"""Stage 3 (CPU): post-hoc calibration, conformal prediction and selective-prediction metrics
for every finished run (+ deep ensembles over seeds). Fully recomputed from preds.npz (cheap)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .calib import apply_temperature, apply_vector, fit_temperature, fit_vector_scaling, softmax_np
from .common import read_json
from .conformal import conformal_sets, set_metrics
from .metrics import all_metrics


def _finite_rows(a):
    """Boolean mask of rows whose values are all finite (works for (n, C) and MC (S, n, C))."""
    a = np.asarray(a)
    return np.isfinite(a).all(-1) if a.ndim == 2 else np.isfinite(a).all(-1).all(0)


def _probs_for_run(z):
    """Returns {method: {part: probs}} and per-part labels, plus fitted params."""
    out, params = {}, {}
    L = {p: z[f"logits_{p}"].astype(np.float64) for p in ("calfit", "calconf", "test")}
    out["raw"] = {p: softmax_np(v) for p, v in L.items()}
    okf = _finite_rows(L["calfit"])  # non-finite calfit rows are excluded from TS/VS fitting
    params["n_calfit_nonfinite"] = int((~okf).sum())
    T = fit_temperature(L["calfit"][okf], z["y_calfit"][okf])
    params["temperature"] = T
    out["ts"] = {p: apply_temperature(v, T) for p, v in L.items()}
    wb = fit_vector_scaling(L["calfit"][okf], z["y_calfit"][okf])
    out["vs"] = {p: apply_vector(v, wb) for p, v in L.items()}
    if "mc_test" in z:
        out["mcd"] = {p: softmax_np(z[f"mc_{p}"].astype(np.float64).reshape(-1, L[p].shape[1])).reshape(
            z[f"mc_{p}"].shape).mean(0) for p in L}
    return out, params


def evaluate_all(cfg, log=print):
    out_dir = Path(cfg["out_dir"])
    res_dir = out_dir / "results"
    (res_dir / "probs").mkdir(parents=True, exist_ok=True)
    ecfg = cfg.get("eval", {})
    alphas = ecfg.get("alphas", [0.05, 0.1])
    scores = ecfg.get("conformal_scores", ["lac", "aps", "raps"])
    conf_methods = set(ecfg.get("conformal_methods", ["raw", "ts", "vs", "mcd", "ens", "ens_ts"]))
    n_bins = ecfg.get("ece_bins", 15)
    raps = ecfg.get("raps", dict(lam=0.01, k_reg=2))

    done = sorted((out_dir / "runs").glob("*/done.json"))
    rows, crows, prows = [], [], []
    nf_rows = []  # per run x method: non-finite rows in calconf (dropped) and test (reported)
    groups = defaultdict(list)

    def evaluate_method(keys, method, P, ys, classes):
        # non-finite TEST rows (e.g. fp16 overflow -> NaN logits) are dropped, counted and reported,
        # consistent with the calibration-row handling (otherwise ECE/NLL/Brier become NaN).
        ok_te = np.isfinite(P["test"]).all(1)
        n_te_bad, n_test = int((~ok_te).sum()), int(len(ok_te))
        if n_te_bad:
            P = dict(P, test=P["test"][ok_te]); ys = dict(ys, test=ys["test"][ok_te])
        m = all_metrics(P["test"], ys["test"], classes, n_bins)
        per = {k: v for k, v in m.items() if k.startswith("f1::")}
        m = {k: v for k, v in m.items() if not k.startswith("f1::")}
        rows.append(dict(keys, method=method, **m))
        for k, v in per.items():
            prows.append(dict(keys, method=method, cls=k[4:], f1=v))
        np.savez_compressed(res_dir / "probs" / f"{keys['run_id']}__{method}.npz",
                            probs=P["test"].astype(np.float32), y=ys["test"])
        ok_cal = np.isfinite(P["calconf"]).all(1)
        n_drop = int((~ok_cal).sum())
        if n_drop or n_te_bad:
            nf_rows.append(dict(run_id=keys["run_id"], method=method, n_calconf=int(len(ok_cal)),
                                calconf_dropped=n_drop, n_test=n_test, n_test_dropped=n_te_bad,
                                calconf_idx=";".join(map(str, np.where(~ok_cal)[0].tolist())),
                                test_idx=";".join(map(str, np.where(~ok_te)[0].tolist()))))
        if method in conf_methods:
            for sc in scores:
                for a in alphas:
                    for mond in (False, True):
                        sets, _ = conformal_sets(P["calconf"], ys["calconf"], P["test"], sc, a, mond,
                                                 seed=int(keys.get("seed", 0)) + 7, **raps)
                        sm = set_metrics(sets, ys["test"], P["test"])
                        cc = sm.pop("class_cov")
                        crows.append(dict(keys, method=method, score=sc, alpha=a, mondrian=mond, **sm,
                                          class_cov=";".join(f"{c:.4f}" for c in cc),
                                          n_cal=int(ok_cal.sum()), n_cal_dropped=n_drop))

    for dj in done:
        info = read_json(dj)
        pz = dj.parent / "preds.npz"
        if not pz.exists():
            continue
        z = dict(np.load(pz))
        ys = {p: z[f"y_{p}"] for p in ("calfit", "calconf", "test")}
        classes = info["classes"]
        keys = {k: info[k] for k in ("run_id", "dataset", "protocol", "preprocess", "model", "loss", "fold", "seed", "block")}
        P, params = _probs_for_run(z)
        if params["n_calfit_nonfinite"]:
            log(f"WARNING evaluate: {info['run_id']} {params['n_calfit_nonfinite']} non-finite calfit rows "
                f"excluded from TS/VS fitting")
        keys_r = dict(keys, temperature=params["temperature"])
        for meth, pp in P.items():
            evaluate_method(keys_r, meth, pp, ys, classes)
        gk = info["run_id"].rsplit("-s", 1)[0]
        groups[gk].append((keys, P, ys, classes))

    # deep ensembles from seeds (identical splits across seeds)
    for gk, members in groups.items():
        if len(members) < 2:
            continue
        keys, _, ys, classes = members[0]
        if any(len(m[2]["test"]) != len(ys["test"]) or not np.array_equal(m[2]["test"], ys["test"]) for m in members):
            continue
        ek = dict(keys, run_id=f"{gk}-ens{len(members)}", seed=-1, temperature=np.nan)
        for meth, src in (("ens", "raw"), ("ens_ts", "ts")):
            P = {p: np.mean([m[1][src][p] for m in members], 0) for p in ("calfit", "calconf", "test")}
            evaluate_method(ek, meth, P, ys, classes)

    pd.DataFrame(rows).to_csv(res_dir / "metrics_long.csv", index=False)
    pd.DataFrame(crows).to_csv(res_dir / "conformal_long.csv", index=False)
    pd.DataFrame(prows).to_csv(res_dir / "perclass_long.csv", index=False)
    nf = pd.DataFrame(nf_rows, columns=["run_id", "method", "n_calconf", "calconf_dropped", "n_test",
                                        "n_test_dropped", "calconf_idx", "test_idx"])
    nf.to_csv(res_dir / "nonfinite_rows.csv", index=False)
    single = nf[~nf.run_id.str.contains("-ens")]
    for rid, g in single.groupby("run_id"):
        r0 = g.iloc[0]
        log(f"evaluate: non-finite rows dropped: {rid} calconf {int(g.calconf_dropped.max())}/{r0.n_calconf}"
            f" (idx {r0.calconf_idx or '-'}), test {int(g.n_test_dropped.max())}/{r0.n_test} (idx {r0.test_idx or '-'})"
            f" [methods {','.join(g.method)}]")
    per_run = single.groupby("run_id")[["calconf_dropped", "n_test_dropped"]].max()
    log(f"evaluate: NONFINITE SUMMARY calconf rows dropped = {int(per_run.calconf_dropped.sum())} in "
        f"{int((per_run.calconf_dropped > 0).sum())} single run(s); test rows dropped = {int(per_run.n_test_dropped.sum())} in "
        f"{int((per_run.n_test_dropped > 0).sum())} single run(s) (+ {nf[nf.run_id.str.contains('-ens')].run_id.nunique()} "
        f"ensemble(s) affected) -> results/nonfinite_rows.csv")
    log(f"evaluate: {len(done)} runs -> {len(rows)} metric rows, {len(crows)} conformal rows")
    return len(rows)
