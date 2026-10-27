#!/usr/bin/env python
"""n1_finalize.py - post-training pipeline: (optional merge of run folders trained elsewhere) +
eval / cross / gradcam / analyze, each stage idempotent and logged.

This is the release version of the script used to produce the final results of the paper. The
original ran in Google Colab and mirrored every output to a cloud-drive folder; that mirroring and
all account-specific paths have been removed. The stage logic (overwrite rules, skip stamps,
non-finite-row reporting) is unchanged.

    python n1_finalize.py                                    # eval cross gradcam analyze
    python n1_finalize.py --merge-from /path/to/other/outputs/runs   # first merge run folders
    python n1_finalize.py --only eval analyze --force eval analyze   # re-run post-hoc stages

Stages (finished stages are skipped unless --force):
  merge    copy run folders from --merge-from into <out_dir>/runs. A destination run that already
           has full results (done.json without marker_only + preds.npz) is NEVER overwritten;
           marker-only / incomplete / missing destinations are (re)written, done.json last.
  prepare  run_all.py --stages prepare (no-op when work/<ds>/ already holds index, groups, splits, caches)
  retrain  configured runs still without results are trained (run_all.py --stages train --retry-failed);
           disabled with --no-retrain, and refused if more than --max-retrain runs are missing
  eval     run_all.py --stages eval      (CPU)
  cross    run_all.py --stages cross     (GPU; RealWaste models -> TrashNet / TrashBox)
  gradcam  run_all.py --stages gradcam   (GPU)
  analyze  run_all.py --stages analyze   (CPU; tables + figures)
  pack     tar <out_dir>/results/probs -> <out_dir>/results/probs.tar

Configuration: --config (default configs/full.yaml, or env N1_CONFIG). All paths (data_root, work_dir,
out_dir, log_dir) come from the config; relative paths are resolved against the repository root.
Log: <log_dir>/finalize.log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter
from pathlib import Path

PKG = Path(__file__).resolve().parent
STAGES = ["merge", "prepare", "retrain", "eval", "cross", "gradcam", "analyze", "pack"]
DEFAULT_STAGES = ["eval", "cross", "gradcam", "analyze", "pack"]
SKIP_SUFFIX = (".tmp", ".part", ".lock")
SKIP_NAMES = {"ckpt.pt", "ckpt.pt.tmp"}

CONFIG = os.environ.get("N1_CONFIG", "configs/full.yaml")
L_OUT: Path = PKG / "outputs"
LOG: Path = PKG / "logs" / "finalize.log"


# ------------------------------------------------------------------ logging / helpers
def log(msg=""):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def raw_log(text):
    print(text, end="", flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text)


def md5_file(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def run_state(d: Path) -> str:
    """full | marker | incomplete | corrupt | missing"""
    dj = d / "done.json"
    if not dj.exists():
        return "missing"
    try:
        info = json.loads(dj.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "corrupt"
    if info.get("marker_only"):
        return "marker"
    if not (d / "preds.npz").exists() or "classes" not in info:
        return "incomplete"
    return "full"


def same_file(a: Path, b: Path) -> bool:
    return b.exists() and a.stat().st_size == b.stat().st_size


def copy_run(src: Path, dst: Path) -> str:
    """Copy one run folder with the overwrite rules. Returns an action label."""
    st = run_state(dst)
    if st == "full":
        if (src / "done.json").read_bytes() == (dst / "done.json").read_bytes():
            n = 0
            for p in src.iterdir():
                if p.is_file() and p.name not in SKIP_NAMES and not (dst / p.name).exists():
                    atomic_copy(p, dst / p.name)
                    n += 1
            return "already" if n == 0 else "completed"
        return "KEPT_EXISTING_FULL"
    files = [p for p in src.iterdir() if p.is_file() and p.name not in SKIP_NAMES and not p.name.endswith(SKIP_SUFFIX)]
    for p in sorted(files, key=lambda p: p.name == "done.json"):  # done.json last
        if p.name != "done.json" and same_file(p, dst / p.name):
            continue
        atomic_copy(p, dst / p.name)
    return {"marker": "replaced_marker", "missing": "new"}.get(st, f"replaced_{st}")


def load_cfg_runs():
    sys.path.insert(0, str(PKG))
    from n1lib.common import expand_runs, load_config  # noqa: E402
    cfg = load_config(PKG / CONFIG if not Path(CONFIG).is_absolute() else CONFIG)
    return cfg, expand_runs(cfg)


def run_all(args) -> int:
    cmd = [sys.executable, "-u", "run_all.py", "--config", CONFIG, *args]
    log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=PKG, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        raw_log(line)
    p.wait()
    log(f"exit code {p.returncode}")
    return p.returncode


def sig(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def stamp_read(name):
    p = L_OUT / "results" / f".finalize_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def stamp_write(name, obj):
    p = L_OUT / "results" / f".finalize_{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))


def full_run_ids(root: Path):
    rr = root / "runs"
    return sorted(d.name for d in rr.iterdir() if d.is_dir() and run_state(d) == "full") if rr.exists() else []


# ------------------------------------------------------------------ stages
def check_prereqs(cfg, stages):
    problems = []
    if not (PKG / "run_all.py").exists():
        problems.append(f"{PKG}/run_all.py missing")
    if any(s in stages for s in ("prepare", "retrain", "cross", "gradcam")):
        work = Path(cfg["work_dir"])
        dsets = sorted({r.dataset for r in expand_runs_cached(cfg)})
        for ds in dsets:
            if not (work / ds / "splits_grouped.npz").exists() and not (Path(cfg["data_root"]) / "raw" / ds).exists():
                problems.append(f"{work / ds}/splits_grouped.npz and {cfg['data_root']}/raw/{ds} missing "
                                f"(run download_data.py and scripts/install_splits.py)")
    if problems:
        raise RuntimeError("prerequisites missing:\n  " + "\n  ".join(problems))


_RUNS = None


def expand_runs_cached(cfg):
    global _RUNS
    if _RUNS is None:
        from n1lib.common import expand_runs
        _RUNS = expand_runs(cfg)
    return _RUNS


def stage_merge(runs, src_root: Path | None):
    if src_root is None:
        log("merge: no --merge-from given -> skip")
        return
    src_root = Path(src_root)
    ids = sorted(d.name for d in src_root.iterdir() if d.is_dir() and run_state(d) == "full")
    cfg_ids = {r.run_id for r in runs}
    extra = sorted(set(ids) - cfg_ids)
    if extra:
        log(f"merge: WARNING {len(extra)} source run(s) not in the config (still copied): {extra[:5]}")
    t0 = time.time()
    acts = Counter()
    for i in ids:
        a = copy_run(src_root / i, L_OUT / "runs" / i)
        acts[a] += 1
        if a == "KEPT_EXISTING_FULL":
            log(f"  WARNING: {i} already has different full results -> NOT overwritten")
    log(f"merge {src_root} -> {L_OUT / 'runs'}: {dict(acts)} ({time.time() - t0:.0f}s)")
    report_counts(runs)


def report_counts(runs):
    cfg_ids = [r.run_id for r in runs]
    st = Counter()
    missing = []
    for i in cfg_ids:
        s = run_state(L_OUT / "runs" / i)
        st[s] += 1
        if s != "full":
            missing.append(f"{i} [{s}]")
    log(f"COUNT: {st['full']}/{len(cfg_ids)} configured runs have full results (done.json + preds.npz); "
        f"states={dict(st)}")
    for m in missing[:25]:
        log(f"   not full: {m}")
    if len(missing) > 25:
        log(f"   ... and {len(missing) - 25} more")
    w = sorted((L_OUT / "runs").glob("*/weights_fp16.pt"))
    log(f"   runs with weights_fp16.pt (needed by cross/gradcam): {len(w)}")


def stage_retrain(runs, enabled=True, max_runs=6):
    missing = [r for r in runs if run_state(L_OUT / "runs" / r.run_id) != "full"]
    if not missing:
        log("retrain: nothing missing -> skip")
        return
    log(f"retrain: {len(missing)} run(s) without results: {[r.run_id for r in missing][:25]}")
    if not enabled:
        log("retrain: disabled (--no-retrain) -> continuing with the runs that exist")
        return
    if len(missing) > max_runs:
        log(f"WARNING retrain: {len(missing)} runs missing (> {max_runs}); train them with run_all.py instead "
            f"-> continuing with the runs that exist")
        return
    for r in missing:  # stale marker / partial folder would make run_all think it is done
        dj = L_OUT / "runs" / r.run_id / "done.json"
        if dj.exists():
            dj.unlink()
    rc = run_all(["--stages", "train", "--retry-failed", "--blocks", *sorted({r.block for r in missing})])
    still = [r.run_id for r in missing if run_state(L_OUT / "runs" / r.run_id) != "full"]
    if still or rc != 0:
        log(f"WARNING retrain: rc={rc}, still missing {still} -> analysis continues without them")
    else:
        log(f"retrain: {len(missing)} run(s) trained")


def quarantine_local():
    """Run folders whose done.json is a placeholder/incomplete would break eval/analyze
    (analyze reads every runs/*/done.json). Move them out of <out_dir>/runs."""
    q = L_OUT.parent / (L_OUT.name + "_quarantine")
    rr = L_OUT / "runs"
    if not rr.exists():
        return
    for d in sorted(rr.iterdir()):
        if d.is_dir() and (d / "done.json").exists() and run_state(d) != "full":
            q.mkdir(exist_ok=True)
            if (q / d.name).exists():
                shutil.rmtree(q / d.name)
            shutil.move(str(d), str(q / d.name))
            log(f"WARNING: {d.name} has a non-full done.json ({run_state(q / d.name)}) -> moved to {q}")


def stage_eval(force=False):
    quarantine_local()
    ids = full_run_ids(L_OUT)
    s = dict(ids=sig(ids), n=len(ids))
    ml = L_OUT / "results" / "metrics_long.csv"
    if not force and ml.exists() and stamp_read("eval") == s:
        log(f"eval: up to date for {len(ids)} runs -> skip")
        return False
    t0 = time.time()
    if run_all(["--stages", "eval"]) != 0 or not ml.exists():
        raise RuntimeError("eval stage failed (see log above)")
    import pandas as pd
    M = pd.read_csv(ml)
    log(f"eval: metrics_long.csv {len(M)} rows, {M[M.seed >= 0].run_id.nunique()} single runs (expected {len(ids)}), "
        f"{M[M.seed < 0].run_id.nunique()} ensembles ({time.time() - t0:.0f}s)")
    C = pd.read_csv(L_OUT / "results" / "conformal_long.csv")
    z = C[C.coverage <= 0]
    log(f"eval: conformal_long.csv {len(C)} rows, rows with coverage 0 = {len(z)}"
        + (f" (runs {sorted(set(z.run_id))[:10]})" if len(z) else ""))
    nfp = L_OUT / "results" / "nonfinite_rows.csv"
    if nfp.exists():
        NF = pd.read_csv(nfp)
        if len(NF):
            g = NF.groupby("run_id")[["calconf_dropped", "n_test_dropped"]].max()
            log(f"eval: nonfinite_rows.csv: {len(g)} run(s) affected, calconf rows dropped = "
                f"{int(g.calconf_dropped.sum())}, test rows dropped = {int(g.n_test_dropped.sum())}")
        else:
            log("eval: nonfinite_rows.csv: no non-finite rows")
    stamp_write("eval", s)
    return True


def stage_cross(cfg, force=False):
    src = (cfg.get("crossdata") or {}).get("source", "realwaste")
    w = sorted((L_OUT / "runs").glob(f"{src}-grouped-*-ce-f*-s*/weights_fp16.pt"))
    cx = L_OUT / "results" / "crossdata.csv"
    s = dict(w=sig([p.parent.name for p in w]))
    if not force and cx.exists() and stamp_read("cross") == s:
        log("cross: up to date -> skip")
        return False
    if not w:
        log("WARNING cross: no weights_fp16.pt in runs -> skipped")
        return False
    t0 = time.time()
    run_all(["--stages", "cross"])
    if not cx.exists():
        log("WARNING cross: results/crossdata.csv not produced (see 'cross-dataset stage failed' above)")
        return False
    import pandas as pd
    X = pd.read_csv(cx)
    log(f"cross: crossdata.csv {len(X)} rows (targets {sorted(set(X.target))}, {len(w)} source models) "
        f"({time.time() - t0:.0f}s)")
    stamp_write("cross", s)
    return True


def stage_gradcam(force=False):
    fp = L_OUT / "figures" / "fig_gradcam_rejected.png"
    if fp.exists() and not force:
        log("gradcam: figure exists -> skip")
        return False
    t0 = time.time()
    run_all(["--stages", "gradcam"])
    if not fp.exists():
        log("WARNING gradcam: figures/fig_gradcam_rejected.png not produced (see 'gradcam' lines above)")
        return False
    log(f"gradcam: written ({time.time() - t0:.0f}s)")
    return True


def stage_analyze(force=False):
    quarantine_local()
    res = L_OUT / "results"
    inputs = [res / n for n in ("metrics_long.csv", "conformal_long.csv", "perclass_long.csv", "crossdata.csv")]
    inputs.append(L_OUT / "figures" / "fig_gradcam_rejected.png")
    s = dict(inp=sig([(p.name, md5_file(p)) for p in inputs if p.exists()]))
    allt = L_OUT / "tables" / "ALL_TABLES.md"
    if not force and allt.exists() and stamp_read("analyze") == s:
        log("analyze: up to date -> skip")
        return False
    t0 = time.time()
    if run_all(["--stages", "analyze"]) != 0 or not allt.exists():
        raise RuntimeError("analyze stage failed (see log above)")
    log(f"analyze: {len(list((L_OUT / 'tables').glob('*.csv')))} tables, "
        f"{len(list((L_OUT / 'figures').glob('*.png')))} PNG figures ({time.time() - t0:.0f}s)")
    stamp_write("analyze", s)
    return True


def stage_pack():
    res = L_OUT / "results"
    if not (res / "probs").exists():
        log("pack: no results/probs -> skip")
        return
    tmp = res / "probs.tar.part"
    with tarfile.open(tmp, "w") as tf:
        tf.add(res / "probs", arcname="probs")
    os.replace(tmp, res / "probs.tar")
    log(f"pack: results/probs.tar {(res / 'probs.tar').stat().st_size / 2**20:.1f} MiB, md5 {md5_file(res / 'probs.tar')}")


# ------------------------------------------------------------------ main
def main():
    global CONFIG, L_OUT, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--only", nargs="+", choices=STAGES, default=None,
                    help=f"stages to run (default: {' '.join(DEFAULT_STAGES)}; plus 'merge' when --merge-from is given)")
    ap.add_argument("--force", nargs="+", choices=STAGES, default=[])
    ap.add_argument("--merge-from", default=None, help="folder with <run_id>/ subfolders trained elsewhere")
    ap.add_argument("--no-retrain", action="store_true")
    ap.add_argument("--max-retrain", type=int, default=6)
    a = ap.parse_args()
    CONFIG = a.config
    stages = a.only or (["merge"] if a.merge_from else []) + DEFAULT_STAGES

    cfg, runs = load_cfg_runs()
    L_OUT = Path(cfg["out_dir"])
    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir = log_dir if log_dir.is_absolute() else PKG / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    LOG = log_dir / "finalize.log"
    t00 = time.time()
    try:
        log(f"=== finalize start stages={stages} force={a.force} config={CONFIG} out={L_OUT}")
        log(f"config: {len(runs)} runs {dict(Counter(r.block for r in runs))}")
        check_prereqs(cfg, stages)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        (L_OUT / "runs").mkdir(parents=True, exist_ok=True)
        f = set(a.force)
        if "merge" in stages:
            stage_merge(runs, a.merge_from)
        if "prepare" in stages and run_all(["--stages", "prepare"]) != 0:
            raise RuntimeError("prepare failed")
        if "retrain" in stages:
            stage_retrain(runs, not a.no_retrain, a.max_retrain)
        if "eval" in stages:
            stage_eval("eval" in f)
        if "cross" in stages:
            stage_cross(cfg, "cross" in f)
        if "gradcam" in stages:
            stage_gradcam("gradcam" in f)
        if "analyze" in stages:
            stage_analyze("analyze" in f)
        if "pack" in stages:
            stage_pack()
        report_counts(runs)
        log(f"outputs: {L_OUT}/results, /tables, /figures")
        log(f"=== finalize finished ({(time.time() - t00) / 60:.1f} min)")
        rc = 0
    except BaseException:  # noqa: BLE001
        log("Traceback:\n" + traceback.format_exc())
        log("=== finalize FAILED (re-run the same command; finished stages are skipped)")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
