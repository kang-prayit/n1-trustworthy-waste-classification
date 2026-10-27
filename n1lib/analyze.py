"""Stage 4: aggregate results into publication tables (CSV + Markdown + LaTeX) and figures
(PNG 300 dpi + PDF + EPS) for Scientific Reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_json
from .metrics import ece_bins, risk_coverage
from .stats import holm, mcnemar_exact, wilcoxon_paired

REF_PAPER = {"resnet101": 0.9886, "hybrid_r101_incv3": 0.9000}  # Kumar et al. 2026, Table 3 (val acc)
MODEL_LABEL = {"resnet101": "ResNet101", "hybrid_r101_incv3": "ResNet101+InceptionV3",
               "efficientnetv2_s": "EfficientNetV2-S", "efficientnet_b0": "EfficientNet-B0",
               "mobilenetv3_large": "MobileNetV3-L", "convnext_tiny": "ConvNeXt-T", "deit_small": "DeiT-S",
               "tiny_cnn": "TinyCNN", "tiny_hybrid": "TinyHybrid"}


def ml(m):
    return MODEL_LABEL.get(m, m)


def pm(s, d=3, scale=1.0):
    s = pd.Series(s).dropna() * scale
    if len(s) == 0:
        return "–"
    if len(s) == 1:
        return f"{s.iloc[0]:.{d}f}"
    return f"{s.mean():.{d}f} ± {s.std(ddof=1):.{d}f}"


# ------------------------------------------------------------------ writers
def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def _tex_escape(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("±", "$\\pm$"), ("α", "$\\alpha$"), ("≥", "$\\geq$")):
        s = s.replace(a, b)
    return s


def _tex(df, caption, label):
    cols = "l" * len(df.columns)
    lines = ["\\begin{table}[ht]", "\\centering", "\\small", f"\\caption{{{_tex_escape(caption)}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{cols}}}", "\\hline",
             " & ".join(_tex_escape(c) for c in df.columns) + " \\\\", "\\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(_tex_escape(v) for v in r.values) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


class Writer:
    def __init__(self, tab_dir):
        self.dir = Path(tab_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.md = []

    def table(self, df, name, caption):
        if df is None or len(df) == 0:
            return
        df.to_csv(self.dir / f"{name}.csv", index=False)
        (self.dir / f"{name}.tex").write_text(_tex(df, caption, f"tab:{name}"), encoding="utf-8")
        self.md.append(f"## {name}\n\n{caption}\n\n{_md(df)}\n")

    def close(self):
        (self.dir / "ALL_TABLES.md").write_text("\n".join(self.md), encoding="utf-8")


def savefig(fig, fig_dir, name):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    try:
        fig.savefig(fig_dir / f"{name}.eps", bbox_inches="tight")
    except Exception:
        pass


# ------------------------------------------------------------------ main

def optimism_gap_tests(M, metrics=("acc", "macro_f1"), naive_protocols=("naive_aug", "naive"), ref_protocol="grouped",
                       method="raw", loss="ce", preprocess="none"):
    """Optimism-gap statistics: naive (item-level) protocol versus leakage-aware grouped CV.

    For every dataset x model that was run under a naive protocol, the per-(fold, seed) metric values of the naive
    run and the grouped run are paired on identical (fold, seed) keys and compared with a two-sided Wilcoxon
    signed-rank test (zero differences discarded, as in stats.wilcoxon_paired).  P values are Holm-adjusted across
    all model x dataset comparisons within one family = (metric, naive protocol).  Naive protocols use a single
    70/15/15 split (fold 0), so the number of pairs equals the number of seeds run on both protocols; with n pairs
    the smallest attainable two-sided exact P is 2 / 2**n (0.25 for n = 3), reported as 'Min attainable p'.
    Returns a DataFrame with one row per comparison (empty if no naive runs exist)."""
    cols = ["Family", "Metric", "Naive protocol", "Dataset", "Model", "Pairs", "Naive mean", "Grouped mean",
            "Mean diff (naive - grouped)", "Median diff", "W", "p", "p (Holm)", "Min attainable p"]
    if M is None or len(M) == 0:
        return pd.DataFrame(columns=cols)
    d0 = M[(M.seed >= 0) & (M.method == method) & (M.loss == loss)]
    if "preprocess" in d0:
        d0 = d0[d0.preprocess == preprocess]
    rows = []
    for metric in metrics:
        if metric not in d0:
            continue
        for prot in naive_protocols:
            fam = []
            for ds in sorted(set(d0.dataset)):
                d = d0[d0.dataset == ds]
                for m in sorted(set(d[d.protocol == prot].model)):
                    a = d[(d.model == m) & (d.protocol == prot)].groupby(["fold", "seed"])[metric].mean()
                    b = d[(d.model == m) & (d.protocol == ref_protocol)].groupby(["fold", "seed"])[metric].mean()
                    j = a.index.intersection(b.index)
                    if len(j) == 0:
                        continue
                    x, y = a[j].to_numpy(float), b[j].to_numpy(float)
                    diff = x - y
                    st, p_ = wilcoxon_paired(x, y)
                    nz = int(np.sum(diff != 0))
                    fam.append({"Family": f"{metric} | {prot} vs {ref_protocol}", "Metric": f"{metric} ({method})",
                                "Naive protocol": prot, "Dataset": ds, "Model": ml(m), "Pairs": int(len(j)),
                                "Naive mean": float(x.mean()), "Grouped mean": float(y.mean()),
                                "Mean diff (naive - grouped)": float(diff.mean()), "Median diff": float(np.median(diff)),
                                "W": st, "p": float(p_), "Min attainable p": float(min(1.0, 2.0 / 2 ** nz)) if nz else 1.0})
            for r, pa in zip(fam, holm([r["p"] for r in fam]) if fam else []):
                r["p (Holm)"] = float(pa)
                rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def analyze(cfg, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6,
                         "savefig.transparent": False})
    out = Path(cfg["out_dir"])
    res = out / "results"
    tab_dir, fig_dir = out / "tables", out / "figures"
    W = Writer(tab_dir)
    M = pd.read_csv(res / "metrics_long.csv")
    C = pd.read_csv(res / "conformal_long.csv") if (res / "conformal_long.csv").exists() else pd.DataFrame()
    PC = pd.read_csv(res / "perclass_long.csv") if (res / "perclass_long.csv").exists() else pd.DataFrame()
    main_ds = cfg.get("analysis", {}).get("main_dataset", "realwaste")
    ref_model = cfg.get("analysis", {}).get("reference_model", "resnet101")
    single = M[M.seed >= 0]

    # ---- T1 datasets & duplicates
    rows = []
    for d in sorted(set(M.dataset)):
        sp = Path(cfg["work_dir"]) / d / "dedup_summary.json"
        if not sp.exists():
            continue
        s = read_json(sp)
        r = dict(Dataset=d, Images=s["n_images"], Classes=len(s["class_counts"]),
                 **{"Exact (MD5) dup. images": s["n_exact_md5_dup_images"],
                    "Near-dup pairs": s["n_near_dup_pairs"], "Dup. groups": s["n_dup_groups"],
                    "Images in dup. groups (%)": f"{s['n_images_in_dup_groups']} ({s['pct_images_in_dup_groups']})",
                    "Redundant images": s["n_redundant_images"], "Cross-class pairs": s["n_cross_class_pairs"],
                    "Largest group": s["largest_group"]})
        for a in (0, 3):
            rp = Path(cfg["work_dir"]) / d / f"splits_naive_aug{a}_report.json"
            if rp.exists():
                rr = read_json(rp)
                r[f"Naive{'+aug' if a else ''}: test items w/ dup. group in train (%)"] = rr["pct_test_dupgroup_in_train"]
        gp = Path(cfg["work_dir"]) / d / "splits_grouped_report.json"
        if gp.exists():
            r["Grouped CV: test w/ dup. group in train (%)"] = max(x["pct_test_dupgroup_in_train"] for x in read_json(gp))
        rows.append(r)
    W.table(pd.DataFrame(rows), "tab1_datasets_duplicates",
            "Datasets and perceptual-hash duplicate analysis (pHash and dHash agreement plus MD5).")

    # ---- T2 optimism gap
    g = single[(single.dataset == main_ds) & (single.method == "raw") & (single.loss == "ce") & (single.preprocess == "none")]
    rows = []
    for m in [x for x in REF_PAPER if x in set(g.model)] + sorted(set(g[g.protocol != "grouped"].model) - set(REF_PAPER)):
        r = {"Model": ml(m), "Kumar et al. 2026 (reported, single split)": f"{REF_PAPER.get(m, np.nan):.4f}" if m in REF_PAPER else "–"}
        for p, lab in (("naive_aug", "Naive + aug-before-split"), ("naive", "Naive random split"), ("grouped", "Leakage-aware grouped 5-fold CV")):
            s = g[(g.model == m) & (g.protocol == p)]
            r[f"{lab} acc"] = pm(s.acc, 4)
            r[f"{lab} macro-F1"] = pm(s.macro_f1, 4)
        a = g[(g.model == m) & (g.protocol == "naive_aug")].acc.mean()
        b = g[(g.model == m) & (g.protocol == "grouped")].acc.mean()
        r["Optimism gap (pp)"] = f"{100 * (a - b):.2f}" if np.isfinite(a) and np.isfinite(b) else "–"
        rows.append(r)
    W.table(pd.DataFrame(rows), "tab2_optimism_gap", "Optimism of the naive protocol versus leakage-aware evaluation (test accuracy, mean ± sd over seeds/folds).")
    # optimism-gap significance: paired Wilcoxon on per-(fold, seed) naive - grouped differences, Holm across model x dataset
    OG = optimism_gap_tests(M)
    if len(OG):
        og = OG.copy()
        for c in ("Naive mean", "Grouped mean", "Mean diff (naive - grouped)", "Median diff"):
            og[c] = og[c].map(lambda v: f"{v:+.4f}" if "diff" in c else f"{v:.4f}")
        for c in ("W", "p", "p (Holm)", "Min attainable p"):
            og[c] = og[c].map(lambda v: "–" if pd.isna(v) else f"{v:.4g}")
        W.table(og, "tab2b_optimism_gap_wilcoxon",
                "Optimism gap: two-sided Wilcoxon signed-rank test on per-(fold, seed) differences naive minus grouped "
                "(paired on identical fold and seed); Holm-adjusted across model × dataset within each metric and naive protocol.")

    # ---- T3 main benchmark per dataset
    rows = []
    gm = M[(M.protocol == "grouped") & (M.loss == "ce") & (M.preprocess == "none")]
    for d in sorted(set(gm.dataset)):
        for m in sorted(set(gm[gm.dataset == d].model), key=lambda x: list(MODEL_LABEL).index(x) if x in MODEL_LABEL else 99):
            s = gm[(gm.dataset == d) & (gm.model == m)]
            raw, ts = s[s.method == "raw"], s[s.method == "ts"]
            ens = s[s.method == "ens_ts"]
            rows.append({"Dataset": d, "Model": ml(m), "Runs": len(raw), "Accuracy": pm(raw.acc, 4),
                         "Macro-F1": pm(raw.macro_f1, 4), "ECE raw": pm(raw.ece), "ECE TS": pm(ts.ece),
                         "NLL raw": pm(raw.nll), "NLL TS": pm(ts.nll), "Brier TS": pm(ts.brier),
                         "AURC TS": pm(ts.aurc, 4), "Acc@90% cov.": pm(ts.acc_cov90, 4),
                         "Ensemble acc": pm(ens.acc, 4), "Ensemble ECE (TS)": pm(ens.ece)})
    W.table(pd.DataFrame(rows), "tab3_main_benchmark", "Leakage-aware benchmark (grouped stratified 5-fold CV; mean ± sd over folds × seeds).")

    # ---- T4 calibration methods (main dataset)
    approaches = [("ce", "raw", "CE (uncalibrated)"), ("ce", "ts", "CE + temperature scaling"),
                  ("ce", "vs", "CE + vector scaling"), ("ce", "mcd", "CE + MC-dropout"),
                  ("ls", "raw", "Label smoothing"), ("ls", "ts", "Label smoothing + TS"),
                  ("focal", "raw", "Focal loss"), ("focal", "ts", "Focal loss + TS"),
                  ("ce", "ens", "Deep ensemble"), ("ce", "ens_ts", "Deep ensemble + TS")]
    gg = M[(M.dataset == main_ds) & (M.protocol == "grouped") & (M.preprocess == "none")]
    rows = []
    for m in sorted(set(gg.model), key=lambda x: list(MODEL_LABEL).index(x) if x in MODEL_LABEL else 99):
        for loss, meth, lab in approaches:
            s = gg[(gg.model == m) & (gg.loss == loss) & (gg.method == meth)]
            if len(s) == 0:
                continue
            rows.append({"Model": ml(m), "Approach": lab, "n": len(s), "Accuracy": pm(s.acc, 4), "ECE": pm(s.ece),
                         "Classwise-ECE": pm(s.cw_ece), "MCE": pm(s.mce), "Brier": pm(s.brier), "NLL": pm(s.nll)})
    W.table(pd.DataFrame(rows), "tab4_calibration_methods", f"Calibration methods on {main_ds} (ECE with 15 equal-width bins).")

    # ---- T5 conformal
    if len(C):
        cg = C[(C.dataset == main_ds) & (C.protocol == "grouped") & (C.loss == "ce") & (C.preprocess == "none") & (C.method == "ts")]
        rows = []
        for m in sorted(set(cg.model), key=lambda x: list(MODEL_LABEL).index(x) if x in MODEL_LABEL else 99):
            for sc in ("lac", "aps", "raps"):
                for a in sorted(set(cg.alpha)):
                    for mond in (False, True):
                        s = cg[(cg.model == m) & (cg.score == sc) & (cg.alpha == a) & (cg.mondrian == mond)]
                        if len(s) == 0:
                            continue
                        rows.append({"Model": ml(m), "Score": sc.upper(), "α": a, "Class-conditional": "yes" if mond else "no",
                                     "Coverage": pm(s.coverage), "Target": f"{1 - a:.2f}", "Avg set size": pm(s.avg_size, 2),
                                     "Singleton rate": pm(s.singleton_rate), "Worst-class coverage": pm(s.worst_class_cov)})
        W.table(pd.DataFrame(rows), "tab5_conformal", "Split conformal prediction on temperature-scaled probabilities (conformal-calibration split).")

        # ---- T6 selective
        rows = []
        for m in sorted(set(cg.model), key=lambda x: list(MODEL_LABEL).index(x) if x in MODEL_LABEL else 99):
            s = gg[(gg.model == m) & (gg.loss == "ce") & (gg.method == "ts")]
            r0 = gg[(gg.model == m) & (gg.loss == "ce") & (gg.method == "raw")]
            c = cg[(cg.model == m) & (cg.score == "aps") & (cg.alpha == 0.1) & (~cg.mondrian.astype(bool))]
            rows.append({"Model": ml(m), "Full-coverage acc": pm(s.acc, 4), "AURC": pm(s.aurc, 4), "E-AURC": pm(s.eaurc, 4),
                         "Acc@80% (MSP)": pm(s.acc_cov80, 4), "Acc@90% (MSP)": pm(s.acc_cov90, 4),
                         "AURC raw": pm(r0.aurc, 4),
                         "Singleton coverage (APS α=0.1)": pm(c.sel_singleton_coverage),
                         "Singleton acc (APS α=0.1)": pm(c.sel_singleton_acc, 4),
                         "Human review load (%)": pm(1 - c.sel_singleton_coverage, 1, 100)})
        W.table(pd.DataFrame(rows), "tab6_selective_prediction", "Selective prediction: max-softmax versus conformal set-size deferral.")

    # ---- T7 per-class F1
    if len(PC):
        p = PC[(PC.dataset == main_ds) & (PC.protocol == "grouped") & (PC.loss == "ce") & (PC.preprocess == "none") & (PC.method == "raw") & (PC.seed >= 0)]
        if len(p):
            t = p.groupby(["model", "cls"]).f1.apply(lambda s: pm(s, 3)).unstack("cls").reset_index()
            t["model"] = t["model"].map(ml)
            W.table(t.rename(columns={"model": "Model"}), "tab7_perclass_f1", "Per-class F1 (mean ± sd over folds × seeds).")

    # ---- T8 statistics
    rows = []
    s0 = single[(single.dataset == main_ds) & (single.protocol == "grouped") & (single.loss == "ce") & (single.preprocess == "none")]
    others = [m for m in sorted(set(s0.model)) if m != ref_model]
    for metric, meth in (("acc", "raw"), ("macro_f1", "raw"), ("ece", "ts"), ("nll", "ts")):
        pv, tmp = [], []
        for m in others:
            a = s0[(s0.model == m) & (s0.method == meth)].set_index(["fold", "seed"])[metric]
            b = s0[(s0.model == ref_model) & (s0.method == meth)].set_index(["fold", "seed"])[metric]
            j = a.index.intersection(b.index)
            st, p_ = wilcoxon_paired(a[j], b[j])
            pv.append(p_)
            tmp.append({"Test": "Wilcoxon signed-rank", "Metric": f"{metric} ({meth})", "Comparison": f"{ml(m)} vs {ml(ref_model)}",
                        "Pairs": len(j), "Mean diff": f"{(a[j] - b[j]).mean():+.4f}", "p": f"{p_:.4g}"})
        for r, pa in zip(tmp, holm(pv) if pv else []):
            r["p (Holm)"] = f"{pa:.4g}"
            rows.append(r)
    for m in sorted(set(s0.model)):
        a = s0[(s0.model == m) & (s0.method == "raw")].set_index(["fold", "seed"]).ece
        b = s0[(s0.model == m) & (s0.method == "ts")].set_index(["fold", "seed"]).ece
        j = a.index.intersection(b.index)
        st, p_ = wilcoxon_paired(a[j], b[j])
        rows.append({"Test": "Wilcoxon signed-rank", "Metric": "ECE raw vs TS", "Comparison": ml(m), "Pairs": len(j),
                     "Mean diff": f"{(a[j] - b[j]).mean():+.4f}", "p": f"{p_:.4g}", "p (Holm)": ""})
    # McNemar: pooled out-of-fold test predictions (seed 0, all folds) vs reference model
    def pooled_correct(model):
        cs = []
        for f in sorted(set(s0.fold)):
            fp = res / "probs" / f"{main_ds}-grouped-{model}-ce-f{f}-s0__raw.npz"
            if not fp.exists():
                return None
            z = np.load(fp)
            cs.append(z["probs"].argmax(1) == z["y"])
        return np.concatenate(cs)
    ref_c = pooled_correct(ref_model)
    if ref_c is not None:
        pv, tmp = [], []
        for m in others:
            c = pooled_correct(m)
            if c is None or len(c) != len(ref_c):
                continue
            b, cc, p_ = mcnemar_exact(c, ref_c)
            pv.append(p_)
            tmp.append({"Test": "McNemar (exact)", "Metric": "accuracy (pooled OOF, seed 0)", "Comparison": f"{ml(m)} vs {ml(ref_model)}",
                        "Pairs": len(c), "Mean diff": f"{c.mean() - ref_c.mean():+.4f} (b={b}, c={cc})", "p": f"{p_:.4g}"})
        for r, pa in zip(tmp, holm(pv) if pv else []):
            r["p (Holm)"] = f"{pa:.4g}"
            rows.append(r)
    for _, r in OG.iterrows():
        rows.append({"Test": "Wilcoxon signed-rank (optimism gap)", "Metric": r["Metric"],
                     "Comparison": f"{r['Model']}, {r['Dataset']}: {r['Naive protocol']} vs grouped", "Pairs": int(r["Pairs"]),
                     "Mean diff": f"{r['Mean diff (naive - grouped)']:+.4f}", "p": f"{r['p']:.4g}",
                     "p (Holm)": f"{r['p (Holm)']:.4g}"})
    W.table(pd.DataFrame(rows), "tab8_statistical_tests", "Paired statistical tests (Holm-corrected within each family).")

    # ---- T9 segmentation ablation
    sg = single[(single.dataset == main_ds) & (single.protocol == "grouped") & (single.loss == "ce")]
    seg_models = sorted(set(sg[sg.preprocess == "seg"].model))
    rows = []
    for m in seg_models:
        for pre in ("none", "seg"):
            folds = set(sg[(sg.model == m) & (sg.preprocess == "seg")].fold)
            seeds = set(sg[(sg.model == m) & (sg.preprocess == "seg")].seed)
            s = sg[(sg.model == m) & (sg.preprocess == pre) & sg.fold.isin(folds) & sg.seed.isin(seeds)]
            rows.append({"Model": ml(m), "Input": "segmented" if pre == "seg" else "raw image",
                         "Accuracy": pm(s[s.method == "raw"].acc, 4), "Macro-F1": pm(s[s.method == "raw"].macro_f1, 4),
                         "ECE raw": pm(s[s.method == "raw"].ece), "ECE TS": pm(s[s.method == "ts"].ece),
                         "AURC TS": pm(s[s.method == "ts"].aurc, 4)})
    W.table(pd.DataFrame(rows), "tab9_segmentation_ablation", "Effect of segmentation pre-processing (approximation of Kumar et al. Alg. 1) on accuracy and calibration.")

    # ---- T10 cross-dataset
    cx = res / "crossdata.csv"
    if cx.exists():
        X = pd.read_csv(cx)
        rows = []
        for (t, m), s in X.groupby(["target", "model"]):
            rows.append({"Source → target": f"{s.source.iloc[0]} → {t}", "Model": ml(m), "n": int(s.n.iloc[0]),
                         "In-domain acc": pm(s.in_domain_test_acc, 4), "Shift acc (shared classes)": pm(s.acc_shared, 4),
                         "Shift acc (full label space)": pm(s.acc_full, 4), "ECE TS under shift": pm(s.ece_ts),
                         "LAC cov. (target 0.90)": pm(s["cov_lac_a0.1"]), "LAC size": pm(s["size_lac_a0.1"], 2),
                         "APS cov. (target 0.90)": pm(s["cov_aps_a0.1"]), "APS size": pm(s["size_aps_a0.1"], 2)})
        W.table(pd.DataFrame(rows), "tab10_cross_dataset", "Cross-dataset sanity check: calibration and conformal coverage under distribution shift.")

    # ---- runtime table
    rt = []
    for dj in (out / "runs").glob("*/done.json"):
        i = read_json(dj)
        rt.append(dict(dataset=i["dataset"], protocol=i["protocol"], model=i["model"], epochs=i["epochs"],
                       n_train=i["n_train"], train_min=i["train_time_s"] / 60, gpu=i.get("env", {}).get("gpu", "cpu")))
    if rt:
        R = pd.DataFrame(rt)
        R = R.groupby(["dataset", "protocol", "model", "epochs", "n_train", "gpu"]).train_min.agg(["count", "mean", "sum"]).reset_index()
        R[["mean", "sum"]] = R[["mean", "sum"]].round(2)
        W.table(R.rename(columns={"count": "runs", "mean": "min/run", "sum": "total min"}), "tab_s1_runtime", "Measured training time.")
    W.close()

    # ============================================================== figures
    models_main = [m for m in MODEL_LABEL if m in set(s0.model)]
    def pooled_probs(model, method, ds=main_ds, loss="ce", seed=0, protocol="grouped"):
        P, Y = [], []
        for f in range(cfg["cv"]["n_folds"]):
            fp = res / "probs" / f"{ds}-{protocol}-{model}-{loss}-f{f}-s{seed}__{method}.npz"
            if fp.exists():
                z = np.load(fp)
                P.append(z["probs"]); Y.append(z["y"])
        return (np.concatenate(P), np.concatenate(Y)) if P else (None, None)

    # reliability diagrams
    rel_models = [m for m in ("resnet101", "hybrid_r101_incv3", "convnext_tiny", "deit_small") if m in models_main] or models_main[:3]
    if rel_models:
        fig, axes = plt.subplots(2, len(rel_models), figsize=(2.3 * len(rel_models), 4.6), squeeze=False)
        for c, m in enumerate(rel_models):
            for r, meth in enumerate(("raw", "ts")):
                ax = axes[r, c]
                P, Y = pooled_probs(m, meth)
                if P is None:
                    ax.axis("off"); continue
                cnt, conf, acc = ece_bins(P.max(1), P.argmax(1) == Y, 15)
                centers = (np.arange(15) + 0.5) / 15
                ok = cnt > 0
                ax.bar(centers[ok], acc[ok], width=1 / 15, edgecolor="black", linewidth=0.4, color="#4C72B0", label="Accuracy")
                ax.bar(centers[ok], conf[ok] - acc[ok], bottom=acc[ok], width=1 / 15, edgecolor="#C44E52",
                       linewidth=0.4, color="#F2B8B9", hatch="//", label="Gap")
                ax.plot([0, 1], [0, 1], "k--", lw=0.7)
                from .metrics import ece as _ece
                ax.set_title(f"{ml(m)} – {'uncalibrated' if meth == 'raw' else 'temp. scaled'}\nECE={_ece(P, Y):.3f}", fontsize=7)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                if r == 1: ax.set_xlabel("Confidence")
                if c == 0: ax.set_ylabel("Accuracy")
        axes[0, 0].legend(fontsize=6, loc="upper left", frameon=False)
        fig.tight_layout()
        savefig(fig, fig_dir, "fig_reliability_diagrams")
        plt.close(fig)

    # risk-coverage
    if models_main:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        for m in models_main:
            P, Y = pooled_probs(m, "ts")
            if P is None: continue
            cov, risk, a = risk_coverage(P.max(1), P.argmax(1) == Y)
            ax.plot(cov, risk, lw=1, label=f"{ml(m)} (AURC={a:.3f})")
        ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error rate)")
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0); ax.legend(fontsize=5.5, frameon=False)
        ax.axvline(0.8, color="grey", lw=0.5, ls=":"); ax.axvline(0.9, color="grey", lw=0.5, ls=":")
        fig.tight_layout(); savefig(fig, fig_dir, "fig_risk_coverage"); plt.close(fig)

    # conformal coverage vs set size
    if len(C):
        cg = C[(C.dataset == main_ds) & (C.protocol == "grouped") & (C.loss == "ce") & (C.preprocess == "none") & (C.method == "ts") & (C.seed >= 0)]
        if len(cg):
            fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
            mk = {"lac": "o", "aps": "s", "raps": "^"}
            for j, a in enumerate(sorted(set(cg.alpha))[:2]):
                ax = axes[j]
                for m in models_main:
                    for sc in ("lac", "aps", "raps"):
                        s = cg[(cg.model == m) & (cg.score == sc) & (cg.alpha == a) & (~cg.mondrian.astype(bool))]
                        if len(s):
                            ax.errorbar(s.avg_size.mean(), s.coverage.mean(), xerr=s.avg_size.std(), yerr=s.coverage.std(),
                                        marker=mk[sc], ms=3.5, lw=0.6, capsize=1.5, color=f"C{models_main.index(m)}")
                ax.axhline(1 - a, color="k", ls="--", lw=0.7)
                ax.set_title(f"α = {a}", fontsize=8); ax.set_xlabel("Average set size"); ax.set_ylabel("Empirical coverage")
            from matplotlib.lines import Line2D
            h = [Line2D([], [], color=f"C{i}", lw=1.5, label=ml(m)) for i, m in enumerate(models_main)]
            h += [Line2D([], [], color="k", marker=v, ls="", label=k.upper()) for k, v in mk.items()]
            axes[1].legend(handles=h, fontsize=5.5, frameon=False, loc="lower right")
            fig.tight_layout(); savefig(fig, fig_dir, "fig_conformal_coverage_size"); plt.close(fig)

            # class-conditional coverage marginal vs Mondrian
            m = ref_model if ref_model in models_main else models_main[0]
            classes = read_json(next((out / "runs").glob(f"{main_ds}-grouped-{m}-ce-f0-s0/done.json")))["classes"] if list((out / "runs").glob(f"{main_ds}-grouped-{m}-ce-f0-s0/done.json")) else None
            if classes:
                fig, ax = plt.subplots(figsize=(4.5, 2.6))
                w = 0.38
                for k, mond in enumerate((False, True)):
                    s = cg[(cg.model == m) & (cg.score == "aps") & (cg.alpha == 0.1) & (cg.mondrian.astype(bool) == mond)]
                    if not len(s): continue
                    cc = np.array([[float(v) for v in str(x).split(";")] for x in s.class_cov])
                    ax.bar(np.arange(len(classes)) + (k - 0.5) * w, np.nanmean(cc, 0), yerr=np.nanstd(cc, 0), width=w,
                           capsize=1.5, label="class-conditional (Mondrian)" if mond else "marginal",
                           color="#55A868" if mond else "#8172B2", error_kw=dict(lw=0.5))
                ax.axhline(0.9, color="k", ls="--", lw=0.7)
                ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=6.5)
                ax.set_ylim(0.5, 1.02); ax.set_ylabel("Coverage (APS, α = 0.1)"); ax.legend(fontsize=6, frameon=False)
                ax.set_title(ml(m), fontsize=8)
                fig.tight_layout(); savefig(fig, fig_dir, "fig_classwise_coverage"); plt.close(fig)

    # optimism gap
    g = single[(single.dataset == main_ds) & (single.method == "raw") & (single.loss == "ce") & (single.preprocess == "none")]
    gm_ = [m for m in REF_PAPER if m in set(g[g.protocol != "grouped"].model)]
    if gm_:
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        prot = [("ref", "Reported [Kumar 2026]"), ("naive_aug", "Naive + aug-before-split"), ("naive", "Naive random split"), ("grouped", "Leakage-aware CV")]
        w = 0.2
        for k, (p, lab) in enumerate(prot):
            vals, errs = [], []
            for m in gm_:
                if p == "ref":
                    vals.append(REF_PAPER[m]); errs.append(0)
                else:
                    s = g[(g.model == m) & (g.protocol == p)].acc
                    vals.append(s.mean() if len(s) else np.nan); errs.append(s.std() if len(s) > 1 else 0)
            ax.bar(np.arange(len(gm_)) + (k - 1.5) * w, vals, yerr=errs, width=w, label=lab, capsize=1.5,
                   color=["#999999", "#DD8452", "#CCB974", "#4C72B0"][k], error_kw=dict(lw=0.5))
        ax.set_xticks(range(len(gm_))); ax.set_xticklabels([ml(m) for m in gm_], fontsize=7)
        ax.set_ylabel("Accuracy"); lo = np.nanmin(g.acc) if len(g) else 0.5
        ax.set_ylim(max(0, lo - 0.1), 1.0); ax.legend(fontsize=5.5, frameon=False, loc="lower right")
        fig.tight_layout(); savefig(fig, fig_dir, "fig_optimism_gap"); plt.close(fig)

    # ECE across calibration approaches
    rows = []
    for m in models_main:
        for loss, meth, lab in approaches:
            s = gg[(gg.model == m) & (gg.loss == loss) & (gg.method == meth)]
            if len(s):
                rows.append((m, lab, s.ece.mean(), s.ece.std() if len(s) > 1 else 0))
    if rows:
        D = pd.DataFrame(rows, columns=["model", "approach", "mean", "sd"])
        labs = [a[2] for a in approaches if a[2] in set(D.approach)]
        fig, ax = plt.subplots(figsize=(7, 2.8))
        w = 0.8 / len(labs)
        for k, lab in enumerate(labs):
            d = D[D.approach == lab].set_index("model").reindex(models_main)
            ax.bar(np.arange(len(models_main)) + (k - len(labs) / 2 + 0.5) * w, d["mean"], yerr=d["sd"], width=w,
                   label=lab, color=plt.cm.tab10(k % 10), capsize=1, error_kw=dict(lw=0.4))
        ax.set_xticks(range(len(models_main))); ax.set_xticklabels([ml(m) for m in models_main], fontsize=7)
        ax.set_ylabel("ECE (15 bins)"); ax.legend(fontsize=5.5, ncol=2, frameon=False)
        fig.tight_layout(); savefig(fig, fig_dir, "fig_ece_calibration_methods"); plt.close(fig)

    # per-class F1 heatmap
    if len(PC):
        p = PC[(PC.dataset == main_ds) & (PC.protocol == "grouped") & (PC.loss == "ce") & (PC.preprocess == "none") & (PC.method == "raw") & (PC.seed >= 0)]
        if len(p):
            H = p.groupby(["model", "cls"]).f1.mean().unstack("cls").reindex([m for m in models_main if m in set(p.model)])
            fig, ax = plt.subplots(figsize=(0.65 * H.shape[1] + 1.8, 0.4 * H.shape[0] + 1.2))
            im = ax.imshow(H.values, cmap="viridis", vmin=max(0, np.nanmin(H.values) - 0.05), vmax=1)
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if H.values[i, j] < 0.85 else "black")
            ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=35, ha="right", fontsize=6.5)
            ax.set_yticks(range(H.shape[0])); ax.set_yticklabels([ml(m) for m in H.index], fontsize=6.5)
            fig.colorbar(im, ax=ax, fraction=0.03, label="F1")
            fig.tight_layout(); savefig(fig, fig_dir, "fig_perclass_f1"); plt.close(fig)

    # duplicate examples
    try:
        _dup_figure(cfg, main_ds, fig_dir, plt)
    except Exception as e:  # noqa: BLE001
        log(f"dup figure skipped: {e}")
    log(f"analysis written to {tab_dir} and {fig_dir}")


def _dup_figure(cfg, ds, fig_dir, plt, n=6):
    from PIL import Image
    wd = Path(cfg["work_dir"]) / ds
    pp = wd / "duplicate_pairs.csv"
    if not pp.exists():
        return
    P = pd.read_csv(pp)
    if len(P) == 0:
        return
    P = P.sort_values(["cross_class", "phash_dist"], ascending=[False, True]).head(n)
    raw = Path(cfg["data_root"]) / "raw" / ds
    fig, axes = plt.subplots(2, len(P), figsize=(1.6 * len(P), 3.4), squeeze=False)
    for c, (_, r) in enumerate(P.iterrows()):
        for k, col in enumerate(("path_i", "path_j")):
            ax = axes[k, c]
            ax.imshow(Image.open(raw / r[col]).convert("RGB"))
            ax.set_title(r[col].split("/")[-1][:22], fontsize=5)
            ax.axis("off")
        axes[1, c].text(0.5, -0.12, f"pHash d={r.phash_dist}, dHash d={r.dhash_dist}" + (" (cross-class)" if r.cross_class else ""),
                        transform=axes[1, c].transAxes, ha="center", va="top", fontsize=5)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig_duplicate_examples")
    plt.close(fig)
