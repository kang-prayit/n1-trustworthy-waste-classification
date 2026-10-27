"""Publication figures for Scientific Reports (main Figs 1-6 + Supplementary Figs S1-S6, numbered in order of citation).

Every data figure is regenerated from analysis outputs only:
  <results_root>/tables/*.csv            (written by analyze.py; "mean ± sd" strings are parsed)
  <results_root>/results/figdata/*.csv   (per-sample-derived curves written by export_figdata())
  <results_root>/results/probs/*.npz     (used to (re)build figdata when present)
  <prep_dir>/                            (preprocessing export: duplicate audit, final numbers)
No number is typed in by hand. Panels whose inputs are missing are drawn as labelled placeholders.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import figstyle as S

MODEL_LABEL = {"resnet101": "ResNet101", "hybrid_r101_incv3": "ResNet101+InceptionV3",
               "efficientnetv2_s": "EfficientNetV2-S", "efficientnet_b0": "EfficientNet-B0",
               "mobilenetv3_large": "MobileNetV3-L", "convnext_tiny": "ConvNeXt-T", "deit_small": "DeiT-S",
               "tiny_cnn": "TinyCNN", "tiny_hybrid": "TinyHybrid"}
LABEL2KEY = {v: k for k, v in MODEL_LABEL.items()}
MODEL_ORDER = list(MODEL_LABEL)
DRAFT_TEXT = "DRAFT - quick run, not final results"
DS_LABEL = {"realwaste": "RealWaste", "trashnet": "TrashNet", "trashbox": "TrashBox"}
FIGNAMES = {
    "fig1": "fig1_pipeline_schematic", "fig2": "fig2_duplicate_audit", "fig3": "fig3_optimism_gap",
    "fig4": "fig4_reliability_calibration", "fig5": "fig5_conformal_coverage_size", "fig6": "fig6_risk_coverage",
    "figS1": "figS1_dedup_threshold_sensitivity", "figS2": "figS2_perclass_f1", "figS3": "figS3_confusion_matrices",
    "figS4": "figS4_perclass_ece", "figS5": "figS5_classwise_coverage", "figS6": "figS6_gradcam_rejected",
    "figS7": "figS7_split_leakage_example",
}


def ml(m):
    return MODEL_LABEL.get(m, m)


def mkey(label):
    return LABEL2KEY.get(label, label)


def morder(keys):
    return sorted(set(keys), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)


def mcol(m):
    return S.MODEL_COLOR.get(m, "#000000")


def mmark(m):
    return S.MODEL_MARKER.get(m, "o")


def parse_pm(v):
    """'0.861 ± 0.007' -> (0.861, 0.007); '0.9886' -> (0.9886, nan); '–' -> (nan, nan)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v), np.nan
    s = str(v).strip()
    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:±\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?", s)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else np.nan)


# ====================================================================== data access
class Source:
    def __init__(self, root, prep_dir=None, dataset="realwaste", n_bins=15):
        self.root = Path(root)
        self.tab = self.root / "tables" if (self.root / "tables").is_dir() else self.root
        self.res = self.root / "results" if (self.root / "results").is_dir() else None
        self.figdata = (self.res / "figdata") if self.res is not None else self.root / "figdata"
        self.legacy_fig = self.root / "figures"
        self.prep = Path(prep_dir) if prep_dir else None
        self.ds = dataset
        self.n_bins = n_bins

    def table(self, stem):
        hits = sorted(self.tab.glob(f"{stem}*.csv"))
        return pd.read_csv(hits[0]) if hits else None

    def fd(self, name):
        p = self.figdata / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None

    def runs_epochs(self):
        t = self.table("tab_s1_runtime")
        if t is not None and "epochs" in t:
            return t.epochs.astype(float).tolist()
        ep = []
        for dj in (self.root / "runs").glob("*/done.json"):
            try:
                ep.append(float(json.loads(dj.read_text())["epochs"]))
            except Exception:  # noqa: BLE001
                pass
        return ep

    def is_quick(self, min_epochs=10):
        ep = self.runs_epochs()
        return bool(ep) and max(ep) < min_epochs


def _save(fig, out_dir, wm, name, log=print):
    fmts = ("pdf", "tiff600", "png300") if wm else ("pdf", "eps", "tiff600", "png300")
    return S.save(fig, out_dir, name, formats=fmts, log=log)


_SHORT = {"Food Organics": "Food Org.", "Miscellaneous Trash": "Misc. Trash", "Textile Trash": "Textile"}


def _short(c):
    return _SHORT.get(c, c)


def placeholder(ax, text):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (3, 2))); s.set_color("#999999"); s.set_linewidth(1.0)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=6.5, color="#555555", transform=ax.transAxes,
            wrap=True)


def axmm(fig, x, y, w, h, **kw):
    """Axes from a rectangle in millimetres measured from the bottom-left corner of the figure."""
    W, H = fig.get_figwidth() / S.MM, fig.get_figheight() / S.MM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def plabel(fig, ax, label, dx=-9.0, dy=1.0):
    """Bold lower-case 8-pt panel label dx mm left of / dy mm above the axes' top-left corner."""
    W, H = fig.get_figwidth() / S.MM, fig.get_figheight() / S.MM
    p = ax.get_position()
    fig.text(p.x0 + dx / W, p.y1 + dy / H, label, fontsize=8, fontweight="bold", ha="left", va="bottom")


def newfig(w_mm, h_mm):
    import matplotlib.pyplot as plt
    assert h_mm <= 170 + 1e-6, "Nature/Sci Rep maximum figure height is 170 mm"
    return plt.figure(figsize=(w_mm * S.MM, h_mm * S.MM))


# ====================================================================== figdata export (per-sample -> csv)
def export_figdata(results_root, dataset="realwaste", n_bins=15, protocol="grouped", loss="ce", seed=0,
                   log=print):
    """Pool out-of-fold test probabilities (seed 0, all folds) and write the curve data behind
    Fig 4a, Fig 6a, Fig S3, Fig S4 and Fig S5 as small CSVs, so figures can be rebuilt from tables only."""
    from .metrics import ece_bins, risk_coverage
    root = Path(results_root)
    res = root / "results" if (root / "results").is_dir() else root
    pdir = res / "probs"
    out = res / "figdata"
    out.mkdir(parents=True, exist_ok=True)
    classes = None
    for dj in sorted((root / "runs").glob(f"{dataset}-{protocol}-*-f0-s0/done.json")):
        try:
            classes = json.loads(dj.read_text())["classes"]; break
        except Exception:  # noqa: BLE001
            pass
    models = set()
    for f in pdir.glob(f"{dataset}-{protocol}-*-{loss}-f*-s{seed}__raw.npz"):
        m = f.name[len(f"{dataset}-{protocol}-"):].split(f"-{loss}-f")[0]
        if not m.startswith("seg-"):
            models.add(m)
    models = morder(models)

    def pooled(m, meth):
        P, Y = [], []
        for f in sorted(pdir.glob(f"{dataset}-{protocol}-{m}-{loss}-f*-s{seed}__{meth}.npz")):
            z = np.load(f); P.append(z["probs"]); Y.append(z["y"])
        return (np.concatenate(P), np.concatenate(Y)) if P else (None, None)

    rel, rc, cm, pce = [], [], [], []
    for m in models:
        for meth in ("raw", "ts"):
            P, Y = pooled(m, meth)
            if P is None:
                continue
            K = P.shape[1]
            cls = classes if classes and len(classes) == K else [f"class {k}" for k in range(K)]
            conf, pred = P.max(1), P.argmax(1)
            corr = pred == Y
            cnt, c, a = ece_bins(conf, corr, n_bins)
            okb = cnt > 0
            e = float(np.sum(cnt[okb] / len(Y) * np.abs(c[okb] - a[okb])))
            for b in range(n_bins):
                rel.append(dict(model=m, method=meth, bin=b, lo=b / n_bins, hi=(b + 1) / n_bins, count=int(cnt[b]),
                                conf=c[b], acc=a[b], ece=e, n=len(Y)))
            cov, risk, aurc = risk_coverage(conf, corr)
            grid = np.linspace(0.01, 1, 100)
            for g_, r_ in zip(grid, np.interp(grid, cov, risk)):
                rc.append(dict(model=m, method=meth, coverage=g_, risk=r_, aurc=aurc, n=len(Y)))
            if meth == "raw":
                M = np.zeros((K, K), int)
                np.add.at(M, (Y, pred), 1)
                for i in range(K):
                    for j in range(K):
                        cm.append(dict(model=m, true=cls[i], pred=cls[j], count=int(M[i, j])))
            for k in range(K):
                pk, ok = P[:, k], (Y == k).astype(float)
                cntk, ck, ak = ece_bins(pk, ok, n_bins)
                mk = cntk > 0
                pce.append(dict(model=m, method=meth, cls=cls[k],
                                ece=float(np.sum(cntk[mk] / len(Y) * np.abs(ck[mk] - ak[mk]))), n=len(Y)))
    for name, rows in (("reliability_bins", rel), ("risk_coverage", rc), ("confusion_matrix", cm),
                       ("perclass_ece", pce)):
        if rows:
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    # class-wise conformal coverage (from conformal_long.csv)
    cl = res / "conformal_long.csv"
    if cl.exists():
        C = pd.read_csv(cl)
        C = C[(C.dataset == dataset) & (C.protocol == protocol) & (C.loss == loss) & (C.preprocess == "none")
              & (C.method == "ts") & (C.seed >= 0)]
        rows = []
        for (m, sc, a, mo), s in C.groupby(["model", "score", "alpha", "mondrian"]):
            cc = np.array([[float(v) for v in str(x).split(";")] for x in s.class_cov])
            cls = classes if classes and len(classes) == cc.shape[1] else [f"class {k}" for k in range(cc.shape[1])]
            for k in range(cc.shape[1]):
                rows.append(dict(model=m, score=sc, alpha=a, mondrian=bool(mo), cls=cls[k], mean=np.nanmean(cc[:, k]),
                                 sd=np.nanstd(cc[:, k], ddof=1) if len(cc) > 1 else np.nan, n=len(cc)))
        if rows:
            pd.DataFrame(rows).to_csv(out / "classwise_coverage.csv", index=False)
    log(f"figdata written to {out} ({len(models)} models)")
    return out


# ====================================================================== Fig 1 schematic
def fig1_pipeline(src, out_dir, wm=None, log=print):
    """Fig. 1 (illustrative vector schematic): (a) edge deployment at the sorting line - camera, calibrated
    softmax(z/T), conformal set C(x), accept (|C| = 1) or defer to operator review; (b) offline leakage-aware
    development - duplicate audit, group-stratified split, training/calibration, conformal threshold.
    Probabilities in (a) are illustrative, not model output. No data needed."""
    import matplotlib.pyplot as plt
    from . import schematics as SC
    H = 104.0
    fig = newfig(183, H)
    ax = axmm(fig, 0, 0, 183, H)
    ax.set_xlim(0, 183); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off"); ax.set_autoscale_on(False)
    SC.draw_fig1_scene(ax, H)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["fig1"], log=log)
    plt.close(fig)
    return files


def figS7_split_example(src, out_dir, raw_root=None, wm=None, log=print):
    """Supplementary Fig. S7 (illustrative): naive versus grouped split of one real RealWaste duplicate group
    (fold 0) and augment-before-split versus augment-inside-training. Needs the preprocessing export and raw images."""
    import matplotlib.pyplot as plt
    from . import schematics as SC
    prep = src.prep
    need = ["realwaste_folds_grouped.csv", "realwaste_split_naive_aug0.csv", "realwaste_split_naive_aug3.csv",
            "realwaste_duplicate_links.csv"]
    if prep is None or raw_root is None or not all((Path(prep) / "realwaste" / n).exists() for n in need) \
            or not (Path(raw_root) / "realwaste").is_dir():
        log("figS7: preprocessing export or raw RealWaste images missing, skipped"); return []
    H = 88.0
    fig = newfig(183, H)
    ax = axmm(fig, 0, 0, 183, H)
    ax.set_xlim(0, 183); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off"); ax.set_autoscale_on(False)
    info = SC.draw_figS7(ax, prep, raw_root, H)
    log(f"figS7: {info}")
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["figS7"], log=log)
    plt.close(fig)
    return files


# ====================================================================== Fig 2 duplicate audit
def _criterion_counts(links_csv, n_images):
    L = pd.read_csv(links_csv)
    res = {}
    for key, pat in (("MD5 (exact)", "MD5"), ("pHash & dHash", "pHash"), ("DINOv2 cosine", "DINOv2")):
        s = L[L.criterion.astype(str).str.contains(pat, regex=False)]
        imgs = set(s.relpath_i) | set(s.relpath_j)
        res[key] = 100 * len(imgs) / n_images
    return res


def fig2_duplicate_audit(src, out_dir, raw_root=None, wm=None, log=print, image_datasets=("realwaste", "trashnet")):
    import matplotlib.pyplot as plt
    if src.prep is None or not (src.prep / "preprocessing_summary.csv").exists():
        log("fig2: no preprocessing export, skipped"); return []
    P = pd.read_csv(src.prep / "preprocessing_summary.csv").set_index("dataset")
    dss = [d for d in ("realwaste", "trashnet", "trashbox") if d in P.index]
    fig = newfig(183, 128)
    # ---- a: % images per criterion
    ax = axmm(fig, 13, 72, 70, 50)
    crit_cols = {"MD5 (exact)": "#D55E00", "pHash & dHash": "#E69F00", "DINOv2 cosine": "#0072B2", "Any (union)": "#000000"}
    w = 0.2
    for i, d in enumerate(dss):
        lk = src.prep / d / f"{d}_duplicate_links.csv"
        vals = _criterion_counts(lk, int(P.loc[d, "n_images"])) if lk.exists() else {}
        vals["Any (union)"] = float(P.loc[d, "pct_images_in_duplicate_groups"])
        for k, (c, col) in enumerate(crit_cols.items()):
            v = vals.get(c, np.nan)
            x = i + (k - 1.5) * w
            ax.bar(x, v, width=w, color=col if c != "Any (union)" else "white", edgecolor="#000000", lw=1.0,
                   label=c if i == 0 else None, hatch="////" if c == "Any (union)" else None)
            if np.isfinite(v):
                ax.text(x, v + 0.25, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=5,
                        rotation=90)
    ax.set_xticks(range(len(dss))); ax.set_xticklabels([DS_LABEL[d] for d in dss])
    ax.set_ylabel("Images in duplicate groups (%)"); ax.set_ylim(0, max(13, float(P.pct_images_in_duplicate_groups.max()) * 1.3))
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, fontsize=6)
    plabel(fig, ax, "a", dx=-12)
    # ---- b: test-to-train leakage
    ax = axmm(fig, 118, 72, 62, 50)
    rows = []  # (label, value, colour)

    def gmax(d):
        gp = src.prep / "raw_artifacts" / d / "splits_grouped_report.json"
        if gp.exists():
            return max(x["pct_test_dupgroup_in_train"] for x in json.loads(gp.read_text()))
        return np.nan
    for d in dss:
        r = P.loc[d]
        if pd.notna(r.naive_aug0_leak):
            rows.append((f"{DS_LABEL[d]}, naive 70/15/15", float(r.naive_aug0_leak), "#E69F00"))
        if pd.notna(r.naive_aug3_leak):
            rows.append((f"{DS_LABEL[d]}, augment ×3 then split", float(r.naive_aug3_leak), "#D55E00"))
    if "trashbox" in dss:
        r = P.loc["trashbox"]
        for part, a, b, c in (("test", "tb_test_in_train", "tb_test_in_train_mh", "tb_test_in_train_md5"),
                              ("val", "tb_val_in_train", "tb_val_in_train_mh", "tb_val_in_train_md5")):
            if pd.notna(r[a]):
                rows.append((f"TrashBox, official {part}", (float(r[a]), float(r[b]), float(r[c])), "#CC79A7"))
    gm = [gmax(d) for d in dss]
    if np.isfinite(gm).any():
        rows.append(("All, grouped 5-fold (max)", float(np.nanmax(gm)), "#009E73"))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, col) in zip(y, rows):
        if isinstance(v, tuple):
            ax.barh(yy, v[0], color="#F4E1EC", edgecolor="#000000", lw=1.0, height=0.7)
            ax.barh(yy, v[1], color="#CC79A7", edgecolor="#000000", lw=1.0, height=0.7)
            ax.barh(yy, v[2], color="#7A3D63", edgecolor="#000000", lw=1.0, height=0.7)
            ax.text(v[0] + 1.5, yy, f"{v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}", va="center", fontsize=5.5)
        else:
            ax.barh(yy, v, color=col, edgecolor="#000000", lw=1.0, height=0.7)
            ax.text(min(v, 100) + 1.5 if v < 80 else v - 1.5, yy, f"{v:.2f}", va="center", fontsize=5.5,
                    ha="left" if v < 80 else "right", color="#000000" if v < 80 else "white")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_xlim(0, 100); ax.set_xlabel("Test items with a duplicate in train (%)")
    if "trashbox" in dss:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(fc="#F4E1EC", ec="k", label="any criterion"), Patch(fc="#CC79A7", ec="k", label="MD5 or hash"),
                           Patch(fc="#7A3D63", ec="k", label="MD5 only")], loc="center right", fontsize=5.5,
                  title="TrashBox official", handlelength=1.0, bbox_to_anchor=(1.0, 0.42))
    plabel(fig, ax, "b", dx=-37)
    # ---- c: image pairs (RealWaste and TrashNet only; TrashBox images are not shown: no licence)
    pairs = []
    if raw_root is not None:
        for d in image_datasets:
            lk = src.prep / d / f"{d}_duplicate_links.csv"
            if not lk.exists():
                continue
            L = pd.read_csv(lk)
            L["cross_class"] = L.cross_class.astype(str).str.lower() == "true"
            sel = []
            if d == "trashnet":
                sel += list(L[L.cross_class & L.criterion.str.contains("MD5")].head(2).itertuples())
                sel += list(L[(~L.cross_class) & L.criterion.str.contains("pHash")].sort_values("phash_dist").head(1).itertuples())
            else:
                sel += list(L[L.criterion.str.contains("pHash")].sort_values("phash_dist").head(2).itertuples())
                sel += list(L[L.criterion == "DINOv2cos>=0.95"].sort_values("cosine", ascending=False).head(1).itertuples())
            pairs += [(d, r) for r in sel]
    n = len(pairs)
    if n == 0:
        placeholder(axmm(fig, 2, 3, 179, 55), "duplicate example images unavailable (raw images not found)")
    else:
        from PIL import Image
        cw, gap, x0 = 28.0, 2.2, 1.5
        th = 22.0
        for c, (d, r) in enumerate(pairs):
            x = x0 + c * (cw + gap)
            for k, (pcol, ccol) in enumerate((("relpath_i", "class_i"), ("relpath_j", "class_j"))):
                a = axmm(fig, x, 33 - k * (th + 4.5), cw, th)
                p = Path(raw_root) / d / getattr(r, pcol)
                if p.exists():
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((700, 700))
                    a.imshow(im, aspect="auto", interpolation="lanczos")
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_visible(True); s.set_linewidth(1.0)
                    s.set_color("#D55E00" if r.cross_class else "#000000")
                a.text(0.0, 1.02, f"{getattr(r, ccol)} · {Path(getattr(r, pcol)).stem.split('_')[-1]}", fontsize=5,
                       transform=a.transAxes, ha="left", va="bottom")
            crit = str(r.criterion)
            bits = []
            if "MD5" in crit:
                bits.append("MD5 identical")
            if "pHash" in crit and pd.notna(r.phash_dist):
                bits.append(f"pHash {int(r.phash_dist)}, dHash {int(r.dhash_dist)}")
            if "DINOv2" in crit and pd.notna(r.cosine):
                bits.append(f"cos {r.cosine:.3f}")
            lab = DS_LABEL[d] + (", cross-class" if r.cross_class else "") + "\n" + "; ".join(bits)
            fig.text((x + cw / 2) / 183, 1.0 / 128, lab, fontsize=5, ha="center", va="bottom", linespacing=1.2)
        fig.text(0.3 / 183, (33 + th + 4.6) / 128, "c", fontsize=8, fontweight="bold", ha="left", va="bottom")
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["fig2"], log=log)
    plt.close(fig)
    return files


# ====================================================================== Fig 3 optimism gap
PROTO = [("Kumar et al. 2026 (reported, single split)", "Reported (Kumar et al.)"),
         ("Naive + aug-before-split", "Naive, augment before split"),
         ("Naive random split", "Naive random split"),
         ("Leakage-aware grouped 5-fold CV", "Leakage-aware grouped CV")]


def fig3_optimism_gap(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    T = src.table("tab2_optimism_gap")
    if T is None or len(T) == 0:
        log("fig3: tab2 missing, skipped"); return []
    fig = newfig(120, 52)
    models = [mkey(m) for m in T.Model]
    for j, (metric, xlabel) in enumerate((("acc", "Test accuracy"), ("macro-F1", "Macro-F1"))):
        ax = axmm(fig, 40 + j * 42, 10, 36, 38)
        ylab = []
        for i, (col, lab) in enumerate(PROTO):
            ylab.append(lab)
            for k, (m, (_, r)) in enumerate(zip(models, T.iterrows())):
                if i == 0:
                    if metric != "acc":
                        continue
                    v, s = parse_pm(r[col])
                else:
                    v, s = parse_pm(r.get(f"{col} {metric}", np.nan))
                if not np.isfinite(v):
                    continue
                yy = len(PROTO) - 1 - i + (k - (len(models) - 1) / 2) * 0.22
                ax.errorbar(v, yy, xerr=s if np.isfinite(s) else None, fmt=mmark(m), color=mcol(m),
                            mfc="white" if i == 0 else mcol(m), ms=4, elinewidth=1.0, capsize=1.5, lw=0)
        ax.set_yticks(range(len(PROTO))[::-1])
        ax.set_yticklabels(ylab if j == 0 else [""] * len(PROTO))
        ax.set_xlabel(xlabel); ax.set_ylim(-0.6, len(PROTO) - 0.4)
        ax.axhline(0.5, color="#999999", lw=1.0, ls=(0, (2, 2)))
        plabel(fig, ax, "ab"[j], dx=-38 if j == 0 else -4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=mcol(m), marker=mmark(m), ls="", ms=4, label=ml(m)) for m in models]
    h.append(Line2D([], [], color="#000000", marker="o", mfc="white", ls="", ms=4, label="reported value"))
    fig.axes[-1].legend(handles=h, loc="upper left", fontsize=5.5, handletextpad=0.3)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["fig3"], log=log)
    plt.close(fig)
    return files


# ====================================================================== Fig 4 calibration
APPROACH_ORDER = ["CE (uncalibrated)", "CE + temperature scaling", "CE + vector scaling", "CE + MC-dropout",
                  "Label smoothing", "Label smoothing + TS", "Focal loss", "Focal loss + TS", "Deep ensemble",
                  "Deep ensemble + TS"]


def _rel_models(models):
    pref = [m for m in ("resnet101", "hybrid_r101_incv3", "convnext_tiny", "deit_small") if m in models]
    return pref or models[:4]


def fig4_calibration(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    T4 = src.table("tab4_calibration_methods")
    R = src.fd("reliability_bins")
    fig = newfig(183, 112)
    # ---- a: reliability diagrams (raw vs TS)
    if R is not None and len(R):
        rm = _rel_models(morder(R.model))
        wa = (183 - 14 - 6 * (len(rm) - 1)) / len(rm)
        wa = min(wa, 40)
        for c, m in enumerate(rm):
            ax = axmm(fig, 14 + c * (wa + 6), 64, wa, 40)
            ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls=(0, (2, 2)))
            for meth, col, mk, lab in (("raw", "#D55E00", "o", "uncalibrated"), ("ts", "#0072B2", "s", "temperature scaled")):
                d = R[(R.model == m) & (R.method == meth) & (R["count"] > 0)]
                if not len(d):
                    continue
                ax.plot(d.conf, d.acc, color=col, marker=mk, ms=2.8, lw=1.25,
                        label=f"{lab} (ECE {d.ece.iloc[0]:.3f})")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("auto")
            ax.set_xlabel("Confidence")
            if c == 0:
                ax.set_ylabel("Accuracy")
            else:
                ax.set_yticklabels([])
            ax.text(0.03, 0.97, ml(m), transform=ax.transAxes, fontsize=6.5, va="top", fontweight="bold")
            ax.legend(loc="lower right", fontsize=5, handlelength=1.2)
            if c == 0:
                plabel(fig, ax, "a", dx=-12)
    else:
        ax = axmm(fig, 14, 64, 166, 40)
        placeholder(ax, "Reliability diagrams (uncalibrated vs temperature-scaled), pooled out-of-fold test predictions\n"
                        "requires results/figdata/reliability_bins.csv or results/probs/*.npz - not in the quick-run export")
        plabel(fig, ax, "a", dx=-12)
    # ---- b: ECE across approaches
    ax = axmm(fig, 42, 10, 138, 40)
    if T4 is not None and len(T4):
        appr = [a for a in APPROACH_ORDER if a in set(T4.Approach)] + [a for a in dict.fromkeys(T4.Approach) if a not in APPROACH_ORDER]
        models = morder(mkey(m) for m in T4.Model)
        n = len(models)
        for k, m in enumerate(models):
            d = T4[T4.Model == ml(m)]
            for _, r in d.iterrows():
                v, s = parse_pm(r.ECE)
                i = appr.index(r.Approach)
                yy = len(appr) - 1 - i + (k - (n - 1) / 2) * min(0.18, 0.8 / n)
                ax.errorbar(v, yy, xerr=s if np.isfinite(s) else None, fmt=mmark(m), color=mcol(m), ms=3.2,
                            elinewidth=1.0, capsize=1.2, lw=0, label=ml(m) if _ == d.index[0] else None)
        ax.set_yticks(range(len(appr))[::-1]); ax.set_yticklabels(appr, fontsize=6)
        ax.set_ylim(-0.6, len(appr) - 0.4); ax.set_xlim(left=0)
        ax.set_xlabel(f"Expected calibration error (ECE, {src.n_bins} bins)")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(models), fontsize=6, borderaxespad=0.2)
    else:
        placeholder(ax, "tab4_calibration_methods.csv missing")
    plabel(fig, ax, "b", dx=-40)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["fig4"], log=log)
    plt.close(fig)
    return files


# ====================================================================== Fig 5 conformal
SCORE_MARK = {"LAC": "o", "APS": "s", "RAPS": "^"}


def fig5_conformal(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    T = src.table("tab5_conformal")
    if T is None or len(T) == 0:
        log("fig5: tab5 missing, skipped"); return []
    T = T.copy()
    T["alpha"] = T["α"].astype(float)
    T["mond"] = T["Class-conditional"].astype(str).str.lower().isin(["yes", "true"])
    alphas = sorted(set(T.alpha), reverse=True)[:2]
    models = morder(mkey(m) for m in T.Model)
    fig = newfig(183, 66)
    for j, a in enumerate(alphas):
        ax = axmm(fig, 12 + j * 58, 11, 48, 42)
        d = T[T.alpha == a]
        for _, r in d.iterrows():
            m = mkey(r.Model)
            cv, cs = parse_pm(r.Coverage); sv, ss = parse_pm(r["Avg set size"])
            ax.errorbar(sv, cv, xerr=ss if np.isfinite(ss) else None, yerr=cs if np.isfinite(cs) else None,
                        fmt=SCORE_MARK.get(r.Score, "o"), color=mcol(m), mfc="white" if r.mond else mcol(m),
                        ms=3.5, elinewidth=1.0, capsize=1.2, lw=0)
        ax.axhline(1 - a, color="#000000", lw=1.0, ls=(0, (3, 2)))
        ax.text(0.98, 1 - a, f"target {1 - a:.2f}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.5)
        ax.set_xlabel("Average prediction-set size")
        ax.set_ylabel("Empirical coverage" if j == 0 else "")
        ax.text(0.03, 0.03, f"α = {a:g}", transform=ax.transAxes, fontsize=6.5, fontweight="bold")
        plabel(fig, ax, "ab"[j], dx=-11)
    # c: worst-class coverage marginal -> Mondrian at the larger alpha
    ax = axmm(fig, 138, 11, 42, 42)
    a = alphas[0]
    d = T[T.alpha == a]
    labs, y = [], 0
    for m in models:
        for sc in ("LAC", "APS", "RAPS"):
            r0 = d[(d.Model == ml(m)) & (d.Score == sc) & (~d.mond)]
            r1 = d[(d.Model == ml(m)) & (d.Score == sc) & (d.mond)]
            if not len(r0) and not len(r1):
                continue
            v0 = parse_pm(r0["Worst-class coverage"].iloc[0])[0] if len(r0) else np.nan
            v1 = parse_pm(r1["Worst-class coverage"].iloc[0])[0] if len(r1) else np.nan
            ax.plot([v0, v1], [y, y], color=mcol(m), lw=1.25)
            ax.plot(v0, y, SCORE_MARK[sc], color=mcol(m), ms=3.5)
            ax.plot(v1, y, SCORE_MARK[sc], color=mcol(m), mfc="white", ms=3.5)
            labs.append(f"{ml(m)} {sc}"); y += 1
    ax.axvline(1 - a, color="#000000", lw=1.0, ls=(0, (3, 2)))
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([l.split(" ")[-1] for l in labs], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"Worst-class coverage (α = {a:g})")
    plabel(fig, ax, "c", dx=-9)
    h = [Line2D([], [], color=mcol(m), lw=1.5, label=ml(m)) for m in models]
    h += [Line2D([], [], color="#000000", marker=v, ls="", ms=3.5, label=k) for k, v in SCORE_MARK.items()]
    h += [Line2D([], [], color="#000000", marker="o", ls="", ms=3.5, label="marginal"),
          Line2D([], [], color="#000000", marker="o", mfc="white", ls="", ms=3.5, label="class-conditional")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(h), fontsize=5.5, handlelength=1.2,
               columnspacing=1.0, handletextpad=0.3)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["fig5"], log=log)
    plt.close(fig)
    return files


# ====================================================================== Fig 6 selective prediction
def fig6_selective(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    T = src.table("tab6_selective_prediction")
    RC = src.fd("risk_coverage")
    fig = newfig(183, 66)
    ax = axmm(fig, 12, 11, 70, 50)
    models = morder(mkey(m) for m in T.Model) if T is not None else []
    if RC is not None and len(RC):
        d0 = RC[RC.method == "ts"]
        for m in morder(d0.model):
            d = d0[d0.model == m]
            ax.plot(d.coverage, d.risk, color=mcol(m), lw=1.25, label=f"{ml(m)} (AURC {d.aurc.iloc[0]:.3f})")
        ax.set_xlim(0, 1)
    elif T is not None:
        for m in models:
            r = T[T.Model == ml(m)].iloc[0]
            pts = [(1.0, r["Full-coverage acc"]), (0.9, r["Acc@90% (MSP)"]), (0.8, r["Acc@80% (MSP)"])]
            xs = [p[0] for p in pts]
            ys = [1 - parse_pm(p[1])[0] for p in pts]
            es = [parse_pm(p[1])[1] for p in pts]
            ax.errorbar(xs, ys, yerr=es, color=mcol(m), marker=mmark(m), ms=3.5, lw=1.0, ls=(0, (2, 1.5)),
                        elinewidth=1.0, capsize=1.2, label=ml(m))
        ax.text(0.02, 0.98, "operating points only\n(full curves need per-sample probabilities)",
                transform=ax.transAxes, fontsize=5.5, va="top", color="#555555")
        ax.set_xlim(0.75, 1.02)
    for c in (0.8, 0.9):
        ax.axvline(c, color="#999999", lw=1.0, ls=(0, (1, 2)))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage (fraction of items sorted automatically)")
    ax.set_ylabel("Selective risk (error rate)")
    ax.legend(loc="center left" if RC is None else "upper left", fontsize=5.5, bbox_to_anchor=(0.0, 0.62) if RC is None else None)
    plabel(fig, ax, "a", dx=-11)
    # b: review load vs accuracy of auto-sorted items
    ax = axmm(fig, 106, 11, 74, 50)
    if T is not None and len(T):
        for m in models:
            r = T[T.Model == ml(m)].iloc[0]
            pts = [(0, r["Full-coverage acc"]), (10, r["Acc@90% (MSP)"]), (20, r["Acc@80% (MSP)"])]
            xs = [p[0] for p in pts]; ys = [parse_pm(p[1])[0] for p in pts]; es = [parse_pm(p[1])[1] for p in pts]
            ax.errorbar(xs, ys, yerr=es, color=mcol(m), marker=mmark(m), ms=3.5, lw=1.0, elinewidth=1.0, capsize=1.2,
                        label=f"{ml(m)}, confidence threshold")
            lv, ls_ = parse_pm(r["Human review load (%)"])
            av, as_ = parse_pm(r["Singleton acc (APS α=0.1)"])
            ax.errorbar(lv, av, xerr=ls_, yerr=as_, color=mcol(m), marker=mmark(m), mfc="white", ms=4.5, lw=0,
                        elinewidth=1.0, capsize=1.2, label=f"{ml(m)}, APS singleton (α = 0.1)")
        ax.set_xlabel("Items deferred to operator review (%)")
        ax.set_ylabel("Accuracy of automatically sorted items")
        ax.set_xlim(left=-2)
        ax.legend(loc="lower right", fontsize=5.0, ncol=1, labelspacing=0.3)
    else:
        placeholder(ax, "tab6 missing")
    plabel(fig, ax, "b", dx=-12)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["fig6"], log=log)
    plt.close(fig)
    return files


# ====================================================================== Supplementary
def figS6_gradcam(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    npz = src.figdata / "gradcam_panels.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        imgs, cams, meta = z["imgs"], z["cams"], json.loads(str(z["meta"]))
        models = list(dict.fromkeys(mm["model"] for mm in meta))
        ncol = max(sum(1 for mm in meta if mm["model"] == m) for m in models)
        import matplotlib
        cmap = matplotlib.colormaps["magma"]
        cw = (183 - 8) / ncol
        rh = cw + 9
        h = min(len(models) * rh + 6, 170)
        fig = newfig(183, h)
        for r_, m in enumerate(models):
            idx = [i for i, mm in enumerate(meta) if mm["model"] == m]
            for c, i in enumerate(idx):
                a = axmm(fig, 7 + c * cw, h - 4 - (r_ + 1) * rh + 8, cw - 1.5, cw - 1.5)
                # alpha-blend in RGB (no transparency => identical PDF/EPS/TIFF output)
                base = imgs[i].astype(float) / (255.0 if imgs[i].max() > 1 else 1.0)
                heat = cmap(np.clip(cams[i], 0, 1))[..., :3]
                a.imshow(np.clip(0.55 * base + 0.45 * heat, 0, 1))
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                mm = meta[i]
                a.text(0, -0.03, f"true: {mm['true']}\npred: {mm['pred']} ({mm['conf']:.2f}); |C| = {mm['set_size']}",
                       transform=a.transAxes, fontsize=5, va="top")
            fig.text(2 / 183, (h - 4 - r_ * rh - rh / 2 + 4) / h, ml(m), rotation=90, fontsize=6.5, va="center")
            plabel(fig, fig.axes[-len(idx)], "abcdef"[r_], dx=-6)
    else:
        legacy = src.legacy_fig / "fig_gradcam_rejected.png"
        fig = newfig(183, 75)
        a = axmm(fig, 0, 0, 183, 75)
        if legacy.exists():
            from PIL import Image
            im = Image.open(legacy).convert("RGB")
            a.imshow(im); a.axis("off")
            a.text(0.5, 0.0, "legacy analyze.py render (quick run); restyled automatically from results/figdata/gradcam_panels.npz on the full run",
                   transform=a.transAxes, ha="center", va="bottom", fontsize=5.5, color="#555555")
        else:
            placeholder(a, "Grad-CAM of deferred samples: requires results/figdata/gradcam_panels.npz (written by gradcam.py)")
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["figS6"], log=log)
    plt.close(fig)
    return files


def figS3_confusion(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    CM = src.fd("confusion_matrix")
    if CM is None or not len(CM):
        fig = newfig(183, 60)
        placeholder(axmm(fig, 5, 5, 173, 50), "Row-normalised confusion matrices (pooled out-of-fold test predictions, seed 0)\n"
                    "requires results/figdata/confusion_matrix.csv or results/probs/*.npz - not in the quick-run export")
    else:
        models = morder(CM.model)
        ncol = min(3, len(models)); nrow = int(np.ceil(len(models) / ncol))
        cell = 50
        h = min(170, nrow * (cell + 14) + 4)
        fig = newfig(183, h)
        for k, m in enumerate(models):
            d = CM[CM.model == m]
            cls = list(dict.fromkeys(d["true"]))
            M = d.pivot(index="true", columns="pred", values="count").reindex(index=cls, columns=cls).values.astype(float)
            Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)
            r_, c = divmod(k, ncol)
            ax = axmm(fig, 22 + c * 54, h - (r_ + 1) * (cell + 14) + 12, cell - 18, cell - 18)
            ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(cls)):
                for j in range(len(cls)):
                    if Mn[i, j] >= 0.01:
                        ax.text(j, i, f"{Mn[i, j]:.2f}"[1:] if Mn[i, j] < 1 else "1", ha="center", va="center", fontsize=4,
                                color="white" if Mn[i, j] > 0.6 else "black")
            ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
            ax.set_xticklabels([_short(c_) for c_ in cls], rotation=90, fontsize=4.5)
            ax.set_yticklabels([_short(c_) for c_ in cls] if c == 0 else [], fontsize=4.5)
            ax.set_title(ml(m), fontsize=6.5, pad=2)
            plabel(fig, ax, "abcdefghi"[k], dx=-6 if c else -20)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["figS3"], log=log)
    plt.close(fig)
    return files


def _heat(fig, ax, H, rows, cols, label, fmt="{:.2f}", cmap="viridis", vmin=None, vmax=None, sd=None):
    im = ax.imshow(H, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.isfinite(H[i, j]):
                t = fmt.format(H[i, j])
                rgba = im.cmap(im.norm(H[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, t, ha="center", va="center", fontsize=5, color="black" if lum > 0.5 else "white")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, fontsize=6); cb.outline.set_linewidth(1.0); cb.ax.tick_params(labelsize=5.5, width=1.0)
    return im


def figS4_perclass_ece(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    D = src.fd("perclass_ece")
    fig = newfig(183, 70)
    if D is None or not len(D):
        placeholder(axmm(fig, 5, 5, 173, 60), "Per-class (one-vs-rest) ECE before and after temperature scaling\n"
                    "requires results/figdata/perclass_ece.csv or results/probs/*.npz - not in the quick-run export")
    else:
        for j, meth in enumerate(("raw", "ts")):
            d = D[D.method == meth]
            if not len(d):
                continue
            models = morder(d.model); cls = list(dict.fromkeys(d.cls))
            H = d.pivot(index="model", columns="cls", values="ece").reindex(index=models, columns=cls).values
            ax = axmm(fig, 30 + j * 80, 20, 62, 42)
            _heat(fig, ax, H, [ml(m) for m in models] if j == 0 else [""] * len(models), cls, "one-vs-rest ECE",
                  fmt="{:.3f}", vmin=0, vmax=float(np.nanmax(D.ece)))
            ax.set_title("uncalibrated" if meth == "raw" else "temperature scaled", fontsize=6.5, pad=2)
            plabel(fig, ax, "ab"[j], dx=-28 if j == 0 else -4)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["figS4"], log=log)
    plt.close(fig)
    return files


def figS2_perclass_f1(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    T = src.table("tab7_perclass_f1")
    if T is None or not len(T):
        return []
    models = morder(mkey(m) for m in T.Model)
    T = T.set_index(T.Model.map(mkey)).reindex(models)
    cls = [c for c in T.columns if c != "Model"]
    H = np.array([[parse_pm(T.loc[m, c])[0] for c in cls] for m in models])
    fig = newfig(136, 22 + 6 * len(models))
    ax = axmm(fig, 30, 16, 92, 6 * len(models))
    _heat(fig, ax, H, [ml(m) for m in models], cls, "F1", vmin=np.floor(np.nanmin(H) * 10) / 10, vmax=1)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["figS2"], log=log)
    plt.close(fig)
    return files


def figS5_classwise_coverage(src, out_dir, wm=None, log=print, alpha=0.1, score="aps"):
    import matplotlib.pyplot as plt
    D = src.fd("classwise_coverage")
    fig = newfig(183, 62)
    if D is None or not len(D):
        placeholder(axmm(fig, 5, 5, 173, 52), f"Class-wise coverage, marginal vs class-conditional ({score.upper()}, α = {alpha})\n"
                    "requires results/figdata/classwise_coverage.csv or results/conformal_long.csv - not in the quick-run export")
    else:
        d = D[(D.score.str.lower() == score) & (np.isclose(D.alpha.astype(float), alpha))]
        models = morder(d.model)[:3]
        cls = list(dict.fromkeys(d.cls))
        for k, m in enumerate(models):
            ax = axmm(fig, 12 + k * 57, 22, 52, 34)
            w = 0.38
            for j, mo in enumerate((False, True)):
                s = d[(d.model == m) & (d.mondrian.astype(str).str.lower() == str(mo).lower())].set_index("cls").reindex(cls)
                ax.bar(np.arange(len(cls)) + (j - 0.5) * w, s["mean"], yerr=s["sd"], width=w,
                       color="#0072B2" if not mo else "white", edgecolor="#0072B2", lw=1.0,
                       error_kw=dict(lw=1.0, capsize=1.0), label="class-conditional" if mo else "marginal")
            ax.axhline(1 - alpha, color="#000000", lw=1.0, ls=(0, (3, 2)))
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(cls))); ax.set_xticklabels([_short(c) for c in cls], rotation=60, ha="right", fontsize=5)
            ax.set_ylabel("Coverage" if k == 0 else "")
            ax.text(0.02, 1.0, ml(m), transform=ax.transAxes, fontsize=6, va="bottom", fontweight="bold")
            if k == 0:
                ax.legend(loc="lower left", fontsize=5.5)
            plabel(fig, ax, "abc"[k], dx=-9)
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["figS5"], log=log)
    plt.close(fig)
    return files


def figS1_threshold(src, out_dir, wm=None, log=print):
    import matplotlib.pyplot as plt
    if src.prep is None:
        return []
    fig = newfig(136, 55)
    found = False
    for j, (pat, xlabel, chosen) in enumerate((("pHash", "pHash Hamming threshold t (pairs with distance ≤ t)", 6),
                                               ("DINOv2", "DINOv2 cosine threshold t (pairs with cosine ≥ t)", 0.95))):
        ax = axmm(fig, 13 + j * 64, 12, 52, 38)
        for d, col, mk in (("realwaste", "#0072B2", "o"), ("trashnet", "#D55E00", "s"), ("trashbox", "#999999", "^")):
            p = src.prep / d / f"{d}_threshold_sensitivity.csv"
            if not p.exists():
                continue
            T = pd.read_csv(p)
            T = T[T.criterion.str.contains(pat, regex=False)]
            if not len(T):
                continue
            found = True
            ax.plot(T.threshold, T.n_pairs, color=col, marker=mk, ms=3, lw=1.25, label=DS_LABEL[d])
        ax.axvline(chosen, color="#000000", lw=1.0, ls=(0, (3, 2)))
        ax.set_yscale("symlog", linthresh=1, linscale=0.5); ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_ylabel("Candidate pairs (symmetric log scale)" if j == 0 else "")
        if j == 0:
            ax.legend(loc="upper left", fontsize=5.5)
        plabel(fig, ax, "ab"[j], dx=-11)
    if not found:
        plt.close(fig); return []
    S.watermark(fig, wm)
    files = _save(fig, out_dir, wm, FIGNAMES["figS1"], log=log)
    plt.close(fig)
    return files


# ====================================================================== driver
def make_all(results_root, out_dir, prep_dir=None, raw_root=None, dataset="realwaste", draft=None, only=None,
             log=print):
    """Build every figure. draft: None -> automatic watermark when the run is a quick run (<10 epochs);
    True/False forces it on/off; a string sets custom watermark text."""
    import matplotlib
    matplotlib.use("Agg")
    font = S.apply()
    src = Source(results_root, prep_dir, dataset)
    if src.res is not None and (src.res / "probs").is_dir() and not (src.figdata / "reliability_bins.csv").exists():
        export_figdata(src.root, dataset, log=log)
    if draft is None:
        wm = DRAFT_TEXT if src.is_quick() else None
    elif draft is True:
        wm = DRAFT_TEXT
    elif draft is False:
        wm = None
    else:
        wm = str(draft)
    log(f"font: {font}; watermark: {wm!r}")
    jobs = {
        "fig1": lambda: fig1_pipeline(src, out_dir, None, log),            # schematic: design, not results
        "fig2": lambda: fig2_duplicate_audit(src, out_dir, raw_root, None, log),  # final preprocessing numbers
        "fig3": lambda: fig3_optimism_gap(src, out_dir, wm, log),
        "fig4": lambda: fig4_calibration(src, out_dir, wm, log),
        "fig5": lambda: fig5_conformal(src, out_dir, wm, log),
        "fig6": lambda: fig6_selective(src, out_dir, wm, log),
        "figS1": lambda: figS1_threshold(src, out_dir, None, log),       # final preprocessing numbers
        "figS2": lambda: figS2_perclass_f1(src, out_dir, wm, log),
        "figS3": lambda: figS3_confusion(src, out_dir, wm, log),
        "figS4": lambda: figS4_perclass_ece(src, out_dir, wm, log),
        "figS5": lambda: figS5_classwise_coverage(src, out_dir, wm, log),
        "figS6": lambda: figS6_gradcam(src, out_dir, wm, log),
        "figS7": lambda: figS7_split_example(src, out_dir, raw_root, None, log),  # illustrative, preprocessing only
    }
    written = {}
    for k, f in jobs.items():
        if only and k not in only:
            continue
        try:
            written[k] = f()
            log(f"{k}: {written[k]}")
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"{k} FAILED: {e}\n{traceback.format_exc()}")
            written[k] = []
    return written
