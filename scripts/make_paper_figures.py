#!/usr/bin/env python
"""Regenerate the manuscript figures (Figs 1-6, Supplementary Figs S1-S7) with n1lib.figures.

Either point --results at a pipeline output folder (the out_dir of run_all.py / n1_finalize.py, which
contains tables/, results/ (with probs/ or probs.tar) and figures/), or use the published results:

    # published results: tables and CSVs from this repository (+ probs.tar, available on request)
    python scripts/make_paper_figures.py --raw data/raw --out build/paper_figures
    python scripts/make_paper_figures.py --probs-tar probs.tar --raw data/raw --out build/paper_figures

    # your own run
    python scripts/make_paper_figures.py --results outputs --raw data/raw --out build/paper_figures

Inputs: tables/*.csv (analyze stage), results/probs/*.npz (per-run test probabilities) or, without them,
the curve data exported from them in results/figdata/ (Fig 4a, Fig 6a, Figs S3-S5), splits/ (duplicate audit, Fig 2 and Fig S1), raw images (only for the
image panels of Fig 2c and Fig S7; skipped without --raw). Supplementary Fig S6 is the Grad-CAM render
written by the gradcam stage (figures/fig_gradcam_rejected.*), copied unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stage_published(staging: Path, probs_tar: Path | None, splits: Path):
    """Assemble an out_dir-like folder from the files shipped in this repository."""
    import pandas as pd
    (staging / "results").mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "results" / "tables", staging / "tables", dirs_exist_ok=True)
    for p in (ROOT / "results" / "data").glob("*.csv"):
        shutil.copyfile(p, staging / "results" / p.name)
    for p in (ROOT / "results" / "data").glob("*.csv.gz"):  # conformal_long / perclass_long are shipped gzipped
        with gzip.open(p, "rb") as fi, open(staging / "results" / p.name[:-3], "wb") as fo:
            shutil.copyfileobj(fi, fo)
    fd = ROOT / "results" / "figdata"  # curve data exported from probs.tar (export_figdata); used when no probs are given
    if fd.is_dir() and probs_tar is None:
        shutil.copytree(fd, staging / "results" / "figdata", dirs_exist_ok=True)
    gc = ROOT / "results" / "figures" / "paper" / "figS6_gradcam_rejected.png"  # Grad-CAM render (preview)
    (staging / "figures").mkdir(exist_ok=True)
    if gc.exists():
        shutil.copyfile(gc, staging / "figures" / "fig_gradcam_rejected.png")
    if probs_tar is not None and not (staging / "results" / "probs").exists():
        with tarfile.open(probs_tar) as tf:
            try:
                tf.extractall(staging / "results", filter="data")
            except TypeError:  # Python < 3.12
                tf.extractall(staging / "results")
    # class names for the fold-0 runs (export_figdata reads them from runs/*/done.json)
    for ds in ("realwaste", "trashnet", "trashbox"):
        idx = splits / "raw_artifacts" / ds / "index.csv"
        if not idx.exists():
            continue
        I = pd.read_csv(idx, usecols=["label", "cls"]).drop_duplicates().sort_values("label")
        d = staging / "runs" / f"{ds}-grouped-classes-f0-s0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "done.json").write_text(json.dumps({"classes": I.cls.tolist(), "marker_only": True}))
    return staging


def main():
    from n1lib import figstyle
    from n1lib import figures as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="pipeline out_dir (tables/, results/, figures/)")
    ap.add_argument("--probs-tar", default=None, help="probs.tar with per-run test probabilities (not in the repository; available from the corresponding author on reasonable request). Needed for Figs 4a, 6a, S3, S4")
    ap.add_argument("--splits", default=str(ROOT / "splits"), help="preprocessing export (duplicate audit)")
    ap.add_argument("--raw", default=None, help="raw images root: <raw>/<dataset>/<class>/<file>")
    ap.add_argument("--out", default=str(ROOT / "build" / "paper_figures"))
    ap.add_argument("--staging", default=str(ROOT / "build" / "paper_fig_inputs"))
    ap.add_argument("--dataset", default="realwaste")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    if a.results:
        root = Path(a.results)
        res = root / "results"
        if not (res / "probs").exists() and (res / "probs.tar").exists():
            with tarfile.open(res / "probs.tar") as tf:
                tf.extractall(res)
    else:
        root = stage_published(Path(a.staging), Path(a.probs_tar) if a.probs_tar else None, Path(a.splits))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figstyle.write_mplstyle(out / "srep.mplstyle")
    F.make_all(root, out, a.splits, a.raw, a.dataset, False, a.only)
    gc = root / "figures" / "fig_gradcam_rejected.png"
    if gc.exists() and (not a.only or "figS6" in a.only):
        for ext in ("png", "pdf"):
            p = gc.with_suffix(f".{ext}")
            if p.exists():
                shutil.copyfile(p, out / f"{F.FIGNAMES['figS6']}.{ext}")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
