#!/usr/bin/env python
"""Write contact sheets of detected duplicate pairs for manual review (recommended before the
full run, and for the manuscript's supplementary material).

    python scripts/inspect_duplicates.py --config configs/full.yaml --dataset realwaste
Output: work/<dataset>/review_hash_pairs.jpg and review_embedding_pairs.jpg (up to 60 pairs each).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from n1lib.common import load_config  # noqa: E402


def sheet(df, raw, out, label_col, n=60, tile=150):
    df = df.head(n)
    if len(df) == 0:
        return
    W = Image.new("RGB", (2 * tile + 260, tile * len(df)), "white")
    d = ImageDraw.Draw(W)
    for r, (_, row) in enumerate(df.iterrows()):
        for c, col in enumerate(("path_i", "path_j")):
            W.paste(Image.open(raw / row[col]).convert("RGB").resize((tile - 2, tile - 2)), (c * tile, r * tile))
        d.text((2 * tile + 5, r * tile + 5), f"{row['path_i']}\n{row['path_j']}\n{label_col}: {row[label_col]}", fill="black")
    W.save(out, quality=90)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--dataset", default="realwaste")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    wd = Path(cfg["work_dir"]) / a.dataset
    raw = Path(cfg["data_root"]) / "raw" / a.dataset
    if (wd / "duplicate_pairs.csv").exists():
        P = pd.read_csv(wd / "duplicate_pairs.csv")
        P["dist"] = P.phash_dist.astype(str) + "/" + P.dhash_dist.astype(str)
        sheet(P.sort_values("phash_dist"), raw, wd / "review_hash_pairs.jpg", "dist")
    if (wd / "embedding_pairs.csv").exists():
        E = pd.read_csv(wd / "embedding_pairs.csv")
        E["cosine"] = E.cosine.round(3)
        sheet(E.sort_values("cosine"), raw, wd / "review_embedding_pairs.jpg", "cosine")


if __name__ == "__main__":
    main()
