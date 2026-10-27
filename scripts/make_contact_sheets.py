#!/usr/bin/env python
"""Contact sheets (JPG) of example duplicate groups, all cross-class duplicates and (TrashBox)
official test/val images that have a duplicate in the official train split.

Needs the raw images (data/raw/<dataset>/<class>/...) and the exported preprocessing package
(scripts/export_preprocessing.py). Only Pillow + pandas are required.

    python scripts/make_contact_sheets.py --package release/preprocessing --raw-root data/raw
    python scripts/make_contact_sheets.py --package <unzipped preprocessing.zip> --raw-root C:/path/to/data/raw --datasets trashbox

Output: <package>/contact_sheets/<dataset>_*.jpg
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TILE, CAP, HDR, MAXCOL = 170, 44, 22, 6


def font(size=12):
    for f in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


F, FB = font(11), font(13)


def thumb(path: Path) -> Image.Image:
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((TILE - 4, TILE - 4))
    except Exception:  # noqa: BLE001
        im = Image.new("RGB", (TILE - 4, TILE - 4), (200, 200, 200))
        ImageDraw.Draw(im).text((10, 70), "missing", fill="red", font=F)
    c = Image.new("RGB", (TILE - 4, TILE - 4), "white")
    c.paste(im, ((TILE - 4 - im.width) // 2, (TILE - 4 - im.height) // 2))
    return c


def draw_rows(rows, raw: Path, title: str, out: Path):
    """rows: list of (header text, [(relpath, caption), ...])."""
    if not rows:
        return None
    ncol = min(MAXCOL, max(len(r[1]) for r in rows))
    W = max(ncol * TILE, 640)
    H = 34 + sum(HDR + TILE + CAP for _ in rows)
    S = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(S)
    d.text((6, 8), title, fill="black", font=FB)
    y = 34
    for hdr, items in rows:
        d.text((6, y + 4), hdr, fill=(0, 0, 140), font=F)
        y += HDR
        for c, (rp, cap) in enumerate(items[:ncol]):
            x = c * TILE
            S.paste(thumb(raw / rp), (x + 2, y + 2))
            d.rectangle([x + 1, y + 1, x + TILE - 2, y + TILE - 2], outline=(180, 180, 180))
            d.multiline_text((x + 3, y + TILE + 1), "\n".join(textwrap.wrap(cap, 27)[:3]), fill="black", font=F, spacing=1)
        if len(items) > ncol:
            d.text((W - 90, y - HDR + 4), f"+{len(items) - ncol} more", fill="gray", font=F)
        y += TILE + CAP
        d.line([0, y - 2, W, y - 2], fill=(220, 220, 220))
    out.parent.mkdir(parents=True, exist_ok=True)
    S.save(out, quality=90)
    print("wrote", out)
    return out


def pick_groups(G: pd.DataFrame, n: int):
    """Deterministic, diverse choice: largest groups first, then one per criterion type round-robin."""
    info = G.groupby("group_id").agg(size=("relpath", "size"), crit=("criterion", lambda s: "+".join(sorted(set("+".join(s).split("+"))))),
                                     mixed=("group_classes", lambda s: "|" in s.iloc[0])).reset_index()
    chosen = list(info.sort_values(["size", "group_id"], ascending=[False, True]).group_id[:3])
    buckets = {k: list(v.sort_values("group_id").group_id) for k, v in info.groupby("crit")}
    while len(chosen) < min(n, len(info)) and any(buckets.values()):
        for k in sorted(buckets):
            while buckets[k] and buckets[k][0] in chosen:
                buckets[k].pop(0)
            if buckets[k] and len(chosen) < n:
                chosen.append(buckets[k].pop(0))
    return chosen, info.set_index("group_id")


def caption(r):
    ev = []
    if pd.notna(r.get("min_phash_dist")):
        ev.append(f"pH{int(r['min_phash_dist'])}/dH{int(r['min_dhash_dist'])}")
    if pd.notna(r.get("max_cosine")):
        ev.append(f"cos{float(r['max_cosine']):.3f}")
    return f"{Path(r['relpath']).name} [{r['class']}] {' '.join(ev)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="release/preprocessing")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--datasets", nargs="+", default=["realwaste", "trashnet", "trashbox"])
    ap.add_argument("--n-groups", type=int, default=12)
    a = ap.parse_args()
    pkg, out_dir = Path(a.package), Path(a.package) / "contact_sheets"
    for ds in a.datasets:
        raw = Path(a.raw_root) / ds
        gfile = pkg / ds / f"{ds}_duplicate_groups.csv"
        if not gfile.exists() or not raw.exists():
            print(f"skip {ds}: need {gfile} and {raw}")
            continue
        G = pd.read_csv(gfile)
        chosen, info = pick_groups(G, a.n_groups)
        rows = []
        for g in chosen:
            m = G[G.group_id == g].sort_values("relpath")
            rows.append((f"group {g}  size {int(info.loc[g, 'size'])}  criterion: {info.loc[g, 'crit']}"
                         + ("  MIXED LABELS" if info.loc[g, "mixed"] else ""),
                         [(r["relpath"], caption(r)) for _, r in m.iterrows()]))
        draw_rows(rows, raw, f"{ds}: example duplicate groups (MD5 | pHash<=6 & dHash<=6 | DINOv2 cos>=0.95)",
                  out_dir / f"{ds}_duplicate_group_examples.jpg")
        X = pd.read_csv(pkg / ds / f"{ds}_cross_class_duplicates.csv")
        rows = []
        for r in X.itertuples():
            ev = "MD5 exact" if "MD5" in r.criterion else f"pHash {r.phash_dist} / dHash {r.dhash_dist}"
            rows.append((f"group {r.group_id}: {r.class_i} vs {r.class_j} ({ev})",
                         [(r.relpath_i, f"{Path(r.relpath_i).name} [{r.class_i}]"),
                          (r.relpath_j, f"{Path(r.relpath_j).name} [{r.class_j}]")]))
        draw_rows(rows, raw, f"{ds}: all cross-class duplicate pairs ({len(X)})", out_dir / f"{ds}_cross_class_duplicates.jpg")
        if ds == "trashbox" and "official_split" in G.columns:
            rows = []
            for g, m in G.groupby("group_id"):
                sp = set(m.official_split)
                if "train" in sp and (sp & {"test", "val"}):
                    m = m.sort_values(["official_split", "relpath"])
                    rows.append((f"group {g}: official {'/'.join(sorted(sp))} ({info.loc[g, 'crit']})",
                                 [(r["relpath"], f"{r['official_split']}: {caption(r)}") for _, r in m.iterrows()]))
                if len(rows) >= a.n_groups:
                    break
            draw_rows(rows, raw, "TrashBox: official test/val images with a duplicate in the official train split (examples)",
                      out_dir / "trashbox_official_split_leakage_examples.jpg")


if __name__ == "__main__":
    main()
