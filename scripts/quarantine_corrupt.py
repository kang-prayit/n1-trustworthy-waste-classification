#!/usr/bin/env python
"""Standalone copy of the corrupt-image quarantine step (identical logic to
the corrupt-file check used before indexing; the quarantined list is in splits/quarantined_corrupt_files.csv). Run after downloading, before indexing:

    python quarantine_corrupt.py --data-root data            # all datasets in data/raw/
    python quarantine_corrupt.py --data-root data --datasets trashbox --dry-run

A file is quarantined if PIL cannot fully decode it (Image.open(p).convert('RGB').load() raises,
truncated-image loading disabled). Files are moved data/raw/<ds>/<rel> -> data/corrupt/<ds>/<rel>
and listed (dataset-relative) in data/corrupt/quarantined_files.txt.
"""
import argparse
import os
import shutil
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.data_root)
    dsets = a.datasets or sorted(p.name for p in (root / "raw").iterdir() if p.is_dir())
    moved = []
    for ds in dsets:
        raw = root / "raw" / ds
        for dp, _, fs in os.walk(raw):
            for f in sorted(fs):
                if not f.lower().endswith(IMG_EXT):
                    continue
                p = Path(dp) / f
                try:
                    with Image.open(p) as im:
                        im.convert("RGB").load()
                except Exception as e:  # noqa: BLE001
                    rel = p.relative_to(raw).as_posix()
                    print(("WOULD MOVE " if a.dry_run else "QUARANTINED ") + f"{ds}/{rel} ({type(e).__name__})")
                    if not a.dry_run:
                        dst = root / "corrupt" / ds / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(p), str(dst))
                    moved.append(f"{ds}/{rel}\t{type(e).__name__}")
    if moved and not a.dry_run:
        lst = root / "corrupt" / "quarantined_files.txt"
        old = set(lst.read_text(encoding="utf-8").splitlines()) if lst.exists() else set()
        lst.write_text("\n".join(sorted(old | set(moved))) + "\n", encoding="utf-8")
    print("TOTAL_BAD", len(moved))


if __name__ == "__main__":
    main()
