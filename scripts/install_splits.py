#!/usr/bin/env python
"""Install the published duplicate groups and grouped CV splits into the pipeline's work directory,
so that `run_all.py --stages prepare` reuses them (it only builds the image caches) instead of
recomputing hashes / embeddings / splits.

    python scripts/install_splits.py --config configs/full.yaml               # all three datasets
    python scripts/install_splits.py --config configs/full.yaml --check-md5   # also verify image bytes

Copies splits/raw_artifacts/<ds>/{index.csv, groups.npy, splits_*.npz, *_report.json, *pairs.csv,
dedup_summary.json} to <work_dir>/<ds>/ and checks that every file listed in index.csv exists under
<data_root>/raw/<ds>/ (run download_data.py and scripts/quarantine_corrupt.py first). With
--check-md5 the MD5 of every image is compared with index.csv; a mismatch means the downloaded
archive differs from the one used in the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def md5(p, bs=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import pandas as pd
    from n1lib.common import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--splits", default=str(ROOT / "splits" / "raw_artifacts"))
    ap.add_argument("--datasets", nargs="+", default=["realwaste", "trashnet", "trashbox"])
    ap.add_argument("--check-md5", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing files in work_dir")
    ap.add_argument("--no-image-check", action="store_true", help="only copy the split files")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config if not Path(a.config).is_absolute() else a.config)
    bad_total = 0
    for ds in a.datasets:
        src, dst = Path(a.splits) / ds, Path(cfg["work_dir"]) / ds
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(src.iterdir()):
            q = dst / p.name
            if q.exists() and not a.force:
                if q.read_bytes() != p.read_bytes():
                    sys.exit(f"{q} exists and differs from the published file (use --force to overwrite)")
                continue
            shutil.copyfile(p, q)
            n += 1
        idx = pd.read_csv(dst / "index.csv", dtype={"md5": str})
        msg = f"{ds}: {n} file(s) installed in {dst}; index has {len(idx)} images"
        if not a.no_image_check:
            raw = Path(cfg["data_root"]) / "raw" / ds
            missing = [r for r in idx.path if not (raw / r).exists()]
            bad = []
            if a.check_md5 and not missing:
                bad = [r for r, m in zip(idx.path, idx.md5) if md5(raw / r) != m]
            msg += f"; missing under {raw}: {len(missing)}" + (f"; md5 mismatches: {len(bad)}" if a.check_md5 else "")
            for r in (missing + bad)[:10]:
                msg += f"\n   {r}"
            bad_total += len(missing) + len(bad)
        print(msg)
    if bad_total:
        sys.exit(f"{bad_total} problem(s) found - see above")
    print("OK")


if __name__ == "__main__":
    main()
