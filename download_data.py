#!/usr/bin/env python
"""Download, extract and verify RealWaste, TrashNet and TrashBox into data/raw/<dataset>/<class>/.

Usage (Windows PowerShell or any shell):
    python download_data.py                      # all three datasets
    python download_data.py --datasets realwaste trashnet
    python download_data.py --fake               # tiny synthetic dataset for smoke tests

Sources (verified 2026-10-25):
  RealWaste  UCI ML Repository id 908 (CC BY 4.0 on UCI; GitHub README says CC BY-NC-SA 4.0)
             https://archive.ics.uci.edu/static/public/908/realwaste.zip   (688.5 MB, 4752 images, 9 classes)
  TrashNet   https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip (42.8 MB, 2527 images, 6 classes)
             mirror: https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip
  TrashBox   https://github.com/nikhilvenkatkumsetty/TrashBox (train set, 14282 images)
             https://github.com/nikhilvenkatkumsetty/TrashBox-testandvalid (test 1793 + val 1781 images)
             downloaded as GitHub zip archives (4.46 GB + 1.06 GB); pooled to 17856 images, 7 classes.
Resumable: partially downloaded files are continued (HTTP Range) and finished datasets are skipped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from n1lib.data import DATASETS, IMG_EXT  # noqa: E402

SOURCES = {
    "realwaste": [("realwaste.zip", ["https://archive.ics.uci.edu/static/public/908/realwaste.zip"])],
    "trashnet": [("trashnet-dataset-resized.zip", [
        "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
        "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"])],
    "trashbox": [
        ("TrashBox-main.zip", ["https://codeload.github.com/nikhilvenkatkumsetty/TrashBox/zip/refs/heads/main"]),
        ("TrashBox-testandvalid-main.zip", ["https://codeload.github.com/nikhilvenkatkumsetty/TrashBox-testandvalid/zip/refs/heads/main"]),
    ],
}


def download(url: str, dest: Path, retries: int = 5) -> None:
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "n1-downloader/1.0"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range -> restart
                    have = 0
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have else "wb"
                with open(part, mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                                 desc=dest.name) as bar:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
            part.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed for {url}: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"download failed: {url}")


def fetch(ds: str, dl_dir: Path) -> list[Path]:
    paths = []
    for fname, urls in SOURCES[ds]:
        dest = dl_dir / fname
        if dest.exists() and zipfile.is_zipfile(dest):
            print(f"  {fname} already downloaded")
        else:
            for u in urls:
                try:
                    print(f"  downloading {u}")
                    download(u, dest)
                    if zipfile.is_zipfile(dest):
                        break
                    dest.unlink()
                except RuntimeError as e:
                    print(f"  {e}; trying next mirror")
            if not dest.exists():
                raise RuntimeError(f"could not download {fname}. See README 'Manual download'.")
        paths.append(dest)
    return paths


def extract_classes(zpath: Path, out: Path, rule) -> int:
    """rule(member_path_parts) -> (class_name, new_file_name) or None."""
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if Path(parts[-1]).suffix.lower() not in IMG_EXT or parts[-1].startswith("."):
                continue
            r = rule(parts)
            if r is None:
                continue
            cls, name = r
            dst = out / cls / name
            if dst.exists() and dst.stat().st_size == info.file_size:
                n += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            n += 1
    return n


def rule_realwaste(parts):
    classes = DATASETS["realwaste"]["classes"]
    for i, p in enumerate(parts[:-1]):
        if p in classes:
            return p, parts[-1]
    return None


def rule_trashnet(parts):
    classes = DATASETS["trashnet"]["classes"]
    if len(parts) >= 2 and parts[-2] in classes:
        return parts[-2], parts[-1]
    return None


def rule_trashbox(parts):
    """Use TrashBox_train_set/<cls>/* and TrashBox_testandvalid_set/{test,val}/<cls>/*.
    The *_subfolders copies are duplicates organised by sub-class and are ignored."""
    classes = DATASETS["trashbox"]["classes"]
    if "TrashBox_train_set" in parts:
        i = parts.index("TrashBox_train_set")
        if len(parts) > i + 2 and parts[i + 1] in classes:
            return parts[i + 1], "train__" + parts[-1]
    if "TrashBox_testandvalid_set" in parts:
        i = parts.index("TrashBox_testandvalid_set")
        if len(parts) > i + 3 and parts[i + 1] in ("test", "val") and parts[i + 2] in classes:
            return parts[i + 2], f"{parts[i + 1]}__" + parts[-1]
    return None


RULES = {"realwaste": rule_realwaste, "trashnet": rule_trashnet, "trashbox": rule_trashbox}


def verify(ds: str, out: Path) -> dict:
    exp = DATASETS[ds]["expected"]
    counts = {c: sum(1 for p in (out / c).iterdir() if p.suffix.lower() in IMG_EXT) if (out / c).exists() else 0
              for c in DATASETS[ds]["classes"]}
    ok = all(counts[c] == n for c, n in exp.items())
    rep = dict(dataset=ds, n_classes=len(counts), total=sum(counts.values()), counts=counts,
               expected=exp, expected_total=sum(exp.values()), matches_expected=ok)
    (out / "VERIFIED.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"  {ds}: {rep['total']} images in {rep['n_classes']} classes; matches expected: {ok}")
    for c in counts:
        flag = "" if counts[c] == exp[c] else f"   <-- expected {exp[c]}"
        print(f"     {c:22s} {counts[c]:6d}{flag}")
    return rep


def make_fake(data_root: Path, n_per_class=40, size=96, seed=0):
    """Synthetic 3-class dataset with planted exact / near / cross-class duplicates."""
    import numpy as np
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    out = data_root / "raw" / "fake"
    if out.exists():
        shutil.rmtree(out)
    classes = DATASETS["fake"]["classes"]
    for ci, c in enumerate(classes):
        (out / c).mkdir(parents=True)
        for i in range(n_per_class):
            bg = tuple(int(v) for v in rng.integers(0, 255, 3))
            im = Image.new("RGB", (size, size), bg)
            d = ImageDraw.Draw(im)
            x0, y0 = rng.integers(5, size // 2, 2)
            x1, y1 = x0 + rng.integers(20, size // 2), y0 + rng.integers(20, size // 2)
            col = [(220, 30, 30), (30, 200, 30), (30, 30, 220)][ci]
            [d.rectangle, d.ellipse, d.polygon][ci]([x0, y0, x1, y1] if ci < 2 else [(x0, y1), (x1, y1), ((x0 + x1) // 2, y0)], fill=col)
            arr = np.asarray(im).astype(int) + rng.integers(-25, 25, (size, size, 3))
            Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(out / c / f"{c}_{i:03d}.jpg", quality=92)
        for k in range(3):  # exact duplicates
            shutil.copy(out / c / f"{c}_{k:03d}.jpg", out / c / f"{c}_dupexact_{k}.jpg")
        for k in range(3, 6):  # near duplicates (brightness change + re-encode)
            im = Image.open(out / c / f"{c}_{k:03d}.jpg")
            arr = np.clip(np.asarray(im).astype(int) + 6, 0, 255).astype("uint8")
            Image.fromarray(arr).save(out / c / f"{c}_dupnear_{k}.jpg", quality=80)
    shutil.copy(out / classes[0] / f"{classes[0]}_010.jpg", out / classes[1] / "crossclass_dup.jpg")
    print(f"fake dataset written to {out}")


def main():
    for _s in (sys.stdout, sys.stderr):  # Windows consoles/redirects may not be UTF-8
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["realwaste", "trashnet", "trashbox"])
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--delete-zips", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-extract even if verified")
    a = ap.parse_args()
    data_root = Path(a.data_root)
    if a.fake:
        make_fake(data_root)
        return
    dl = data_root / "downloads"
    summary = {}
    for ds in a.datasets:
        print(f"== {ds}")
        out = data_root / "raw" / ds
        vf = out / "VERIFIED.json"
        if vf.exists() and not a.force and json.loads(vf.read_text())["matches_expected"]:
            print("  already extracted and verified")
            summary[ds] = json.loads(vf.read_text())
            continue
        zips = fetch(ds, dl)
        for z in zips:
            n = extract_classes(z, out, RULES[ds])
            print(f"  extracted {n} images from {z.name}")
        summary[ds] = verify(ds, out)
        if a.delete_zips:
            for z in zips:
                z.unlink()
    (data_root / "DATASETS_VERIFIED.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bad = [d for d, r in summary.items() if not r["matches_expected"]]
    if bad:
        print(f"WARNING: counts differ from expected for {bad} (see data/raw/<ds>/VERIFIED.json). "
              "Experiments still run on what was found; report the actual counts.")


if __name__ == "__main__":
    main()
