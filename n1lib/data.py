"""Dataset registry, raw-image index, uint8 memmap cache and torch Dataset.

Raw layout produced by download_data.py:  <data_root>/raw/<dataset>/<ClassName>/<image files>
Cache: <work_dir>/<dataset>/cache_<preprocess>_<size>.npy  (N x S x S x 3 uint8, np.memmap)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DATASETS = {
    "realwaste": dict(
        classes=["Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash",
                 "Paper", "Plastic", "Textile Trash", "Vegetation"],
        expected={"Cardboard": 461, "Food Organics": 411, "Glass": 420, "Metal": 790,
                  "Miscellaneous Trash": 495, "Paper": 500, "Plastic": 921,
                  "Textile Trash": 318, "Vegetation": 436}),           # 4752
    "trashnet": dict(
        classes=["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        expected={"cardboard": 403, "glass": 501, "metal": 410, "paper": 594,
                  "plastic": 482, "trash": 137}),                       # 2527
    "trashbox": dict(
        classes=["cardboard", "e-waste", "glass", "medical", "metal", "paper", "plastic"],
        # GitHub train set + TrashBox-testandvalid (test+val) pooled, verified via git trees
        expected={"cardboard": 2413, "e-waste": 3008, "glass": 2528, "medical": 1957,
                  "metal": 2586, "paper": 2695, "plastic": 2669}),      # 17856
    "fake": dict(classes=["alpha", "beta", "gamma"], expected=None),
}


def dataset_classes(name: str, data_root: str | Path) -> list[str]:
    if name in DATASETS:
        return DATASETS[name]["classes"]
    root = Path(data_root) / "raw" / name
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def build_index(name: str, data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root) / "raw" / name
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run: python download_data.py --datasets {name}")
    classes = dataset_classes(name, data_root)
    rows = []
    for ci, c in enumerate(classes):
        cdir = root / c
        if not cdir.exists():
            raise FileNotFoundError(f"class folder missing: {cdir}")
        for p in sorted(cdir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith("."):
                rows.append(dict(path=str(p.relative_to(root)).replace("\\", "/"), label=ci, cls=c))
    df = pd.DataFrame(rows)
    return df


def load_rgb(path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("P", "LA") or (im.mode == "RGBA"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def resize_center_square(im: Image.Image, size: int) -> np.ndarray:
    w, h = im.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return np.asarray(im.crop((left, top, left + size, top + size)), dtype=np.uint8)


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_one(args):
    path, size, preprocess = args
    try:
        im = load_rgb(path)
        arr = resize_center_square(im, size)
        if preprocess == "seg":
            from .seg import segment_and_mask
            arr = segment_and_mask(arr)
        return arr, True
    except Exception as e:  # corrupt image -> black tile, flagged
        print(f"[warn] cannot read {path}: {e}")
        return np.zeros((size, size, 3), np.uint8), False


def build_cache(index: pd.DataFrame, raw_root: Path, out_path: Path, size: int,
                preprocess: str = "none", workers: int = 4) -> Path:
    """Write an N x S x S x 3 uint8 .npy (memmap-able). Resumable via a .done marker."""
    out_path = Path(out_path)
    done = out_path.with_suffix(".done")
    if out_path.exists() and done.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(len(index), size, size, 3))
    jobs = [(str(raw_root / p), size, preprocess) for p in index["path"]]
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    ok = np.ones(len(jobs), bool)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for i, (arr, good) in enumerate(tqdm(ex.map(_cache_one, jobs, chunksize=16), total=len(jobs),
                                                 desc=f"cache {out_path.name}")):
                mm[i] = arr
                ok[i] = good
    else:
        for i, j in enumerate(tqdm(jobs, desc=f"cache {out_path.name}")):
            mm[i], ok[i] = _cache_one(j)
    mm.flush()
    del mm
    np.save(out_path.with_suffix(".ok.npy"), ok)
    done.write_text("ok")
    return out_path


# ----------------------------------------------------------------------------------------------
# torch side
# ----------------------------------------------------------------------------------------------
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402


def train_transform(img_size: int):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.02)], p=0.8),
    ])


def eval_transform(img_size: int):
    return T.CenterCrop(img_size)


def offline_aug_transform():
    """Deterministic 'offline' augmentation used ONLY by the naive protocol
    (copies created before the split, as in pipelines that augment the whole pool)."""
    return T.Compose([
        T.RandomRotation(25), T.RandomHorizontalFlip(), T.RandomResizedCrop(256, scale=(0.75, 1.0), antialias=True),
        T.ColorJitter(0.25, 0.25, 0.25, 0.02),
    ])


class CachedImages(Dataset):
    """Reads from the uint8 memmap. Items are (image_index, aug_id); aug_id>0 = offline copy."""

    def __init__(self, cache_path, items, labels, mode: str, img_size: int):
        self.cache_path = str(cache_path)
        self.items = np.asarray(items, dtype=np.int64).reshape(-1, 2)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mode = mode
        self.img_size = img_size
        self._mm = None
        self.tf = train_transform(img_size) if mode == "train" else eval_transform(img_size)
        self.offline = offline_aug_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self._mm is None:  # opened lazily inside each worker (Windows spawn-safe)
            self._mm = np.load(self.cache_path, mmap_mode="r")
        idx, aug = self.items[i]
        x = torch.from_numpy(np.array(self._mm[idx])).permute(2, 0, 1).contiguous()
        if aug > 0:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(idx) * 1000 + int(aug))
                x = self.offline(x)
        x = self.tf(x)
        return x, int(self.labels[i])
