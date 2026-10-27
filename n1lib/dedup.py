"""Exact (MD5) and perceptual (pHash + dHash) duplicate detection and grouping.

Two images are linked if MD5 is identical, or if BOTH Hamming(pHash) <= t_p AND
Hamming(dHash) <= t_d (requiring agreement of two hashes limits false merges between
distinct objects photographed on the same background). Linked images are merged into
connected components (union-find) -> 'groups' used by group-aware splitting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hash_one(path):
    import imagehash
    from .data import load_rgb, file_md5
    try:
        im = load_rgb(path)
        ph = int(str(imagehash.phash(im, hash_size=8)), 16)
        dh = int(str(imagehash.dhash(im, hash_size=8)), 16)
        return file_md5(path), ph, dh, im.size[0], im.size[1]
    except Exception:
        return file_md5(path), 0, 0, -1, -1


def compute_hashes(index: pd.DataFrame, raw_root: Path, workers: int = 4) -> pd.DataFrame:
    from tqdm import tqdm
    paths = [str(Path(raw_root) / p) for p in index["path"]]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(tqdm(ex.map(_hash_one, paths, chunksize=32), total=len(paths), desc="hashing"))
    else:
        res = [_hash_one(p) for p in tqdm(paths, desc="hashing")]
    out = index.copy()
    out["md5"] = [r[0] for r in res]
    out["phash"] = np.array([r[1] for r in res], dtype=np.uint64)
    out["dhash"] = np.array([r[2] for r in res], dtype=np.uint64)
    out["width"] = [r[3] for r in res]
    out["height"] = [r[4] for r in res]
    return out


def hamming_matrix_rows(h: np.ndarray, rows: slice) -> np.ndarray:
    """Hamming distances between h[rows] and all h (uint64 arrays) -> uint8 matrix."""
    x = np.bitwise_xor(h[rows, None], h[None, :])
    b = x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
    return _POP[b].sum(-1, dtype=np.uint8)


def near_duplicate_pairs(ph: np.ndarray, dh: np.ndarray, t_p: int, t_d: int, chunk: int = 512):
    """Return (i, j, dp, dd) arrays for i<j with dp<=max(t_p) and dd<=t_d."""
    n = len(ph)
    I, J, DP, DD = [], [], [], []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dp = hamming_matrix_rows(ph, slice(s, e))
        cand = np.argwhere(dp <= t_p)
        cand = cand[cand[:, 1] > cand[:, 0] + s]
        if len(cand) == 0:
            continue
        i = cand[:, 0] + s
        j = cand[:, 1]
        dd = _POP[np.bitwise_xor(dh[i], dh[j]).view(np.uint8).reshape(-1, 8)].sum(-1)
        keep = dd <= t_d
        I.append(i[keep]); J.append(j[keep]); DP.append(dp[cand[keep, 0], cand[keep, 1]]); DD.append(dd[keep])
    if not I:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    return np.concatenate(I), np.concatenate(J), np.concatenate(DP), np.concatenate(DD)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def compute_embeddings(index: pd.DataFrame, raw_root: Path, model_name: str, batch: int = 64) -> np.ndarray:
    """L2-normalised self-supervised embeddings (default DINOv2 ViT-S/14 at 224 px) used to find
    repeated shots of the same physical object that perceptual hashes miss."""
    import timm
    import torch
    import torchvision.transforms as T
    from tqdm import tqdm
    from .data import load_rgb
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(img_size=224) if "dinov2" in model_name else {}
    m = timm.create_model(model_name, pretrained=True, num_classes=0, **kw).eval().to(dev)
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    paths = [Path(raw_root) / p for p in index["path"]]
    feats = []
    with torch.no_grad():
        for s in tqdm(range(0, len(paths), batch), desc="embeddings"):
            x = torch.stack([tf(load_rgb(p)) for p in paths[s:s + batch]]).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                f = m(x)
            feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(feats).numpy().astype(np.float32)


def embedding_pairs(emb: np.ndarray, labels: np.ndarray, thresh: float, same_class_only: bool, chunk: int = 2048):
    I, J, S = [], [], []
    for s in range(0, len(emb), chunk):
        sim = emb[s:s + chunk] @ emb.T
        i, j = np.where(sim >= thresh)
        i = i + s
        k = j > i
        i, j = i[k], j[k]
        if same_class_only:
            k = labels[i] == labels[j]
            i, j = i[k], j[k]
        I.append(i); J.append(j); S.append(sim[i - s, j])
    return np.concatenate(I), np.concatenate(J), np.concatenate(S)


def group_duplicates(h: pd.DataFrame, t_p: int, t_d: int, extra_pairs=None):
    """Returns (group ids, pairs dataframe, summary dict). extra_pairs: optional (i, j) arrays
    (e.g. embedding near-duplicates) that are also merged."""
    n = len(h)
    uf = UnionFind(n)
    # exact byte duplicates
    for _, idx in h.groupby("md5").indices.items():
        for k in idx[1:]:
            uf.union(idx[0], k)
    ph = h["phash"].to_numpy(np.uint64)
    dh = h["dhash"].to_numpy(np.uint64)
    i, j, dp, dd = near_duplicate_pairs(ph, dh, t_p, t_d)
    for a, b in zip(i, j):
        uf.union(int(a), int(b))
    n_extra = 0
    if extra_pairs is not None and len(extra_pairs[0]):
        n_extra = len(extra_pairs[0])
        for a, b in zip(*extra_pairs):
            uf.union(int(a), int(b))
    groups = np.array([uf.find(k) for k in range(n)])
    # relabel 0..G-1
    _, groups = np.unique(groups, return_inverse=True)
    lab = h["label"].to_numpy()
    pairs = pd.DataFrame(dict(i=i, j=j, phash_dist=dp, dhash_dist=dd,
                              path_i=h["path"].to_numpy()[i] if len(i) else [],
                              path_j=h["path"].to_numpy()[j] if len(j) else [],
                              cross_class=(lab[i] != lab[j]) if len(i) else []))
    sizes = np.bincount(groups)
    in_dup = sizes[groups] > 1
    exact_dup_imgs = int((h.groupby("md5")["md5"].transform("size") > 1).sum())
    mixed = 0
    for g in np.where(sizes > 1)[0]:
        if len(np.unique(lab[groups == g])) > 1:
            mixed += 1
    # sensitivity: number of near-dup pairs at several pHash thresholds (dHash ignored)
    sens = {}
    for t in (0, 2, 4, 6, 8, 10):
        a, _, _, _ = near_duplicate_pairs(ph, dh, t, 64)
        sens[f"pairs_phash_le_{t}"] = int(len(a))
    summary = dict(
        n_images=n, n_groups=int(len(sizes)), n_dup_groups=int((sizes > 1).sum()),
        n_images_in_dup_groups=int(in_dup.sum()),
        pct_images_in_dup_groups=round(100 * in_dup.mean(), 2),
        n_redundant_images=int(n - len(sizes)), largest_group=int(sizes.max()),
        n_exact_md5_dup_images=exact_dup_imgs, n_near_dup_pairs=int(len(i)), n_embedding_pairs=int(n_extra),
        n_cross_class_pairs=int(pairs["cross_class"].sum()) if len(pairs) else 0,
        n_mixed_label_groups=int(mixed), phash_thresh=t_p, dhash_thresh=t_d, **sens)
    return groups, pairs, summary
