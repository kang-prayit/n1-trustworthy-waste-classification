"""Leakage-aware (group-aware, stratified) CV splits and the naive reference protocol."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def _sgkf(y, g, n_splits, seed):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def grouped_cv_splits(y: np.ndarray, groups: np.ndarray, n_folds: int = 5, cal_frac: float = 0.2,
                      seed: int = 2026, conformal_split: str = "half") -> list[dict]:
    """Per fold k: test = outer fold k (~1/n_folds). The remaining data is split group-aware into
    train (1-cal_frac) and cal (cal_frac). cal is split group-aware into calfit (temperature/vector
    scaling) and calconf (conformal quantiles) when conformal_split == 'half'; otherwise both = cal.
    No group is ever shared between train/cal/test."""
    idx = np.arange(len(y))
    folds = []
    outer = _sgkf(y, groups, n_folds, seed)
    n_inner = max(2, int(round(1 / cal_frac)))
    for k, (trval, test) in enumerate(outer.split(idx, y, groups)):
        inner = _sgkf(y[trval], groups[trval], n_inner, seed + 100 + k)
        tr_i, cal_i = next(inner.split(trval, y[trval], groups[trval]))
        train, cal = trval[tr_i], trval[cal_i]
        if conformal_split == "half":
            c = _sgkf(y[cal], groups[cal], 2, seed + 200 + k)
            a, b = next(c.split(cal, y[cal], groups[cal]))
            calfit, calconf = cal[a], cal[b]
        else:
            calfit = calconf = cal
        folds.append(dict(train=train, calfit=calfit, calconf=calconf, test=test))
        _check_disjoint(folds[-1], groups)
    return folds


def _check_disjoint(f, groups):
    sets = {k: set(groups[v]) for k, v in f.items()}
    assert not (sets["train"] & sets["test"]), "group leakage train/test"
    assert not (sets["train"] & sets["calfit"]) and not (sets["train"] & sets["calconf"]), "leak train/cal"
    assert not (sets["test"] & sets["calfit"]) and not (sets["test"] & sets["calconf"]), "leak test/cal"


def naive_split(y: np.ndarray, aug_copies: int = 0, seed: int = 2026, ratios=(0.70, 0.15, 0.15)) -> dict:
    """Reference-style protocol: (optional) offline augmentation of the WHOLE pool first
    (each image -> 1 original + aug_copies augmented copies), then a random stratified
    70:15:15 split at the item level, ignoring duplicates/groups. The 15% 'validation' part
    is used as the calibration split (calfit/calconf halves). Returns item arrays (img_idx, aug_id)."""
    n = len(y)
    items = np.array([(i, a) for a in range(aug_copies + 1) for i in range(n)], dtype=np.int64)
    yi = y[items[:, 0]]
    tr, rest = train_test_split(np.arange(len(items)), test_size=ratios[1] + ratios[2],
                                stratify=yi, random_state=seed)
    cal, te = train_test_split(rest, test_size=ratios[2] / (ratios[1] + ratios[2]),
                               stratify=yi[rest], random_state=seed + 1)
    cf, cc = train_test_split(cal, test_size=0.5, stratify=yi[cal], random_state=seed + 2)
    return dict(train=items[tr], calfit=items[cf], calconf=items[cc], test=items[te])


def leakage_report(split: dict, groups: np.ndarray) -> dict:
    """How many test items share a duplicate group (or source image) with a training item."""
    def gi(arr):
        arr = np.asarray(arr)
        img = arr[:, 0] if arr.ndim == 2 else arr
        return img, groups[img]
    tr_img, tr_g = gi(split["train"])
    te_img, te_g = gi(split["test"])
    same_src = np.isin(te_img, tr_img).mean()
    same_grp = np.isin(te_g, tr_g).mean()
    return dict(test_items=int(len(te_img)), pct_test_source_image_in_train=round(100 * same_src, 2),
                pct_test_dupgroup_in_train=round(100 * same_grp, 2))
