"""Approximate re-implementation of the segmentation pre-processing of Kumar et al. (2026, Alg. 1):
majority vote of four binary masks (global threshold, adaptive threshold, Otsu, 2-means clustering),
morphological clean-up, then watershed boundaries subtracted; background set to black.
Pure numpy/scipy (no OpenCV). Per-image, parameter-free -> no cross-split leakage."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return int(np.nanargmax(sb))


def _kmeans2(x, iters=10):
    c = np.percentile(x, [25, 75], axis=0)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for k in range(2):
            if (a == k).any():
                c[k] = x[a == k].mean(0)
    return a


def _orient(m):
    """Choose polarity so that the foreground touches the image border less."""
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    return ~m if border.mean() > 0.5 else m


def segment_mask(rgb: np.ndarray) -> np.ndarray:
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    m1 = _orient(g > g.mean())
    m2 = _orient(g > ndi.uniform_filter(g.astype(float), 31) - 2)
    m3 = _orient(g > _otsu(g))
    small = rgb[::4, ::4].reshape(-1, 3).astype(float)
    a = _kmeans2(small).reshape(rgb[::4, ::4].shape[:2])
    a = np.kron(a, np.ones((4, 4), int))[: rgb.shape[0], : rgb.shape[1]].astype(bool)
    m4 = _orient(a)
    vote = (m1.astype(int) + m2 + m3 + m4) >= 2
    vote = ndi.binary_opening(vote, iterations=2)
    vote = ndi.binary_closing(vote, iterations=3)
    vote = ndi.binary_fill_holes(vote)
    # watershed refinement on distance transform; boundaries between basins removed
    dist = ndi.distance_transform_edt(vote)
    if dist.max() > 0:
        peaks = (dist == ndi.maximum_filter(dist, size=25)) & (dist > 0.3 * dist.max())
        markers, n = ndi.label(peaks)
        if n > 1:
            inv = (255 * (1 - dist / dist.max())).astype(np.uint8)
            markers = markers.astype(np.int16)
            markers[~vote] = -1
            ws = ndi.watershed_ift(inv, markers)
            edges = (ndi.maximum_filter(ws, 3) != ndi.minimum_filter(ws, 3)) & vote
            vote = vote & ~edges
    if vote.mean() < 0.05:  # segmentation failed -> keep full image
        vote[:] = True
    return vote


def segment_and_mask(rgb: np.ndarray) -> np.ndarray:
    m = segment_mask(rgb)
    out = rgb.copy()
    out[~m] = 0
    return out
