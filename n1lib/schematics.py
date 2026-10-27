"""Schematic (illustrative) figures drawn on a millimetre canvas: Fig. 1 (sorting-line scene) and
Supplementary Fig. S7 (naive versus leakage-aware split of one real duplicate group).

Split naming follows the manuscript: *calibration-fit* split (fits temperature / vector scaling; `calfit` in
code, role `cal_ts` in the exported split CSVs) and *conformal-calibration* split (fits conformal thresholds;
`calconf` in code, role `cal_conf`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.patches import (Rectangle, FancyBboxPatch, Circle, Ellipse, Polygon, Wedge,
                                FancyArrowPatch, Arc)
from PIL import Image, ImageOps, ImageEnhance

# Okabe-Ito
BLUE, VERM, GREEN, ORANGE, SKY, PURPLE, YELLOW, BLACK = ("#0072B2", "#D55E00", "#009E73", "#E69F00",
                                                         "#56B4E9", "#CC79A7", "#F0E442", "#000000")
MM = 1 / 25.4
W = 183.0
GREY, LGREY = "#7F7F7F", "#E6E6E6"
ROLE = {"train": BLUE, "cal_ts": ORANGE, "cal_conf": GREEN, "test": VERM}
# readable split names used in the manuscript; code keys: calfit/calconf (npz), cal_ts/cal_conf (split CSV roles)
ROLE_LABEL = {"train": "train", "cal_ts": "calibration-fit", "cal_conf": "conformal-calibration", "test": "test"}


def canvas(h):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    ax.set_autoscale_on(False)
    return fig, ax


def load(rel, ds="realwaste", px=360, crop=1.0, raw_root=None):
    im = Image.open(Path(raw_root) / ds / rel).convert("RGB")
    s = min(im.size) * crop
    l, t = (im.size[0] - s) / 2, (im.size[1] - s) / 2
    return im.crop((int(l), int(t), int(l + s), int(t + s))).resize((px, px), Image.LANCZOS)


def thumb(ax, img, x, y, s, frame=None, lw=1.2, z=3, ls="-"):
    if isinstance(img, (str, Path)):
        img = load(img)
    ax.imshow(np.asarray(img), extent=[x, x + s, y, y + s], zorder=z, interpolation="lanczos")
    if frame:
        ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=frame, lw=lw, zorder=z + 1, ls=ls))


def placeholder(ax, x, y, s, text="image\nplaceholder"):
    ax.add_patch(Rectangle((x, y), s, s, fc=LGREY, ec=GREY, lw=1, ls="--", zorder=3))
    ax.text(x + s / 2, y + s / 2, text, ha="center", va="center", fontsize=5, color=GREY, zorder=4)


def fake_aug(img, seed):
    """Illustrative rendering of an offline augmentation (rotation <= 25 deg, flip, crop 0.75-1, colour jitter).
    NOT the exact tensor produced in the experiment."""
    r = np.random.default_rng(seed)
    im = img.rotate(r.uniform(-25, 25), resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    if r.random() < 0.5:
        im = ImageOps.mirror(im)
    s = r.uniform(0.75, 1.0) ** 0.5
    w = im.size[0]; c = int(w * s); o = r.integers(0, w - c + 1, 2)
    im = im.crop((o[0], o[1], o[0] + c, o[1] + c)).resize((w, w), Image.LANCZOS)
    for E in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        im = E(im).enhance(r.uniform(0.75, 1.25))
    return im


def rbox(ax, x, y, w, h, fc="white", ec=BLACK, lw=1.0, r=1.5, z=1, ls="-", alpha=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, ls=ls, alpha=alpha)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, c=BLACK, lw=1.0, ms=6, z=5, ls="-", cs="arc3,rad=0", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, lw=lw, color=c,
                                 zorder=z, ls=ls, connectionstyle=cs, shrinkA=0, shrinkB=0))


def label(ax, x, y, s):
    ax.text(x, y, s, fontsize=8, fontweight="bold", ha="left", va="top", zorder=20)


def head(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 7); kw.setdefault("fontweight", "bold")
    ax.text(x, y, s, ha=kw.pop("ha", "left"), va=kw.pop("va", "top"), zorder=20, **kw)


def txt(ax, x, y, s, size=6, **kw):
    ax.text(x, y, s, fontsize=size, ha=kw.pop("ha", "left"), va=kw.pop("va", "top"), zorder=kw.pop("zorder", 20), **kw)


def worker(ax, x, y, h=14, vest=ORANGE, helmet=YELLOW, z=6):
    """Simple operator icon with helmet and high-visibility vest (K3/PPE), feet at (x, y)."""
    u = h / 14
    ax.add_patch(FancyBboxPatch((x - 2.6 * u, y), 5.2 * u, 7.8 * u, boxstyle=f"round,pad=0,rounding_size={1.6*u}",
                                fc=vest, ec=BLACK, lw=1, zorder=z))
    for yy in (y + 2.6 * u, y + 4.6 * u):
        ax.add_patch(Rectangle((x - 2.6 * u, yy), 5.2 * u, 0.55 * u, fc="#FFFFFF", ec="none", zorder=z + 1))
    ax.add_patch(Circle((x, y + 9.9 * u), 1.9 * u, fc="#F2D0B0", ec=BLACK, lw=1, zorder=z))
    ax.add_patch(Wedge((x, y + 10.5 * u), 2.3 * u, 0, 180, fc=helmet, ec=BLACK, lw=1, zorder=z + 1))
    ax.add_patch(Rectangle((x - 2.9 * u, y + 10.3 * u), 5.8 * u, 0.5 * u, fc=helmet, ec=BLACK, lw=0.8, zorder=z + 1))


def bin_icon(ax, x, y, w, h, c, lab=None, z=4, size=5):
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w * 0.88, y), (x + w * 0.12, y)], closed=True,
                         fc=c, ec=BLACK, lw=1, zorder=z, alpha=0.9))
    ax.add_patch(Rectangle((x - 0.6, y + h), w + 1.2, 1.0, fc=c, ec=BLACK, lw=1, zorder=z + 1))
    if lab:
        txt(ax, x + w / 2, y + h / 2, lab, size=size, ha="center", va="center", color="white", fontweight="bold")


def check(ax, x, y, s=3, c=GREEN, lw=1.6, z=8):
    ax.plot([x - s * 0.5, x - s * 0.1, x + s * 0.6], [y, y - s * 0.45, y + s * 0.55], c=c, lw=lw,
            solid_capstyle="round", zorder=z)


def draw_fig1_scene(ax, H=104.0):
    """Fig. 1: (a) sorting-line deployment scene, (b) offline leakage-aware development strip. Pure vector."""
    def bottle(x, y, h, c):
        u = h / 10
        pts = [(x - 1.6*u, y), (x + 1.6*u, y), (x + 1.6*u, y + 5.6*u), (x + 0.6*u, y + 7.4*u), (x + 0.6*u, y + 9.4*u),
               (x - 0.6*u, y + 9.4*u), (x - 0.6*u, y + 7.4*u), (x - 1.6*u, y + 5.6*u)]
        ax.add_patch(Polygon(pts, closed=True, fc=c, ec=BLACK, lw=0.9, zorder=6))
        ax.add_patch(Rectangle((x - 0.7*u, y + 9.4*u), 1.4*u, 0.8*u, fc=GREY, ec=BLACK, lw=0.7, zorder=6))
        ax.add_patch(Rectangle((x - 1.6*u, y + 2*u), 3.2*u, 2*u, fc="white", ec="none", alpha=0.7, zorder=7))


    def can(x, y, h, c="#B8B8B8"):
        u = h / 10
        ax.add_patch(Rectangle((x - 2.3*u, y + 0.6*u), 4.6*u, 8.6*u, fc=c, ec=BLACK, lw=0.9, zorder=6))
        for yy in (y + 0.6*u, y + 9.2*u):
            ax.add_patch(Ellipse((x, yy), 4.6*u, 1.2*u, fc="#D9D9D9", ec=BLACK, lw=0.9, zorder=7))
        ax.add_patch(Rectangle((x - 2.3*u, y + 3.5*u), 4.6*u, 2.6*u, fc=VERM, ec="none", alpha=0.8, zorder=6.5))


    def box(x, y, h, c="#C8A165"):
        u = h / 10
        ax.add_patch(Rectangle((x - 4*u, y), 8*u, 6.5*u, fc=c, ec=BLACK, lw=0.9, zorder=6))
        ax.add_patch(Polygon([(x - 4*u, y + 6.5*u), (x - 1.2*u, y + 9*u), (x - 0.4*u, y + 6.5*u)], fc=c, ec=BLACK, lw=0.9, zorder=6))
        ax.add_patch(Polygon([(x + 4*u, y + 6.5*u), (x + 1.4*u, y + 9.2*u), (x + 0.4*u, y + 6.5*u)], fc=c, ec=BLACK, lw=0.9, zorder=6))
        ax.plot([x - 1.5*u, x + 1.5*u], [y + 3.2*u, y + 3.2*u], c="#7A5A2B", lw=1.2, zorder=7)


    def leaf(x, y, h, c="#4E9A3A"):
        u = h / 10
        ax.add_patch(Ellipse((x, y + 3.5*u), 7.5*u, 3.6*u, angle=25, fc=c, ec=BLACK, lw=0.9, zorder=6))
        ax.add_patch(Ellipse((x + 1.8*u, y + 2.2*u), 6*u, 2.8*u, angle=-20, fc="#6FB34F", ec=BLACK, lw=0.9, zorder=6))
        ax.plot([x - 3*u, x + 3.4*u], [y + 1.7*u, y + 5.2*u], c="#2D5E20", lw=0.8, zorder=7)


    def cup(x, y, h, c=SKY):
        u = h / 10
        ax.add_patch(Polygon([(x - 2.0*u, y), (x + 2.0*u, y), (x + 2.8*u, y + 7.5*u), (x - 2.8*u, y + 7.5*u)], fc=c, ec=BLACK, lw=0.9, zorder=6, alpha=0.9))
        ax.add_patch(Rectangle((x - 3.1*u, y + 7.5*u), 6.2*u, 0.9*u, fc="white", ec=BLACK, lw=0.8, zorder=6))
        ax.plot([x + 0.5*u, x + 1.5*u], [y + 8.4*u, y + 12*u], c=VERM, lw=1.2, zorder=5)


    def paper(x, y, h, c="#F4F1E8"):
        u = h / 10
        ax.add_patch(Polygon([(x - 3.2*u, y), (x + 3.4*u, y + 0.5*u), (x + 2.8*u, y + 5.2*u), (x - 3.6*u, y + 4.4*u)], fc=c, ec=BLACK, lw=0.9, zorder=6))
        for k in range(3):
            ax.plot([x - 2.3*u, x + 2.1*u], [y + (1.4 + k*1.1)*u, y + (1.7 + k*1.1)*u], c=GREY, lw=0.6, zorder=7)


    label(ax, 2, H - 1.5, "a")
    head(ax, 6, H - 1.6, "At the sorting line (edge deployment)")

    txt(ax, 6, 90, "Mixed waste on the belt;\none image per item.", size=5.5, linespacing=1.2)
    # ---- conveyor
    yb = 52
    ax.add_patch(FancyBboxPatch((6, yb - 4), 110, 4, boxstyle="round,pad=0,rounding_size=2", fc="#4D4D4D", ec=BLACK, lw=1, zorder=3))
    for xr in np.arange(9, 115, 7.5):
        ax.add_patch(Circle((xr, yb - 2), 1.3, fc="#9E9E9E", ec=BLACK, lw=0.6, zorder=4))
    for xl in (12, 108):
        ax.plot([xl, xl], [yb - 4, yb - 14], c="#4D4D4D", lw=2, zorder=2)
    ax.plot([2, 120], [yb - 14, yb - 14], c="#BDBDBD", lw=1, zorder=1)
    arrow(ax, 20, yb - 7.5, 40, yb - 7.5, c=GREY, lw=1, ms=6)
    txt(ax, 42, yb - 7.5, "belt direction", size=5, va="center", color=GREY)
    # items
    bottle(14, yb, 9, "#2E8B57"); box(27, yb, 8.5); can(41, yb, 8); leaf(53, yb, 8)
    cup(92, yb, 7.5); bottle(102, yb, 9, "#8B5A2B")

    # ---- camera gantry and field of view
    gx0, gx1, gy = 58, 84, 88
    for gx in (gx0, gx1):
        ax.add_patch(Rectangle((gx - 0.8, yb - 14), 1.6, gy - yb + 14, fc="#9E9E9E", ec=BLACK, lw=0.7, zorder=2))
    ax.add_patch(Rectangle((gx0 - 0.8, gy), gx1 - gx0 + 1.6, 1.8, fc="#9E9E9E", ec=BLACK, lw=0.7, zorder=2))
    cx = (gx0 + gx1) / 2
    ax.add_patch(Polygon([(cx - 2.2, gy - 6), (cx + 2.2, gy - 6), (cx + 11, yb + 0.3), (cx - 11, yb + 0.3)],
                         fc=SKY, ec="none", alpha=0.22, zorder=1))
    ax.add_patch(FancyBboxPatch((cx - 4, gy - 5.8), 8, 5, boxstyle="round,pad=0,rounding_size=0.8", fc="#333333", ec=BLACK, lw=0.8, zorder=6))
    ax.add_patch(Circle((cx, gy - 6.2), 1.6, fc=SKY, ec=BLACK, lw=0.8, zorder=7))
    txt(ax, cx - 5.2, gy - 3.2, "RGB\ncamera", size=5, ha="right", va="center", linespacing=1.05)
    # item under the camera is highlighted
    ax.add_patch(Rectangle((cx - 7.5, yb - 0.2), 15, 13.6, fill=False, ec=SKY, lw=1.0, ls="--", zorder=8))
    can(cx, yb, 8, "#B0B0B0")

    # ---- edge device callout
    ex0, ey0, ew, eh = 88, 64, 34, 34
    rbox(ax, ex0, ey0, ew, eh, fc="#F5F9FC", ec=BLUE, lw=1.2, r=2, z=3)
    for k in range(5):
        ax.plot([ex0 + 5 + k*6, ex0 + 5 + k*6], [ey0 - 1.2, ey0], c=BLUE, lw=1.2, zorder=3)
    txt(ax, ex0 + 2, ey0 + eh - 1.5, "Edge device", size=6, fontweight="bold", color=BLUE)
    ax.plot([cx + 4, ex0], [gy - 3.3, gy - 3.3], c=BLACK, lw=0.8, zorder=2)
    # mini steps inside
    steps = [("CNN", "logits z"), ("softmax(z/T)", "calibrated p"), ("p ≥ 1 − q̂", "set C(x)")]
    for k, (a, b) in enumerate(steps):
        xx = ex0 + 2 + k * 10.7
        rbox(ax, xx, ey0 + 22.5, 9.4, 6.2, fc="white", ec=GREY, lw=0.8, r=1, z=4)
        txt(ax, xx + 4.7, ey0 + 26.6, a, size=5.5 if k == 0 else 4.4, ha="center", va="center", fontweight="bold")
        txt(ax, xx + 4.7, ey0 + 24.0, b, size=4.4, ha="center", va="center", color="#444444")
        if k:
            arrow(ax, xx - 1.2, ey0 + 25.6, xx - 0.1, ey0 + 25.6, lw=0.7, ms=3.5)
    # illustrative calibrated probabilities with threshold
    cls = ["Metal", "Plastic", "Glass", "Paper", "Cardb."]
    pv = [0.52, 0.31, 0.09, 0.05, 0.03]
    bx0, by0, bh = ex0 + 3, ey0 + 4.5, 13
    for k, (c_, v) in enumerate(zip(cls, pv)):
        xk = bx0 + k * 6.0
        inset = v >= 0.25
        ax.add_patch(Rectangle((xk, by0), 4.2, v * bh * 1.6, fc=BLUE if inset else "#BFD6EA", ec="none", zorder=5))
        txt(ax, xk + 2.1, by0 - 0.6, c_, size=4.3, ha="center", va="top")
    ax.plot([bx0 - 1, bx0 + 29], [by0 + 0.25 * bh * 1.6] * 2, c=VERM, lw=0.9, ls="--", zorder=6)
    txt(ax, bx0 + 29.5, by0 + 0.25 * bh * 1.6 + 0.4, "1 − q̂", size=4.5, ha="right", va="bottom", color=VERM)
    txt(ax, ex0 + ew / 2, ey0 + 21.3, "illustrative probabilities", size=4.4, ha="center", color=GREY, style="italic")

    # ---- diverter at belt end
    arrow(ax, 117, yb - 1.5, 128, 66, c=GREEN, lw=1.4, ms=7, cs="arc3,rad=-0.25")
    arrow(ax, 117, yb - 2.5, 128, 40, c=ORANGE, lw=1.4, ms=7, cs="arc3,rad=0.25")
    txt(ax, 118.5, 59.5, "|C(x)| = 1", size=5, ha="right", color=GREEN, fontweight="bold")
    txt(ax, 118.5, 44.5, "|C(x)| ≠ 1", size=5, ha="right", color=ORANGE, fontweight="bold")
    # bins
    names = [("Glass", "#2E8B57"), ("Metal", "#7F7F7F"), ("Plastic", SKY), ("Paper", "#C8A165"), ("Organic", "#4E9A3A")]
    for k, (n, c) in enumerate(names):
        bin_icon(ax, 129.5 + k * 10.3, 64, 8.4, 10, c, lab=None)
        txt(ax, 133.7 + k * 10.3, 63.0, n, size=5, ha="center")
    txt(ax, 129.5, 82.5, "Accept: automatic sorting", size=6, fontweight="bold", color=GREEN)
    txt(ax, 129.5, 79.2, "one label in the conformal set", size=5.2)
    # review station
    rbox(ax, 129, 31, 51, 19.5, fc="#FFF6E5", ec=ORANGE, lw=1.0, r=2, z=1)
    txt(ax, 131, 48.5, "Defer: operator review station", size=6, fontweight="bold", color=ORANGE)
    ax.add_patch(Rectangle((148, 37.3), 22, 1.5, fc="#8C6D46", ec=BLACK, lw=0.8, zorder=4))
    for xl in (150, 168):
        ax.plot([xl, xl], [32.3, 37.3], c=BLACK, lw=1.2, zorder=3)
    can(153, 38.8, 5.5); cup(158, 38.8, 5.0); paper(165, 38.8, 5.0)
    worker(ax, 140, 31.8, h=11.5)
    txt(ax, 131, 45.2, "ambiguous or empty set", size=5.2)
    txt(ax, 129.5, 30.3, "Human-in-the-loop: only deferred items reach the operator,\nat a guarded table away from the moving belt (K3 safety).",
        size=5.0, linespacing=1.15)

    # ---- b: offline development strip
    ax.plot([2, 181], [27.6 - 3.0, 27.6 - 3.0], c=LGREY, lw=0.8)
    label(ax, 2, 23.5, "b")
    head(ax, 6, 23.4, "Before deployment: leakage-aware development")
    cards = [("Duplicate audit", "MD5 · pHash/dHash · DINOv2\n→ duplicate groups"),
             ("Group-stratified split", "train 64 · calibration-fit 8 ·\nconformal-calibration 8 · test 20 %;\nno group crosses roles"),
             ("Train and calibrate", "six architectures, 5 folds × 3 seeds;\nT fitted on the calibration-fit\nsplit only"),
             ("Conformal threshold", "q̂ from scores on the\nconformal-calibration split\n(α = 0.10)")]
    cw = 34.5
    for k, (a, b) in enumerate(cards):
        x = 6 + k * (cw + 4.5)
        rbox(ax, x, 3.6, cw, 16.0, fc="white", ec=GREY, lw=0.9, r=1.5, z=2)
        txt(ax, x + 2, 18.4, f"{k+1}  {a}", size=5.8, fontweight="bold")
        txt(ax, x + 2, 15.0, b, size=5.0, linespacing=1.2)
        if k:
            arrow(ax, x - 4.0, 12.1, x - 0.4, 12.1, lw=1, ms=5)
    # segment bar on card 2
    xb = 6 + (cw + 4.5) + 2
    for frac, role in ((0.64, "train"), (0.08, "cal_ts"), (0.08, "cal_conf"), (0.20, "test")):
        ax.add_patch(Rectangle((xb, 4.6), frac * 30, 1.4, fc=ROLE[role], ec="white", lw=0.4, zorder=4))
        xb += frac * 30
    arrow(ax, 6 + 4 * (cw + 4.5) - 4.0, 12.1, 178, 12.1, lw=1, ms=5)
    txt(ax, 160.6, 13.9, "deploy model, T, q̂\nto the edge device (a)", size=5, ha="left", va="bottom", linespacing=1.1)
    txt(ax, 181, 0.2, "Illustrative vector schematic; probabilities are illustrative, not model output.", size=4.6, ha="right", va="bottom", color=GREY)


S7_GROUP = 4331  # RealWaste duplicate group shown in Supplementary Fig. S7 (16 Vegetation images, test in fold 0)
S7_ORDER = [76, 74, 133, 138, 136, 125, 126, 129, 140, 139, 130, 134, 137, 210, 225, 75]
S7_LEAKS = [(74, 76), (133, 138), (136, 138), (139, 140)]  # real DINOv2 links crossing naive train/test
S7_AUG_SOURCE = "Cardboard/Cardboard_101.jpg"


def draw_figS7(ax, prep_dir, raw_root, H=88.0):
    """Supplementary Fig. S7 (illustrative): one real RealWaste duplicate group under the naive item-level split
    versus the leakage-aware grouped split (fold 0), and augment-before-split versus augment-inside-training.
    Roles, links and cosine values are read from the preprocessing export; augmented copies are illustrative
    renderings of the stated transforms, not the stored tensors."""
    import pandas as pd
    prep = Path(prep_dir) / "realwaste"
    F = pd.read_csv(prep / "realwaste_folds_grouped.csv")
    N0 = pd.read_csv(prep / "realwaste_split_naive_aug0.csv")
    N3 = pd.read_csv(prep / "realwaste_split_naive_aug3.csv")
    L = pd.read_csv(prep / "realwaste_duplicate_links.csv")
    naive = dict(zip(N0.relpath, N0.role))
    ld = lambda p, px=220: load(p, px=px, raw_root=raw_root)  # noqa: E731

    XL, XR, CW = 2.0, 94.0, 87.0
    for x, t, c in ((XL, "Naive: split images at random", VERM), (XR, "Leakage-aware: split duplicate groups", BLUE)):
        ax.add_patch(Rectangle((x, H - 7.2), CW, 5.6, fc=c, ec="none", alpha=0.14, zorder=0))
        head(ax, x + 2, H - 2.6, t, color=c)
    ax.plot([91, 91], [9, H - 2], c=LGREY, lw=1)
    # legend + credit
    lx = 2
    for role, w in (("train", 11), ("cal_ts", 22), ("cal_conf", 30), ("test", 10)):
        ax.add_patch(Rectangle((lx, 2.2), 3, 3, fill=False, ec=ROLE[role], lw=1.2))
        txt(ax, lx + 3.8, 3.7, ROLE_LABEL[role] + (" split" if role.startswith("cal") else ""), size=5, va="center")
        lx += w
    ax.plot([lx + 1, lx + 6], [3.7, 3.7], c=PURPLE, lw=1.4)
    txt(ax, lx + 7, 3.7, "duplicate link across train/test (leak)", size=5, va="center")
    txt(ax, 181, 3.7, "Illustrative example. RealWaste images: Single et al., UCI\n(CC BY 4.0; licence pending confirmation).",
        size=4.5, ha="right", va="center", color="#444444", linespacing=1.1)

    # ---- row 1: one real duplicate group
    gid = S7_GROUP
    mem = [f"Vegetation/Vegetation_{i}.jpg" for i in S7_ORDER]
    assert sorted(mem) == sorted(F[F.group_id == gid].relpath.tolist()), "group membership changed"
    imgs = {p: ld(p) for p in mem}
    label(ax, 2, H - 9.0, "a"); label(ax, XR - 1.5, H - 9.0, "b")
    txt(ax, XL + 4, H - 9.4, f"One duplicate group (RealWaste Vegetation, {len(mem)} images)", size=6, fontweight="bold")
    txt(ax, XR + 2.5, H - 9.4, "Same group, kept together", size=6, fontweight="bold")
    ts, g = 9.0, 1.4
    pos = {}
    for side, x0 in (("naive", XL + 4), ("group", XR + 4)):
        for i, p in enumerate(mem):
            cx = x0 + (i % 8) * (ts + g)
            cy = H - 25.5 - (i // 8) * (ts + 3.0)
            role = naive[p] if side == "naive" else F.loc[F.relpath == p, "role_fold0"].item()
            thumb(ax, imgs[p], cx, cy, ts, frame=ROLE[role], lw=1.4)
            if side == "naive":
                pos[p] = (cx, cy)
        if side == "group":
            rbox(ax, x0 - 1.3, H - 25.5 - ts - 3.0 - 1.3, 8 * (ts + g) - g + 2.6, 2 * ts + 3.0 + 2.6, fc="none",
                 ec=VERM, lw=1.0, ls="--", r=1.5, z=6)
    keep = {frozenset((f"Vegetation/Vegetation_{a}.jpg", f"Vegetation/Vegetation_{b}.jpg")) for a, b in S7_LEAKS}
    leaks = [(r.relpath_i, r.relpath_j, r.cosine) for r in L[L.group_id == gid].itertuples()
             if {naive.get(r.relpath_i), naive.get(r.relpath_j)} == {"test", "train"}
             and frozenset((r.relpath_i, r.relpath_j)) in keep]
    assert len(leaks) == len(S7_LEAKS), leaks
    for a, b, cs in leaks:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        up = ya > H - 30
        yy = ya + (ts if up else 0)
        ax.add_patch(FancyArrowPatch((xa + ts / 2, yy), (xb + ts / 2, yy), arrowstyle="-", lw=1.3, color=PURPLE, zorder=7,
                                     connectionstyle=f"arc3,rad={(-0.6 if up else 0.6) * (1 if xa < xb else -1)}"))
        txt(ax, (xa + xb) / 2 + ts / 2, yy + (3.4 if up else -3.4), f"{cs:.3f}", size=4.4, ha="center",
            va="bottom" if up else "top", color=PURPLE)
    nt = sum(naive[p] == "test" for p in mem); ntr = sum(naive[p] == "train" for p in mem)
    txt(ax, XL + 4, H - 43.6, f"Random split: {ntr} train, {nt} test, the rest calibration. Linked look-alikes sit on\n"
        "both sides (arcs: real DINOv2 links with cosine), so the test set partly re-tests memorised images.",
        size=5.3, linespacing=1.2)
    gr0 = sorted(set(F.loc[F.group_id == gid, "role_fold0"]))
    txt(ax, XR + 2.5, H - 43.6, f"StratifiedGroupKFold assigns the whole group to one role (here: {'/'.join(gr0)} in\n"
        "fold 0). No test image has a near-duplicate in training.", size=5.3, linespacing=1.2)

    # ---- row 2: augmentation before vs after the split
    y2 = 17.0
    label(ax, 2, y2 + 21.5, "c"); label(ax, XR - 1.5, y2 + 21.5, "d")
    txt(ax, XL + 4, y2 + 21.1, "Augment first, then split", size=6, fontweight="bold")
    txt(ax, XR + 2.5, y2 + 21.1, "Split first, augment only inside training", size=6, fontweight="bold")
    srcp = S7_AUG_SOURCE
    rows = N3[N3.relpath == srcp].sort_values("aug_id")
    base = ld(srcp, 300)
    s2 = 12.0
    xs = XL + 4
    for r in rows.itertuples():
        im = base if r.aug_id == 0 else fake_aug(base, 1000 + int(r.aug_id))
        thumb(ax, im, xs, y2, s2, frame=ROLE[r.role], lw=1.5)
        txt(ax, xs + s2 / 2, y2 - 0.8, ("original" if r.aug_id == 0 else f"copy {int(r.aug_id)}") + f"\n{ROLE_LABEL[r.role]}",
            size=5, ha="center", linespacing=1.1, color=ROLE[r.role])
        if r.aug_id >= 1:
            ax.add_patch(FancyArrowPatch((XL + 4 + s2 / 2, y2 + s2), (xs + s2 / 2, y2 + s2), arrowstyle="-", lw=1.2,
                                         color=PURPLE, zorder=7, connectionstyle="arc3,rad=-0.17"))
        xs += s2 + 3.2
    roles = dict(zip(rows.aug_id.astype(int), rows.role))
    ncopy_tr = sum(roles.get(k) == "train" for k in (1, 2, 3))
    txt(ax, xs + 0.3, y2 + s2 + 0.5, f"Each image and its three stored\ncopies are split as independent\nitems: the {roles.get(0)} original of\n"
        f"Cardboard 101 has {ncopy_tr} of 3\ncopies in train.", size=5.0, linespacing=1.2)
    gr = F.loc[F.relpath == srcp, "role_fold0"].item()
    xs = XR + 4
    thumb(ax, base, xs, y2, s2, frame=ROLE[gr], lw=1.5)
    txt(ax, xs + s2 / 2, y2 - 0.8, f"original\n{ROLE_LABEL[gr]} (fold 0)", size=5, ha="center", linespacing=1.1, color=ROLE[gr])
    arrow(ax, xs + s2 + 1.0, y2 + s2 / 2, xs + s2 + 7.5, y2 + s2 / 2, lw=1, ms=5)
    txt(ax, xs + s2 + 4.2, y2 + s2 / 2 + 1.0, "each\nepoch", size=4.6, ha="center", va="bottom", linespacing=1.0)
    xs2 = xs + s2 + 9.0
    for k in range(3):
        thumb(ax, fake_aug(base, 7000 + k), xs2 + k * (s2 * 0.72 + 1.6), y2 + 1.8, s2 * 0.72, frame=ROLE[gr], lw=1.0, ls="--")
    txt(ax, xs2 + 1.5 * (s2 * 0.72 + 1.6) - 0.8, y2 - 0.8, "random views, never stored\nor split", size=5, ha="center",
        linespacing=1.1, color=ROLE[gr])
    txt(ax, XR + 63, y2 + s2 - 0.5, "Calibration and test\nimages are never\naugmented.", size=5.0, linespacing=1.2)
    txt(ax, XL + 4, y2 - 6.4, "Copies are illustrative renderings of the stated transforms (rotation ≤ 25°, flip, crop,\n"
        "colour jitter), not the stored tensors.", size=4.6, color=GREY, linespacing=1.1)
    return dict(group=gid, naive_train=ntr, naive_test=nt, leaks=[(a, b, round(float(c), 4)) for a, b, c in leaks],
                aug_roles=roles, grouped_role=gr)
