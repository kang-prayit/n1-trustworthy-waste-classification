"""Scientific Reports figure style (single source of truth).

Implements the Nature/Scientific Reports artwork rules used in this project:
* widths 89 mm (single column), 120-136 mm (1.5 column), 183 mm (double column); max height 170 mm
* sans-serif type (Arial/Helvetica; metric-compatible Liberation Sans/Arimo as fallback), 7 pt text,
  5 pt minimum, panel labels 8 pt bold lower case (a, b, c), no titles inside panels
* lines >= 1 pt for data; axes/ticks/error bars 1 pt (Sci Rep: thinnest line >= 1 pt; Nature: 0.25-1 pt)
* Okabe-Ito colour-blind-safe palette (Wong 2011, Nat. Methods 8, 441), white background, no 3D
* vector PDF/EPS with embedded TrueType (Type 42) text + 300/600 dpi TIFF/PNG; RGB
"""
from __future__ import annotations

from pathlib import Path

MM = 1 / 25.4
WIDTH_SINGLE, WIDTH_ONEHALF, WIDTH_DOUBLE, MAX_HEIGHT = 89 * MM, 136 * MM, 183 * MM, 170 * MM

OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
             "grey": "#999999"}
CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
# fixed colour per model so that every figure uses the same mapping
MODEL_COLOR = {"resnet101": "#0072B2", "hybrid_r101_incv3": "#D55E00", "efficientnetv2_s": "#009E73",
               "mobilenetv3_large": "#CC79A7", "convnext_tiny": "#E69F00", "deit_small": "#56B4E9",
               "efficientnet_b0": "#000000", "tiny_cnn": "#0072B2", "tiny_hybrid": "#D55E00"}
MODEL_MARKER = {"resnet101": "o", "hybrid_r101_incv3": "s", "efficientnetv2_s": "^", "mobilenetv3_large": "v",
                "convnext_tiny": "D", "deit_small": "P", "efficientnet_b0": "X", "tiny_cnn": "o", "tiny_hybrid": "s"}
FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "Arimo", "DejaVu Sans"]

RC = {
    "font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 7,
    "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.title_fontsize": 6.5, "legend.frameon": False,
    "legend.handlelength": 1.5, "legend.borderaxespad": 0.3,
    "axes.linewidth": 1.0, "axes.edgecolor": "#000000", "axes.labelcolor": "#000000",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.prop_cycle": "cycler('color', ['0072B2','D55E00','009E73','CC79A7','E69F00','56B4E9','000000','F0E442'])",
    "axes.labelpad": 2.0, "axes.unicode_minus": True,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.color": "#000000", "ytick.color": "#000000",
    "lines.linewidth": 1.25, "lines.markersize": 3.5, "lines.markeredgewidth": 1.0,
    "patch.linewidth": 1.0, "hatch.linewidth": 1.0,
    "errorbar.capsize": 1.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.transparent": False, "figure.dpi": 150, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "image.cmap": "viridis", "mathtext.default": "regular",
}


def _rc_line(k, v):
    if isinstance(v, list):
        v = ", ".join(v)
    elif isinstance(v, bool):
        v = str(v)
    elif isinstance(v, str) and v.startswith("#"):
        v = v[1:]  # '#' starts a comment in .mplstyle files
    return f"{k}: {v}"


def write_mplstyle(path):
    lines = ["# Scientific Reports figure style - generated from n1lib/figstyle.py (do not edit by hand)",
             "# widths: 89 mm single, 136 mm 1.5-col, 183 mm double; max height 170 mm",
             "# Okabe-Ito palette; Arial/Helvetica (Liberation Sans/Arimo metric-compatible fallback)"]
    for k, v in RC.items():
        if k == "axes.prop_cycle":
            v = v.replace("'", "")
            v = "cycler('color', [" + ", ".join(f"'{c}'" for c in v.split("[")[1].split("]")[0].split(",")) + "])"
            lines.append(f"{k}: {v}")
            continue
        lines.append(_rc_line(k, v))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply():
    import matplotlib
    matplotlib.rcParams.update(RC)
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONT_STACK if f in names), "DejaVu Sans")


def panel_label(ax, label, x=None, y=1.0, fig=None, dx_mm=-1.5):
    """Bold lower-case panel label (8 pt) placed at the top-left outside the axes."""
    import matplotlib.transforms as mtrans
    fig = fig or ax.figure
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    xx = bbox.x0 + dx_mm * MM / fig.get_figwidth() if x is None else x
    pos = ax.get_position()
    yy = max(pos.y1, bbox.y1) if y == 1.0 else y
    fig.text(xx, yy, label, fontsize=8, fontweight="bold", ha="left", va="top" if y != 1.0 else "bottom",
             transform=fig.transFigure)


def watermark(fig, text):
    if not text:
        return
    fig.text(0.5, 0.5, text, fontsize=16, color="#D55E00", alpha=0.28, ha="center", va="center",
             rotation=28, fontweight="bold", transform=fig.transFigure, zorder=1000)


def save(fig, out_dir, name, formats=("pdf", "eps", "tiff600", "png300"), log=print):
    """Save a figure at its final size (no bbox cropping => exact column width)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in formats:
        try:
            if f == "pdf":
                fig.savefig(out_dir / f"{name}.pdf")
                written.append(f"{name}.pdf")
            elif f == "eps":
                fig.savefig(out_dir / f"{name}.eps")
                written.append(f"{name}.eps")
            elif f.startswith("tiff"):
                dpi = int(f[4:])
                fig.savefig(out_dir / f"{name}.tif", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
                written.append(f"{name}.tif")
            elif f.startswith("png"):
                dpi = int(f[3:])
                fig.savefig(out_dir / f"{name}.png", dpi=dpi)
                written.append(f"{name}.png")
        except Exception as e:  # noqa: BLE001 - EPS can fail with transparency; keep going
            log(f"  could not write {name}.{f}: {e}")
    return written
