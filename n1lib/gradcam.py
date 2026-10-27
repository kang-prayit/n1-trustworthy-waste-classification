"""Grad-CAM (Selvaraju et al. 2017) on rejected / uncertain test samples (fold 0, seed 0)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .calib import apply_temperature, fit_temperature
from .common import get_device, read_json
from .conformal import conformal_sets
from .data import dataset_classes
from .models import GRADCAM_OK, build_model
from .prepare import cache_path


def gradcam(model, x, target):
    """x: (1,3,H,W) float [0,1]. Uses the final feature map of the first (CNN) branch."""
    br = model.branches[0]
    br.keep_fmap = True
    x = x.clone().requires_grad_(True)
    out = model(x)
    fm = br.fmap
    model.zero_grad()
    out[0, target].backward()
    g = fm.grad
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fm).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    br.keep_fmap = False
    return cam.detach().cpu().numpy()


def make_gradcam_figure(cfg, fig_dir, log=print, n_show=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gc = cfg.get("gradcam", {})
    ds = gc.get("dataset", "realwaste")
    models = [m for m in gc.get("models", ["resnet101", "convnext_tiny"]) if m in GRADCAM_OK]
    out_dir = Path(cfg["out_dir"])
    classes = dataset_classes(ds, cfg["data_root"])
    panels = []
    device = get_device()
    for m in models:
        rd = out_dir / "runs" / f"{ds}-grouped-{m}-ce-f0-s0"
        if not (rd / "weights_fp16.pt").exists():
            log(f"gradcam: no weights for {rd.name}, skipped")
            continue
        info = read_json(rd / "done.json")
        z = dict(np.load(rd / "preds.npz"))
        T = fit_temperature(z["logits_calfit"], z["y_calfit"])
        P = apply_temperature(z["logits_test"], T)
        Pc = apply_temperature(z["logits_calconf"], T)
        sets, _ = conformal_sets(Pc, z["y_calconf"], P, "aps", 0.1)
        conf = P.max(1)
        thr = np.quantile(conf, 0.2)  # rejected at 80 % coverage
        cand = np.where((conf <= thr) | (sets.sum(1) > 1))[0]
        cand = cand[np.argsort(conf[cand])][:n_show]
        model = build_model(m, len(classes), info["train_cfg"]["drop_rate"], pretrained=False)
        sd = torch.load(rd / "weights_fp16.pt", map_location="cpu")
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        model.to(device).eval()
        mm = np.load(cache_path(cfg, ds, "none"), mmap_mode="r")
        S, s = cfg["cache_size"], cfg["img_size"]
        o = (S - s) // 2
        for i in cand:
            img = np.array(mm[z["items_test"][i][0]])[o:o + s, o:o + s]
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(device)
            pred = int(P[i].argmax())
            cam = gradcam(model, x, pred)
            setlab = [classes[k] for k in np.where(sets[i])[0]]
            panels.append((m, img, cam, classes[int(z["y_test"][i])], classes[pred], float(conf[i]), setlab))
    if not panels:
        return None
    rows = len(models)
    by = {m: [p for p in panels if p[0] == m] for m in models}
    rows = [m for m in models if by[m]]
    fig, axes = plt.subplots(len(rows), n_show, figsize=(2.2 * n_show, 2.6 * len(rows)), squeeze=False)
    for r, m in enumerate(rows):
        for c in range(n_show):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(by[m]):
                _, img, cam, yt, yp, cf, st = by[m][c]
                ax.imshow(img)
                ax.imshow(cam, cmap="jet", alpha=0.4)
                ax.set_title(f"true: {yt}\npred: {yp} ({cf:.2f})\n|C|={len(st)}", fontsize=6.5)
        axes[r, 0].text(-0.08, 0.5, m, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_gradcam_rejected.png", dpi=300)
    fig.savefig(fig_dir / "fig_gradcam_rejected.pdf")
    plt.close(fig)
    log("gradcam figure written")
    return fig_dir / "fig_gradcam_rejected.png"
