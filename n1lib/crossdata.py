"""Optional (cheap) cross-dataset sanity check: models trained on a source dataset (weights kept
for seed 0) are applied to shared classes of a target dataset. Reports accuracy (full-label argmax
and restricted to shared classes), ECE, and conformal coverage/set size using the SOURCE
calibration split, i.e. conformal guarantees under distribution shift (exchangeability broken)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .calib import apply_temperature, fit_temperature, softmax_np
from .common import get_device, read_json
from .conformal import conformal_sets, set_metrics
from .data import CachedImages, dataset_classes
from .metrics import ece
from .models import build_model
from .prepare import cache_path, load_index
from .train import _loader, predict


def run_crossdata(cfg, log=print):
    cc = cfg.get("crossdata")
    if not cc:
        return
    out_dir = Path(cfg["out_dir"])
    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    device = get_device()
    rows = []
    src = cc["source"]
    src_classes = dataset_classes(src, cfg["data_root"])
    for tgt, mapping in cc["targets"].items():
        if not cache_path(cfg, tgt, "none").exists():
            log(f"crossdata: target {tgt} not prepared, skipped")
            continue
        tidx = load_index(cfg, tgt)
        tcls = dataset_classes(tgt, cfg["data_root"])
        inv = {t: s for s, t in mapping.items()}
        keep = tidx["cls"].isin(inv.keys()).to_numpy()
        items = np.stack([np.where(keep)[0], np.zeros(keep.sum(), int)], 1)
        ysrc = np.array([src_classes.index(inv[c]) for c in tidx["cls"][keep]])
        shared = sorted({src_classes.index(s) for s in mapping})
        ds = CachedImages(cache_path(cfg, tgt, "none"), items, ysrc, "eval", cfg["img_size"])
        for wp in sorted((out_dir / "runs").glob(f"{src}-grouped-*-ce-f*-s*/weights_fp16.pt")):
            rd = wp.parent
            info = read_json(rd / "done.json")
            outp = rd / f"cross_{tgt}.npz"
            if not outp.exists():
                model = build_model(info["model"], len(src_classes), info["train_cfg"]["drop_rate"], pretrained=False)
                sd = torch.load(wp, map_location="cpu")
                model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
                model.to(device)
                ld = _loader(ds, 128, False, int(cfg["train"]["num_workers"]), 0, device)
                p = predict(model, ld, device, device.type == "cuda", device.type == "cuda", 0,
                            cfg["train"].get("max_eval_batches"))
                np.savez_compressed(outp, logits=p["logits"], y=p["y"])
                del model
            z = np.load(outp)
            pz = dict(np.load(rd / "preds.npz"))
            T = fit_temperature(pz["logits_calfit"], pz["y_calfit"])
            y = z["y"]
            Pt = apply_temperature(z["logits"], T)
            Pr = softmax_np(z["logits"])
            restricted = softmax_np(z["logits"][:, shared] / T)
            yr = np.array([shared.index(v) for v in y])
            cal = apply_temperature(pz["logits_calconf"], T)
            row = dict(source=src, target=tgt, run_id=info["run_id"], model=info["model"], fold=info["fold"],
                       n=len(y), acc_full=float((Pt.argmax(1) == y).mean()),
                       acc_shared=float((restricted.argmax(1) == yr).mean()),
                       ece_raw=ece(Pr, y), ece_ts=ece(Pt, y),
                       in_domain_test_acc=info["test_acc_raw"])
            for a in (0.05, 0.1):
                sets, _ = conformal_sets(cal, pz["y_calconf"], Pt, "lac", a)
                sm = set_metrics(sets, y)
                row[f"cov_lac_a{a}"] = sm["coverage"]
                row[f"size_lac_a{a}"] = sm["avg_size"]
                sets, _ = conformal_sets(cal, pz["y_calconf"], Pt, "aps", a)
                sm = set_metrics(sets, y)
                row[f"cov_aps_a{a}"] = sm["coverage"]
                row[f"size_aps_a{a}"] = sm["avg_size"]
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(res / "crossdata.csv", index=False)
        log(f"crossdata: {len(rows)} rows")
