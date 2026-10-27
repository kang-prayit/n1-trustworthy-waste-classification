"""Runtime estimation. Prior throughputs are conservative guesses for an RTX 4070 (12 GB) with
AMP fp16 + channels_last, batch 64, 224 px (hybrid includes the 299 px InceptionV3 branch),
capped by a Windows DataLoader limit. Measure the real values with
    python run_all.py --config configs/full.yaml --benchmark
which writes logs/throughput.json and is used automatically afterwards."""
from __future__ import annotations

import json
import time
from pathlib import Path

PRIOR_TRAIN_IPS = {  # images / second, training (fwd+bwd)
    "resnet101": 330, "hybrid_r101_incv3": 200, "efficientnetv2_s": 330, "efficientnet_b0": 850,
    "mobilenetv3_large": 1200, "convnext_tiny": 400, "deit_small": 700, "tiny_cnn": 2000, "tiny_hybrid": 1500,
}
LOADER_CAP_IPS = 900       # uint8 memmap + torchvision v2 aug, 6 workers on Windows
INFER_FACTOR = 3.0          # inference ~3x faster than training
RUN_OVERHEAD_S = 45         # model build, worker spawn, checkpoint I/O, eval loaders
N_IMAGES = {"realwaste": 4752, "trashnet": 2527, "trashbox": 17856, "fake": 132}


def load_throughput(log_dir):
    p = Path(log_dir) / "throughput.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def estimate_run_seconds(run, cfg, measured=None):
    measured = measured or {}
    n = N_IMAGES.get(run.dataset, 5000)
    if run.protocol == "grouped":
        n_train = n * (1 - 1 / cfg["cv"]["n_folds"]) * (1 - cfg["cv"]["cal_frac"])
        n_eval = n - n_train
    else:
        k = (run.naive_aug_copies + 1) if run.protocol == "naive_aug" else 1
        n_train, n_eval = 0.7 * n * k, 0.3 * n * k
    ips = min(measured.get(run.model, PRIOR_TRAIN_IPS.get(run.model, 400)), LOADER_CAP_IPS)
    tr = run.train["epochs"] * n_train / ips
    ev = n_eval / (ips * INFER_FACTOR)
    return tr + ev + RUN_OVERHEAD_S


def benchmark(models, cfg, log_dir, iters=25, bs=64):
    import torch
    from .models import build_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = load_throughput(log_dir)
    for m in models:
        try:
            model = build_model(m, 9, 0.2, pretrained=False).to(dev).to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(model.parameters(), 1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
            x = torch.rand(bs, 3, cfg["img_size"], cfg["img_size"], device=dev).contiguous(memory_format=torch.channels_last)
            y = torch.randint(0, 9, (bs,), device=dev)
            for i in range(iters + 5):
                if i == 5:
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    loss = torch.nn.functional.cross_entropy(model(x).float(), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ips = iters * bs / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0
            out[m] = round(ips, 1)
            print(f"  {m:20s} {ips:8.1f} img/s  peak mem {mem:.2f} GB")
            del model, opt
            if dev.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: benchmark failed: {e}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / "throughput.json").write_text(json.dumps(out, indent=2))
    return out


def estimate_table(runs, cfg, measured=None):
    import pandas as pd
    rows = [dict(block=r.block, dataset=r.dataset, protocol=r.protocol, model=r.model, loss=r.loss,
                 epochs=r.train["epochs"], est_min=estimate_run_seconds(r, cfg, measured) / 60) for r in runs]
    df = pd.DataFrame(rows)
    t = df.groupby(["block", "dataset", "protocol", "model", "epochs"]).est_min.agg(["count", "mean", "sum"]).reset_index()
    t.columns = ["block", "dataset", "protocol", "model", "epochs", "runs", "min_per_run", "total_min"]
    t[["min_per_run", "total_min"]] = t[["min_per_run", "total_min"]].round(1)
    return t, df.est_min.sum() / 60
