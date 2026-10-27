"""Stage 2 (GPU): train one run with AMP + channels_last, epoch checkpoints (resumable), then dump
logits for calfit/calconf/test and MC-dropout logits. Writes <out>/runs/<run_id>/done.json."""
from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import atomic_write_json, get_device, python_info, seed_everything
from .data import CachedImages, dataset_classes
from .losses import make_loss
from .models import build_model
from .prepare import cache_path, load_index, load_split


def run_dir(cfg, run):
    return Path(cfg["out_dir"]) / "runs" / run.run_id


def is_done(cfg, run):
    return (run_dir(cfg, run) / "done.json").exists()


def _worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)


def _loader(ds, bs, shuffle, nw, seed, device):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=device.type == "cuda",
                      drop_last=shuffle and len(ds) > bs, persistent_workers=nw > 0, generator=g,
                      worker_init_fn=_worker_init, prefetch_factor=4 if nw > 0 else None)


def _to_input(x, device, cl):
    x = x.to(device, non_blocking=True).float().div_(255.0)
    return x.contiguous(memory_format=torch.channels_last) if cl else x


@torch.no_grad()
def predict(model, loader, device, amp, cl, mc_samples=0, max_batches=None):
    model.eval()
    L, MC, Y = [], [], []
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = _to_input(x, device, cl)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            f = model.features(x)
        f = f.float()
        L.append(model.fc(f).cpu())
        if mc_samples:
            MC.append(torch.stack([model.head(f, mc=True) for _ in range(mc_samples)], 0).half().cpu())
        Y.append(y)
    out = dict(logits=torch.cat(L).numpy(), y=torch.cat(Y).numpy())
    if mc_samples:
        out["mc"] = torch.cat(MC, 1).numpy()
    return out


def train_run(cfg, run, progress=None, run_index=0, n_runs=1):
    rd = run_dir(cfg, run)
    rd.mkdir(parents=True, exist_ok=True)
    tc = run.train
    device = get_device()
    amp = bool(tc["amp"]) and device.type == "cuda"
    cl = bool(tc["channels_last"]) and device.type == "cuda"
    seed_everything(1000 * run.seed + run.fold)
    torch.backends.cudnn.benchmark = True

    classes = dataset_classes(run.dataset, cfg["data_root"])
    idx = load_index(cfg, run.dataset)
    labels = idx["label"].to_numpy()
    split = load_split(cfg, run.dataset, run.protocol, run.fold, run.naive_aug_copies)
    cpath = cache_path(cfg, run.dataset, run.preprocess)
    mk = lambda part, mode: CachedImages(cpath, split[part], labels[split[part][:, 0]], mode, cfg["img_size"])
    nw = int(tc["num_workers"])
    tr_loader = _loader(mk("train", "train"), tc["batch_size"], True, nw, run.seed, device)

    model = build_model(run.model, len(classes), tc["drop_rate"], tc["pretrained"]).to(device)
    if cl:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.param_groups(tc["lr_backbone"], tc["lr_head"], tc["weight_decay"]))
    steps_per_epoch = len(tr_loader) if not tc["max_train_batches"] else min(len(tr_loader), tc["max_train_batches"])
    total = tc["epochs"] * steps_per_epoch
    warm = tc["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = make_loss(run.loss, tc)

    ck = rd / "ckpt.pt"
    start_epoch, history, t_used = 0, [], 0.0
    if ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_epoch, history, t_used = st["epoch"], st["history"], st.get("time", 0.0)
        torch.set_rng_state(st["rng_cpu"].cpu() if hasattr(st["rng_cpu"], "cpu") else st["rng_cpu"])
        if progress:
            progress.info(f"resume {run.run_id} from epoch {start_epoch}")

    for ep in range(start_epoch, tc["epochs"]):
        t0 = time.time()
        model.train()
        tl, tc_, tn = 0.0, 0, 0
        for bi, (x, y) in enumerate(tr_loader):
            if tc["max_train_batches"] and bi >= tc["max_train_batches"]:
                break
            x = _to_input(x, device, cl)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tc["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y); tc_ += (out.argmax(1) == y).sum().item(); tn += len(y)
        dt = time.time() - t0
        t_used += dt
        history.append(dict(epoch=ep + 1, train_loss=tl / max(1, tn), train_acc=tc_ / max(1, tn), sec=dt,
                            lr=opt.param_groups[0]["lr"]))
        st = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  scaler=scaler.state_dict(), epoch=ep + 1, history=history, time=t_used,
                  rng_cpu=torch.get_rng_state())
        torch.save(st, str(ck) + ".tmp")
        os.replace(str(ck) + ".tmp", ck)
        if os.environ.get("N1_STOP_AFTER_EPOCH") == str(ep + 1):  # used only to test resume
            raise KeyboardInterrupt("simulated interruption")
        if progress:
            remaining = (tc["epochs"] - ep - 1) * dt
            progress.log(stage="train", run_id=run.run_id, run_index=run_index, n_runs=n_runs, status="epoch",
                         epoch=ep + 1, epochs=tc["epochs"], train_loss=round(tl / max(1, tn), 4),
                         train_acc=round(tc_ / max(1, tn), 4), run_elapsed_s=round(t_used, 1),
                         note=f"run_eta_min={remaining / 60:.1f}")

    # ---- inference dumps -------------------------------------------------------------------
    t0 = time.time()
    dumps = {}
    for part in ("calfit", "calconf", "test"):
        ld = _loader(mk(part, "eval"), tc["eval_batch_size"], False, nw, 0, device)
        p = predict(model, ld, device, amp, cl, tc["mc_samples"], tc["max_eval_batches"])
        for k, v in p.items():
            dumps[f"{k}_{part}"] = v
        n = len(p["y"])
        dumps[f"items_{part}"] = split[part][:n]
    np.savez_compressed(rd / "preds.npz", **dumps)
    test_acc = float((dumps["logits_test"].argmax(1) == dumps["y_test"]).mean())
    if run.keep_weights:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   rd / "weights_fp16.pt")
    with open(rd / "history.csv", "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,sec,lr\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']:.5f},{h['train_acc']:.5f},{h['sec']:.1f},{h['lr']:.3e}\n")
    info = dict(run.keys(), classes=classes, test_acc_raw=test_acc, train_time_s=t_used,
                infer_time_s=time.time() - t0, epochs=tc["epochs"], train_cfg=tc,
                n_train=int(len(split["train"])), env=python_info())
    atomic_write_json(rd / "done.json", info)
    if ck.exists():
        ck.unlink()
    return info
