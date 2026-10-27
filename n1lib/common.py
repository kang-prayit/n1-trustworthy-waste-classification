"""Shared utilities: config loading, run enumeration, logging, atomic I/O, seeding."""
from __future__ import annotations

import copy
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN = dict(
    epochs=30, batch_size=64, eval_batch_size=128, lr_backbone=1e-4, lr_head=1e-3,
    weight_decay=0.05, warmup_epochs=1, drop_rate=0.2, num_workers=6, amp=True,
    channels_last=True, label_smoothing=0.1, focal_gamma=2.0, mc_samples=30,
    pretrained=True, grad_clip=1.0, max_train_batches=None, max_eval_batches=None,
)


def deep_update(base: dict, upd: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:  # simple inheritance
        base = load_config(path.parent / cfg.pop("base"))
        for k in ("experiments", "crossdata", "gradcam", "analysis", "datasets"):
            if k in cfg:  # these sections are replaced, not merged
                base.pop(k, None)
        for k in ("data_root", "work_dir", "out_dir", "_config_path"):
            if k in cfg:
                base.pop(k, None)
        cfg = deep_update(base, cfg)
    cfg.setdefault("train", {})
    cfg["train"] = deep_update(DEFAULT_TRAIN, cfg["train"])
    for k in ("data_root", "work_dir", "out_dir"):
        p = Path(cfg.get(k, k.split("_")[0]))
        cfg[k] = str(p if p.is_absolute() else (ROOT / p))
    cfg["_config_path"] = str(path)
    return cfg


@dataclass
class Run:
    dataset: str
    protocol: str          # grouped | naive | naive_aug
    preprocess: str        # none | seg
    model: str
    loss: str              # ce | ls | focal
    fold: int
    seed: int
    block: str
    train: dict = field(default_factory=dict)
    keep_weights: bool = False
    naive_aug_copies: int = 0

    @property
    def run_id(self) -> str:
        pre = "" if self.preprocess == "none" else f"-{self.preprocess}"
        return f"{self.dataset}-{self.protocol}{pre}-{self.model}-{self.loss}-f{self.fold}-s{self.seed}"

    @property
    def group_key(self) -> str:
        """Key shared by the seeds of one (dataset, protocol, model, loss, fold) -> ensembles."""
        pre = "" if self.preprocess == "none" else f"-{self.preprocess}"
        return f"{self.dataset}-{self.protocol}{pre}-{self.model}-{self.loss}-f{self.fold}"

    def keys(self) -> dict:
        return dict(run_id=self.run_id, dataset=self.dataset, protocol=self.protocol,
                    preprocess=self.preprocess, model=self.model, loss=self.loss,
                    fold=self.fold, seed=self.seed, block=self.block)


def expand_runs(cfg: dict) -> list[Run]:
    runs, seen = [], set()
    n_folds = cfg["cv"]["n_folds"]
    for blk in cfg["experiments"]:
        if not blk.get("enabled", True):
            continue
        train = deep_update(cfg["train"], blk.get("train"))
        folds = blk.get("folds", "all")
        folds = list(range(n_folds)) if folds == "all" else list(folds)
        if blk.get("protocol", "grouped") != "grouped":
            folds = [0]  # naive protocols use one random 70/15/15 split
        keep_seeds = set(blk.get("keep_weights_seeds", []))
        for seed in blk.get("seeds", [0]):
            for fold in folds:
                for model in blk["models"]:
                    for loss in blk.get("losses", ["ce"]):
                        r = Run(dataset=blk["dataset"], protocol=blk.get("protocol", "grouped"),
                                preprocess=blk.get("preprocess", "none"), model=model, loss=loss,
                                fold=fold, seed=seed, block=blk["name"], train=train,
                                keep_weights=seed in keep_seeds,
                                naive_aug_copies=blk.get("naive_aug_copies", 0))
                        if r.run_id in seen:
                            continue
                        seen.add(r.run_id)
                        runs.append(r)
    return runs


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Progress:
    """Append-only CSV progress log (logs/progress.csv) + human-readable log file."""
    FIELDS = ["time", "stage", "run_id", "run_index", "n_runs", "status", "epoch", "epochs",
              "train_loss", "train_acc", "test_acc", "run_elapsed_s", "total_elapsed_s", "eta_h", "note"]

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "progress.csv"
        self.txt_path = self.log_dir / "run_all.log"
        self.t0 = time.time()
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, **kw):
        kw.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        kw.setdefault("total_elapsed_s", round(time.time() - self.t0, 1))
        row = [kw.get(k, "") for k in self.FIELDS]
        for _ in range(5):  # tolerate the file being open in Excel for a moment
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                break
            except PermissionError:
                time.sleep(2)
        self.info(" | ".join(f"{k}={kw[k]}" for k in self.FIELDS if kw.get(k, "") != ""))

    def info(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(self.txt_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except PermissionError:
            pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def python_info() -> dict:
    import torch
    d = dict(python=sys.version.split()[0], torch=torch.__version__,
             cuda_available=torch.cuda.is_available(), cuda=torch.version.cuda)
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
    try:
        import timm
        d["timm"] = timm.__version__
    except Exception:
        pass
    return d
