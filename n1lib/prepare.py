"""Stage 1 (CPU): index -> hashes -> duplicate groups -> splits -> uint8 cache. All resumable."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_write_json, read_json
from .data import DATASETS, build_cache, build_index
from .dedup import compute_embeddings, compute_hashes, embedding_pairs, group_duplicates
from .splits import grouped_cv_splits, leakage_report, naive_split


def ds_dir(cfg, ds):
    return Path(cfg["work_dir"]) / ds


def prepare_dataset(cfg, ds, preprocesses=("none",), naive_copies=(0,), log=print):
    wd = ds_dir(cfg, ds)
    wd.mkdir(parents=True, exist_ok=True)
    raw = Path(cfg["data_root"]) / "raw" / ds
    workers = int(cfg.get("prep_workers", 4))
    meta_p = wd / "index.csv"
    if not meta_p.exists():
        idx = build_index(ds, cfg["data_root"])
        exp = DATASETS.get(ds, {}).get("expected")
        counts = idx["cls"].value_counts().to_dict()
        if exp:
            for c, n in exp.items():
                if counts.get(c, 0) != n:
                    log(f"[warn] {ds}/{c}: found {counts.get(c, 0)} images, expected {n}")
        log(f"{ds}: {len(idx)} images, classes={counts}")
        h = compute_hashes(idx, raw, workers)
        h.to_csv(meta_p, index=False)
    h = pd.read_csv(meta_p, dtype={"md5": str})
    h["phash"] = h["phash"].astype(np.uint64)
    h["dhash"] = h["dhash"].astype(np.uint64)
    y = h["label"].to_numpy()

    dcfg = cfg.get("dedup", {})
    grp_p = wd / "groups.npy"
    if not grp_p.exists():
        extra = None
        if dcfg.get("embed", False):
            ep = wd / "embeddings.npy"
            if not ep.exists():
                np.save(ep, compute_embeddings(h, raw, dcfg.get("embed_model", "vit_small_patch14_dinov2.lvd142m")))
            emb = np.load(ep)
            ei, ej, es = embedding_pairs(emb, y, dcfg.get("embed_thresh", 0.95), dcfg.get("embed_same_class_only", True))
            pd.DataFrame(dict(i=ei, j=ej, cosine=es, path_i=h["path"].to_numpy()[ei], path_j=h["path"].to_numpy()[ej])
                         ).to_csv(wd / "embedding_pairs.csv", index=False)
            extra = (ei, ej)
            emb_sens = {}
            for t in (0.90, 0.92, 0.95, 0.98):
                a_, _, _ = embedding_pairs(emb, y, t, dcfg.get("embed_same_class_only", True))
                emb_sens[f"embed_pairs_cos_ge_{t}"] = int(len(a_))
        groups, pairs, summ = group_duplicates(h, dcfg.get("phash_thresh", 6), dcfg.get("dhash_thresh", 6), extra)
        summ["dataset"] = ds
        if dcfg.get("embed", False):
            summ.update(emb_sens, embed_thresh=dcfg.get("embed_thresh", 0.95))
        summ["class_counts"] = h["cls"].value_counts().to_dict()
        pairs.to_csv(wd / "duplicate_pairs.csv", index=False)
        atomic_write_json(wd / "dedup_summary.json", summ)
        np.save(grp_p, groups)
        log(f"{ds}: dedup -> {summ['n_dup_groups']} duplicate groups, {summ['n_images_in_dup_groups']} images "
            f"({summ['pct_images_in_dup_groups']}%), cross-class pairs {summ['n_cross_class_pairs']}")
    groups = np.load(grp_p)

    cv = cfg["cv"]
    sp = wd / "splits_grouped.npz"
    if not sp.exists():
        folds = grouped_cv_splits(y, groups, cv["n_folds"], cv["cal_frac"], cv["split_seed"],
                                  cv.get("conformal_split", "half"))
        np.savez(sp, **{f"f{k}_{part}": v for k, f in enumerate(folds) for part, v in f.items()})
        rep = []
        for k, f in enumerate(folds):
            r = {part: int(len(v)) for part, v in f.items()}
            r.update(fold=k, **leakage_report(f, groups))
            rep.append(r)
        atomic_write_json(wd / "splits_grouped_report.json", rep)
    for a in naive_copies:
        npth = wd / f"splits_naive_aug{a}.npz"
        if not npth.exists():
            s = naive_split(y, a, cv["split_seed"])
            np.savez(npth, **s)
            atomic_write_json(wd / f"splits_naive_aug{a}_report.json",
                              dict({k: int(len(v)) for k, v in s.items()}, **leakage_report(s, groups)))
    for pre in preprocesses:
        build_cache(h, raw, wd / f"cache_{pre}_{cfg['cache_size']}.npy", cfg["cache_size"], pre, workers)
    return h


def load_split(cfg, ds, protocol, fold, naive_copies=0):
    """Returns dict part -> items array (N,2) [image_idx, aug_id]."""
    wd = ds_dir(cfg, ds)
    if protocol == "grouped":
        z = np.load(wd / "splits_grouped.npz")
        out = {}
        for part in ("train", "calfit", "calconf", "test"):
            v = z[f"f{fold}_{part}"]
            out[part] = np.stack([v, np.zeros_like(v)], 1)
        return out
    a = naive_copies if protocol == "naive_aug" else 0
    z = np.load(wd / f"splits_naive_aug{a}.npz")
    return {k: z[k] for k in ("train", "calfit", "calconf", "test")}


def load_index(cfg, ds):
    return pd.read_csv(ds_dir(cfg, ds) / "index.csv", dtype={"md5": str})


def cache_path(cfg, ds, pre):
    return ds_dir(cfg, ds) / f"cache_{pre}_{cfg['cache_size']}.npy"
