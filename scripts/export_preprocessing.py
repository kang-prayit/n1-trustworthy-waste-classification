#!/usr/bin/env python
"""Export the preprocessing artifacts of the prepare stage (work/<ds>/) as a self-describing,
release-ready supplementary data package (CSV/JSON/Markdown; no images).

    python scripts/export_preprocessing.py --work-dir work --data-root data --out release/preprocessing
    python scripts/export_preprocessing.py --work-dir <unzipped artifacts> --corrupt-list <...>/quarantined_files.txt

Inputs per dataset (written by run_all.py --stages prepare):
  index.csv, groups.npy, duplicate_pairs.csv, embedding_pairs.csv (optional), dedup_summary.json,
  splits_grouped.npz (+ _report.json), splits_naive_aug<k>.npz (+ _report.json)
Only numpy and pandas are required. Every number in the summary is recomputed from these files,
and the duplicate groups are re-derived from the exported links and checked against groups.npy.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROLE = {"train": "train", "calfit": "cal_ts", "calconf": "cal_conf", "test": "test"}
CRIT_MD5, CRIT_HASH, CRIT_EMB = "MD5", "pHash<=6&dHash<=6", "DINOv2cos>=0.95"
OFFICIAL = re.compile(r"^(train|test|val)__")


def relpath(p: str) -> str:
    """Dataset-relative POSIX path (strips any absolute Windows/Unix prefix up to raw/<ds>/)."""
    p = str(p).replace("\\", "/")
    m = re.search(r"/raw/[^/]+/(.+)$", p)
    return m.group(1) if m else p.lstrip("/")


def union_find(n, edges):
    par = np.arange(n)

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    return np.unique([f(k) for k in range(n)], return_inverse=True)[1]


def same_partition(a, b) -> bool:
    return len(set(zip(a.tolist(), b.tolist()))) == len(np.unique(a)) == len(np.unique(b))


def official_split(paths):
    return np.array([(OFFICIAL.match(Path(p).name).group(1) if OFFICIAL.match(Path(p).name) else "") for p in paths])


def leak_pct(src_mask, ref_mask, groups):
    """% of images in src_mask whose group contains at least one image of ref_mask."""
    if src_mask.sum() == 0:
        return None
    ref_groups = set(groups[ref_mask].tolist())
    return round(100 * float(np.mean([g in ref_groups for g in groups[src_mask]])), 2)


def export_dataset(ds: str, wd: Path, out: Path, embed_thresh: float, ph_t: int, dh_t: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(wd / "index.csv", dtype={"md5": str})
    idx["path"] = idx["path"].map(relpath)
    n = len(idx)
    groups = np.load(wd / "groups.npy")
    assert len(groups) == n, f"{ds}: groups.npy length {len(groups)} != index {n}"
    summ = json.loads((wd / "dedup_summary.json").read_text())
    paths, cls = idx["path"].to_numpy(), idx["cls"].to_numpy()

    # ---------------- links (edges) with their detection criterion ----------------
    links = {}

    def add(i, j, crit, **kw):
        i, j = (int(i), int(j)) if i < j else (int(j), int(i))
        e = links.setdefault((i, j), dict(i=i, j=j, criteria=set(), phash_dist=np.nan, dhash_dist=np.nan, cosine=np.nan))
        e["criteria"].add(crit)
        for k, v in kw.items():
            e[k] = v
    for _, ix in idx.groupby("md5").indices.items():   # exact duplicates: star to the first file
        for k in ix[1:]:
            add(ix[0], k, CRIT_MD5, phash_dist=0, dhash_dist=0)
    hp = pd.read_csv(wd / "duplicate_pairs.csv")
    md5 = idx["md5"].to_numpy()
    for r in hp.itertuples():
        add(r.i, r.j, CRIT_MD5 if md5[r.i] == md5[r.j] else CRIT_HASH, phash_dist=int(r.phash_dist), dhash_dist=int(r.dhash_dist))
    ep_file = wd / "embedding_pairs.csv"
    n_emb = 0
    if ep_file.exists():
        ep = pd.read_csv(ep_file)
        n_emb = len(ep)
        for r in ep.itertuples():
            add(r.i, r.j, CRIT_EMB, cosine=round(float(r.cosine), 4))
    L = pd.DataFrame(list(links.values())).sort_values(["i", "j"]) if links else pd.DataFrame(
        columns=["i", "j", "criteria", "phash_dist", "dhash_dist", "cosine"])
    order = [CRIT_MD5, CRIT_HASH, CRIT_EMB]
    L["criterion"] = L["criteria"].map(lambda s: "+".join(c for c in order if c in s))
    re_groups = union_find(n, zip(L["i"], L["j"]))
    assert same_partition(re_groups, groups), f"{ds}: exported links do not reproduce groups.npy"
    sizes = np.bincount(groups)

    L = L.assign(group_id=groups[L["i"].to_numpy()], relpath_i=paths[L["i"]], relpath_j=paths[L["j"]],
                 class_i=cls[L["i"]], class_j=cls[L["j"]])
    L["cross_class"] = L["class_i"] != L["class_j"]
    if ds == "trashbox":
        osp = official_split(paths)
        L["official_split_i"], L["official_split_j"] = osp[L["i"]], osp[L["j"]]
    cols = ["group_id", "relpath_i", "class_i", "relpath_j", "class_j", "criterion", "phash_dist", "dhash_dist",
            "cosine", "cross_class"] + (["official_split_i", "official_split_j"] if ds == "trashbox" else [])
    L = L.sort_values(["group_id", "relpath_i", "relpath_j"])
    Lw = L[cols].copy()
    for c in ("phash_dist", "dhash_dist"):
        Lw[c] = Lw[c].astype("Int64")
    Lw.to_csv(out / f"{ds}_duplicate_links.csv", index=False)

    # ---------------- per-image duplicate-group membership ----------------
    rows = []
    per_img = {}
    for r in L.itertuples():
        for a in (r.relpath_i, r.relpath_j):
            d = per_img.setdefault(a, dict(criteria=set(), ph=np.inf, dh=np.inf, cos=-np.inf))
            d["criteria"].update(r.criterion.split("+"))
            if not np.isnan(r.phash_dist):
                d["ph"] = min(d["ph"], r.phash_dist); d["dh"] = min(d["dh"], r.dhash_dist)
            if not np.isnan(r.cosine):
                d["cos"] = max(d["cos"], r.cosine)
    in_dup = sizes[groups] > 1
    for k in np.where(in_dup)[0]:
        d = per_img[paths[k]]
        g = groups[k]
        rows.append(dict(relpath=paths[k], **{"class": cls[k]}, group_id=int(g), group_size=int(sizes[g]),
                         group_classes="|".join(sorted(set(cls[groups == g]))),
                         criterion="+".join(c for c in order if c in d["criteria"]),
                         min_phash_dist=None if np.isinf(d["ph"]) else int(d["ph"]),
                         min_dhash_dist=None if np.isinf(d["dh"]) else int(d["dh"]),
                         max_cosine=None if np.isinf(d["cos"]) else d["cos"],
                         md5=md5[k]))
    G = pd.DataFrame(rows).sort_values(["group_id", "relpath"])
    if ds == "trashbox":
        G["official_split"] = official_split(G["relpath"])
    for c in ("min_phash_dist", "min_dhash_dist"):
        G[c] = G[c].astype("Int64")
    G.to_csv(out / f"{ds}_duplicate_groups.csv", index=False)

    # ---------------- cross-class duplicates ----------------
    X = Lw[Lw["cross_class"]]
    X.to_csv(out / f"{ds}_cross_class_duplicates.csv", index=False)
    mixed = [g for g in np.where(sizes > 1)[0] if len(set(cls[groups == g])) > 1]

    # ---------------- image index (all images) ----------------
    I = pd.DataFrame(dict(relpath=paths, **{"class": cls}, label=idx["label"], md5=md5,
                          phash_hex=[f"{int(v):016x}" for v in idx["phash"].astype(np.uint64)],
                          dhash_hex=[f"{int(v):016x}" for v in idx["dhash"].astype(np.uint64)],
                          width=idx["width"], height=idx["height"], group_id=groups, group_size=sizes[groups]))
    if ds == "trashbox":
        I["official_split"] = official_split(paths)
    # keep-one-per-group representative (lowest index) -> deduplicated view
    first = np.zeros(n, bool)
    first[np.unique(groups, return_index=True)[1]] = True
    I["dedup_keep"] = first
    I.to_csv(out / f"{ds}_image_index.csv", index=False)

    # ---------------- grouped 5-fold protocol ----------------
    z = np.load(wd / "splits_grouped.npz")
    nf = len({k.split("_")[0] for k in z.files})
    F = pd.DataFrame(dict(relpath=paths, **{"class": cls}, group_id=groups))
    test_fold = np.full(n, -1)
    for k in range(nf):
        role = np.full(n, "", dtype=object)
        for part, name in ROLE.items():
            v = z[f"f{k}_{part}"]
            assert (role[v] == "").all(), f"{ds}: image in two parts of fold {k}"
            role[v] = name
        assert (role != "").all(), f"{ds}: fold {k} does not cover all images"
        F[f"role_fold{k}"] = role
        test_fold[z[f"f{k}_test"]] = k
        for a in ROLE.values():   # group-disjointness check
            for b in ROLE.values():
                if a < b:
                    assert not set(groups[role == a]) & set(groups[role == b]), f"{ds}: group leak {a}/{b} fold {k}"
    F.insert(3, "test_fold", test_fold)
    F.to_csv(out / f"{ds}_folds_grouped.csv", index=False)
    fold_rep = [dict(fold=k, **{r: int((F[f"role_fold{k}"] == r).sum()) for r in ROLE.values()}) for k in range(nf)]

    # ---------------- naive splits ----------------
    naive_rep = {}
    for f in sorted(wd.glob("splits_naive_aug*.npz")):
        a = int(re.search(r"aug(\d+)", f.name).group(1))
        zz = np.load(f)
        parts = []
        for part, name in ROLE.items():
            it = zz[part]
            parts.append(pd.DataFrame(dict(relpath=paths[it[:, 0]], **{"class": cls[it[:, 0]]}, group_id=groups[it[:, 0]],
                                           aug_id=it[:, 1], role=name)))
        N = pd.concat(parts).sort_values(["relpath", "aug_id"])
        N.to_csv(out / f"{ds}_split_naive_aug{a}.csv", index=False)
        te, tr = N[N.role == "test"], N[N.role == "train"]
        naive_rep[f"aug{a}"] = dict(
            items={r: int((N.role == r).sum()) for r in ROLE.values()},
            pct_test_items_whose_source_image_in_train=round(100 * te.relpath.isin(set(tr.relpath)).mean(), 2),
            pct_test_items_whose_dup_group_in_train=round(100 * te.group_id.isin(set(tr.group_id)).mean(), 2))

    # ---------------- class statistics ----------------
    C = []
    for c in dict.fromkeys(cls):
        m = cls == c
        C.append({"class": c, "n_images": int(m.sum()), "n_images_in_dup_groups": int((m & in_dup).sum()),
                  "pct_in_dup_groups": round(100 * (m & in_dup).sum() / m.sum(), 2),
                  "n_dup_groups_touching_class": int(len(set(groups[m & in_dup]))),
                  "n_after_dedup_keep_one_per_group": int((m & first).sum()),
                  "n_removed_by_dedup": int((m & ~first).sum())})
    C = pd.DataFrame(C)
    tot = C.drop(columns="class").sum(numeric_only=True)
    tot["pct_in_dup_groups"] = round(100 * tot["n_images_in_dup_groups"] / tot["n_images"], 2)
    tot["n_dup_groups_touching_class"] = int((sizes > 1).sum())  # unique groups (classes can share mixed groups)
    C = pd.concat([C, pd.DataFrame([dict(tot, **{"class": "TOTAL"})])], ignore_index=True)
    for c in C.columns:
        if c not in ("class", "pct_in_dup_groups"):
            C[c] = C[c].astype(int)
    C.to_csv(out / f"{ds}_class_counts.csv", index=False)

    # ---------------- threshold sensitivity ----------------
    S = [dict(criterion="pHash Hamming <= t (dHash ignored), pairs", threshold=k.rsplit("_", 1)[1], n_pairs=v)
         for k, v in summ.items() if k.startswith("pairs_phash_le_")]
    S += [dict(criterion="DINOv2 ViT-S/14 cosine >= t (same class), pairs", threshold=k.rsplit("_", 1)[1], n_pairs=v)
          for k, v in summ.items() if k.startswith("embed_pairs_cos_ge_")]
    pd.DataFrame(S).to_csv(out / f"{ds}_threshold_sensitivity.csv", index=False)
    shutil.copy(wd / "dedup_summary.json", out / f"{ds}_dedup_summary_raw.json")

    # ---------------- summary numbers (recomputed) ----------------
    md5_counts = idx.groupby("md5")["md5"].transform("size").to_numpy()
    rec = dict(
        dataset=ds, n_classes=int(len(set(cls))), n_images=n, n_groups=int(len(sizes)),
        n_duplicate_groups=int((sizes > 1).sum()), n_images_in_duplicate_groups=int(in_dup.sum()),
        pct_images_in_duplicate_groups=round(100 * in_dup.mean(), 2),
        n_redundant_images=int(n - len(sizes)), n_images_after_dedup=int(len(sizes)), largest_group=int(sizes.max()),
        n_exact_md5_duplicate_images=int((md5_counts > 1).sum()),
        n_exact_md5_groups=int((idx["md5"].value_counts() > 1).sum()),
        n_hash_pairs=int(len(hp)), n_embedding_pairs=int(n_emb), n_links_total=int(len(L)),
        n_cross_class_links=int(L["cross_class"].sum()),
        n_cross_class_exact_md5_links=int((L["cross_class"] & L["criterion"].str.contains("MD5", regex=False)).sum()),
        n_mixed_label_groups=int(len(mixed)),
        n_groups_by_criterion_only={c: int(sum(1 for g in np.where(sizes > 1)[0]
                                               if set(L[L.group_id == g]["criterion"].str.split("+").explode()) == {c}))
                                    for c in order},
        phash_thresh=ph_t, dhash_thresh=dh_t, embed_thresh=embed_thresh,
        folds=fold_rep, naive=naive_rep,
        check_groups_reproduced_from_links=True, check_folds_group_disjoint=True,
    )
    if ds == "trashbox":
        osp = official_split(paths)
        tr, te, va = osp == "train", osp == "test", osp == "val"
        # groups restricted to subsets of criteria
        def groups_for(crits):
            e = [(int(a), int(b)) for (a, b), d in links.items() if d["criteria"] & crits]
            return union_find(n, e)
        g_m = groups_for({CRIT_MD5})
        g_mh = groups_for({CRIT_MD5, CRIT_HASH})
        rec["official_split_sizes"] = dict(train=int(tr.sum()), test=int(te.sum()), val=int(va.sum()))
        rec["official_leakage_pct"] = {
            lab: dict(test_in_train=leak_pct(te, tr, g), val_in_train=leak_pct(va, tr, g), test_in_val=leak_pct(te, va, g),
                      n_test_images_with_train_dup=int(sum(1 for x in g[te] if x in set(g[tr].tolist()))),
                      n_val_images_with_train_dup=int(sum(1 for x in g[va] if x in set(g[tr].tolist()))))
            for lab, g in (("md5_only", g_m), ("md5_or_hash", g_mh), ("all_criteria", groups))}
    (out / f"{ds}_preprocessing_stats.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


SUMMARY_COLS = [
    ("dataset", "Dataset"), ("n_classes", "Classes"), ("n_images", "Images (after quarantine)"),
    ("n_quarantined", "Quarantined corrupt files"),
    ("n_duplicate_groups", "Duplicate groups (size>=2)"), ("n_images_in_duplicate_groups", "Images in duplicate groups"),
    ("pct_images_in_duplicate_groups", "% images in duplicate groups"), ("largest_group", "Largest group"),
    ("n_redundant_images", "Redundant images (N - #groups)"), ("n_images_after_dedup", "Images after dedup (1 per group)"),
    ("n_exact_md5_duplicate_images", "Exact MD5 duplicate images"), ("n_exact_md5_groups", "Exact MD5 groups"),
    ("n_hash_pairs", "pHash&dHash pairs"), ("n_embedding_pairs", "DINOv2 pairs (cos>=0.95, same class)"),
    ("n_cross_class_links", "Cross-class duplicate pairs"), ("n_cross_class_exact_md5_links", "  of which exact MD5"),
    ("n_mixed_label_groups", "Mixed-label groups"),
    ("naive_aug0_leak", "Naive 70/15/15: % test items with dup group in train"),
    ("naive_aug3_leak", "Naive + offline aug x3: % test items with source image in train"),
    ("tb_test_in_train", "TrashBox official test with dup in train (% , all criteria)"),
    ("tb_val_in_train", "TrashBox official val with dup in train (% , all criteria)"),
    ("tb_test_in_train_mh", "TrashBox official test with dup in train (%, MD5 or pHash&dHash only)"),
    ("tb_val_in_train_mh", "TrashBox official val with dup in train (%, MD5 or pHash&dHash only)"),
    ("tb_test_in_train_md5", "TrashBox official test with exact copy in train (%)"),
    ("tb_val_in_train_md5", "TrashBox official val with exact copy in train (%)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out", default="release/preprocessing")
    ap.add_argument("--datasets", nargs="+", default=["realwaste", "trashnet", "trashbox"])
    ap.add_argument("--corrupt-list", default="data/corrupt/quarantined_files.txt")
    a = ap.parse_args()
    wd, out = Path(a.work_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # quarantined files -> dataset-relative list
    q = []
    cl = Path(a.corrupt_list)
    if cl.exists():
        for line in cl.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.strip().split("\t")[0].replace("\\", "/")
            m = re.search(r"corrupt/([^/]+)/(.+)$", p) or re.match(r"^([^/]+)/(.+\.\w+)$", p)
            if m and Path(p).suffix:
                q.append(dict(dataset=m.group(1), relpath=m.group(2), split_of_origin=(OFFICIAL.match(Path(p).name).group(1)
                              if OFFICIAL.match(Path(p).name) else ""), reason="PIL cannot decode (Image.open().convert('RGB').load() raises)"))
    Q = pd.DataFrame(q, columns=["dataset", "relpath", "split_of_origin", "reason"])
    Q.to_csv(out / "quarantined_corrupt_files.csv", index=False)
    recs = []
    for ds in a.datasets:
        d = wd / ds
        if not (d / "groups.npy").exists():
            print(f"skip {ds}: no artifacts in {d}")
            continue
        s = json.loads((d / "dedup_summary.json").read_text())
        r = export_dataset(ds, d, out / ds, s.get("embed_thresh", 0.95), s.get("phash_thresh", 6), s.get("dhash_thresh", 6))
        r["n_quarantined"] = int((Q.dataset == ds).sum())
        (out / ds / f"{ds}_preprocessing_stats.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        recs.append(r)
        print(f"{ds}: {r['n_images']} images, {r['n_duplicate_groups']} dup groups, {r['n_images_in_duplicate_groups']} "
              f"images ({r['pct_images_in_duplicate_groups']}%), cross-class {r['n_cross_class_links']}")
    rows = []
    for r in recs:
        row = {k: r.get(k) for k, _ in SUMMARY_COLS}
        row["naive_aug0_leak"] = r["naive"].get("aug0", {}).get("pct_test_items_whose_dup_group_in_train")
        row["naive_aug3_leak"] = r["naive"].get("aug3", {}).get("pct_test_items_whose_source_image_in_train")
        ol = r.get("official_leakage_pct")
        if ol:
            row.update(tb_test_in_train=ol["all_criteria"]["test_in_train"], tb_val_in_train=ol["all_criteria"]["val_in_train"],
                       tb_test_in_train_md5=ol["md5_only"]["test_in_train"],
                       tb_test_in_train_mh=ol["md5_or_hash"]["test_in_train"], tb_val_in_train_mh=ol["md5_or_hash"]["val_in_train"], tb_val_in_train_md5=ol["md5_only"]["val_in_train"])
        rows.append(row)
    T = pd.DataFrame(rows)[[k for k, _ in SUMMARY_COLS]]
    T.to_csv(out / "preprocessing_summary.csv", index=False)
    lab = dict(SUMMARY_COLS)
    W = T.set_index("dataset").T
    W.index = [lab[i] for i in W.index]
    def _f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return str(int(v)) if float(v).is_integer() and not isinstance(v, str) else str(v)
    fmt = W.astype(object).apply(lambda col: col.map(lambda v: v if isinstance(v, str) else _f(v)))
    md = ["| Quantity | " + " | ".join(fmt.columns) + " |", "|---|" + "---:|" * len(fmt.columns)]
    md += ["| " + i + " | " + " | ".join(str(v) for v in fmt.loc[i]) + " |" for i in fmt.index]
    (out / "preprocessing_summary.md").write_text(
        "# Preprocessing summary (computed from the released artifacts)\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    print((out / "preprocessing_summary.md").read_text())


if __name__ == "__main__":
    main()
