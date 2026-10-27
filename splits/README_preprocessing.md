# Preprocessing artefacts — duplicate audit, corrupt-file quarantine and leakage-aware splits

Companion data package for:

> Prayitno and Karisma Trinanda Putra. *Leakage-Aware Re-evaluation, Confidence Calibration, and Conformal
> Selective Prediction for Trustworthy Real-World Waste Classification.* Scientific Reports (under review), 2026.
> Code: https://github.com/kang-prayit/n1-trustworthy-waste-classification · Archived release: Zenodo, DOI `[PLACEHOLDER: Zenodo software DOI]`

This package contains **no images**. Instead it holds everything you need to reproduce, inspect or reuse the
preprocessing of the three public datasets used in the paper:
- the near-duplicate graph and the duplicate groups,
- cross-class (label-noise) duplicates,
- corrupt files that were quarantined,
- the grouped 5-fold assignments with train / temperature-scaling / conformal-calibration / test roles,
- the naive (leaky) baseline splits,
- class counts before and after deduplication,
- the leakage of the official TrashBox test/val split into its train split.

Every number in `preprocessing_summary.*` is recomputed from the files in this package by
`scripts/export_preprocessing.py`. None of them were typed in by hand.

| Dataset | Source | Classes | Images used |
|---|---|---:|---:|
| RealWaste | UCI ML Repository, doi:10.24432/C5SS4G (Single, Iranmanesh & Raad 2023); UCI page: CC BY 4.0, authors' repository: CC BY-NC-SA 4.0 | 9 | 4,752 |
| TrashNet | github.com/garythung/trashnet (`dataset-resized`), MIT | 6 | 2,527 |
| TrashBox | github.com/nikhilvenkatkumsetty/TrashBox (+ TrashBox-testandvalid); Kumsetty et al., FRUCT 2022, doi:10.23919/FRUCT54823.2022.9770922 | 7 | 17,853 (train+test+val pooled, after quarantining 3 files) |

Relative paths (`relpath`) are given relative to each dataset's class-folder root after `download_data.py` in
the code repository has run. That is `data/raw/<dataset>/<class>/<file>`. TrashBox files are renamed
`<split>__<original name>`, where split is train/test/val, so the three official splits can be pooled without
any name collisions.

---

## 1. Procedure and thresholds

1. **Corrupt-file quarantine.** Before indexing, each image is fully decoded with PIL. Truncated-image loading
   is disabled for this check: `Image.open(p).convert("RGB").load()`. Any file that fails is moved to
   `data/corrupt/` and excluded. Only 3 TrashBox train images fail, with `UnidentifiedImageError`. They are
   listed in `quarantined_corrupt_files.csv`. Without this step those files would crash training or be
   silently decoded as garbage.
2. **Hashing.** For each image we compute MD5 (file bytes), 64-bit pHash and 64-bit dHash (`imagehash` 4.3.2,
   default hash size 8, on the RGB image).
3. **Duplicate links.** Two images are linked (an edge in the duplicate graph) if any of the following holds:
   - **MD5**: identical file bytes (exact copy, any class).
   - **pHash≤6 & dHash≤6**: Hamming distance ≤ 6 on *both* hashes (any class).
   - **DINOv2cos≥0.95**: cosine similarity ≥ 0.95 between L2-normalised DINOv2 ViT-S/14 CLS embeddings
     (`timm` `vit_small_patch14_dinov2.lvd142m`, 224 px, ImageNet normalisation). This criterion applies
     **only within the same class**, so it cannot merge genuinely different objects across labels.
4. **Groups.** Duplicate groups are the connected components of the graph (union–find). The relation is
   transitive, so a group can be larger than any single pair suggests. The RealWaste group of 16 is a chain of
   near-identical Vegetation photos. Every image gets a `group_id`, and singletons are groups of size 1.
5. **Grouped folds (leakage-aware protocol).** `StratifiedGroupKFold(n_splits=5, shuffle=True,
   random_state=2026)` over groups gives each image one test fold (≈20 %). Within every fold, the remaining
   groups are split again, grouped and stratified, into
   - `train` (≈64 % of all images),
   - `cal_ts` (≈8 %, used to fit temperature scaling and the other calibrators),
   - `cal_conf` (≈8 %, used to calibrate conformal prediction and selective thresholds).

   No group spans two roles in any fold. The export script checks this.
6. **Naive baseline (leaky protocol).** A stratified random 70 / 7.5 / 7.5 / 15 split over *items* that
   ignores groups (seed 2026):
   - `aug0`: one item per image.
   - `aug3` (RealWaste only): the "augment-then-split" protocol common in the literature. Each image is first
     expanded into 4 items (`aug_id` 0 = original, 1–3 = offline augmentations) and only then split. Copies of
     the same photograph therefore end up in both train and test.
7. **Deduplicated counts.** "After dedup" keeps one image per group, the lowest index (`dedup_keep=True`). The
   paper does *not* delete duplicates for training. It keeps them and keeps each group in one role. The
   dedup counts measure redundancy only.

**Threshold sensitivity.** `<ds>_threshold_sensitivity.csv` gives the number of pairs at pHash Hamming
t ∈ {0,2,4,6,8,10}, with dHash ignored, and at DINOv2 cosine t ∈ {0.90,0.92,0.95,0.98}. We chose the
operating point (pHash≤6 ∧ dHash≤6, cos≥0.95) after visually inspecting the pairs. The contact sheets (`scripts/make_contact_sheets.py`, needs the images) show
typical groups.

---

## 2. Summary (computed from this package)

| Quantity | RealWaste | TrashNet | TrashBox |
|---|---:|---:|---:|
| Images (after quarantine) | 4,752 | 2,527 | 17,853 |
| Quarantined corrupt files | 0 | 0 | 3 |
| Duplicate groups (size ≥ 2) | 16 | 39 | 855 |
| Images in duplicate groups | 63 (1.33 %) | 79 (3.13 %) | 1,889 (10.58 %) |
| Largest group | 16 | 3 | 53 |
| Redundant images (N − #groups) | 47 | 40 | 1,034 |
| Images after dedup | 4,705 | 2,487 | 16,819 |
| Exact-copy (MD5) images / groups | 0 / 0 | 6 / 3 | 724 / 344 |
| pHash&dHash pairs | 2 | 29 | 842 |
| DINOv2 pairs (same class) | 92 | 31 | 1,360 |
| Cross-class duplicate links (of which MD5) | 0 | 3 (3) | 11 (3) |
| Mixed-label groups | 0 | 3 | 7 |
| Naive aug0: % test items whose dup group is in train | 1.26 | 1.84 | 8.55 |
| Naive aug3: % test items whose source image is in train | 97.41 | – | – |

**TrashBox official split leakage.** Official sizes after quarantine are train 14,279, test 1,793 and
val 1,781. The table shows how many official test/val images have a duplicate in the official train split.

| Criterion | test → train | val → train |
|---|---:|---:|
| Any criterion (MD5, hash or DINOv2) | 8.37 % (150 images) | 9.10 % (162 images) |
| MD5 or pHash&dHash only | 6.36 % | 7.13 % |
| Exact copy (MD5) only | 2.90 % | 3.54 % |

Between test and val themselves, 1.17 % of test images have a duplicate in val.

**Notes worth knowing.**
- **TrashNet** has 3 exact copies filed under two classes: glass115 = metal91, glass176 = plastic152 and
  glass389 = plastic332. These are label noise. Each pair forms one mixed-label group that stays in a single
  role.
- **TrashBox** has 11 cross-class links:
  - 3 are exact copies (cardboard700 = paper2235, glass1433 / glass1832 = plastic1212) and are true label
    noise.
  - The other 8 are pHash&dHash-only links. Visual inspection (contact sheet `trashbox_cross_class_duplicates.jpg`)
    shows they are mostly **false positives**: different objects photographed on a plain white background,
    which hash alike. Keeping them grouped is conservative, because it can only reduce leakage. It affects
    < 0.1 % of images.
- In **RealWaste**, the duplicates are concentrated in *Vegetation*: 49 images in groups (11.2 %), 40 of them
  redundant. In TrashBox, *glass* is the most affected class (23.5 % of images in groups). See
  `<ds>_class_counts.csv`.

---

## 3. Files

```
splits/
├── README_preprocessing.md          this file
├── RINGKASAN_ID.md                  short summary in Indonesian
├── preprocessing_summary.md / .csv  dataset-level numbers (table above), one row per dataset in the CSV
├── quarantined_corrupt_files.csv    dataset, relpath, split_of_origin, reason
├── <dataset>/                       realwaste/, trashnet/, trashbox/
│   ├── <ds>_image_index.csv
│   ├── <ds>_duplicate_groups.csv
│   ├── <ds>_duplicate_links.csv
│   ├── <ds>_cross_class_duplicates.csv
│   ├── <ds>_folds_grouped.csv
│   ├── <ds>_split_naive_aug0.csv    (+ _aug3 for RealWaste)
│   ├── <ds>_class_counts.csv
│   ├── <ds>_threshold_sensitivity.csv
│   ├── <ds>_preprocessing_stats.json
│   └── <ds>_dedup_summary_raw.json
├── raw_artifacts/<ds>/              native pipeline files (index.csv, groups.npy, *.npz, pair lists, reports)
└── (scripts live in ../scripts/: export_preprocessing.py, make_contact_sheets.py, quarantine_corrupt.py, install_splits.py)
```

### Column definitions

**`<ds>_image_index.csv`** (one row per image, in pipeline index order):
- `relpath`, `class` and `label`: the integer label in alphabetical class order.
- `md5`.
- `phash_hex` and `dhash_hex`: 64-bit hashes in hex.
- `width` and `height`.
- `group_id` and `group_size`.
- `official_split`: TrashBox only (train/test/val).
- `dedup_keep`: True for the representative of each group.

**`<ds>_duplicate_groups.csv`** has one row per image in a group of size ≥ 2:
- `relpath`, `class`, `group_id` and `group_size`.
- `group_classes`: classes present, joined with `|`.
- `criterion`: the criteria of the links touching this image, joined with `+`.
- `min_phash_dist`, `min_dhash_dist` and `max_cosine`: the strongest similarity to any group member.
- `md5` and `official_split`.

**`<ds>_duplicate_links.csv`** has one row per edge of the duplicate graph:
- `group_id`.
- `relpath_i`, `class_i`, `relpath_j` and `class_j`.
- `criterion`: `MD5`, `pHash<=6&dHash<=6` and/or `DINOv2cos>=0.95`, joined with `+`.
- `phash_dist`, `dhash_dist` and `cosine`: similarity values, empty if not measured for that criterion.
- `cross_class`.
- `official_split_i` and `official_split_j`.

For MD5 groups, every member is linked to the first file (a star). The union of all links reproduces
`raw_artifacts/<ds>/groups.npy` exactly, and the export script checks this.

**`<ds>_cross_class_duplicates.csv`** has the same columns as the links file, filtered to `cross_class=True`.

**`<ds>_folds_grouped.csv`**: `relpath`, `class`, `group_id`, `test_fold` (0–4) and `role_fold0` … `role_fold4`
∈ {`train`, `cal_ts`, `cal_conf`, `test`}. Read each column as "the role of this image when fold k is the
test fold". The paper's grouped results use these assignments.

**`<ds>_split_naive_aug{0,3}.csv`**: `relpath`, `class`, `group_id`, `aug_id` and `role`. There is one row per
split item (image × augmentation copy).

**`<ds>_class_counts.csv`**: `class`, `n_images`, `n_images_in_dup_groups`, `pct_in_dup_groups`,
`n_dup_groups_touching_class`, `n_after_dedup_keep_one_per_group` and `n_removed_by_dedup`.

**`raw_artifacts/<ds>/`** holds the files that `n1lib/prepare.py` (stage `prepare`) writes and that training reads directly:
- `index.csv`.
- `groups.npy` (group id per index row).
- `splits_grouped.npz`, with keys `f{k}_{train,calfit,calconf,test}` holding int64 row indices.
- `splits_naive_aug{k}.npz`, with keys `train`, `calfit`, `calconf` and `test` holding (n, 2) arrays of
  `[row_index, aug_id]`.
- `duplicate_pairs.csv` (hash pairs) and `embedding_pairs.csv` (same-class pairs with cosine ≥ 0.95).
- The JSON reports.

The `calfit` and `calconf` keys are the `cal_ts` and `cal_conf` roles above. `python scripts/install_splits.py` copies these into
`work/<ds>/` of the code repository and checks them against the downloaded images; training then reuses the
exact splits of the paper and the embedding step is skipped.

---

## 4. Reproducing

**From the raw images, with the full pipeline (recommended).** Follow the code repository:

```bash
python download_data.py                      # RealWaste, TrashNet, TrashBox (+test/val) -> data/raw/
python scripts/quarantine_corrupt.py         # moves the 3 undecodable TrashBox files to data/corrupt/
python run_all.py --config configs/full.yaml --stages prepare   # hashes, DINOv2 embeddings, groups, splits -> work/
python scripts/export_preprocessing.py --work-dir work --out preprocessing \
       --corrupt-list data/corrupt/quarantined_files.txt
python scripts/make_contact_sheets.py --package preprocessing --raw-root data/raw   # optional, JPG thumbnails
```

The `prepare` stage is a small part of the total runtime and also runs on a CPU. The hashes and MD5 are
deterministic. DINOv2 cosines can differ in the 4th decimal across GPU/CPU and library versions, which may
flip pairs lying exactly at the 0.95 boundary. To get bit-identical splits, use `raw_artifacts/` as described
above.

**From this package alone.** `python scripts/export_preprocessing.py --work-dir splits/raw_artifacts --out <dir>
--corrupt-list splits/raw_artifacts/corrupt/quarantined_files.txt` regenerates every CSV and summary in this
package from `raw_artifacts/`. It needs only numpy, pandas and scikit-learn, and runs in about 10 s on a
CPU (verified: the output is byte-identical to the shipped CSVs). `make_contact_sheets.py` additionally needs the images.

---

## 5. Licence

The derived metadata in this package (CSV, JSON, NPY and NPZ: file names, hashes, group and fold assignments)
is released under **CC BY 4.0**. The scripts are MIT, matching the code repository. The
**images belong to their creators** and are not redistributed here. Obtain them from the original sources
and follow their terms (RealWaste: CC BY 4.0 on the UCI page but CC BY-NC-SA 4.0 in the authors' repository and paper, so use it for non-commercial research; TrashNet: MIT; TrashBox: no explicit licence in its repository,
so cite Kumsetty et al. 2022). The contact sheets (thumbnails of dataset images) are therefore not distributed; regenerate them locally with
`scripts/make_contact_sheets.py`.

Please cite the paper and the three datasets when you use these files.
