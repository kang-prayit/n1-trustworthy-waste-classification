# Leakage-aware re-evaluation, calibration and conformal selective prediction for real-world waste classification

[![Code licence: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data licence: CC BY 4.0](https://img.shields.io/badge/results%20%26%20splits-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA.md)
<!-- PLACEHOLDER: after the first Zenodo release, paste the DOI badge here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

This repository contains the code, the leakage-aware data splits and the final results of the paper

> Prayitno and Karisma Trinanda Putra. **Leakage-Aware Re-evaluation, Confidence Calibration, and Conformal
> Selective Prediction for Trustworthy Real-World Waste Classification.** *Scientific Reports* (under review), 2026.
> DOI: `[PLACEHOLDER: paper DOI]`

- **Repository:** https://github.com/kang-prayit/n1-trustworthy-waste-classification
- **Archived code (Zenodo):** DOI `[PLACEHOLDER: Zenodo software DOI]`
- **What is included:** code, tests, environment files, the duplicate-group and split files (`splits/`), and the final result CSVs, tables and figures (`results/`). Per-sample prediction probabilities (`probs.tar`, per-run test-set probabilities of all 252 runs) are not included in this repository; they are available from the corresponding author on reasonable request. The datasets are third-party and are not redistributed (see [Datasets](#datasets)).

## Overview

Public waste-image datasets contain exact and near-duplicate photographs. When images are split item by item, or are augmented *before* splitting, copies of the same object end up in both the training and the test set, and accuracy is overestimated. This repository does the following:

1. **Duplicate audit.** It detects duplicates with MD5, perceptual and difference hashing (pHash ≤ 6 and dHash ≤ 6) and DINOv2-S/14 cosine similarity (≥ 0.95, same class only). Duplicates are merged into groups with union–find.
2. **Grouped splits.** It builds **grouped stratified five-fold splits** (StratifiedGroupKFold, seed 2026). Each fold has four disjoint roles: *train*, *calibration-fit* (temperature/vector scaling), *conformal calibration* and *test*. No duplicate group spans two roles.
3. **Training.** It trains six architectures (ResNet101, a ResNet101 + InceptionV3 hybrid, EfficientNetV2-S, MobileNetV3-L, ConvNeXt-T and DeiT-S) on **RealWaste**, **TrashNet** and **TrashBox**. There are 252 training runs: the main benchmark, naive and augment-then-split baselines, loss variants, a segmentation ablation, and TrashNet and TrashBox replications.
4. **Calibration.** It evaluates temperature scaling, vector scaling, MC dropout and deep ensembles.
5. **Uncertainty and deferral.** It evaluates split conformal prediction (LAC, APS and RAPS; marginal and class-conditional/Mondrian; α = 0.05 and 0.10), selective prediction (risk–coverage, AURC) and the resulting human-review load.
6. **Analysis.** It runs statistical tests (Wilcoxon signed-rank with Holm correction, McNemar), cross-dataset transfer and Grad-CAM, and regenerates every table and figure of the paper.

## Repository structure

```
.
├── run_all.py                 resumable pipeline: prepare -> train -> eval -> cross -> gradcam -> analyze
├── n1_finalize.py             post-training stages (optional merge of runs trained elsewhere, eval, cross,
│                              gradcam, analyze, pack probs.tar); idempotent, used for the final results
├── download_data.py           download + verify RealWaste, TrashNet, TrashBox into data/raw/ (or --fake)
├── n1lib/                     the library
│   ├── common.py              config loading (YAML with `base:` inheritance), run expansion, paths
│   ├── dedup.py prepare.py    hashing, DINOv2 embeddings, duplicate groups, image caches
│   ├── splits.py              grouped / naive / augment-then-split protocols
│   ├── data.py models.py losses.py seg.py train.py runtime.py
│   ├── calib.py               temperature scaling, vector scaling, MC dropout, ensembles
│   ├── conformal.py           LAC / APS / RAPS, marginal and Mondrian
│   ├── metrics.py stats.py    ECE, MCE, NLL, Brier, AURC; Wilcoxon/Holm, McNemar
│   ├── evaluate.py            post-hoc evaluation of every run (incl. non-finite handling)
│   ├── crossdata.py gradcam.py
│   ├── analyze.py             tables (results/tables) and pipeline figures
│   └── figures.py figstyle.py schematics.py   publication figures (Figs 1-6, S1-S7)
├── configs/                   full.yaml (paper), quick.yaml, smoke_cpu.yaml, local.example.yaml
├── scripts/
│   ├── install_splits.py      copy the published splits into work/ and verify them against the images
│   ├── quarantine_corrupt.py  move the 3 undecodable TrashBox files out of data/raw
│   ├── export_preprocessing.py, make_contact_sheets.py, inspect_duplicates.py
│   ├── make_paper_figures.py  regenerate the manuscript figures
│   └── windows/               PowerShell helpers (setup check, background run, progress)
├── splits/                    duplicate audit + grouped CV and naive splits (no images), see splits/README_preprocessing.md
│   ├── <dataset>/             human-readable CSVs (image index, duplicate groups/links, folds, naive splits, ...)
│   └── raw_artifacts/<dataset>/   the exact files used by the pipeline (index.csv, groups.npy, splits_*.npz)
├── results/
│   ├── data/                  metrics_long.csv, conformal_long.csv.gz, perclass_long.csv.gz, crossdata.csv, nonfinite_rows.csv
│   ├── figdata/               curve data behind Figs 4a, 6a, S3-S5 (pooled from the per-run probabilities)
│   ├── tables/                tab1-tab10, tab2b, tab_s1 (.csv + .tex) and ALL_TABLES.md
│   └── figures/paper/         PNG previews of Figs 1-6 and S1-S7 + vector PDFs of the data figures
├── docs/NONFINITE_HANDLING.md
├── notebooks/colab_reproduce.ipynb
├── tests/                     pytest unit tests
├── requirements.txt, requirements-cpu-tested.txt, environment.yml
├── CITATION.cff, .zenodo.json, LICENSE (MIT), LICENSE-DATA.md (CC BY 4.0)
```

The figure PNGs in `results/figures/paper/` are previews, at most 1400 px wide, so that the repository stays small. Vector PDFs are included for the data figures; the full-resolution figures are those published with the article. `results/data/conformal_long.csv.gz` and `perclass_long.csv.gz` are gzip-compressed (pandas reads them directly, e.g. `pd.read_csv('results/data/conformal_long.csv.gz')`). The diagnostic figures written by the `analyze` stage (`outputs/figures/`) are not included; they are regenerated by the pipeline. Per-sample prediction probabilities (`probs.tar`, per-run test-set probabilities of all 252 runs) are not included in this repository; they are available from the corresponding author on reasonable request.

## Installation

The pipeline needs Python ≥ 3.10 and, for training, a CUDA GPU. The **recorded** environment of the paper's evaluation and analysis was an NVIDIA A100-SXM4-80GB with Python 3.13, PyTorch 2.11.0 (CUDA 13.0) and timm 1.0.30. Preprocessing used imagehash 4.3.2. Only these versions were recorded. All other packages are listed without a pinned version.

```bash
git clone https://github.com/kang-prayit/n1-trustworthy-waste-classification.git
cd n1-trustworthy-waste-classification
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -m pytest -q tests                                     # 24 tests, ~15 s on a CPU
```

You can also create a conda environment with `conda env create -f environment.yml`. `requirements-cpu-tested.txt` lists the exact versions of the CPU-only environment in which this release was checked: unit tests, CPU smoke run and regeneration of the paper figures.

## Datasets

**No images are redistributed.** Download them from the original sources and respect their licences. The licence statements below were checked on the source pages on 27 October 2026.

| Dataset | Source | Licence stated by the source | Images used |
|---|---|---|---|
| RealWaste (9 classes) | UCI ML Repository, https://doi.org/10.24432/C5SS4G (`https://archive.ics.uci.edu/static/public/908/realwaste.zip`); authors' repository https://github.com/sam-single/realwaste | The **UCI page states CC BY 4.0**, but the **authors' repository README and paper state CC BY-NC-SA 4.0**. We used the data for non-commercial research only; please do the same, or clarify the licence with the authors. | 4,752 |
| TrashNet (6 classes) | https://github.com/garythung/trashnet (`data/dataset-resized.zip`) | MIT licence (© 2017 Gary Thung) | 2,527 |
| TrashBox (7 classes) | https://github.com/nikhilvenkatkumsetty/TrashBox (train) and https://github.com/nikhilvenkatkumsetty/TrashBox-testandvalid (test, val) | No licence file. The README asks users to cite Kumsetty *et al.*, FRUCT 2022, https://doi.org/10.23919/FRUCT54823.2022.9770922 | 17,853 (train + test + val pooled, after quarantining 3 undecodable files) |

```bash
python download_data.py                  # all three datasets (~6.2 GB of zips); resumable
python scripts/quarantine_corrupt.py     # TrashBox: moves 3 undecodable train images to data/corrupt/
```

Expected layout, which `download_data.py` produces:

```
data/raw/realwaste/<class>/<file>.jpg            e.g. data/raw/realwaste/Vegetation/Vegetation_133.jpg
data/raw/trashnet/<class>/<file>.jpg             e.g. data/raw/trashnet/metal/metal55.jpg
data/raw/trashbox/<class>/<split>__<file>        split = train | test | val (official splits pooled)
```

To use the **exact splits of the paper**, install the published duplicate groups and folds. The script verifies that every indexed image exists. With `--check-md5` it also checks the image bytes against the MD5 sums in the index:

```bash
python scripts/install_splits.py --config configs/full.yaml --check-md5
```

After this step, the `prepare` stage only builds the image caches. It does not recompute hashes, DINOv2 embeddings or splits. DINOv2 cosine values can differ in the 4th decimal across hardware, so recomputing can flip pairs that lie exactly at the 0.95 threshold. For the file formats, see `splits/README_preprocessing.md`.

### Configuring paths

All paths are set in the YAML config. Nothing is hard-coded. Copy `configs/local.example.yaml` to `configs/local.yaml`, which git ignores, and set `data_root`, `work_dir`, `out_dir` and `log_dir`. Every other setting is inherited from `full.yaml` through `base:`. Relative paths are resolved against the repository root. Then pass `--config configs/local.yaml` to every command below.

## Reproducing the results

| Step | Command | Output |
|---|---|---|
| Smoke test (CPU, ~1 min) | `python download_data.py --fake && python run_all.py --config configs/smoke_cpu.yaml` | `outputs_smoke/` |
| Runtime estimate | `python run_all.py --config configs/full.yaml --estimate` | |
| **prepare** | `python run_all.py --config configs/full.yaml --stages prepare` | `work/<ds>/` (index, groups, splits, caches) |
| **train** (252 runs, resumable) | `python run_all.py --config configs/full.yaml --stages train` (optionally `--blocks A_main_realwaste ...`, `--max-runs N`) | `outputs/runs/<run_id>/` (`done.json`, `preds.npz`, weights for seed 0) |
| **eval** (CPU) | `python run_all.py --config configs/full.yaml --stages eval` | `outputs/results/{metrics,conformal,perclass}_long.csv`, `nonfinite_rows.csv`, `probs/` |
| **cross** (GPU) | `python run_all.py --config configs/full.yaml --stages cross` | `outputs/results/crossdata.csv` |
| **gradcam** (GPU) | `python run_all.py --config configs/full.yaml --stages gradcam` | `outputs/figures/fig_gradcam_rejected.*` |
| **analyze** (CPU) | `python run_all.py --config configs/full.yaml --stages analyze` | `outputs/tables/*`, `outputs/figures/*` |
| **finalize** | `python n1_finalize.py --config configs/full.yaml [--merge-from <other>/outputs/runs]` | eval, cross, gradcam, analyze and `results/probs.tar`, each skipped when up to date |
| paper figures | `python scripts/make_paper_figures.py --results outputs --raw data/raw` | `build/paper_figures/` |

`run_all.py` without `--stages` runs every stage in order. A finished run is never repeated, so if the job is interrupted, re-run the same command. The final results of the paper were produced with `n1_finalize.py`: training runs from two machines were merged with `--merge-from`, then eval, cross, gradcam and analyze were run on one A100.

**Without a GPU and without the datasets**, you can check the published numbers and figures. Run the tests, then regenerate all paper figures from the tables and CSVs in this repository. The curves of Figs 4a, 6a and S3–S5 come from `results/figdata/`, which holds small CSVs pooled from the per-run probabilities (seed 0, all folds). Fig 2c and Fig S7 additionally need the raw images in `--raw`. With `--probs-tar`, the curve data are recomputed from the probabilities themselves.

```bash
python scripts/make_paper_figures.py --raw data/raw --out build/paper_figures                         # uses results/figdata
python scripts/make_paper_figures.py --probs-tar probs.tar --raw data/raw --out build/paper_figures   # all figures
```

This was verified for this release: both ways, the regenerated PNGs are pixel-identical to the published figures. The only exception is Fig S6, which is a copy of the Grad-CAM render; the repository holds a 1400 px preview of it.

## From scripts to the paper's tables and figures

Tables are written by `n1lib/analyze.py` (stage `analyze`) to `results/tables/`. The manuscript tables are rounded or regrouped copies of these files; none of their numbers were computed separately.

| Paper | File in `results/tables/` | Produced from |
|---|---|---|
| Table 1 (datasets and duplicate structure) | `tab1_datasets_duplicates` (also `splits/preprocessing_summary.csv`) | prepare + analyze |
| Table 2 (benchmark on RealWaste) | `tab3_main_benchmark` (RealWaste rows) | eval + analyze |
| Suppl. Table S1 (optimism gap) | `tab2_optimism_gap` | eval + analyze |
| Suppl. Table S2 (per-class F1) | `tab7_perclass_f1` | eval + analyze |
| Suppl. Table S3 (statistical tests) | `tab8_statistical_tests`, `tab2b_optimism_gap_wilcoxon` | analyze (`n1lib/stats.py`) |
| Suppl. Table S4 (TrashNet and TrashBox benchmark) | `tab3_main_benchmark` (TrashNet / TrashBox rows) | eval + analyze |
| Suppl. Table S5 (training time) | `tab_s1_runtime` (regrouped by GPU) | `done.json` of every run |
| Suppl. Table S6 (calibration approaches) | `tab4_calibration_methods` | eval + analyze |
| Suppl. Table S7 (conformal prediction) | `tab5_conformal` | eval + analyze |
| Suppl. Table S8 (selective prediction) | `tab6_selective_prediction` | eval + analyze |
| Suppl. Table S9 (segmentation ablation) | `tab9_segmentation_ablation` | eval + analyze |
| Suppl. Table S10 (cross-dataset transfer) | `tab10_cross_dataset` | cross + analyze |

Figures are written by `scripts/make_paper_figures.py --only <id>`, which calls `n1lib/figures.py`. Previews are in `results/figures/paper/`.

| Paper | Id | Inputs |
|---|---|---|
| Fig 1 (pipeline schematic) | `fig1` | none (`n1lib/schematics.py`) |
| Fig 2 (duplicate audit and leakage) | `fig2` | `splits/` (+ raw images for panel c) |
| Fig 3 (optimism gap) | `fig3` | `tab2_optimism_gap` |
| Fig 4 (reliability and calibration) | `fig4` | `figdata/reliability_bins.csv` (from `probs/`) + `tab4_calibration_methods` |
| Fig 5 (conformal coverage and set size) | `fig5` | `tab5_conformal` |
| Fig 6 (selective prediction) | `fig6` | `figdata/risk_coverage.csv` (from `probs/`) + `tab6_selective_prediction` |
| Suppl. Fig S1 (threshold sensitivity) | `figS1` | `splits/<ds>/<ds>_threshold_sensitivity.csv` |
| Suppl. Fig S2 (per-class F1) | `figS2` | `tab7_perclass_f1` |
| Suppl. Fig S3 (confusion matrices) | `figS3` | `figdata/confusion_matrix.csv` (from `probs/`) |
| Suppl. Fig S4 (per-class ECE) | `figS4` | `figdata/perclass_ece.csv` (from `probs/`) |
| Suppl. Fig S5 (class-wise coverage) | `figS5` | `figdata/classwise_coverage.csv` (from `conformal_long`) |
| Suppl. Fig S6 (Grad-CAM of deferred images) | `figS6` | the render written by the `gradcam` stage (`fig_gradcam_rejected`), copied unchanged |
| Suppl. Fig S7 (split-leakage example) | `figS7` | `splits/realwaste/*` + raw images |

Supplementary Algorithms S1–S3 correspond to `n1lib/dedup.py`, `prepare.py` and `splits.py` (S1, the leakage-aware protocol), `n1lib/calib.py` (S2) and `n1lib/conformal.py` (S3). The `analyze` stage also writes diagnostic versions of these figures to `outputs/figures/`.

## Hardware and runtime

All numbers below are taken from `results/tables/tab_s1_runtime.csv`, the wall-clock times that the training code logged in each run's `done.json`.

| GPU | Training runs | Total time |
|---|---:|---:|
| NVIDIA GeForce RTX 4070 (12 GB) | 94 | 1,457 min ≈ 24.3 h |
| NVIDIA A100-SXM4-80GB | 125 | 333 min ≈ 5.6 h |
| NVIDIA A100-SXM4-40GB | 33 | 57 min ≈ 1.0 h |
| **Total** | **252** | **≈ 30.8 GPU-hours** |

The RTX 4070 runs used Python 3.11 with PyTorch 2.6.0 + CUDA 12.4. The A100 runs and all post-training stages used Python 3.13 with PyTorch 2.11.0 + CUDA 13.0. Both used timm 1.0.30. The final `eval` + `analyze` pass over all 252 runs takes about 1.6 min. TrashBox models were trained for 12 epochs, the naive + augmentation baseline for 15 epochs, and everything else for 30 epochs.

## Non-finite outputs

One TrashNet image (`metal/metal55.jpg`) produced NaN logits for MobileNetV3-L under fp16 inference. This affected 3 of 252 runs. The image was dropped from conformal calibration in folds 0 and 3 (n_cal = 202 of 203) and from the test set in fold 4 (504 of 505 images evaluated). Every dropped row is listed in `results/data/nonfinite_rows.csv`. Only the TrashNet / MobileNetV3-L row of `tab3_main_benchmark` (Suppl. Table S4) is affected. See [`docs/NONFINITE_HANDLING.md`](docs/NONFINITE_HANDLING.md) for details. One more value looks like an error but is not: in `tab1_datasets_duplicates`, the `nan` ("Naive+aug" column, TrashNet and TrashBox rows) means *not run* (the protocol was run on RealWaste only); the paper shows it as "–".

## Licence

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **Results, tables, figures and split files** (`results/`, `splits/`): CC BY 4.0 ([`LICENSE-DATA.md`](LICENSE-DATA.md)).
- **Dataset images** that appear inside some figures (Fig 2c, S6 and S7) are reproduced under the licences of their datasets. They are not covered by CC BY 4.0.
- The datasets themselves are not part of this repository; see [Datasets](#datasets).

## How to cite

If you use this code, the splits or the results, please cite the paper and the archived software. GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

```bibtex
@article{prayitno2026leakage,
  author  = {Prayitno and Putra, Karisma Trinanda},
  title   = {Leakage-Aware Re-evaluation, Confidence Calibration, and Conformal Selective Prediction
             for Trustworthy Real-World Waste Classification},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under review},
  doi     = {[PLACEHOLDER: paper DOI]}
}

@software{prayitno2026n1code,
  author    = {Prayitno and Putra, Karisma Trinanda},
  title     = {Leakage-Aware Re-evaluation, Confidence Calibration, and Conformal Selective Prediction
               for Trustworthy Real-World Waste Classification: code, splits and results},
  version   = {1.0.0},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {[PLACEHOLDER: Zenodo software DOI]},
  url       = {https://github.com/kang-prayit/n1-trustworthy-waste-classification}
}
```

Please also cite the datasets: RealWaste (Single, Iranmanesh & Raad, *Information* 14, 633, 2023, https://doi.org/10.3390/info14120633; UCI https://doi.org/10.24432/C5SS4G), TrashNet (Thung & Yang, 2016) and TrashBox (Kumsetty *et al.*, FRUCT 2022, https://doi.org/10.23919/FRUCT54823.2022.9770922).

## Contact

- Karisma Trinanda Putra (corresponding author), Department of Electrical Engineering, Universitas Muhammadiyah Yogyakarta, karisma@umy.ac.id, ORCID [0000-0003-2887-3527](https://orcid.org/0000-0003-2887-3527)
- Prayitno, Politeknik Negeri Semarang / Universitas Muhammadiyah Yogyakarta / SEAMEO SEAMOLEC, prayitno@polines.ac.id, ORCID [0000-0002-4794-2957](https://orcid.org/0000-0002-4794-2957)

Please report bugs and questions through the GitHub issue tracker. Requests for the per-run prediction probabilities: corresponding author.
