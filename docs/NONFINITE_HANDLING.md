# Non-finite outputs: what happened and how they are handled

## What happened
Out of 252 training runs, only one image produced non-finite outputs: TrashNet `metal/metal55.jpg` (index 1265 in `splits/raw_artifacts/trashnet/index.csv`, class *metal*). With MobileNetV3-Large under mixed-precision (fp16/AMP) inference it gave NaN logits. This happened in all logit sets and in all 30 MC-dropout passes, which is consistent with an fp16 overflow. No other model gave non-finite outputs for this image, and the same is true of every other image, dataset and run.

The image falls in a different split role in three of the five grouped-CV folds:

| run (TrashNet, MobileNetV3-L, CE, seed 0) | split role | row | handling |
|---|---|---|---|
| fold 0 | conformal calibration (`calconf`, 203 rows) | 98 | dropped before conformal calibration (n_cal = 202) |
| fold 3 | conformal calibration (`calconf`, 203 rows) | 103 | dropped before conformal calibration (n_cal = 202) |
| fold 4 | test (505 rows) | 253 | dropped before every metric (504 test images evaluated) |

The `calfit` subsets, used to fit temperature and vector scaling, contain no non-finite rows, so those fits are unaffected.

## How the code handles it
- `n1lib/conformal.py` (`conformal_sets`) drops calibration rows whose probabilities are non-finite. Without this, the empirical quantile would be NaN and every prediction set would be empty.
- `n1lib/evaluate.py` (`evaluate_method`) drops non-finite rows per post-hoc method (raw / TS / VS / MC-dropout), for both calconf and test. The drop happens before the metrics, the saved `results/probs/*.npz`, the conformal sets and the set metrics are computed. Every dropped row is written to `results/nonfinite_rows.csv` with the columns `run_id, method, n_calconf, calconf_dropped, n_test, n_test_dropped, calconf_idx, test_idx`, and is logged.
- `n1_finalize.py` (stage `eval`) prints a per-run summary of the dropped rows.
- `tests/test_nonfinite.py` covers both paths.

The published `results/data/nonfinite_rows.csv` contains exactly the 12 rows described above: 3 runs × 4 methods.

## Effect on the reported results
Only the TrashNet / MobileNetV3-L row of the main benchmark table changes: `tab3_main_benchmark`, which is Table 2 of the paper and Supplementary Table S4. All other tables and all figures are unaffected, because every figure uses RealWaste only.

Before this fix, the NaN test row in fold 4 made that fold's ECE, NLL and Brier values NaN. As a result, those cells were averaged over 4 folds instead of 5. All cells of the row now use 5 folds:

| Cell | before | final (published) |
|---|---|---|
| Accuracy | 0.9248 ± 0.0102 | 0.9252 ± 0.0105 |
| Macro-F1 | 0.9116 ± 0.0155 | 0.9120 ± 0.0159 |
| ECE raw | 0.048 ± 0.007 (4 folds) | 0.045 ± 0.010 |
| ECE TS | 0.027 ± 0.007 (4 folds) | 0.029 ± 0.007 |
| NLL raw | 0.300 ± 0.046 (4 folds) | 0.276 ± 0.067 |
| NLL TS | 0.252 ± 0.032 (4 folds) | 0.237 ± 0.043 |
| Brier TS | 0.116 ± 0.012 (4 folds) | 0.111 ± 0.015 |
| AURC TS | 0.0112 ± 0.0035 | 0.0111 ± 0.0035 |
| Acc@90 % coverage | 0.9692 ± 0.0092 | 0.9692 ± 0.0092 |

## Sentence used in the Methods
> With mixed-precision (fp16) inference, a single TrashNet image (metal55.jpg) produced non-finite logits for MobileNetV3-L in 3 of its 5 grouped-CV folds; it was removed from the conformal calibration set in folds 0 and 3 (n_cal = 202 of 203) and from the test set in fold 4 (504 of 505 test images evaluated), and no other model, dataset or run (of 252) produced non-finite outputs.

## Remaining NaN values that are expected
- `metrics_long.csv`, column `temperature`: this value only exists for temperature scaling, so it is NaN for every other method.
- `conformal_long.csv`, column `sel_singleton_acc`: NaN when a configuration produces no singleton sets. This happens in one fold-4 row (TS / APS / α = 0.05 / Mondrian). The column is not used in any table.
- `tab1_datasets_duplicates`, column "Naive+aug … (%)": NaN for TrashNet and TrashBox, because the naive + augmentation protocol was run on RealWaste only. The paper shows this as "–".
