## tab1_datasets_duplicates

Datasets and perceptual-hash duplicate analysis (pHash and dHash agreement plus MD5).

| Dataset | Images | Classes | Exact (MD5) dup. images | Near-dup pairs | Dup. groups | Images in dup. groups (%) | Redundant images | Cross-class pairs | Largest group | Naive: test items w/ dup. group in train (%) | Naive+aug: test items w/ dup. group in train (%) | Grouped CV: test w/ dup. group in train (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| realwaste | 4752 | 9 | 0 | 2 | 16 | 63 (1.33) | 47 | 0 | 16 | 1.26 | 97.41 | 0.0 |
| trashbox | 17853 | 7 | 724 | 842 | 855 | 1889 (10.58) | 1034 | 11 | 53 | 8.55 | nan | 0.0 |
| trashnet | 2527 | 6 | 6 | 29 | 39 | 79 (3.13) | 40 | 3 | 3 | 1.84 | nan | 0.0 |

## tab2_optimism_gap

Optimism of the naive protocol versus leakage-aware evaluation (test accuracy, mean ± sd over seeds/folds).

| Model | Kumar et al. 2026 (reported, single split) | Naive + aug-before-split acc | Naive + aug-before-split macro-F1 | Naive random split acc | Naive random split macro-F1 | Leakage-aware grouped 5-fold CV acc | Leakage-aware grouped 5-fold CV macro-F1 | Optimism gap (pp) |
|---|---|---|---|---|---|---|---|---|
| ResNet101 | 0.9886 | 0.9977 ± 0.0002 | 0.9978 ± 0.0003 | 0.9336 ± 0.0072 | 0.9345 ± 0.0063 | 0.9356 ± 0.0052 | 0.9373 ± 0.0061 | 6.21 |
| ResNet101+InceptionV3 | 0.9000 | 0.9978 ± 0.0002 | 0.9978 ± 0.0004 | 0.9345 ± 0.0045 | 0.9386 ± 0.0048 | 0.9341 ± 0.0089 | 0.9359 ± 0.0095 | 6.37 |

## tab2b_optimism_gap_wilcoxon

Optimism gap: two-sided Wilcoxon signed-rank test on per-(fold, seed) differences naive minus grouped (paired on identical fold and seed); Holm-adjusted across model × dataset within each metric and naive protocol.

| Family | Metric | Naive protocol | Dataset | Model | Pairs | Naive mean | Grouped mean | Mean diff (naive - grouped) | Median diff | W | p | p (Holm) | Min attainable p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | naive_aug vs grouped | acc (raw) | naive_aug | realwaste | ResNet101+InceptionV3 | 3 | 0.9978 | 0.9443 | +0.0535 | +0.0536 | 0 | 0.25 | 0.5 | 0.25 |
| acc | naive_aug vs grouped | acc (raw) | naive_aug | realwaste | ResNet101 | 3 | 0.9977 | 0.9398 | +0.0579 | +0.0578 | 0 | 0.25 | 0.5 | 0.25 |
| acc | naive vs grouped | acc (raw) | naive | realwaste | ResNet101+InceptionV3 | 3 | 0.9345 | 0.9443 | -0.0098 | -0.0053 | 0 | 0.25 | 0.5 | 0.25 |
| acc | naive vs grouped | acc (raw) | naive | realwaste | ResNet101 | 3 | 0.9336 | 0.9398 | -0.0062 | -0.0036 | 0 | 0.25 | 0.5 | 0.25 |
| macro_f1 | naive_aug vs grouped | macro_f1 (raw) | naive_aug | realwaste | ResNet101+InceptionV3 | 3 | 0.9978 | 0.9476 | +0.0502 | +0.0507 | 0 | 0.25 | 0.5 | 0.25 |
| macro_f1 | naive_aug vs grouped | macro_f1 (raw) | naive_aug | realwaste | ResNet101 | 3 | 0.9978 | 0.9422 | +0.0556 | +0.0549 | 0 | 0.25 | 0.5 | 0.25 |
| macro_f1 | naive vs grouped | macro_f1 (raw) | naive | realwaste | ResNet101+InceptionV3 | 3 | 0.9386 | 0.9476 | -0.0090 | -0.0042 | 0 | 0.25 | 0.5 | 0.25 |
| macro_f1 | naive vs grouped | macro_f1 (raw) | naive | realwaste | ResNet101 | 3 | 0.9345 | 0.9422 | -0.0077 | -0.0045 | 0 | 0.25 | 0.5 | 0.25 |

## tab3_main_benchmark

Leakage-aware benchmark (grouped stratified 5-fold CV; mean ± sd over folds × seeds).

| Dataset | Model | Runs | Accuracy | Macro-F1 | ECE raw | ECE TS | NLL raw | NLL TS | Brier TS | AURC TS | Acc@90% cov. | Ensemble acc | Ensemble ECE (TS) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| realwaste | ResNet101 | 15 | 0.9356 ± 0.0052 | 0.9373 ± 0.0061 | 0.044 ± 0.005 | 0.019 ± 0.005 | 0.336 ± 0.061 | 0.223 ± 0.027 | 0.102 ± 0.010 | 0.0097 ± 0.0032 | 0.9703 ± 0.0073 | 0.9453 ± 0.0069 | 0.027 ± 0.004 |
| realwaste | ResNet101+InceptionV3 | 15 | 0.9341 ± 0.0089 | 0.9359 ± 0.0095 | 0.048 ± 0.010 | 0.022 ± 0.004 | 0.359 ± 0.066 | 0.224 ± 0.027 | 0.103 ± 0.012 | 0.0099 ± 0.0029 | 0.9706 ± 0.0073 | 0.9442 ± 0.0070 | 0.025 ± 0.009 |
| realwaste | EfficientNetV2-S | 15 | 0.9145 ± 0.0064 | 0.9167 ± 0.0076 | 0.054 ± 0.006 | 0.021 ± 0.003 | 0.355 ± 0.036 | 0.283 ± 0.023 | 0.130 ± 0.008 | 0.0179 ± 0.0051 | 0.9546 ± 0.0068 | 0.9274 ± 0.0057 | 0.029 ± 0.006 |
| realwaste | MobileNetV3-L | 15 | 0.9057 ± 0.0108 | 0.9079 ± 0.0111 | 0.052 ± 0.010 | 0.024 ± 0.003 | 0.374 ± 0.056 | 0.297 ± 0.030 | 0.141 ± 0.014 | 0.0175 ± 0.0043 | 0.9472 ± 0.0102 | 0.9120 ± 0.0142 | 0.023 ± 0.010 |
| realwaste | ConvNeXt-T | 15 | 0.9414 ± 0.0059 | 0.9433 ± 0.0068 | 0.038 ± 0.006 | 0.017 ± 0.005 | 0.242 ± 0.027 | 0.191 ± 0.017 | 0.089 ± 0.009 | 0.0079 ± 0.0018 | 0.9761 ± 0.0053 | 0.9476 ± 0.0051 | 0.018 ± 0.004 |
| realwaste | DeiT-S | 15 | 0.9428 ± 0.0092 | 0.9443 ± 0.0093 | 0.042 ± 0.008 | 0.016 ± 0.003 | 0.295 ± 0.054 | 0.199 ± 0.029 | 0.089 ± 0.013 | 0.0083 ± 0.0027 | 0.9758 ± 0.0065 | 0.9508 ± 0.0078 | 0.017 ± 0.004 |
| trashbox | ResNet101 | 5 | 0.9246 ± 0.0050 | 0.9242 ± 0.0050 | 0.047 ± 0.005 | 0.011 ± 0.004 | 0.334 ± 0.027 | 0.242 ± 0.014 | 0.113 ± 0.007 | 0.0104 ± 0.0013 | 0.9664 ± 0.0049 | – | – |
| trashbox | ResNet101+InceptionV3 | 5 | 0.9326 ± 0.0035 | 0.9323 ± 0.0035 | 0.043 ± 0.004 | 0.011 ± 0.003 | 0.318 ± 0.030 | 0.219 ± 0.012 | 0.101 ± 0.005 | 0.0091 ± 0.0013 | 0.9721 ± 0.0016 | – | – |
| trashbox | EfficientNetV2-S | 5 | 0.9383 ± 0.0038 | 0.9383 ± 0.0037 | 0.034 ± 0.004 | 0.011 ± 0.002 | 0.252 ± 0.021 | 0.205 ± 0.015 | 0.095 ± 0.006 | 0.0083 ± 0.0013 | 0.9748 ± 0.0031 | – | – |
| trashbox | MobileNetV3-L | 5 | 0.9100 ± 0.0033 | 0.9094 ± 0.0032 | 0.040 ± 0.004 | 0.014 ± 0.005 | 0.330 ± 0.009 | 0.285 ± 0.006 | 0.134 ± 0.003 | 0.0145 ± 0.0006 | 0.9521 ± 0.0021 | – | – |
| trashbox | ConvNeXt-T | 5 | 0.9414 ± 0.0028 | 0.9412 ± 0.0029 | 0.036 ± 0.002 | 0.012 ± 0.002 | 0.236 ± 0.021 | 0.188 ± 0.012 | 0.088 ± 0.005 | 0.0078 ± 0.0016 | 0.9769 ± 0.0025 | – | – |
| trashbox | DeiT-S | 5 | 0.9346 ± 0.0027 | 0.9346 ± 0.0029 | 0.044 ± 0.003 | 0.010 ± 0.004 | 0.310 ± 0.021 | 0.215 ± 0.009 | 0.099 ± 0.004 | 0.0088 ± 0.0010 | 0.9737 ± 0.0022 | – | – |
| trashnet | ResNet101 | 15 | 0.9475 ± 0.0078 | 0.9384 ± 0.0104 | 0.038 ± 0.007 | 0.023 ± 0.005 | 0.239 ± 0.073 | 0.171 ± 0.033 | 0.081 ± 0.013 | 0.0065 ± 0.0039 | 0.9824 ± 0.0067 | 0.9565 ± 0.0061 | 0.032 ± 0.007 |
| trashnet | ResNet101+InceptionV3 | 5 | 0.9450 ± 0.0088 | 0.9346 ± 0.0102 | 0.041 ± 0.008 | 0.023 ± 0.003 | 0.268 ± 0.073 | 0.179 ± 0.034 | 0.085 ± 0.014 | 0.0079 ± 0.0034 | 0.9793 ± 0.0077 | – | – |
| trashnet | EfficientNetV2-S | 5 | 0.9280 ± 0.0099 | 0.9131 ± 0.0148 | 0.043 ± 0.010 | 0.021 ± 0.006 | 0.264 ± 0.063 | 0.219 ± 0.038 | 0.105 ± 0.013 | 0.0111 ± 0.0048 | 0.9701 ± 0.0048 | – | – |
| trashnet | MobileNetV3-L | 5 | 0.9252 ± 0.0105 | 0.9120 ± 0.0159 | 0.045 ± 0.010 | 0.029 ± 0.007 | 0.276 ± 0.067 | 0.237 ± 0.043 | 0.111 ± 0.015 | 0.0111 ± 0.0035 | 0.9692 ± 0.0092 | – | – |
| trashnet | ConvNeXt-T | 15 | 0.9478 ± 0.0117 | 0.9336 ± 0.0155 | 0.035 ± 0.009 | 0.020 ± 0.006 | 0.193 ± 0.055 | 0.160 ± 0.035 | 0.078 ± 0.016 | 0.0050 ± 0.0022 | 0.9836 ± 0.0077 | 0.9517 ± 0.0146 | 0.025 ± 0.007 |
| trashnet | DeiT-S | 5 | 0.9418 ± 0.0071 | 0.9304 ± 0.0088 | 0.041 ± 0.009 | 0.021 ± 0.008 | 0.248 ± 0.054 | 0.176 ± 0.029 | 0.085 ± 0.011 | 0.0057 ± 0.0016 | 0.9815 ± 0.0053 | – | – |

## tab4_calibration_methods

Calibration methods on realwaste (ECE with 15 equal-width bins).

| Model | Approach | n | Accuracy | ECE | Classwise-ECE | MCE | Brier | NLL |
|---|---|---|---|---|---|---|---|---|
| ResNet101 | CE (uncalibrated) | 15 | 0.9356 ± 0.0052 | 0.044 ± 0.005 | 0.013 ± 0.001 | 0.449 ± 0.133 | 0.110 ± 0.010 | 0.336 ± 0.061 |
| ResNet101 | CE + temperature scaling | 15 | 0.9356 ± 0.0052 | 0.019 ± 0.005 | 0.012 ± 0.001 | 0.411 ± 0.171 | 0.102 ± 0.010 | 0.223 ± 0.027 |
| ResNet101 | CE + vector scaling | 15 | 0.9353 ± 0.0057 | 0.019 ± 0.005 | 0.010 ± 0.001 | 0.364 ± 0.117 | 0.100 ± 0.010 | 0.219 ± 0.027 |
| ResNet101 | CE + MC-dropout | 15 | 0.9357 ± 0.0048 | 0.045 ± 0.005 | 0.013 ± 0.001 | 0.469 ± 0.120 | 0.109 ± 0.010 | 0.332 ± 0.060 |
| ResNet101 | Label smoothing | 5 | 0.9320 ± 0.0071 | 0.076 ± 0.004 | 0.024 ± 0.001 | 0.427 ± 0.227 | 0.111 ± 0.007 | 0.294 ± 0.016 |
| ResNet101 | Label smoothing + TS | 5 | 0.9320 ± 0.0071 | 0.023 ± 0.006 | 0.012 ± 0.001 | 0.438 ± 0.219 | 0.108 ± 0.008 | 0.251 ± 0.021 |
| ResNet101 | Focal loss | 5 | 0.9306 ± 0.0095 | 0.024 ± 0.007 | 0.012 ± 0.002 | 0.238 ± 0.090 | 0.108 ± 0.014 | 0.237 ± 0.032 |
| ResNet101 | Focal loss + TS | 5 | 0.9306 ± 0.0095 | 0.021 ± 0.006 | 0.012 ± 0.002 | 0.479 ± 0.221 | 0.108 ± 0.014 | 0.229 ± 0.027 |
| ResNet101 | Deep ensemble | 5 | 0.9453 ± 0.0050 | 0.023 ± 0.008 | 0.010 ± 0.002 | 0.367 ± 0.132 | 0.086 ± 0.010 | 0.235 ± 0.046 |
| ResNet101 | Deep ensemble + TS | 5 | 0.9453 ± 0.0069 | 0.027 ± 0.004 | 0.012 ± 0.002 | 0.354 ± 0.076 | 0.087 ± 0.010 | 0.193 ± 0.024 |
| ResNet101+InceptionV3 | CE (uncalibrated) | 15 | 0.9341 ± 0.0089 | 0.048 ± 0.010 | 0.013 ± 0.002 | 0.497 ± 0.112 | 0.111 ± 0.016 | 0.359 ± 0.066 |
| ResNet101+InceptionV3 | CE + temperature scaling | 15 | 0.9341 ± 0.0089 | 0.022 ± 0.004 | 0.012 ± 0.002 | 0.438 ± 0.225 | 0.103 ± 0.012 | 0.224 ± 0.027 |
| ResNet101+InceptionV3 | CE + vector scaling | 15 | 0.9329 ± 0.0060 | 0.023 ± 0.006 | 0.011 ± 0.001 | 0.390 ± 0.158 | 0.103 ± 0.011 | 0.226 ± 0.029 |
| ResNet101+InceptionV3 | CE + MC-dropout | 15 | 0.9339 ± 0.0086 | 0.047 ± 0.009 | 0.013 ± 0.002 | 0.500 ± 0.134 | 0.111 ± 0.016 | 0.355 ± 0.065 |
| ResNet101+InceptionV3 | Label smoothing | 5 | 0.9379 ± 0.0055 | 0.095 ± 0.006 | 0.027 ± 0.001 | 0.375 ± 0.203 | 0.106 ± 0.006 | 0.288 ± 0.011 |
| ResNet101+InceptionV3 | Label smoothing + TS | 5 | 0.9379 ± 0.0055 | 0.016 ± 0.003 | 0.011 ± 0.001 | 0.392 ± 0.166 | 0.096 ± 0.006 | 0.220 ± 0.013 |
| ResNet101+InceptionV3 | Focal loss | 5 | 0.9287 ± 0.0051 | 0.028 ± 0.006 | 0.012 ± 0.001 | 0.378 ± 0.166 | 0.107 ± 0.004 | 0.242 ± 0.007 |
| ResNet101+InceptionV3 | Focal loss + TS | 5 | 0.9287 ± 0.0051 | 0.018 ± 0.003 | 0.012 ± 0.001 | 0.357 ± 0.202 | 0.106 ± 0.003 | 0.225 ± 0.003 |
| ResNet101+InceptionV3 | Deep ensemble | 5 | 0.9436 ± 0.0073 | 0.027 ± 0.005 | 0.010 ± 0.001 | 0.461 ± 0.147 | 0.088 ± 0.014 | 0.244 ± 0.051 |
| ResNet101+InceptionV3 | Deep ensemble + TS | 5 | 0.9442 ± 0.0070 | 0.025 ± 0.009 | 0.012 ± 0.002 | 0.357 ± 0.181 | 0.088 ± 0.012 | 0.192 ± 0.024 |
| EfficientNetV2-S | CE (uncalibrated) | 15 | 0.9145 ± 0.0064 | 0.054 ± 0.006 | 0.015 ± 0.001 | 0.430 ± 0.124 | 0.140 ± 0.009 | 0.355 ± 0.036 |
| EfficientNetV2-S | CE + temperature scaling | 15 | 0.9145 ± 0.0064 | 0.021 ± 0.003 | 0.014 ± 0.001 | 0.312 ± 0.086 | 0.130 ± 0.008 | 0.283 ± 0.023 |
| EfficientNetV2-S | CE + vector scaling | 15 | 0.9162 ± 0.0083 | 0.026 ± 0.007 | 0.012 ± 0.001 | 0.352 ± 0.129 | 0.129 ± 0.009 | 0.283 ± 0.021 |
| EfficientNetV2-S | CE + MC-dropout | 15 | 0.9142 ± 0.0072 | 0.053 ± 0.005 | 0.015 ± 0.001 | 0.461 ± 0.141 | 0.139 ± 0.009 | 0.351 ± 0.036 |
| EfficientNetV2-S | Label smoothing | 5 | 0.9236 ± 0.0052 | 0.088 ± 0.002 | 0.027 ± 0.001 | 0.546 ± 0.291 | 0.126 ± 0.006 | 0.326 ± 0.009 |
| EfficientNetV2-S | Label smoothing + TS | 5 | 0.9236 ± 0.0052 | 0.023 ± 0.004 | 0.012 ± 0.001 | 0.534 ± 0.248 | 0.119 ± 0.006 | 0.270 ± 0.011 |
| EfficientNetV2-S | Focal loss | 5 | 0.9043 ± 0.0095 | 0.024 ± 0.004 | 0.014 ± 0.001 | 0.339 ± 0.190 | 0.141 ± 0.012 | 0.291 ± 0.017 |
| EfficientNetV2-S | Focal loss + TS | 5 | 0.9043 ± 0.0095 | 0.021 ± 0.003 | 0.013 ± 0.001 | 0.288 ± 0.062 | 0.141 ± 0.012 | 0.291 ± 0.017 |
| EfficientNetV2-S | Deep ensemble | 5 | 0.9257 ± 0.0075 | 0.028 ± 0.003 | 0.012 ± 0.001 | 0.485 ± 0.165 | 0.113 ± 0.007 | 0.259 ± 0.028 |
| EfficientNetV2-S | Deep ensemble + TS | 5 | 0.9274 ± 0.0057 | 0.029 ± 0.006 | 0.014 ± 0.001 | 0.413 ± 0.304 | 0.112 ± 0.007 | 0.242 ± 0.019 |
| MobileNetV3-L | CE (uncalibrated) | 15 | 0.9057 ± 0.0108 | 0.052 ± 0.010 | 0.016 ± 0.002 | 0.469 ± 0.175 | 0.149 ± 0.018 | 0.374 ± 0.056 |
| MobileNetV3-L | CE + temperature scaling | 15 | 0.9057 ± 0.0108 | 0.024 ± 0.003 | 0.015 ± 0.002 | 0.430 ± 0.257 | 0.141 ± 0.014 | 0.297 ± 0.030 |
| MobileNetV3-L | CE + vector scaling | 15 | 0.9105 ± 0.0062 | 0.023 ± 0.009 | 0.012 ± 0.001 | 0.387 ± 0.204 | 0.136 ± 0.010 | 0.287 ± 0.028 |
| MobileNetV3-L | CE + MC-dropout | 15 | 0.9058 ± 0.0107 | 0.048 ± 0.011 | 0.016 ± 0.002 | 0.517 ± 0.218 | 0.147 ± 0.018 | 0.359 ± 0.053 |
| MobileNetV3-L | Label smoothing | 5 | 0.9141 ± 0.0108 | 0.124 ± 0.011 | 0.034 ± 0.001 | 0.332 ± 0.093 | 0.149 ± 0.011 | 0.378 ± 0.025 |
| MobileNetV3-L | Label smoothing + TS | 5 | 0.9141 ± 0.0108 | 0.022 ± 0.005 | 0.014 ± 0.001 | 0.375 ± 0.223 | 0.127 ± 0.013 | 0.285 ± 0.036 |
| MobileNetV3-L | Focal loss | 5 | 0.8954 ± 0.0121 | 0.025 ± 0.006 | 0.015 ± 0.001 | 0.398 ± 0.166 | 0.151 ± 0.014 | 0.306 ± 0.027 |
| MobileNetV3-L | Focal loss + TS | 5 | 0.8954 ± 0.0121 | 0.021 ± 0.003 | 0.015 ± 0.001 | 0.395 ± 0.165 | 0.151 ± 0.014 | 0.305 ± 0.028 |
| MobileNetV3-L | Deep ensemble | 5 | 0.9114 ± 0.0145 | 0.037 ± 0.011 | 0.015 ± 0.002 | 0.426 ± 0.152 | 0.133 ± 0.019 | 0.312 ± 0.053 |
| MobileNetV3-L | Deep ensemble + TS | 5 | 0.9120 ± 0.0142 | 0.023 ± 0.010 | 0.015 ± 0.001 | 0.371 ± 0.195 | 0.130 ± 0.016 | 0.274 ± 0.031 |
| ConvNeXt-T | CE (uncalibrated) | 15 | 0.9414 ± 0.0059 | 0.038 ± 0.006 | 0.011 ± 0.001 | 0.464 ± 0.101 | 0.095 ± 0.010 | 0.242 ± 0.027 |
| ConvNeXt-T | CE + temperature scaling | 15 | 0.9414 ± 0.0059 | 0.017 ± 0.005 | 0.010 ± 0.001 | 0.440 ± 0.122 | 0.089 ± 0.009 | 0.191 ± 0.017 |
| ConvNeXt-T | CE + vector scaling | 15 | 0.9414 ± 0.0071 | 0.019 ± 0.008 | 0.010 ± 0.002 | 0.489 ± 0.139 | 0.091 ± 0.011 | 0.198 ± 0.024 |
| ConvNeXt-T | CE + MC-dropout | 15 | 0.9414 ± 0.0057 | 0.038 ± 0.006 | 0.011 ± 0.001 | 0.468 ± 0.126 | 0.095 ± 0.010 | 0.238 ± 0.027 |
| ConvNeXt-T | Label smoothing | 5 | 0.9428 ± 0.0064 | 0.085 ± 0.005 | 0.023 ± 0.001 | 0.481 ± 0.211 | 0.097 ± 0.009 | 0.265 ± 0.016 |
| ConvNeXt-T | Label smoothing + TS | 5 | 0.9428 ± 0.0064 | 0.022 ± 0.009 | 0.010 ± 0.002 | 0.497 ± 0.151 | 0.092 ± 0.011 | 0.213 ± 0.023 |
| ConvNeXt-T | Focal loss | 5 | 0.9430 ± 0.0075 | 0.028 ± 0.007 | 0.012 ± 0.000 | 0.614 ± 0.132 | 0.092 ± 0.010 | 0.192 ± 0.016 |
| ConvNeXt-T | Focal loss + TS | 5 | 0.9430 ± 0.0075 | 0.021 ± 0.001 | 0.010 ± 0.001 | 0.500 ± 0.209 | 0.091 ± 0.011 | 0.186 ± 0.021 |
| ConvNeXt-T | Deep ensemble | 5 | 0.9480 ± 0.0055 | 0.026 ± 0.005 | 0.010 ± 0.001 | 0.425 ± 0.147 | 0.083 ± 0.009 | 0.197 ± 0.023 |
| ConvNeXt-T | Deep ensemble + TS | 5 | 0.9476 ± 0.0051 | 0.018 ± 0.004 | 0.010 ± 0.002 | 0.459 ± 0.181 | 0.081 ± 0.008 | 0.172 ± 0.016 |
| DeiT-S | CE (uncalibrated) | 15 | 0.9428 ± 0.0092 | 0.042 ± 0.008 | 0.011 ± 0.002 | 0.506 ± 0.093 | 0.097 ± 0.015 | 0.295 ± 0.054 |
| DeiT-S | CE + temperature scaling | 15 | 0.9428 ± 0.0092 | 0.016 ± 0.003 | 0.010 ± 0.001 | 0.467 ± 0.223 | 0.089 ± 0.013 | 0.199 ± 0.029 |
| DeiT-S | CE + vector scaling | 15 | 0.9411 ± 0.0077 | 0.019 ± 0.005 | 0.010 ± 0.001 | 0.425 ± 0.169 | 0.092 ± 0.013 | 0.206 ± 0.030 |
| DeiT-S | CE + MC-dropout | 15 | 0.9428 ± 0.0092 | 0.041 ± 0.007 | 0.011 ± 0.002 | 0.480 ± 0.108 | 0.096 ± 0.014 | 0.284 ± 0.052 |
| DeiT-S | Label smoothing | 5 | 0.9430 ± 0.0080 | 0.072 ± 0.006 | 0.020 ± 0.001 | 0.316 ± 0.109 | 0.096 ± 0.011 | 0.270 ± 0.021 |
| DeiT-S | Label smoothing + TS | 5 | 0.9430 ± 0.0080 | 0.025 ± 0.009 | 0.010 ± 0.001 | 0.504 ± 0.127 | 0.094 ± 0.013 | 0.237 ± 0.025 |
| DeiT-S | Focal loss | 5 | 0.9390 ± 0.0086 | 0.019 ± 0.007 | 0.010 ± 0.001 | 0.415 ± 0.251 | 0.094 ± 0.014 | 0.199 ± 0.028 |
| DeiT-S | Focal loss + TS | 5 | 0.9390 ± 0.0086 | 0.018 ± 0.003 | 0.010 ± 0.001 | 0.389 ± 0.201 | 0.094 ± 0.013 | 0.197 ± 0.025 |
| DeiT-S | Deep ensemble | 5 | 0.9514 ± 0.0072 | 0.027 ± 0.006 | 0.010 ± 0.002 | 0.483 ± 0.198 | 0.078 ± 0.013 | 0.209 ± 0.049 |
| DeiT-S | Deep ensemble + TS | 5 | 0.9508 ± 0.0078 | 0.017 ± 0.004 | 0.010 ± 0.002 | 0.412 ± 0.216 | 0.077 ± 0.012 | 0.170 ± 0.028 |

## tab5_conformal

Split conformal prediction on temperature-scaled probabilities (conformal-calibration split).

| Model | Score | α | Class-conditional | Coverage | Target | Avg set size | Singleton rate | Worst-class coverage |
|---|---|---|---|---|---|---|---|---|
| ResNet101 | LAC | 0.05 | no | 0.952 ± 0.015 | 0.95 | 1.06 ± 0.05 | 0.939 ± 0.030 | 0.858 ± 0.064 |
| ResNet101 | LAC | 0.05 | yes | 0.972 ± 0.011 | 0.95 | 1.53 ± 0.33 | 0.646 ± 0.195 | 0.915 ± 0.048 |
| ResNet101 | LAC | 0.1 | no | 0.903 ± 0.021 | 0.90 | 0.94 ± 0.03 | 0.941 ± 0.025 | 0.755 ± 0.065 |
| ResNet101 | LAC | 0.1 | yes | 0.931 ± 0.015 | 0.90 | 1.09 ± 0.06 | 0.887 ± 0.036 | 0.837 ± 0.051 |
| ResNet101 | APS | 0.05 | no | 0.954 ± 0.015 | 0.95 | 1.38 ± 0.08 | 0.763 ± 0.020 | 0.891 ± 0.038 |
| ResNet101 | APS | 0.05 | yes | 0.971 ± 0.011 | 0.95 | 1.99 ± 0.26 | 0.540 ± 0.139 | 0.884 ± 0.055 |
| ResNet101 | APS | 0.1 | no | 0.905 ± 0.016 | 0.90 | 1.15 ± 0.04 | 0.797 ± 0.016 | 0.814 ± 0.031 |
| ResNet101 | APS | 0.1 | yes | 0.936 ± 0.014 | 0.90 | 1.41 ± 0.11 | 0.734 ± 0.035 | 0.839 ± 0.036 |
| ResNet101 | RAPS | 0.05 | no | 0.953 ± 0.015 | 0.95 | 1.33 ± 0.07 | 0.759 ± 0.023 | 0.889 ± 0.040 |
| ResNet101 | RAPS | 0.05 | yes | 0.971 ± 0.010 | 0.95 | 2.08 ± 0.23 | 0.314 ± 0.119 | 0.884 ± 0.055 |
| ResNet101 | RAPS | 0.1 | no | 0.904 ± 0.015 | 0.90 | 1.14 ± 0.04 | 0.797 ± 0.016 | 0.816 ± 0.029 |
| ResNet101 | RAPS | 0.1 | yes | 0.936 ± 0.014 | 0.90 | 1.37 ± 0.14 | 0.699 ± 0.082 | 0.837 ± 0.035 |
| ResNet101+InceptionV3 | LAC | 0.05 | no | 0.952 ± 0.017 | 0.95 | 1.06 ± 0.06 | 0.939 ± 0.036 | 0.854 ± 0.065 |
| ResNet101+InceptionV3 | LAC | 0.05 | yes | 0.974 ± 0.008 | 0.95 | 1.48 ± 0.20 | 0.682 ± 0.108 | 0.935 ± 0.014 |
| ResNet101+InceptionV3 | LAC | 0.1 | no | 0.898 ± 0.020 | 0.90 | 0.94 ± 0.03 | 0.939 ± 0.032 | 0.749 ± 0.086 |
| ResNet101+InceptionV3 | LAC | 0.1 | yes | 0.930 ± 0.018 | 0.90 | 1.07 ± 0.08 | 0.901 ± 0.044 | 0.846 ± 0.042 |
| ResNet101+InceptionV3 | APS | 0.05 | no | 0.954 ± 0.012 | 0.95 | 1.37 ± 0.07 | 0.769 ± 0.020 | 0.892 ± 0.029 |
| ResNet101+InceptionV3 | APS | 0.05 | yes | 0.974 ± 0.012 | 0.95 | 1.93 ± 0.18 | 0.585 ± 0.072 | 0.897 ± 0.055 |
| ResNet101+InceptionV3 | APS | 0.1 | no | 0.902 ± 0.018 | 0.90 | 1.14 ± 0.05 | 0.801 ± 0.017 | 0.811 ± 0.034 |
| ResNet101+InceptionV3 | APS | 0.1 | yes | 0.939 ± 0.013 | 0.90 | 1.38 ± 0.11 | 0.742 ± 0.040 | 0.847 ± 0.044 |
| ResNet101+InceptionV3 | RAPS | 0.05 | no | 0.954 ± 0.012 | 0.95 | 1.32 ± 0.06 | 0.766 ± 0.019 | 0.892 ± 0.030 |
| ResNet101+InceptionV3 | RAPS | 0.05 | yes | 0.974 ± 0.012 | 0.95 | 2.00 ± 0.27 | 0.358 ± 0.133 | 0.895 ± 0.052 |
| ResNet101+InceptionV3 | RAPS | 0.1 | no | 0.903 ± 0.018 | 0.90 | 1.13 ± 0.04 | 0.802 ± 0.018 | 0.817 ± 0.036 |
| ResNet101+InceptionV3 | RAPS | 0.1 | yes | 0.938 ± 0.013 | 0.90 | 1.34 ± 0.13 | 0.723 ± 0.093 | 0.844 ± 0.042 |
| EfficientNetV2-S | LAC | 0.05 | no | 0.953 ± 0.010 | 0.95 | 1.14 ± 0.05 | 0.882 ± 0.036 | 0.870 ± 0.045 |
| EfficientNetV2-S | LAC | 0.05 | yes | 0.972 ± 0.008 | 0.95 | 1.96 ± 0.56 | 0.462 ± 0.218 | 0.923 ± 0.021 |
| EfficientNetV2-S | LAC | 0.1 | no | 0.895 ± 0.018 | 0.90 | 0.96 ± 0.03 | 0.954 ± 0.020 | 0.756 ± 0.077 |
| EfficientNetV2-S | LAC | 0.1 | yes | 0.932 ± 0.016 | 0.90 | 1.25 ± 0.23 | 0.770 ± 0.195 | 0.851 ± 0.039 |
| EfficientNetV2-S | APS | 0.05 | no | 0.958 ± 0.008 | 0.95 | 1.54 ± 0.11 | 0.715 ± 0.037 | 0.910 ± 0.026 |
| EfficientNetV2-S | APS | 0.05 | yes | 0.974 ± 0.009 | 0.95 | 2.48 ± 0.51 | 0.382 ± 0.165 | 0.896 ± 0.060 |
| EfficientNetV2-S | APS | 0.1 | no | 0.907 ± 0.016 | 0.90 | 1.23 ± 0.08 | 0.773 ± 0.028 | 0.832 ± 0.038 |
| EfficientNetV2-S | APS | 0.1 | yes | 0.938 ± 0.011 | 0.90 | 1.62 ± 0.23 | 0.624 ± 0.136 | 0.848 ± 0.036 |
| EfficientNetV2-S | RAPS | 0.05 | no | 0.958 ± 0.008 | 0.95 | 1.46 ± 0.10 | 0.709 ± 0.041 | 0.908 ± 0.025 |
| EfficientNetV2-S | RAPS | 0.05 | yes | 0.974 ± 0.009 | 0.95 | 2.27 ± 0.28 | 0.269 ± 0.115 | 0.897 ± 0.058 |
| EfficientNetV2-S | RAPS | 0.1 | no | 0.907 ± 0.017 | 0.90 | 1.21 ± 0.07 | 0.771 ± 0.030 | 0.831 ± 0.037 |
| EfficientNetV2-S | RAPS | 0.1 | yes | 0.939 ± 0.011 | 0.90 | 1.56 ± 0.20 | 0.593 ± 0.130 | 0.848 ± 0.036 |
| MobileNetV3-L | LAC | 0.05 | no | 0.949 ± 0.019 | 0.95 | 1.18 ± 0.11 | 0.852 ± 0.062 | 0.871 ± 0.057 |
| MobileNetV3-L | LAC | 0.05 | yes | 0.967 ± 0.012 | 0.95 | 1.66 ± 0.29 | 0.557 ± 0.200 | 0.898 ± 0.042 |
| MobileNetV3-L | LAC | 0.1 | no | 0.900 ± 0.017 | 0.90 | 0.99 ± 0.03 | 0.965 ± 0.011 | 0.782 ± 0.057 |
| MobileNetV3-L | LAC | 0.1 | yes | 0.933 ± 0.018 | 0.90 | 1.18 ± 0.11 | 0.842 ± 0.065 | 0.850 ± 0.029 |
| MobileNetV3-L | APS | 0.05 | no | 0.950 ± 0.018 | 0.95 | 1.49 ± 0.13 | 0.694 ± 0.032 | 0.896 ± 0.036 |
| MobileNetV3-L | APS | 0.05 | yes | 0.968 ± 0.015 | 0.95 | 2.14 ± 0.39 | 0.445 ± 0.183 | 0.863 ± 0.073 |
| MobileNetV3-L | APS | 0.1 | no | 0.901 ± 0.027 | 0.90 | 1.22 ± 0.06 | 0.753 ± 0.019 | 0.825 ± 0.048 |
| MobileNetV3-L | APS | 0.1 | yes | 0.931 ± 0.021 | 0.90 | 1.50 ± 0.14 | 0.663 ± 0.040 | 0.818 ± 0.058 |
| MobileNetV3-L | RAPS | 0.05 | no | 0.951 ± 0.018 | 0.95 | 1.45 ± 0.10 | 0.682 ± 0.037 | 0.896 ± 0.043 |
| MobileNetV3-L | RAPS | 0.05 | yes | 0.968 ± 0.014 | 0.95 | 2.23 ± 0.42 | 0.250 ± 0.221 | 0.865 ± 0.074 |
| MobileNetV3-L | RAPS | 0.1 | no | 0.901 ± 0.027 | 0.90 | 1.21 ± 0.06 | 0.752 ± 0.019 | 0.824 ± 0.046 |
| MobileNetV3-L | RAPS | 0.1 | yes | 0.930 ± 0.019 | 0.90 | 1.49 ± 0.19 | 0.613 ± 0.130 | 0.820 ± 0.055 |
| ConvNeXt-T | LAC | 0.05 | no | 0.949 ± 0.018 | 0.95 | 1.03 ± 0.04 | 0.962 ± 0.023 | 0.858 ± 0.059 |
| ConvNeXt-T | LAC | 0.05 | yes | 0.966 ± 0.016 | 0.95 | 1.41 ± 0.25 | 0.700 ± 0.173 | 0.894 ± 0.043 |
| ConvNeXt-T | LAC | 0.1 | no | 0.901 ± 0.022 | 0.90 | 0.93 ± 0.03 | 0.931 ± 0.027 | 0.756 ± 0.066 |
| ConvNeXt-T | LAC | 0.1 | yes | 0.935 ± 0.020 | 0.90 | 1.02 ± 0.04 | 0.935 ± 0.019 | 0.849 ± 0.046 |
| ConvNeXt-T | APS | 0.05 | no | 0.955 ± 0.011 | 0.95 | 1.23 ± 0.05 | 0.820 ± 0.021 | 0.907 ± 0.027 |
| ConvNeXt-T | APS | 0.05 | yes | 0.973 ± 0.010 | 0.95 | 1.82 ± 0.26 | 0.557 ± 0.159 | 0.893 ± 0.053 |
| ConvNeXt-T | APS | 0.1 | no | 0.906 ± 0.017 | 0.90 | 1.07 ± 0.03 | 0.838 ± 0.020 | 0.832 ± 0.032 |
| ConvNeXt-T | APS | 0.1 | yes | 0.933 ± 0.010 | 0.90 | 1.21 ± 0.06 | 0.807 ± 0.028 | 0.835 ± 0.031 |
| ConvNeXt-T | RAPS | 0.05 | no | 0.954 ± 0.011 | 0.95 | 1.20 ± 0.04 | 0.820 ± 0.021 | 0.905 ± 0.027 |
| ConvNeXt-T | RAPS | 0.05 | yes | 0.972 ± 0.011 | 0.95 | 1.72 ± 0.34 | 0.488 ± 0.228 | 0.892 ± 0.050 |
| ConvNeXt-T | RAPS | 0.1 | no | 0.905 ± 0.017 | 0.90 | 1.06 ± 0.03 | 0.838 ± 0.020 | 0.831 ± 0.033 |
| ConvNeXt-T | RAPS | 0.1 | yes | 0.933 ± 0.010 | 0.90 | 1.18 ± 0.06 | 0.803 ± 0.040 | 0.836 ± 0.030 |
| DeiT-S | LAC | 0.05 | no | 0.951 ± 0.016 | 0.95 | 1.03 ± 0.04 | 0.966 ± 0.026 | 0.860 ± 0.054 |
| DeiT-S | LAC | 0.05 | yes | 0.973 ± 0.012 | 0.95 | 1.47 ± 0.29 | 0.696 ± 0.161 | 0.931 ± 0.032 |
| DeiT-S | LAC | 0.1 | no | 0.901 ± 0.026 | 0.90 | 0.93 ± 0.03 | 0.930 ± 0.030 | 0.754 ± 0.070 |
| DeiT-S | LAC | 0.1 | yes | 0.933 ± 0.019 | 0.90 | 1.04 ± 0.04 | 0.922 ± 0.031 | 0.851 ± 0.063 |
| DeiT-S | APS | 0.05 | no | 0.954 ± 0.014 | 0.95 | 1.28 ± 0.07 | 0.815 ± 0.019 | 0.912 ± 0.036 |
| DeiT-S | APS | 0.05 | yes | 0.973 ± 0.011 | 0.95 | 1.81 ± 0.21 | 0.603 ± 0.103 | 0.893 ± 0.052 |
| DeiT-S | APS | 0.1 | no | 0.906 ± 0.017 | 0.90 | 1.08 ± 0.03 | 0.835 ± 0.013 | 0.838 ± 0.026 |
| DeiT-S | APS | 0.1 | yes | 0.937 ± 0.013 | 0.90 | 1.29 ± 0.08 | 0.781 ± 0.034 | 0.841 ± 0.037 |
| DeiT-S | RAPS | 0.05 | no | 0.953 ± 0.014 | 0.95 | 1.23 ± 0.05 | 0.814 ± 0.019 | 0.909 ± 0.035 |
| DeiT-S | RAPS | 0.05 | yes | 0.973 ± 0.011 | 0.95 | 1.82 ± 0.27 | 0.436 ± 0.179 | 0.894 ± 0.051 |
| DeiT-S | RAPS | 0.1 | no | 0.906 ± 0.017 | 0.90 | 1.07 ± 0.03 | 0.835 ± 0.013 | 0.837 ± 0.028 |
| DeiT-S | RAPS | 0.1 | yes | 0.937 ± 0.013 | 0.90 | 1.29 ± 0.16 | 0.741 ± 0.127 | 0.839 ± 0.038 |

## tab6_selective_prediction

Selective prediction: max-softmax versus conformal set-size deferral.

| Model | Full-coverage acc | AURC | E-AURC | Acc@80% (MSP) | Acc@90% (MSP) | AURC raw | Singleton coverage (APS α=0.1) | Singleton acc (APS α=0.1) | Human review load (%) |
|---|---|---|---|---|---|---|---|---|---|
| ResNet101 | 0.9356 ± 0.0052 | 0.0097 ± 0.0032 | 0.0076 ± 0.0031 | 0.9846 ± 0.0050 | 0.9703 ± 0.0073 | 0.0096 ± 0.0031 | 0.797 ± 0.016 | 0.9740 ± 0.0087 | 20.3 ± 1.6 |
| ResNet101+InceptionV3 | 0.9341 ± 0.0089 | 0.0099 ± 0.0029 | 0.0077 ± 0.0026 | 0.9832 ± 0.0061 | 0.9706 ± 0.0073 | 0.0100 ± 0.0028 | 0.801 ± 0.017 | 0.9719 ± 0.0084 | 19.9 ± 1.7 |
| EfficientNetV2-S | 0.9145 ± 0.0064 | 0.0179 ± 0.0051 | 0.0141 ± 0.0048 | 0.9763 ± 0.0050 | 0.9546 ± 0.0068 | 0.0172 ± 0.0048 | 0.773 ± 0.028 | 0.9680 ± 0.0054 | 22.7 ± 2.8 |
| MobileNetV3-L | 0.9057 ± 0.0108 | 0.0175 ± 0.0043 | 0.0128 ± 0.0034 | 0.9713 ± 0.0062 | 0.9472 ± 0.0102 | 0.0172 ± 0.0042 | 0.753 ± 0.019 | 0.9631 ± 0.0104 | 24.7 ± 1.9 |
| ConvNeXt-T | 0.9414 ± 0.0059 | 0.0079 ± 0.0018 | 0.0061 ± 0.0015 | 0.9875 ± 0.0045 | 0.9761 ± 0.0053 | 0.0078 ± 0.0017 | 0.838 ± 0.020 | 0.9724 ± 0.0049 | 16.2 ± 2.0 |
| DeiT-S | 0.9428 ± 0.0092 | 0.0083 ± 0.0027 | 0.0066 ± 0.0023 | 0.9874 ± 0.0060 | 0.9758 ± 0.0065 | 0.0082 ± 0.0026 | 0.835 ± 0.013 | 0.9740 ± 0.0079 | 16.5 ± 1.3 |

## tab7_perclass_f1

Per-class F1 (mean ± sd over folds × seeds).

| Model | Cardboard | Food Organics | Glass | Metal | Miscellaneous Trash | Paper | Plastic | Textile Trash | Vegetation |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-T | 0.945 ± 0.010 | 0.971 ± 0.011 | 0.964 ± 0.011 | 0.948 ± 0.010 | 0.870 ± 0.021 | 0.954 ± 0.011 | 0.925 ± 0.009 | 0.934 ± 0.026 | 0.978 ± 0.012 |
| DeiT-S | 0.952 ± 0.015 | 0.973 ± 0.012 | 0.959 ± 0.018 | 0.948 ± 0.014 | 0.879 ± 0.026 | 0.950 ± 0.018 | 0.929 ± 0.015 | 0.934 ± 0.023 | 0.975 ± 0.010 |
| EfficientNetV2-S | 0.913 ± 0.019 | 0.951 ± 0.013 | 0.933 ± 0.013 | 0.924 ± 0.013 | 0.836 ± 0.030 | 0.921 ± 0.008 | 0.894 ± 0.014 | 0.909 ± 0.023 | 0.969 ± 0.012 |
| ResNet101+InceptionV3 | 0.941 ± 0.016 | 0.965 ± 0.015 | 0.957 ± 0.011 | 0.936 ± 0.015 | 0.860 ± 0.023 | 0.953 ± 0.014 | 0.920 ± 0.020 | 0.921 ± 0.032 | 0.972 ± 0.014 |
| MobileNetV3-L | 0.889 ± 0.037 | 0.949 ± 0.018 | 0.929 ± 0.022 | 0.908 ± 0.016 | 0.841 ± 0.033 | 0.915 ± 0.018 | 0.889 ± 0.018 | 0.889 ± 0.028 | 0.961 ± 0.009 |
| ResNet101 | 0.943 ± 0.019 | 0.965 ± 0.011 | 0.952 ± 0.017 | 0.935 ± 0.010 | 0.867 ± 0.021 | 0.953 ± 0.011 | 0.924 ± 0.013 | 0.926 ± 0.023 | 0.969 ± 0.012 |

## tab8_statistical_tests

Paired statistical tests (Holm-corrected within each family).

| Test | Metric | Comparison | Pairs | Mean diff | p | p (Holm) |
|---|---|---|---|---|---|---|
| Wilcoxon signed-rank | acc (raw) | ConvNeXt-T vs ResNet101 | 15 | +0.0058 | 0.01746 | 0.03492 |
| Wilcoxon signed-rank | acc (raw) | DeiT-S vs ResNet101 | 15 | +0.0072 | 0.006957 | 0.02087 |
| Wilcoxon signed-rank | acc (raw) | EfficientNetV2-S vs ResNet101 | 15 | -0.0211 | 0.0006517 | 0.002607 |
| Wilcoxon signed-rank | acc (raw) | ResNet101+InceptionV3 vs ResNet101 | 15 | -0.0015 | 0.57 | 0.57 |
| Wilcoxon signed-rank | acc (raw) | MobileNetV3-L vs ResNet101 | 15 | -0.0299 | 6.104e-05 | 0.0003052 |
| Wilcoxon signed-rank | macro_f1 (raw) | ConvNeXt-T vs ResNet101 | 15 | +0.0060 | 0.03015 | 0.0603 |
| Wilcoxon signed-rank | macro_f1 (raw) | DeiT-S vs ResNet101 | 15 | +0.0071 | 0.001526 | 0.004578 |
| Wilcoxon signed-rank | macro_f1 (raw) | EfficientNetV2-S vs ResNet101 | 15 | -0.0205 | 6.104e-05 | 0.0003052 |
| Wilcoxon signed-rank | macro_f1 (raw) | ResNet101+InceptionV3 vs ResNet101 | 15 | -0.0013 | 0.5245 | 0.5245 |
| Wilcoxon signed-rank | macro_f1 (raw) | MobileNetV3-L vs ResNet101 | 15 | -0.0294 | 6.104e-05 | 0.0003052 |
| Wilcoxon signed-rank | ece (ts) | ConvNeXt-T vs ResNet101 | 15 | -0.0028 | 0.2078 | 0.4543 |
| Wilcoxon signed-rank | ece (ts) | DeiT-S vs ResNet101 | 15 | -0.0030 | 0.073 | 0.292 |
| Wilcoxon signed-rank | ece (ts) | EfficientNetV2-S vs ResNet101 | 15 | +0.0021 | 0.3028 | 0.4543 |
| Wilcoxon signed-rank | ece (ts) | ResNet101+InceptionV3 vs ResNet101 | 15 | +0.0024 | 0.1514 | 0.4543 |
| Wilcoxon signed-rank | ece (ts) | MobileNetV3-L vs ResNet101 | 15 | +0.0049 | 0.008362 | 0.04181 |
| Wilcoxon signed-rank | nll (ts) | ConvNeXt-T vs ResNet101 | 15 | -0.0315 | 0.002014 | 0.004028 |
| Wilcoxon signed-rank | nll (ts) | DeiT-S vs ResNet101 | 15 | -0.0237 | 0.0004272 | 0.001282 |
| Wilcoxon signed-rank | nll (ts) | EfficientNetV2-S vs ResNet101 | 15 | +0.0603 | 6.104e-05 | 0.0003052 |
| Wilcoxon signed-rank | nll (ts) | ResNet101+InceptionV3 vs ResNet101 | 15 | +0.0016 | 0.5995 | 0.5995 |
| Wilcoxon signed-rank | nll (ts) | MobileNetV3-L vs ResNet101 | 15 | +0.0739 | 6.104e-05 | 0.0003052 |
| Wilcoxon signed-rank | ECE raw vs TS | ConvNeXt-T | 15 | +0.0217 | 6.104e-05 |  |
| Wilcoxon signed-rank | ECE raw vs TS | DeiT-S | 15 | +0.0256 | 6.104e-05 |  |
| Wilcoxon signed-rank | ECE raw vs TS | EfficientNetV2-S | 15 | +0.0322 | 6.104e-05 |  |
| Wilcoxon signed-rank | ECE raw vs TS | ResNet101+InceptionV3 | 15 | +0.0258 | 6.104e-05 |  |
| Wilcoxon signed-rank | ECE raw vs TS | MobileNetV3-L | 15 | +0.0279 | 6.104e-05 |  |
| Wilcoxon signed-rank | ECE raw vs TS | ResNet101 | 15 | +0.0249 | 6.104e-05 |  |
| McNemar (exact) | accuracy (pooled OOF, seed 0) | ConvNeXt-T vs ResNet101 | 4752 | +0.0036 (b=151, c=134) | 0.3433 | 0.6865 |
| McNemar (exact) | accuracy (pooled OOF, seed 0) | DeiT-S vs ResNet101 | 4752 | +0.0074 (b=148, c=113) | 0.03513 | 0.1054 |
| McNemar (exact) | accuracy (pooled OOF, seed 0) | EfficientNetV2-S vs ResNet101 | 4752 | -0.0227 (b=117, c=225) | 5.431e-09 | 2.173e-08 |
| McNemar (exact) | accuracy (pooled OOF, seed 0) | ResNet101+InceptionV3 vs ResNet101 | 4752 | +0.0004 (b=117, c=115) | 0.9477 | 0.9477 |
| McNemar (exact) | accuracy (pooled OOF, seed 0) | MobileNetV3-L vs ResNet101 | 4752 | -0.0322 (b=109, c=262) | 1.173e-15 | 5.865e-15 |
| Wilcoxon signed-rank (optimism gap) | acc (raw) | ResNet101+InceptionV3, realwaste: naive_aug vs grouped | 3 | +0.0535 | 0.25 | 0.5 |
| Wilcoxon signed-rank (optimism gap) | acc (raw) | ResNet101, realwaste: naive_aug vs grouped | 3 | +0.0579 | 0.25 | 0.5 |
| Wilcoxon signed-rank (optimism gap) | acc (raw) | ResNet101+InceptionV3, realwaste: naive vs grouped | 3 | -0.0098 | 0.25 | 0.5 |
| Wilcoxon signed-rank (optimism gap) | acc (raw) | ResNet101, realwaste: naive vs grouped | 3 | -0.0062 | 0.25 | 0.5 |
| Wilcoxon signed-rank (optimism gap) | macro_f1 (raw) | ResNet101+InceptionV3, realwaste: naive_aug vs grouped | 3 | +0.0502 | 0.25 | 0.5 |
| Wilcoxon signed-rank (optimism gap) | macro_f1 (raw) | ResNet101, realwaste: naive_aug vs grouped | 3 | +0.0556 | 0.25 | 0.5 |
| Wilcoxon signed-rank (optimism gap) | macro_f1 (raw) | ResNet101+InceptionV3, realwaste: naive vs grouped | 3 | -0.0090 | 0.25 | 0.5 |
| Wilcoxon signed-rank (optimism gap) | macro_f1 (raw) | ResNet101, realwaste: naive vs grouped | 3 | -0.0077 | 0.25 | 0.5 |

## tab9_segmentation_ablation

Effect of segmentation pre-processing (approximation of Kumar et al. Alg. 1) on accuracy and calibration.

| Model | Input | Accuracy | Macro-F1 | ECE raw | ECE TS | AURC TS |
|---|---|---|---|---|---|---|
| ResNet101+InceptionV3 | raw image | 0.9364 ± 0.0114 | 0.9384 ± 0.0103 | 0.045 ± 0.011 | 0.021 ± 0.002 | 0.0090 ± 0.0026 |
| ResNet101+InceptionV3 | segmented | 0.8493 ± 0.0066 | 0.8513 ± 0.0055 | 0.102 ± 0.006 | 0.029 ± 0.009 | 0.0346 ± 0.0017 |
| ResNet101 | raw image | 0.9360 ± 0.0028 | 0.9380 ± 0.0016 | 0.044 ± 0.002 | 0.020 ± 0.007 | 0.0098 ± 0.0040 |
| ResNet101 | segmented | 0.8443 ± 0.0102 | 0.8427 ± 0.0101 | 0.107 ± 0.012 | 0.028 ± 0.009 | 0.0375 ± 0.0031 |

## tab10_cross_dataset

Cross-dataset sanity check: calibration and conformal coverage under distribution shift.

| Source → target | Model | n | In-domain acc | Shift acc (shared classes) | Shift acc (full label space) | ECE TS under shift | LAC cov. (target 0.90) | LAC size | APS cov. (target 0.90) | APS size |
|---|---|---|---|---|---|---|---|---|---|---|
| realwaste → trashbox | ConvNeXt-T | 12889 | 0.9396 ± 0.0068 | 0.6248 ± 0.0069 | 0.5911 ± 0.0075 | 0.109 ± 0.034 | 0.364 ± 0.044 | 0.46 ± 0.07 | 0.869 ± 0.022 | 3.51 ± 0.36 |
| realwaste → trashbox | DeiT-S | 12889 | 0.9434 ± 0.0087 | 0.5757 ± 0.0109 | 0.5319 ± 0.0106 | 0.162 ± 0.015 | 0.332 ± 0.023 | 0.48 ± 0.06 | 0.802 ± 0.022 | 3.36 ± 0.15 |
| realwaste → trashbox | EfficientNetV2-S | 12889 | 0.9133 ± 0.0059 | 0.4936 ± 0.0178 | 0.4587 ± 0.0146 | 0.256 ± 0.037 | 0.384 ± 0.032 | 0.74 ± 0.12 | 0.708 ± 0.021 | 2.93 ± 0.19 |
| realwaste → trashbox | ResNet101+InceptionV3 | 12889 | 0.9364 ± 0.0114 | 0.4585 ± 0.0197 | 0.4222 ± 0.0175 | 0.353 ± 0.033 | 0.350 ± 0.044 | 0.73 ± 0.16 | 0.605 ± 0.032 | 2.10 ± 0.14 |
| realwaste → trashbox | MobileNetV3-L | 12889 | 0.9038 ± 0.0104 | 0.3940 ± 0.0271 | 0.3685 ± 0.0251 | 0.335 ± 0.054 | 0.334 ± 0.028 | 0.84 ± 0.12 | 0.622 ± 0.049 | 2.84 ± 0.24 |
| realwaste → trashbox | ResNet101 | 12889 | 0.9360 ± 0.0028 | 0.4525 ± 0.0266 | 0.4204 ± 0.0236 | 0.351 ± 0.018 | 0.346 ± 0.024 | 0.72 ± 0.06 | 0.612 ± 0.027 | 2.17 ± 0.19 |
| realwaste → trashnet | ConvNeXt-T | 2390 | 0.9396 ± 0.0068 | 0.7069 ± 0.0162 | 0.7018 ± 0.0140 | 0.092 ± 0.023 | 0.504 ± 0.057 | 0.62 ± 0.08 | 0.888 ± 0.015 | 2.14 ± 0.25 |
| realwaste → trashnet | DeiT-S | 2390 | 0.9434 ± 0.0087 | 0.6126 ± 0.0171 | 0.6056 ± 0.0191 | 0.178 ± 0.029 | 0.422 ± 0.049 | 0.61 ± 0.10 | 0.813 ± 0.025 | 2.27 ± 0.11 |
| realwaste → trashnet | EfficientNetV2-S | 2390 | 0.9133 ± 0.0059 | 0.5569 ± 0.0326 | 0.5421 ± 0.0313 | 0.216 ± 0.047 | 0.467 ± 0.029 | 0.81 ± 0.10 | 0.781 ± 0.041 | 2.52 ± 0.19 |
| realwaste → trashnet | ResNet101+InceptionV3 | 2390 | 0.9364 ± 0.0114 | 0.5173 ± 0.0392 | 0.5099 ± 0.0385 | 0.273 ± 0.048 | 0.416 ± 0.062 | 0.75 ± 0.15 | 0.712 ± 0.059 | 1.95 ± 0.16 |
| realwaste → trashnet | MobileNetV3-L | 2390 | 0.9038 ± 0.0104 | 0.3908 ± 0.0344 | 0.3762 ± 0.0334 | 0.327 ± 0.058 | 0.336 ± 0.045 | 0.85 ± 0.13 | 0.676 ± 0.058 | 2.78 ± 0.24 |
| realwaste → trashnet | ResNet101 | 2390 | 0.9360 ± 0.0028 | 0.4998 ± 0.0191 | 0.4917 ± 0.0203 | 0.263 ± 0.024 | 0.388 ± 0.023 | 0.70 ± 0.07 | 0.742 ± 0.036 | 2.19 ± 0.16 |

## tab_s1_runtime

Measured training time.

| dataset | protocol | model | epochs | n_train | gpu | runs | min/run | total min |
|---|---|---|---|---|---|---|---|---|
| realwaste | grouped | convnext_tiny | 30 | 3039 | NVIDIA A100-SXM4-80GB | 2 | 2.0 | 4.0 |
| realwaste | grouped | convnext_tiny | 30 | 3039 | NVIDIA GeForce RTX 4070 | 3 | 2.85 | 8.54 |
| realwaste | grouped | convnext_tiny | 30 | 3041 | NVIDIA A100-SXM4-80GB | 8 | 1.99 | 15.95 |
| realwaste | grouped | convnext_tiny | 30 | 3041 | NVIDIA GeForce RTX 4070 | 12 | 2.78 | 33.35 |
| realwaste | grouped | deit_small | 30 | 3039 | NVIDIA A100-SXM4-80GB | 2 | 1.77 | 3.55 |
| realwaste | grouped | deit_small | 30 | 3039 | NVIDIA GeForce RTX 4070 | 3 | 1.89 | 5.67 |
| realwaste | grouped | deit_small | 30 | 3041 | NVIDIA A100-SXM4-80GB | 8 | 1.79 | 14.31 |
| realwaste | grouped | deit_small | 30 | 3041 | NVIDIA GeForce RTX 4070 | 12 | 1.88 | 22.54 |
| realwaste | grouped | efficientnetv2_s | 30 | 3039 | NVIDIA A100-SXM4-80GB | 2 | 3.29 | 6.59 |
| realwaste | grouped | efficientnetv2_s | 30 | 3039 | NVIDIA GeForce RTX 4070 | 3 | 15.62 | 46.86 |
| realwaste | grouped | efficientnetv2_s | 30 | 3041 | NVIDIA A100-SXM4-80GB | 8 | 3.26 | 26.06 |
| realwaste | grouped | efficientnetv2_s | 30 | 3041 | NVIDIA GeForce RTX 4070 | 12 | 15.55 | 186.55 |
| realwaste | grouped | hybrid_r101_incv3 | 30 | 3039 | NVIDIA A100-SXM4-80GB | 3 | 3.81 | 11.43 |
| realwaste | grouped | hybrid_r101_incv3 | 30 | 3039 | NVIDIA GeForce RTX 4070 | 3 | 34.73 | 104.18 |
| realwaste | grouped | hybrid_r101_incv3 | 30 | 3041 | NVIDIA A100-SXM4-80GB | 12 | 3.76 | 45.16 |
| realwaste | grouped | hybrid_r101_incv3 | 30 | 3041 | NVIDIA GeForce RTX 4070 | 12 | 34.55 | 414.59 |
| realwaste | grouped | mobilenetv3_large | 30 | 3039 | NVIDIA A100-SXM4-80GB | 2 | 2.03 | 4.05 |
| realwaste | grouped | mobilenetv3_large | 30 | 3039 | NVIDIA GeForce RTX 4070 | 3 | 5.61 | 16.84 |
| realwaste | grouped | mobilenetv3_large | 30 | 3041 | NVIDIA A100-SXM4-80GB | 8 | 1.92 | 15.36 |
| realwaste | grouped | mobilenetv3_large | 30 | 3041 | NVIDIA GeForce RTX 4070 | 12 | 5.58 | 66.93 |
| realwaste | grouped | resnet101 | 30 | 3039 | NVIDIA A100-SXM4-80GB | 3 | 2.4 | 7.2 |
| realwaste | grouped | resnet101 | 30 | 3039 | NVIDIA GeForce RTX 4070 | 3 | 20.72 | 62.16 |
| realwaste | grouped | resnet101 | 30 | 3041 | NVIDIA A100-SXM4-80GB | 12 | 2.37 | 28.45 |
| realwaste | grouped | resnet101 | 30 | 3041 | NVIDIA GeForce RTX 4070 | 12 | 20.55 | 246.62 |
| realwaste | naive | hybrid_r101_incv3 | 30 | 3326 | NVIDIA A100-SXM4-80GB | 3 | 3.9 | 11.71 |
| realwaste | naive | resnet101 | 30 | 3326 | NVIDIA A100-SXM4-80GB | 3 | 2.46 | 7.38 |
| realwaste | naive_aug | hybrid_r101_incv3 | 15 | 13305 | NVIDIA A100-SXM4-80GB | 1 | 9.68 | 9.68 |
| realwaste | naive_aug | hybrid_r101_incv3 | 15 | 13305 | NVIDIA GeForce RTX 4070 | 2 | 76.12 | 152.25 |
| realwaste | naive_aug | resnet101 | 15 | 13305 | NVIDIA A100-SXM4-80GB | 1 | 8.85 | 8.85 |
| realwaste | naive_aug | resnet101 | 15 | 13305 | NVIDIA GeForce RTX 4070 | 2 | 45.17 | 90.35 |
| trashbox | grouped | convnext_tiny | 12 | 11425 | NVIDIA A100-SXM4-40GB | 1 | 2.8 | 2.8 |
| trashbox | grouped | convnext_tiny | 12 | 11425 | NVIDIA A100-SXM4-80GB | 2 | 2.73 | 5.45 |
| trashbox | grouped | convnext_tiny | 12 | 11426 | NVIDIA A100-SXM4-80GB | 2 | 2.76 | 5.51 |
| trashbox | grouped | deit_small | 12 | 11425 | NVIDIA A100-SXM4-40GB | 1 | 2.44 | 2.44 |
| trashbox | grouped | deit_small | 12 | 11425 | NVIDIA A100-SXM4-80GB | 2 | 2.49 | 4.98 |
| trashbox | grouped | deit_small | 12 | 11426 | NVIDIA A100-SXM4-80GB | 2 | 2.52 | 5.05 |
| trashbox | grouped | efficientnetv2_s | 12 | 11425 | NVIDIA A100-SXM4-40GB | 1 | 4.31 | 4.31 |
| trashbox | grouped | efficientnetv2_s | 12 | 11425 | NVIDIA A100-SXM4-80GB | 2 | 4.31 | 8.62 |
| trashbox | grouped | efficientnetv2_s | 12 | 11426 | NVIDIA A100-SXM4-80GB | 2 | 4.33 | 8.66 |
| trashbox | grouped | hybrid_r101_incv3 | 12 | 11425 | NVIDIA A100-SXM4-80GB | 3 | 5.01 | 15.03 |
| trashbox | grouped | hybrid_r101_incv3 | 12 | 11426 | NVIDIA A100-SXM4-40GB | 1 | 5.07 | 5.07 |
| trashbox | grouped | hybrid_r101_incv3 | 12 | 11426 | NVIDIA A100-SXM4-80GB | 1 | 4.96 | 4.96 |
| trashbox | grouped | mobilenetv3_large | 12 | 11425 | NVIDIA A100-SXM4-40GB | 1 | 2.86 | 2.86 |
| trashbox | grouped | mobilenetv3_large | 12 | 11425 | NVIDIA A100-SXM4-80GB | 2 | 2.75 | 5.49 |
| trashbox | grouped | mobilenetv3_large | 12 | 11426 | NVIDIA A100-SXM4-80GB | 2 | 2.77 | 5.54 |
| trashbox | grouped | resnet101 | 12 | 11425 | NVIDIA A100-SXM4-40GB | 1 | 3.14 | 3.14 |
| trashbox | grouped | resnet101 | 12 | 11425 | NVIDIA A100-SXM4-80GB | 2 | 3.16 | 6.32 |
| trashbox | grouped | resnet101 | 12 | 11426 | NVIDIA A100-SXM4-40GB | 1 | 3.16 | 3.16 |
| trashbox | grouped | resnet101 | 12 | 11426 | NVIDIA A100-SXM4-80GB | 1 | 3.14 | 3.14 |
| trashnet | grouped | convnext_tiny | 30 | 1616 | NVIDIA A100-SXM4-40GB | 2 | 1.22 | 2.44 |
| trashnet | grouped | convnext_tiny | 30 | 1616 | NVIDIA A100-SXM4-80GB | 1 | 1.32 | 1.32 |
| trashnet | grouped | convnext_tiny | 30 | 1617 | NVIDIA A100-SXM4-40GB | 9 | 1.13 | 10.18 |
| trashnet | grouped | convnext_tiny | 30 | 1617 | NVIDIA A100-SXM4-80GB | 3 | 1.19 | 3.56 |
| trashnet | grouped | deit_small | 30 | 1616 | NVIDIA A100-SXM4-80GB | 1 | 1.01 | 1.01 |
| trashnet | grouped | deit_small | 30 | 1617 | NVIDIA A100-SXM4-40GB | 1 | 0.97 | 0.97 |
| trashnet | grouped | deit_small | 30 | 1617 | NVIDIA A100-SXM4-80GB | 3 | 1.01 | 3.03 |
| trashnet | grouped | efficientnetv2_s | 30 | 1616 | NVIDIA A100-SXM4-80GB | 1 | 2.01 | 2.01 |
| trashnet | grouped | efficientnetv2_s | 30 | 1617 | NVIDIA A100-SXM4-40GB | 1 | 1.92 | 1.92 |
| trashnet | grouped | efficientnetv2_s | 30 | 1617 | NVIDIA A100-SXM4-80GB | 3 | 1.83 | 5.49 |
| trashnet | grouped | hybrid_r101_incv3 | 30 | 1616 | NVIDIA A100-SXM4-80GB | 1 | 2.06 | 2.06 |
| trashnet | grouped | hybrid_r101_incv3 | 30 | 1617 | NVIDIA A100-SXM4-40GB | 1 | 2.16 | 2.16 |
| trashnet | grouped | hybrid_r101_incv3 | 30 | 1617 | NVIDIA A100-SXM4-80GB | 3 | 2.13 | 6.4 |
| trashnet | grouped | mobilenetv3_large | 30 | 1616 | NVIDIA A100-SXM4-80GB | 1 | 1.44 | 1.44 |
| trashnet | grouped | mobilenetv3_large | 30 | 1617 | NVIDIA A100-SXM4-40GB | 1 | 1.37 | 1.37 |
| trashnet | grouped | mobilenetv3_large | 30 | 1617 | NVIDIA A100-SXM4-80GB | 3 | 1.16 | 3.49 |
| trashnet | grouped | resnet101 | 30 | 1616 | NVIDIA A100-SXM4-40GB | 2 | 1.37 | 2.74 |
| trashnet | grouped | resnet101 | 30 | 1616 | NVIDIA A100-SXM4-80GB | 1 | 1.31 | 1.31 |
| trashnet | grouped | resnet101 | 30 | 1617 | NVIDIA A100-SXM4-40GB | 9 | 1.29 | 11.63 |
| trashnet | grouped | resnet101 | 30 | 1617 | NVIDIA A100-SXM4-80GB | 3 | 1.28 | 3.85 |
