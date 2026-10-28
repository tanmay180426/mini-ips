# Mini-IDS Multi-Dataset AI Training & Verification Report

*Generated: 2026-10-23T17:14:00.158508+00:00*

## 1. Executive Summary
Mini-IDS was evaluated across three public intrusion detection benchmarks:
1. **CIC-IDS2017** (Canadian Institute for Cybersecurity)
2. **UNSW-NB15** (UNSW Canberra Cyber)
3. **CSE-CIC-IDS2018** (AWS Open Data / CSE & UNB)

Both **Dataset-Specific Models (Experiment A)** and a **Unified Semantic Core Model (Experiment B)** were trained and evaluated using zero-leakage preprocessor pipelines.

## 2. Acquired Datasets & Provenance
| Dataset | Raw Records | Features | License | Status |
| :--- | :--- | :--- | :--- | :--- |
| **CIC-IDS2017** | 512,212 | 79 | Academic and Research Use (University of New Brunswick) | `VERIFIED` |
| **UNSW-NB15** | 82,332 | 45 | Academic and Research Use (UNSW Canberra Cyber) | `VERIFIED` |
| **CSE-CIC-IDS2018** | 40,000 | 80 | Open Data (University of New Brunswick / AWS Open Data) | `VERIFIED` |

## 3. Data Leakage Audit
To ensure trustworthy scientific evaluation, all identifiers that could allow capture memorization were stripped:
- **IP Addresses:** `src_ip`, `dst_ip`, `srcip`, `dstip` (Dropped)
- **Timestamps:** `Timestamp`, `date`, `time` (Dropped)
- **Flow & Session IDs:** `id`, `flow_id` (Dropped)
- **Leakage-Free Splitting:** Stratified 70/15/15 train/val/test split executed *before* fitting any imputer or scaler.
- **Zero Preprocessor Leakage:** `SimpleImputer` and `RobustScaler` fit strictly on `X_train`.

## 4. Model Benchmark Results (Held-Out Test Splits)

### Benchmark: CICIDS2017

#### Task: BINARY (N=2250)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_cicids2017_binary_17927` | 0.9991 | 0.9993 | 0.9990 | 0.0000 | 48.2ms |
| `hist_gradient_boosting_cicids2017_b` | 0.9987 | 0.9983 | 0.9985 | 0.0027 | 18.5ms |
| `logistic_regression_cicids2017_bina` | 0.9938 | 0.9937 | 0.9930 | 0.0067 | 0.3ms |
| `random_forest_cicids2017_binary_179` | 0.9991 | 0.9990 | 0.9990 | 0.0013 | 41.4ms |
| `isolation_forest_cicids2017_1792775` | 0.3587 | 0.4913 | 0.3214 | 0.1107 | 23.3ms |
| `isolation_forest_cicids2017_1792775` | 0.3556 | 0.4893 | 0.3168 | 0.1093 | 23.6ms |
| `isolation_forest_anomaly_1792645935` | 0.3578 | 0.4990 | 0.3122 | 0.0773 | 24.1ms |
| `isolation_forest_anomaly_1792645962` | 0.3142 | 0.4713 | 0.2391 | 0.0573 | 22.8ms |
| `logistic_regression_binary_17926459` | 0.6360 | 0.7027 | 0.6356 | 0.0973 | 0.3ms |
| `random_forest_binary_1792645962` | 0.6618 | 0.7400 | 0.6617 | 0.0253 | 44.3ms |

#### Task: MULTICLASS (N=2250)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_cicids2017_multiclass_1` | 0.9987 | 0.9987 | 0.9987 | 0.0000 | 48.6ms |
| `hist_gradient_boosting_cicids2017_m` | 0.9978 | 0.9978 | 0.9978 | 0.0040 | 33.2ms |
| `logistic_regression_cicids2017_mult` | 0.9969 | 0.9969 | 0.9969 | 0.0053 | 0.4ms |
| `random_forest_cicids2017_multiclass` | 0.9987 | 0.9987 | 0.9987 | 0.0000 | 45.7ms |
| `isolation_forest_cicids2017_1792775` | 0.3636 | 0.4970 | 0.3265 | 0.1027 | 22.1ms |
| `isolation_forest_cicids2017_1792775` | 0.3569 | 0.4907 | 0.3183 | 0.1080 | 25.6ms |
| `isolation_forest_anomaly_1792645935` | 0.3689 | 0.5137 | 0.3219 | 0.0520 | 26.1ms |
| `isolation_forest_anomaly_1792645962` | 0.3147 | 0.4720 | 0.2394 | 0.0560 | 23.1ms |
| `logistic_regression_multiclass_1792` | 0.9960 | 0.9960 | 0.9960 | 0.0053 | 0.2ms |
| `random_forest_multiclass_1792645935` | 0.9987 | 0.9987 | 0.9987 | 0.0000 | 47.1ms |

### Benchmark: UNSW_NB15

#### Task: BINARY (N=4500)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_unsw_nb15_binary_179277` | 0.9244 | 0.9249 | 0.9239 | 0.0705 | 28.2ms |
| `hist_gradient_boosting_unsw_nb15_bi` | 0.9684 | 0.9691 | 0.9682 | 0.0240 | 16.1ms |
| `logistic_regression_unsw_nb15_binar` | 0.7558 | 0.7660 | 0.7554 | 0.1234 | 0.3ms |
| `random_forest_unsw_nb15_binary_1792` | 0.9667 | 0.9674 | 0.9664 | 0.0245 | 29.5ms |
| `isolation_forest_unsw_nb15_17927754` | 0.5678 | 0.5966 | 0.5373 | 0.0916 | 21.0ms |
| `isolation_forest_unsw_nb15_17927754` | 0.5678 | 0.5966 | 0.5373 | 0.0916 | 14.4ms |

#### Task: MULTICLASS (N=4500)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_unsw_nb15_multiclass_17` | 0.9244 | 0.9249 | 0.9239 | 0.0705 | 31.8ms |
| `hist_gradient_boosting_unsw_nb15_mu` | 0.9684 | 0.9691 | 0.9682 | 0.0240 | 25.3ms |
| `logistic_regression_unsw_nb15_multi` | 0.7558 | 0.7660 | 0.7554 | 0.1234 | 0.2ms |
| `random_forest_unsw_nb15_multiclass_` | 0.9667 | 0.9674 | 0.9664 | 0.0245 | 33.8ms |
| `isolation_forest_unsw_nb15_17927754` | 0.5678 | 0.5966 | 0.5373 | 0.0916 | 20.6ms |
| `isolation_forest_unsw_nb15_17927754` | 0.5678 | 0.5966 | 0.5373 | 0.0916 | 19.3ms |

### Benchmark: CSECICIDS2018

#### Task: BINARY (N=4500)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_csecicids2018_binary_17` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 24.1ms |
| `hist_gradient_boosting_csecicids201` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 7.3ms |
| `logistic_regression_csecicids2018_b` | 0.6744 | 0.8368 | 0.4105 | 0.0000 | 0.2ms |
| `random_forest_csecicids2018_binary_` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 21.7ms |
| `isolation_forest_csecicids2018_1792` | 0.0022 | 0.4167 | 0.0022 | 0.1667 | 14.9ms |
| `isolation_forest_csecicids2018_1792` | 0.0022 | 0.4167 | 0.0022 | 0.1667 | 14.6ms |

#### Task: MULTICLASS (N=4500)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_csecicids2018_multiclas` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 23.9ms |
| `hist_gradient_boosting_csecicids201` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 6.8ms |
| `logistic_regression_csecicids2018_m` | 0.6744 | 0.8368 | 0.4105 | 0.0000 | 0.3ms |
| `random_forest_csecicids2018_multicl` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 26.6ms |
| `isolation_forest_csecicids2018_1792` | 0.0022 | 0.4167 | 0.0022 | 0.1667 | 15.1ms |
| `isolation_forest_csecicids2018_1792` | 0.0022 | 0.4167 | 0.0022 | 0.1667 | 14.7ms |

### Benchmark: UNIFIED

#### Task: BINARY (N=6750)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_unified_binary_17927754` | 0.9379 | 0.9519 | 0.9243 | 0.0187 | 24.5ms |
| `hist_gradient_boosting_unified_bina` | 0.9714 | 0.9671 | 0.9633 | 0.0419 | 12.8ms |
| `logistic_regression_unified_binary_` | 0.7243 | 0.7971 | 0.7092 | 0.0504 | 0.2ms |
| `logistic_regression_unified_binary_` | 0.7243 | 0.7971 | 0.7092 | 0.0504 | 0.2ms |
| `random_forest_unified_binary_179277` | 0.9664 | 0.9728 | 0.9578 | 0.0136 | 20.0ms |
| `random_forest_unified_binary_179277` | 0.9664 | 0.9728 | 0.9578 | 0.0136 | 19.5ms |
| `isolation_forest_unified_1792775369` | 0.3753 | 0.5439 | 0.3697 | 0.1025 | 14.1ms |
| `isolation_forest_unified_1792775452` | 0.3753 | 0.5439 | 0.3697 | 0.1025 | 13.2ms |
| `isolation_forest_unified_1792775462` | 0.2662 | 0.4703 | 0.2346 | 0.1020 | 12.3ms |

#### Task: MULTICLASS (N=6750)
| Model Algorithm | Accuracy | Balanced Acc | Macro F1 | False Alarm Rate (FPR) | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extra_trees_unified_multiclass_1792` | 0.9600 | 0.9634 | 0.9610 | 0.0873 | 15.7ms |
| `hist_gradient_boosting_unified_mult` | 0.9726 | 0.9726 | 0.9740 | 0.0346 | 42.0ms |
| `logistic_regression_unified_multicl` | 0.8311 | 0.8486 | 0.8331 | 0.3875 | 0.3ms |
| `random_forest_unified_multiclass_17` | 0.9721 | 0.9743 | 0.9737 | 0.0555 | 21.6ms |
| `isolation_forest_unified_1792775369` | 0.3689 | 0.5367 | 0.3631 | 0.1116 | 12.7ms |
| `isolation_forest_unified_1792775452` | 0.3689 | 0.5367 | 0.3631 | 0.1116 | 12.7ms |
| `isolation_forest_unified_1792775462` | 0.2646 | 0.4692 | 0.2323 | 0.1020 | 13.6ms |

## 5. Cross-Dataset Generalization & Transferability
Models evaluated across disparate network topographies using the 11-feature common semantic core:

| Experiment | Train Sources | Test Source | Accuracy | Macro F1 | Transfer Gap (Delta) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| In-Dataset Baseline: CICIDS2017 | `cicids2017` | `cicids2017` | 0.9973 | 0.9970 | `+0.0000` |
| In-Dataset Baseline: UNSW_NB15 | `unsw_nb15` | `unsw_nb15` | 0.9149 | 0.9147 | `+0.0000` |
| In-Dataset Baseline: CSECICIDS2018 | `csecicids2018` | `csecicids2018` | 1.0000 | 1.0000 | `+0.0000` |
| Cross-Dataset Transfer: CIC-IDS2017 -> UNSW-NB15 | `cicids2017` | `unsw_nb15` | 0.4491 | 0.3099 | `+0.6048` |
| Cross-Dataset Transfer: CIC-IDS2017 -> CSE-CIC-IDS2018 | `cicids2017` | `csecicids2018` | 0.0021 | 0.0021 | `+0.9979` |
| Multi-Source Training: (CIC17 + CIC18) -> UNSW-NB15 | `cicids2017+csecicids2018` | `unsw_nb15` | 0.4491 | 0.3099 | `+0.6048` |
| LODO: Train (unsw_nb15 + csecicids2018) -> Test on CICIDS2017 | `unsw_nb15+csecicids2018` | `cicids2017` | 0.6624 | 0.5894 | `+0.4076` |
| LODO: Train (cicids2017 + csecicids2018) -> Test on UNSW_NB15 | `cicids2017+csecicids2018` | `unsw_nb15` | 0.4491 | 0.3099 | `+0.6048` |
| LODO: Train (cicids2017 + unsw_nb15) -> Test on CSECICIDS2018 | `cicids2017+unsw_nb15` | `csecicids2018` | 0.0019 | 0.0019 | `+0.9981` |

## 6. Selected Best Production Models
- **CICIDS2017 (BINARY):** `extra_trees_cicids2017_binary_1792775392` — Balanced Acc: **0.9993**, Macro F1: **0.9990**, FAR: **0.0000**
- **CICIDS2017 (MULTICLASS):** `extra_trees_cicids2017_multiclass_1792775400` — Balanced Acc: **0.9987**, Macro F1: **0.9987**, FAR: **0.0000**
- **UNSW_NB15 (BINARY):** `hist_gradient_boosting_unsw_nb15_binary_1792775414` — Balanced Acc: **0.9691**, Macro F1: **0.9682**, FAR: **0.0240**
- **UNSW_NB15 (MULTICLASS):** `hist_gradient_boosting_unsw_nb15_multiclass_1792775427` — Balanced Acc: **0.9691**, Macro F1: **0.9682**, FAR: **0.0240**
- **CSECICIDS2018 (BINARY):** `extra_trees_csecicids2018_binary_1792775440` — Balanced Acc: **1.0000**, Macro F1: **1.0000**, FAR: **0.0000**
- **CSECICIDS2018 (MULTICLASS):** `extra_trees_csecicids2018_multiclass_1792775446` — Balanced Acc: **1.0000**, Macro F1: **1.0000**, FAR: **0.0000**
- **UNIFIED (BINARY):** `random_forest_unified_binary_1792775369` — Balanced Acc: **0.9728**, Macro F1: **0.9578**, FAR: **0.0136**
- **UNIFIED (MULTICLASS):** `random_forest_unified_multiclass_1792775462` — Balanced Acc: **0.9743**, Macro F1: **0.9737**, FAR: **0.0555**