# Mini-IDS: Model Evaluation & Performance Benchmark Report

## 1. Evaluation Methodology

All metrics reported in this document are derived from real, reproducible evaluation runs executed on held-out test splits (`data/datasets/cicids2017/processed/test.csv`).

- **Test Dataset:** CICIDS2017 (241 held-out test flow records).
- **Partitioning:** Stratified 70/10/20 train/val/test split.
- **Evaluation Pipeline:** `ml/evaluation.py` measuring accuracy, precision, recall, F1 scores (macro and weighted), false positive rates (FPR), false negative rates (FNR), ROC-AUC, and inference latency per sample.

---

## 2. Supervised & Anomaly Detection Performance Summary

| Model Version | Algorithm | Task Type | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | ROC-AUC | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_forest_binary_1792500911` | Random Forest | Binary | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 0.0681 ms/sample |
| `logistic_regression_binary_1792500911` | Logistic Regression | Binary | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.0012 ms/sample** |
| `random_forest_multiclass_1792500921` | Random Forest | Multiclass | **99.59%** | **99.62%** | **99.59%** | **99.59%** | **1.0000** | 0.0636 ms/sample |
| `isolation_forest_binary_1792500912` | Isolation Forest | Anomaly | *Unsupervised* | *N/A* | *N/A* | *N/A* | *N/A* | 0.0205 ms/sample |

---

## 3. Detailed Multiclass Evaluation Breakdown

Model: `random_forest_multiclass_1792500921`

### 3.1 Per-Class Metrics
| Attack Family | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | 100.00% | 99.31% | 99.65% | 145 |
| `BruteForce` | 92.31% | 100.00% | 96.00% | 12 |
| `DoS/DDoS` | 100.00% | 100.00% | 100.00% | 36 |
| `PortScan` | 100.00% | 100.00% | 100.00% | 48 |

### 3.2 Multiclass Confusion Matrix
```
Classes: ['BENIGN', 'BruteForce', 'DoS/DDoS', 'PortScan']
[[144    1    0    0]    <- BENIGN (1 false positive categorized as BruteForce)
 [  0   12    0    0]    <- BruteForce (100% recall)
 [  0    0   36    0]    <- DoS/DDoS (100% recall)
 [  0    0    0   48]]   <- PortScan (100% recall)
```

---

## 4. Binary Classification Breakdown

Model: `random_forest_binary_1792500911`

### 4.1 Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `BENIGN` (0) | 100.00% | 100.00% | 100.00% | 145 |
| `ATTACK` (1) | 100.00% | 100.00% | 100.00% | 96 |

### 4.2 Binary Confusion Matrix
```
Classes: ['BENIGN', 'ATTACK']
[[145    0]    <- 0 False Positives (FPR: 0.00%)
 [  0   96]]   <- 0 False Negatives (FNR: 0.00%)
```

---

## 5. Unsupervised Anomaly Detection Analysis

Model: `isolation_forest_binary_1792500912`
- **Methodology:** Trained exclusively on benign baseline traffic with a configured contamination parameter of `0.05`.
- **Inference Behavior:** Calculates empirical decision distances. Attack flows featuring extreme volumetric surges (e.g. DoS floods) and abnormal connection characteristics produce high normalized anomaly scores (`> 0.85`), triggering `ML_ANOMALY_OUTLIER` alerts.

---

## 6. Model Explainability & Triage Guidance

Mini-IDS pairs detection alerts with human-interpretable forensic explanations (`ml/explainability.py`):
1. **Global Feature Importance:** Tree-based models expose top decision drivers:
   - `destination_port` (service-targeted reconnaissance)
   - `flow_packets_s` & `flow_bytes_s` (volumetric denial of service)
   - `syn_flag_count` & `fwd_iat_mean` (connection probing and automation)
2. **Defensive Playbook Actions:** Automatically recommends operational containment steps tailored to the detected attack class (e.g., firewall drop rules, SYN cookies activation, MFA enforcement).
