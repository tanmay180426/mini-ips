# Model Evaluation Report: random_forest (`random_forest_multiclass_1792945214`)

- **Dataset:** `cicids2017`
- **Task Type:** `multiclass`
- **Split Methodology:** `stratified`
- **Test Samples Evaluated:** 241
- **Inference Latency:** 0.1787 ms/sample

## Overall Performance Summary

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 99.59% |
| **Precision (Weighted)** | 99.62% |
| **Recall (Weighted)** | 99.59% |
| **F1 Score (Weighted)** | 99.59% |
| **F1 Score (Macro)** | 98.91% |
| **ROC-AUC Score** | 1.0000 |

## Per-Class Classification Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | 100.00% | 99.31% | 99.65% | 145 |
| `BruteForce` | 92.31% | 100.00% | 96.00% | 12 |
| `DoS/DDoS` | 100.00% | 100.00% | 100.00% | 36 |
| `PortScan` | 100.00% | 100.00% | 100.00% | 48 |

## Confusion Matrix

```
Classes: ['BENIGN', 'BruteForce', 'DoS/DDoS', 'PortScan']
[[144   1   0   0]
 [  0  12   0   0]
 [  0   0  36   0]
 [  0   0   0  48]]
```
