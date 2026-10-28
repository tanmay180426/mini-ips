# Model Evaluation Report: logistic_regression (`logistic_regression_multiclass_1792945215`)

- **Dataset:** `cicids2017`
- **Task Type:** `multiclass`
- **Split Methodology:** `stratified`
- **Test Samples Evaluated:** 241
- **Inference Latency:** 0.0021 ms/sample

## Overall Performance Summary

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 99.17% |
| **Precision (Weighted)** | 99.29% |
| **Recall (Weighted)** | 99.17% |
| **F1 Score (Weighted)** | 99.20% |
| **F1 Score (Macro)** | 97.90% |
| **ROC-AUC Score** | 0.9999 |

## Per-Class Classification Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | 100.00% | 98.62% | 99.31% | 145 |
| `BruteForce` | 85.71% | 100.00% | 92.31% | 12 |
| `DoS/DDoS` | 100.00% | 100.00% | 100.00% | 36 |
| `PortScan` | 100.00% | 100.00% | 100.00% | 48 |

## Confusion Matrix

```
Classes: ['BENIGN', 'BruteForce', 'DoS/DDoS', 'PortScan']
[[143   2   0   0]
 [  0  12   0   0]
 [  0   0  36   0]
 [  0   0   0  48]]
```
