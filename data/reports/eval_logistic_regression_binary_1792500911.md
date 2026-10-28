# Model Evaluation Report: logistic_regression (`logistic_regression_binary_1792500911`)

- **Dataset:** `cicids2017`
- **Task Type:** `binary`
- **Split Methodology:** `stratified`
- **Test Samples Evaluated:** 241
- **Inference Latency:** 0.0012 ms/sample

## Overall Performance Summary

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 100.00% |
| **Precision (Weighted)** | 100.00% |
| **Recall (Weighted)** | 100.00% |
| **F1 Score (Weighted)** | 100.00% |
| **F1 Score (Macro)** | 100.00% |
| **False Positive Rate (FPR)** | 0.00% |
| **False Negative Rate (FNR)** | 0.00% |
| **ROC-AUC Score** | 1.0000 |

## Per-Class Classification Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | 100.00% | 100.00% | 100.00% | 145 |
| `ATTACK` | 100.00% | 100.00% | 100.00% | 96 |

## Confusion Matrix

```
Classes: ['BENIGN', 'ATTACK']
[[145   0]
 [  0  96]]
```
