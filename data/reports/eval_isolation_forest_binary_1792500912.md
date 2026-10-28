# Model Evaluation Report: IsolationForest_CICIDS2017 (`isolation_forest_binary_1792500912`)

- **Dataset:** `cicids2017`
- **Task Type:** `anomaly`
- **Split Methodology:** `stratified`
- **Test Samples Evaluated:** 241
- **Inference Latency:** 0.0205 ms/sample

## Overall Performance Summary

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 5.39% |
| **Precision (Weighted)** | 3.70% |
| **Recall (Weighted)** | 5.39% |
| **F1 Score (Weighted)** | 4.39% |
| **F1 Score (Macro)** | 3.67% |

## Per-Class Classification Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `Class_-1` | 0.00% | 0.00% | 0.00% | 0 |
| `Class_0` | 0.00% | 0.00% | 0.00% | 145 |
| `Class_1` | 9.29% | 13.54% | 11.02% | 96 |

## Confusion Matrix

```
Classes: ['Class_-1', 'Class_0', 'Class_1']
[[  0   0   0]
 [ 18   0 127]
 [ 83   0  13]]
```
