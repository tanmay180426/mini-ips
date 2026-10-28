# Mini-IDS: Machine Learning & Feature Engineering Pipeline

## 1. Feature Engineering & Bidirectional Flow Aggregation

Mini-IDS bridges raw packet captures (`NormalizedPacket`) with flow-based tabular machine learning classifiers using a stateful flow aggregator (`ml/feature_extraction.py`).

### 1.1 Canonical 5-Tuple Key
To ensure bidirectional traffic (requests and responses) is unified into a single flow record regardless of direction:
```python
canonical_key = (
    min(src_ip, dst_ip),
    max(src_ip, dst_ip),
    min(src_port, dst_port),
    max(src_port, dst_port),
    protocol.upper()
)
```

### 1.2 Extracted Tabular Features
The aggregator tracks packet length distributions, inter-arrival times (IAT), and TCP control flags, outputting 28 standardized tabular features:

| Feature Name | Description |
| :--- | :--- |
| `flow_duration` | Total duration of flow in seconds |
| `total_fwd_packets` | Packets transmitted in forward direction |
| `total_bwd_packets` | Packets transmitted in backward direction |
| `total_length_of_fwd_packets` | Cumulative forward payload bytes |
| `total_length_of_bwd_packets` | Cumulative backward payload bytes |
| `fwd_packet_length_max` / `min` / `mean` / `std` | Statistical moments of forward packet sizes |
| `bwd_packet_length_max` / `min` / `mean` / `std` | Statistical moments of backward packet sizes |
| `flow_bytes_s` | Bandwidth throughput (bytes/second) |
| `flow_packets_s` | Packet rate (packets/second) |
| `flow_iat_mean` / `std` / `max` / `min` | Inter-arrival time statistics across entire flow |
| `fwd_iat_mean` / `std` / `max` / `min` | Forward direction inter-arrival times |
| `bwd_iat_mean` / `std` / `max` / `min` | Backward direction inter-arrival times |
| `fin_flag_count`, `syn_flag_count`, `rst_flag_count` | TCP control flag counters |
| `psh_flag_count`, `ack_flag_count` | TCP push and acknowledgement counters |
| `down_up_ratio` | Ratio of backward packets to forward packets |
| `average_packet_size` | Global mean packet size across the flow |

---

## 2. Preprocessing & Leakage Prevention (`ml/preprocessing.py`)

To ensure scientific rigor and zero data leakage:
1. **Leakage Exclusion:** Columns that could cause artificial memorization (e.g. `timestamp`, `flow_id`, raw IP addresses) are explicitly removed from feature matrices.
2. **Infinite Value Handling:** Flow rates (`flow_bytes_s`, `flow_packets_s`) can yield mathematical infinities on zero-duration connections. Infinities are converted to `NaN` and imputed.
3. **Transformers Fitted on Train Only:** `SimpleImputer` and `StandardScaler` are fitted strictly on the training partition (`X_train`) and applied downstream to validation (`X_val`) and test (`X_test`) partitions.
4. **Partition Strategies:**
   - **Stratified Split:** Preserves rare attack class representations across train, validation, and test splits.
   - **Temporal Split:** Partitions records chronologically to evaluate generalization to future traffic.

---

## 3. Supervised Classification (`ml/train_supervised.py`)

Supports multiple scikit-learn algorithms for binary (BENIGN vs ATTACK) and multiclass classification:
- **Random Forest:** Ensemble of decision trees with balanced class weighting, robust to unscaled outliers and non-linear boundaries.
- **Logistic Regression:** Regularized linear model optimizing cross-entropy loss, providing low-latency inference.
- **HistGradientBoosting:** High-speed histogram-based gradient boosted trees for large tabular datasets.

---

## 4. Unsupervised Anomaly Detection (`ml/train_anomaly.py`)

Detects zero-day threats and behavioral shifts without ground-truth attack labels:
- **Algorithm:** Isolation Forest (`sklearn.ensemble.IsolationForest`).
- **Baseline Training:** Fitted exclusively on benign network traffic records.
- **Normalized Anomaly Scoring:** Uses a calibrated sigmoid function mapping raw decision margins to `[0.0, 1.0]`, where values close to `1.0` indicate extreme statistical outliers.

---

## 5. Model Registry & Verification (`ml/model_registry.py`)

Models are serialized into versioned artifact bundles in `data/models/<version>/`:
- `model.joblib`: Serialized estimator binary.
- `preprocessor.joblib`: Fitted scaler, imputer, and feature schema.
- `metadata.json`: Dataset provenance, hyperparameters, evaluation metrics, and schema.

Before any model is set to active status, `registry.validate_artifact()` conducts dry-run validation with dummy inputs to verify schema alignment and prevent runtime crashes.
