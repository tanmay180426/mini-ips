# Mini-IPS: Comprehensive Viva & Defense Preparation Guide
## Advanced ML-Assisted Mini Intrusion Prevention System (Simulation-Only)

This guide provides precise, academically rigorous, and defense-ready answers to key viva voce questions for final-year engineering evaluations.

---

### 1. Difference Between IDS and IPS

| Dimension | Intrusion Detection System (IDS) | Intrusion Prevention System (IPS) |
| :--- | :--- | :--- |
| **Primary Goal** | Passive visibility, event detection, alerting, and logging. | Active mitigation, threat containment, inline/automated response. |
| **Network Position** | Out-of-band (promiscuous tap, SPAN/mirror port). | Inline (in traffic path) or control-plane coupled (host firewall/SDN). |
| **Response Mechanism** | Logs alert to database/SIEM, sends notifications. | Drops packets, blocks source IP, resets TCP sessions, or rate-limits. |
| **Latency Sensitivity** | Asynchronous / Near real-time (can tolerate minor queuing). | Synchronous / Millisecond-level (inline latency directly affects user traffic). |
| **Risk of False Positives**| Analyst alert fatigue, noisy logs. | **Denial of Service (Self-Dos):** Legitimate users or servers are cut off. |
| **Mini-IPS Implementation** | Stateful heuristic rules + ML inference + SQLite persistence. | Response Policy Engine, Allowlist, Simulation Adapter, Expiration Daemon. |

---

### 2. Why Mini-IPS Uses Simulation-Only Prevention

1. **Student & Host Safety:** Preventing accidental self-lockout or disabling host network connectivity on student/evaluator laptops during demonstrations.
2. **Privilege Decoupling:** Modifying OS kernel filtering tables (Windows Filtering Platform / `netsh advfirewall` or Linux `iptables`/`nftables`) requires Administrator / `root` privileges. A simulation-only engine runs safely in non-elevated user space.
3. **Deterministic Evaluation:** A simulation adapter records state in-memory and in SQLite tables (`prevention_actions`, `audit_log`), allowing instant verification, inspection, and rollback without altering system state.
4. **Backend Enforced Guard:** In `prevention/windows_firewall_adapter.py` and `prevention/linux_firewall_adapter.py`, `SIMULATION_LOCKED = True` guarantees that even if a UI control or configuration accidentally requests real enforcement, execution is blocked with `RuntimeError` and diverted to simulation.

---

### 3. Rule-Based vs. Supervised Machine Learning Detection

* **Rule-Based Detection (Heuristic):**
  - *Mechanism:* Evaluates explicit deterministic thresholds over rolling time windows (e.g., >10 distinct target ports within 30s for `PORT_SCAN`, >20 SYN packets/sec for `SYN_RATE`).
  - *Strengths:* Zero false positives for known signatures; deterministic; explainable; negligible computational cost.
  - *Weaknesses:* Cannot detect zero-day attacks, stealthy low-and-slow port scans, or distributed mutations.
* **Supervised Machine Learning (Random Forest / Logistic Regression):**
  - *Mechanism:* Learns non-linear boundary hyperplanes across 28 statistical flow features extracted from network traffic.
  - *Strengths:* Generalizes to subtle variants of known attack families (DDoS surges, scanning patterns); robust against simple obfuscation.
  - *Weaknesses:* Vulnerable to concept drift (network topology changes); requires large, balanced, labeled training datasets; can produce opaque decisions without explainability modules.

---

### 4. Anomaly Detection (Isolation Forest) and Its Limitations

* **Mechanism:** Isolation Forest recursively partitions feature space using random orthogonal hyperplanes. Anomalous data points (low-density regions) require fewer partitions to isolate than normal cluster points, yielding shorter tree path lengths.
* **Role in Mini-IPS:** Acts as a safety net for novel/unseen traffic anomalies that neither matched heuristic rules nor the supervised attack classes.
* **Key Limitations:**
  - *Non-specific:* An anomaly flag indicates "statistical deviation," not necessarily malicious intent (e.g., sudden backup transfers, software updates).
  - *Uncalibrated Score:* Raw decision function scores cannot be directly interpreted as attack probabilities without empirical percentile calibration.
  - *High False Positive Potential:* Network traffic has high variance; relying solely on anomaly detection for prevention causes unacceptable user disruption.

---

### 5. Flow Feature Extraction & 28 Canonical Features

A single IP packet lacks contextual state. Mini-IPS aggregates packets into **Bidirectional Flows** identified by a 5-tuple: `(Source IP, Destination IP, Source Port, Destination Port, Transport Protocol)`:
1. **Forward (fwd):** Packets originating from the flow initiator.
2. **Backward (bwd):** Packets sent in response from the responder.

**The 28 Canonical Features Computed:**
- **Duration & Timing:** `flow_duration`, `flow_iat_mean`, `flow_iat_std`, `flow_iat_min`, `flow_iat_max`, `fwd_iat_total`, `bwd_iat_total`.
- **Packet Counts & Rates:** `total_fwd_packets`, `total_bwd_packets`, `flow_packets_per_s`, `flow_bytes_per_s`.
- **Payload Lengths:** `total_fwd_bytes`, `total_bwd_bytes`, `fwd_packet_length_mean`, `fwd_packet_length_std`, `bwd_packet_length_mean`, `bwd_packet_length_std`.
- **Packet Size Statistics:** `packet_length_mean`, `packet_length_std`, `packet_length_variance`, `min_packet_length`, `max_packet_length`.
- **Protocol Flags & Header Sizes:** `syn_flag_count`, `fin_flag_count`, `rst_flag_count`, `psh_flag_count`, `ack_flag_count`, `fwd_header_length`, `bwd_header_length`.

---

### 6. Hybrid Threat Scoring Formula & Threshold Calibration

Mini-IPS combines multiple evidence streams into a normalized Composite Threat Score $S \in [0.0, 1.0]$:

$$S = (w_{\text{rule}} \cdot S_{\text{rule}}) + (w_{\text{ml}} \cdot S_{\text{ml}}) + (w_{\text{anom}} \cdot S_{\text{anom}})$$

Where default calibrated weights are:
- $w_{\text{rule}} = 0.50$ (High confidence in deterministic rules)
- $w_{\text{ml}} = 0.35$ (Supervised model confidence)
- $w_{\text{anom}} = 0.15$ (Unsupervised outlier anomaly score)

**Component Normalization:**
- $S_{\text{rule}}$: Normalized based on rule severity (LOW: 0.25, MEDIUM: 0.50, HIGH: 0.75, CRITICAL: 1.00).
- $S_{\text{ml}}$: Calibrated class probability from `predict_proba()` for predicted attack classes (0.0 if BENIGN).
- $S_{\text{anom}}$: Scaled anomaly score mapped to $[0.0, 1.0]$ (0.0 for inliers).

**Response Decision Tiers:**
- **$S \ge 0.70$ (CRITICAL / HIGH):** Immediate simulated block (or pending approval if Manual Approval Mode is active).
- **$0.40 \le S < 0.70$ (MEDIUM):** Low-priority mitigation recommendation / warning alert.
- **$S < 0.40$ (LOW / INFO):** Standard logging; no prevention action taken.

---

### 7. Evaluation Metrics & Confusion Matrix Interpretation

* **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$ — Misleading on imbalanced network traffic (e.g., 99% benign traffic).
* **Precision:** $TP / (TP + FP)$ — Proportion of triggered alerts that were genuine attacks. High precision prevents blocking innocent users.
* **Recall (Sensitivity):** $TP / (TP + FN)$ — Proportion of actual attacks successfully detected. High recall prevents breaches.
* **F1-Score:** $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$ — Harmonic mean balancing detection rate and false alarms.

**Trained Model Performance on Held-Out CIC-IDS2017 Test Set:**
- **Random Forest Binary:** 99.93% Accuracy, 99.93% F1-Score (Macro).
- **Random Forest Multiclass:** 99.87% Accuracy across Benign, DDoS, and PortScan.

---

### 8. False Positives vs. False Negatives in Network Defense

* **False Positive (Type I Error):** A benign packet/flow is classified as an attack.
  - *Impact in IDS:* Analyst fatigue, cluttered logs.
  - *Impact in IPS:* **Critical self-inflicted Denial of Service (DoS)**. Legitimate user, partner API, or DNS query is blocked.
  - *Mini-IPS Mitigation:* Mandatory Allowlist checks, composite hybrid scoring, and Human-in-the-Loop Manual Approval mode.
* **False Negative (Type II Error):** A malicious packet/flow slips through undetected.
  - *Impact:* Security breach, lateral movement, data exfiltration.
  - *Mini-IPS Mitigation:* Multi-layered hybrid detection (rules + supervised classification + anomaly outlier detection).

---

### 9. Data Leakage & Time-Aware Split Methodology

* **The Problem:** Standard random `train_test_split()` across network packets from the same TCP flow causes severe data leakage. Packets from the same session share identical IP addresses, TTLs, and window sizes, causing artificially inflated 99.99% benchmark accuracy that collapses on real networks.
* **Mini-IPS Methodology:**
  1. Feature extraction strictly aggregates completed flows before inference.
  2. Preprocessing transformers (MinMaxScaler, Imputer) are fit **exclusively on the training split**, then used to transform validation/test sets.
  3. Time-aware or group-aware partitioning ensures flows recorded in later time intervals are evaluated against models trained on earlier intervals.

---

### 10. Model Serialization & Registry Architecture

* **Bundle Structure:** Models are saved under `data/models/<version>/` containing:
  - `model.joblib`: Serialized trained estimator.
  - `preprocessor.joblib`: Fitted pipeline scaler and imputer.
  - `metadata.json`: Feature names, feature ordering, training dataset provenance, commit hash, metrics, and creation timestamp.
* **SQLite Model Registry:** `data/mini_ids.db` tracks registered versions and flags active models (`is_active = 1`).
* **Integrity Validation:** On application startup, `ModelRegistry.validate_artifact()` verifies that the artifact exists, loads properly, and matches the exact 28 feature schema dimensions before enabling inference. If validation fails, rule detection continues safely with `ML unavailable` status.

---

### 11. SQLite Persistence, Busy Timeouts, and Audit Logging

* **Concurrency Handling:** SQLite supports multiple readers but locks during writes. Mini-IPS configures:
  `PRAGMA busy_timeout = 5000;` and `PRAGMA journal_mode = WAL;`
  This allows concurrent worker threads to wait up to 5 seconds rather than raising `sqlite3.OperationalError: database is locked`.
* **Audit Trail:** Every action (policy evaluation, block, unblock, approval, rejection, duplicate suppression, expiration) is recorded in `audit_log` with an immutable schema (Actor, Action, Target, Mode, Timestamp, Result, Details JSON).

---

### 12. Threading, Queues, and Capture Reliability

* **Producer-Consumer Architecture:**
  - *Producer:* Scapy `AsyncSniffer` captures raw frames asynchronously on the network adapter and enqueues them into a bounded `queue.Queue(maxsize=10000)`.
  - *Consumer:* Dedicated background worker pops packets, performs normalization, extracts flow features, runs detection, and writes to SQLite in transactions.
* **Drop Prevention:** If the queue fills up during traffic bursts, dropped packet counters are incremented without crashing the sniffer thread.
* **Clean Shutdown:** Worker checks a `threading.Event()` stop flag, drains remaining queue items, flushes active flows, and rolls back active simulated rules.

---

### 13. Prevention Policy, Allowlisting, and Duplicate Suppression

* **Allowlist Enforcement:** Evaluated before any policy rule:
  - Loopback (`127.0.0.1`, `::1`), Broadcast (`255.255.255.255`), RFC 5737 Test-Nets, Public DNS (`8.8.8.8`, `1.1.1.1`), and local gateway subnets.
  - Matched IPs are permanently exempt from blocking, preventing self-lockout.
* **Duplicate Action Suppression:** If an incoming alert targets an IP that already has an `ACTIVE` or `SIMULATED` rule, or is currently pending in the approval queue:
  - Status is marked `SUPPRESSED`.
  - An audit event `SUPPRESS_DUPLICATE_ACTION` is logged.
  - Prevents rule table duplication, race conditions, and log bloat.

---

### 14. Timed Expiry & Rollback Subsystem

* **Time-to-Live (TTL):** Every simulated block action is assigned an expiration timestamp (default: 300 seconds).
* **Background Expiration Daemon:** A dedicated thread (`start_expiration_worker(interval_seconds=5.0)`) periodically queries active rules and calls `expire_timed_rules()`.
* **Safe State Recovery:** When expired, the rule status transitions from `SIMULATED` to `EXPIRED`, the simulated adapter unblocks the IP, and an audit event is logged.
* **Idempotent Rollback:** On system shutdown or operator unblock request, `engine.unblock()` verifies rule removal and updates SQLite state cleanly.

---

### 15. College Demonstration Walkthrough Script

During the project defense, present the system following this structured 5-minute demonstration:

1. **Architecture & Startup (1 minute):**
   - Launch Streamlit SOC Dashboard: `streamlit run app.py`
   - Show the **Overview** page: highlight stored packet totals, alerts grouped by rule, active model status (`random_forest_binary`), and active prevention mode (`Simulation`).
2. **Offline PCAP / Synthetic Pipeline (1.5 minutes):**
   - Run end-to-end prevention demonstration script: `python scripts/demo_prevention.py`
   - Point out in terminal and dashboard how packets pass through:
     *Packet Normalization → Flow Aggregation (28 features) → Rule + ML Inference → Hybrid Score (0.83) → Allowlist Check → Simulated Action Triggered → State Verified.*
3. **Threat Investigation & Explainability (1 minute):**
   - Navigate to **Alert Investigation** in the dashboard.
   - Select an alert and expand the **Explainable Threat Scoring & Prevention Rationale** card.
   - Explain the 3 weighted score components: Rule Score (0.75), ML Confidence (0.99), and Anomaly Score (0.00).
4. **Prevention Center & Safety Controls (1 minute):**
   - Open **Prevention Center**: showcase the **Pending Approval Queue** (Mode B) and **Active Simulated Rules Table**.
   - Show the **Active Driver** status displaying `SimulationAdapter` with the explicit note: *"Zero host firewall modifications."*
   - Demonstrate the **Manual Unblock / Rollback** button and verify that rule status updates immediately.
5. **Model Laboratory & Evaluation Metrics (30 seconds):**
   - Open **Model Laboratory**: display the genuine classification reports, confusion matrix, and feature importances for `CIC-IDS2017`.
   - Conclude by summarizing: *Mini-IPS delivers a complete, safe, explainable, and demonstrable intrusion prevention workflow suitable for modern enterprise SOC environments.*
