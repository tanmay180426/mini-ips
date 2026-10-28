# Mini-IPS: Advanced ML-Assisted Intrusion Prevention System (Simulation-Only)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64%20%7C%20Linux-lightgrey.svg)](https://www.microsoft.com/windows)
[![Tests](https://img.shields.io/badge/Tests-231%20Passed-brightgreen.svg)](tests/)
[![Safety](https://img.shields.io/badge/Safety-Simulation--Only%20Guard-success.svg)](prevention/simulation_adapter.py)
[![Architecture](https://img.shields.io/badge/IDPS-Hybrid%20Detection%20%2B%20Explainable%20Scoring-orange.svg)](docs/architecture.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An integrated, machine-learning-assisted **Mini Intrusion Prevention System (Mini-IPS)** and SOC Security Analytics Platform built with Python, Scapy, scikit-learn, SQLite, and Streamlit.

Designed for Computer Engineering final-year evaluations, technical reviews, and cybersecurity operations research, Mini-IPS operates in **Strict Simulation-Only Mode**:
1. **Cybersecurity & Network Defense:** Stateful heuristic rule detection, raw packet dissection, bidirectional flow reconstruction (28 canonical features), explainable composite threat scoring, simulation-only prevention execution with active duplicate suppression, verification, and automated background expiration.
2. **Artificial Intelligence & Machine Learning:** Supervised classification and unsupervised anomaly detection can load versioned artifacts with their fitted preprocessors. The active artifacts are currently associated with generated CICIDS-style sample data; original benchmark provenance and benchmark performance are not established.
3. **Data Science & SOC Analytics:** Statistical dataset profiling, correlation analysis, interactive exploratory data analysis (EDA), and forensic audit trails.

---

## 1. System Architecture & Prevention Pipeline

```
                       Network Traffic / Uploaded PCAP
                                      |
                                      v
                              Traffic Ingestion
                        (Live Sniffer / PCAP Reader)
                                      |
                                      v
                        Packet Parsing & Normalization
                                      |
                                      v
                        Bidirectional Flow Aggregation
                           (28 Canonical Features)
                                      |
                                      v
                           Hybrid Detection Engine
             +------------------------+------------------------+
             |                        |                        |
             v                        v                        v
     Heuristic Rules         Supervised Classifiers    Isolation Forest
 (PortScan, SynRate, Vol)      (RF, Logistic Reg)      (Anomaly Outlier)
             |                        |                        |
             +------------------------+------------------------+
                                      |
                                      v
                        Threat Scoring & Correlation
                        (Composite Score 0.0 - 1.0)
                                      |
                                      v
                            Response Policy Engine
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
   [Allowlist Match / Low Risk]                            [High Threat Score]
         |                                                         |
         v                                                         v
    LOG_ALERT Only                                         Prevention Mode
 (Zero Action Taken)                         +---------------------+---------------------+
                                             |                     |                     |
                                             v                     v                     v
                                          MODE A:               MODE B:               MODE C:
                                        Simulation          Manual Approval          Authorized
                                      (Safe Default)        (Operator Queue)    Compatibility Mode
                                             |                     |          (Simulation Only)
                                             |                     v                     |
                                             |              [Human Review]               |
                                             |              Approve / Reject             |
                                             +---------------------+---------------------+
                                                                   |
                                                                   v
                                                        Simulation Adapter Execution
                                                        (No OS Firewall Commands)
                                                                   |
                                                                   v
                                                        Simulation Verification
                                                      (Check Simulated Rule State)
                                                                   |
                                                                   v
                                                        SQLite Audit Logging
                                                      (sqlite3 audit_log table)
                                                                   |
                                                                   v
                                                      Streamlit Prevention Center
                                                        & Forensic Audit Console
```

---

## 2. Key Prevention Safeguards & Operating Modes

Mini-IDS enforces strict defense-in-depth safeguards to ensure safe, ethical, and reliable prevention operations:

### 2.1 Multi-Mode Prevention Engine
- **Mode A — Simulation Mode (Safe Default):** Detections generate calibrated response recommendations and simulated firewall entries in memory and database. The host operating system firewall is **never** altered. Recommended for evaluations, classroom settings, and non-administrative environments.
- **Mode B — Manual Approval Mode (Human-in-the-Loop):** Suspected attacks are placed in the **Pending Approvals Queue**. An authorized security analyst must explicitly inspect the rationale, threat score, and target before approving execution.
- **Mode C — Compatibility Selector:** The legacy `AUTHORIZED_ENFORCEMENT` enum/UI value is retained for compatibility, but the prevention engine routes it to the simulation adapter. The application does not perform host firewall enforcement; no `netsh` or `iptables` command is executed by the prevention path.

### 2.2 Built-in Safety Constraints
1. **Self-Lockout Immunity:** Core infrastructure addresses can **never** receive a prevention simulation:
   - Loopback: `127.0.0.1`, `::1`, `localhost`
   - Network Infrastructure: `0.0.0.0`, `255.255.255.255`
   - Public Resolvers: `8.8.8.8`, `8.8.4.4`, `1.1.1.1`, `1.0.0.1`, `9.9.9.9`
   - Configurable local subnets (gateways and operator IPs)
2. **Offline Data Containment:** Ingestion of offline PCAP captures or synthetic demonstration data is restricted to **Simulation Mode**. The simulation records decisions in memory and never blocks network traffic.
3. **Simulation Rule Scoping:** Simulated rule names use the `MiniIDS_Block_*` prefix and live in the in-memory simulation adapter.
4. **Simulation Verification:** Verification checks simulated adapter state; it does not query or verify the host firewall.
5. **Timed Simulation Lifecycle:** Simulated actions support expiration and revocation. This does not remove or alter host firewall rules.

---

## 3. Model Artifacts and Evaluation Provenance

The active model artifacts load and perform inference, but their recorded dataset name (`cicids2017`) does not establish benchmark provenance. The local dataset manifest identifies `cicids2017_sample_1200.csv` as produced by the Mini-IDS deterministic statistical generator using the CICIDS2017 schema. Accordingly, the older benchmark metrics and confusion matrix previously listed here are not supported as results on the original CIC-IDS2017 corpus and are removed. Treat the active artifacts as demonstrations on generated sample data until they are retrained and evaluated on a provenance-verified benchmark dataset.

---

## 4. Installation & Windows Setup

### 4.1 Prerequisites
1. **Python 3.10+ (Tested through Python 3.14):** Verify with `python --version`.
2. **Npcap for Windows (Required only for live interface capture):**
   - Download and install [Npcap](https://npcap.com/#download).
   - Ensure the option **"Install Npcap in WinPcap API-compatible Mode"** is checked.
   - *Note: Npcap is NOT required for offline PCAP analysis, model training, prevention simulation, or demonstration mode.*

### 4.2 Clone & Virtual Environment Setup
Open PowerShell as Administrator (or standard user for simulation/demo):

```powershell
# 1. Enable execution of PowerShell scripts if needed
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 2. Clone repository and navigate to workspace
git clone https://github.com/anaygore45/mini-ids.git
cd mini-ids

# 3. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Quick Start & Demonstration

### 5.1 Run the Interactive Prevention Demonstration
Execute the end-to-end prevention verification script to observe threat scoring, simulation, manual approval, allowlist protection, and rollback:
```powershell
python scripts/demo_prevention.py
```

### 5.2 Launch the Streamlit SOC Dashboard and Offline Demo
From the repository root:
```powershell
streamlit run app.py
```
Open `http://localhost:8501`. The sidebar's **Demonstration Mode (Synthetic)** toggle uses a separate deterministic demo database at `data/demo_synthetic.db` (or the path in `MINI_IDS_DEMO_DB_PATH`). It does not populate the selected production database. The demo includes synthetic packets and alerts for investigation and policy simulation.

Suggested demo workflow:
1. Enable **Demonstration Mode (Synthetic)**.
2. Review Overview, Traffic Intelligence, and Alert Investigation. Demo records are labeled synthetic.
3. Open Prevention Center and choose an existing stored demo alert under **Evaluate a stored demonstration detection**.
4. Click **Evaluate simulation proposal** and inspect the resulting decision/action in Prevention Center.
5. Open Audit Log to review the policy and simulation events.

All prevention outcomes are simulated. They do not block traffic or change host firewall rules. For an empty local database, ingest an offline capture explicitly with `python main.py --pcap <capture.pcap> --db <path-to-database>`.

### 5.3 Offline PCAP Hybrid Analysis
To analyze a recorded network capture and correlate heuristic rules with ML models and policy recommendations:
```powershell
python main.py --pcap data/sample.pcap --hybrid
```

### 5.4 Live Windows Packet Capture
```powershell
# List available network adapters discovered by Npcap
python main.py --list-interfaces

# Sniff packets live on interface (e.g. Wi-Fi) with limits
python main.py --interface "Wi-Fi" --count 500 --timeout 30 --hybrid
```

---

## 6. Multi-Dataset AI Training & Generalization Pipeline

Mini-IDS implements a reproducible, multi-dataset machine learning and cross-dataset evaluation architecture across three public benchmark corpora:
1. **Dataset A: CIC-IDS2017** (Supervised binary & multiclass flow telemetry)
2. **Dataset B: UNSW-NB15** (Disparate modern topology, high-dimensional packet statistics)
3. **Dataset C: CSE-CIC-IDS2018** (Collaborative AWS Open Data network attack flows)

### 6.1 Two Experimental Methodologies
- **Experiment A (Dataset-Specific Baselines):** Native high-dimensional models trained on verified feature schemas for each individual dataset.
- **Experiment B (Unified Cross-Dataset Core):** Harmonized 11-feature common semantic core (`flow_duration_sec`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`, `total_packets`, `total_bytes`, `packet_rate`, `byte_rate`, `fwd_pkt_len_mean`, `bwd_pkt_len_mean`) with standardized binary and multiclass taxonomies.

### 6.2 Executable Pipeline Commands
```powershell
# 1. Acquire and verify all three benchmark datasets
python -m training.download_datasets --dataset all

# 2. Audit schema health, verify SHA256 checksums, and check data leakage
python -m training.validate_datasets --dataset all

# 3. Prepare zero-leakage partitions (fit RobustScaler & imputers strictly on X_train)
python -m training.prepare_datasets --dataset all

# 4. Train 5 ML estimators (RF, ET, Logistic Reg, HistGradientBoosting, Isolation Forest)
python -m training.train_models --dataset all --task both

# 5. Evaluate models on held-out test sets across Acc, Bal Acc, F1, and False Alarm Rates
python -m training.evaluate_models --dataset all --task both

# 6. Benchmark cross-dataset transferability and domain shift across all source permutations
python -m training.cross_dataset_evaluation

# 7. Compile consolidated reports and artifact metadata
python -m training.generate_report
```

### 6.3 Executable 20-Step Jupyter Notebook
For interactive thesis defense, research walkthroughs, and visual verification:
```powershell
jupyter notebook notebooks/multi_dataset_training.ipynb
```
The notebook executes:
1. Dataset Download & Verification
2. Dataset Inspection
3. Data Cleaning
4. Feature and Label Analysis
5. Class Distribution Visualization
6. Feature-Schema Reconciliation
7. Data Leakage Checks
8. Binary Label Standardization
9. Multiclass Label Mapping
10. Training/Validation/Test Partitioning
11. Model Training
12. Hyperparameter Selection
13. Evaluation
14. Cross-Dataset Testing
15. Confusion Matrix Visualization
16. Model Comparison
17. Feature Importance
18. Model Saving and Reloading
19. Example Inference on Unseen Records
20. Final Report Generation

---

## 7. Running the Automated Test Suite

The test suite contains unit, integration, model, prevention, and Streamlit UI checks. Test totals and failures depend on the current checkout; use the commands below to obtain the current result:

```powershell
# Run the complete test suite
python -m pytest

# Run with concise summary
python -m pytest -q

# Run multi-dataset and prevention suites specifically
python -m pytest tests/test_data_validation.py tests/test_preprocessing.py tests/test_models.py tests/test_cross_dataset.py tests/test_inference.py tests/test_integration.py tests/test_prevention.py
```

---

## 8. 11-Page Security Analytics Interface

The Streamlit dashboard provides 11 distinct, focused analytical workspaces:

1. **Overview:** Stored security metrics, alert distribution charts, capture controls, and runtime model bundle/inference health when persisted flows are available.
2. **Live Monitoring:** Interface selection, packet/duration limits, active sniffer telemetry, and real-time event streaming.
3. **Prevention Center (NEW):** Simulation and manual review workflows, simulated action history, manual simulated revocation, and allowlist safeguards. All available modes are simulation-only; network traffic is never claimed to be blocked.
4. **Audit Log (NEW):** SQLite audit records for policy evaluations, simulated actions, approvals, and rollbacks. The log is not described as immutable or tamper-proof.
5. **Alert Investigation:** Searchable, filterable incident workspace with stored packet evidence, linked prevention decisions, audit-backed notes, and triage/resolution state.
6. **Traffic Intelligence:** Persisted packet and flow counts, protocol breakdowns, destination ports, and indexed flow summaries. Alerts are not treated as packet-level classifications.
7. **AI/ML Detection:** Active bundle loading and inference probes on persisted flow features, with explicit unavailable/incompatible status. A successful probe alone does not claim that ML contributes to live detection.
8. **Model Laboratory:** Registered model versions, evaluation metrics comparison table, feature importance ranking, and one-click model activation.
9. **Data Explorer:** Tabular dataset profiling, missing/infinite value health checks, class distribution charts, and Pearson correlation matrices. Opening the page does not generate sample data.
10. **Detection Rules:** Configuration and threshold documentation for heuristic rules and ML confidence boundaries.
11. **System & Settings:** Database diagnostics, table row counts, schema status, log viewer, and demonstration mode toggle.

---

## 9. Limitations & Ethical Considerations

> [!CAUTION]
> **Defensive and Educational Scope Only:**
> Mini-IDS is designed strictly for authorized security monitoring, educational demonstrations, and defense research. It must not be deployed to monitor or alter networks or hosts without explicit authorization.

- **Prevention Mode:** All application prevention actions route through the simulation adapter. Windows and Linux firewall adapter objects exist for guarded compatibility, but their OS-changing methods are not called by the application path.
- **Model Generalization:** Current active artifacts have not been established as trained on the original CIC-IDS2017 corpus. Their metrics must not be treated as benchmark performance; generated sample data does not establish real-world detection quality.
- **High-Throughput Limitations:** Python and Scapy are optimized for educational clarity and modularity. In gigabit enterprise production environments, dedicated kernel-bypass solutions (e.g. DPDK, PF_RING) and hardware-accelerated engines (e.g. Suricata, Zeek) are standard.

---

## 10. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
