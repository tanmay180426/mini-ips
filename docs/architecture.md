# Mini-IDS: Hybrid Security Analytics & Machine Learning Architecture

## 1. Executive Summary

**Mini-IDS** is an intelligent, hybrid Network Intrusion Detection and Security Analytics Platform designed for Windows environments. It bridges traditional cybersecurity detection heuristics with supervised machine learning and unsupervised anomaly detection, underpinned by an interactive exploratory data science workspace.

Mini-IDS operates under a strict **passive detection** paradigm: it observes, reconstructs, extracts tabular flow features, and evaluates traffic against heuristic rules and machine learning models to generate actionable security intelligence without inline disruption (e.g. packet dropping or session RST injection).

---

## 2. End-to-End System Architecture

The following diagram illustrates the complete data flow from network packets and PCAP traces to the Streamlit analytics interface:

```
+---------------------------------------------------------------------------------------+
|                                    TRAFFIC INGESTION                                  |
|                                                                                       |
|  +---------------------------+                     +-------------------------------+  |
|  |  Offline PCAP/PCAPNG File |                     |  Live Network Interface       |  |
|  |  (capture/packet_capture) |                     |  (capture/live_capture.py)    |  |
|  +-------------+-------------+                     +---------------+---------------+  |
+----------------|---------------------------------------------------|------------------+
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             PACKET NORMALIZATION LAYER                                |
|                                                                                       |
|   NormalizedPacket: timestamp, src_ip, dst_ip, src_port, dst_port, protocol,          |
|                     tcp_flags, length                                                 |
+------------------------------------------+--------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+--------------------------------------+      +-----------------------------------------+
|     STREAMING HEURISTIC ENGINE       |      |     BIDIRECTIONAL FLOW AGGREGATOR       |
|                                      |      |     (ml/feature_extraction.py)          |
|  +--------------------------------+  |      |                                         |
|  | PortScanDetector (Windowed)    |  |      |  - Canonical 5-Tuple Key Tracking       |
|  +--------------------------------+  |      |  - Forward & Backward Packet Tracking   |
|  | SynRateDetector (Windowed)     |  |      |  - Timeout & FIN/RST Flow Eviction      |
|  +--------------------------------+  |      |  - 28 Tabular Statistical Features      |
|  | TrafficAnomalyDetector (Vol.)  |  |      +--------------------+--------------------+
|  +--------------------------------+  |                           |
+-------------------+------------------+                           | Completed Flows
                    |                                              v
                    | Heuristic Alerts        +-----------------------------------------+
                    |                         |        MACHINE LEARNING PIPELINE        |
                    |                         |                                         |
                    |                         |  +-----------------------------------+  |
                    |                         |  | Scaler & Imputer Transformers     |  |
                    |                         |  +-----------------------------------+  |
                    |                         |  | Supervised Classifier             |  |
                    |                         |  | (Random Forest / Logistic Reg.)   |  |
                    |                         |  +-----------------------------------+  |
                    |                         |  | Unsupervised Anomaly Detector     |  |
                    |                         |  | (Isolation Forest Baseline)       |  |
                    |                         |  +-----------------------------------+  |
                    |                         +--------------------+--------------------+
                    |                                              |
                    +----------------------+-----------------------+
                                           | Correlated Alerts & Flows
                                           v
+---------------------------------------------------------------------------------------+
|                             PERSISTENT SQLITE STORAGE                                 |
|                                                                                       |
|   Tables: packets | alerts | flows | models                                           |
|   WAL Journaling Mode | B-Tree Indexes on timestamp, IPs, rule_name                   |
+------------------------------------------+--------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                       STREAMLIT 9-PAGE SECURITY DASHBOARD                             |
|                                                                                       |
|  [1] Overview            [4] Traffic Intelligence     [7] Data Explorer               |
|  [2] Live Monitoring     [5] AI/ML Detection          [8] Detection Rules             |
|  [3] Alert Investigation [6] Model Laboratory         [9] System & Settings           |
+---------------------------------------------------------------------------------------+
```

---

## 3. Subsystem Breakdown

### 3.1 Ingestion & Normalization (`capture/`)
- **Offline PCAP Reader (`capture/packet_capture.py`):** Iteratively yields packets using Scapy's `PcapReader` without memory bloat.
- **Live Windows Sniffer (`capture/live_capture.py`):** Interacts with Npcap via Scapy's `AsyncSniffer`. Implements a decoupled multi-threaded worker using a thread-safe `queue.Queue` to separate raw packet capture from disk I/O and detection logic.
- **Packet Normalizer:** Standardizes disparate protocol structures into a uniform `NormalizedPacket` dataclass, ensuring that offline replays and live network streams execute identical downstream logic.

### 3.2 Bidirectional Flow Aggregation (`ml/feature_extraction.py`)
- Maintains active in-flight network flows using a canonical, direction-agnostic 5-tuple:
  `canonical_key = (min(ip1, ip2), max(ip1, ip2), min(port1, port2), max(port1, port2), protocol)`
- Tracks directional forward (`fwd_stats`) and backward (`bwd_stats`) counters, packet length distributions, and inter-arrival times (IAT).
- Evicts completed flows either upon observation of TCP `FIN`/`RST` termination flags or when the idle timeout (default: 15.0 seconds) expires.
- Computes 28 statistical features matching the standard **CICIDS2017** feature conventions.

### 3.3 Hybrid Detection Engine (`detection/hybrid_detection.py`)
The `HybridDetectionEngine` coordinates multiple detection methodologies:
1. **Rule-Based Heuristic Detectors:**
   - `PortScanDetector`: Tracks unique destination ports contacted per source IP over sliding time windows.
   - `SynRateDetector`: Evaluates TCP SYN flag arrival rates to identify volumetric SYN flood attempts.
   - `TrafficAnomalyDetector`: Flags byte or packet rate surges exceeding rolling baseline thresholds.
2. **Supervised ML Classification:**
   - Transforms extracted flow features using a serialized preprocessing pipeline.
   - Evaluates Random Forest, Logistic Regression, or Gradient Boosted trees.
   - Generates alerts tagged with `detection_method='ML_SUPERVISED'` along with model version and confidence probability.
3. **Unsupervised Anomaly Detection:**
   - Applies an Isolation Forest model trained exclusively on benign baseline flow distributions.
   - Evaluates decision functions and generates normalized anomaly scores `[0.0, 1.0]`.
   - Alerts on flows exceeding calibrated contamination thresholds with `detection_method='ML_ANOMALY'`.

### 3.4 Model Registry & Artifact Management (`ml/model_registry.py`)
- Versioned storage for trained models, preprocessors, and JSON metadata manifests in `data/models/<version>/`.
- Atomic registration and activation management tracked within the SQLite `models` table.
- Dry-run validation of model integrity and feature schema alignment prior to activation.

### 3.5 Database & Storage Layer (`storage/database.py`)
Persistent storage is implemented in SQLite with additive, backward-compatible migrations:
- **`packets` table:** Normalized packet logs indexed by timestamp, source IP, destination IP, and protocol.
- **`alerts` table:** Security alert events with `detection_method`, `model_version`, `confidence`, and `anomaly_score`.
- **`flows` table:** Completed network flow records with extracted statistical features serialized as JSON.
- **`models` table:** Registered machine learning model metadata, performance metrics, and activation flags.

---

## 4. UI/UX Design System

The platform user interface is built in **Streamlit** following a professional, restrained aesthetic:
- **Palette:**
  - Background: `#F6F5F0` (Warm Canvas)
  - Sidebar: `#202C3B` (Dark Navy Slate)
  - Cards: `#FFFFFF` (Crisp White with `#E2E3DE` borders)
  - Primary Accent: `#C76D45` (Terracotta)
  - Secondary Accent: `#52877F` (Sage Teal)
  - Critical Alerts: `#C65353` (Crimson)
  - Warnings: `#D5A13B` (Amber)
  - Text: `#252A31` (Deep Charcoal)
- **Visualizations:** High-density, accessible Altair charts utilizing discrete interval binning to eliminate misleading interpolation across observation gaps.
