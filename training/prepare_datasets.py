"""Multi-Dataset Preparation, Harmonization, and Leakage-Free Preprocessing Pipeline.

Supports:
- Experiment A: Dataset-specific models (CIC-IDS2017, UNSW-NB15, CSE-CIC-IDS2018).
- Experiment B: Unified cross-dataset model on a common semantic feature subset.
- Zero-leakage stratified train/val/test partitioning.
- Median imputation and RobustScaler fitted strictly on X_train.
- Standardized binary (BENIGN / ATTACK) and multiclass taxonomy.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Canonical 11-feature common semantic core for Experiment B (Unified Model)
UNIFIED_CORE_FEATURES = [
    "flow_duration_sec",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "total_packets",
    "total_bytes",
    "packet_rate",
    "byte_rate",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
]

# Standard Multiclass Taxonomy
MULTICLASS_TAXONOMY = {
    0: "BENIGN",
    1: "DOS_DDOS",
    2: "PORT_SCAN_RECONNAISSANCE",
    3: "BRUTE_FORCE",
    4: "BOTNET",
    5: "WEB_ATTACK",
    6: "OTHER_ATTACK",
}

TAXONOMY_MAP = {v: k for k, v in MULTICLASS_TAXONOMY.items()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mini_ids.training.prepare_datasets")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"


def map_label_to_taxonomy(raw_label: str, dataset_key: str) -> Tuple[int, str, int, str]:
    """Map raw dataset label to binary (0=BENIGN, 1=ATTACK) and standard multiclass taxonomy.

    Returns:
        Tuple[binary_int, binary_str, multiclass_int, multiclass_str]
    """
    clean_lbl = str(raw_label).strip()
    upper = clean_lbl.upper()

    # Default
    b_int = 1
    b_str = "ATTACK"
    m_str = "OTHER_ATTACK"

    if dataset_key in ("cicids2017", "csecicids2018"):
        if upper in ("BENIGN", "NORMAL"):
            b_int = 0
            b_str = "BENIGN"
            m_str = "BENIGN"
        elif any(k in upper for k in ["DDOS", "DOS", "SLOWLORIS", "SLOWHTTPTEST", "HULK", "GOLDENEYE", "LOIC", "HOIC"]):
            m_str = "DOS_DDOS"
        elif any(k in upper for k in ["PORTSCAN", "PORT_SCAN"]):
            m_str = "PORT_SCAN_RECONNAISSANCE"
        elif any(k in upper for k in ["PATATOR", "BRUTEFORCE", "BRUTE_FORCE", "BRUTE FORCE"]):
            if "WEB" in upper or "XSS" in upper:
                m_str = "WEB_ATTACK"
            else:
                m_str = "BRUTE_FORCE"
        elif "BOT" in upper:
            m_str = "BOTNET"
        elif any(k in upper for k in ["WEB", "XSS", "SQL INJECTION", "INJECTION"]):
            m_str = "WEB_ATTACK"
        else:
            m_str = "OTHER_ATTACK"

    elif dataset_key == "unsw_nb15":
        if upper in ("NORMAL", "BENIGN", "0"):
            b_int = 0
            b_str = "BENIGN"
            m_str = "BENIGN"
        elif "DOS" in upper:
            m_str = "DOS_DDOS"
        elif any(k in upper for k in ["RECONNAISSANCE", "RECON"]):
            m_str = "PORT_SCAN_RECONNAISSANCE"
        elif any(k in upper for k in ["BACKDOOR", "WORMS"]):
            m_str = "BOTNET"
        else:
            m_str = "OTHER_ATTACK"

    m_int = TAXONOMY_MAP.get(m_str, 6)
    return b_int, b_str, m_int, m_str


def extract_unified_features_from_row(row: pd.Series, dataset_key: str) -> Dict[str, float]:
    """Map a raw record into the harmonized 11-feature core schema with matching units."""
    feat: Dict[str, float] = {}

    if dataset_key == "cicids2017":
        # Flow Duration is in microseconds in CIC-IDS2017
        dur_us = float(row.get("Flow Duration", row.get("flow_duration", 0.0)))
        dur_sec = max(0.0, dur_us / 1_000_000.0)
        fwd_p = float(row.get("Total Fwd Packets", row.get("total_fwd_packets", 0.0)))
        bwd_p = float(row.get("Total Backward Packets", row.get("total_backward_packets", 0.0)))
        fwd_b = float(row.get("Total Length of Fwd Packets", row.get("total_length_of_fwd_packets", 0.0)))
        bwd_b = float(row.get("Total Length of Bwd Packets", row.get("total_length_of_bwd_packets", 0.0)))
        pkt_rate = float(row.get("Flow Packets/s", row.get("flow_packets_s", (fwd_p + bwd_p) / max(dur_sec, 1e-4))))
        byte_rate = float(row.get("Flow Bytes/s", row.get("flow_bytes_s", (fwd_b + bwd_b) / max(dur_sec, 1e-4))))
        fwd_len_mean = float(row.get("Fwd Packet Length Mean", row.get("fwd_packet_length_mean", 0.0)))
        bwd_len_mean = float(row.get("Bwd Packet Length Mean", row.get("bwd_packet_length_mean", 0.0)))

    elif dataset_key == "csecicids2018":
        # Flow Duration is in microseconds in CSE-CIC-IDS2018
        dur_us = float(row.get("Flow Duration", 0.0))
        dur_sec = max(0.0, dur_us / 1_000_000.0)
        fwd_p = float(row.get("Tot Fwd Pkts", 0.0))
        bwd_p = float(row.get("Tot Bwd Pkts", 0.0))
        fwd_b = float(row.get("TotLen Fwd Pkts", 0.0))
        bwd_b = float(row.get("TotLen Bwd Pkts", 0.0))
        pkt_rate = float(row.get("Flow Pkts/s", (fwd_p + bwd_p) / max(dur_sec, 1e-4)))
        byte_rate = float(row.get("Flow Byts/s", (fwd_b + bwd_b) / max(dur_sec, 1e-4)))
        fwd_len_mean = float(row.get("Fwd Pkt Len Mean", 0.0))
        bwd_len_mean = float(row.get("Bwd Pkt Len Mean", 0.0))

    elif dataset_key == "unsw_nb15":
        # dur is already in seconds in UNSW-NB15
        dur_sec = max(0.0, float(row.get("dur", 0.0)))
        fwd_p = float(row.get("spkts", 0.0))
        bwd_p = float(row.get("dpkts", 0.0))
        fwd_b = float(row.get("sbytes", 0.0))
        bwd_b = float(row.get("dbytes", 0.0))
        pkt_rate = float(row.get("rate", (fwd_p + bwd_p) / max(dur_sec, 1e-4)))
        byte_rate = (fwd_b + bwd_b) / max(dur_sec, 1e-4)
        fwd_len_mean = float(row.get("smean", 0.0))
        bwd_len_mean = float(row.get("dmean", 0.0))

    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    feat["flow_duration_sec"] = dur_sec
    feat["fwd_packets"] = max(0.0, fwd_p)
    feat["bwd_packets"] = max(0.0, bwd_p)
    feat["fwd_bytes"] = max(0.0, fwd_b)
    feat["bwd_bytes"] = max(0.0, bwd_b)
    feat["total_packets"] = feat["fwd_packets"] + feat["bwd_packets"]
    feat["total_bytes"] = feat["fwd_bytes"] + feat["bwd_bytes"]
    feat["packet_rate"] = max(0.0, pkt_rate)
    feat["byte_rate"] = max(0.0, byte_rate)
    feat["fwd_pkt_len_mean"] = max(0.0, fwd_len_mean)
    feat["bwd_pkt_len_mean"] = max(0.0, bwd_len_mean)

    return feat


def load_raw_dataset(
    dataset_key: str,
    max_records: int = 50_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load raw dataset files for a given dataset key."""
    raw_sub = RAW_DIR / dataset_key
    if not raw_sub.exists():
        raw_sub = DATA_DIR / "datasets" / dataset_key / "raw"

    if not raw_sub.exists():
        raise FileNotFoundError(f"Raw directory not found for {dataset_key}: {raw_sub}")

    # Prioritize balanced CSV if available, else parquets, else raw CSVs
    csv_files = list(raw_sub.glob("*.csv"))
    parquet_files = list(raw_sub.glob("*.parquet"))

    if dataset_key == "cicids2017":
        multi_csv = raw_sub / "cicids2017_multiclass.csv"
        if multi_csv.exists():
            df = pd.read_csv(multi_csv, low_memory=False)
        elif parquet_files:
            dfs = []
            for pq_file in parquet_files:
                sub_df = pd.read_parquet(pq_file)
                if len(sub_df) > max_records // 2:
                    sub_df = sub_df.sample(n=max_records // 2, random_state=random_seed)
                dfs.append(sub_df)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(csv_files[0], low_memory=False)

    elif dataset_key == "unsw_nb15":
        test_csv = raw_sub / "UNSW_NB15_testing-set.csv"
        target_f = test_csv if test_csv.exists() else csv_files[0]
        # utf-8-sig to handle BOM
        df = pd.read_csv(target_f, encoding="utf-8-sig", low_memory=False)

    elif dataset_key == "csecicids2018":
        wed_csv = raw_sub / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
        target_f = wed_csv if wed_csv.exists() else csv_files[0]
        df = pd.read_csv(target_f, low_memory=False)

    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    df.columns = [str(c).strip() for c in df.columns]

    if len(df) > max_records:
        df = df.sample(n=max_records, random_state=random_seed)

    return df


def prepare_dataset_specific(
    dataset_key: str,
    output_dir: Optional[Path] = None,
    max_records: int = 30_000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Prepare dataset-specific models (Experiment A)."""
    out_dir = output_dir or (PROCESSED_DIR / dataset_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data for %s...", dataset_key)
    df = load_raw_dataset(dataset_key, max_records=max_records, random_seed=random_seed)
    logger.info("Loaded %d records, %d columns", len(df), len(df.columns))

    # Identify label column
    label_candidates = ["Label", "label", "attack_cat", "class", "Class"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if not label_col:
        raise ValueError(f"No label column found in {dataset_key}. Columns: {list(df.columns)}")

    # Extract target series
    raw_labels = df[label_col].astype(str).str.strip()

    # Map to binary and multiclass
    b_ints, b_strs, m_ints, m_strs = [], [], [], []
    for lbl in raw_labels:
        b_i, b_s, m_i, m_s = map_label_to_taxonomy(lbl, dataset_key)
        b_ints.append(b_i)
        b_strs.append(b_s)
        m_ints.append(m_i)
        m_strs.append(m_s)

    y_binary = np.array(b_ints, dtype=np.int32)
    y_multiclass = np.array(m_ints, dtype=np.int32)

    # Filter feature matrix: drop leakage and targets
    leakage_cols = {
        label_col, "label", "Label", "attack_cat", "id", "flow_id", "timestamp", "Timestamp",
        "src_ip", "dst_ip", "srcip", "dstip", "sport", "dport", "Destination Port", "Dst Port"
    }
    feature_candidates = [c for c in df.columns if c not in leakage_cols]

    # Keep only numeric columns
    numeric_df = df[feature_candidates].select_dtypes(include=[np.number]).copy()
    feature_names = list(numeric_df.columns)
    logger.info("Kept %d numeric features for %s", len(feature_names), dataset_key)

    # Clean infinities to NaNs for imputer
    X_raw = numeric_df.values.astype(np.float32)
    X_raw[np.isinf(X_raw)] = np.nan

    artifacts = {}

    for task in ["binary", "multiclass"]:
        y_target = y_binary if task == "binary" else y_multiclass
        # Ensure at least 2 classes present
        unique_classes = np.unique(y_target)
        if len(unique_classes) < 2:
            logger.warning("Only 1 class found for %s (%s). Skipping.", dataset_key, task)
            continue

        # Split 1: Split off test set (zero leakage)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_raw, y_target, test_size=test_size, random_state=random_seed, stratify=y_target
        )
        # Split 2: Split remaining into train and validation
        adj_val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=adj_val_ratio, random_state=random_seed, stratify=y_temp
        )

        # Impute and Scale strictly fit on X_train
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        scaler = RobustScaler()
        X_train_scl = scaler.fit_transform(X_train_imp)
        X_val_scl = scaler.transform(X_val_imp)
        X_test_scl = scaler.transform(X_test_imp)

        # Save arrays and pipeline
        np.save(out_dir / f"X_train_{task}.npy", X_train_scl)
        np.save(out_dir / f"X_val_{task}.npy", X_val_scl)
        np.save(out_dir / f"X_test_{task}.npy", X_test_scl)
        np.save(out_dir / f"y_train_{task}.npy", y_train)
        np.save(out_dir / f"y_val_{task}.npy", y_val)
        np.save(out_dir / f"y_test_{task}.npy", y_test)

        preprocessor_bundle = {
            "imputer": imputer,
            "scaler": scaler,
            "feature_names": feature_names,
            "task_type": task,
            "dataset_key": dataset_key,
        }
        joblib.dump(preprocessor_bundle, out_dir / f"preprocessor_{task}.joblib")
        joblib.dump(scaler, out_dir / f"scaler_{task}.joblib")

        if task == "binary":
            lbl_map = {"BENIGN": 0, "ATTACK": 1}
            inv_lbl_map = {"0": "BENIGN", "1": "ATTACK"}
        else:
            lbl_map = {MULTICLASS_TAXONOMY[int(c)]: int(c) for c in unique_classes}
            inv_lbl_map = {str(int(c)): MULTICLASS_TAXONOMY[int(c)] for c in unique_classes}

        meta = {
            "dataset_key": dataset_key,
            "task_type": task,
            "experiment": "Experiment A (Dataset-Specific)",
            "num_features": len(feature_names),
            "feature_names": feature_names,
            "label_mapping": lbl_map,
            "inverse_label_mapping": inv_lbl_map,
            "train_samples": len(X_train_scl),
            "val_samples": len(X_val_scl),
            "test_samples": len(X_test_scl),
            "class_distribution_train": {int(k): int((y_train == k).sum()) for k in unique_classes},
            "class_distribution_test": {int(k): int((y_test == k).sum()) for k in unique_classes},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_dir / f"metadata_{task}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        artifacts[task] = {
            "train_scaled": out_dir / f"X_train_{task}.npy",
            "val_scaled": out_dir / f"X_val_{task}.npy",
            "test_scaled": out_dir / f"X_test_{task}.npy",
            "y_train": out_dir / f"y_train_{task}.npy",
            "y_val": out_dir / f"y_val_{task}.npy",
            "y_test": out_dir / f"y_test_{task}.npy",
            "scaler": out_dir / f"scaler_{task}.joblib",
            "metadata": out_dir / f"metadata_{task}.json",
        }

    return artifacts


def prepare_unified_dataset(
    output_dir: Optional[Path] = None,
    samples_per_dataset: int = 15_000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Prepare unified multi-dataset corpus on common 11-feature core (Experiment B)."""
    out_dir = output_dir or (PROCESSED_DIR / "unified")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Constructing Unified Cross-Dataset Corpus (Experiment B)...")
    datasets = ["cicids2017", "unsw_nb15", "csecicids2018"]

    all_features = []
    all_b_labels = []
    all_m_labels = []
    all_sources = []

    for ds_key in datasets:
        try:
            df = load_raw_dataset(ds_key, max_records=samples_per_dataset, random_seed=random_seed)
            label_col = next((c for c in ["Label", "label", "attack_cat"] if c in df.columns), None)
            if not label_col:
                continue

            for _, row in df.iterrows():
                feat = extract_unified_features_from_row(row, ds_key)
                b_i, _, m_i, _ = map_label_to_taxonomy(row[label_col], ds_key)

                all_features.append([feat[f] for f in UNIFIED_CORE_FEATURES])
                all_b_labels.append(b_i)
                all_m_labels.append(m_i)
                all_sources.append(ds_key)

            logger.info("Extracted %d unified records from %s", len(df), ds_key)
        except Exception as err:
            logger.warning("Could not process %s for unified dataset: %s", ds_key, err)

    X_raw = np.array(all_features, dtype=np.float32)
    X_raw[np.isinf(X_raw)] = np.nan
    y_binary = np.array(all_b_labels, dtype=np.int32)
    y_multiclass = np.array(all_m_labels, dtype=np.int32)
    sources = np.array(all_sources)

    artifacts = {}

    for task in ["binary", "multiclass"]:
        y_target = y_binary if task == "binary" else y_multiclass
        unique_classes = np.unique(y_target)

        # Split with source awareness
        X_temp, X_test, y_temp, y_test, src_temp, src_test = train_test_split(
            X_raw, y_target, sources, test_size=test_size, random_state=random_seed, stratify=y_target
        )
        adj_val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val, src_train, src_val = train_test_split(
            X_temp, y_temp, src_temp, test_size=adj_val_ratio, random_state=random_seed, stratify=y_temp
        )

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        scaler = RobustScaler()
        X_train_scl = scaler.fit_transform(X_train_imp)
        X_val_scl = scaler.transform(X_val_imp)
        X_test_scl = scaler.transform(X_test_imp)

        np.save(out_dir / f"X_train_{task}.npy", X_train_scl)
        np.save(out_dir / f"X_val_{task}.npy", X_val_scl)
        np.save(out_dir / f"X_test_{task}.npy", X_test_scl)
        np.save(out_dir / f"y_train_{task}.npy", y_train)
        np.save(out_dir / f"y_val_{task}.npy", y_val)
        np.save(out_dir / f"y_test_{task}.npy", y_test)
        np.save(out_dir / f"src_test_{task}.npy", src_test)

        bundle = {
            "imputer": imputer,
            "scaler": scaler,
            "feature_names": UNIFIED_CORE_FEATURES,
            "task_type": task,
            "dataset_key": "unified",
        }
        joblib.dump(bundle, out_dir / f"preprocessor_{task}.joblib")
        joblib.dump(scaler, out_dir / f"scaler_{task}.joblib")

        if task == "binary":
            lbl_map = {"BENIGN": 0, "ATTACK": 1}
            inv_lbl_map = {"0": "BENIGN", "1": "ATTACK"}
        else:
            lbl_map = {MULTICLASS_TAXONOMY[int(c)]: int(c) for c in unique_classes}
            inv_lbl_map = {str(int(c)): MULTICLASS_TAXONOMY[int(c)] for c in unique_classes}

        meta = {
            "dataset_key": "unified",
            "task_type": task,
            "experiment": "Experiment B (Unified Cross-Dataset)",
            "num_features": len(UNIFIED_CORE_FEATURES),
            "feature_names": UNIFIED_CORE_FEATURES,
            "label_mapping": lbl_map,
            "inverse_label_mapping": inv_lbl_map,
            "sources": list(set(all_sources)),
            "train_samples": len(X_train_scl),
            "val_samples": len(X_val_scl),
            "test_samples": len(X_test_scl),
            "class_distribution_train": {int(k): int((y_train == k).sum()) for k in unique_classes},
            "class_distribution_test": {int(k): int((y_test == k).sum()) for k in unique_classes},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_dir / f"metadata_{task}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        artifacts[task] = {
            "train_scaled": out_dir / f"X_train_{task}.npy",
            "val_scaled": out_dir / f"X_val_{task}.npy",
            "test_scaled": out_dir / f"X_test_{task}.npy",
            "y_train": out_dir / f"y_train_{task}.npy",
            "y_val": out_dir / f"y_val_{task}.npy",
            "y_test": out_dir / f"y_test_{task}.npy",
            "scaler": out_dir / f"scaler_{task}.joblib",
            "metadata": out_dir / f"metadata_{task}.json",
        }

    return artifacts


def prepare_all_datasets() -> Dict[str, Any]:
    """Execute complete preprocessing across all individual datasets and unified schema."""
    results = {}
    for ds_key in ["cicids2017", "unsw_nb15", "csecicids2018"]:
        logger.info("=== Preparing Dataset: %s ===", ds_key)
        try:
            results[ds_key] = prepare_dataset_specific(ds_key)
        except Exception as err:
            logger.error("Failed preparing %s: %s", ds_key, err)
            results[ds_key] = {"error": str(err)}

    logger.info("=== Preparing Unified Multi-Dataset Corpus ===")
    try:
        results["unified"] = prepare_unified_dataset()
    except Exception as err:
        logger.error("Failed preparing unified dataset: %s", err)
        results["unified"] = {"error": str(err)}

    return results


def main() -> None:
    """CLI Entrypoint for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare and harmonize IDPS datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "cicids2017", "unsw_nb15", "csecicids2018", "unified"],
        default="all",
        help="Dataset to prepare (default: all)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        res = prepare_all_datasets()
    elif args.dataset == "unified":
        res = {"unified": prepare_unified_dataset()}
    else:
        res = {args.dataset: prepare_dataset_specific(args.dataset)}

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)
    for k, v in res.items():
        if "error" in v:
            print(f"[{k.upper()}] FAILED: {v['error']}")
        else:
            print(f"[{k.upper()}] SUCCESS: Generated splits for {list(v.keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
