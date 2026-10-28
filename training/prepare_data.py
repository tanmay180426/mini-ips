"""Data Preparation and Preprocessing Pipeline for Mini-IDS.

Loads raw CIC-IDS2017 benchmark dataset files, normalizes columns to the canonical
28-feature schema, performs leakage-free stratified splitting, fits preprocessors,
and saves the train/val/test splits and fitted preprocessor artifacts.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Ensure parent directory in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.feature_schema import (
    CANONICAL_FLOW_FEATURES,
    extract_canonical_features,
    validate_feature_dataframe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_cicids_dataset(
    input_path: Path,
    output_dir: Path,
    task_type: str = "multiclass",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Path]:
    """Load, harmonize, split, and persist processed dataset splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reading raw dataset: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Raw shape: %s", df.shape)

    # 1. Clean and normalize columns to canonical feature schema
    X_clean, y_raw = extract_canonical_features(df, target_col="label")
    report = validate_feature_dataframe(X_clean)
    logger.info("Schema validation: %s (features: %d, rows: %d)",
                report.is_valid, report.total_features, len(X_clean))

    # 2. Encode labels
    y_str = y_raw.astype(str).str.strip()
    if task_type == "binary":
        y = np.where(y_str.str.upper() == "BENIGN", 0, 1)
        label_map = {"BENIGN": 0, "ATTACK": 1}
        inv_map = {0: "BENIGN", 1: "ATTACK"}
    else:
        # Multiclass: preserve distinct attack categories
        unique_labels = sorted(y_str.unique(), key=lambda x: (x != "BENIGN", x))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        inv_map = {idx: label for label, idx in label_map.items()}
        y = np.array([label_map[item] for item in y_str])

    logger.info("Class distribution: %s", {k: int((y == v).sum()) for k, v in label_map.items()})

    # 3. Leakage-free split: Train, Validation, Test
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_clean, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    # Split remaining into train and validation
    adjusted_val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_ratio, random_state=random_seed, stratify=y_temp
    )

    logger.info("Split sizes — Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # 4. Fit scaler strictly on X_train only (zero data leakage)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Persist splits and artifacts
    files = {
        "train_scaled": output_dir / f"X_train_{task_type}.npy",
        "val_scaled": output_dir / f"X_val_{task_type}.npy",
        "test_scaled": output_dir / f"X_test_{task_type}.npy",
        "y_train": output_dir / f"y_train_{task_type}.npy",
        "y_val": output_dir / f"y_val_{task_type}.npy",
        "y_test": output_dir / f"y_test_{task_type}.npy",
        "scaler": output_dir / f"scaler_{task_type}.joblib",
        "metadata": output_dir / f"metadata_{task_type}.json",
    }

    np.save(files["train_scaled"], X_train_scaled)
    np.save(files["val_scaled"], X_val_scaled)
    np.save(files["test_scaled"], X_test_scaled)
    np.save(files["y_train"], y_train)
    np.save(files["y_val"], y_val)
    np.save(files["y_test"], y_test)
    joblib.dump(scaler, files["scaler"])

    # Save summary metadata
    meta = {
        "source_dataset": str(input_path),
        "task_type": task_type,
        "feature_names": CANONICAL_FLOW_FEATURES,
        "feature_count": len(CANONICAL_FLOW_FEATURES),
        "label_mapping": label_map,
        "inverse_label_mapping": {str(k): v for k, v in inv_map.items()},
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "scaler_type": "RobustScaler",
    }
    with open(files["metadata"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Successfully prepared and saved all splits to %s", output_dir)
    return files


def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess Mini-IDS training datasets.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/datasets/cicids2017/raw/cicids2017_multiclass.csv",
        help="Path to raw CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets/cicids2017/processed",
        help="Directory to save preprocessed splits",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["binary", "multiclass"],
        default="multiclass",
        help="Target classification task type",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        logger.error("Dataset not found: %s. Run python training/download_dataset.py first.", input_file)
        sys.exit(1)

    prepare_cicids_dataset(
        input_path=input_file,
        output_dir=Path(args.output_dir),
        task_type=args.task_type,
    )


if __name__ == "__main__":
    main()
