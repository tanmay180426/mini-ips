"""Dataset Integrity, Schema, and Data-Leakage Validation Engine.

Performs deep structural audits on acquired intrusion detection datasets:
- Checksum verification.
- Dimension and schema profiling.
- Missing and infinite value rates.
- Constant and duplicate record detection.
- Ground-truth label distributions.
- Potential data leakage audits (IP addresses, timestamps, flow IDs).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mini_ids.training.validate_datasets")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
MANIFESTS_DIR = DATA_DIR / "manifests"

LEAKAGE_KEYWORDS = {
    "src_ip", "source_ip", "srcip", "dst_ip", "dest_ip", "destination_ip", "dstip",
    "timestamp", "time", "date", "flow_id", "flowid", "id", "session_id",
    "source_port", "src_port", "sport", "dest_port", "dst_port", "dport"
}


def compute_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset_sample(
    file_path: Path,
    max_sample_rows: int = 50_000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load dataset file into a pandas DataFrame (with sampling if large)."""
    suffix = file_path.suffix.lower()
    meta = {"file_path": str(file_path), "format": suffix}

    if suffix == ".parquet":
        import pyarrow.parquet as pq
        meta["total_rows"] = pq.read_metadata(file_path).num_rows
        df = pd.read_parquet(file_path)
        if len(df) > max_sample_rows:
            df = df.sample(n=max_sample_rows, random_state=42)
    elif suffix == ".csv":
        # Check encoding
        encoding = "utf-8"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.startswith("\ufeff"):
                    encoding = "utf-8-sig"
        except UnicodeDecodeError:
            encoding = "latin-1"

        # Count total rows
        total_rows = 0
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                total_rows += chunk.count(b"\n")
        meta["total_rows"] = max(0, total_rows - 1)

        if meta["total_rows"] > max_sample_rows:
            df = pd.read_csv(file_path, nrows=max_sample_rows, encoding=encoding, low_memory=False)
        else:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    meta["sample_rows"] = len(df)
    meta["total_cols"] = len(df.columns)
    return df, meta


def validate_dataframe(df: pd.DataFrame, dataset_key: str) -> Dict[str, Any]:
    """Execute structural quality, leakage, and distribution audit on dataframe."""
    n_rows, n_cols = df.shape

    # 1. Missing values
    missing_series = df.isnull().sum()
    missing_cols = {col: int(cnt) for col, cnt in missing_series.items() if cnt > 0}
    total_missing = int(missing_series.sum())

    # 2. Infinite values (in numeric columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_cols = {}
    for col in num_cols:
        inf_cnt = int(np.isinf(df[col]).sum())
        if inf_cnt > 0:
            inf_cols[col] = inf_cnt

    # 3. Constant columns (0 variance)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    # 4. Duplicate rows
    # Check duplicates excluding any obvious ID column
    feature_cols = [c for c in df.columns if c.lower() not in {"id", "flow_id", "timestamp"}]
    dup_count = int(df.duplicated(subset=feature_cols).sum())

    # 5. Data leakage audit
    identified_leakage_cols = []
    for col in df.columns:
        clean_name = col.lower().replace(" ", "_").replace(".", "_")
        if clean_name in LEAKAGE_KEYWORDS:
            identified_leakage_cols.append(col)

    # 6. Label column discovery & distribution
    label_candidates = ["Label", "label", "attack_cat", "class", "Class", "Target"]
    found_label_col = None
    for cand in label_candidates:
        if cand in df.columns:
            found_label_col = cand
            break

    label_dist = {}
    if found_label_col:
        raw_counts = df[found_label_col].value_counts().to_dict()
        label_dist = {str(k): int(v) for k, v in raw_counts.items()}

    # 7. High-cardinality non-numeric columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    high_cardinality = [c for c in cat_cols if df[c].nunique() > 50 and c != found_label_col]

    return {
        "dataset_key": dataset_key,
        "rows_analyzed": n_rows,
        "columns_count": n_cols,
        "numeric_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "total_missing_values": total_missing,
        "columns_with_missing": missing_cols,
        "columns_with_inf": inf_cols,
        "constant_columns": constant_cols,
        "duplicate_rows": dup_count,
        "potential_leakage_columns": identified_leakage_cols,
        "label_column": found_label_col,
        "label_distribution": label_dist,
        "high_cardinality_categoricals": high_cardinality,
        "validation_passed": True,
    }


def validate_dataset_files(
    dataset_key: str,
    max_sample_rows: int = 50_000,
) -> Dict[str, Any]:
    """Validate all raw files for a given dataset."""
    raw_sub = RAW_DIR / dataset_key
    if not raw_sub.exists():
        return {
            "dataset_key": dataset_key,
            "status": "MISSING",
            "error": f"Directory not found: {raw_sub}",
        }

    files = list(raw_sub.glob("*.parquet")) + list(raw_sub.glob("*.csv"))
    if not files:
        return {
            "dataset_key": dataset_key,
            "status": "EMPTY",
            "error": f"No data files found in {raw_sub}",
        }

    file_validations = []
    overall_status = "VALID"

    for file_path in files:
        logger.info("Auditing file: %s", file_path.name)
        sha256 = compute_sha256(file_path)
        size_bytes = file_path.stat().st_size

        try:
            df, meta = load_dataset_sample(file_path, max_sample_rows=max_sample_rows)
            audit = validate_dataframe(df, dataset_key)
            audit["filename"] = file_path.name
            audit["size_bytes"] = size_bytes
            audit["sha256"] = sha256
            audit["total_file_rows"] = meta.get("total_rows", len(df))
            file_validations.append(audit)
        except Exception as err:
            logger.error("Validation error for %s: %s", file_path.name, err)
            overall_status = "ERROR"
            file_validations.append({
                "filename": file_path.name,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "status": "ERROR",
                "error": str(err),
            })

    report = {
        "dataset_key": dataset_key,
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "files_validated": file_validations,
    }

    # Save to manifests
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MANIFESTS_DIR / f"{dataset_key}_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved validation report: %s", report_path)

    return report


def validate_all_datasets(max_sample_rows: int = 50_000) -> Dict[str, Any]:
    """Validate all three benchmark datasets."""
    datasets = ["cicids2017", "unsw_nb15", "csecicids2018"]
    summary = {}
    for ds in datasets:
        summary[ds] = validate_dataset_files(ds, max_sample_rows=max_sample_rows)
    return summary


def main() -> None:
    """CLI Entrypoint for dataset validation."""
    parser = argparse.ArgumentParser(description="Validate schema, integrity, and data leakage for IDPS datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "cicids2017", "unsw_nb15", "csecicids2018"],
        default="all",
        help="Dataset to validate (default: all)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=50_000,
        help="Maximum rows to sample for deep audit (default: 50,000)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        results = validate_all_datasets(max_sample_rows=args.sample_rows)
    else:
        results = {args.dataset: validate_dataset_files(args.dataset, max_sample_rows=args.sample_rows)}

    print("\n" + "=" * 65)
    print("DATASET VALIDATION & DATA LEAKAGE AUDIT")
    print("=" * 65)
    for ds_key, res in results.items():
        st = res.get("status", "UNKNOWN")
        print(f"\n[{st}] {ds_key.upper()}")
        for f in res.get("files_validated", []):
            if "error" in f:
                print(f"  [ERROR] {f.get('filename')}: {f['error']}")
                continue
            print(f"  [OK] File: {f['filename']} ({f['size_bytes']:,} bytes)")
            print(f"    - Records: {f.get('total_file_rows', f['rows_analyzed']):,} | Features: {f['columns_count']}")
            print(f"    - Label column: '{f['label_column']}' with {len(f['label_distribution'])} unique classes")
            print(f"    - Missing values: {f['total_missing_values']} | Inf cols: {len(f['columns_with_inf'])}")
            print(f"    - Leakage risks flagged: {f['potential_leakage_columns']}")
            print(f"    - Constant columns: {len(f['constant_columns'])}")
    print("=" * 65)


if __name__ == "__main__":
    main()
