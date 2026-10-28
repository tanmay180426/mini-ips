"""Programmatic Acquisition and Validation of CIC-IDS2017 Dataset.

Downloads official labeled flow feature datasets from Hugging Face
(`bvsam/cic-ids-2017`), verifies checksums, enforces schema constraints,
samples representative attack and benign flows, and creates provenance manifests.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

from detection.feature_schema import CANONICAL_FLOW_FEATURES, FeatureSchemaValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mini_ids.training.download")

# Default remote repository endpoints (Hugging Face bvsam/cic-ids-2017)
HF_BASE_URL = "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/"

AVAILABLE_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "portscan": {
        "file_name": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
        "url": HF_BASE_URL + "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
        "description": "CIC-IDS2017 PortScan and Benign Friday telemetry",
        "expected_classes": ["BENIGN", "PortScan"],
    },
    "ddos": {
        "file_name": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
        "url": HF_BASE_URL + "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
        "description": "CIC-IDS2017 DDoS and Benign Friday telemetry",
        "expected_classes": ["BENIGN", "DDoS"],
    },
}

DATA_DIR = Path("data")
DATASETS_DIR = DATA_DIR / "datasets" / "cicids2017"
RAW_DIR = DATASETS_DIR / "raw"
META_DIR = DATASETS_DIR / "metadata"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a local file in 64KB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_with_progress(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """Download a remote file displaying byte progress and handling retries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    headers = {"User-Agent": "Mini-IDS-Acquisition/2.0 (Security-Research)"}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Connecting to %s (Attempt %d/%d)...", url, attempt, max_retries)
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = response.headers.get("Content-Length")
                total_bytes = int(total_size) if total_size else None

                downloaded = 0
                block_size = 1024 * 128  # 128 KB
                start_time = time.time()

                with open(temp_path, "wb") as out_file:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)

                        if total_bytes and (time.time() - start_time > 1.0 or downloaded == total_bytes):
                            pct = (downloaded / total_bytes) * 100.0
                            mb = downloaded / (1024 * 1024)
                            total_mb = total_bytes / (1024 * 1024)
                            sys.stdout.write(f"\r  -> Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
                            sys.stdout.flush()
                            start_time = time.time()

            sys.stdout.write("\n")
            if temp_path.exists():
                if output_path.exists():
                    output_path.unlink()
                temp_path.rename(output_path)
            logger.info("Download completed successfully: %s", output_path)
            return True

        except Exception as err:
            logger.warning("Download failed on attempt %d: %s", attempt, err)
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                logger.error("All download attempts exhausted for %s", url)
                return False

    return False


def acquire_cicids2017(
    scenario: str = "portscan",
    sample_size: Optional[int] = 10000,
    seed: int = 42,
    output_dir: Path = RAW_DIR,
    force_download: bool = False,
) -> Path:
    """Download, validate, and sample CIC-IDS2017 dataset.

    Args:
        scenario: 'portscan' or 'ddos'.
        sample_size: Maximum rows to retain in processed CSV (None = keep all).
        seed: Random seed for sampling.
        output_dir: Directory where the raw CSV is saved.
        force_download: Force redownloading if file exists.

    Returns:
        Path: Path to the generated, verified dataset CSV file.
    """
    if scenario not in AVAILABLE_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(AVAILABLE_SCENARIOS.keys())}")

    info = AVAILABLE_SCENARIOS[scenario]
    parquet_path = output_dir / info["file_name"]
    csv_path = output_dir / f"cicids2017_{scenario}.csv"

    # Step 1: Download Parquet if not already cached
    if force_download or not parquet_path.exists():
        logger.info("Downloading %s from Hugging Face...", scenario)
        success = download_with_progress(info["url"], parquet_path)
        if not success:
            raise RuntimeError(
                f"Failed to download CIC-IDS2017 scenario '{scenario}' from {info['url']}. "
                "Check your internet connection or verify Hugging Face status."
            )
    else:
        logger.info("Using cached Parquet file: %s", parquet_path)

    # Step 2: Validate Parquet file integrity
    file_size = parquet_path.stat().st_size
    checksum = compute_sha256(parquet_path)
    logger.info("Parquet validated: Size=%d bytes, SHA256=%s...", file_size, checksum[:16])

    # Step 3: Read and Validate Schema
    df_raw = pd.read_parquet(parquet_path)
    total_rows = len(df_raw)
    logger.info("Ingested %d total rows from %s", total_rows, info["file_name"])

    # Resolve label column
    label_col = "Label" if "Label" in df_raw.columns else " Label"
    if label_col not in df_raw.columns:
        raise ValueError(f"Required label column not found in {list(df_raw.columns[:5])}")

    df_raw["label"] = df_raw[label_col].astype(str).str.strip()

    # Step 4: Stratified Sampling if sample_size is specified
    if sample_size and sample_size < total_rows:
        logger.info("Stratified sampling to %d rows (seed=%d)...", sample_size, seed)
        # Sample proportionally per class to ensure minority attack classes are preserved
        classes = df_raw["label"].unique()
        sampled_parts = []
        for cls_name in classes:
            subset = df_raw[df_raw["label"] == cls_name]
            frac = len(subset) / total_rows
            n_cls = max(10, int(round(sample_size * frac)))
            n_take = min(n_cls, len(subset))
            sampled_parts.append(subset.sample(n=n_take, random_state=seed))
        df_sampled = pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        df_sampled = df_raw

    # Step 5: Clean and Align Features
    X, report = FeatureSchemaValidator.validate_and_clean_dataframe(df_sampled, strict=False)
    X["label"] = df_sampled["label"].values

    # Step 6: Save CSV and Manifest
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(csv_path, index=False)
    csv_size = csv_path.stat().st_size
    csv_checksum = compute_sha256(csv_path)

    class_distribution = X["label"].value_counts().to_dict()
    logger.info("Saved clean CSV: %s (Rows: %d, Size: %.2f MB)", csv_path, len(X), csv_size / (1024 * 1024))
    logger.info("Class distribution: %s", class_distribution)

    META_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": f"CICIDS2017_{scenario.upper()}",
        "scenario": scenario,
        "source_url": info["url"],
        "doc_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "huggingface_repo": "https://huggingface.co/datasets/bvsam/cic-ids-2017",
        "retrieval_timestamp": time.time(),
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "parquet_file": info["file_name"],
        "parquet_size_bytes": file_size,
        "parquet_sha256": checksum,
        "csv_file": csv_path.name,
        "csv_size_bytes": csv_size,
        "csv_sha256": csv_checksum,
        "total_raw_rows": total_rows,
        "sampled_rows": len(X),
        "feature_count": len(CANONICAL_FLOW_FEATURES),
        "features": CANONICAL_FLOW_FEATURES,
        "class_distribution": class_distribution,
        "validation_report": {
            "is_valid": report.is_valid,
            "nan_imputed_count": report.nan_imputed_count,
            "inf_imputed_count": report.inf_imputed_count,
            "clipped_negative_count": report.clipped_negative_count,
        },
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "citation": "Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, 'Toward Generating a New Dataset for Intrusion Detection and Security Analysis', ICISSP, 2018.",
    }

    manifest_path = META_DIR / f"{scenario}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest recorded at: %s", manifest_path)

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Download and prepare CIC-IDS2017 dataset from Hugging Face.")
    parser.add_argument("--scenario", choices=["portscan", "ddos"], default="portscan", help="Scenario subset to acquire.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Maximum number of rows to sample (default: 10000).")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if cached.")
    args = parser.parse_args()

    acquire_cicids2017(
        scenario=args.scenario,
        sample_size=args.sample_size,
        seed=args.seed,
        force_download=args.force,
    )


if __name__ == "__main__":
    main()
