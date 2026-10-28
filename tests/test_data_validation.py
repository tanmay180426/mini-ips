"""Tests for Dataset Validation, Integrity Checks, and Data Leakage Audits."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from training.download_datasets import compute_sha256, probe_remote_file
from training.validate_datasets import validate_dataframe, validate_dataset_files


def test_compute_sha256(tmp_path):
    """Verify SHA-256 calculation matches known cryptographic digest."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Mini-IDS cryptographic test payload\n", encoding="utf-8")
    digest = compute_sha256(test_file)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_validate_dataframe_clean():
    """Verify validation passes on clean tabular flow dataframe."""
    df = pd.DataFrame({
        "flow_duration": [1.0, 2.0, 3.5, 4.0],
        "total_fwd_packets": [10, 20, 15, 30],
        "packet_rate": [10.0, 10.0, 4.3, 7.5],
        "Label": ["BENIGN", "BENIGN", "PortScan", "BENIGN"],
    })
    audit = validate_dataframe(df, dataset_key="test_ds")
    assert audit["validation_passed"] is True
    assert audit["total_missing_values"] == 0
    assert len(audit["columns_with_inf"]) == 0
    assert audit["label_column"] == "Label"
    assert audit["label_distribution"]["BENIGN"] == 3
    assert audit["label_distribution"]["PortScan"] == 1


def test_validate_dataframe_detects_leakage():
    """Verify data leakage columns (IPs, timestamps, session IDs) are flagged."""
    df = pd.DataFrame({
        "src_ip": ["192.168.1.10", "192.168.1.20"],
        "dst_ip": ["10.0.0.1", "10.0.0.2"],
        "timestamp": ["2026-10-23 10:00:00", "2026-10-23 10:01:00"],
        "flow_id": ["flow_1", "flow_2"],
        "fwd_pkts": [5, 12],
        "Label": ["BENIGN", "ATTACK"],
    })
    audit = validate_dataframe(df, dataset_key="leakage_ds")
    assert "src_ip" in audit["potential_leakage_columns"]
    assert "dst_ip" in audit["potential_leakage_columns"]
    assert "timestamp" in audit["potential_leakage_columns"]
    assert "flow_id" in audit["potential_leakage_columns"]


def test_validate_dataframe_detects_anomalies():
    """Verify missing, infinite, constant, and duplicate records are accurately recorded."""
    df = pd.DataFrame({
        "duration": [1.0, np.nan, 3.0, 3.0],
        "rate": [10.0, np.inf, 5.0, 5.0],
        "const_col": [42, 42, 42, 42],
        "Label": ["BENIGN", "ATTACK", "ATTACK", "ATTACK"],
    })
    audit = validate_dataframe(df, dataset_key="anomaly_ds")
    assert audit["total_missing_values"] == 1
    assert "rate" in audit["columns_with_inf"]
    assert "const_col" in audit["constant_columns"]
    assert audit["duplicate_rows"] == 1


def test_probe_remote_file_invalid_endpoint():
    """Ensure probing invalid URL returns graceful failure without throwing unhandled exceptions."""
    is_avail, size, err = probe_remote_file("http://invalid-nonexistent-domain-test.org/test.csv", timeout=2.0)
    assert is_avail is False
    assert size == -1
    assert len(err) > 0
