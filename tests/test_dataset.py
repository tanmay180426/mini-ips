"""Unit tests for Dataset Acquisition, Manifest Tracking, and Sample Generation.

Tests cover:
- Directory structuring and path discovery (ml/dataset_manager.py).
- SHA256 checksum computation and validation.
- Acquisition manifest recording and JSON deserialization.
- Deterministic synthetic sample fixture generation with multi-class distributions.
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from ml.dataset_manager import DatasetManager, DatasetManifest, DATASET_METADATA


def test_dataset_manager_directory_structure(tmp_path: Path):
    """Verify DatasetManager sets up raw, processed, and metadata subdirectories."""
    manager = DatasetManager(base_dir=tmp_path)
    paths = manager.get_dataset_paths("cicids2017")

    assert paths["root"].is_dir()
    assert paths["raw"].is_dir()
    assert paths["processed"].is_dir()
    assert paths["metadata"].is_dir()


def test_dataset_manager_sha256_computation(tmp_path: Path):
    """Verify SHA256 calculation matches known cryptographic digest."""
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"Mini-IDS Machine Learning Test Payload")

    manager = DatasetManager(base_dir=tmp_path)
    digest = manager.compute_sha256(test_file)

    assert isinstance(digest, str)
    assert len(digest) == 64
    # Known sha256 for this exact byte sequence
    import hashlib
    expected = hashlib.sha256(b"Mini-IDS Machine Learning Test Payload").hexdigest()
    assert digest == expected


def test_dataset_manager_manifest_lifecycle(tmp_path: Path):
    """Verify writing and reading JSON acquisition manifests."""
    manager = DatasetManager(base_dir=tmp_path)
    paths = manager.get_dataset_paths("cicids2017")

    manifest = DatasetManifest(
        dataset_name="CICIDS2017",
        original_source="http://mock.source/data.csv",
        retrieval_timestamp=1700000000.0,
        file_name="mock_flows.csv",
        file_size_bytes=1024,
        sha256_checksum="abcdef1234567890" * 4,
        verification_status="VERIFIED",
        doc_reference="https://www.unb.ca/cic/datasets/ids-2017.html",
        processing_status="RAW",
    )

    manifest_file = manager.record_manifest(manifest, paths["metadata"])
    assert manifest_file.is_file()

    loaded = manager.load_manifests("cicids2017")
    assert len(loaded) == 1
    assert loaded[0]["dataset_name"] == "CICIDS2017"
    assert loaded[0]["file_name"] == "mock_flows.csv"
    assert loaded[0]["verification_status"] == "VERIFIED"


def test_dataset_manager_reproducible_sample_generation(tmp_path: Path):
    """Verify deterministic sample generation with valid schema and label proportions."""
    manager = DatasetManager(base_dir=tmp_path)
    sample_path = manager.generate_reproducible_sample_dataset(
        dataset_name="cicids2017",
        n_samples=200,
        random_seed=42,
    )

    assert sample_path.is_file()
    df = pd.read_csv(sample_path)
    assert len(df) == 200

    # Required column presence
    assert "Flow Duration" in df.columns
    assert "Total Fwd Packets" in df.columns
    assert "Label" in df.columns

    # Verify label distributions
    counts = df["Label"].value_counts().to_dict()
    assert "BENIGN" in counts
    assert "PortScan" in counts
    assert "DoS" in counts
    assert counts["BENIGN"] > counts["PortScan"]
