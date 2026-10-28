"""Tests for Cross-Dataset Generalization and Domain Adaptation."""

import json
from pathlib import Path
import numpy as np
import pytest

from training.cross_dataset_evaluation import fit_and_evaluate_transfer


def test_fit_and_evaluate_transfer():
    """Verify zero-leakage transfer training and evaluation pipeline."""
    np.random.seed(42)
    # Source domain A
    X_train_a = np.random.normal(loc=0.0, scale=1.0, size=(100, 11)).astype(np.float32)
    y_train_a = (X_train_a[:, 0] > 0).astype(np.int32)

    # Target domain B (with mean shift to test generalization)
    X_test_b = np.random.normal(loc=0.5, scale=1.2, size=(50, 11)).astype(np.float32)
    y_test_b = (X_test_b[:, 0] > 0).astype(np.int32)

    results = fit_and_evaluate_transfer(
        X_train=X_train_a,
        y_train=y_train_a,
        X_test=X_test_b,
        y_test=y_test_b,
        model_type="rf",
        random_seed=42,
    )

    assert "accuracy" in results
    assert "f1_macro" in results
    assert "false_positive_rate" in results
    assert results["accuracy"] > 0.60
    assert results["fit_duration_s"] >= 0.0


def test_cross_dataset_report_integrity():
    """Ensure precomputed cross-dataset benchmark report contains expected structure."""
    rep_file = Path("reports/metrics/cross_dataset_results.json")
    if not rep_file.exists():
        pytest.skip("Cross dataset report not generated yet.")

    with open(rep_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "experiments" in data
    assert len(data["experiments"]) >= 5
    for exp in data["experiments"]:
        assert "name" in exp
        assert "train_sources" in exp
        assert "test_source" in exp
        assert "metrics" in exp
        assert "accuracy" in exp["metrics"]
        assert "f1_macro" in exp["metrics"]
