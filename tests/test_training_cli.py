"""Tests for training CLI scripts: prepare_data, train_models, evaluate_models."""

import json
from pathlib import Path
import pytest

from training.evaluate_models import evaluate_models_on_test_set
from training.prepare_data import prepare_cicids_dataset
from training.train_models import train_and_register_models


def test_prepare_data_and_train_pipeline(tmp_path):
    """Test full cycle: data preparation -> model training -> evaluation report."""
    raw_csv = Path("data/datasets/cicids2017/raw/cicids2017_multiclass.csv")
    if not raw_csv.exists():
        pytest.skip("Raw multiclass dataset not downloaded yet.")

    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    # 1. Prepare data
    files = prepare_cicids_dataset(
        input_path=raw_csv,
        output_dir=processed_dir,
        task_type="multiclass",
        test_size=0.10,
        val_size=0.10,
        random_seed=42,
    )
    assert files["train_scaled"].exists()
    assert files["test_scaled"].exists()
    assert files["scaler"].exists()

    # 2. Train a lightweight Random Forest and Isolation Forest
    results = train_and_register_models(
        task_type="multiclass",
        models_to_train=["random_forest", "isolation_forest"],
        data_dir=processed_dir,
        output_dir=models_dir,
        random_seed=42,
        db_path=tmp_path / "test.db",
    )
    assert "random_forest" in results
    assert "isolation_forest" in results
    assert results["random_forest"]["metrics"]["accuracy"] > 0.90

    # 3. Evaluate models
    eval_report_file = models_dir / "test_report.json"
    report = evaluate_models_on_test_set(
        task_type="multiclass",
        models_dir=models_dir,
        data_dir=processed_dir,
        output_report_path=eval_report_file,
    )
    assert eval_report_file.exists()
    assert len(report["models"]) >= 1
    rf_eval = next(v for k, v in report["models"].items() if "random_forest" in k)
    assert rf_eval["accuracy"] > 0.90
