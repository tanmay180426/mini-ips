"""Tests for Model Training, Estimator Algorithms, and Persistence."""

from pathlib import Path
import joblib
import numpy as np
import pytest

from training.evaluate_models import evaluate_single_model
from training.train_models import ALL_ALGORITHMS, train_and_register_models


def test_train_and_register_algorithms(tmp_path):
    """Verify training of all 5 supported estimators on a minimal fixture."""
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Synthetic binary dataset fixture
    np.random.seed(42)
    n_samples = 200
    n_feats = 11
    X = np.random.randn(n_samples, n_feats).astype(np.float32)
    # Signal: sum of first 3 features > 0 is attack
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(np.int32)

    # Splits
    np.save(data_dir / "X_train_binary.npy", X[:140])
    np.save(data_dir / "y_train_binary.npy", y[:140])
    np.save(data_dir / "X_val_binary.npy", X[140:170])
    np.save(data_dir / "y_val_binary.npy", y[140:170])
    np.save(data_dir / "X_test_binary.npy", X[170:])
    np.save(data_dir / "y_test_binary.npy", y[170:])

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler().fit(X[:140])
    joblib.dump(scaler, data_dir / "scaler_binary.joblib")

    import json
    meta = {
        "dataset_key": "test_ds",
        "task_type": "binary",
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
        "inverse_label_mapping": {"0": "BENIGN", "1": "ATTACK"},
        "feature_names": [f"f_{i}" for i in range(n_feats)],
    }
    with open(data_dir / "metadata_binary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    results = train_and_register_models(
        task_type="binary",
        models_to_train=ALL_ALGORITHMS,
        data_dir=data_dir,
        output_dir=models_dir,
        dataset_key="test_ds",
        db_path=tmp_path / "test.db",
    )

    for algo in ALL_ALGORITHMS:
        assert algo in results
        assert results[algo]["metrics"]["accuracy"] >= 0.0

    # Verify persisted joblibs
    sup_files = list((models_dir / "supervised").glob("*.joblib"))
    anom_files = list((models_dir / "anomaly").glob("*.joblib"))
    assert len(sup_files) == 4  # RF, ET, LR, HGB
    assert len(anom_files) == 1  # Isolation Forest


def test_evaluate_single_model_metrics():
    """Verify evaluation metric calculations produce mathematically coherent scores."""
    from sklearn.ensemble import RandomForestClassifier

    X_train = np.array([[1, 2], [2, 3], [10, 12], [11, 13]], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1], dtype=np.int32)
    X_test = np.array([[1.5, 2.5], [10.5, 12.5]], dtype=np.float32)
    y_test = np.array([0, 1], dtype=np.int32)

    rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    metrics = evaluate_single_model(
        model=rf,
        X_test=X_test,
        y_test=y_test,
        inv_map={0: "BENIGN", 1: "ATTACK"},
        benign_idx=0,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0
    assert "BENIGN" in metrics["per_class"]
    assert "ATTACK" in metrics["per_class"]
