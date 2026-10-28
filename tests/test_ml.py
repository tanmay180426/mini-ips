"""Comprehensive unit tests for Machine Learning and Feature Engineering modules.

Tests cover:
- NetworkFlow stateful tracking and flow feature extraction (ml/feature_extraction.py).
- Preprocessing, scaling, imputation, and leakage-free dataset splitting (ml/preprocessing.py).
- Supervised classification model training and inference (ml/train_supervised.py).
- Unsupervised anomaly detection model training and scoring (ml/train_anomaly.py).
- Model registry lifecycle: validation, registration, activation, querying (ml/model_registry.py).
- Explainability and defensive recommendations (ml/explainability.py).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from capture.packet_capture import NormalizedPacket
from ml.feature_extraction import FlowFeatureExtractor, NetworkFlow
from ml.preprocessing import FlowDataPreprocessor
from ml.train_supervised import SupervisedModelTrainer
from ml.train_anomaly import AnomalyDetectorTrainer
from ml.model_registry import ModelRegistry
from ml.explainability import ModelExplainer
from storage.database import DatabaseManager


# ---------------------------------------------------------------------------
# 1. Flow Feature Extraction Tests
# ---------------------------------------------------------------------------


def test_network_flow_bidirectional_tracking():
    """Verify NetworkFlow correctly tracks bidirectional packets and computes duration."""
    flow = NetworkFlow(
        flow_id="test_flow_1",
        src_ip="192.168.1.10",
        dst_ip="10.0.0.1",
        src_port=54321,
        dst_port=80,
        protocol="TCP",
        start_time=100.0,
        end_time=100.0,
    )
    assert flow.total_packets == 0
    assert flow.duration == 0.0

    extractor = FlowFeatureExtractor(flow_timeout=15.0)

    # Forward packet
    p1 = NormalizedPacket(100.0, "192.168.1.10", "10.0.0.1", "TCP", 64, 54321, 80, "S")
    out1 = extractor.process_packet(p1)
    assert len(out1) == 0

    # Backward packet
    p2 = NormalizedPacket(101.5, "10.0.0.1", "192.168.1.10", "TCP", 128, 80, 54321, "SA")
    out2 = extractor.process_packet(p2)
    assert len(out2) == 0

    active = extractor.active_flows
    assert len(active) == 1
    active_flow = list(active.values())[0]

    assert active_flow.total_packets == 2
    assert active_flow.fwd_stats.packet_count == 1
    assert active_flow.bwd_stats.packet_count == 1
    assert active_flow.duration == 1.5

    features = active_flow.extract_features()
    assert features["flow_duration"] == 1.5
    assert features["total_fwd_packets"] == 1
    assert features["total_bwd_packets"] == 1
    assert features["syn_flag_count"] == 2.0


def test_flow_feature_extractor_lifecycle():
    """Verify FlowFeatureExtractor aggregates packets into flows and evicts on timeout."""
    extractor = FlowFeatureExtractor(flow_timeout=5.0)

    p1 = NormalizedPacket(10.0, "192.168.1.5", "8.8.8.8", "UDP", 100, 50000, 53, "")
    p2 = NormalizedPacket(11.0, "8.8.8.8", "192.168.1.5", "UDP", 200, 53, 50000, "")

    f1 = extractor.process_packet(p1)
    assert len(f1) == 0  # Flow remains active

    f2 = extractor.process_packet(p2)
    assert len(f2) == 0  # Flow still within timeout

    # Send a packet long after timeout to trigger eviction
    p3 = NormalizedPacket(25.0, "10.0.0.2", "10.0.0.3", "TCP", 64, 1111, 22, "S")
    evicted = extractor.process_packet(p3)

    # Evicted list contains the timed-out UDP flow
    assert len(evicted) == 1
    assert evicted[0].src_ip == "192.168.1.5"
    assert evicted[0].total_packets == 2

    # Active flow count should now be 1 (for p3)
    assert len(extractor.active_flows) == 1

    # Flush remaining flows
    flushed = extractor.flush()
    assert len(flushed) == 1
    assert len(extractor.active_flows) == 0


# ---------------------------------------------------------------------------
# 2. Preprocessing & Split Leakage Prevention Tests
# ---------------------------------------------------------------------------


def test_flow_data_preprocessor_fit_transform():
    """Verify Preprocessor handles infinity, NaN imputation, and scaling without leakage."""
    np.random.seed(42)
    n = 60
    data = pd.DataFrame({
        "flow_duration": np.concatenate([np.random.uniform(1.0, 50.0, n - 2), [np.inf, np.nan]]),
        "total_fwd_packets": np.concatenate([np.random.randint(1, 20, n - 1), [np.nan]]),
        "destination_port": np.random.randint(1, 65535, n + 1)[:n],
        "label": ["BENIGN"] * 30 + ["PortScan"] * 30,
    })

    preprocessor = FlowDataPreprocessor(target_col="label", task_type="binary", test_size=0.2, val_size=0.2)
    partitions, report = preprocessor.fit_transform(data)

    assert "X_train" in partitions
    assert "X_test" in partitions
    assert "y_train" in partitions
    assert "y_test" in partitions

    X_train = partitions["X_train"]
    assert not np.isnan(X_train).any()
    assert not np.isinf(X_train).any()

    # Preprocessing report verification
    assert report.final_feature_count > 0
    assert report.split_method == "stratified"
    assert report.train_rows == len(partitions["X_train"])


def test_flow_data_preprocessor_train_test_split_no_leakage():
    """Verify train_test_split preserves temporal/stratified partitions and fits scaler on train only."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "flow_duration": np.random.exponential(scale=10.0, size=n),
        "total_fwd_packets": np.random.randint(1, 100, size=n),
        "total_bwd_packets": np.random.randint(0, 100, size=n),
        "label": ["BENIGN"] * 80 + ["DDoS"] * 20,
    })

    preprocessor = FlowDataPreprocessor(
        target_col="label",
        task_type="binary",
        split_method="stratified",
        test_size=0.20,
        val_size=0.10,
    )
    partitions, report = preprocessor.fit_transform(df)

    X_train = partitions["X_train"]
    y_train = partitions["y_train"]
    X_test = partitions["X_test"]
    y_test = partitions["y_test"]

    # Target class distribution preserved in stratified split
    train_positive_ratio = np.mean(y_train == 1)
    test_positive_ratio = np.mean(y_test == 1)
    assert abs(train_positive_ratio - 0.20) < 0.05
    assert abs(test_positive_ratio - 0.20) < 0.05

    # Verify scaler was fitted on train set: mean of scaled train is approx 0
    assert np.allclose(np.mean(X_train, axis=0), 0.0, atol=1e-1)


# ---------------------------------------------------------------------------
# 3. Supervised Model Training Tests
# ---------------------------------------------------------------------------


def test_supervised_model_trainer_random_forest():
    """Verify training Random Forest binary classifier generates legitimate model bundle."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=0.0, scale=1.0, size=(50, 4)),
        np.random.normal(loc=4.0, scale=1.0, size=(50, 4)),
    ])
    y = np.array([0] * 50 + [1] * 50)

    trainer = SupervisedModelTrainer(algorithm="random_forest", task_type="binary")
    model, summary = trainer.train(X, y)

    assert model is not None
    assert summary.algorithm == "random_forest"
    assert summary.task_type == "binary"
    assert summary.feature_count == 4
    assert summary.train_samples == 100

    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == 100
    assert probs.shape == (100, 2)
    # Model should easily separate separated Gaussian clusters
    accuracy = np.mean(preds == y)
    assert accuracy >= 0.95


def test_supervised_model_trainer_multiclass():
    """Verify training multiclass model with class label mapping."""
    np.random.seed(42)
    X = np.random.randn(60, 3)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)

    trainer = SupervisedModelTrainer(algorithm="logistic_regression", task_type="multiclass")
    model, summary = trainer.train(X, y)

    assert summary.task_type == "multiclass"
    preds = model.predict(X)
    assert set(preds).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# 4. Unsupervised Anomaly Detector Tests
# ---------------------------------------------------------------------------


def test_anomaly_detector_trainer_isolation_forest():
    """Verify training Isolation Forest on benign baseline flows produces valid anomaly scores."""
    np.random.seed(42)
    # Normal benign baseline
    X_benign = np.random.normal(loc=0.0, scale=1.0, size=(100, 3))

    trainer = AnomalyDetectorTrainer(contamination=0.05)
    model, summary = trainer.train(X_benign)

    assert summary.algorithm == "isolation_forest"
    assert summary.train_samples == 100

    # Test on benign-like point vs extreme anomalous point
    X_test_normal = np.array([[0.1, -0.2, 0.05]])
    X_test_anomaly = np.array([[100.0, -85.0, 500.0]])

    _, score_normal = trainer.score_samples(X_test_normal)
    _, score_anomaly = trainer.score_samples(X_test_anomaly)

    # Normalized anomaly score: higher score = more anomalous (0.0 to 1.0)
    assert 0.0 <= score_normal[0] <= 1.0
    assert 0.0 <= score_anomaly[0] <= 1.0
    assert score_anomaly[0] > score_normal[0]


# ---------------------------------------------------------------------------
# 5. Model Registry Lifecycle Tests
# ---------------------------------------------------------------------------


def test_model_registry_registration_and_activation(tmp_path: Path):
    """Verify model registry stores bundles, updates SQLite, and manages active models."""
    db_file = tmp_path / "test_reg.db"
    models_dir = tmp_path / "models"
    with DatabaseManager(db_file) as db:
        db.init_db()

    registry = ModelRegistry(models_dir=models_dir, db_path=db_file)

    # Create dummy trained bundle
    X = np.random.randn(20, 2)
    y = np.array([0] * 10 + [1] * 10)
    trainer = SupervisedModelTrainer(algorithm="logistic_regression", task_type="binary")
    model, summary = trainer.train(X, y)
    from sklearn.impute import SimpleImputer
    from ml.preprocessing import FlowDataPreprocessor
    preprocessor = FlowDataPreprocessor()
    preprocessor.feature_names_ = ["feat_a", "feat_b"]
    preprocessor.imputer = SimpleImputer(strategy="median").fit(
        pd.DataFrame(X, columns=preprocessor.feature_names_)
    )

    metrics = {"accuracy": 0.95, "f1_weighted": 0.94, "precision": 0.96, "recall": 0.95}

    # Register model bundle
    meta = registry.save_model_bundle(
        name="test_log_reg",
        version="v1.0.0-test",
        model=model,
        preprocessor=preprocessor,
        model_type="supervised_binary",
        algorithm="logistic_regression",
        dataset_name="CICIDS2017",
        task_type="binary",
        feature_schema=["feat_a", "feat_b"],
        metrics=metrics,
        set_active=True,
    )

    assert meta.version == "v1.0.0-test"
    assert meta.is_active is True

    # Validate saved artifact
    is_valid, msg, val_meta = registry.validate_artifact("v1.0.0-test")
    assert is_valid is True
    assert val_meta["algorithm"] == "logistic_regression"

    # Query active model from registry
    bundle = registry.load_active_bundle(model_type="supervised_binary")
    assert bundle is not None
    loaded_model, loaded_preprocessor, loaded_meta = bundle
    assert loaded_meta["algorithm"] == "logistic_regression"
    assert loaded_preprocessor.transform_inference({"feat_a": 0.0, "feat_b": 0.0}).shape == (1, 2)


# ---------------------------------------------------------------------------
# 6. Model Explainability Tests
# ---------------------------------------------------------------------------


def test_model_explainer_feature_importance_and_attribution():
    """Verify ModelExplainer extracts feature importances and defensive guidance."""
    np.random.seed(42)
    X = np.random.randn(40, 3)
    y = np.array([0] * 20 + [1] * 20)
    feature_names = ["destination_port", "flow_duration", "total_fwd_packets"]

    trainer = SupervisedModelTrainer(algorithm="random_forest", task_type="binary")
    model, _ = trainer.train(X, y)

    importances = ModelExplainer.get_global_feature_importance(model, feature_names)
    assert len(importances) == 3
    assert set(item["feature"] for item in importances) == set(feature_names)

    # Test flow attribution
    sample_flow = {
        "flow_duration": 0.05,
        "total_fwd_packets": 2,
        "total_bwd_packets": 0,
        "average_packet_size": 60.0,
    }
    attribution = ModelExplainer.explain_flow_prediction(sample_flow)
    assert isinstance(attribution, list)
    assert len(attribution) >= 1

    # Defensive recommendation for PortScan
    recs = ModelExplainer.get_defensive_recommendations("PortScan")
    assert isinstance(recs, list)
    assert len(recs) >= 1
    assert any("firewall" in r.lower() or "scan" in r.lower() or "rate" in r.lower() for r in recs)
