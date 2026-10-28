"""Stage 3: Genuine ML Artifact Activation and Inference Validation Tests."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict
import joblib, numpy as np, pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
RF_BINARY_DIR = WORKSPACE_ROOT / "data" / "models" / "random_forest_binary_1792944312"
IF_BINARY_DIR = WORKSPACE_ROOT / "data" / "models" / "isolation_forest_binary_1792944313"
RF_MULTICLASS_DIR = WORKSPACE_ROOT / "data" / "models" / "random_forest_multiclass_1792945214"


def isolated_registry(tmp_path: Path, version: str):
    """Point the supported registry API at a temporary SQLite registry."""
    from ml.model_registry import ModelRegistry
    from storage.database import DatabaseManager

    artifact_dir = WORKSPACE_ROOT / "data" / "models" / version
    meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    db_path = tmp_path / f"{version}.db"
    with DatabaseManager(db_path) as db:
        db.init_db()
        db.insert_model({
            "name": meta.get("name", version),
            "version": version,
            "model_type": meta["model_type"],
            "algorithm": meta.get("algorithm", "unknown"),
            "dataset_name": meta.get("dataset_name", "unknown"),
            "task_type": meta.get("task_type", "unknown"),
            "metrics": meta.get("metrics", {}),
            "feature_schema": meta.get("feature_schema", []),
            "artifact_path": str(artifact_dir),
            "created_at": meta.get("created_at", 0.0),
            "is_active": True,
        })
    return ModelRegistry(models_dir=WORKSPACE_ROOT / "data" / "models", db_path=db_path), db_path

BENIGN_FLOW: Dict[str, float] = {
    "flow_duration": 5.0, "total_fwd_packets": 3.0, "total_bwd_packets": 3.0,
    "total_length_of_fwd_packets": 300.0, "total_length_of_bwd_packets": 400.0,
    "fwd_packet_length_mean": 100.0, "fwd_packet_length_std": 5.0,
    "fwd_packet_length_min": 80.0, "fwd_packet_length_max": 120.0,
    "bwd_packet_length_mean": 133.0, "bwd_packet_length_std": 6.0,
    "bwd_packet_length_min": 100.0, "bwd_packet_length_max": 180.0,
    "flow_bytes_s": 140.0, "flow_packets_s": 1.2,
    "flow_iat_mean": 0.8, "flow_iat_std": 0.1, "flow_iat_min": 0.5, "flow_iat_max": 1.2,
    "fwd_iat_mean": 1.6, "fwd_iat_std": 0.2, "bwd_iat_mean": 1.7, "bwd_iat_std": 0.1,
    "fin_flag_count": 1.0, "syn_flag_count": 1.0, "rst_flag_count": 0.0,
    "psh_flag_count": 2.0, "ack_flag_count": 6.0, "down_up_ratio": 1.0, "average_packet_size": 116.0,
}
ATTACK_FLOW: Dict[str, float] = {
    "flow_duration": 0.001, "total_fwd_packets": 1000.0, "total_bwd_packets": 0.0,
    "total_length_of_fwd_packets": 40000.0, "total_length_of_bwd_packets": 0.0,
    "fwd_packet_length_mean": 40.0, "fwd_packet_length_std": 0.0,
    "fwd_packet_length_min": 40.0, "fwd_packet_length_max": 40.0,
    "bwd_packet_length_mean": 0.0, "bwd_packet_length_std": 0.0,
    "bwd_packet_length_min": 0.0, "bwd_packet_length_max": 0.0,
    "flow_bytes_s": 40_000_000.0, "flow_packets_s": 1_000_000.0,
    "flow_iat_mean": 1e-6, "flow_iat_std": 0.0, "flow_iat_min": 0.0, "flow_iat_max": 2e-6,
    "fwd_iat_mean": 1e-6, "fwd_iat_std": 0.0, "bwd_iat_mean": 0.0, "bwd_iat_std": 0.0,
    "fin_flag_count": 0.0, "syn_flag_count": 1000.0, "rst_flag_count": 0.0,
    "psh_flag_count": 0.0, "ack_flag_count": 0.0, "down_up_ratio": 0.0, "average_packet_size": 40.0,
}

@pytest.fixture
def rf_binary_bundle():
    pytest.importorskip("joblib")
    if not (RF_BINARY_DIR / "model.joblib").exists():
        pytest.skip("RF binary artifact missing")
    return joblib.load(RF_BINARY_DIR / "model.joblib"), joblib.load(RF_BINARY_DIR / "preprocessor.joblib")

@pytest.fixture
def if_binary_bundle():
    pytest.importorskip("joblib")
    if not (IF_BINARY_DIR / "model.joblib").exists():
        pytest.skip("IF binary artifact missing")
    return joblib.load(IF_BINARY_DIR / "model.joblib"), joblib.load(IF_BINARY_DIR / "preprocessor.joblib")

@pytest.fixture
def rf_multiclass_bundle():
    pytest.importorskip("joblib")
    if not (RF_MULTICLASS_DIR / "model.joblib").exists():
        pytest.skip("RF multiclass artifact missing")
    return joblib.load(RF_MULTICLASS_DIR / "model.joblib"), joblib.load(RF_MULTICLASS_DIR / "preprocessor.joblib")

def test_rf_binary_artifact_exists():
    assert (RF_BINARY_DIR / "model.joblib").is_file()
    assert (RF_BINARY_DIR / "preprocessor.joblib").is_file()
    assert (RF_BINARY_DIR / "metadata.json").is_file()

def test_if_binary_artifact_exists():
    assert (IF_BINARY_DIR / "model.joblib").is_file()
    assert (IF_BINARY_DIR / "preprocessor.joblib").is_file()
    assert (IF_BINARY_DIR / "metadata.json").is_file()

def test_rf_multiclass_artifact_exists():
    assert (RF_MULTICLASS_DIR / "model.joblib").is_file()
    assert (RF_MULTICLASS_DIR / "preprocessor.joblib").is_file()
    assert (RF_MULTICLASS_DIR / "metadata.json").is_file()

def test_registry_validates_rf_binary():
    import sys; sys.path.insert(0, str(WORKSPACE_ROOT))
    from ml.model_registry import ModelRegistry
    ok, msg, meta = ModelRegistry().validate_artifact("random_forest_binary_1792944312")
    assert ok, f"validate_artifact failed: {msg}"
    assert len(meta.get("feature_schema", [])) == 28

def test_registry_validates_if_binary():
    import sys; sys.path.insert(0, str(WORKSPACE_ROOT))
    from ml.model_registry import ModelRegistry
    ok, msg, meta = ModelRegistry().validate_artifact("isolation_forest_binary_1792944313")
    assert ok, f"validate_artifact failed: {msg}"
    assert meta.get("task_type") == "anomaly"

def test_registry_validate_missing_artifact():
    import sys; sys.path.insert(0, str(WORKSPACE_ROOT))
    from ml.model_registry import ModelRegistry
    ok, msg, meta = ModelRegistry().validate_artifact("nonexistent_version_99999")
    assert ok is False and meta is None

def test_isolated_registry_loads_real_supervised_binary(tmp_path):
    registry, db_path = isolated_registry(tmp_path, "random_forest_binary_1792944312")
    before = db_path.stat().st_mtime_ns
    result = registry.load_active_bundle("supervised_binary")
    assert result is not None
    model, prep, meta = result
    assert hasattr(model, "predict") and hasattr(prep, "transform_inference")
    assert meta.get("model_type") == "supervised_binary"
    assert db_path.stat().st_mtime_ns == before

def test_artifact_fallback_respects_requested_model_type(tmp_path):
    from ml.model_registry import ModelRegistry
    from storage.database import DatabaseManager

    db_path = tmp_path / "empty_registry.db"
    with DatabaseManager(db_path) as db:
        db.init_db()
    registry = ModelRegistry(models_dir=WORKSPACE_ROOT / "data" / "models", db_path=db_path)
    result = registry.load_active_bundle("supervised_binary")
    assert result is not None
    assert result[2].get("model_type") == "supervised_binary"

def test_load_active_bundle_supervised_binary(tmp_path):
    registry, _ = isolated_registry(tmp_path, "random_forest_binary_1792944312")
    result = registry.load_active_bundle("supervised_binary")
    assert result is not None
    model, prep, meta = result
    assert hasattr(model, "predict") and hasattr(prep, "transform_inference")
    assert meta.get("model_type") == "supervised_binary"

def test_load_active_bundle_anomaly(tmp_path):
    registry, _ = isolated_registry(tmp_path, "isolation_forest_binary_1792944313")
    result = registry.load_active_bundle("anomaly_isolation_forest")
    assert result is not None
    model, prep, meta = result
    assert type(model).__name__ == "IsolationForest"
    assert meta.get("task_type") == "anomaly"

def test_preprocessor_feature_names_count(rf_binary_bundle):
    _, prep = rf_binary_bundle
    assert len(prep.feature_names_) == 28

def test_preprocessor_includes_model_training_features(rf_binary_bundle):
    _, prep = rf_binary_bundle
    names = list(prep.feature_names_)
    assert "down_up_ratio" in names
    assert "average_packet_size" in names

def test_preprocessor_transform_inference_shape(rf_binary_bundle):
    _, prep = rf_binary_bundle
    X = prep.transform_inference(BENIGN_FLOW)
    assert X.shape == (1, 28)
    assert not np.any(np.isnan(X)) and not np.any(np.isinf(X))

def test_preprocessor_handles_missing_keys(rf_binary_bundle):
    _, prep = rf_binary_bundle
    X = prep.transform_inference({"flow_duration": 1.5, "syn_flag_count": 5.0})
    assert X.shape == (1, 28) and not np.any(np.isnan(X))

def test_preprocessor_handles_nan_values(rf_binary_bundle):
    """transform_inference handles NaN: SimpleImputer replaces with median from training."""
    _, prep = rf_binary_bundle
    nan_flow = {**BENIGN_FLOW, "flow_bytes_s": float("nan"), "rst_flag_count": float("nan")}
    X = prep.transform_inference(nan_flow)
    assert X.shape == (1, 28)
    assert not np.any(np.isnan(X)), "NaN should be imputed to median"


def test_preprocessor_raises_on_inf_documented(rf_binary_bundle):
    """BEHAVIOR DOCUMENTED: transform_inference raises ValueError on Inf inputs.

    sklearn's SimpleImputer validates finite-only input. Callers (e.g., HybridDetectionEngine)
    must replace Inf with np.nan before calling transform_inference().
    This is a known constraint of the FlowDataPreprocessor contract.
    """
    _, prep = rf_binary_bundle
    inf_flow = {**BENIGN_FLOW, "flow_packets_s": float("inf")}
    with pytest.raises(ValueError, match="infinity"):
        prep.transform_inference(inf_flow)

def test_rf_binary_benign_prediction(rf_binary_bundle):
    model, prep = rf_binary_bundle
    pred = model.predict(prep.transform_inference(BENIGN_FLOW))[0]
    assert prep.inverse_label_mapping_.get(pred, str(pred)) == "BENIGN"

def test_rf_binary_attack_prediction(rf_binary_bundle):
    model, prep = rf_binary_bundle
    pred = model.predict(prep.transform_inference(ATTACK_FLOW))[0]
    assert prep.inverse_label_mapping_.get(pred, str(pred)) == "ATTACK"

def test_rf_binary_predict_proba_shape(rf_binary_bundle):
    model, prep = rf_binary_bundle
    assert hasattr(model, "predict_proba")
    proba = model.predict_proba(prep.transform_inference(ATTACK_FLOW))
    assert proba.shape == (1, 2)
    assert all(0.0 <= p <= 1.0 for p in proba[0])

def test_rf_binary_attack_confidence_above_threshold(rf_binary_bundle):
    model, prep = rf_binary_bundle
    proba = model.predict_proba(prep.transform_inference(ATTACK_FLOW))[0]
    assert max(proba) > 0.60, f"Attack confidence {max(proba):.3f} too low"

def test_if_binary_predict_returns_valid_label(if_binary_bundle):
    model, prep = if_binary_bundle
    pred = model.predict(prep.transform_inference(ATTACK_FLOW))[0]
    assert pred in {1, -1}

def test_if_binary_attack_is_outlier(if_binary_bundle):
    model, prep = if_binary_bundle
    assert model.predict(prep.transform_inference(ATTACK_FLOW))[0] == -1

def test_if_binary_decision_function_returns_scalar(if_binary_bundle):
    model, prep = if_binary_bundle
    score = model.decision_function(prep.transform_inference(BENIGN_FLOW))[0]
    assert not math.isnan(score) and not math.isinf(score)

def test_if_normalized_anomaly_score_range(if_binary_bundle):
    model, prep = if_binary_bundle
    dec = model.decision_function(prep.transform_inference(ATTACK_FLOW))[0]
    norm = 1.0 / (1.0 + math.exp(dec * 5.0))
    assert 0.0 < norm < 1.0

def test_if_not_treated_as_probability_classifier(if_binary_bundle):
    model, _ = if_binary_bundle
    assert not hasattr(model, "predict_proba"), "IsolationForest must not have predict_proba"

def test_rf_multiclass_classes_count(rf_multiclass_bundle):
    model, _ = rf_multiclass_bundle
    assert len(model.classes_) == 4

def test_rf_multiclass_predict_returns_valid_class(rf_multiclass_bundle):
    model, prep = rf_multiclass_bundle
    pred = model.predict(prep.transform_inference(ATTACK_FLOW))[0]
    assert pred in model.classes_

def test_ml_available_contract():
    import sys; sys.path.insert(0, str(WORKSPACE_ROOT))
    from detection.inference import is_ml_available, get_ml_status
    avail, reason = is_ml_available()
    assert isinstance(avail, bool) and isinstance(reason, str)
    status = get_ml_status()
    assert "available" in status and "joblib" in status and "sklearn" in status
    assert avail is True, f"ML should be available: {reason}"

def test_central_inference_engine_status():
    import sys; sys.path.insert(0, str(WORKSPACE_ROOT))
    from detection.inference import CentralInferenceEngine
    engine = CentralInferenceEngine(model_path=str(RF_BINARY_DIR / "model.joblib"))
    status = engine.get_status()
    assert status["has_model"] is True and status["model_bundle_loaded"] is True
    assert status["ml_available"] is False
    assert status["has_model_preprocessor"] is True
    assert status["ml_status"] == "ML_UNAVAILABLE"  # No inference has completed yet.

    result = engine.predict_flow(BENIGN_FLOW)
    assert result.predicted_label != "ERROR_MALFORMED"
    ready_status = engine.get_status()
    assert ready_status["ml_status"] == "ML_READY"
    assert ready_status["ml_available"] is True


def test_central_inference_refuses_model_without_fitted_preprocessor(tmp_path):
    from detection.inference import CentralInferenceEngine

    engine = CentralInferenceEngine(
        model_path=str(RF_BINARY_DIR / "model.joblib"),
        preprocessor_path=tmp_path / "missing_preprocessor.joblib",
    )
    status = engine.get_status()

    assert status["has_model"] is True
    assert status["has_model_preprocessor"] is False
    assert status["ml_available"] is False
    assert status["ml_status"] == "ML_UNAVAILABLE"

def test_central_inference_engine_rule_only_when_no_model():
    import sys; sys.path.insert(0, str(WORKSPACE_ROOT))
    from detection.inference import CentralInferenceEngine, DetectionResult
    result = CentralInferenceEngine().predict_flow(BENIGN_FLOW)
    assert isinstance(result, DetectionResult) and result.is_attack is False

def test_hybrid_engine_generates_alerts(rf_binary_bundle, if_binary_bundle):
    """HybridDetectionEngine must produce at least one alert for a SYN-flood sequence."""
    import sys; sys.path.insert(0, str(WORKSPACE_ROOT))
    from detection.hybrid_detection import HybridDetectionEngine
    from capture.packet_capture import NormalizedPacket
    rf_model, rf_prep = rf_binary_bundle
    if_model, if_prep = if_binary_bundle
    engine = HybridDetectionEngine(
        supervised_model=rf_model, supervised_preprocessor=rf_prep,
        supervised_meta={"task_type": "binary", "version": "rf_test", "algorithm": "Random Forest"},
        anomaly_model=if_model, anomaly_preprocessor=if_prep,
        anomaly_meta={"version": "if_test"},
    )
    all_alerts = []
    for i in range(15):
        pkt = NormalizedPacket(
            timestamp=1700000000.0 + i * 0.00001,
            src_ip="192.168.1.100", dst_ip="10.0.0.1",
            src_port=54321 + i, dst_port=80,
            protocol="TCP", length=40, tcp_flags="SYN",
        )
        alerts, _ = engine.process_packet_all(pkt)
        all_alerts.extend(alerts)
    flush_alerts, _ = engine.flush()
    all_alerts.extend(flush_alerts)
    assert len(all_alerts) > 0, "No alerts generated for SYN-flood sequence"
    methods = {a.detection_method for a in all_alerts}
    assert any(m in methods for m in ("HEURISTIC", "ML_SUPERVISED", "ML_ANOMALY"))

def test_dataset_overlap_documented():
    """Document: 200-row dataset is a partial subset of 1200-row (non-independent splits)."""
    import pandas as pd
    small_path = WORKSPACE_ROOT / "data" / "sample" / "cicids2017_sample.csv"
    large_path = WORKSPACE_ROOT / "data" / "datasets" / "cicids2017" / "raw" / "cicids2017_sample_1200.csv"
    if not small_path.is_file() or not large_path.is_file():
        pytest.skip("Dataset files not present")
    small = pd.read_csv(small_path).drop(columns=["Timestamp", "Label"], errors="ignore")
    large = pd.read_csv(large_path).drop(columns=["Timestamp", "Label"], errors="ignore")
    overlap = pd.merge(small, large, how="inner")
    assert len(overlap) >= 100, f"Unexpectedly low overlap: {len(overlap)}"

def test_temporary_registry_uses_its_isolated_paths(tmp_path):
    from ml.model_registry import ModelRegistry
    reg = ModelRegistry(models_dir=tmp_path / "models", db_path=tmp_path / "test.db")
    reg.validate_artifact("random_forest_binary_1792944312")
    assert reg.db_path == tmp_path / "test.db"
    assert reg.models_dir == tmp_path / "models"
