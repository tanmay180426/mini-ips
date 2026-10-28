"""Tests for Central Inference Engine, Anomaly Detection, and Rule Integration."""

from pathlib import Path
import numpy as np
import pytest

from detection.anomaly_detection import AnomalyDetectionEngine
from detection.inference import CentralInferenceEngine, DetectionResult
from detection.rule_engine import RuleEngine


def test_inference_engine_missing_artifacts_fallback():
    """Verify inference engine safely falls back without raising unhandled errors."""
    engine = CentralInferenceEngine(
        model_path="nonexistent_model.joblib",
        scaler_path="nonexistent_scaler.joblib",
    )
    flow = {
        "flow_duration": 1000,
        "total_fwd_packets": 5,
        "packet_rate": 5.0,
    }
    result = engine.predict_flow(flow)
    assert isinstance(result, DetectionResult)
    assert result.is_attack is False
    assert result.severity == "LOW"


def test_inference_engine_malformed_input():
    """Verify malformed input dictionary is handled gracefully."""
    engine = CentralInferenceEngine()
    bad_flow = {"invalid_feature": "non_numeric_data"}
    result = engine.predict_flow(bad_flow)
    assert isinstance(result, DetectionResult)
    assert "error" in result.predicted_label.lower() or result.severity == "LOW"


def test_rule_engine_port_scan_detection():
    """Verify deterministic RuleEngine triggers on simulated port scan activity."""
    engine = RuleEngine(port_scan_threshold=5, port_scan_window=10.0)

    alerts = []
    # Send 6 packets to different destination ports from same source IP
    for port in range(1000, 1007):
        pkt = {
            "timestamp": 1700000000.0,
            "src_ip": "192.168.1.50",
            "dst_ip": "10.0.0.1",
            "protocol": "TCP",
            "src_port": 54321,
            "dst_port": port,
            "tcp_flags": "SYN",
            "packet_length": 60,
        }
        res = engine.evaluate_packet(pkt)
        if res:
            alerts.extend(res)

    assert len(alerts) >= 1
    assert any(a.rule_name == "PORT_SCAN" for a in alerts)


def test_anomaly_detection_engine():
    """Verify AnomalyDetectionEngine scores vectors against an IsolationForest."""
    from sklearn.ensemble import IsolationForest

    X_train = np.random.normal(loc=0.0, scale=1.0, size=(100, 5))
    iso = IsolationForest(random_state=42).fit(X_train)

    engine = AnomalyDetectionEngine(model=iso, threshold=0.50)

    # Inlier near mean 0
    is_anom_in, score_in, _ = engine.score_vector(np.array([0.1, 0.0, -0.1, 0.2, 0.0]))
    # Extreme outlier
    is_anom_out, score_out, _ = engine.score_vector(np.array([50.0, 45.0, 60.0, 80.0, 75.0]))

    assert score_out > score_in
    assert is_anom_out is True
