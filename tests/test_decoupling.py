"""Defensive Import Decoupling and Baseline Isolation Tests.

Verifies:
1. Pure rule-based detection and packet normalization operate independently of optional ML dependencies (joblib, scikit-learn).
2. 'from detection import Alert' and 'from capture.packet_capture import NormalizedPacket' succeed under all conditions.
3. ML availability and readiness are explicitly inspectable via is_ml_available() and get_ml_status().
4. CentralInferenceEngine and DetectionPreprocessor provide safe fallback results on uninitialized or missing model artifacts without raising unhandled exceptions.
"""

import sys
from typing import Any, Dict
import pytest

from capture.packet_capture import NormalizedPacket
from detection import (
    Alert,
    PortScanDetector,
    RuleEngine,
    SynRateDetector,
    TrafficAnomalyDetector,
    get_ml_status,
    is_ml_available,
)
from detection.inference import CentralInferenceEngine, DetectionResult
from detection.preprocessing import DetectionPreprocessor


def test_import_alert_and_normalized_packet():
    """Verify Alert and NormalizedPacket data structures instantiate cleanly."""
    pkt = NormalizedPacket(
        timestamp=1700000000.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.1",
        protocol="TCP",
        length=64,
        src_port=44444,
        dst_port=80,
        tcp_flags="S",
    )
    assert pkt.src_ip == "192.168.1.10"
    assert pkt.protocol == "TCP"

    alert = Alert(
        timestamp=1700000000.0,
        rule_name="TEST_RULE",
        src_ip="192.168.1.10",
        dst_ip="10.0.0.1",
        observed_value=15.0,
        threshold=10.0,
        window_seconds=10.0,
        severity="HIGH",
        message="Test alert message",
    )
    assert alert.rule_name == "TEST_RULE"
    assert alert.severity == "HIGH"
    d = alert.to_dict()
    assert d["rule_name"] == "TEST_RULE"
    assert d["detection_method"] == "HEURISTIC"


def test_import_without_ml_in_sys_modules(monkeypatch):
    """Verify Alert and NormalizedPacket work when ML packages are completely unimportable."""
    monkeypatch.setitem(sys.modules, "joblib", None)
    monkeypatch.setitem(sys.modules, "sklearn", None)

    from detection import Alert as DecoupledAlert
    from capture.packet_capture import NormalizedPacket as DecoupledPacket

    pkt = DecoupledPacket(1.0, "1.1.1.1", "2.2.2.2", "TCP", 64, 1, 2, "S")
    assert pkt.src_ip == "1.1.1.1"

    alert = DecoupledAlert(1.0, "PORT_SCAN", "1.1.1.1", "2.2.2.2", 10, 5, 10.0, "HIGH", "Scan")
    assert alert.rule_name == "PORT_SCAN"


def test_rule_detectors_work_independently():
    """Verify heuristic detectors operate completely independently of ML dependencies."""
    engine = RuleEngine(
        port_scan_threshold=3,
        port_scan_window=10.0,
        syn_rate_threshold=3,
        syn_window=10.0,
    )

    # 1. Trigger Port Scan
    alerts = []
    for port in [21, 22, 23, 80]:
        pkt = NormalizedPacket(100.0, "10.0.0.99", "192.168.1.1", "TCP", 64, 50000, port, "S")
        res = engine.evaluate_packet(pkt)
        if res:
            alerts.extend(res)

    assert len(alerts) >= 1
    assert any(a.rule_name == "PORT_SCAN" for a in alerts)

    # 2. Trigger SYN Rate
    syn_alerts = []
    for _ in range(5):
        pkt = NormalizedPacket(200.0, "10.0.0.88", "192.168.1.1", "TCP", 64, 50001, 443, "S")
        res = engine.evaluate_packet(pkt)
        if res:
            syn_alerts.extend(res)

    assert len(syn_alerts) >= 1
    assert any(a.rule_name == "SYN_RATE" for a in syn_alerts)


def test_ml_availability_status_inspection():
    """Verify is_ml_available() and get_ml_status() provide inspectable diagnostics."""
    available, reason = is_ml_available()
    assert isinstance(available, bool)
    assert isinstance(reason, str)
    assert len(reason) > 0

    status = get_ml_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "reason" in status
    assert "joblib" in status
    assert "sklearn" in status
    assert status["available"] == (status["joblib"] and status["sklearn"])


def test_central_inference_engine_safe_fallback_on_uninitialized_model():
    """Verify CentralInferenceEngine safely falls back without raising unhandled exceptions."""
    engine = CentralInferenceEngine(
        model_path=None,
        scaler_path=None,
        anomaly_model_path=None,
    )

    status = engine.get_status()
    assert status["has_model"] is False
    assert status["has_scaler"] is False
    assert status["has_anomaly_model"] is False

    # Feed dummy flow
    flow: Dict[str, Any] = {
        "flow_duration": 1.5,
        "total_fwd_packets": 10,
        "total_bwd_packets": 8,
        "total_length_of_fwd_packets": 500,
        "total_length_of_bwd_packets": 800,
    }
    result = engine.predict_flow(flow)
    assert isinstance(result, DetectionResult)
    assert result.is_attack is False
    assert result.predicted_label == "BENIGN"
    assert result.confidence >= 0.0
    assert result.is_anomaly is False
    assert "baseline limits" in result.explanation.lower() or "normal" in result.explanation.lower()


def test_preprocessor_works_without_fitted_scalers():
    """Verify DetectionPreprocessor converts flow dictionaries using pure numpy when scalers are absent."""
    preproc = DetectionPreprocessor(schema_type="canonical", scaler=None, imputer=None)
    assert preproc.feature_names is not None
    assert len(preproc.feature_names) == 28

    flow = {
        "flow_duration": 2.0,
        "total_fwd_packets": 5,
        "total_bwd_packets": 3,
        "total_length_of_fwd_packets": 200,
        "total_length_of_bwd_packets": 150,
    }
    vec = preproc.prepare_flow(flow)
    assert vec.shape == (1, 28)
    assert vec.dtype.kind in "fi"
