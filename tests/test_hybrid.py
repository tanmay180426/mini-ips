"""Unit tests for the Hybrid Intrusion Detection Engine.

Tests cover:
- HybridDetectionEngine initialization and default state.
- Heuristic alert passthrough with metadata tagging (detection_method='HEURISTIC').
- Integration of supervised ML classification on completed network flows.
- Integration of unsupervised anomaly detection on completed network flows.
- Graceful degradation when models are missing or disabled.
- Flow flush lifecycle.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from capture.packet_capture import NormalizedPacket
from detection import Alert, PortScanDetector
from detection.hybrid_detection import HybridDetectionEngine
from ml.feature_extraction import NetworkFlow


def test_hybrid_engine_initialization():
    """Verify default engine instantiates all heuristic detectors and flow extractor."""
    engine = HybridDetectionEngine()
    assert engine.enable_heuristic is True
    assert engine.enable_supervised is True
    assert engine.enable_anomaly is True
    assert engine.port_scan is not None
    assert engine.syn_rate is not None
    assert engine.traffic_anomaly is not None
    assert engine.flow_extractor is not None


def test_hybrid_engine_heuristic_alert_passthrough():
    """Verify heuristic detectors trigger alerts tagged with detection_method='HEURISTIC'."""
    # Create an engine with a mock detector that fires an alert
    mock_ps = MagicMock()
    fake_alert = Alert(
        timestamp=100.0,
        rule_name="PORT_SCAN",
        severity="HIGH",
        src_ip="192.168.1.50",
        dst_ip="10.0.0.1",
        observed_value=15.0,
        threshold=10.0,
        window_seconds=10.0,
        message="Port scan detected",
    )
    mock_ps.process_packet.return_value = fake_alert

    engine = HybridDetectionEngine(port_scan_detector=mock_ps)

    packet = NormalizedPacket(100.0, "192.168.1.50", "10.0.0.1", "TCP", 64, 1234, 80, "S")
    alerts, completed_flows = engine.process_packet_all(packet)

    assert len(alerts) >= 1
    heuristic_alert = alerts[0]
    assert heuristic_alert.rule_name == "PORT_SCAN"
    assert heuristic_alert.detection_method == "HEURISTIC"


def test_hybrid_engine_supervised_ml_flow_alert():
    """Verify completed flow classified as attack triggers ML_SUPERVISED alert."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])  # Class 1 = ATTACK
    mock_model.predict_proba.return_value = np.array([[0.05, 0.95]])

    mock_preproc = MagicMock()
    mock_preproc.transform_inference.return_value = np.zeros((1, 5))
    mock_preproc.inverse_label_mapping_ = {0: "BENIGN", 1: "PortScan"}

    meta = {
        "version": "rf_binary_v1",
        "algorithm": "random_forest",
        "task_type": "binary",
    }

    engine = HybridDetectionEngine(
        supervised_model=mock_model,
        supervised_preprocessor=mock_preproc,
        supervised_meta=meta,
        enable_heuristic=False,
        flow_timeout=5.0,
    )

    # Ingest packets that will trigger flow completion via RST flag
    p1 = NormalizedPacket(10.0, "192.168.1.50", "10.0.0.1", "TCP", 64, 5000, 80, "S")
    p2 = NormalizedPacket(10.5, "10.0.0.1", "192.168.1.50", "TCP", 64, 80, 5000, "R")

    engine.process_packet_all(p1)
    alerts, completed = engine.process_packet_all(p2)

    assert len(completed) == 1
    assert len(alerts) == 1
    ml_alert = alerts[0]
    assert ml_alert.detection_method == "ML_SUPERVISED"
    assert ml_alert.model_version == "rf_binary_v1"
    assert ml_alert.confidence is not None
    assert ml_alert.confidence >= 0.9


def test_hybrid_engine_graceful_degradation_without_models():
    """Verify engine functions cleanly as pure heuristic engine when ML models are None."""
    engine = HybridDetectionEngine(
        supervised_model=None,
        anomaly_model=None,
        enable_heuristic=True,
    )

    packet = NormalizedPacket(100.0, "192.168.1.10", "10.0.0.1", "TCP", 64, 1000, 80, "S")
    # Processing should execute without any exceptions
    single_alert = engine.process_packet(packet)
    assert single_alert is None or isinstance(single_alert, Alert)

    alerts, completed = engine.process_packet_all(packet)
    assert isinstance(alerts, list)
    assert isinstance(completed, list)

    flush_alerts, flushed_flows = engine.flush()
    assert isinstance(flush_alerts, list)
    assert isinstance(flushed_flows, list)


def test_runtime_mode_requires_successful_ml_inference():
    """Configured model files alone do not promote rule fallback to HYBRID."""
    no_model_engine = HybridDetectionEngine(enable_heuristic=True)
    status = no_model_engine.get_runtime_status()
    assert status["detection_mode"] == "RULE_ONLY"
    assert status["ml_status"] == "ML_UNAVAILABLE"

    model = MagicMock()
    model.predict.return_value = np.array([0])
    preprocessor = MagicMock()
    preprocessor.transform_inference.return_value = np.zeros((1, 2))
    engine = HybridDetectionEngine(
        supervised_model=model,
        supervised_preprocessor=preprocessor,
        supervised_meta={"task_type": "binary"},
        enable_heuristic=True,
    )
    assert engine.get_runtime_status()["detection_mode"] == "RULE_ONLY"

    engine.process_packet_all(NormalizedPacket(10.0, "192.0.2.1", "192.0.2.2", "TCP", 64, 1000, 80, "S"))
    engine.process_packet_all(NormalizedPacket(10.5, "192.0.2.2", "192.0.2.1", "TCP", 64, 80, 1000, "R"))

    status = engine.get_runtime_status()
    assert status["ml_status"] == "ML_READY"
    assert status["detection_mode"] == "HYBRID"
    assert status["ml_inference_count"] == 1


def test_failed_ml_inference_keeps_rule_only_mode():
    model = MagicMock()
    model.predict.side_effect = ValueError("incompatible model input")
    preprocessor = MagicMock()
    preprocessor.transform_inference.return_value = np.zeros((1, 2))
    engine = HybridDetectionEngine(
        supervised_model=model,
        supervised_preprocessor=preprocessor,
        enable_heuristic=True,
    )

    engine.process_packet_all(NormalizedPacket(10.0, "192.0.2.1", "192.0.2.2", "TCP", 64, 1000, 80, "S"))
    engine.process_packet_all(NormalizedPacket(10.5, "192.0.2.2", "192.0.2.1", "TCP", 64, 80, 1000, "R"))

    status = engine.get_runtime_status()
    assert status["ml_status"] == "ML_UNAVAILABLE"
    assert status["detection_mode"] == "RULE_ONLY"
    assert "incompatible model input" in status["reason"]
