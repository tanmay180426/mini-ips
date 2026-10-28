"""End-to-End System Integration Tests for Mini-IDS.

Validates the full detection-to-prevention workflow:
Traffic Ingestion -> Schema Normalization -> Inference -> Threat Scoring -> Response Policy -> Prevention Action -> Verification & Persistence.
"""

from pathlib import Path
import time
import pytest

from detection.inference import CentralInferenceEngine
from prevention import (
    ActionStatus,
    ActionType,
    ApprovalManager,
    PolicyEngine,
    PreventionEngine,
    PreventionMode,
    SimulationAdapter,
)
from storage.database import DatabaseManager


def test_full_detection_to_prevention_pipeline(tmp_path):
    """Verify that a high-threat flow triggers policy evaluation and simulated prevention."""
    db_file = tmp_path / "test_integration.db"
    db = DatabaseManager(str(db_file))
    db.init_db()

    # 1. Inference Engine
    engine = CentralInferenceEngine(schema_type="unified")
    suspicious_flow = {
        "flow_duration": 500,
        "total_fwd_packets": 500,
        "total_backward_packets": 0,
        "total_length_of_fwd_packets": 32000,
        "total_length_of_bwd_packets": 0,
        "fwd_packet_length_mean": 64.0,
        "bwd_packet_length_mean": 0.0,
        "flow_packets_s": 1000.0,
        "flow_bytes_s": 64000.0,
    }
    det_res = engine.predict_flow(suspicious_flow)
    assert det_res is not None

    # 2. Prevention Engine in Simulation Mode
    prev_engine = PreventionEngine(
        db_manager=db,
        initial_mode=PreventionMode.SIMULATION,
    )
    action = prev_engine.process_threat(
        src_ip="198.51.100.42",
        rule_name="DDOS_VOLUMETRIC",
        alert_id="alert-integration-1",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.95,
        anomaly_score=0.88,
        predicted_class="DDoS",
    )

    assert action is not None
    assert action.mode == PreventionMode.SIMULATION
    assert action.target_ip == "198.51.100.42"

    # 3. Verify simulated block
    sim_adapter = prev_engine.simulation_adapter
    assert action.status == ActionStatus.SIMULATED
    simulation_exists = sim_adapter.verify_rule(action.rule_name, action.target_ip)
    assert simulation_exists is True


def test_manual_approval_workflow_integration(tmp_path):
    """Verify manual approval workflow transitions from pending to approved."""
    db_file = tmp_path / "test_approval.db"
    db = DatabaseManager(str(db_file))
    db.init_db()

    engine = PreventionEngine(
        db_manager=db,
        initial_mode=PreventionMode.MANUAL_APPROVAL,
    )

    action = engine.process_threat(
        src_ip="203.0.113.88",
        rule_name="PORT_SCAN",
        alert_id="alert-integration-approval-1",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.92,
        anomaly_score=0.85,
        predicted_class="PortScan",
    )
    assert action.status == ActionStatus.AWAITING_APPROVAL

    # Check pending queue
    queue = engine.approval_manager.get_pending_queue()
    assert any(q["action_id"] == action.action_id for q in queue)

    # Approve action
    approved = engine.approve_and_execute(
        action_id=action.action_id,
        operator="Security_Analyst",
        reason="Confirmed anomalous probe cluster",
    )
    assert approved is True

    # Confirm updated action in DB
    updated = db.get_prevention_action(action.action_id)
    assert updated is not None
    assert updated["status"] == ActionStatus.SIMULATED.value
