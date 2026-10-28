"""Comprehensive Unit and Integration Tests for Mini-IDS Prevention Engine.

Tests threat scoring, allowlist safeguards, simulation adapter, manual approval workflow,
rule verification, timed rollback, and audit log persistence.
"""

import time
import pytest

from prevention import (
    ActionStatus,
    ActionType,
    ApprovalManager,
    BaseFirewallAdapter,
    PolicyEngine,
    PreventionAction,
    PreventionEngine,
    PreventionMode,
    RiskLevel,
    RollbackManager,
    SimulationAdapter,
    VerificationManager,
    WindowsFirewallAdapter,
)
from storage.database import DatabaseManager


@pytest.fixture
def in_memory_db():
    """Create an isolated in-memory DatabaseManager for testing."""
    db = DatabaseManager(":memory:")
    db.init_db()
    return db


@pytest.fixture
def sim_adapter():
    """Create a fresh SimulationAdapter."""
    return SimulationAdapter()


@pytest.fixture
def policy_engine():
    """Create a fresh PolicyEngine with default simulation mode."""
    return PolicyEngine(
        default_mode=PreventionMode.SIMULATION,
        block_threshold=0.70,
        rate_limit_threshold=0.50,
        default_block_duration=300,
        custom_allowlist=["10.0.0.1", "192.168.1.0/24"],
    )


# --------------------------------------------------------------------------
# 1. Policy Engine & Safeguards Tests
# --------------------------------------------------------------------------

def test_allowlist_protection_loopback(policy_engine):
    """Verify loopback addresses are strictly protected from blocking."""
    is_prot, reason = policy_engine.is_allowlisted("127.0.0.1")
    assert is_prot is True
    assert "loopback" in reason.lower()

    is_prot_v6, _ = policy_engine.is_allowlisted("::1")
    assert is_prot_v6 is True


def test_allowlist_protection_dns_and_custom(policy_engine):
    """Verify DNS resolvers and custom CIDRs are protected."""
    is_prot, _ = policy_engine.is_allowlisted("8.8.8.8")
    assert is_prot is True

    is_prot_custom, _ = policy_engine.is_allowlisted("10.0.0.1")
    assert is_prot_custom is True

    is_prot_subnet, _ = policy_engine.is_allowlisted("192.168.1.50")
    assert is_prot_subnet is True

    # Untrusted IP should not be allowlisted
    is_untrusted, _ = policy_engine.is_allowlisted("45.33.32.156")
    assert is_untrusted is False


def test_threat_scoring_ddos_high_risk(policy_engine):
    """Verify high-confidence DDoS triggers CRITICAL risk and BLOCK_IP recommendation."""
    score, risk = policy_engine.calculate_threat_score(
        rule_name="ML_SUPERVISED_DDOS",
        heuristic_severity="CRITICAL",
        ml_confidence=0.99,
        anomaly_score=0.95,
        predicted_class="DDoS",
    )
    assert score >= 0.85
    assert risk == RiskLevel.CRITICAL

    action = policy_engine.evaluate(
        src_ip="198.51.100.22",
        rule_name="ML_SUPERVISED_DDOS",
        alert_id="alert-policy-1",
        source_tag="live",
        heuristic_severity="CRITICAL",
        ml_confidence=0.99,
        anomaly_score=0.95,
        predicted_class="DDoS",
    )
    assert action.action_type == ActionType.BLOCK_IP
    assert action.threat_score >= 0.85


def test_threat_scoring_allowlisted_target_never_blocked(policy_engine):
    """Ensure that even high-threat traffic on an allowlisted IP produces LOG_ALERT only."""
    action = policy_engine.evaluate(
        src_ip="127.0.0.1",
        rule_name="ML_SUPERVISED_DDOS",
        alert_id="alert-allowlist-1",
        source_tag="live",
        heuristic_severity="CRITICAL",
        ml_confidence=0.99,
    )
    assert action.action_type == ActionType.LOG_ALERT
    assert "ALLOWLIST" in action.details.get("reason", "")


def test_offline_pcap_safeguard_forces_simulation(policy_engine):
    """Ensure traffic tagged as offline PCAP cannot be placed in Authorized Enforcement."""
    action = policy_engine.evaluate(
        src_ip="203.0.113.19",
        rule_name="PORTSCAN",
        alert_id="alert-offline-1",
        heuristic_severity="HIGH",
        ml_confidence=0.95,
        source_tag="pcap",
        mode_override=PreventionMode.AUTHORIZED_ENFORCEMENT,
    )
    assert action.mode == PreventionMode.SIMULATION


# --------------------------------------------------------------------------
# 2. Simulation Adapter Tests
# --------------------------------------------------------------------------

def test_simulation_adapter_lifecycle(sim_adapter):
    """Test simulated rule addition, verification, listing, and deletion."""
    avail, _ = sim_adapter.is_available()
    assert avail is True

    rule_name = "test_block_01"
    target_ip = "198.51.100.44"

    # 1. Apply rule
    success, msg = sim_adapter.apply_rule(rule_name, target_ip, duration_seconds=60)
    assert success is True
    assert "[SIMULATED]" in msg

    # 2. Verify rule
    assert sim_adapter.verify_rule(rule_name, target_ip) is True
    assert sim_adapter.verify_rule(rule_name, "1.2.3.4") is False

    # 3. List active rules
    active = sim_adapter.list_active_rules()
    assert len(active) == 1
    assert active[0]["target_ip"] == target_ip

    # 4. Remove rule
    rem_success, rem_msg = sim_adapter.remove_rule(rule_name)
    assert rem_success is True
    assert sim_adapter.verify_rule(rule_name, target_ip) is False
    assert len(sim_adapter.list_active_rules()) == 0


def test_simulation_adapter_expiration(sim_adapter):
    """Verify that expired simulated rules are pruned."""
    sim_adapter.apply_rule("expiring_rule", "198.51.100.99", duration_seconds=0)
    # With 0 duration, it should immediately be expired
    time.sleep(0.01)
    assert sim_adapter.verify_rule("expiring_rule", "198.51.100.99") is False
    assert len(sim_adapter.list_active_rules()) == 0


# --------------------------------------------------------------------------
# 3. Windows Firewall Adapter Prefix Scoping Test
# --------------------------------------------------------------------------

def test_windows_firewall_prefix_scoping():
    """Verify WindowsFirewallAdapter strictly scopes rule names to MiniIDS_Block_."""
    wf = WindowsFirewallAdapter()
    scoped_1 = wf._ensure_scoped_name("test_123")
    assert scoped_1 == f"{BaseFirewallAdapter.RULE_PREFIX}test_123"

    scoped_2 = wf._ensure_scoped_name(f"{BaseFirewallAdapter.RULE_PREFIX}already_scoped")
    assert scoped_2 == f"{BaseFirewallAdapter.RULE_PREFIX}already_scoped"


# --------------------------------------------------------------------------
# 4. Approval Workflow Tests
# --------------------------------------------------------------------------

def test_manual_approval_workflow(in_memory_db):
    """Verify complete Manual Approval lifecycle: Queue -> Approve/Reject -> DB & Audit log."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.MANUAL_APPROVAL,
    )

    action = engine.process_threat(
        src_ip="198.51.100.77",
        rule_name="ML_SUPERVISED_PORTSCAN",
        alert_id="alert-approval-1",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.92,
        anomaly_score=0.88,
        predicted_class="PortScan",
    )
    assert action.status == ActionStatus.AWAITING_APPROVAL

    # Queue check
    queue = engine.approval_manager.get_pending_queue()
    assert len(queue) >= 1
    assert any(q["action_id"] == action.action_id for q in queue)

    # Approve and execute
    app_res = engine.approve_and_execute(action.action_id, operator="SecOps_Analyst", reason="Verified scan pattern")
    assert app_res is True

    # Check updated action in DB
    updated = in_memory_db.get_prevention_action(action.action_id)
    assert updated["status"] == ActionStatus.SIMULATED.value
    assert updated["approved_by"] == "SecOps_Analyst"

    # Check approval events table
    events = in_memory_db.get_approval_events(action.action_id)
    assert len(events) == 1
    assert events[0]["decision"] == "APPROVED"
    assert events[0]["operator"] == "SecOps_Analyst"

    # Check audit log
    audit = in_memory_db.get_audit_logs(limit=10)
    assert any(a["action"] == "MANUAL_APPROVE" for a in audit)


def test_manual_rejection_workflow(in_memory_db):
    """Verify operator rejection transitions status and records audit event."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.MANUAL_APPROVAL,
    )

    action = engine.process_threat(
        src_ip="198.51.100.88",
        rule_name="SUSPICIOUS_PROBE",
        alert_id="alert-reject-1",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.85,
    )
    assert action.status == ActionStatus.AWAITING_APPROVAL

    rej_res = engine.reject(action.action_id, operator="SecOps_Lead", reason="Known benign scanner")
    assert rej_res is True

    updated = in_memory_db.get_prevention_action(action.action_id)
    assert updated["status"] == ActionStatus.REJECTED.value

    events = in_memory_db.get_approval_events(action.action_id)
    assert len(events) == 1
    assert events[0]["decision"] == "REJECTED"


# --------------------------------------------------------------------------
# 5. Rollback and Expiration Tests
# --------------------------------------------------------------------------

def test_manual_unblock_and_rollback(in_memory_db):
    """Test operator unblock in simulation mode."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.SIMULATION,
    )

    action = engine.process_threat(
        src_ip="198.51.100.111",
        rule_name="DDOS_SYN_FLOOD",
        alert_id="alert-unblock-1",
        source_tag="live",
        heuristic_severity="CRITICAL",
        ml_confidence=0.98,
    )
    assert action.status == ActionStatus.SIMULATED

    unblock_ok, unblock_msg = engine.unblock(action.action_id, operator="Incident_Responder")
    assert unblock_ok is True
    assert "Simulation revoked" in unblock_msg

    updated = in_memory_db.get_prevention_action(action.action_id)
    assert updated["status"] == ActionStatus.ROLLED_BACK.value
    assert updated["rolled_back_at"] is not None

    audit = in_memory_db.get_audit_logs(action="REVOKE_SIMULATION")
    assert len(audit) >= 1
    assert audit[0]["result"] == "SUCCESS"


def test_clean_shutdown_rollback(in_memory_db):
    """Verify clean shutdown rolls back all active rules."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.SIMULATION,
    )

    engine.process_threat(src_ip="198.51.100.201", rule_name="TEST_ATTACK_1", alert_id="alert-shutdown-1", source_tag="live", heuristic_severity="HIGH", ml_confidence=0.95)
    engine.process_threat(src_ip="198.51.100.202", rule_name="TEST_ATTACK_2", alert_id="alert-shutdown-2", source_tag="live", heuristic_severity="HIGH", ml_confidence=0.95)

    stats_before = in_memory_db.get_prevention_stats()
    assert stats_before["active_rules"] == 2

    cleaned = engine.shutdown()
    assert cleaned == 2

    stats_after = in_memory_db.get_prevention_stats()
    assert stats_after["active_rules"] == 0


# --------------------------------------------------------------------------
# 6. Backend Simulation Guard and Safety Tests
# --------------------------------------------------------------------------

def test_backend_simulation_guard_prevents_os_firewall():
    """Verify that host firewall adapters are locked and fail closed in Simulation-Only mode."""
    wf = WindowsFirewallAdapter()
    avail, reason = wf.is_available()
    assert avail is False
    assert "Simulation-Only Mode" in reason

    import pytest
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY GUARD"):
        wf.apply_rule("test_rule", "198.51.100.99")

    with pytest.raises(RuntimeError, match="CRITICAL SAFETY GUARD"):
        wf.remove_rule("test_rule")

    from prevention.linux_firewall_adapter import LinuxFirewallAdapter
    lf = LinuxFirewallAdapter()
    lf_avail, lf_reason = lf.is_available()
    assert lf_avail is False
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY GUARD"):
        lf.apply_rule("test_rule", "198.51.100.99")


def test_prevention_engine_strictly_routes_to_simulation(in_memory_db):
    """Verify PreventionEngine routes all execution to SimulationAdapter even under ENFORCEMENT mode."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.AUTHORIZED_ENFORCEMENT,
    )
    # Active adapter must always be SimulationAdapter
    assert engine.get_active_adapter() == engine.simulation_adapter
    assert engine.get_active_adapter(mode=PreventionMode.AUTHORIZED_ENFORCEMENT) == engine.simulation_adapter

    engine.set_mode(PreventionMode.AUTHORIZED_ENFORCEMENT)
    assert engine.mode == PreventionMode.SIMULATION
    action = engine.process_threat(
        src_ip="198.51.100.55",
        rule_name="DDOS_SYN_FLOOD",
        alert_id="alert-safety-1",
        source_tag="live",
        heuristic_severity="CRITICAL",
        ml_confidence=0.99,
    )
    assert action.status == ActionStatus.SIMULATED
    # Verified in simulation adapter, not host OS
    assert engine.simulation_adapter.verify_rule(action.rule_name, "198.51.100.55") is True

    # Audit log must show DIVERT_TO_SIMULATION safety guard event
    audit = in_memory_db.get_audit_logs(action="CHANGE_PREVENTION_MODE")
    assert len(audit) >= 1
    assert audit[0]["result"] == "DENIED"


# --------------------------------------------------------------------------
# 7. Duplicate Action Suppression Tests
# --------------------------------------------------------------------------

def test_duplicate_action_suppression_active_rule(in_memory_db):
    """Verify subsequent alerts for an already blocked IP are marked SUPPRESSED."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.SIMULATION,
    )

    action1 = engine.process_threat(
        src_ip="198.51.100.60",
        rule_name="PORTSCAN_INITIAL",
        alert_id="alert-dup-1",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.91,
    )
    assert action1.status == ActionStatus.SIMULATED

    # Second attack event from same IP
    action2 = engine.process_threat(
        src_ip="198.51.100.60",
        rule_name="PORTSCAN_REPEAT",
        alert_id="alert-dup-2",
        source_tag="live",
        heuristic_severity="CRITICAL",
        ml_confidence=0.95,
    )
    assert action2.status == ActionStatus.SUPPRESSED
    assert action2.details.get("suppressed_duplicate") is True

    # Ensure only 1 active rule exists in simulation adapter
    rules = [r for r in engine.simulation_adapter.list_active_rules() if r.get("target_ip") == "198.51.100.60"]
    assert len(rules) == 1

    # Check suppression audit event
    supp_audit = in_memory_db.get_audit_logs(action="SUPPRESS_DUPLICATE_ACTION")
    assert len(supp_audit) >= 1
    assert supp_audit[0]["target"] == "198.51.100.60"


def test_duplicate_action_suppression_pending_queue(in_memory_db):
    """Verify subsequent alerts for an IP already awaiting manual approval are marked SUPPRESSED."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.MANUAL_APPROVAL,
    )

    action1 = engine.process_threat(
        src_ip="198.51.100.61",
        rule_name="SUSPICIOUS_PROBE_1",
        alert_id="alert-pending-1",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.88,
    )
    assert action1.status == ActionStatus.AWAITING_APPROVAL

    # Second event for same IP
    action2 = engine.process_threat(
        src_ip="198.51.100.61",
        rule_name="SUSPICIOUS_PROBE_2",
        alert_id="alert-pending-2",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.90,
    )
    assert action2.status == ActionStatus.SUPPRESSED
    assert action2.details.get("suppressed_duplicate") is True

    # Check pending queue length remains 1
    pending = [p for p in engine.approval_manager.get_pending_queue() if p.get("target_ip") == "198.51.100.61"]
    assert len(pending) == 1


# --------------------------------------------------------------------------
# 8. Explainability and Background Expiration Tests
# --------------------------------------------------------------------------

def test_explainable_risk_scoring(in_memory_db):
    """Verify threat action details contain comprehensive explainability breakdown."""
    engine = PreventionEngine(db_manager=in_memory_db)
    action = engine.process_threat(
        src_ip="198.51.100.70",
        rule_name="PORT_SCAN",
        alert_id="alert-explain-1",
        source_tag="live",
        heuristic_severity="HIGH",
        ml_confidence=0.94,
        anomaly_score=0.78,
        predicted_class="PortScan",
    )
    explanation = action.details.get("explanation")
    assert explanation is not None
    assert explanation["rule_name"] == "PORT_SCAN"
    assert explanation["rule_score"] > 0
    assert explanation["ml_model_active"] is True
    assert explanation["ml_predicted_class"] == "PortScan"
    assert explanation["ml_confidence"] == 0.94
    assert explanation["composite_score"] > 0.70
    assert explanation["risk_level"] in ["HIGH", "CRITICAL"]
    assert explanation["policy_decision"] == "BLOCK_IP"
    assert "recommendation" in explanation


def test_background_expiration_worker(in_memory_db):
    """Verify background expiration worker automatically expires simulation rules."""
    engine = PreventionEngine(
        db_manager=in_memory_db,
        initial_mode=PreventionMode.SIMULATION,
        default_block_duration=1,  # 1 second duration
    )
    action = engine.process_threat(
        src_ip="198.51.100.80",
        rule_name="FAST_EXPIRE_TEST",
        alert_id="alert-expire-1",
        source_tag="live",
        heuristic_severity="CRITICAL",
        ml_confidence=0.99,
    )
    assert action.status == ActionStatus.SIMULATED

    # Start worker with 0.2s polling interval
    engine.start_expiration_worker(interval_seconds=0.2)
    assert engine._expiration_thread is not None
    assert engine._expiration_thread.is_alive()

    # Wait for expiration
    time.sleep(1.4)
    # Verify rule expired in simulation adapter
    assert engine.simulation_adapter.verify_rule(action.rule_name, "198.51.100.80") is False

    engine.stop_expiration_worker()
    assert engine._expiration_thread is None

