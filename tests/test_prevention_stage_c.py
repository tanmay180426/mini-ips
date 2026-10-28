"""Stage C prevention lifecycle and simulation-only safety regressions."""

import time

import pytest

from prevention import ActionStatus, ActionType, PreventionEngine, PreventionMode, PolicyEngine
from prevention.linux_firewall_adapter import LinuxFirewallAdapter
from prevention.windows_firewall_adapter import WindowsFirewallAdapter
from storage.database import DatabaseManager


@pytest.fixture
def db():
    manager = DatabaseManager(":memory:")
    manager.init_db()
    yield manager
    manager.close()


def evaluate(**overrides):
    fields = {
        "src_ip": "198.51.100.9",
        "rule_name": "TEST_PORTSCAN",
        "alert_id": "alert-stage-c-1",
        "source_tag": "live",
        "heuristic_severity": "CRITICAL",
        "ml_confidence": 0.99,
        "anomaly_score": 0.9,
        "predicted_class": "PortScan",
    }
    fields.update(overrides)
    return PolicyEngine().evaluate(**fields)


def test_policy_rejects_invalid_input_and_missing_context():
    for params in (
        {"src_ip": "bad-ip"},
        {"alert_id": None},
        {"source_tag": "unknown-source"},
        {"heuristic_severity": "SEVERE"},
        {"target_port": 70000},
        {"protocol": "GRE"},
        {"ml_confidence": float("nan")},
    ):
        action = evaluate(**params)
        assert action.status == ActionStatus.FAILED
        assert action.action_type != ActionType.BLOCK_IP
        assert action.details["policy_version"]


def test_severity_threshold_and_source_requirements_are_deterministic():
    assert evaluate(heuristic_severity="LOW").action_type == ActionType.RATE_LIMIT
    assert evaluate(ml_confidence=0.1, anomaly_score=0.0).action_type != ActionType.BLOCK_IP
    # The source event and source tag are mandatory even with a very high score.
    assert evaluate(alert_id="").status == ActionStatus.FAILED


def test_allowlist_and_protected_assets_never_receive_simulation():
    policy = PolicyEngine(custom_allowlist=["192.168.1.0/24"])
    protected = policy.evaluate(src_ip="192.168.1.20", rule_name="TEST", alert_id="alert-allowlisted", source_tag="live", heuristic_severity="CRITICAL", ml_confidence=0.99)
    assert protected.action_type == ActionType.LOG_ALERT
    assert protected.details["allowlisted"] is True
    assert evaluate(src_ip="127.0.0.1").action_type == ActionType.LOG_ALERT
    with pytest.raises(ValueError):
        PolicyEngine(custom_allowlist=["not-an-ip"])


def test_policy_configuration_and_allowlist_entries_are_validated():
    with pytest.raises(ValueError):
        PolicyEngine(block_threshold=0.4, rate_limit_threshold=0.5)
    policy = PolicyEngine()
    with pytest.raises(ValueError):
        policy.add_to_allowlist("invalid")


def test_simulation_status_reason_policy_version_and_audit_trace(db):
    engine = PreventionEngine(db_manager=db, cooldown_seconds=0)
    result = engine.process_threat(**{
        "src_ip": "198.51.100.10", "rule_name": "DDOS_TEST", "alert_id": "alert-persist-1",
        "source_tag": "live", "heuristic_severity": "CRITICAL", "ml_confidence": 0.99,
    })
    assert result.status == ActionStatus.SIMULATED
    assert "network traffic was not blocked" in result.details["simulation_result"].lower()
    stored = db.get_prevention_action(result.action_id)
    assert stored["alert_id"] == "alert-persist-1"
    assert stored["status"] == ActionStatus.SIMULATED.value
    assert stored["details"]["policy_version"]
    assert stored["details"]["reason"]
    events = db.get_audit_logs(limit=None)
    assert any(e["details"].get("action_id") == result.action_id for e in events)
    assert all("BLOCKED" not in e["result"] for e in events)


def test_duplicate_cooldown_and_rate_limit_suppress_simulation(db):
    engine = PreventionEngine(db_manager=db, cooldown_seconds=0)
    first = engine.process_threat("198.51.100.11", "DDOS_FIRST", alert_id="alert-dup-a", source_tag="live", heuristic_severity="CRITICAL")
    duplicate = engine.process_threat("198.51.100.11", "DDOS_SECOND", alert_id="alert-dup-b", source_tag="live", heuristic_severity="CRITICAL")
    assert first.status == ActionStatus.SIMULATED
    assert duplicate.status == ActionStatus.SUPPRESSED
    assert duplicate.details.get("suppressed_duplicate") is True

    cooldown_engine = PreventionEngine(db_manager=DatabaseManager(":memory:"), cooldown_seconds=60)
    cooldown_engine.db.init_db()
    one = cooldown_engine.process_threat("198.51.100.12", "DDOS_ONE", alert_id="alert-cool-a", source_tag="live", heuristic_severity="CRITICAL")
    assert cooldown_engine.unblock(one.action_id)[0] is True
    assert all(rule["target_ip"] != "198.51.100.12" for rule in cooldown_engine.simulation_adapter.list_active_rules())
    later = cooldown_engine.process_threat("198.51.100.12", "DDOS_TWO", alert_id="alert-cool-b", source_tag="live", heuristic_severity="CRITICAL")
    assert later.status == ActionStatus.SUPPRESSED
    assert later.details.get("suppression") == "COOLDOWN"

    limited_db = DatabaseManager(":memory:")
    limited_db.init_db()
    limited = PreventionEngine(db_manager=limited_db, max_actions_per_window=1)
    limited.process_threat("198.51.100.13", "DDOS_ONE", alert_id="alert-rate-a", source_tag="live", heuristic_severity="CRITICAL")
    capped = limited.process_threat("198.51.100.14", "DDOS_TWO", alert_id="alert-rate-b", source_tag="live", heuristic_severity="CRITICAL")
    assert capped.status == ActionStatus.SUPPRESSED
    assert capped.details.get("suppression") == "RATE_LIMIT"


def test_approval_requires_pending_action_and_executes_simulation_only(db):
    engine = PreventionEngine(db_manager=db, initial_mode=PreventionMode.MANUAL_APPROVAL)
    queued = engine.process_threat("198.51.100.15", "DDOS_APPROVAL", alert_id="alert-approval", source_tag="live", heuristic_severity="CRITICAL")
    assert queued.status == ActionStatus.AWAITING_APPROVAL
    assert engine.approve_and_execute(queued.action_id, reason="reviewed") is True
    assert db.get_prevention_action(queued.action_id)["status"] == ActionStatus.SIMULATED.value
    assert engine.approve_and_execute(queued.action_id) is False


def test_expiration_and_revocation_audit_only_simulation_state(db, monkeypatch):
    engine = PreventionEngine(db_manager=db, default_block_duration=1, cooldown_seconds=0)
    result = engine.process_threat("198.51.100.16", "DDOS_EXPIRE", alert_id="alert-expiration", source_tag="live", heuristic_severity="CRITICAL")
    assert result.status == ActionStatus.SIMULATED
    import prevention.rollback as rollback_module
    real_time = time.time
    monkeypatch.setattr(rollback_module.time, "time", lambda: real_time() + 2)
    assert engine.expire_timed_rules() == 1
    assert db.get_prevention_action(result.action_id)["status"] == ActionStatus.EXPIRED.value
    assert all(rule["target_ip"] != "198.51.100.16" for rule in engine.simulation_adapter.list_active_rules())
    audits = db.get_audit_logs(limit=None)
    assert any(e["action"] == "EXPIRE_SIMULATION" and e["details"].get("action_id") == result.action_id for e in audits)


def test_policy_component_and_simulation_adapter_fail_closed(db, monkeypatch):
    engine = PreventionEngine(db_manager=db)
    monkeypatch.setattr(engine.policy_engine, "evaluate", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("policy down")))
    failed_policy = engine.process_threat("198.51.100.17", "DDOS_TEST", alert_id="alert-policy-failure", source_tag="live", heuristic_severity="CRITICAL")
    assert failed_policy.status == ActionStatus.FAILED
    assert engine.simulation_adapter.list_active_rules() == []

    engine2 = PreventionEngine(db_manager=DatabaseManager(":memory:"))
    engine2.db.init_db()
    monkeypatch.setattr(engine2.simulation_adapter, "apply_rule", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("adapter down")))
    failed_adapter = engine2.process_threat("198.51.100.18", "DDOS_TEST", alert_id="alert-adapter-failure", source_tag="live", heuristic_severity="CRITICAL")
    assert failed_adapter.status == ActionStatus.FAILED
    assert "adapter failure" in failed_adapter.error_message.lower()

    engine3 = PreventionEngine(db_manager=DatabaseManager(":memory:"))
    engine3.db.init_db()
    monkeypatch.setattr(engine3.simulation_adapter, "verify_rule", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("verify down")))
    failed_verify = engine3.process_threat("198.51.100.21", "DDOS_TEST", alert_id="alert-verify-failure", source_tag="live", heuristic_severity="CRITICAL")
    assert failed_verify.status == ActionStatus.FAILED
    assert engine3.db.get_prevention_action(failed_verify.action_id)["status"] == ActionStatus.FAILED.value


def test_no_os_adapter_or_command_path_is_reached(db, monkeypatch):
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("OS command execution must not be reached")

    monkeypatch.setattr(subprocess, "run", forbidden)
    for adapter in (WindowsFirewallAdapter(), LinuxFirewallAdapter()):
        adapter.SIMULATION_LOCKED = False  # Runtime configuration must not unlock legacy execution.
        assert adapter.verify_rule("rule", "198.51.100.1") is False
        assert adapter.list_active_rules() == []
        with pytest.raises(RuntimeError):
            adapter.apply_rule("rule", "198.51.100.1")
        with pytest.raises(RuntimeError):
            adapter.remove_rule("rule")

    engine = PreventionEngine(db_manager=db)
    for name in ("apply_rule", "verify_rule", "remove_rule", "list_active_rules"):
        monkeypatch.setattr(WindowsFirewallAdapter, name, forbidden)
        monkeypatch.setattr(LinuxFirewallAdapter, name, forbidden)
    result = engine.process_threat("198.51.100.19", "DDOS_SAFETY", alert_id="alert-safety", source_tag="live", heuristic_severity="CRITICAL")
    assert result.status == ActionStatus.SIMULATED
    audit = db.get_audit_logs(action="RECORD_SIMULATION")
    assert audit and audit[0]["details"]["traffic_prevented"] is False


def test_legacy_enforcement_mode_is_normalized_or_denied(db):
    engine = PreventionEngine(db_manager=db, initial_mode=PreventionMode.AUTHORIZED_ENFORCEMENT)
    assert engine.mode == PreventionMode.SIMULATION
    engine.set_mode(PreventionMode.AUTHORIZED_ENFORCEMENT)
    assert engine.mode == PreventionMode.SIMULATION
    assert db.get_audit_logs(action="CHANGE_PREVENTION_MODE")[0]["result"] == "DENIED"
    engine.policy_engine.mode = PreventionMode.AUTHORIZED_ENFORCEMENT
    result = engine.process_threat("198.51.100.20", "DDOS_DIRECT_MODE", alert_id="alert-direct-mode", source_tag="live", heuristic_severity="CRITICAL")
    assert result.status == ActionStatus.SIMULATED
    assert result.mode == PreventionMode.SIMULATION
