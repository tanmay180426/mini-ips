"""Focused dashboard workflow regressions using disposable database paths."""

from pathlib import Path

import numpy as np
import pytest

from app import _detection_mode_from_records, _probe_active_model, _project_path
from streamlit.testing.v1 import AppTest
from detection import Alert
from storage.database import DatabaseManager


def test_detection_mode_requires_real_ml_attribution_or_successful_probe():
    assert _detection_mode_from_records([])[0] == "ML_UNAVAILABLE"
    assert _detection_mode_from_records([{"detection_method": "HEURISTIC"}])[0] == "RULE_ONLY"
    assert _detection_mode_from_records([], model_probe_succeeded=True)[0] == "ML_READY"
    assert _detection_mode_from_records([
        {"detection_method": "HEURISTIC"}, {"detection_method": "ML_SUPERVISED", "model_version": "m1"}
    ])[0] == "HYBRID"


def test_model_probe_runs_on_persisted_flow_feature_payload():
    class Preprocessor:
        feature_names_ = ["feature_a"]

        def transform_inference(self, values):
            assert values == {"feature_a": 2.0}
            return np.array([[2.0]])

    class Model:
        n_features_in_ = 1

        def predict(self, features):
            assert features.tolist() == [[2.0]]
            return [1]

    class Registry:
        def load_active_bundle(self, model_type):
            return Model(), Preprocessor(), {
                "version": "isolated-v1", "algorithm": "test-model", "feature_schema": ["feature_a"]
            }

    probe = _probe_active_model(Registry(), "supervised_binary", {"features_json": '{"feature_a": 2.0}'})
    assert probe["bundle_status"] == "LOADED"
    assert probe["schema_compatible"] is True
    assert probe["inference_status"] == "SUCCESS"


def test_model_probe_reports_missing_flow_without_claiming_inference():
    class Registry:
        def load_active_bundle(self, model_type):
            raise FileNotFoundError("artifact missing")

    probe = _probe_active_model(Registry(), "supervised_binary", None)
    assert probe["bundle_status"] == "UNAVAILABLE"
    assert probe["inference_status"] == "FAILED"
    assert "artifact missing" in probe["reason"]


def test_project_relative_paths_anchor_to_repository_root():
    path = _project_path("data/reports")
    assert path.is_absolute()
    assert path == Path(__file__).resolve().parents[1] / "data" / "reports"


def test_incident_notes_and_state_are_append_only_audited(tmp_path: Path):
    with DatabaseManager(tmp_path / "incidents.db") as db:
        db.init_db()
        alert_id = db.insert_alert(Alert(
            timestamp=100.0, rule_name="TEST_RULE", severity="HIGH",
            src_ip="192.0.2.1", dst_ip="198.51.100.2", observed_value=2,
            threshold=1, window_seconds=5, message="fixture",
        ))
        assert db.get_alert_investigation_state(alert_id)["status"] == "OPEN"
        assert db.update_alert_investigation(alert_id, "TRIAGED", "Reviewed evidence", actor="TEST_ANALYST")
        assert db.update_alert_investigation(alert_id, "RESOLVED", "Closed with evidence", actor="TEST_ANALYST")
        state = db.get_alert_investigation_state(alert_id)
        assert state["status"] == "RESOLVED"
        assert state["notes"] == "Closed with evidence"
        history = db.get_audit_logs(limit=None, action="UPDATE_ALERT_INVESTIGATION")
        assert [entry["result"] for entry in reversed(history)] == ["TRIAGED", "RESOLVED"]
        assert all(entry["details"]["alert_id"] == alert_id for entry in history)


def test_incident_update_rejects_invalid_state_and_missing_alert(tmp_path: Path):
    with DatabaseManager(tmp_path / "incidents.db") as db:
        db.init_db()
        with pytest.raises(ValueError):
            db.update_alert_investigation(1, "BLOCKED", "bad")
        assert db.update_alert_investigation(999, "OPEN", "missing") is False


def test_primary_dashboard_pages_render_without_exceptions(monkeypatch):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    monkeypatch.setenv("MINI_IDS_DEMO_MODE", "1")
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    for page in [
        "Overview", "Live Monitoring", "Prevention Center", "Audit Log",
        "Alert Investigation", "Traffic Intelligence", "AI/ML Detection",
        "Model Laboratory", "Data Explorer", "Detection Rules", "System & Settings",
    ]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Dashboard page {page!r} raised: {[e.message for e in at.exception]}"


def test_demo_prevention_button_persists_simulation_and_audit(monkeypatch, tmp_path):
    from storage.database import DatabaseManager

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    demo_db = tmp_path / "demo-flow.db"
    monkeypatch.setenv("MINI_IDS_DEMO_MODE", "1")
    monkeypatch.setenv("MINI_IDS_DEMO_DB_PATH", str(demo_db))
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Prevention Center").run()
    next(widget for widget in at.selectbox if widget.label == "Stored demo alert").set_value("1").run()
    at.button(key="evaluate_demo_prevention").click().run()
    assert not at.exception
    assert any("Policy evaluation persisted" in item.value for item in at.info)
    with DatabaseManager(demo_db) as db:
        actions = db.get_prevention_actions(limit=None)
        assert actions
        assert all(item["mode"] in {"SIMULATION", "MANUAL_APPROVAL"} for item in actions)
        assert all(item["status"] != "BLOCKED" for item in actions)
        assert all(item["alert_id"] for item in actions)
        assert all(item.get("details", {}).get("source_tag") == "demo" for item in actions)
        assert all(item.get("details", {}).get("reason") for item in actions)
        audits = db.get_audit_logs(limit=None, action="POLICY_EVALUATION")
        assert audits and audits[0]["details"].get("alert_id")


def test_empty_default_database_keeps_demo_opt_in(monkeypatch, tmp_path):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    local_db = tmp_path / "empty-local.db"
    demo_db = tmp_path / "demo-opt-in.db"
    monkeypatch.setenv("MINI_IDS_DB_PATH", str(local_db))
    monkeypatch.setenv("MINI_IDS_DEMO_DB_PATH", str(demo_db))
    monkeypatch.delenv("MINI_IDS_DEMO_MODE", raising=False)
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert any("Database file not found" in item.value for item in at.warning)
    assert any("No database exists at the specified path" in item.value for item in at.info)
    assert not demo_db.exists()
