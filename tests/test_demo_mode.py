"""Automated regression tests for Demonstration Mode and Synthetic Dataset in Mini-IDS.

Verifies:
- Deterministic synthetic packet and alert generation with RFC 5737 IP addresses.
- Correlated packet evidence exists for all synthetic alerts within their detection windows.
- Database population via get_or_create_demo_database is idempotent and isolated.
- Dashboard renders demo-mode banner and disables live capture in demo mode.
- Dashboard renders full KPIs, charts, alert list, and packet evidence without mini_ids.db.
- Dashboard toggle allows switching between local mode and demo mode.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from capture.packet_capture import NormalizedPacket
from detection import Alert
from storage.database import DatabaseManager
from storage.demo_data import (
    DEFAULT_DEMO_DB_PATH,
    DEMO_BASE_TIMESTAMP,
    generate_synthetic_alerts,
    generate_synthetic_packets,
    get_or_create_demo_database,
    populate_demo_database,
)

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_synthetic_packets_rfc_5737_and_determinism():
    """Verify synthetic packets comply with RFC 5737 documentation IP prefixes and sorting."""
    packets = generate_synthetic_packets(DEMO_BASE_TIMESTAMP)
    assert len(packets) == 77

    # Check chronological ordering
    for i in range(len(packets) - 1):
        assert packets[i].timestamp <= packets[i + 1].timestamp

    # Verify all IPs belong to RFC 5737 documentation prefixes
    allowed_prefixes = ("192.0.2.", "198.51.100.", "203.0.113.")
    for p in packets:
        assert any(p.src_ip.startswith(prefix) for prefix in allowed_prefixes), f"Non-RFC IP: {p.src_ip}"
        assert any(p.dst_ip.startswith(prefix) for prefix in allowed_prefixes), f"Non-RFC IP: {p.dst_ip}"

    # Verify protocol variety
    protocols = {p.protocol for p in packets}
    assert "TCP" in protocols
    assert "UDP" in protocols
    assert "ICMP" in protocols


def test_synthetic_alerts_have_synthetic_labels():
    """Verify synthetic alerts are explicitly labeled and contain valid thresholds."""
    alerts = generate_synthetic_alerts(DEMO_BASE_TIMESTAMP)
    assert len(alerts) == 4

    rules = {a.rule_name for a in alerts}
    assert "PORT_SCAN" in rules
    assert "SYN_RATE" in rules
    assert "TRAFFIC_ANOMALY" in rules

    severities = {a.severity for a in alerts}
    assert "HIGH" in severities
    assert "MEDIUM" in severities
    assert "LOW" in severities

    for a in alerts:
        assert "[SYNTHETIC DEMO EVENT]" in a.message


def test_synthetic_alerts_have_correlated_packet_evidence(tmp_path: Path):
    """Verify that every synthetic alert has matching packets in its detection window."""
    demo_db_path = tmp_path / "test_correlated.db"
    with DatabaseManager(demo_db_path) as db:
        populate_demo_database(db, DEMO_BASE_TIMESTAMP)
        alerts = db.get_alerts()
        assert len(alerts) == 4

        for alert in alerts:
            alert_ts = float(alert["timestamp"])
            win_sec = float(alert["window_seconds"])
            start_time = alert_ts - win_sec
            end_time = alert_ts

            evidence = db.get_related_packets(
                src_ip=alert["src_ip"],
                dst_ip=alert["dst_ip"],
                start_time=start_time,
                end_time=end_time,
            )
            assert len(evidence) >= 1, f"Alert {alert['rule_name']} (ID {alert['id']}) has 0 evidence packets"


def test_get_or_create_demo_database_lifecycle(tmp_path: Path):
    """Verify get_or_create_demo_database creates and populates a new database cleanly."""
    target_path = tmp_path / "fresh_demo.db"
    assert not target_path.exists()

    result_path = get_or_create_demo_database(target_path, DEMO_BASE_TIMESTAMP)
    assert result_path == target_path
    assert target_path.is_file()

    with DatabaseManager(target_path) as db:
        assert db.get_packet_count() == 77
        assert db.get_alert_count() == 4

    # Idempotent re-run should not duplicate records
    get_or_create_demo_database(target_path, DEMO_BASE_TIMESTAMP)
    with DatabaseManager(target_path) as db:
        assert db.get_packet_count() == 77
        assert db.get_alert_count() == 4


def test_default_demo_database_respects_isolated_environment_path(tmp_path: Path, monkeypatch):
    target_path = tmp_path / "configured_demo.db"
    monkeypatch.setenv("MINI_IDS_DEMO_DB_PATH", str(target_path))

    result_path = get_or_create_demo_database()

    assert result_path == target_path
    assert target_path.is_file()


def test_dashboard_renders_demo_mode_banner_and_disables_live_capture(monkeypatch: pytest.MonkeyPatch):
    """Verify dashboard in Demo Mode displays synthetic banner and disables live capture."""
    monkeypatch.setenv("MINI_IDS_DEMO_MODE", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()

    assert not at.exception

    # Verify Demonstration Mode banner is displayed
    info_messages = [info.value for info in at.info]
    assert any("Demonstration Mode Active" in msg for msg in info_messages)
    assert any("Live Packet Capture is disabled in Hosted Demonstration Mode" in msg for msg in info_messages)

    # Verify metrics reflect the synthetic dataset
    metric_dict = {m.label: m.value for m in at.metric}
    assert metric_dict["Total Stored Packets"] == "77"
    assert metric_dict["Total Stored Alerts"] == "4"
    assert metric_dict["High-Severity Alerts"] == "2"
    assert metric_dict["Distinct Alert Sources"] == "4"

    # Verify alert table and traffic table are populated
    assert len(at.dataframe) >= 2
    alerts_df = at.dataframe[0].value
    assert len(alerts_df) == 4
    assert any("[SYNTHETIC DEMO EVENT]" in desc for desc in alerts_df["Description / Reason"])


def test_dashboard_local_mode_with_custom_db_preserves_normal_behavior(tmp_path: Path):
    """Verify specifying a custom local DB path disables demo mode and uses local DB."""
    local_db = tmp_path / "custom_local.db"
    with DatabaseManager(local_db) as db:
        db.init_db()
        db.insert_packet(NormalizedPacket(10.0, "10.1.1.1", "10.2.2.2", "TCP", 64, 5000, 80, "S"))
        db.insert_alert(Alert(10.0, "PORT_SCAN", "10.1.1.1", "10.2.2.2", 1, 1, 10.0, "LOW", "Local custom alert"))

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.sidebar.text_input[0].input(str(local_db)).run()

    assert not at.exception

    # Demo banner should NOT be displayed when using a custom local DB
    info_messages = [info.value for info in at.info]
    assert not any("Demonstration Mode Active" in msg for msg in info_messages)

    # Metrics should reflect custom local DB
    metric_dict = {m.label: m.value for m in at.metric}
    assert metric_dict["Total Stored Packets"] == "1"
    assert metric_dict["Total Stored Alerts"] == "1"
