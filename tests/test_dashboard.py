"""Headless integration tests for Streamlit security monitoring dashboard.

Uses Streamlit's official headless AppTest framework to verify:
- Non-existent database path displays warning without creating database file.
- Empty database (0 rows) displays helpful empty-state guidance.
- Populated database correctly displays cumulative metrics, rule counts, and tables.
- Alert and traffic filters correctly narrow displayed records.
- Database read errors are handled gracefully with concise error messages.
- Timestamp formatting utility behaves predictably on valid, invalid, and None values.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from app import format_timestamp, build_timeline_chart
from capture.packet_capture import NormalizedPacket
from detection import Alert
from storage.database import DatabaseManager

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_format_timestamp_valid():
    """Verify conversion of UNIX epoch timestamp to readable UTC string."""
    # 1700000000 = 2023-11-14 22:13:20 UTC
    formatted = format_timestamp(1700000000.0)
    assert formatted == "2023-11-14 22:13:20 UTC"


def test_format_timestamp_none_and_invalid():
    """Verify fallback handling for None or malformed timestamps."""
    assert format_timestamp(None) == "N/A"
    assert format_timestamp("invalid_epoch") == "invalid_epoch"


def test_dashboard_missing_database_shows_warning_and_does_not_create_file(tmp_path: Path):
    """Verify that a non-existent database path shows empty state without creating a file."""
    non_existent_db = tmp_path / "does_not_exist.db"
    assert not non_existent_db.exists()

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()

    # Input the missing database path in the sidebar
    at.sidebar.text_input[0].input(str(non_existent_db)).run()

    # Verify warning and info elements
    assert len(at.warning) >= 1
    assert "Database file not found" in at.warning[0].value
    assert len(at.info) >= 1
    assert "No database exists at the specified path" in at.info[0].value

    # CRITICAL: Verify file was NOT silently created on disk
    assert not non_existent_db.exists()


def test_dashboard_empty_database_shows_info_message(tmp_path: Path):
    """Verify that an empty database (0 packets, 0 alerts) displays an empty-state message."""
    empty_db = tmp_path / "empty_ids.db"
    with DatabaseManager(empty_db) as db:
        db.init_db()

    assert empty_db.is_file()

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(empty_db)).run()

    assert len(at.info) >= 1
    assert "contains no stored packets or alerts" in at.info[0].value
    # No tables rendered when empty
    assert len(at.dataframe) == 0


def test_dashboard_populated_database_displays_metrics_and_tables(tmp_path: Path):
    """Verify populated database displays cumulative stored totals, rule breakdown, and tables."""
    test_db = tmp_path / "populated_ids.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        # Insert 2 packets
        db.insert_packet(
            NormalizedPacket(
                timestamp=1700000001.0,
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                protocol="TCP",
                length=64,
                src_port=54321,
                dst_port=80,
                tcp_flags="S",
            )
        )
        db.insert_packet(
            NormalizedPacket(
                timestamp=1700000002.0,
                src_ip="192.168.1.100",
                dst_ip="8.8.8.8",
                protocol="UDP",
                length=128,
                src_port=54322,
                dst_port=53,
                tcp_flags=None,
            )
        )
        # Insert 2 alerts
        db.insert_alert(
            Alert(
                timestamp=1700000001.0,
                rule_name="PORT_SCAN",
                severity="HIGH",
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                observed_value=12.0,
                threshold=10.0,
                window_seconds=30.0,
                message="Port scan threshold exceeded",
            )
        )
        db.insert_alert(
            Alert(
                timestamp=1700000002.0,
                rule_name="SYN_RATE",
                severity="MEDIUM",
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                observed_value=25.0,
                threshold=20.0,
                window_seconds=10.0,
                message="Elevated SYN rate detected",
            )
        )

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(test_db)).run()

    # Verify no exceptions
    assert not at.exception

    # Verify cumulative metrics
    metric_dict = {m.label: m.value for m in at.metric}
    assert metric_dict["Total Stored Packets"] == "2"
    assert metric_dict["Total Stored Alerts"] == "2"
    assert metric_dict["High-Severity Alerts"] == "1"
    assert metric_dict["Distinct Alert Sources"] == "1"
    assert metric_dict["PORT_SCAN"] == "1"
    assert metric_dict["SYN_RATE"] == "1"

    # Verify dataframes rendered for alerts, investigation packet evidence, and traffic
    assert len(at.dataframe) == 3
    alerts_df = at.dataframe[0].value
    evidence_df = at.dataframe[1].value
    packets_df = at.dataframe[2].value

    assert len(alerts_df) == 2
    assert "Rule" in alerts_df.columns
    assert "Severity" in alerts_df.columns
    assert "Description / Reason" in alerts_df.columns

    # Verify related packet evidence dataframe
    assert len(evidence_df) >= 1
    assert "Protocol" in evidence_df.columns
    assert "TCP Flags" in evidence_df.columns
    assert "Length (Bytes)" in evidence_df.columns

    assert len(packets_df) == 2
    assert "Protocol" in packets_df.columns
    assert "Length (Bytes)" in packets_df.columns
    assert "TCP Flags" in packets_df.columns


def test_dashboard_alert_filtering(tmp_path: Path):
    """Verify that sidebar filter options correctly filter alert records."""
    test_db = tmp_path / "filter_test.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        db.insert_packet(
            NormalizedPacket(
                timestamp=100.0,
                src_ip="1.1.1.1",
                dst_ip="2.2.2.2",
                protocol="TCP",
                length=60,
                src_port=1000,
                dst_port=80,
                tcp_flags="S",
            )
        )
        db.insert_alert(
            Alert(
                timestamp=100.0,
                rule_name="PORT_SCAN",
                severity="HIGH",
                src_ip="1.1.1.1",
                dst_ip="2.2.2.2",
                observed_value=5.0,
                threshold=5.0,
                window_seconds=10.0,
                message="Test port scan",
            )
        )
        db.insert_alert(
            Alert(
                timestamp=101.0,
                rule_name="SYN_RATE",
                severity="MEDIUM",
                src_ip="1.1.1.1",
                dst_ip="2.2.2.2",
                observed_value=20.0,
                threshold=20.0,
                window_seconds=10.0,
                message="Test syn rate",
            )
        )

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(test_db)).run()

    # Filter by Rule = PORT_SCAN
    rule_select = at.sidebar.selectbox[0]
    rule_select.select("PORT_SCAN").run()

    alerts_df = at.dataframe[0].value
    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["Rule"] == "PORT_SCAN"


def test_dashboard_alert_investigation_empty_evidence(tmp_path: Path):
    """Verify alert investigation displays clear message when no matching packets exist."""
    test_db = tmp_path / "empty_evidence.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        # Insert packet with different IPs
        db.insert_packet(
            NormalizedPacket(
                timestamp=100.0,
                src_ip="8.8.8.8",
                dst_ip="8.8.4.4",
                protocol="UDP",
                length=50,
            )
        )
        # Insert alert with different IPs
        db.insert_alert(
            Alert(
                timestamp=100.0,
                rule_name="PORT_SCAN",
                severity="HIGH",
                src_ip="192.168.1.100",
                dst_ip="10.0.0.5",
                observed_value=15.0,
                threshold=10.0,
                window_seconds=30.0,
                message="Port scan with no matching packets in db",
            )
        )

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(test_db)).run()

    # Alert table and traffic table rendered, but evidence table should be replaced with warning
    assert len(at.dataframe) == 2
    # Warning message for empty packet evidence
    warnings = [w.value for w in at.warning]
    assert any("No matching packet records found" in w for w in warnings)


def test_dashboard_alert_investigation_selection_updates_view(tmp_path: Path):
    """Verify selecting an alert by ID updates the investigation details."""
    test_db = tmp_path / "select_investigation.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        # Packets for host 1
        db.insert_packet(NormalizedPacket(10.0, "10.0.0.1", "10.0.0.2", "TCP", 64, 1000, 80, "S"))
        # Packets for host 2
        db.insert_packet(NormalizedPacket(20.0, "10.0.0.3", "10.0.0.4", "TCP", 64, 2000, 443, "S"))

        db.insert_alert(
            Alert(10.0, "PORT_SCAN", "10.0.0.1", "10.0.0.2", 15.0, 10.0, 5.0, "HIGH", "Host 1 scan")
        )
        db.insert_alert(
            Alert(20.0, "SYN_RATE", "10.0.0.3", "10.0.0.4", 25.0, 20.0, 5.0, "MEDIUM", "Host 2 syn flood")
        )

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(test_db)).run()

    # The main selectbox is the alert investigation selector
    investigate_select = at.selectbox[0]
    # Select Alert #1 (PORT_SCAN for 10.0.0.1 -> 10.0.0.2)
    investigate_select.select(1).run()

    # Check evidence dataframe has packet for 10.0.0.1 -> 10.0.0.2
    evidence_df = at.dataframe[1].value
    assert len(evidence_df) == 1
    assert evidence_df.iloc[0]["Source IP"] == "10.0.0.1"
    assert evidence_df.iloc[0]["Destination IP"] == "10.0.0.2"


def test_dashboard_corrupt_database_error_handling(tmp_path: Path):
    """Verify corrupt database files produce a graceful Streamlit error message."""
    bad_db = tmp_path / "corrupt.db"
    bad_db.write_text("NOT A VALID SQLITE DATABASE", encoding="utf-8")

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(bad_db)).run()

    assert len(at.error) >= 1
    assert "Database error" in at.error[0].value or "file is not a database" in at.error[0].value


def test_dashboard_time_range_scope_filter(tmp_path: Path):
    """Verify time range scope filter isolates alerts and packets within the specified range."""
    test_db = tmp_path / "time_scope.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        # Run 1: ts=100
        db.insert_packet(NormalizedPacket(100.0, "1.1.1.1", "2.2.2.2", "TCP", 64, 100, 80, "S"))
        db.insert_alert(Alert(100.0, "PORT_SCAN", "1.1.1.1", "2.2.2.2", 10, 10, 5.0, "HIGH", "Run 1 scan"))

        # Run 2: ts=500
        db.insert_packet(NormalizedPacket(500.0, "3.3.3.3", "4.4.4.4", "UDP", 128, 200, 53, None))
        db.insert_alert(Alert(500.0, "TRAFFIC_VOLUME", "3.3.3.3", "4.4.4.4", 20, 20, 5.0, "MED", "Run 2 surge"))

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(test_db)).run()

    # Initially both alerts are in the alerts table
    assert len(at.dataframe[0].value) == 2

    # Enable time filter checkbox
    time_checkbox = at.sidebar.checkbox[0]
    time_checkbox.check().run()

    # Set start to 450 and end to 550 to isolate Run 2
    at.sidebar.number_input[0].set_value(450.0).run()
    at.sidebar.number_input[1].set_value(550.0).run()

    # Alerts dataframe should now only contain Run 2
    alerts_df = at.dataframe[0].value
    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["Rule"] == "TRAFFIC_VOLUME"


def test_dashboard_security_analytics_kpis_and_tabs(tmp_path: Path):
    """Verify that Security Analytics tabs and multi-severity KPIs render accurately."""
    test_db = tmp_path / "analytics_test.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        # 3 packets
        db.insert_packets([
            NormalizedPacket(100.0, "192.168.1.1", "10.0.0.1", "TCP", 64, 1000, 80, "S"),
            NormalizedPacket(101.0, "192.168.1.2", "10.0.0.1", "UDP", 128, 2000, 53, None),
            NormalizedPacket(102.0, "192.168.1.3", "10.0.0.2", "ICMP", 32, None, None, None),
        ])
        # 3 alerts with various severities and distinct sources
        db.insert_alerts([
            Alert(100.0, "PORT_SCAN", "192.168.1.1", "10.0.0.1", 15, 10, 30.0, "CRITICAL", "Critical scan"),
            Alert(101.0, "SYN_RATE", "192.168.1.2", "10.0.0.1", 25, 20, 10.0, "HIGH", "High syn rate"),
            Alert(102.0, "TRAFFIC_VOLUME", "192.168.1.1", "10.0.0.2", 1000, 500, 10.0, "MEDIUM", "Med volume"),
        ])

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(test_db)).run()

    assert not at.exception

    # Check KPI metrics
    metric_dict = {m.label: m.value for m in at.metric}
    assert metric_dict["Total Stored Packets"] == "3"
    assert metric_dict["Total Stored Alerts"] == "3"
    assert metric_dict["High-Severity Alerts"] == "2"  # 1 CRITICAL + 1 HIGH
    assert metric_dict["Distinct Alert Sources"] == "2"  # 192.168.1.1, 192.168.1.2


def test_dashboard_refresh_button(tmp_path: Path):
    """Verify clicking refresh button runs without error."""
    test_db = tmp_path / "refresh_test.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        db.insert_packet(NormalizedPacket(1.0, "1.1.1.1", "2.2.2.2", "TCP", 64))
        db.insert_alert(Alert(1.0, "RULE", "1.1.1.1", "2.2.2.2", 1, 1, 10.0, "LOW", "Msg"))

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    at.sidebar.text_input[0].input(str(test_db)).run()

    # Click the refresh button in sidebar
    refresh_button = at.sidebar.button[0]
    refresh_button.click().run()

    assert not at.exception


def test_build_timeline_chart_single_sparse_empty():
    """Verify build_timeline_chart generates discrete bar charts for empty, single, and sparse data."""
    # 1. Empty data returns None
    assert build_timeline_chart([]) is None

    # 2. Single-point dataset generates bar mark
    single = [{"bucket": "2023-11-14 22:00 UTC", "count": 5}]
    chart_single = build_timeline_chart(single, y_field="count", y_title="Alerts")
    assert chart_single is not None
    chart_dict = chart_single.to_dict()
    assert chart_dict["mark"]["type"] == "bar"

    # 3. Sparse multi-year gap dataset uses discrete bars without continuous interpolation
    sparse = [
        {"bucket": "2015-10-20 UTC", "count": 242},
        {"bucket": "2026-10-16 UTC", "count": 2},
    ]
    chart_sparse = build_timeline_chart(sparse, y_field="count", y_title="Alerts")
    assert chart_sparse is not None
    sparse_dict = chart_sparse.to_dict()
    assert sparse_dict["mark"]["type"] == "bar"

