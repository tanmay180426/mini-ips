"""Unit tests for SQLite database storage module.

Tests verify:
- Idempotent schema initialization (init_db multiple times).
- Insertion and retrieval of single and batch NormalizedPacket instances.
- Insertion and retrieval of single and batch Alert instances.
- Correct handling of nullable fields (ports, tcp_flags, IPs).
- Chronological sorting (ASC and DESC) with limits and offsets.
- Empty database query behavior.
- Context manager and connection close resource cleanup.
- Input validation and exception raising on invalid types.
"""

from pathlib import Path
import pytest

from capture.packet_capture import NormalizedPacket
from detection import Alert
from storage.database import DatabaseManager, init_db


def test_schema_creation_and_repeated_initialization(tmp_path: Path):
    """Verify that init_db creates tables and can be called repeatedly safely."""
    db_file = tmp_path / "test_init.db"
    db = DatabaseManager(db_file)

    # Initialize once
    db.init_db()

    # Verify tables exist
    conn = db.get_connection()
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    assert "packets" in tables
    assert "alerts" in tables

    # Initialize a second time (should not fail or recreate existing tables)
    db.init_db()
    db.close()


def test_insert_and_get_single_packet(tmp_path: Path):
    """Verify inserting and retrieving a single NormalizedPacket."""
    db_file = tmp_path / "test_single_packet.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        pkt = NormalizedPacket(
            timestamp=1700000001.5,
            src_ip="192.168.1.50",
            dst_ip="10.0.0.1",
            protocol="TCP",
            length=64,
            src_port=54321,
            dst_port=443,
            tcp_flags="S",
        )
        row_id = db.insert_packet(pkt)
        assert row_id == 1
        assert db.get_packet_count() == 1

        packets = db.get_packets()
        assert len(packets) == 1
        record = packets[0]

        assert record["id"] == 1
        assert record["timestamp"] == 1700000001.5
        assert record["src_ip"] == "192.168.1.50"
        assert record["dst_ip"] == "10.0.0.1"
        assert record["protocol"] == "TCP"
        assert record["length"] == 64
        assert record["src_port"] == 54321
        assert record["dst_port"] == 443
        assert record["tcp_flags"] == "S"


def test_insert_packet_with_nullable_fields(tmp_path: Path):
    """Verify inserting an ICMP packet with None for ports and flags."""
    db_file = tmp_path / "test_nullable_packet.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        icmp_pkt = NormalizedPacket(
            timestamp=1700000002.0,
            src_ip="192.168.1.1",
            dst_ip="8.8.8.8",
            protocol="ICMP",
            length=32,
            src_port=None,
            dst_port=None,
            tcp_flags=None,
        )
        row_id = db.insert_packet(icmp_pkt)
        assert row_id == 1

        records = db.get_packets()
        assert len(records) == 1
        assert records[0]["src_port"] is None
        assert records[0]["dst_port"] is None
        assert records[0]["tcp_flags"] is None


def test_insert_and_get_single_alert(tmp_path: Path):
    """Verify inserting and retrieving a single Alert."""
    db_file = tmp_path / "test_single_alert.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        alert = Alert(
            timestamp=1700000010.0,
            rule_name="PORT_SCAN",
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            observed_value=15,
            threshold=10,
            window_seconds=30.0,
            severity="MEDIUM",
            message="TCP port scan detected.",
        )
        row_id = db.insert_alert(alert)
        assert row_id == 1
        assert db.get_alert_count() == 1

        alerts = db.get_alerts()
        assert len(alerts) == 1
        record = alerts[0]

        assert record["id"] == 1
        assert record["timestamp"] == 1700000010.0
        assert record["rule_name"] == "PORT_SCAN"
        assert record["src_ip"] == "192.168.1.100"
        assert record["dst_ip"] == "10.0.0.5"
        assert record["observed_value"] == 15.0
        assert record["threshold"] == 10.0
        assert record["window_seconds"] == 30.0
        assert record["severity"] == "MEDIUM"
        assert record["message"] == "TCP port scan detected."


def test_insert_alert_with_nullable_fields(tmp_path: Path):
    """Verify inserting a global traffic anomaly alert where src/dst IP may be None."""
    db_file = tmp_path / "test_nullable_alert.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        alert = Alert(
            timestamp=1700000020.0,
            rule_name="TRAFFIC_VOLUME",
            src_ip=None,
            dst_ip=None,
            observed_value=1000000,
            threshold=500000,
            window_seconds=10.0,
            severity="HIGH",
            message="Global stream traffic volume surge.",
        )
        db.insert_alert(alert)

        records = db.get_alerts()
        assert len(records) == 1
        assert records[0]["src_ip"] is None
        assert records[0]["dst_ip"] is None


def test_chronological_ordering_and_pagination(tmp_path: Path):
    """Verify packets and alerts are retrieved in chronological order with limits."""
    db_file = tmp_path / "test_ordering.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        # Insert 5 packets out of timestamp order
        pkts = [
            NormalizedPacket(105.0, "10.0.0.1", "10.0.0.2", "TCP", 60),
            NormalizedPacket(101.0, "10.0.0.1", "10.0.0.2", "TCP", 60),
            NormalizedPacket(103.0, "10.0.0.1", "10.0.0.2", "TCP", 60),
            NormalizedPacket(102.0, "10.0.0.1", "10.0.0.2", "TCP", 60),
            NormalizedPacket(104.0, "10.0.0.1", "10.0.0.2", "TCP", 60),
        ]
        inserted = db.insert_packets(pkts)
        assert inserted == 5

        # Ascending (default)
        asc = db.get_packets(order="ASC")
        timestamps = [p["timestamp"] for p in asc]
        assert timestamps == [101.0, 102.0, 103.0, 104.0, 105.0]

        # Descending
        desc = db.get_packets(order="DESC")
        timestamps_desc = [p["timestamp"] for p in desc]
        assert timestamps_desc == [105.0, 104.0, 103.0, 102.0, 101.0]

        # Limit and offset
        limited = db.get_packets(limit=2, offset=1, order="ASC")
        assert len(limited) == 2
        assert [p["timestamp"] for p in limited] == [102.0, 103.0]


def test_batch_insert_alerts(tmp_path: Path):
    """Verify batch insertion of alerts."""
    db_file = tmp_path / "test_batch_alerts.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        alerts = [
            Alert(10.0, "SYN_RATE", "1.1.1.1", "2.2.2.2", 20, 20, 10.0, "MED", "Msg 1"),
            Alert(20.0, "PORT_SCAN", "1.1.1.1", "2.2.2.2", 15, 15, 30.0, "HIGH", "Msg 2"),
        ]
        count = db.insert_alerts(alerts)
        assert count == 2
        assert db.get_alert_count() == 2

        # Empty batch returns 0
        assert db.insert_alerts([]) == 0


def test_empty_database_retrieval(tmp_path: Path):
    """Verify querying an empty database returns empty lists without error."""
    db_file = tmp_path / "test_empty.db"
    with DatabaseManager(db_file) as db:
        db.init_db()
        assert db.get_packets() == []
        assert db.get_alerts() == []
        assert db.get_packet_count() == 0
        assert db.get_alert_count() == 0


def test_validation_errors(tmp_path: Path):
    """Verify ValueError is raised on invalid object types."""
    db_file = tmp_path / "test_val.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        with pytest.raises(ValueError):
            db.insert_packet({"not": "a NormalizedPacket"})  # type: ignore

        with pytest.raises(ValueError):
            db.insert_alert({"not": "an Alert"})  # type: ignore


def test_init_db_convenience_function(tmp_path: Path):
    """Verify top-level init_db function initializes schema and returns manager."""
    db_file = tmp_path / "test_conv.db"
    db = init_db(db_file)
    assert db.db_path == db_file
    assert db.get_packet_count() == 0
    db.close()


def test_wal_journal_mode_enabled(tmp_path: Path):
    """Verify that SQLite connection explicitly sets and activates WAL journal mode."""
    db_file = tmp_path / "test_wal_mode.db"
    with DatabaseManager(db_file) as db:
        conn = db.get_connection()
        cursor = conn.execute("PRAGMA journal_mode;")
        active_mode = cursor.fetchone()[0]
        assert str(active_mode).lower() == "wal"


def test_get_alert_by_id(tmp_path: Path):
    """Verify get_alert_by_id returns specific alert row or None if missing."""
    db_file = tmp_path / "test_get_alert_id.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        alert = Alert(
            timestamp=100.5,
            rule_name="PORT_SCAN",
            src_ip="192.168.1.10",
            dst_ip="10.0.0.1",
            observed_value=15,
            threshold=10,
            window_seconds=30.0,
            severity="HIGH",
            message="Port scan detected",
        )
        alert_id = db.insert_alert(alert)
        assert alert_id == 1

        # Found alert
        fetched = db.get_alert_by_id(alert_id)
        assert fetched is not None
        assert fetched["id"] == alert_id
        assert fetched["rule_name"] == "PORT_SCAN"
        assert fetched["src_ip"] == "192.168.1.10"
        assert fetched["dst_ip"] == "10.0.0.1"
        assert fetched["observed_value"] == 15.0
        assert fetched["threshold"] == 10.0
        assert fetched["window_seconds"] == 30.0
        assert fetched["severity"] == "HIGH"
        assert fetched["message"] == "Port scan detected"

        # Non-existent alert
        assert db.get_alert_by_id(999) is None


def test_get_related_packets_filtering(tmp_path: Path):
    """Verify get_related_packets filters by IP pair, time interval, protocol, and handles empty sets."""
    db_file = tmp_path / "test_related_packets.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        pkts = [
            # Matching IP and time (10.0.0.1 -> 10.0.0.2, ts=100.0)
            NormalizedPacket(100.0, "10.0.0.1", "10.0.0.2", "TCP", 64, 1001, 80, "S"),
            # Matching IP and time (10.0.0.1 -> 10.0.0.2, ts=105.0)
            NormalizedPacket(105.0, "10.0.0.1", "10.0.0.2", "TCP", 64, 1002, 81, "S"),
            # Matching IP but outside time window (ts=150.0)
            NormalizedPacket(150.0, "10.0.0.1", "10.0.0.2", "TCP", 64, 1003, 82, "S"),
            # Different source IP (ts=102.0)
            NormalizedPacket(102.0, "10.0.0.99", "10.0.0.2", "TCP", 64, 1004, 80, "S"),
            # Different destination IP (ts=103.0)
            NormalizedPacket(103.0, "10.0.0.1", "10.0.0.99", "TCP", 64, 1005, 80, "S"),
            # UDP protocol
            NormalizedPacket(104.0, "10.0.0.1", "10.0.0.2", "UDP", 128, 5000, 53, None),
        ]
        db.insert_packets(pkts)

        # 1. Filter by IP pair and time range [100.0, 110.0]
        related = db.get_related_packets(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            start_time=100.0,
            end_time=110.0,
        )
        assert len(related) == 3  # Two TCP + One UDP
        assert [p["timestamp"] for p in related] == [100.0, 104.0, 105.0]

        # 2. Filter by IP pair, time range, and TCP protocol
        tcp_related = db.get_related_packets(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            start_time=100.0,
            end_time=110.0,
            protocol="TCP",
        )
        assert len(tcp_related) == 2
        assert [p["timestamp"] for p in tcp_related] == [100.0, 105.0]

        # 3. Non-matching IP pair returns empty list
        no_ip_match = db.get_related_packets(
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
        )
        assert no_ip_match == []

        # 4. Non-matching time range returns empty list
        no_time_match = db.get_related_packets(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            start_time=200.0,
            end_time=300.0,
        )
        assert no_time_match == []

        # 5. Parameterized query safety with special characters
        safe_query = db.get_related_packets(
            src_ip="10.0.0.1' OR '1'='1",
            dst_ip="10.0.0.2",
        )
        assert safe_query == []


def test_get_timestamp_bounds(tmp_path: Path):
    """Verify get_timestamp_bounds correctly reports bounds for empty and populated DB."""
    db_file = tmp_path / "test_bounds.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        # Empty DB returns None
        empty_bounds = db.get_timestamp_bounds()
        assert empty_bounds["min_timestamp"] is None
        assert empty_bounds["max_timestamp"] is None

        # Add packets and alert with known timestamps
        db.insert_packet(NormalizedPacket(50.0, "1.1.1.1", "2.2.2.2", "TCP", 64))
        db.insert_packet(NormalizedPacket(200.0, "1.1.1.1", "2.2.2.2", "TCP", 64))
        db.insert_alert(Alert(250.0, "RULE", "1.1.1.1", "2.2.2.2", 1, 1, 10.0, "LOW", "Msg"))

        bounds = db.get_timestamp_bounds()
        assert bounds["min_timestamp"] == 50.0
        assert bounds["max_timestamp"] == 250.0


def test_get_related_packets_none_ips_and_pagination(tmp_path: Path):
    """Verify get_related_packets handles None for src/dst IP (global query) and pagination."""
    db_file = tmp_path / "test_related_none_ip.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        pkts = [
            NormalizedPacket(10.0, "1.1.1.1", "2.2.2.2", "TCP", 100),
            NormalizedPacket(11.0, "3.3.3.3", "4.4.4.4", "UDP", 200),
            NormalizedPacket(12.0, "5.5.5.5", "6.6.6.6", "ICMP", 50),
        ]
        db.insert_packets(pkts)

        # Query with src_ip=None, dst_ip=None retrieves all packets within time bounds
        all_in_window = db.get_related_packets(
            src_ip=None,
            dst_ip=None,
            start_time=10.0,
            end_time=12.0,
        )
        assert len(all_in_window) == 3

        # Test limit and offset
        paginated = db.get_related_packets(
            src_ip=None,
            dst_ip=None,
            limit=2,
            offset=1,
            order="ASC",
        )
        assert len(paginated) == 2
        assert paginated[0]["timestamp"] == 11.0
        assert paginated[1]["timestamp"] == 12.0


def test_get_security_kpis_empty_and_populated(tmp_path: Path):
    """Verify get_security_kpis computes accurate metrics on empty and populated data."""
    db_file = tmp_path / "test_kpis.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        # 1. Empty database behavior
        empty_kpis = db.get_security_kpis()
        assert empty_kpis == {
            "total_packets": 0,
            "total_alerts": 0,
            "high_severity_alerts": 0,
            "distinct_alert_sources": 0,
        }

        # 2. Populate records
        db.insert_packets([
            NormalizedPacket(100.0, "10.0.0.1", "10.0.0.2", "TCP", 64),
            NormalizedPacket(101.0, "10.0.0.1", "10.0.0.2", "TCP", 128),
            NormalizedPacket(500.0, "10.0.0.3", "10.0.0.4", "UDP", 256),
        ])

        db.insert_alerts([
            # Alert 1: HIGH severity from 10.0.0.1
            Alert(100.0, "PORT_SCAN", "10.0.0.1", "10.0.0.2", 15, 10, 30.0, "HIGH", "Port scan"),
            # Alert 2: CRITICAL severity from 10.0.0.1 (duplicate source)
            Alert(101.0, "SYN_RATE", "10.0.0.1", "10.0.0.2", 25, 20, 10.0, "CRITICAL", "SYN flood"),
            # Alert 3: MEDIUM severity from 10.0.0.3 (distinct source)
            Alert(102.0, "PORT_SCAN", "10.0.0.3", "10.0.0.2", 12, 10, 30.0, "MEDIUM", "Port scan"),
            # Alert 4: LOW severity with NULL source (e.g. global volume anomaly)
            Alert(103.0, "TRAFFIC_VOLUME", None, None, 1000, 500, 10.0, "LOW", "Global volume"),
            # Alert 5: Outside initial time range (ts=500.0)
            Alert(500.0, "PORT_SCAN", "10.0.0.5", "10.0.0.2", 15, 10, 30.0, "HIGH", "Later scan"),
        ])

        # Full database KPIs
        kpis = db.get_security_kpis()
        assert kpis["total_packets"] == 3
        assert kpis["total_alerts"] == 5
        assert kpis["high_severity_alerts"] == 3  # 2 HIGH + 1 CRITICAL
        assert kpis["distinct_alert_sources"] == 3  # 10.0.0.1, 10.0.0.3, 10.0.0.5 (NULL safely excluded)

        # Scoped KPIs for [90.0, 110.0]
        scoped = db.get_security_kpis(start_time=90.0, end_time=110.0)
        assert scoped["total_packets"] == 2
        assert scoped["total_alerts"] == 4
        assert scoped["high_severity_alerts"] == 2  # Alert 1 and Alert 2
        assert scoped["distinct_alert_sources"] == 2  # 10.0.0.1, 10.0.0.3


def test_alert_analytics_aggregations_and_rankings(tmp_path: Path):
    """Verify severity distribution, rules breakdown, and top sources/destinations."""
    db_file = tmp_path / "test_analytics.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        alerts = [
            Alert(10.0, "PORT_SCAN", "1.1.1.1", "2.2.2.2", 15, 10, 30.0, "HIGH", "Msg"),
            Alert(20.0, "PORT_SCAN", "1.1.1.1", "2.2.2.2", 15, 10, 30.0, "HIGH", "Msg"),
            Alert(30.0, "SYN_RATE", "3.3.3.3", "2.2.2.2", 25, 20, 10.0, "MEDIUM", "Msg"),
            Alert(40.0, "PORT_SCAN", "1.1.1.1", "4.4.4.4", 15, 10, 30.0, "LOW", "Msg"),
        ]
        db.insert_alerts(alerts)

        # 1. Severity distribution
        sev_dist = db.get_alert_severity_distribution()
        sev_map = {item["severity"]: item["count"] for item in sev_dist}
        assert sev_map["HIGH"] == 2
        assert sev_map["MEDIUM"] == 1
        assert sev_map["LOW"] == 1

        # 2. Filtered severity distribution by rule
        syn_sev = db.get_alert_severity_distribution(rule="SYN_RATE")
        assert len(syn_sev) == 1
        assert syn_sev[0]["severity"] == "MEDIUM"

        # 3. Rules breakdown
        rule_dist = db.get_alerts_by_rule()
        rule_map = {item["rule_name"]: item["count"] for item in rule_dist}
        assert rule_map["PORT_SCAN"] == 3
        assert rule_map["SYN_RATE"] == 1

        # 4. Top alert sources
        top_srcs = db.get_top_alert_sources(limit=2)
        assert len(top_srcs) == 2
        assert top_srcs[0]["src_ip"] == "1.1.1.1"
        assert top_srcs[0]["count"] == 3
        assert top_srcs[1]["src_ip"] == "3.3.3.3"
        assert top_srcs[1]["count"] == 1

        # 5. Top alert destinations
        top_dsts = db.get_top_alert_destinations(limit=2)
        assert len(top_dsts) == 2
        assert top_dsts[0]["dst_ip"] == "2.2.2.2"
        assert top_dsts[0]["count"] == 3
        assert top_dsts[1]["dst_ip"] == "4.4.4.4"
        assert top_dsts[1]["count"] == 1


def test_timelines_and_traffic_analytics(tmp_path: Path):
    """Verify alert and traffic timeline bucketing, protocol distribution, and ports."""
    db_file = tmp_path / "test_traffic_analytics.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        # 1700000000 = 2023-11-14 22:13:20 UTC
        pkts = [
            NormalizedPacket(1700000000.0, "10.0.0.1", "10.0.0.2", "TCP", 100, 1024, 80, "S"),
            NormalizedPacket(1700000010.0, "10.0.0.1", "10.0.0.2", "TCP", 200, 1025, 80, "A"),
            NormalizedPacket(1700000020.0, "10.0.0.1", "10.0.0.2", "UDP", 300, 2000, 53, None),
            NormalizedPacket(1700003600.0, "10.0.0.3", "10.0.0.2", "ICMP", 50, None, None, None),
        ]
        db.insert_packets(pkts)

        alerts = [
            Alert(1700000000.0, "PORT_SCAN", "10.0.0.1", "10.0.0.2", 15, 10, 30.0, "HIGH", "Msg 1"),
            Alert(1700000010.0, "PORT_SCAN", "10.0.0.1", "10.0.0.2", 16, 10, 30.0, "HIGH", "Msg 2"),
        ]
        db.insert_alerts(alerts)

        # 1. Alert timeline
        alert_tl = db.get_alert_timeline(bucket_resolution="hour")
        assert len(alert_tl) == 1
        assert "2023-11-14 22:00 UTC" in alert_tl[0]["bucket"]
        assert alert_tl[0]["count"] == 2

        # 2. Protocol distribution
        proto_dist = db.get_protocol_distribution()
        proto_map = {item["protocol"]: (item["count"], item["total_bytes"]) for item in proto_dist}
        assert proto_map["TCP"] == (2, 300)
        assert proto_map["UDP"] == (1, 300)
        assert proto_map["ICMP"] == (1, 50)

        # 3. Traffic timeline
        traffic_tl = db.get_traffic_timeline(bucket_resolution="hour")
        assert len(traffic_tl) >= 1
        assert sum(b["packet_count"] for b in traffic_tl) == 4
        assert sum(b["total_bytes"] for b in traffic_tl) == 650

        # 4. Top destination ports
        top_ports = db.get_top_destination_ports(limit=5)
        port_map = {item["dst_port"]: item["packet_count"] for item in top_ports}
        assert port_map[80] == 2
        assert port_map[53] == 1

        # 5. Top packet destinations
        top_dsts = db.get_top_packet_destinations(limit=5)
        assert len(top_dsts) == 1
        assert top_dsts[0]["dst_ip"] == "10.0.0.2"
        assert top_dsts[0]["packet_count"] == 4
        assert top_dsts[0]["total_bytes"] == 650


def test_protocol_distribution_filtering(tmp_path: Path):
    """Verify get_protocol_distribution respects protocol filters and case-insensitivity."""
    db_file = tmp_path / "proto_filter.db"
    with DatabaseManager(db_file) as db:
        db.init_db()
        db.insert_packets([
            NormalizedPacket(100.0, "1.1.1.1", "2.2.2.2", "TCP", 100),
            NormalizedPacket(101.0, "1.1.1.1", "2.2.2.2", "TCP", 200),
            NormalizedPacket(102.0, "1.1.1.1", "2.2.2.2", "UDP", 300),
            NormalizedPacket(103.0, "1.1.1.1", "2.2.2.2", "ICMP", 50),
        ])

        # Filter by TCP
        tcp_dist = db.get_protocol_distribution(protocol="TCP")
        assert len(tcp_dist) == 1
        assert tcp_dist[0]["protocol"] == "TCP"
        assert tcp_dist[0]["count"] == 2
        assert tcp_dist[0]["total_bytes"] == 300

        # Filter by lowercase udp
        udp_dist = db.get_protocol_distribution(protocol="udp")
        assert len(udp_dist) == 1
        assert udp_dist[0]["protocol"] == "UDP"
        assert udp_dist[0]["count"] == 1
        assert udp_dist[0]["total_bytes"] == 300

        # Filter by All returns all 3
        all_dist = db.get_protocol_distribution(protocol="All")
        assert len(all_dist) == 3

        # Filter by non-existent protocol returns empty list
        none_dist = db.get_protocol_distribution(protocol="SCTP")
        assert len(none_dist) == 0


def test_timeline_sparse_single_point_and_empty(tmp_path: Path):
    """Verify timeline aggregation handles empty, single-point, and sparse multi-year records."""
    db_file = tmp_path / "sparse_timeline.db"
    with DatabaseManager(db_file) as db:
        db.init_db()

        # 1. Empty database returns empty list
        assert db.get_alert_timeline() == []
        assert db.get_traffic_timeline() == []

        # 2. Single-point dataset (span = 0)
        db.insert_packet(NormalizedPacket(1700000000.0, "1.1.1.1", "2.2.2.2", "TCP", 100))
        db.insert_alert(Alert(1700000000.0, "PORT_SCAN", "1.1.1.1", "2.2.2.2", 15, 10, 30.0, "HIGH", "Single"))

        single_alert_tl = db.get_alert_timeline(bucket_resolution="auto")
        assert len(single_alert_tl) == 1
        assert single_alert_tl[0]["count"] == 1

        single_traffic_tl = db.get_traffic_timeline(bucket_resolution="auto")
        assert len(single_traffic_tl) == 1
        assert single_traffic_tl[0]["packet_count"] == 1

        # 3. Sparse multi-year gap dataset (e.g. 2015 and 2026 timestamps)
        # 1445353834 = 2015-10-20, 1792165244 = 2026-10-16
        db.insert_packet(NormalizedPacket(1445353834.0, "10.0.0.1", "10.0.0.2", "TCP", 64))
        db.insert_alert(Alert(1445353834.0, "PORT_SCAN", "10.0.0.1", "10.0.0.2", 15, 10, 30.0, "HIGH", "Old"))

        sparse_alert_tl = db.get_alert_timeline(bucket_resolution="auto")
        # Exactly 2 discrete buckets should be returned — no fabricated continuous intermediate points
        assert len(sparse_alert_tl) == 2
        buckets = [b["bucket"] for b in sparse_alert_tl]
        assert any("2015-10-20" in b for b in buckets)
        assert any("2023-11-14" in b for b in buckets)

        sparse_traffic_tl = db.get_traffic_timeline(bucket_resolution="auto")
        assert len(sparse_traffic_tl) == 2


