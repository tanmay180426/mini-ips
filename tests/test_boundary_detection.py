"""Repeatable boundary and edge-condition tests for Mini-IDS detection rules.

Validates exact threshold boundaries, window expiration, and severity transitions:
- PortScanDetector:
  - Immediately below distinct-port threshold (N - 1) produces no alert.
  - Exactly at distinct-port threshold (N) triggers a PORT_SCAN alert.
  - Sliding time window boundary: expired packets outside window do not count.
  - Duplicate destination port packets do not increment distinct count at boundary.
  - High severity boundary transition at (2 * threshold).
- SynRateDetector:
  - Immediately below initial SYN threshold (N - 1) produces no alert.
  - Exactly at initial SYN threshold (N) triggers a SYN_RATE alert within window.
  - Sliding time window boundary: SYN packets outside window do not trigger early.
  - Non-initial SYN packets (SYN-ACK, ACK, UDP) interleaved at boundary do not count.
  - High severity boundary transition at (2 * threshold).
- End-to-end PCAP Boundary Integration:
  - Validates full pipeline (iter_pcap -> normalization -> detection -> SQLite)
    at sub-threshold (14 ports) vs at-threshold (15 ports) in isolated tmp_path.
"""

from pathlib import Path
from typing import List, Optional
import pytest
from scapy.all import Ether, IP, TCP, wrpcap

from capture.packet_capture import NormalizedPacket
from detection import (
    Alert,
    PortScanDetector,
    SynRateDetector,
)
from main import process_pcap_stream
from storage.database import DatabaseManager


def make_test_packet(
    timestamp: float,
    src_ip: str = "192.168.1.50",
    dst_ip: str = "192.168.1.10",
    protocol: str = "TCP",
    length: int = 64,
    src_port: int = 50000,
    dst_port: int = 80,
    tcp_flags: Optional[str] = "S",
) -> NormalizedPacket:
    """Helper to synthesize a NormalizedPacket with deterministic values."""
    return NormalizedPacket(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        protocol=protocol,
        length=length,
        src_port=src_port,
        dst_port=dst_port,
        tcp_flags=tcp_flags,
    )


# ============================================================================
# 1. PortScanDetector Boundary Tests
# ============================================================================


def test_port_scan_default_threshold_sub_boundary():
    """Verify default threshold - 1 (14 distinct ports) triggers zero alerts."""
    detector = PortScanDetector()  # Default: threshold=15, window=30.0s
    assert detector.threshold == 15
    assert detector.window_seconds == 30.0

    alerts: List[Alert] = []
    # Send 14 distinct ports, 1 second apart (timestamps 100.0 to 113.0)
    for port in range(1, 15):
        pkt = make_test_packet(
            timestamp=100.0 + port,
            dst_port=port,
            src_port=40000 + port,
        )
        alert = detector.process_packet(pkt)
        if alert is not None:
            alerts.append(alert)

    # 14 < 15: No alert should be generated
    assert len(alerts) == 0


def test_port_scan_default_threshold_exact_boundary():
    """Verify exactly 15 distinct ports (default threshold) triggers exactly one alert on the 15th port."""
    detector = PortScanDetector()  # threshold=15
    alerts: List[Alert] = []

    # First 14 distinct ports
    for port in range(1, 15):
        pkt = make_test_packet(
            timestamp=100.0 + port,
            dst_port=port,
            src_port=40000 + port,
        )
        assert detector.process_packet(pkt) is None

    # 15th distinct port (timestamp 115.0) hits the exact boundary
    pkt_15 = make_test_packet(
        timestamp=115.0,
        dst_port=15,
        src_port=40015,
    )
    alert = detector.process_packet(pkt_15)

    assert alert is not None
    assert isinstance(alert, Alert)
    assert alert.rule_name == "PORT_SCAN"
    assert alert.src_ip == "192.168.1.50"
    assert alert.dst_ip == "192.168.1.10"
    assert alert.observed_value == 15
    assert alert.threshold == 15
    assert alert.window_seconds == 30.0
    assert alert.severity == "MEDIUM"
    assert "15 distinct ports contacted" in alert.message

    # 16th distinct port within same active window must be suppressed
    pkt_16 = make_test_packet(
        timestamp=116.0,
        dst_port=16,
        src_port=40016,
    )
    assert detector.process_packet(pkt_16) is None


def test_port_scan_duplicates_at_sub_boundary_do_not_trigger():
    """Verify sending repeated packets to 14 distinct ports (total > 15 packets) still does not trigger."""
    detector = PortScanDetector(threshold=15, window_seconds=30.0)

    # Send 30 packets, but only cycling through 14 distinct destination ports
    alerts: List[Alert] = []
    for i in range(30):
        port = (i % 14) + 1  # Always in range 1..14
        pkt = make_test_packet(
            timestamp=100.0 + (i * 0.5),
            dst_port=port,
            src_port=45000 + i,
        )
        alert = detector.process_packet(pkt)
        if alert is not None:
            alerts.append(alert)

    # Distinct count is strictly 14 < 15
    assert len(alerts) == 0


def test_port_scan_time_window_boundary_expiration():
    """Verify that distinct ports outside the rolling window expire and do not count toward the boundary."""
    detector = PortScanDetector(threshold=5, window_seconds=10.0)

    # Port 1 at timestamp 100.0
    assert detector.process_packet(make_test_packet(timestamp=100.0, dst_port=1)) is None

    # Ports 2, 3, 4, 5 arrive at timestamp 111.0, 112.0, 113.0, 114.0
    # At t=111.0, cutoff is 111.0 - 10.0 = 101.0 -> Port 1 (t=100.0) is expired!
    # Active window contains:
    # At t=111.0: {2} (count=1)
    # At t=112.0: {2, 3} (count=2)
    # At t=113.0: {2, 3, 4} (count=3)
    # At t=114.0: {2, 3, 4, 5} (count=4 < threshold 5)
    for port in [2, 3, 4, 5]:
        pkt = make_test_packet(timestamp=110.0 + (port - 1), dst_port=port)
        assert detector.process_packet(pkt) is None

    # Even though 5 packets were sent in total across the stream, only 4 are in the 10s window
    # Now send 5th distinct port within the active window (t=115.0, port 6)
    pkt_6 = make_test_packet(timestamp=115.0, dst_port=6)
    alert = detector.process_packet(pkt_6)

    # Window now contains {2, 3, 4, 5, 6} (count=5 >= 5)
    assert alert is not None
    assert alert.observed_value == 5
    assert alert.threshold == 5


def test_port_scan_initial_threshold_severity_is_medium():
    """Verify PortScanDetector assigns MEDIUM severity when crossing the initial threshold (distinct_ports < 2*threshold)."""
    detector = PortScanDetector(threshold=5, window_seconds=10.0)

    alert = None
    for port in range(1, 6):
        res = detector.process_packet(
            make_test_packet(timestamp=100.0 + (port - 1), dst_port=port)
        )
        if res is not None:
            alert = res

    assert alert is not None
    assert alert.rule_name == "PORT_SCAN"
    assert alert.observed_value == 5
    assert alert.threshold == 5
    assert alert.severity == "MEDIUM"


def test_port_scan_high_severity_transition():
    """Verify PortScanDetector escalates severity from MEDIUM to HIGH when distinct ports >= 2*threshold.

    Behavior under test:
    - Initial alert triggers at threshold (N=5) with severity='MEDIUM'.
    - Additional distinct ports arriving during the active suppression window return None.
    - When the rolling window elapses (current_time >= last_alert_time + window_seconds),
      suppression resets.
    - If the active window contains >= 2*threshold distinct ports (here N=10),
      the new alert is emitted with severity='HIGH' and observed_value=10.
    """
    detector = PortScanDetector(threshold=5, window_seconds=10.0)

    # Phase 1: Reach initial threshold (5 distinct ports at t=100.0..104.0)
    initial_alert = None
    for port in range(1, 6):
        res = detector.process_packet(
            make_test_packet(timestamp=100.0 + (port - 1), dst_port=port)
        )
        if res is not None:
            initial_alert = res

    assert initial_alert is not None
    assert initial_alert.observed_value == 5
    assert initial_alert.severity == "MEDIUM"

    # Phase 2: Escalate scan during active suppression (ports 6..13 at t=105.0..112.0)
    # All 8 packets arrive during suppression (t < 104.0 + 10.0 = 114.0) and must return None
    for port in range(6, 14):
        pkt = make_test_packet(timestamp=100.0 + (port - 1), dst_port=port)
        assert detector.process_packet(pkt) is None

    # Phase 3: At t=114.0, a full window (10.0s) has elapsed since last_alert_time (104.0).
    # Suppression resets.
    # History cutoff is 114.0 - 10.0 = 104.0.
    # Retained in active window:
    # - port 5 (t=104.0)
    # - ports 6..13 (t=105.0..112.0, 8 ports)
    # - port 14 (t=114.0, 1 port)
    # Total distinct ports in active window: {5, 6, 7, 8, 9, 10, 11, 12, 13, 14} -> exactly 10 ports.
    # 10 >= 2 * 5 (2 * threshold) -> triggers HIGH severity alert.
    pkt_high = make_test_packet(timestamp=114.0, dst_port=14)
    high_alert = detector.process_packet(pkt_high)

    assert high_alert is not None
    assert isinstance(high_alert, Alert)
    assert high_alert.rule_name == "PORT_SCAN"
    assert high_alert.observed_value == 10
    assert high_alert.threshold == 5
    assert high_alert.severity == "HIGH"
    assert "10 distinct ports contacted" in high_alert.message


# ============================================================================
# 2. SynRateDetector Boundary Tests
# ============================================================================


def test_syn_rate_default_threshold_sub_boundary():
    """Verify default threshold - 1 (19 initial SYN packets) triggers zero alerts."""
    detector = SynRateDetector()  # Default: threshold=20, window=10.0s
    assert detector.threshold == 20
    assert detector.window_seconds == 10.0

    alerts: List[Alert] = []
    # Send 19 initial SYN packets spaced 0.4s apart (total span: 18 * 0.4 = 7.2s < 10.0s)
    for i in range(19):
        pkt = make_test_packet(
            timestamp=200.0 + (i * 0.4),
            tcp_flags="S",
            dst_port=80,
            src_port=50000 + i,
        )
        alert = detector.process_packet(pkt)
        if alert is not None:
            alerts.append(alert)

    # 19 < 20: No alert
    assert len(alerts) == 0


def test_syn_rate_default_threshold_exact_boundary():
    """Verify exactly 20 initial SYN packets (default threshold) triggers exactly one alert on the 20th packet."""
    detector = SynRateDetector()  # threshold=20, window=10.0s

    # Packets 1 to 19 must not alert
    for i in range(19):
        pkt = make_test_packet(
            timestamp=200.0 + (i * 0.4),
            tcp_flags="S",
            dst_port=80,
            src_port=50000 + i,
        )
        assert detector.process_packet(pkt) is None

    # 20th packet (timestamp 207.6s, within the 10.0s window from 200.0)
    pkt_20 = make_test_packet(
        timestamp=200.0 + (19 * 0.4),
        tcp_flags="S",
        dst_port=80,
        src_port=50019,
    )
    alert = detector.process_packet(pkt_20)

    assert alert is not None
    assert isinstance(alert, Alert)
    assert alert.rule_name == "SYN_RATE"
    assert alert.src_ip == "192.168.1.50"
    assert alert.dst_ip == "192.168.1.10"
    assert alert.observed_value == 20
    assert alert.threshold == 20
    assert alert.window_seconds == 10.0
    assert alert.severity == "MEDIUM"
    assert "20 SYN packets in 10.0s" in alert.message

    # 21st packet in same window must be suppressed
    pkt_21 = make_test_packet(
        timestamp=208.0,
        tcp_flags="S",
        dst_port=80,
        src_port=50020,
    )
    assert detector.process_packet(pkt_21) is None


def test_syn_rate_non_syn_interleaved_at_boundary():
    """Verify interleaved non-initial-SYN packets (SYN-ACK, ACK, UDP) do not inflate the SYN boundary."""
    detector = SynRateDetector(threshold=5, window_seconds=10.0)

    # Interleave 4 valid SYNs with 10 non-SYN packets (SYN-ACK, ACK, UDP)
    # Total packets: 14, but valid initial SYNs: 4 (< 5 threshold)
    packets = [
        make_test_packet(timestamp=300.0, tcp_flags="S"),       # SYN 1
        make_test_packet(timestamp=300.5, tcp_flags="SA"),      # Ignored (SYN-ACK)
        make_test_packet(timestamp=301.0, tcp_flags="A"),       # Ignored (ACK)
        make_test_packet(timestamp=301.5, tcp_flags="S"),       # SYN 2
        make_test_packet(timestamp=302.0, protocol="UDP", tcp_flags=None),  # Ignored (UDP)
        make_test_packet(timestamp=302.5, tcp_flags="FA"),      # Ignored (FIN-ACK)
        make_test_packet(timestamp=303.0, tcp_flags="S"),       # SYN 3
        make_test_packet(timestamp=303.5, tcp_flags="SA"),      # Ignored (SYN-ACK)
        make_test_packet(timestamp=304.0, tcp_flags="A"),       # Ignored (ACK)
        make_test_packet(timestamp=304.5, tcp_flags="S"),       # SYN 4
        make_test_packet(timestamp=305.0, tcp_flags="R"),       # Ignored (RST)
    ]

    for pkt in packets:
        assert detector.process_packet(pkt) is None

    # 5th valid initial SYN triggers the alert
    pkt_syn_5 = make_test_packet(timestamp=305.5, tcp_flags="S")
    alert = detector.process_packet(pkt_syn_5)

    assert alert is not None
    assert alert.rule_name == "SYN_RATE"
    assert alert.observed_value == 5
    assert alert.threshold == 5


def test_syn_rate_time_window_boundary_expiration():
    """Verify that SYN packets older than the rolling window expire and do not trigger boundary alerts prematurely."""
    detector = SynRateDetector(threshold=5, window_seconds=5.0)

    # Packet 1 at timestamp 400.0
    assert detector.process_packet(make_test_packet(timestamp=400.0, tcp_flags="S")) is None

    # Packets 2, 3, 4, 5 arrive at t=406.0, 406.5, 407.0, 407.5
    # At t=406.0, cutoff is 406.0 - 5.0 = 401.0 -> Packet 1 (t=400.0) is expired!
    # Active window contains:
    # At t=406.0: 1 SYN
    # At t=406.5: 2 SYNs
    # At t=407.0: 3 SYNs
    # At t=407.5: 4 SYNs (< 5)
    for i in range(4):
        pkt = make_test_packet(timestamp=406.0 + (i * 0.5), tcp_flags="S")
        assert detector.process_packet(pkt) is None

    # 5th SYN within active window arrives at t=408.0
    # Active window [403.0, 408.0] now contains 5 SYNs
    pkt_final = make_test_packet(timestamp=408.0, tcp_flags="S")
    alert = detector.process_packet(pkt_final)

    assert alert is not None
    assert alert.observed_value == 5
    assert alert.threshold == 5


def test_syn_rate_initial_threshold_severity_is_medium():
    """Verify SynRateDetector assigns MEDIUM severity when crossing the initial threshold (syn_count < 2*threshold)."""
    detector = SynRateDetector(threshold=5, window_seconds=10.0)

    alert = None
    for i in range(5):
        res = detector.process_packet(
            make_test_packet(timestamp=200.0 + i, tcp_flags="S", src_port=50000 + i)
        )
        if res is not None:
            alert = res

    assert alert is not None
    assert alert.rule_name == "SYN_RATE"
    assert alert.observed_value == 5
    assert alert.threshold == 5
    assert alert.severity == "MEDIUM"


def test_syn_rate_high_severity_transition():
    """Verify SynRateDetector escalates severity from MEDIUM to HIGH when SYN count >= 2*threshold.

    Behavior under test:
    - Initial alert triggers at threshold (N=5) with severity='MEDIUM'.
    - Subsequent SYN packets arriving during the active suppression window return None.
    - When the rolling window elapses (current_time >= last_alert_time + window_seconds),
      suppression resets.
    - If the active window contains >= 2*threshold SYN packets (here N=10),
      the new alert is emitted with severity='HIGH' and observed_value=10.
    """
    detector = SynRateDetector(threshold=5, window_seconds=10.0)

    # Phase 1: Reach initial threshold (5 initial SYNs at t=200.0..204.0)
    initial_alert = None
    for i in range(5):
        res = detector.process_packet(
            make_test_packet(timestamp=200.0 + i, tcp_flags="S", src_port=50000 + i)
        )
        if res is not None:
            initial_alert = res

    assert initial_alert is not None
    assert initial_alert.observed_value == 5
    assert initial_alert.severity == "MEDIUM"

    # Phase 2: Escalate SYN rate during active suppression (8 SYNs at t=205.0..212.0)
    # All 8 packets arrive during suppression (t < 204.0 + 10.0 = 214.0) and must return None
    for i in range(5, 13):
        pkt = make_test_packet(timestamp=200.0 + i, tcp_flags="S", src_port=50000 + i)
        assert detector.process_packet(pkt) is None

    # Phase 3: At t=214.0, a full window (10.0s) has elapsed since last_alert_time (204.0).
    # Suppression resets.
    # History cutoff is 214.0 - 10.0 = 204.0.
    # Retained in active window:
    # - packet 5 (t=204.0)
    # - packets 6..13 (t=205.0..212.0, 8 packets)
    # - packet 14 (t=214.0, 1 packet)
    # Total SYNs in active window: 1 + 8 + 1 = 10 SYNs.
    # 10 >= 2 * 5 (2 * threshold) -> triggers HIGH severity alert.
    pkt_high = make_test_packet(timestamp=214.0, tcp_flags="S", src_port=50013)
    high_alert = detector.process_packet(pkt_high)

    assert high_alert is not None
    assert isinstance(high_alert, Alert)
    assert high_alert.rule_name == "SYN_RATE"
    assert high_alert.observed_value == 10
    assert high_alert.threshold == 5
    assert high_alert.severity == "HIGH"
    assert "10 SYN packets in 10.0s" in high_alert.message


# ============================================================================
# 3. End-to-End PCAP Boundary Tests (Isolated via tmp_path)
# ============================================================================


def test_pcap_port_scan_boundary_end_to_end(tmp_path: Path):
    """Verify offline PCAP analysis at boundary using synthetic Scapy PCAP files in an isolated tmp_path.

    Tests that:
    - 14 distinct ports in PCAP produces 0 PORT_SCAN alerts.
    - 15 distinct ports in PCAP produces exactly 1 PORT_SCAN alert stored in SQLite.
    """
    mac_src = "00:11:22:33:44:55"
    mac_dst = "aa:bb:cc:dd:ee:ff"
    src_ip = "192.168.1.50"
    dst_ip = "192.168.1.10"

    # --- Sub-boundary: 14 distinct ports ---
    sub_pcap = tmp_path / "boundary_sub_14_ports.pcap"
    sub_db = tmp_path / "boundary_sub.db"

    sub_packets = []
    for port in range(1, 15):
        pkt = (
            Ether(src=mac_src, dst=mac_dst)
            / IP(src=src_ip, dst=dst_ip)
            / TCP(sport=50000 + port, dport=port, flags="S")
        )
        pkt.time = 100.0 + port
        sub_packets.append(pkt)
    wrpcap(str(sub_pcap), sub_packets)

    sub_summary = process_pcap_stream(
        pcap_path=sub_pcap,
        db_path=sub_db,
        print_alerts=False,
    )
    assert sub_summary["packets_processed"] == 14
    assert sub_summary["alerts_generated"] == 0
    with DatabaseManager(sub_db) as db:
        assert db.get_packet_count() == 14
        assert db.get_alert_count() == 0

    # --- Exact boundary: 15 distinct ports ---
    exact_pcap = tmp_path / "boundary_exact_15_ports.pcap"
    exact_db = tmp_path / "boundary_exact.db"

    exact_packets = []
    for port in range(1, 16):
        pkt = (
            Ether(src=mac_src, dst=mac_dst)
            / IP(src=src_ip, dst=dst_ip)
            / TCP(sport=50000 + port, dport=port, flags="S")
        )
        pkt.time = 100.0 + port
        exact_packets.append(pkt)
    wrpcap(str(exact_pcap), exact_packets)

    exact_summary = process_pcap_stream(
        pcap_path=exact_pcap,
        db_path=exact_db,
        print_alerts=False,
    )
    assert exact_summary["packets_processed"] == 15
    assert exact_summary["alerts_generated"] == 1
    assert exact_summary["rule_counts"]["PORT_SCAN"] == 1

    with DatabaseManager(exact_db) as db:
        assert db.get_packet_count() == 15
        assert db.get_alert_count() == 1
        alerts = db.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["rule_name"] == "PORT_SCAN"
        assert alerts[0]["observed_value"] == 15
        assert alerts[0]["threshold"] == 15
        assert alerts[0]["src_ip"] == src_ip
        assert alerts[0]["dst_ip"] == dst_ip
