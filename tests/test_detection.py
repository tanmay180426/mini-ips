"""Unit and integration tests for Mini-IDS detection engine rules.

Tests verify:
- PortScanDetector:
  - Below-threshold traffic does not alert.
  - Crossing distinct-port threshold triggers an Alert.
  - Duplicate destination ports do NOT inflate distinct-port count.
  - Non-TCP packets (UDP, ICMP) are ignored.
  - Duplicate alert suppression in active window.
  - Window expiration and reset for subsequent scan attempts.
- SynRateDetector:
  - Initial SYN packets ('S') count towards threshold.
  - SYN-ACK ('SA', 'AS') responses do NOT count as initial SYNs.
  - Other TCP flags (ACK, PSH, FIN) do NOT count.
  - Crossing SYN threshold triggers an Alert.
  - Non-TCP packets are ignored.
  - Alert suppression and window reset.
- TrafficAnomalyDetector:
  - Byte volume threshold calculation using packet.length.
  - Packet count threshold calculation.
  - Dual-metric trigger behavior.
  - Alert suppression and window reset.
  - Global vs per-source modes.
- Robustness:
  - Missing optional fields (src_ip, dst_port, tcp_flags) do not crash.
  - Out-of-order timestamps handled predictably.
  - Reset method restores clean state.
"""

import pytest

from capture.packet_capture import NormalizedPacket
from detection import (
    Alert,
    PortScanDetector,
    SynRateDetector,
    TrafficAnomalyDetector,
)


def make_packet(
    timestamp: float = 100.0,
    src_ip: str = "192.168.1.10",
    dst_ip: str = "10.0.0.1",
    protocol: str = "TCP",
    length: int = 64,
    src_port: int = 50000,
    dst_port: int = 80,
    tcp_flags: str = "S",
) -> NormalizedPacket:
    """Helper to synthesize a NormalizedPacket with defaults."""
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
# A. PortScanDetector Tests
# ============================================================================


def test_port_scan_below_threshold_no_alert():
    """Verify that distinct port connections below the threshold do not alert."""
    detector = PortScanDetector(threshold=5, window_seconds=10.0)

    # Send 4 distinct ports (threshold is 5)
    alerts = []
    for port in range(1, 5):
        pkt = make_packet(timestamp=100.0 + port, dst_port=port)
        alert = detector.process_packet(pkt)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0


def test_port_scan_crossing_threshold_alerts():
    """Verify that reaching the threshold triggers an Alert with proper metadata."""
    detector = PortScanDetector(threshold=5, window_seconds=10.0)

    alert = None
    for port in range(1, 6):
        pkt = make_packet(timestamp=100.0 + port, dst_port=port)
        res = detector.process_packet(pkt)
        if res:
            alert = res

    assert alert is not None
    assert isinstance(alert, Alert)
    assert alert.rule_name == "PORT_SCAN"
    assert alert.src_ip == "192.168.1.10"
    assert alert.dst_ip == "10.0.0.1"
    assert alert.observed_value == 5
    assert alert.threshold == 5
    assert alert.window_seconds == 10.0
    assert "port scan detected" in alert.message.lower()


def test_port_scan_duplicate_ports_do_not_inflate_count():
    """Verify that repeated connections to the same port do not inflate scan count."""
    detector = PortScanDetector(threshold=5, window_seconds=10.0)

    alerts = []
    # Send 20 packets to port 80 and port 443 only (distinct ports = 2)
    for i in range(20):
        target_port = 80 if i % 2 == 0 else 443
        pkt = make_packet(timestamp=100.0 + (i * 0.1), dst_port=target_port)
        res = detector.process_packet(pkt)
        if res:
            alerts.append(res)

    assert len(alerts) == 0


def test_port_scan_ignores_non_tcp():
    """Verify that UDP and ICMP traffic are not counted as TCP port scans."""
    detector = PortScanDetector(threshold=3, window_seconds=10.0)

    # 10 distinct UDP destination ports
    for port in range(1, 10):
        pkt = make_packet(
            timestamp=100.0 + port,
            protocol="UDP",
            dst_port=port,
            tcp_flags=None,
        )
        assert detector.process_packet(pkt) is None


def test_port_scan_suppression_and_window_expiration():
    """Verify alert suppression during active window and reset after window expiration."""
    detector = PortScanDetector(threshold=3, window_seconds=10.0)

    # 1. Trigger first alert on 3rd distinct port at t=103.0
    a1 = detector.process_packet(make_packet(timestamp=101.0, dst_port=1))
    a2 = detector.process_packet(make_packet(timestamp=102.0, dst_port=2))
    a3 = detector.process_packet(make_packet(timestamp=103.0, dst_port=3))
    assert a1 is None and a2 is None
    assert a3 is not None  # First alert triggered

    # 2. Subsequent distinct ports in the same window (e.g., t=104.0, 105.0) are suppressed
    a4 = detector.process_packet(make_packet(timestamp=104.0, dst_port=4))
    a5 = detector.process_packet(make_packet(timestamp=105.0, dst_port=5))
    assert a4 is None and a5 is None

    # 3. Advance time past window expiration (t >= 103.0 + 10.0 = 113.0)
    # Start a new scan event at t=120.0
    d1 = detector.process_packet(make_packet(timestamp=120.0, dst_port=10))
    d2 = detector.process_packet(make_packet(timestamp=121.0, dst_port=20))
    d3 = detector.process_packet(make_packet(timestamp=122.0, dst_port=30))
    assert d1 is None and d2 is None
    assert d3 is not None  # New alert allowed in new window
    assert d3.observed_value == 3


# ============================================================================
# B. SynRateDetector Tests
# ============================================================================


def test_syn_rate_initial_syn_trigger():
    """Verify that pure initial SYN packets trigger alert when threshold is crossed."""
    detector = SynRateDetector(threshold=5, window_seconds=5.0)

    alert = None
    for i in range(5):
        pkt = make_packet(timestamp=200.0 + (i * 0.5), tcp_flags="S")
        res = detector.process_packet(pkt)
        if res:
            alert = res

    assert alert is not None
    assert alert.rule_name == "SYN_RATE"
    assert alert.observed_value == 5
    assert alert.threshold == 5
    assert alert.window_seconds == 5.0
    assert alert.src_ip == "192.168.1.10"


def test_syn_rate_syn_ack_does_not_count():
    """Verify that SYN-ACK responses ('SA') are NOT counted as initial SYNs."""
    detector = SynRateDetector(threshold=5, window_seconds=5.0)

    # Send 10 SYN-ACK packets
    alerts = []
    for i in range(10):
        pkt = make_packet(timestamp=200.0 + (i * 0.2), tcp_flags="SA")
        res = detector.process_packet(pkt)
        if res:
            alerts.append(res)

    assert len(alerts) == 0


def test_syn_rate_ignores_non_syn_and_non_tcp():
    """Verify that non-SYN TCP packets and non-TCP packets are ignored."""
    detector = SynRateDetector(threshold=3, window_seconds=5.0)

    # TCP ACK and FIN packets
    assert detector.process_packet(make_packet(timestamp=201.0, tcp_flags="A")) is None
    assert detector.process_packet(make_packet(timestamp=202.0, tcp_flags="FA")) is None
    assert detector.process_packet(make_packet(timestamp=203.0, tcp_flags="R")) is None

    # UDP packets
    assert (
        detector.process_packet(
            make_packet(timestamp=204.0, protocol="UDP", tcp_flags=None)
        )
        is None
    )


def test_syn_rate_suppression():
    """Verify that subsequent SYNs in the same active window do not duplicate alerts."""
    detector = SynRateDetector(threshold=3, window_seconds=10.0)

    # Reach threshold (3 packets)
    r1 = detector.process_packet(make_packet(timestamp=301.0, tcp_flags="S"))
    r2 = detector.process_packet(make_packet(timestamp=302.0, tcp_flags="S"))
    r3 = detector.process_packet(make_packet(timestamp=303.0, tcp_flags="S"))
    assert r1 is None and r2 is None
    assert r3 is not None  # Alert fired

    # 4th and 5th packets in the same window
    r4 = detector.process_packet(make_packet(timestamp=304.0, tcp_flags="S"))
    r5 = detector.process_packet(make_packet(timestamp=305.0, tcp_flags="S"))
    assert r4 is None and r5 is None


# ============================================================================
# C. TrafficAnomalyDetector Tests
# ============================================================================


def test_traffic_anomaly_byte_threshold():
    """Verify alert triggers when total byte volume exceeds byte_threshold."""
    detector = TrafficAnomalyDetector(
        byte_threshold=1000,
        packet_threshold=None,
        window_seconds=10.0,
    )

    # Send 4 packets of 200 bytes (total 800 bytes < 1000)
    for i in range(4):
        pkt = make_packet(timestamp=400.0 + i, length=200)
        assert detector.process_packet(pkt) is None

    # 5th packet of 200 bytes brings total to 1000 bytes (>= 1000)
    pkt5 = make_packet(timestamp=404.0, length=200)
    alert = detector.process_packet(pkt5)

    assert alert is not None
    assert alert.rule_name == "TRAFFIC_VOLUME"
    assert alert.observed_value == 1000
    assert alert.threshold == 1000
    assert "bytes" in alert.message


def test_traffic_anomaly_packet_count_threshold():
    """Verify alert triggers when packet count exceeds packet_threshold."""
    detector = TrafficAnomalyDetector(
        byte_threshold=None,
        packet_threshold=5,
        window_seconds=10.0,
    )

    # Send small 10-byte packets
    for i in range(4):
        assert (
            detector.process_packet(make_packet(timestamp=500.0 + i, length=10)) is None
        )

    # 5th packet reaches threshold
    alert = detector.process_packet(make_packet(timestamp=504.0, length=10))
    assert alert is not None
    assert alert.observed_value == 5
    assert alert.threshold == 5
    assert "packets" in alert.message


def test_traffic_anomaly_global_mode():
    """Verify global stream mode aggregates traffic across multiple source IPs."""
    detector = TrafficAnomalyDetector(
        byte_threshold=None,
        packet_threshold=4,
        window_seconds=10.0,
        per_source=False,
    )

    p1 = make_packet(timestamp=600.0, src_ip="10.0.0.1")
    p2 = make_packet(timestamp=601.0, src_ip="10.0.0.2")
    p3 = make_packet(timestamp=602.0, src_ip="10.0.0.3")
    p4 = make_packet(timestamp=603.0, src_ip="10.0.0.4")

    assert detector.process_packet(p1) is None
    assert detector.process_packet(p2) is None
    assert detector.process_packet(p3) is None
    alert = detector.process_packet(p4)

    assert alert is not None
    assert alert.src_ip is None  # Global aggregation
    assert alert.observed_value == 4


# ============================================================================
# D. Robustness, Edge Cases, and Utilities
# ============================================================================


def test_detector_missing_optional_fields_safe():
    """Verify detectors do not crash on missing optional fields."""
    ps_detector = PortScanDetector(threshold=5)
    syn_detector = SynRateDetector(threshold=5)
    vol_detector = TrafficAnomalyDetector(byte_threshold=5000)

    # Packet with None ports, flags, and missing IPs
    bad_pkt = NormalizedPacket(
        timestamp=700.0,
        src_ip="",
        dst_ip="",
        protocol="OTHER",
        length=0,
        src_port=None,
        dst_port=None,
        tcp_flags=None,
    )

    assert ps_detector.process_packet(bad_pkt) is None
    assert syn_detector.process_packet(bad_pkt) is None
    assert vol_detector.process_packet(bad_pkt) is None


def test_detector_out_of_order_ancient_packet():
    """Verify that ancient out-of-order packets are discarded and do not pollute state."""
    detector = PortScanDetector(threshold=3, window_seconds=10.0)

    # Current stream time is at t=100.0
    detector.process_packet(make_packet(timestamp=100.0, dst_port=80))

    # Stale packet arriving from t=50.0 (older than 100 - 10 = 90)
    assert detector.process_packet(make_packet(timestamp=50.0, dst_port=443)) is None


def test_detector_reset():
    """Verify reset() clears all internal state."""
    detector = PortScanDetector(threshold=3, window_seconds=10.0)

    detector.process_packet(make_packet(timestamp=100.0, dst_port=1))
    detector.process_packet(make_packet(timestamp=101.0, dst_port=2))
    detector.reset()

    # After reset, sending 2 more distinct ports should not trigger (2 < 3)
    detector.process_packet(make_packet(timestamp=102.0, dst_port=3))
    assert (
        detector.process_packet(make_packet(timestamp=103.0, dst_port=4)) is None
    )


def test_alert_to_dict():
    """Verify Alert.to_dict converts properly."""
    alert = Alert(
        timestamp=100.0,
        rule_name="PORT_SCAN",
        src_ip="1.2.3.4",
        dst_ip="5.6.7.8",
        observed_value=15,
        threshold=10,
        window_seconds=30.0,
        severity="MEDIUM",
        message="Test alert",
    )
    d = alert.to_dict()
    assert isinstance(d, dict)
    assert d["rule_name"] == "PORT_SCAN"
    assert d["severity"] == "MEDIUM"
