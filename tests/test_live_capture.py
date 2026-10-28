"""Automated Unit and Integration Tests for Mini-IDS Live Packet Capture.

Tests verify:
- Safe enumeration and sorting of network interfaces discovered by Scapy.
- Graceful handling of empty interfaces, permission errors, and missing Npcap.
- Thread-safe background worker lifecycle (start, stop, configure).
- Prevention of duplicate/concurrent capture workers.
- Normalization, detector evaluation, and SQLite persistence of captured packets and alerts.
- Automatic stop on packet limit and duration timeout.
- CLI live capture execution and --list-interfaces output.
- Non-blocking dashboard integration.

All network capture operations are fully mocked to ensure tests run reliably
across all environments without requiring real network access, administrator
rights, or active network adapters.
"""

from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from scapy.all import IP, TCP, UDP
from streamlit.testing.v1 import AppTest

from capture.live_capture import (
    LiveCaptureWorker,
    filter_primary_interfaces,
    get_available_interfaces,
    get_live_worker,
)
from capture.packet_capture import NormalizedPacket
from detection import PortScanDetector, SynRateDetector, TrafficAnomalyDetector
from main import main
from storage.database import DatabaseManager

APP_PATH = str(Path(__file__).parent.parent / "app.py")


class MockScapyInterface:
    """Mock Scapy interface object for deterministic testing."""

    def __init__(self, name: str, description: str, ip: str = "", mac: str = "", guid: str = ""):
        self.name = name
        self.description = description
        self.ip = ip
        self.mac = mac
        self.network_name = guid
        self.pcap_name = guid


# ---------------------------------------------------------------------------
# 1. Interface Enumeration Tests
# ---------------------------------------------------------------------------


def test_get_available_interfaces_formatting_and_sorting():
    """Verify that network adapters are properly extracted, formatted, and sorted."""
    mock_ifaces = {
        "dev1": MockScapyInterface("Wi-Fi", "Realtek WiFi Controller", "192.168.1.100", "00:11:22:33:44:55"),
        "dev2": MockScapyInterface("Loopback Pseudo-Interface 1", "Software Loopback", "127.0.0.1"),
        "dev3": MockScapyInterface("Ethernet", "Realtek GbE Controller", "169.254.10.20"),
        "dev4": MockScapyInterface("Disconnected WAN", "WAN Miniport", ""),
    }

    with patch("capture.live_capture.conf") as mock_conf:
        mock_conf.ifaces = mock_ifaces
        interfaces = get_available_interfaces()

    assert len(interfaces) == 4

    # Active IPv4 should be sorted first
    first = interfaces[0]
    assert first["name"] == "Wi-Fi"
    assert first["ip"] == "192.168.1.100"
    assert first["has_ip"] is True
    assert "192.168.1.100" in first["display_name"]
    assert "Realtek WiFi Controller" in first["display_name"]

    # Loopback adapter should be properly identified
    loopback = next(i for i in interfaces if "Loopback" in i["name"])
    assert loopback["ip"] == "127.0.0.1"

    # Disconnected WAN adapter should have has_ip == False
    wan = next(i for i in interfaces if i["name"] == "Disconnected WAN")
    assert wan["has_ip"] is False
    assert wan["ip"] == ""


def test_get_available_interfaces_handles_exceptions_and_empty():
    """Verify graceful handling when Scapy conf.ifaces is empty or raises an exception."""
    with patch("capture.live_capture.conf") as mock_conf:
        mock_conf.ifaces = {}
        assert get_available_interfaces() == []

    with patch("capture.live_capture.conf") as mock_conf:
        # Simulate property access failure or Npcap missing error
        type(mock_conf).ifaces = property(lambda self: (_ for _ in ()).throw(RuntimeError("Npcap missing")))
        assert get_available_interfaces() == []


def test_filter_primary_interfaces():
    """Verify filter_primary_interfaces filters out virtual WFP/QoS miniports and caps at 3-4 options."""
    mock_raw_ifaces = [
        {"name": "Wi-Fi", "description": "Intel Wi-Fi 6", "ip": "192.168.1.103", "has_ip": True, "display_name": "Wi-Fi (192.168.1.103)"},
        {"name": "Ethernet", "description": "Realtek GbE", "ip": "169.254.189.189", "has_ip": False, "display_name": "Ethernet"},
        {"name": "Loopback Pseudo-Interface 1", "description": "Software Loopback", "ip": "127.0.0.1", "has_ip": False, "is_loopback": True, "display_name": "Loopback"},
        {"name": "Bluetooth Network Connection", "description": "Bluetooth Device", "ip": "169.254.68.167", "has_ip": False, "display_name": "Bluetooth"},
        {"name": "Ethernet-WFP 802.3 MAC Layer LightWeight Filter-0000", "description": "WFP Filter", "ip": "", "has_ip": False, "display_name": "WFP Filter"},
        {"name": "Ethernet-QoS Packet Scheduler-0000", "description": "QoS Scheduler", "ip": "", "has_ip": False, "display_name": "QoS"},
        {"name": "Local Area Connection* 1", "description": "Microsoft Wi-Fi Direct Virtual Adapter", "ip": "", "has_ip": False, "display_name": "Wi-Fi Direct"},
        {"name": "6to4 Adapter", "description": "Microsoft 6to4 Adapter", "ip": "", "has_ip": False, "display_name": "6to4"},
    ]

    filtered = filter_primary_interfaces(mock_raw_ifaces, max_count=4)
    assert len(filtered) == 4
    names = [f["name"] for f in filtered]
    assert "Wi-Fi" in names
    assert "Loopback Pseudo-Interface 1" in names
    assert "Ethernet" in names
    assert "Ethernet-WFP 802.3 MAC Layer LightWeight Filter-0000" not in names
    assert "6to4 Adapter" not in names



# ---------------------------------------------------------------------------
# 2. Worker Lifecycle and Duplicate Prevention Tests
# ---------------------------------------------------------------------------


def test_worker_lifecycle_and_duplicate_start_prevention(tmp_path: Path):
    """Verify worker start, stop, and that duplicate starts are rejected."""
    test_db = tmp_path / "lifecycle_test.db"
    worker = LiveCaptureWorker(interface="MockAdapter", db_path=test_db)

    assert not worker.is_running
    status_idle = worker.get_status()
    assert status_idle["status"] == "IDLE"
    assert status_idle["packets_captured"] == 0

    mock_sniffer_instance = MagicMock()

    with patch("capture.live_capture.AsyncSniffer", return_value=mock_sniffer_instance):
        started = worker.start()
        assert started is True
        assert worker.is_running is True
        assert worker.get_status()["status"] == "RUNNING"

        # Attempting to start an already running worker must raise RuntimeError
        with pytest.raises(RuntimeError, match="already actively running"):
            worker.start()

        # Stop worker cleanly
        worker.stop(timeout=1.0)
        assert worker.is_running is False
        assert worker.get_status()["status"] == "STOPPED"
        mock_sniffer_instance.stop.assert_called()


def test_worker_configure_while_running_raises_error(tmp_path: Path):
    """Verify configure() cannot be called while capture is active."""
    test_db = tmp_path / "config_test.db"
    worker = LiveCaptureWorker(interface="Adapter1", db_path=test_db)

    with patch("capture.live_capture.AsyncSniffer", return_value=MagicMock()):
        worker.start()
        try:
            with pytest.raises(RuntimeError, match="Cannot reconfigure an active capture worker"):
                worker.configure(interface="Adapter2")
        finally:
            worker.stop(timeout=1.0)


def test_worker_start_without_interface_raises_error(tmp_path: Path):
    """Verify starting a worker with no interface specified raises ValueError."""
    worker = LiveCaptureWorker(interface=None, db_path=tmp_path / "test.db")
    with pytest.raises(ValueError, match="No network interface specified"):
        worker.start()


# ---------------------------------------------------------------------------
# 3. Packet Ingestion, Detection, and Persistence Tests
# ---------------------------------------------------------------------------


def test_worker_packet_limit_and_persistence(tmp_path: Path):
    """Verify that worker stops at packet limit and persists packets to SQLite."""
    test_db = tmp_path / "packet_limit.db"
    worker = LiveCaptureWorker(
        interface="MockLoopback",
        db_path=test_db,
        packet_limit=4,
        batch_size=2,
    )

    sniffer_started = threading.Event()
    captured_cb = None

    def mock_sniffer_init(iface, prn, store):
        nonlocal captured_cb
        captured_cb = prn
        mock_s = MagicMock()
        mock_s.start = MagicMock(side_effect=sniffer_started.set)
        mock_s.stop = MagicMock()
        return mock_s

    with patch("capture.live_capture.AsyncSniffer", side_effect=mock_sniffer_init):
        worker.start()
        assert sniffer_started.wait(timeout=2.0), "Worker did not start sniffer in time"
        assert captured_cb is not None

        # Simulate 4 raw Scapy packets arriving via the sniffer callback
        for i in range(4):
            raw_pkt = IP(src=f"10.0.0.{i+1}", dst="10.0.0.100") / TCP(sport=50000 + i, dport=80, flags="S")
            raw_pkt.time = 1700000000.0 + i
            if captured_cb:
                captured_cb(raw_pkt)

        # Wait for worker thread to process queue and auto-stop on packet limit
        timeout = time.time() + 2.0
        while worker.is_running and time.time() < timeout:
            time.sleep(0.05)

        worker.stop(timeout=1.0)

    status = worker.get_status()
    assert status["status"] == "COMPLETED"
    assert status["packets_captured"] == 4
    assert status["packets_persisted"] == 4

    # Verify packets are in database
    with DatabaseManager(test_db) as db:
        assert db.get_packet_count() == 4
        packets = db.get_packets(limit=10)
        assert len(packets) == 4
        src_ips = {p["src_ip"] for p in packets}
        assert "10.0.0.1" in src_ips
        assert "10.0.0.4" in src_ips


def test_worker_detects_alerts_and_persists(tmp_path: Path):
    """Verify that live packets trigger detection rules and persist alerts to SQLite."""
    test_db = tmp_path / "live_alerts.db"

    # Detector with low threshold=3 for testing
    scan_detector = PortScanDetector(threshold=3, window_seconds=10.0)

    worker = LiveCaptureWorker(
        interface="MockEth",
        db_path=test_db,
        packet_limit=4,
        detectors=[scan_detector],
        batch_size=1,
    )

    sniffer_started = threading.Event()
    captured_cb = None

    def mock_sniffer_init(iface, prn, store):
        nonlocal captured_cb
        captured_cb = prn
        mock_s = MagicMock()
        mock_s.start = MagicMock(side_effect=sniffer_started.set)
        mock_s.stop = MagicMock()
        return mock_s

    with patch("capture.live_capture.AsyncSniffer", side_effect=mock_sniffer_init):
        worker.start()
        assert sniffer_started.wait(timeout=2.0), "Worker did not start sniffer in time"
        assert captured_cb is not None

        # Send 3 scan packets targeting 3 distinct ports (threshold=3)
        for port in [80, 443, 8080]:
            pkt = IP(src="192.168.1.50", dst="10.0.0.5") / TCP(sport=55000, dport=port, flags="S")
            pkt.time = 1700000000.0 + (port / 1000.0)
            if captured_cb:
                captured_cb(pkt)

        timeout = time.time() + 2.0
        while worker.is_running and time.time() < timeout:
            time.sleep(0.05)

        worker.stop(timeout=1.0)

    status = worker.get_status()
    assert status["alerts_generated"] >= 1

    with DatabaseManager(test_db) as db:
        assert db.get_alert_count() >= 1
        alerts = db.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["rule_name"] == "PORT_SCAN"
        assert alerts[0]["src_ip"] == "192.168.1.50"
        assert alerts[0]["dst_ip"] == "10.0.0.5"


def test_worker_sniffer_start_failure_handles_error(tmp_path: Path):
    """Verify that when AsyncSniffer fails to start, worker transitions to ERROR status."""
    test_db = tmp_path / "sniff_err.db"
    worker = LiveCaptureWorker(interface="BadAdapter", db_path=test_db)

    with patch("capture.live_capture.AsyncSniffer", side_effect=PermissionError("Npcap access denied")):
        worker.start()
        # Give thread time to fail
        time.sleep(0.15)
        worker.stop(timeout=1.0)

    status = worker.get_status()
    assert status["status"] == "ERROR"
    assert status["is_running"] is False
    assert "Npcap access denied" in str(status["error_message"])


def test_worker_duration_timeout(tmp_path: Path):
    """Verify that worker terminates when duration_seconds expires without packets."""
    test_db = tmp_path / "duration_test.db"
    worker = LiveCaptureWorker(
        interface="MockAdapter",
        db_path=test_db,
        duration_seconds=0.25,
    )

    with patch("capture.live_capture.AsyncSniffer", return_value=MagicMock()):
        worker.start()
        timeout = time.time() + 1.5
        while worker.is_running and time.time() < timeout:
            time.sleep(0.05)

        worker.stop(timeout=1.0)

    status = worker.get_status()
    assert status["status"] == "COMPLETED"
    assert status["is_running"] is False
    assert status["packets_captured"] == 0


# ---------------------------------------------------------------------------
# 4. CLI Live Capture Integration Tests
# ---------------------------------------------------------------------------


def test_cli_list_interfaces(capsys):
    """Verify main.py --list-interfaces executes and prints discovered adapters."""
    mock_ifaces = [
        {"display_name": "Wi-Fi (192.168.0.100) — Realtek"},
        {"display_name": "Loopback Pseudo-Interface 1 (127.0.0.1)"},
    ]

    with patch("main.get_available_interfaces", return_value=mock_ifaces):
        exit_code = main(["--list-interfaces"])
        assert exit_code == 0

    captured = capsys.readouterr()
    assert "Available Network Interfaces" in captured.out
    assert "Wi-Fi (192.168.0.100) — Realtek" in captured.out
    assert "Loopback Pseudo-Interface 1" in captured.out


def test_cli_live_capture_execution(tmp_path: Path, capsys):
    """Verify main.py --interface runs live capture and outputs summary."""
    test_db = tmp_path / "cli_live.db"

    mock_status = {
        "status": "COMPLETED",
        "packets_captured": 15,
        "alerts_generated": 1,
        "elapsed_seconds": 1.5,
        "db_path": str(test_db),
        "error_message": None,
    }

    mock_worker = MagicMock()
    mock_worker.is_running = False
    mock_worker.get_status.return_value = mock_status

    with patch("main.LiveCaptureWorker", return_value=mock_worker):
        exit_code = main([
            "--interface", "MockWiFi",
            "--count", "15",
            "--timeout", "5.0",
            "--db", str(test_db),
        ])
        assert exit_code == 0

    captured = capsys.readouterr()
    assert "Mini-IDS: Live Network Intrusion Detection Capture" in captured.out
    assert "Interface:    MockWiFi" in captured.out
    assert "Packet Limit: 15" in captured.out
    assert "Packets captured: 15" in captured.out
    assert "Alerts triggered: 1" in captured.out


# ---------------------------------------------------------------------------
# 5. Dashboard Live Capture UI Tests
# ---------------------------------------------------------------------------


def test_dashboard_live_capture_section_renders(tmp_path: Path):
    """Verify the Live Packet Capture controls render without crashing the dashboard."""
    test_db = tmp_path / "live_ui.db"
    with DatabaseManager(test_db) as db:
        db.init_db()
        db.insert_packet(NormalizedPacket(100.0, "1.1.1.1", "2.2.2.2", "TCP", 64, 1234, 80, "S"))

    mock_ifaces = [
        {"name": "Wi-Fi", "display_name": "Wi-Fi (192.168.0.1) — Realtek", "guid": "guid1"},
        {"name": "Loopback", "display_name": "Loopback (127.0.0.1)", "guid": "guid2"},
    ]

    with patch("app.get_available_interfaces", return_value=mock_ifaces):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.sidebar.text_input[0].input(str(test_db)).run()

        assert not at.exception
        # Confirm live capture expander is present in UI
        expander_labels = [exp.label for exp in at.expander]
        assert any("Capture Configuration & Controls" in label for label in expander_labels)


def test_dashboard_live_capture_start_stop_interaction(tmp_path: Path):
    """Verify that clicking Start Capture triggers worker configuration and lifecycle."""
    test_db = tmp_path / "live_ui_ctrl.db"
    with DatabaseManager(test_db) as db:
        db.init_db()

    mock_ifaces = [
        {"name": "Wi-Fi", "display_name": "Wi-Fi (192.168.0.1) — Realtek", "guid": "guid1"},
    ]

    mock_worker = MagicMock()
    mock_worker.is_running = False
    mock_worker.get_status.return_value = {
        "status": "IDLE",
        "is_running": False,
        "interface": None,
        "packets_captured": 0,
        "packets_persisted": 0,
        "packets_dropped": 0,
        "alerts_generated": 0,
        "packet_limit": None,
        "duration_seconds": None,
        "elapsed_seconds": 0.0,
        "error_message": None,
        "db_path": str(test_db),
    }

    with patch("app.get_available_interfaces", return_value=mock_ifaces), \
         patch("app.get_live_worker", return_value=mock_worker), \
         patch("capture.live_capture.get_live_worker", return_value=mock_worker):
        at = AppTest.from_file(APP_PATH)
        at.run()
        at.sidebar.text_input[0].input(str(test_db)).run()

        # Find Start Capture button
        start_btn = next((b for b in at.button if b.key == "live_btn_start"), None)
        assert start_btn is not None
        assert not start_btn.disabled
        start_btn.click()
        at.run()

        mock_worker.configure.assert_called_once()
        mock_worker.start.assert_called_once()




# ---------------------------------------------------------------------------
# 6. Deterministic Synthetic Integration Tests (Phase F Verification)
# ---------------------------------------------------------------------------


def test_synthetic_integration_all_detectors_and_persistence(tmp_path: Path):
    """Synthetic Integration Test:
    Verify that all 3 detection rules (Port Scan, SYN Rate, Traffic Anomaly)
    receive synthetic packets, evaluate thresholds, and persist rich alerts with
    valid rule_name, IPs, observed values, thresholds, and severity into SQLite.
    """
    test_db = tmp_path / "synthetic_all_detectors.db"

    # Set up all 3 detectors with custom testing thresholds
    detectors = [
        PortScanDetector(threshold=3, window_seconds=10.0),
        SynRateDetector(threshold=3, window_seconds=10.0),
        TrafficAnomalyDetector(packet_threshold=5, byte_threshold=None, window_seconds=10.0),
    ]

    worker = LiveCaptureWorker(
        interface="MockSyntheticAdapter",
        db_path=test_db,
        packet_limit=6,
        detectors=detectors,
        batch_size=1,
    )

    sniffer_started = threading.Event()
    captured_cb = None

    def mock_sniffer_init(iface, prn, store):
        nonlocal captured_cb
        captured_cb = prn
        mock_s = MagicMock()
        mock_s.start = MagicMock(side_effect=sniffer_started.set)
        mock_s.stop = MagicMock()
        return mock_s

    with patch("capture.live_capture.AsyncSniffer", side_effect=mock_sniffer_init):
        worker.start()
        assert sniffer_started.wait(timeout=2.0), "Worker did not start sniffer in time"
        assert captured_cb is not None

        # Feed 5 synthetic packets from 192.168.1.10 -> 10.0.0.1 targeting 5 distinct ports with SYN flag
        # This will trigger:
        # 1. PortScanDetector: 5 unique ports >= threshold 3
        # 2. SynRateDetector: 5 SYN packets >= threshold 3
        # 3. TrafficAnomalyDetector: 5 packets >= packet_threshold 5
        for i in range(5):
            pkt = IP(src="192.168.1.10", dst="10.0.0.1") / TCP(sport=40000 + i, dport=100 + i, flags="S")
            pkt.time = 1700000000.0 + (i * 0.1)
            if captured_cb:
                captured_cb(pkt)

        # Wait for worker loop
        timeout = time.time() + 2.0
        while worker.is_running and time.time() < timeout:
            time.sleep(0.05)

        worker.stop(timeout=1.0)

    status = worker.get_status()
    assert status["packets_captured"] == 5
    assert status["alerts_generated"] >= 3

    with DatabaseManager(test_db) as db:
        assert db.get_packet_count() == 5
        alerts = db.get_alerts()
        assert len(alerts) >= 3

        rule_names = {a["rule_name"] for a in alerts}
        assert "PORT_SCAN" in rule_names
        assert "SYN_RATE" in rule_names
        assert "TRAFFIC_VOLUME" in rule_names

        for alert in alerts:
            assert alert["src_ip"] == "192.168.1.10"
            assert alert["dst_ip"] == "10.0.0.1"
            assert isinstance(alert["timestamp"], float)
            assert alert["observed_value"] >= alert["threshold"]
            assert alert["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            assert len(alert["message"]) > 0


def test_synthetic_session_state_isolation(tmp_path: Path):
    """Synthetic Integration Test:
    Verify that subsequent capture sessions on fresh workers do not leak
    detector state or alert counts from prior sessions.
    """
    test_db = tmp_path / "synthetic_isolation.db"

    def run_session(port_list):
        worker = LiveCaptureWorker(
            interface="MockAdapter",
            db_path=test_db,
            packet_limit=len(port_list),
            detectors=[PortScanDetector(threshold=3, window_seconds=10.0)],
            batch_size=1,
        )
        sniffer_started = threading.Event()
        captured_cb = None

        def mock_sniffer_init(iface, prn, store):
            nonlocal captured_cb
            captured_cb = prn
            mock_s = MagicMock()
            mock_s.start = MagicMock(side_effect=sniffer_started.set)
            mock_s.stop = MagicMock()
            return mock_s

        with patch("capture.live_capture.AsyncSniffer", side_effect=mock_sniffer_init):
            worker.start()
            assert sniffer_started.wait(timeout=2.0), "Worker did not start sniffer in time"
            assert captured_cb is not None
            for port in port_list:
                pkt = IP(src="10.10.10.10", dst="10.10.10.20") / TCP(sport=50000, dport=port, flags="S")
                pkt.time = 1700000100.0 + (port / 100.0)
                if captured_cb:
                    captured_cb(pkt)

            timeout = time.time() + 2.0
            while worker.is_running and time.time() < timeout:
                time.sleep(0.05)
            worker.stop(timeout=1.0)

        return worker.get_status()

    # Session 1: 2 packets (below threshold 3) -> 0 alerts
    status1 = run_session([80, 81])
    assert status1["packets_captured"] == 2
    assert status1["alerts_generated"] == 0

    # Session 2: 2 packets (below threshold 3) -> should still be 0 alerts if state didn't leak
    status2 = run_session([82, 83])
    assert status2["packets_captured"] == 2
    assert status2["alerts_generated"] == 0


# ---------------------------------------------------------------------------
# 7. Regression Tests: Lifecycle Shutdown & CLI Bounds Validation
# ---------------------------------------------------------------------------


def test_worker_stop_timeout_preserves_running_state_until_termination(tmp_path: Path):
    """Deterministic concurrency test:
    Verify that when worker.stop() times out while the worker thread remains alive:
    1. worker.is_running remains True and status is 'STOPPING'.
    2. A second worker cannot start or reconfigure while the first thread is alive.
    3. Once the thread finishes, status synchronizes to 'STOPPED' and is_running is False.
    """
    test_db = tmp_path / "stop_timeout.db"
    worker = LiveCaptureWorker(interface="MockAdapter", db_path=test_db)

    thread_can_exit = threading.Event()
    worker_started = threading.Event()

    def mock_run_worker():
        worker_started.set()
        # Hold worker thread alive until test signals it to exit
        thread_can_exit.wait(timeout=5.0)
        with worker._lock:
            worker._status = "STOPPED"
            worker._is_running = False

    with patch.object(worker, "_run_worker", side_effect=mock_run_worker):
        worker.start()
        assert worker_started.wait(timeout=2.0), "Worker thread failed to start"
        assert worker.is_running is True

        # Stop with zero/negligible timeout so join times out immediately while thread is alive
        worker.stop(timeout=0.01)

        # Thread is still blocked on thread_can_exit; must remain running
        assert worker.is_running is True
        status = worker.get_status()
        assert status["is_running"] is True
        assert status["status"] == "STOPPING"

        # Duplicate start while thread is still finishing MUST be rejected
        with pytest.raises(RuntimeError, match="already actively running"):
            worker.start()

        # Reconfiguration while thread is still alive MUST be rejected
        with pytest.raises(RuntimeError, match="Cannot reconfigure"):
            worker.configure(interface="AnotherAdapter")

        # Now signal the background thread to finish
        thread_can_exit.set()
        if worker._thread is not None:
            worker._thread.join(timeout=2.0)

        # After actual thread termination, status must correctly report STOPPED
        final_status = worker.get_status()
        assert final_status["is_running"] is False
        assert final_status["status"] == "STOPPED"
        assert worker.is_running is False


def test_cli_rejects_simultaneous_pcap_and_interface(capsys):
    """Verify main.py rejects simultaneous --pcap and --interface with exit code 1."""
    exit_code = main(["--pcap", "data/test.pcap", "--interface", "Wi-Fi"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "--pcap and --interface are mutually exclusive" in captured.err


def test_cli_rejects_invalid_count(capsys):
    """Verify main.py rejects zero and negative packet limits with exit code 1."""
    exit_code_zero = main(["--interface", "Wi-Fi", "--count", "0"])
    assert exit_code_zero == 1
    captured_zero = capsys.readouterr()
    assert "Invalid --count value '0'" in captured_zero.err

    exit_code_neg = main(["--interface", "Wi-Fi", "--count", "-10"])
    assert exit_code_neg == 1
    captured_neg = capsys.readouterr()
    assert "Invalid --count value '-10'" in captured_neg.err


def test_cli_rejects_invalid_timeout(capsys):
    """Verify main.py rejects zero and negative duration timeouts with exit code 1."""
    exit_code_zero = main(["--interface", "Wi-Fi", "--timeout", "0"])
    assert exit_code_zero == 1
    captured_zero = capsys.readouterr()
    assert "Invalid --timeout value '0.0'" in captured_zero.err

    exit_code_neg = main(["--interface", "Wi-Fi", "--timeout", "-5.5"])
    assert exit_code_neg == 1
    captured_neg = capsys.readouterr()
    assert "Invalid --timeout value '-5.5'" in captured_neg.err


def test_cli_rejects_live_args_with_pcap(tmp_path: Path, capsys):
    """Verify main.py rejects --count or --timeout when used with --pcap."""
    pcap_path = str(tmp_path / "dummy.pcap")
    exit_code = main(["--pcap", pcap_path, "--count", "100"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--count and --timeout options are only applicable to live capture" in captured.err

    exit_code_t = main(["--pcap", pcap_path, "--timeout", "30.0"])
    assert exit_code_t == 1
    captured_t = capsys.readouterr()
    assert "--count and --timeout options are only applicable to live capture" in captured_t.err
