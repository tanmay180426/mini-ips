"""End-to-end integration tests for Mini-IDS offline pipeline.

Tests verify:
- Ingestion of synthetic PCAPs containing valid IP traffic and non-IP packets (ARP).
- Propagation of normalized packets through detection rules and SQLite persistence.
- Consistent counts between returned summary dictionary and database queries.
- Correct alert attributes and rule classification.
- Fresh detector state on subsequent pipeline runs against the same PCAP.
- Graceful handling and exit codes for missing or invalid PCAP paths.
"""

from pathlib import Path
import pytest
from scapy.all import ARP, Ether, IP, TCP, UDP, wrpcap

from detection import PortScanDetector, SynRateDetector, TrafficAnomalyDetector
from main import main, process_pcap_stream
from storage.database import DatabaseManager


def create_synthetic_pcap(file_path: Path) -> int:
    """Create a synthetic PCAP with a mix of scan traffic, normal UDP, and ARP.

    Returns:
        int: Number of valid IP packets created (excluding ARP).
    """
    mac_src = "00:11:22:33:44:55"
    mac_dst = "aa:bb:cc:dd:ee:ff"
    src_ip = "192.168.1.50"
    target_ip = "10.0.0.1"

    packets = []

    # 1. Non-IP packet (ARP) - must be ignored by iter_pcap
    packets.append(Ether(src=mac_src, dst=mac_dst) / ARP(pdst=target_ip))

    # 2. 6 TCP connection attempts to distinct ports (target: port scan trigger with threshold=5)
    for port in [21, 22, 23, 25, 80, 443]:
        packets.append(
            Ether(src=mac_src, dst=mac_dst)
            / IP(src=src_ip, dst=target_ip)
            / TCP(sport=50000 + port, dport=port, flags="S")
        )

    # 3. 2 normal UDP packets
    packets.append(
        Ether(src=mac_src, dst=mac_dst)
        / IP(src=src_ip, dst="8.8.8.8")
        / UDP(sport=53000, dport=53)
    )
    packets.append(
        Ether(src=mac_src, dst=mac_dst)
        / IP(src=src_ip, dst="8.8.4.4")
        / UDP(sport=53001, dport=53)
    )

    wrpcap(str(file_path), packets)
    # Total packets: 1 ARP + 6 TCP + 2 UDP = 9 packets
    # Valid IP packets: 8 packets
    return 8


def test_end_to_end_pipeline_execution(tmp_path: Path):
    """Verify that process_pcap_stream accurately analyzes, detects, and persists traffic."""
    pcap_file = tmp_path / "test_traffic.pcap"
    db_file = tmp_path / "test_pipeline.db"

    expected_ip_count = create_synthetic_pcap(pcap_file)

    # Configure custom low-threshold detectors for deterministic testing
    test_detectors = [
        PortScanDetector(threshold=5, window_seconds=30.0),
        SynRateDetector(threshold=20, window_seconds=10.0),
        TrafficAnomalyDetector(byte_threshold=1_000_000, packet_threshold=500),
    ]

    summary = process_pcap_stream(
        pcap_path=pcap_file,
        db_path=db_file,
        detectors=test_detectors,
        batch_size=5,
        print_alerts=False,
    )

    # 1. Verify summary counts
    assert summary["packets_processed"] == expected_ip_count
    assert summary["alerts_generated"] >= 1
    assert "PORT_SCAN" in summary["rule_counts"]
    assert summary["rule_counts"]["PORT_SCAN"] == 1

    # 2. Verify persistence in SQLite database
    with DatabaseManager(db_file) as db:
        assert db.get_packet_count() == expected_ip_count
        assert db.get_alert_count() == summary["alerts_generated"]

        alerts = db.get_alerts()
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["rule_name"] == "PORT_SCAN"
        assert alert["src_ip"] == "192.168.1.50"
        assert alert["dst_ip"] == "10.0.0.1"
        assert alert["observed_value"] >= 5.0
        assert alert["threshold"] == 5.0


def test_end_to_end_fresh_state_on_subsequent_runs(tmp_path: Path):
    """Verify that running a new pipeline invocation uses fresh detector state."""
    pcap_file = tmp_path / "test_traffic2.pcap"
    db_file1 = tmp_path / "test_run1.db"
    db_file2 = tmp_path / "test_run2.db"

    create_synthetic_pcap(pcap_file)

    # Run 1
    summary1 = process_pcap_stream(
        pcap_path=pcap_file,
        db_path=db_file1,
        detectors=[PortScanDetector(threshold=5, window_seconds=30.0)],
        print_alerts=False,
    )
    assert summary1["alerts_generated"] == 1

    # Run 2 against a new database with default fresh detectors
    # If state was accidentally shared, suppression might silence the alert
    summary2 = process_pcap_stream(
        pcap_path=pcap_file,
        db_path=db_file2,
        detectors=[PortScanDetector(threshold=5, window_seconds=30.0)],
        print_alerts=False,
    )
    assert summary2["alerts_generated"] == 1


def test_end_to_end_missing_pcap_raises_error(tmp_path: Path):
    """Verify FileNotFoundError is raised when target PCAP does not exist."""
    db_file = tmp_path / "test_missing.db"
    with pytest.raises(FileNotFoundError):
        process_pcap_stream(
            pcap_path=tmp_path / "does_not_exist.pcap",
            db_path=db_file,
        )


def test_cli_main_success(tmp_path: Path, capsys):
    """Verify CLI main() parses arguments, processes PCAP, and exits with code 0."""
    pcap_file = tmp_path / "cli_test.pcap"
    db_file = tmp_path / "cli_test.db"

    create_synthetic_pcap(pcap_file)

    exit_code = main(["--pcap", str(pcap_file), "--db", str(db_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Mini-IDS: Offline Network Intrusion Detection System" in captured.out
    assert "Analysis Complete!" in captured.out
    assert "Packets analyzed: 8" in captured.out


def test_cli_main_missing_file(tmp_path: Path, capsys):
    """Verify CLI main() returns nonzero exit code when file is missing."""
    exit_code = main(["--pcap", str(tmp_path / "missing.pcap")])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "[ERROR]" in captured.err
