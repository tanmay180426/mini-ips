"""Deterministic Synthetic Demonstration Dataset Module for Mini-IDS.

Provides a safe, self-contained, sanitized demonstration database for hosted
deployments (e.g. Streamlit Community Cloud) and mentor demonstrations where
real local network captures and local SQLite databases must not be exposed.

Key Principles:
- Uses RFC 5737 documentation IP prefixes (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24).
- Fully deterministic timestamps, IP pairings, protocols, and ports.
- Explicitly labels all synthetic records as '[SYNTHETIC DEMO EVENT]'.
- Generates correlated packet evidence matching detector sliding evaluation windows,
  ensuring the dashboard's Alert Investigation workflow functions realistically.
- Never accesses, alters, or replaces the developer's local `data/mini_ids.db`.
"""

import os
from pathlib import Path
from typing import List, Optional

from capture.packet_capture import NormalizedPacket
from detection import Alert
from storage.database import DatabaseManager

# Default path for the sanitized synthetic demo database (git-ignored via *.db)
DEFAULT_DEMO_DB_PATH = Path("data") / "demo_synthetic.db"

# Base reference timestamp (deterministic UNIX epoch in seconds)
DEMO_BASE_TIMESTAMP = 1771500000.0  # Reference epoch: 2026-02-18 11:20:00 UTC


def generate_synthetic_packets(base_ts: float = DEMO_BASE_TIMESTAMP) -> List[NormalizedPacket]:
    """Generate deterministic synthetic packets covering TCP, UDP, and ICMP protocols.

    Args:
        base_ts: Reference starting timestamp in epoch seconds.

    Returns:
        List[NormalizedPacket]: List of normalized synthetic packets.
    """
    packets: List[NormalizedPacket] = []

    # 1. Port Scan Activity: 198.51.100.45 (External Scanner) -> 192.0.2.10 (Protected Web Server)
    # 15 distinct TCP destination ports probed sequentially with SYN flag
    scan_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 3306, 3389, 8080]
    for idx, port in enumerate(scan_ports):
        packets.append(
            NormalizedPacket(
                timestamp=base_ts + 10.0 + (idx * 0.9),
                src_ip="198.51.100.45",
                dst_ip="192.0.2.10",
                protocol="TCP",
                length=60,
                src_port=41000 + idx,
                dst_port=port,
                tcp_flags="S",
            )
        )

    # 2. SYN Flood / Rate Surge: 203.0.113.88 (External Attacker) -> 192.0.2.10 (Port 80 HTTP)
    # 25 SYN packets within a rapid 4.8-second burst
    for idx in range(25):
        packets.append(
            NormalizedPacket(
                timestamp=base_ts + 40.0 + (idx * 0.18),
                src_ip="203.0.113.88",
                dst_ip="192.0.2.10",
                protocol="TCP",
                length=64,
                src_port=52000 + idx,
                dst_port=80,
                tcp_flags="S",
            )
        )

    # 3. Traffic Anomaly / Data Surge: 198.51.100.120 -> 192.0.2.30 (Database Gateway, Port 5432)
    # 10 large data packets totaling 150,000 bytes (15,000 bytes per frame)
    for idx in range(10):
        packets.append(
            NormalizedPacket(
                timestamp=base_ts + 60.0 + (idx * 1.4),
                src_ip="198.51.100.120",
                dst_ip="192.0.2.30",
                protocol="TCP",
                length=15000,
                src_port=39001,
                dst_port=5432,
                tcp_flags="PA",
            )
        )

    # 4. Legitimate Web Traffic: 192.0.2.55 (Internal Client) -> 192.0.2.10 (HTTP/HTTPS)
    # 15 normal TCP packets (handshakes, ACK, payload)
    for idx in range(15):
        packets.append(
            NormalizedPacket(
                timestamp=base_ts + 2.0 + (idx * 5.5),
                src_ip="192.0.2.55",
                dst_ip="192.0.2.10",
                protocol="TCP",
                length=128 + (idx * 40),
                src_port=50100 + idx,
                dst_port=443 if idx % 2 == 0 else 80,
                tcp_flags="PA" if idx % 3 == 0 else "A",
            )
        )

    # 5. Legitimate DNS Telemetry: 192.0.2.55 -> 192.0.2.20 (Internal DNS Server, Port 53)
    # 8 UDP DNS resolution frames
    for idx in range(8):
        packets.append(
            NormalizedPacket(
                timestamp=base_ts + 5.0 + (idx * 11.0),
                src_ip="192.0.2.55",
                dst_ip="192.0.2.20",
                protocol="UDP",
                length=76,
                src_port=55120 + idx,
                dst_port=53,
                tcp_flags=None,
            )
        )

    # 6. Network Diagnostics: 192.0.2.70 -> 192.0.2.10 (ICMP Ping Echo)
    # 4 ICMP diagnostic probes
    for idx in range(4):
        packets.append(
            NormalizedPacket(
                timestamp=base_ts + 12.0 + (idx * 20.0),
                src_ip="192.0.2.70",
                dst_ip="192.0.2.10",
                protocol="ICMP",
                length=84,
                src_port=None,
                dst_port=None,
                tcp_flags=None,
            )
        )

    # Ensure chronological sorting
    packets.sort(key=lambda p: p.timestamp)
    return packets


def generate_synthetic_alerts(base_ts: float = DEMO_BASE_TIMESTAMP) -> List[Alert]:
    """Generate deterministic synthetic intrusion alerts correlated with synthetic packets.

    Args:
        base_ts: Reference starting timestamp in epoch seconds.

    Returns:
        List[Alert]: List of structured demonstration alerts.
    """
    return [
        Alert(
            timestamp=base_ts + 25.0,
            rule_name="PORT_SCAN",
            severity="HIGH",
            src_ip="198.51.100.45",
            dst_ip="192.0.2.10",
            observed_value=15.0,
            threshold=10.0,
            window_seconds=30.0,
            message="[SYNTHETIC DEMO EVENT] TCP port scan detected: 15 unique destination ports probed within 30.0s window.",
        ),
        Alert(
            timestamp=base_ts + 45.0,
            rule_name="SYN_RATE",
            severity="HIGH",
            src_ip="203.0.113.88",
            dst_ip="192.0.2.10",
            observed_value=25.0,
            threshold=20.0,
            window_seconds=10.0,
            message="[SYNTHETIC DEMO EVENT] Elevated TCP SYN rate detected: 25 SYN packets observed within 10.0s window.",
        ),
        Alert(
            timestamp=base_ts + 76.0,
            rule_name="TRAFFIC_ANOMALY",
            severity="MEDIUM",
            src_ip="198.51.100.120",
            dst_ip="192.0.2.30",
            observed_value=150000.0,
            threshold=100000.0,
            window_seconds=60.0,
            message="[SYNTHETIC DEMO EVENT] Traffic volume anomaly: 150,000 bytes observed within 60.0s window (threshold: 100,000 bytes).",
        ),
        Alert(
            timestamp=base_ts + 88.0,
            rule_name="SYN_RATE",
            severity="LOW",
            src_ip="192.0.2.55",
            dst_ip="192.0.2.10",
            observed_value=5.0,
            threshold=5.0,
            window_seconds=10.0,
            message="[SYNTHETIC DEMO EVENT] Baseline connection rate threshold reached: 5 SYN packets observed within 10.0s window.",
        ),
    ]


def populate_demo_database(db: DatabaseManager, base_ts: float = DEMO_BASE_TIMESTAMP) -> None:
    """Populate a DatabaseManager instance with synthetic demonstration records.

    Args:
        db: Initialized or active DatabaseManager instance.
        base_ts: Reference starting timestamp in epoch seconds.
    """
    from ml.feature_extraction import FlowFeatureExtractor

    db.init_db()
    packets = generate_synthetic_packets(base_ts)
    alerts = generate_synthetic_alerts(base_ts)

    db.insert_packets(packets)
    db.insert_alerts(alerts)

    # Also extract and persist synthetic flows
    extractor = FlowFeatureExtractor(flow_timeout=15.0)
    flows = []
    for p in packets:
        flows.extend(extractor.process_packet(p))
    flows.extend(extractor.flush())
    if flows:
        db.insert_flows(flows)


def get_or_create_demo_database(
    db_path: Optional[Path] = None,
    base_ts: float = DEMO_BASE_TIMESTAMP,
) -> Path:
    """Ensure a sanitized demonstration SQLite database exists and contains demo data.

    If the specified database file does not exist or contains 0 records, it is
    automatically initialized and populated with deterministic synthetic data.

    Args:
        db_path: Path to the SQLite database file. Defaults to `data/demo_synthetic.db`.
        base_ts: Reference starting timestamp in epoch seconds.

    Returns:
        Path: Validated path to the populated demonstration database file.
    """
    if db_path is not None:
        target_path = Path(db_path)
    else:
        configured_path = os.environ.get("MINI_IDS_DEMO_DB_PATH")
        target_path = Path(configured_path) if configured_path else DEFAULT_DEMO_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with DatabaseManager(target_path) as db:
        needs_population = False
        if not target_path.exists():
            needs_population = True
        else:
            try:
                pkt_count = db.get_packet_count()
                alert_count = db.get_alert_count()
                if pkt_count == 0 and alert_count == 0:
                    needs_population = True
            except Exception:
                needs_population = True

        if needs_population:
            populate_demo_database(db, base_ts=base_ts)

    return target_path
