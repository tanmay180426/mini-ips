"""Packet Capture and Normalization Module.

This module provides functionality to read offline PCAP and PCAPNG files using
Scapy and normalize captured network packets into a clean, uniform Python data
structure for downstream intrusion detection analysis.

Features:
- Offline PCAP and PCAPNG file ingestion using Scapy PcapReader.
- Extraction of standard metadata (timestamp, IPs, protocol, length, ports).
- Safe handling of non-IP, malformed, or unsupported packets.
- No inspection or persistence of packet payloads (privacy and safety).

NOTE: Live sniffing on network interfaces is deferred to a future milestone.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

# Suppress Scapy runtime warnings on Windows when libpcap/Npcap is absent
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.all import ICMP, IP, IPv6, TCP, UDP, Packet, PcapReader

logger = logging.getLogger(__name__)


@dataclass
class NormalizedPacket:
    """Normalized metadata extracted from a supported network packet.

    Attributes:
        timestamp (float): Packet arrival time in epoch seconds.
        src_ip (str): Source IP address (IPv4 or IPv6).
        dst_ip (str): Destination IP address (IPv4 or IPv6).
        protocol (str): Transport or network protocol ('TCP', 'UDP', 'ICMP', or 'OTHER').
        length (int): Total packet length in bytes.
        src_port (Optional[int]): Source port if TCP or UDP; None otherwise.
        dst_port (Optional[int]): Destination port if TCP or UDP; None otherwise.
        tcp_flags (Optional[str]): TCP control flags (e.g., 'S', 'A', 'FA') if TCP; None otherwise.
    """

    timestamp: float
    src_ip: str
    dst_ip: str
    protocol: str
    length: int
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    tcp_flags: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the normalized packet into a standard Python dictionary."""
        return asdict(self)


def normalize_packet(packet: Packet) -> Optional[NormalizedPacket]:
    """Extract normalized metadata from a single raw Scapy packet.

    Non-IP packets (e.g. ARP, raw Ethernet) and unsupported or malformed packets
    are safely skipped by returning None, ensuring parsing does not crash.
    Packet payloads are intentionally ignored and not stored.

    Args:
        packet: A Scapy Packet object.

    Returns:
        Optional[NormalizedPacket]: A NormalizedPacket instance if supported, or None.
    """
    try:
        # 1. Verify and extract IP network layer (IPv4 or IPv6)
        if packet.haslayer(IP):
            ip_layer = packet[IP]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
        elif packet.haslayer(IPv6):
            ip_layer = packet[IPv6]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
        else:
            # Non-IP packets (e.g. ARP, STP, raw L2 frames) are safely skipped
            return None

        # 2. Extract timestamp and total packet length
        try:
            timestamp = float(packet.time)
        except (AttributeError, ValueError, TypeError):
            timestamp = 0.0

        length = len(packet)

        # 3. Inspect transport layer protocols (TCP, UDP, ICMP, or OTHER)
        src_port: Optional[int] = None
        dst_port: Optional[int] = None
        tcp_flags: Optional[str] = None

        if packet.haslayer(TCP):
            protocol = "TCP"
            tcp_layer = packet[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)
            tcp_flags = str(tcp_layer.flags)
        elif packet.haslayer(UDP):
            protocol = "UDP"
            udp_layer = packet[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)
        elif packet.haslayer(ICMP):
            protocol = "ICMP"
        else:
            protocol = "OTHER"

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

    except Exception as err:
        # Safely catch malformed or unexpected packet structures
        logger.debug("Failed to normalize malformed packet: %s", err)
        return None


def iter_pcap(file_path: Union[str, Path]) -> Iterator[NormalizedPacket]:
    """Stream and yield normalized packets from a PCAP or PCAPNG file.

    Memory-efficient iterator suitable for larger capture files.

    Args:
        file_path: Path to the .pcap or .pcapng file.

    Yields:
        NormalizedPacket: Each successfully normalized packet.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {file_path}")

    with PcapReader(str(path)) as reader:
        for raw_packet in reader:
            normalized = normalize_packet(raw_packet)
            if normalized is not None:
                yield normalized


def read_pcap(file_path: Union[str, Path]) -> List[NormalizedPacket]:
    """Read a PCAP or PCAPNG file and return all normalized packets as a list.

    Unsupported, non-IP, and malformed packets are safely filtered out.

    Args:
        file_path: Path to the .pcap or .pcapng file.

    Returns:
        List[NormalizedPacket]: All successfully normalized packets.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    return list(iter_pcap(file_path))
