# capture/__init__.py
"""Packet capture module for Mini-IDS.

Responsible for reading network packets from offline PCAP files and
live network interfaces using Scapy.
"""

from .packet_capture import (
    NormalizedPacket,
    iter_pcap,
    normalize_packet,
    read_pcap,
)
from .live_capture import (
    LiveCaptureWorker,
    get_available_interfaces,
    get_live_worker,
)

__all__ = [
    "NormalizedPacket",
    "normalize_packet",
    "iter_pcap",
    "read_pcap",
    "LiveCaptureWorker",
    "get_available_interfaces",
    "get_live_worker",
]
