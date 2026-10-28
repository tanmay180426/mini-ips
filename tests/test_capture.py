"""Unit tests for offline PCAP reading and packet normalization.

Tests verify:
- Extraction of metadata from IPv4 and IPv6 packets.
- Proper identification of TCP, UDP, and ICMP protocols.
- Extraction of source/destination ports and TCP flags.
- Safe skipping of non-IP packets (e.g. ARP, raw Ethernet) without exceptions.
- Safe handling of malformed packets.
- Offline PCAP reading using temporary files generated in pytest.
"""

from pathlib import Path
import pytest
from scapy.all import ARP, Ether, ICMP, IP, IPv6, TCP, UDP, wrpcap

from capture.packet_capture import (
    NormalizedPacket,
    iter_pcap,
    normalize_packet,
    read_pcap,
)


def test_normalize_tcp_packet():
    """Verify normalization of a standard IPv4 TCP packet."""
    raw_pkt = IP(src="192.168.1.50", dst="10.0.0.1") / TCP(
        sport=49152, dport=80, flags="S"
    )
    normalized = normalize_packet(raw_pkt)

    assert normalized is not None
    assert isinstance(normalized, NormalizedPacket)
    assert normalized.src_ip == "192.168.1.50"
    assert normalized.dst_ip == "10.0.0.1"
    assert normalized.protocol == "TCP"
    assert normalized.src_port == 49152
    assert normalized.dst_port == 80
    assert normalized.tcp_flags == "S"
    assert normalized.length > 0


def test_normalize_udp_packet():
    """Verify normalization of an IPv4 UDP packet without TCP flags."""
    raw_pkt = IP(src="10.0.0.5", dst="8.8.8.8") / UDP(sport=5353, dport=53)
    normalized = normalize_packet(raw_pkt)

    assert normalized is not None
    assert normalized.src_ip == "10.0.0.5"
    assert normalized.dst_ip == "8.8.8.8"
    assert normalized.protocol == "UDP"
    assert normalized.src_port == 5353
    assert normalized.dst_port == 53
    assert normalized.tcp_flags is None


def test_normalize_icmp_packet():
    """Verify normalization of an ICMP packet (IP packet without TCP/UDP ports)."""
    raw_pkt = IP(src="192.168.1.1", dst="192.168.1.254") / ICMP()
    normalized = normalize_packet(raw_pkt)

    assert normalized is not None
    assert normalized.src_ip == "192.168.1.1"
    assert normalized.dst_ip == "192.168.1.254"
    assert normalized.protocol == "ICMP"
    assert normalized.src_port is None
    assert normalized.dst_port is None
    assert normalized.tcp_flags is None


def test_normalize_ipv6_packet():
    """Verify normalization of an IPv6 packet."""
    raw_pkt = IPv6(src="2001:db8::1", dst="2001:db8::2") / TCP(sport=5000, dport=443)
    normalized = normalize_packet(raw_pkt)

    assert normalized is not None
    assert normalized.src_ip == "2001:db8::1"
    assert normalized.dst_ip == "2001:db8::2"
    assert normalized.protocol == "TCP"
    assert normalized.src_port == 5000
    assert normalized.dst_port == 443


def test_normalize_non_ip_packet():
    """Verify that non-IP packets (like ARP and raw Ethernet) return None safely."""
    arp_pkt = ARP(pdst="192.168.1.1")
    eth_pkt = Ether()

    assert normalize_packet(arp_pkt) is None
    assert normalize_packet(eth_pkt) is None


def test_normalize_malformed_object():
    """Verify that malformed or unsupported objects do not crash normalization."""

    class BrokenPacket:
        def haslayer(self, layer):
            raise RuntimeError("Corrupt layer data")

    result = normalize_packet(BrokenPacket())
    assert result is None


def test_normalized_packet_to_dict():
    """Verify the to_dict method produces a standard dictionary."""
    pkt = NormalizedPacket(
        timestamp=1700000000.0,
        src_ip="192.168.1.10",
        dst_ip="192.168.1.1",
        protocol="TCP",
        length=60,
        src_port=12345,
        dst_port=80,
        tcp_flags="SA",
    )
    d = pkt.to_dict()
    assert isinstance(d, dict)
    assert d["src_ip"] == "192.168.1.10"
    assert d["protocol"] == "TCP"
    assert d["tcp_flags"] == "SA"


def test_read_pcap_synthetic_file(tmp_path: Path):
    """Generate a temporary PCAP with synthetic packets and verify reading."""
    test_pcap_file = tmp_path / "synthetic_test.pcap"

    # Create a mixture of supported and unsupported packets with standard Ethernet encapsulation
    mac_src = "00:11:22:33:44:55"
    mac_dst = "aa:bb:cc:dd:ee:ff"
    packets = [
        Ether(src=mac_src, dst=mac_dst)
        / IP(src="192.168.1.10", dst="192.168.1.20")
        / TCP(sport=1000, dport=80, flags="S"),
        Ether(src=mac_src, dst=mac_dst) / ARP(pdst="192.168.1.1"),  # Should be safely skipped
        Ether(src=mac_src, dst=mac_dst)
        / IP(src="192.168.1.10", dst="8.8.4.4")
        / UDP(sport=2000, dport=53),
        Ether(src=mac_src, dst=mac_dst)
        / IP(src="192.168.1.10", dst="192.168.1.1")
        / ICMP(),
    ]
    wrpcap(str(test_pcap_file), packets)

    # Read PCAP
    results = read_pcap(test_pcap_file)

    # Exactly 3 packets should be normalized (ARP is skipped)
    assert len(results) == 3

    assert results[0].protocol == "TCP"
    assert results[0].src_ip == "192.168.1.10"
    assert results[0].dst_port == 80

    assert results[1].protocol == "UDP"
    assert results[1].src_ip == "192.168.1.10"
    assert results[1].dst_port == 53

    assert results[2].protocol == "ICMP"
    assert results[2].src_port is None


def test_iter_pcap_synthetic_file(tmp_path: Path):
    """Verify iter_pcap generator streams normalized packets."""
    test_pcap_file = tmp_path / "synthetic_stream.pcap"

    packets = [
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1111, dport=22),
        IP(src="10.0.0.1", dst="10.0.0.3") / TCP(sport=1112, dport=22),
    ]
    wrpcap(str(test_pcap_file), packets)

    streamed = list(iter_pcap(test_pcap_file))
    assert len(streamed) == 2
    assert streamed[0].dst_port == 22
    assert streamed[1].dst_port == 22


def test_read_pcap_missing_file():
    """Verify FileNotFoundError is raised when target PCAP does not exist."""
    with pytest.raises(FileNotFoundError):
        read_pcap("non_existent_file_path.pcap")
