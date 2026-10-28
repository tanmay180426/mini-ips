"""Deterministic Security Rule Engine for Mini-IDS.

Combines signature-less heuristic rule detectors:
- Port Scan Detector (destination port threshold across sliding time window)
- SYN Flood Rate Detector (TCP SYN-to-ACK imbalance monitoring)
- Volumetric Traffic Anomaly Detector (packet/byte surge monitoring)
"""

from typing import Any, Dict, List, Optional, Union
from capture.packet_capture import NormalizedPacket
from detection.port_scan import PortScanDetector
from detection.syn_rate import SynRateDetector
from detection.traffic_anomaly import TrafficAnomalyDetector


class RuleEngine:
    """Orchestrates deterministic heuristic rule checks."""

    def __init__(
        self,
        port_scan_threshold: int = 15,
        port_scan_window: float = 30.0,
        syn_rate_threshold: int = 50,
        syn_window: float = 10.0,
    ) -> None:
        self.port_scan_detector = PortScanDetector(
            threshold=port_scan_threshold,
            window_seconds=port_scan_window,
        )
        self.syn_rate_detector = SynRateDetector(
            threshold=syn_rate_threshold,
            window_seconds=syn_window,
        )
        self.volumetric_detector = TrafficAnomalyDetector()

    def evaluate_packet(self, packet_info: Union[NormalizedPacket, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a packet through all deterministic rules and return matched alerts."""
        if isinstance(packet_info, dict):
            pkt = NormalizedPacket(
                timestamp=float(packet_info.get("timestamp", 0.0)),
                src_ip=str(packet_info.get("src_ip", "")),
                dst_ip=str(packet_info.get("dst_ip", "")),
                protocol=str(packet_info.get("protocol", "OTHER")),
                length=int(packet_info.get("length", packet_info.get("packet_length", 60))),
                src_port=packet_info.get("src_port"),
                dst_port=packet_info.get("dst_port"),
                tcp_flags=packet_info.get("tcp_flags"),
            )
        else:
            pkt = packet_info

        alerts: List[Dict[str, Any]] = []

        # 1. Port scan check
        ps_alert = self.port_scan_detector.process_packet(pkt)
        if ps_alert:
            alerts.append(ps_alert)

        # 2. SYN flood check
        syn_alert = self.syn_rate_detector.process_packet(pkt)
        if syn_alert:
            alerts.append(syn_alert)

        # 3. Volumetric check
        vol_alert = self.volumetric_detector.process_packet(pkt)
        if vol_alert:
            alerts.append(vol_alert)

        return alerts


__all__ = ["RuleEngine", "PortScanDetector", "SynRateDetector", "TrafficAnomalyDetector"]
