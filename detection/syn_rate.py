"""TCP SYN Rate Anomaly Detection Module.

Identifies abnormally high frequencies of TCP connection initiation requests (SYN
packets without ACK set) originating from a source host towards a target host within
a configurable sliding time window. Such patterns are characteristic of SYN flood
Denial-of-Service (DoS) attempts or aggressive network scanners.

Detector Characteristics:
- Target pair: Evaluated per (source IP, destination IP) pair.
- Flag filtering: Matches packets where the SYN flag ('S') is set and the ACK flag
  ('A') is NOT set. SYN-ACK ('SA') packets are server responses and are ignored.
- Protocol filter: Strictly inspects TCP packets. UDP, ICMP, and non-IP traffic
  are safely ignored.
- Suppression: Once triggered for a (src, dst) pair, subsequent alerts for that pair
  are suppressed until the observed SYN count drops below the threshold or a full
  window duration elapses.
- Out-of-order handling: Packets with timestamps older than (latest_seen - window)
  are ignored as expired.

NOTE: This detector is a heuristic indicator and may generate false positives
during legitimate bursts of outbound TCP connections (e.g., loading web pages
with numerous embedded assets or downloading multi-threaded files).
"""

from typing import Dict, Iterable, List, Optional, Tuple

from capture.packet_capture import NormalizedPacket
from detection import Alert


class SynRateDetector:
    """Detects unusually high TCP SYN rates across a single traffic stream.

    Attributes:
        threshold (int): Minimum count of initial SYN packets required to trigger an alert.
                         Default: 20 SYN packets.
        window_seconds (float): Time window in seconds across which SYN packets are tracked.
                                Default: 10.0 seconds.
    """

    DEFAULT_THRESHOLD: int = 20
    DEFAULT_WINDOW_SECONDS: float = 10.0

    def __init__(
        self,
        threshold: int = DEFAULT_THRESHOLD,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        if window_seconds <= 0:
            raise ValueError(f"Window duration must be positive, got {window_seconds}")

        self.threshold = threshold
        self.window_seconds = window_seconds

        # Map of (src_ip, dst_ip) -> list of timestamps
        self._history: Dict[Tuple[str, str], List[float]] = {}

        # Suppression tracking: (src_ip, dst_ip) -> (alerted: bool, last_alert_time: float)
        self._suppression: Dict[Tuple[str, str], Tuple[bool, float]] = {}

        # Highest timestamp observed so far across this traffic stream
        self._latest_timestamp: float = 0.0

    @staticmethod
    def is_initial_syn(packet: NormalizedPacket) -> bool:
        """Check whether the packet is a TCP initial SYN (SYN set, ACK not set).

        Args:
            packet: Incoming NormalizedPacket.

        Returns:
            bool: True if protocol is TCP and flags contain 'S' but not 'A'.
        """
        if packet.protocol != "TCP" or not packet.tcp_flags:
            return False
        flags = packet.tcp_flags.upper()
        return ("S" in flags) and ("A" not in flags)

    def process_packet(self, packet: NormalizedPacket) -> Optional[Alert]:
        """Process a single packet and return an Alert if the SYN rate threshold is breached.

        Args:
            packet: The incoming NormalizedPacket.

        Returns:
            Optional[Alert]: An Alert instance if the SYN rate threshold is crossed, else None.
        """
        # 1. Filter for valid TCP initial SYN packets
        if not self.is_initial_syn(packet):
            return None
        if not packet.src_ip or not packet.dst_ip:
            return None

        current_time = float(packet.timestamp)

        # 2. Out-of-order handling: discard packets older than sliding window
        if current_time < (self._latest_timestamp - self.window_seconds):
            return None
        self._latest_timestamp = max(self._latest_timestamp, current_time)

        key = (packet.src_ip, packet.dst_ip)
        if key not in self._history:
            self._history[key] = []
            self._suppression[key] = (False, 0.0)

        history = self._history[key]
        alerted, last_alert_time = self._suppression[key]

        # 3. Append timestamp and prune expired timestamps
        history.append(current_time)
        cutoff = current_time - self.window_seconds
        self._history[key] = [ts for ts in history if ts >= cutoff]
        syn_count = len(self._history[key])

        # 4. Evaluate suppression reset condition
        if alerted:
            if syn_count < self.threshold or (current_time >= last_alert_time + self.window_seconds):
                alerted = False
                self._suppression[key] = (False, last_alert_time)

        # 5. Check threshold and generate alert
        if syn_count >= self.threshold and not alerted:
            self._suppression[key] = (True, current_time)
            rate_per_sec = round(syn_count / self.window_seconds, 2)
            return Alert(
                timestamp=current_time,
                rule_name="SYN_RATE",
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                observed_value=syn_count,
                threshold=self.threshold,
                window_seconds=self.window_seconds,
                severity="HIGH" if syn_count >= (self.threshold * 2) else "MEDIUM",
                message=(
                    f"High TCP SYN rate detected from {packet.src_ip} targeting {packet.dst_ip}: "
                    f"{syn_count} SYN packets in {self.window_seconds}s "
                    f"(~{rate_per_sec} pkts/s, threshold={self.threshold})."
                ),
            )

        return None

    def process_packets(self, packets: Iterable[NormalizedPacket]) -> List[Alert]:
        """Convenience method to process an iterable of packets and collect alerts.

        Args:
            packets: Iterable of NormalizedPacket instances.

        Returns:
            List[Alert]: All alerts triggered during processing.
        """
        alerts: List[Alert] = []
        for packet in packets:
            alert = self.process_packet(packet)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def reset(self) -> None:
        """Clear all sliding window history and alert suppression state."""
        self._history.clear()
        self._suppression.clear()
        self._latest_timestamp = 0.0
