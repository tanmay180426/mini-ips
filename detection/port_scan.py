"""TCP Port Scan Detection Module.

Identifies potential reconnaissance activities where a single source IP connects
to multiple distinct destination TCP ports on the same target host within a
configurable sliding time window.

Detector Characteristics:
- Target pair: Evaluated per (source IP, destination IP) pair.
- Metric: Counts unique destination ports, not raw packet counts (repeated packets
  to the same port do not inflate the metric).
- Protocol filter: Strictly inspects TCP packets. UDP, ICMP, and non-IP traffic
  are safely ignored.
- Suppression: After triggering an alert for a (src, dst) pair, subsequent alerts
  for that pair are suppressed until either distinct ports drop below the threshold
  or the time window has completely elapsed.
- Out-of-order handling: Packets with timestamps older than (latest_seen - window)
  are safely discarded as expired. Out-of-order packets within the active window
  are accepted and evaluated.

NOTE: This detector is a heuristic indicator and may generate false positives
for legitimate multi-port traffic (e.g., active FTP, BitTorrent, or web services).
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

from capture.packet_capture import NormalizedPacket
from detection import Alert


class PortScanDetector:
    """Detects TCP port scan activity across a single traffic stream.

    Attributes:
        threshold (int): Minimum distinct destination ports required to trigger an alert.
                         Default: 15 distinct ports.
        window_seconds (float): Time window in seconds across which distinct ports are tracked.
                                Default: 30.0 seconds.
    """

    DEFAULT_THRESHOLD: int = 15
    DEFAULT_WINDOW_SECONDS: float = 30.0

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

        # Map of (src_ip, dst_ip) -> list of (timestamp, dst_port)
        self._history: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}

        # Suppression tracking: (src_ip, dst_ip) -> (alerted: bool, last_alert_time: float)
        self._suppression: Dict[Tuple[str, str], Tuple[bool, float]] = {}

        # Highest timestamp observed so far across this traffic stream
        self._latest_timestamp: float = 0.0

    def process_packet(self, packet: NormalizedPacket) -> Optional[Alert]:
        """Process a single normalized packet and return an Alert if a scan is detected.

        Args:
            packet: The incoming NormalizedPacket.

        Returns:
            Optional[Alert]: An Alert instance if the scan threshold is crossed, else None.
        """
        # 1. Validate TCP protocol and required fields
        if packet.protocol != "TCP":
            return None
        if not packet.src_ip or not packet.dst_ip or packet.dst_port is None:
            return None

        current_time = float(packet.timestamp)

        # 2. Out-of-order handling: ignore ancient packets older than current sliding window
        if current_time < (self._latest_timestamp - self.window_seconds):
            return None
        self._latest_timestamp = max(self._latest_timestamp, current_time)

        key = (packet.src_ip, packet.dst_ip)
        if key not in self._history:
            self._history[key] = []
            self._suppression[key] = (False, 0.0)

        history = self._history[key]
        alerted, last_alert_time = self._suppression[key]

        # 3. Append current observation and prune expired entries outside sliding window
        history.append((current_time, packet.dst_port))
        cutoff = current_time - self.window_seconds
        self._history[key] = [(ts, port) for ts, port in history if ts >= cutoff]
        history = self._history[key]

        # 4. Count unique destination ports in current window
        distinct_ports: Set[int] = {port for _, port in history}
        distinct_count = len(distinct_ports)

        # 5. Evaluate suppression reset condition
        if alerted:
            # Reset if scan activity dropped below threshold or a full window elapsed since alert
            if distinct_count < self.threshold or (current_time >= last_alert_time + self.window_seconds):
                alerted = False
                self._suppression[key] = (False, last_alert_time)

        # 6. Check threshold and generate alert
        if distinct_count >= self.threshold and not alerted:
            self._suppression[key] = (True, current_time)
            return Alert(
                timestamp=current_time,
                rule_name="PORT_SCAN",
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                observed_value=distinct_count,
                threshold=self.threshold,
                window_seconds=self.window_seconds,
                severity="HIGH" if distinct_count >= (self.threshold * 2) else "MEDIUM",
                message=(
                    f"Potential TCP port scan detected from {packet.src_ip} targeting "
                    f"{packet.dst_ip}: {distinct_count} distinct ports contacted "
                    f"within {self.window_seconds}s (threshold={self.threshold})."
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
