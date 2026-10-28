"""Traffic Volume Anomaly Detection Module.

Identifies sudden surges in network traffic volume within a configurable sliding
time window based on normalized packet lengths and timestamps.

Detector Characteristics:
- Dual-metric tracking:
  - Byte volume: Evaluated against `byte_threshold` using the captured packet frame
    lengths (`NormalizedPacket.length`).
  - Packet count: Evaluated against `packet_threshold` using the total count of packets.
  An alert triggers if either configured threshold is breached.
- Scope: Configurable to track per source IP (`per_source=True`, default) or globally
  across the entire stream (`per_source=False`).
- Heuristic transparency: Uses pure, deterministic sliding-window threshold counting.
  No machine learning, stochastic models, or external statistical dependencies.
- Suppression: Once an anomaly alert fires for a source/stream, repeated alerts are
  suppressed until the volume drops below threshold or the window duration elapses.
- Out-of-order handling: Packets with timestamps older than (latest_seen - window)
  are ignored as expired.

NOTE: This detector is a threshold-based indicator and may produce false positives
during benign bulk operations (e.g., large file downloads, OS updates, or video streaming).
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

from capture.packet_capture import NormalizedPacket
from detection import Alert


class TrafficAnomalyDetector:
    """Detects sudden spikes in packet count or byte volume across a traffic stream.

    Attributes:
        byte_threshold (Optional[int]): Minimum bytes in window to trigger an alert.
                                        Default: 500,000 bytes (~500 KB).
        packet_threshold (Optional[int]): Minimum packet count in window to trigger an alert.
                                          Default: 500 packets.
        window_seconds (float): Duration of the rolling evaluation window in seconds.
                                Default: 10.0 seconds.
        per_source (bool): If True, tracks volume per source IP. If False, tracks aggregate stream volume.
                           Default: True.
    """

    DEFAULT_BYTE_THRESHOLD: int = 500_000
    DEFAULT_PACKET_THRESHOLD: int = 500
    DEFAULT_WINDOW_SECONDS: float = 10.0

    def __init__(
        self,
        byte_threshold: Optional[int] = DEFAULT_BYTE_THRESHOLD,
        packet_threshold: Optional[int] = DEFAULT_PACKET_THRESHOLD,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        per_source: bool = True,
    ) -> None:
        if byte_threshold is None and packet_threshold is None:
            raise ValueError("At least one of byte_threshold or packet_threshold must be provided.")
        if byte_threshold is not None and byte_threshold <= 0:
            raise ValueError(f"byte_threshold must be positive, got {byte_threshold}")
        if packet_threshold is not None and packet_threshold <= 0:
            raise ValueError(f"packet_threshold must be positive, got {packet_threshold}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")

        self.byte_threshold = byte_threshold
        self.packet_threshold = packet_threshold
        self.window_seconds = window_seconds
        self.per_source = per_source

        # Map of key -> list of (timestamp, length)
        # key is src_ip if per_source=True else "GLOBAL_STREAM"
        self._history: Dict[str, List[Tuple[float, int]]] = {}

        # Suppression tracking: key -> (alerted: bool, last_alert_time: float)
        self._suppression: Dict[str, Tuple[bool, float]] = {}

        # Highest timestamp observed so far across this traffic stream
        self._latest_timestamp: float = 0.0

    def process_packet(self, packet: NormalizedPacket) -> Optional[Alert]:
        """Process a single packet and return an Alert if a volume threshold is breached.

        Args:
            packet: The incoming NormalizedPacket.

        Returns:
            Optional[Alert]: An Alert instance if volume threshold is crossed, else None.
        """
        # Determine tracking key
        if self.per_source:
            key = packet.src_ip if packet.src_ip else "UNKNOWN_SRC"
        else:
            key = "GLOBAL_STREAM"

        current_time = float(packet.timestamp)
        pkt_len = int(packet.length) if packet.length else 0

        # Out-of-order handling: discard packets older than sliding window
        if current_time < (self._latest_timestamp - self.window_seconds):
            return None
        self._latest_timestamp = max(self._latest_timestamp, current_time)

        if key not in self._history:
            self._history[key] = []
            self._suppression[key] = (False, 0.0)

        history = self._history[key]
        alerted, last_alert_time = self._suppression[key]

        # Append observation and prune records outside sliding window
        history.append((current_time, pkt_len))
        cutoff = current_time - self.window_seconds
        self._history[key] = [(ts, length) for ts, length in history if ts >= cutoff]
        history = self._history[key]

        observed_packets = len(history)
        observed_bytes = sum(length for _, length in history)

        # Check breach conditions
        byte_breached = self.byte_threshold is not None and observed_bytes >= self.byte_threshold
        packet_breached = self.packet_threshold is not None and observed_packets >= self.packet_threshold
        condition_active = byte_breached or packet_breached

        # Evaluate suppression reset
        if alerted:
            if not condition_active or (current_time >= last_alert_time + self.window_seconds):
                alerted = False
                self._suppression[key] = (False, last_alert_time)

        # Generate alert if condition active and not suppressed
        if condition_active and not alerted:
            self._suppression[key] = (True, current_time)

            if byte_breached and packet_breached:
                metric_name = "bytes and packets"
                observed_value = observed_bytes
                threshold_val = self.byte_threshold
                detail = f"{observed_bytes} bytes across {observed_packets} packets"
            elif byte_breached:
                metric_name = "bytes"
                observed_value = observed_bytes
                threshold_val = self.byte_threshold
                detail = f"{observed_bytes} bytes (threshold={self.byte_threshold})"
            else:
                metric_name = "packets"
                observed_value = observed_packets
                threshold_val = self.packet_threshold
                detail = f"{observed_packets} packets (threshold={self.packet_threshold})"

            src_label = packet.src_ip if self.per_source else "global stream"
            return Alert(
                timestamp=current_time,
                rule_name="TRAFFIC_VOLUME",
                src_ip=packet.src_ip if self.per_source else None,
                dst_ip=packet.dst_ip if self.per_source else None,
                observed_value=observed_value,
                threshold=threshold_val if threshold_val is not None else 0,
                window_seconds=self.window_seconds,
                severity="HIGH" if (observed_bytes >= (self.byte_threshold or 0) * 2) else "MEDIUM",
                message=(
                    f"Traffic volume surge ({metric_name}) detected from {src_label}: "
                    f"{detail} in {self.window_seconds}s."
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
