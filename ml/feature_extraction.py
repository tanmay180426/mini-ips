"""Bidirectional Network Flow Aggregation and Feature Extraction Module.

Transforms individual normalized network packets into stateful, bidirectional
network flows and computes statistical features compatible with standard flow-based
intrusion detection benchmark schemas (e.g. CICIDS2017).
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

from capture.packet_capture import NormalizedPacket


@dataclass
class FlowStatistics:
    """Statistical tracking state for a single flow direction."""
    packet_count: int = 0
    byte_count: int = 0
    packet_lengths: List[int] = field(default_factory=list)
    arrival_times: List[float] = field(default_factory=list)
    syn_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0
    urg_count: int = 0


@dataclass
class NetworkFlow:
    """Stateful bidirectional network flow representation."""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    start_time: float
    end_time: float
    fwd_stats: FlowStatistics = field(default_factory=FlowStatistics)
    bwd_stats: FlowStatistics = field(default_factory=FlowStatistics)
    is_closed: bool = False

    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return max(0.0, self.end_time - self.start_time)

    @property
    def total_packets(self) -> int:
        return self.fwd_stats.packet_count + self.bwd_stats.packet_count

    @property
    def total_bytes(self) -> int:
        return self.fwd_stats.byte_count + self.bwd_stats.byte_count

    def extract_features(self) -> Dict[str, float]:
        """Compute tabular statistical features matching CICIDS2017 schema conventions."""
        dur = self.duration
        fwd_lens = self.fwd_stats.packet_lengths
        bwd_lens = self.bwd_stats.packet_lengths
        all_lens = fwd_lens + bwd_lens

        total_pkts = self.total_packets
        total_bytes = self.total_bytes

        # Rates (safe division by non-zero duration)
        eff_dur = max(dur, 1e-6)
        flow_bytes_s = float(total_bytes) / eff_dur
        flow_pkts_s = float(total_pkts) / eff_dur

        # Packet length stats
        def calc_mean_std(arr: List[int]) -> Tuple[float, float, float, float]:
            if not arr:
                return 0.0, 0.0, 0.0, 0.0
            n = len(arr)
            mean = sum(arr) / n
            variance = sum((x - mean) ** 2 for x in arr) / n if n > 1 else 0.0
            return mean, math.sqrt(variance), float(min(arr)), float(max(arr))

        fwd_mean, fwd_std, fwd_min, fwd_max = calc_mean_std(fwd_lens)
        bwd_mean, bwd_std, bwd_min, bwd_max = calc_mean_std(bwd_lens)
        all_mean, all_std, all_min, all_max = calc_mean_std(all_lens)

        # Inter-arrival times (IAT)
        def calc_iat_stats(times: List[float]) -> Tuple[float, float, float, float]:
            if len(times) < 2:
                return 0.0, 0.0, 0.0, 0.0
            iats = [times[i] - times[i - 1] for i in range(1, len(times))]
            n = len(iats)
            mean = sum(iats) / n
            var = sum((x - mean) ** 2 for x in iats) / n if n > 1 else 0.0
            return mean, math.sqrt(var), min(iats), max(iats)

        fwd_iat_mean, fwd_iat_std, fwd_iat_min, fwd_iat_max = calc_iat_stats(self.fwd_stats.arrival_times)
        bwd_iat_mean, bwd_iat_std, bwd_iat_min, bwd_iat_max = calc_iat_stats(self.bwd_stats.arrival_times)

        all_times = sorted(self.fwd_stats.arrival_times + self.bwd_stats.arrival_times)
        flow_iat_mean, flow_iat_std, flow_iat_min, flow_iat_max = calc_iat_stats(all_times)

        # Flag counts
        syn_count = self.fwd_stats.syn_count + self.bwd_stats.syn_count
        fin_count = self.fwd_stats.fin_count + self.bwd_stats.fin_count
        rst_count = self.fwd_stats.rst_count + self.bwd_stats.rst_count
        psh_count = self.fwd_stats.psh_count + self.bwd_stats.psh_count
        ack_count = self.fwd_stats.ack_count + self.bwd_stats.ack_count

        # Down/Up Ratio
        down_up_ratio = (
            float(self.bwd_stats.packet_count) / float(self.fwd_stats.packet_count)
            if self.fwd_stats.packet_count > 0
            else 0.0
        )

        return {
            "flow_duration": dur,
            "total_fwd_packets": float(self.fwd_stats.packet_count),
            "total_bwd_packets": float(self.bwd_stats.packet_count),
            "total_length_of_fwd_packets": float(self.fwd_stats.byte_count),
            "total_length_of_bwd_packets": float(self.bwd_stats.byte_count),
            "fwd_packet_length_mean": fwd_mean,
            "fwd_packet_length_std": fwd_std,
            "fwd_packet_length_min": fwd_min,
            "fwd_packet_length_max": fwd_max,
            "bwd_packet_length_mean": bwd_mean,
            "bwd_packet_length_std": bwd_std,
            "bwd_packet_length_min": bwd_min,
            "bwd_packet_length_max": bwd_max,
            "flow_bytes_s": flow_bytes_s,
            "flow_packets_s": flow_pkts_s,
            "flow_iat_mean": flow_iat_mean,
            "flow_iat_std": flow_iat_std,
            "flow_iat_min": flow_iat_min,
            "flow_iat_max": flow_iat_max,
            "fwd_iat_mean": fwd_iat_mean,
            "fwd_iat_std": fwd_iat_std,
            "bwd_iat_mean": bwd_iat_mean,
            "bwd_iat_std": bwd_iat_std,
            "fin_flag_count": float(fin_count),
            "syn_flag_count": float(syn_count),
            "rst_flag_count": float(rst_count),
            "psh_flag_count": float(psh_count),
            "ack_flag_count": float(ack_count),
            "down_up_ratio": down_up_ratio,
            "average_packet_size": all_mean,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert flow summary to dictionary for persistence."""
        return {
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "packet_count": self.total_packets,
            "byte_count": self.total_bytes,
            "fwd_packets": self.fwd_stats.packet_count,
            "bwd_packets": self.bwd_stats.packet_count,
            "fwd_bytes": self.fwd_stats.byte_count,
            "bwd_bytes": self.bwd_stats.byte_count,
            "flow_pkts_s": float(self.total_packets) / max(self.duration, 1e-6),
            "flow_bytes_s": float(self.total_bytes) / max(self.duration, 1e-6),
            "features": self.extract_features(),
        }


class FlowFeatureExtractor:
    """Stateful aggregator that streams normalized packets into bidirectional flows.

    Maintains active flows and yields finalized flows when idle timeout expires
    or when TCP FIN/RST termination flags are observed.
    """

    def __init__(self, flow_timeout: float = 15.0) -> None:
        self.flow_timeout = flow_timeout
        self._active_flows: Dict[Tuple[str, str, Optional[int], Optional[int], str], NetworkFlow] = {}

    @property
    def active_flows(self) -> Dict[Tuple[str, str, Optional[int], Optional[int], str], NetworkFlow]:
        """Return the dictionary of active in-flight flows."""
        return self._active_flows

    def _canonical_key(
        self, p: NormalizedPacket
    ) -> Tuple[Tuple[str, str, Optional[int], Optional[int], str], bool]:
        """Generate direction-agnostic canonical 5-tuple key.

        Returns:
            Tuple: (canonical_key, is_forward)
        """
        fwd_pair = (p.src_ip, p.dst_ip, p.src_port, p.dst_port, p.protocol.upper())
        rev_pair = (p.dst_ip, p.src_ip, p.dst_port, p.src_port, p.protocol.upper())

        if fwd_pair in self._active_flows:
            return fwd_pair, True
        if rev_pair in self._active_flows:
            return rev_pair, False

        # First packet defines the forward direction
        return fwd_pair, True

    def process_packet(self, packet: NormalizedPacket) -> List[NetworkFlow]:
        """Ingest a packet, update active flows, and return any expired/completed flows."""
        completed_flows: List[NetworkFlow] = []
        now = float(packet.timestamp)

        # 1. Check for expired flows that timed out
        keys_to_remove = []
        for key, flow in self._active_flows.items():
            if (now - flow.end_time) > self.flow_timeout:
                flow.is_closed = True
                completed_flows.append(flow)
                keys_to_remove.append(key)

        for k in keys_to_remove:
            del self._active_flows[k]

        # 2. Get or initialize flow
        key, is_fwd = self._canonical_key(packet)
        if key not in self._active_flows:
            flow_id = f"{packet.src_ip}:{packet.src_port}->{packet.dst_ip}:{packet.dst_port}_{packet.protocol}_{now:.2f}"
            flow = NetworkFlow(
                flow_id=flow_id,
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                src_port=packet.src_port,
                dst_port=packet.dst_port,
                protocol=packet.protocol.upper(),
                start_time=now,
                end_time=now,
            )
            self._active_flows[key] = flow
        else:
            flow = self._active_flows[key]
            flow.end_time = max(flow.end_time, now)

        # 3. Update directional stats
        stats = flow.fwd_stats if is_fwd else flow.bwd_stats
        stats.packet_count += 1
        stats.byte_count += packet.length
        stats.packet_lengths.append(packet.length)
        stats.arrival_times.append(now)

        flags = (packet.tcp_flags or "").upper()
        if "S" in flags:
            stats.syn_count += 1
        if "F" in flags:
            stats.fin_count += 1
        if "R" in flags:
            stats.rst_count += 1
        if "P" in flags:
            stats.psh_count += 1
        if "A" in flags:
            stats.ack_count += 1
        if "U" in flags:
            stats.urg_count += 1

        # 4. Check TCP termination
        if "F" in flags or "R" in flags:
            # If both directions or RST observed, close flow
            if "R" in flags or (flow.fwd_stats.fin_count > 0 and flow.bwd_stats.fin_count > 0):
                flow.is_closed = True
                completed_flows.append(flow)
                if key in self._active_flows:
                    del self._active_flows[key]

        return completed_flows

    def flush(self) -> List[NetworkFlow]:
        """Flush all remaining active flows (e.g. at end of PCAP)."""
        flows = list(self._active_flows.values())
        for f in flows:
            f.is_closed = True
        self._active_flows.clear()
        return flows
