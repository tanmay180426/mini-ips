"""Traffic Intelligence and Statistical Aggregation Module.

Computes discrete time bucket metrics, protocol/port distributions, and
benign vs. suspicious traffic proportions from SQLite database records.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from storage.database import DatabaseManager


class TrafficStatistics:
    """Aggregates security telemetry and network metrics for visualization."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def get_discrete_traffic_volume(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        protocol: Optional[str] = None,
        bucket_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        """Aggregate packets into discrete time intervals to prevent false slopes."""
        packets = self.db.get_related_packets(
            start_time=start_time,
            end_time=end_time,
            protocol=protocol,
            limit=50000,
        )
        if not packets:
            return []

        df = pd.DataFrame(packets)
        # Bucket by integer division
        df["bucket_epoch"] = (df["timestamp"] // bucket_seconds) * bucket_seconds
        df["bucket_str"] = pd.to_datetime(df["bucket_epoch"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")

        grouped = df.groupby(["bucket_epoch", "bucket_str"]).agg(
            packet_count=("id", "count"),
            byte_count=("length", "sum"),
        ).reset_index().sort_values("bucket_epoch")

        return grouped.to_dict(orient="records")

    def get_benign_vs_suspicious_ratio(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate proportion of normal vs alert-associated traffic."""
        kpis = self.db.get_security_kpis(start_time=start_time, end_time=end_time)
        tot_pkts = kpis["total_packets"]
        tot_alerts = kpis["total_alerts"]

        # Alert packets (upper bound: each alert relates to packet activity)
        suspicious_est = min(tot_pkts, tot_alerts * 5)
        benign_est = max(0, tot_pkts - suspicious_est)

        return {
            "total_packets": tot_pkts,
            "total_alerts": tot_alerts,
            "benign_packets": benign_est,
            "suspicious_packets": suspicious_est,
            "suspicious_ratio_pct": round(suspicious_est * 100.0 / max(tot_pkts, 1), 2),
        }
