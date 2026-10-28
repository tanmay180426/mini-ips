"""Mini-IDS Core CLI Entrypoint.

Command-line interface to coordinate offline PCAP analysis:
1. Ingest packets via the capture module (`iter_pcap`).
2. Feed each normalized packet to detection rules exactly once.
3. Persist normalized packets and generated alerts into SQLite (`DatabaseManager`).
4. Output analysis statistics and alert summaries to console.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Suppress runtime warnings from Scapy if libpcap is absent
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from capture.packet_capture import iter_pcap
from detection import (
    Alert,
    PortScanDetector,
    SynRateDetector,
    TrafficAnomalyDetector,
)
from storage.database import DEFAULT_DB_PATH, DatabaseManager

logger = logging.getLogger("mini_ids")


def process_pcap_stream(
    pcap_path: Union[str, Path],
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    detectors: Optional[List[Any]] = None,
    batch_size: int = 500,
    print_alerts: bool = True,
    enable_hybrid: bool = False,
) -> Dict[str, Any]:
    """Process an offline PCAP/PCAPNG file through detectors and persist results.

    Each normalized packet is passed to every configured detector exactly once.
    Packets and triggered alerts are stored in the SQLite database.

    Args:
        pcap_path: Path to the .pcap or .pcapng file.
        db_path: Path to the target SQLite database file.
        detectors: Optional list of detector instances. If None, default
                   heuristic detectors or hybrid engine are instantiated.
        batch_size: Number of packets to accumulate before committing to database.
        print_alerts: If True, prints alert notifications to stdout as they occur.
        enable_hybrid: If True, activates flow extraction and trained ML models.

    Returns:
        Dict[str, Any]: Summary dictionary containing:
            - 'packets_processed': Total normalized packets ingested and persisted.
            - 'alerts_generated': Total alerts emitted and persisted.
            - 'rule_counts': Dictionary mapping rule names to trigger counts.
            - 'db_path': Path to the SQLite database.

    Raises:
        FileNotFoundError: If pcap_path does not exist or is not a file.
        ValueError: If batch_size <= 0.
    """
    path = Path(pcap_path)
    if not path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # 1. Initialize database and ensure tables exist
    target_db_path = Path(db_path)
    db = DatabaseManager(target_db_path)
    db.init_db()

    # 2. Instantiate detector instances
    if enable_hybrid and detectors is None:
        try:
            from detection.hybrid_detection import HybridDetectionEngine
            from ml.model_registry import ModelRegistry
            from prevention.engine import PreventionEngine
            from prevention.policy_engine import PreventionMode

            prev_engine = PreventionEngine(db_manager=db, initial_mode=PreventionMode.SIMULATION)
            registry = ModelRegistry(db_path=target_db_path)
            sup_bundle = registry.load_active_bundle("supervised_binary") or registry.load_active_bundle()
            anom_bundle = registry.load_active_bundle("anomaly_isolation_forest")

            sup_m, sup_p, sup_meta = sup_bundle if sup_bundle else (None, None, None)
            anom_m, anom_p, anom_meta = anom_bundle if anom_bundle else (None, None, None)

            hybrid_engine = HybridDetectionEngine(
                supervised_model=sup_m,
                supervised_preprocessor=sup_p,
                supervised_meta=sup_meta,
                anomaly_model=anom_m,
                anomaly_preprocessor=anom_p,
                anomaly_meta=anom_meta,
                prevention_engine=prev_engine,
                source_tag="pcap",
            )
            active_detectors = [hybrid_engine]
        except Exception as err:
            logger.warning("Could not initialize hybrid detection engine: %s. Using heuristic detectors.", err)
            active_detectors = [
                PortScanDetector(),
                SynRateDetector(),
                TrafficAnomalyDetector(),
            ]
    elif detectors is None:
        active_detectors = [
            PortScanDetector(),
            SynRateDetector(),
            TrafficAnomalyDetector(),
        ]
    else:
        active_detectors = detectors

    packets_processed = 0
    alerts_generated = 0
    rule_counts: Dict[str, int] = {}
    packet_batch = []

    try:
        # 3. Stream packets iteratively (memory-safe for large captures)
        for packet in iter_pcap(path):
            packets_processed += 1
            packet_batch.append(packet)

            # Flush packet batch periodically
            if len(packet_batch) >= batch_size:
                db.insert_packets(packet_batch)
                packet_batch.clear()

            # 4. Pass each packet to detectors
            for detector in active_detectors:
                if hasattr(detector, "process_packet_all"):
                    det_alerts, flows = detector.process_packet_all(packet)
                    if flows:
                        db.insert_flows(flows)
                else:
                    alt = detector.process_packet(packet)
                    det_alerts = [alt] if alt is not None else []

                for alert in det_alerts:
                    db.insert_alert(alert)
                    alerts_generated += 1
                    rule_counts[alert.rule_name] = rule_counts.get(alert.rule_name, 0) + 1

                    if print_alerts:
                        print(
                            f"  -> [{alert.severity}] {alert.rule_name} | "
                            f"{alert.src_ip} -> {alert.dst_ip} | {alert.message}"
                        )

        # 5. Flush any remaining packets in the buffer
        if packet_batch:
            db.insert_packets(packet_batch)
            packet_batch.clear()

        # Flush remaining flows from detectors
        for detector in active_detectors:
            if hasattr(detector, "flush"):
                flush_alerts, flush_flows = detector.flush()
                if flush_flows:
                    db.insert_flows(flush_flows)
                for alert in flush_alerts:
                    db.insert_alert(alert)
                    alerts_generated += 1
                    rule_counts[alert.rule_name] = rule_counts.get(alert.rule_name, 0) + 1
                    if print_alerts:
                        print(
                            f"  -> [{alert.severity}] {alert.rule_name} | "
                            f"{alert.src_ip} -> {alert.dst_ip} | {alert.message}"
                        )

    finally:
        # Guarantee database connection is cleanly closed
        db.close()

    return {
        "packets_processed": packets_processed,
        "alerts_generated": alerts_generated,
        "rule_counts": rule_counts,
        "db_path": target_db_path,
    }


from capture.live_capture import LiveCaptureWorker, get_available_interfaces


def build_parser() -> argparse.ArgumentParser:
    """Build and return command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mini-ids",
        description="Mini-IDS: Network Intrusion Detection System (Offline PCAP & Live Capture)",
    )
    parser.add_argument(
        "-p",
        "--pcap",
        required=False,
        default=None,
        type=str,
        help="Path to the offline PCAP or PCAPNG capture file to analyze.",
    )
    parser.add_argument(
        "-i",
        "--interface",
        required=False,
        default=None,
        type=str,
        help="Name or device identifier of network interface for live capture.",
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List all network interfaces discovered by Scapy and exit.",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=None,
        help="Maximum packet limit for live capture.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=None,
        help="Capture duration timeout in seconds for live capture.",
    )
    parser.add_argument(
        "-d",
        "--db",
        default=str(DEFAULT_DB_PATH),
        type=str,
        help=f"Path to SQLite database for persisting packets and alerts (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid detection (heuristic rules + active supervised ML classifier + Isolation Forest anomaly detection).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Handle interface listing
    if args.list_interfaces:
        ifaces = get_available_interfaces()
        print("=" * 60)
        print("Mini-IDS: Available Network Interfaces")
        print("=" * 60)
        if not ifaces:
            print("No network interfaces discovered (Npcap may not be installed).")
        else:
            for iface in ifaces:
                print(f"  * {iface['display_name']}")
        print("=" * 60)
        return 0

    # 2. Validate input modes
    if args.pcap and args.interface:
        parser.print_help()
        print(
            "\n[ERROR] --pcap and --interface are mutually exclusive. Specify only one ingestion source.",
            file=sys.stderr,
        )
        return 1

    if not args.pcap and not args.interface:
        parser.print_help()
        print(
            "\n[ERROR] Either --pcap <file> or --interface <name> must be specified (or use --list-interfaces).",
            file=sys.stderr,
        )
        return 1

    # 3. Validate capture limits and timeouts
    if args.count is not None and args.count <= 0:
        print(
            f"\n[ERROR] Invalid --count value '{args.count}'. Packet limit must be a positive integer (>= 1).",
            file=sys.stderr,
        )
        return 1

    if args.timeout is not None and args.timeout <= 0:
        print(
            f"\n[ERROR] Invalid --timeout value '{args.timeout}'. Duration timeout must be greater than 0 seconds.",
            file=sys.stderr,
        )
        return 1

    if args.pcap and (args.count is not None or args.timeout is not None):
        print(
            "\n[ERROR] --count and --timeout options are only applicable to live capture (--interface), not offline PCAP.",
            file=sys.stderr,
        )
        return 1

    db_file = Path(args.db)

    # 4. Handle live capture mode
    if args.interface:
        import time

        print("=" * 60)
        print("Mini-IDS: Live Network Intrusion Detection Capture")
        print("=" * 60)
        print(f"Interface:    {args.interface}")
        print(f"Database:     {db_file}")
        if args.count is not None:
            print(f"Packet Limit: {args.count}")
        if args.timeout is not None:
            print(f"Timeout:      {args.timeout}s")
        print("-" * 60)
        print("Capturing packets and evaluating detection rules (Press Ctrl+C to stop)...")

        worker = LiveCaptureWorker(
            interface=args.interface,
            db_path=db_file,
            packet_limit=args.count,
            duration_seconds=args.timeout,
        )

        try:
            worker.start()
            while worker.is_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping capture...")
        finally:
            worker.stop()

        status = worker.get_status()
        print("-" * 60)
        print("Capture Complete!")
        print(f"  * Status:           {status['status']}")
        print(f"  * Packets captured: {status['packets_captured']}")
        print(f"  * Alerts triggered: {status['alerts_generated']}")
        print(f"  * Elapsed time:     {status['elapsed_seconds']:.2f}s")
        print(f"  * Database saved:   {status['db_path']}")
        if status.get("error_message"):
            print(f"  * Error:            {status['error_message']}", file=sys.stderr)
            return 1
        print("=" * 60)
        return 0

    # 4. Handle offline PCAP mode
    pcap_file = Path(args.pcap)
    print("=" * 60)
    print("Mini-IDS: Offline Network Intrusion Detection System")
    print("=" * 60)
    print(f"Target PCAP: {pcap_file}")
    print(f"Database:    {db_file}")
    print("-" * 60)
    print("Ingesting packets and evaluating detection rules...")

    try:
        summary = process_pcap_stream(
            pcap_path=pcap_file,
            db_path=db_file,
            print_alerts=True,
            enable_hybrid=args.hybrid,
        )
    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR] {fnf_err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\n[ERROR] Failed to process PCAP: {err}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("Analysis Complete!")
    print(f"  * Packets analyzed: {summary['packets_processed']}")
    print(f"  * Alerts triggered: {summary['alerts_generated']}")
    if summary["rule_counts"]:
        print("  * Alerts by Rule:")
        for rule, count in summary["rule_counts"].items():
            print(f"      - {rule}: {count}")
    print(f"  * Database saved:   {summary['db_path']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
