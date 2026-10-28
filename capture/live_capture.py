"""Live Packet Capture and Streaming Ingestion Module.

Provides non-blocking, thread-safe live network packet capture using Scapy's
Windows/Npcap or libpcap backend. Normalizes incoming packets, feeds them to
active detection rules, and persists records to SQLite.

Key Features:
- Safe enumeration of available network interfaces with human-readable labels.
- Background worker thread with queue-based decoupling to prevent packet drops.
- Thread-safe status telemetry, counters, and lifecycle controls (start/stop).
- Automatic batching and thread-confined SQLite database connections.
- Configurable packet count limits and duration timeouts.
- Graceful error handling for missing interfaces, permissions, or Npcap faults.
- Prevention of duplicate or concurrent capture workers.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Suppress Scapy runtime warnings on Windows when libpcap/Npcap is absent
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.all import AsyncSniffer, conf

from capture.packet_capture import NormalizedPacket, normalize_packet
from detection import Alert
# Default location for persistent project storage
DEFAULT_DB_PATH = Path("data") / "mini_ids.db"

logger = logging.getLogger("mini_ids.live_capture")


def get_available_interfaces() -> List[Dict[str, Any]]:
    """Enumerate and format all network interfaces discovered by Scapy.

    Returns:
        List[Dict[str, Any]]: List of interface metadata dictionaries, sorted
        with active IPv4-assigned adapters first, containing:
            - 'name': System or friendly adapter name.
            - 'description': Device description or driver string.
            - 'ip': Assigned IPv4 address (or empty string).
            - 'mac': Hardware MAC address (or empty string).
            - 'device': Device name or Scapy interface identifier.
            - 'display_name': User-friendly formatted label for UI dropdowns.
            - 'has_ip': Boolean indicating whether an active IPv4 is configured.
    """
    interfaces: List[Dict[str, Any]] = []

    try:
        if not hasattr(conf, "ifaces") or not conf.ifaces:
            return interfaces

        for iface in conf.ifaces.values():
            name = str(getattr(iface, "name", "")).strip()
            desc = str(getattr(iface, "description", "")).strip()
            ip = str(getattr(iface, "ip", "")).strip()
            mac = str(getattr(iface, "mac", "")).strip()
            guid = str(getattr(iface, "network_name", getattr(iface, "pcap_name", name))).strip()

            if not name and not desc:
                continue

            # Construct human-readable UI label
            label = name if name else desc
            if ip and ip != "0.0.0.0":
                label += f" ({ip})"
            if desc and desc != name:
                label += f" — {desc}"

            is_active_ip = bool(
                ip and ip != "0.0.0.0" and not ip.startswith("169.254") and not ip.startswith("127.")
            )
            is_loopback = "loopback" in name.lower() or ip.startswith("127.")

            interfaces.append({
                "name": name,
                "description": desc,
                "ip": ip,
                "mac": mac,
                "guid": guid,
                "device": name if name else guid,
                "display_name": label,
                "has_ip": is_active_ip,
                "is_loopback": is_loopback,
            })

        # Sort: Active routable IP adapters first, then loopback, then disconnected/other
        interfaces.sort(key=lambda x: (not x["has_ip"], not x["is_loopback"], x["name"]))

    except Exception as err:
        logger.warning("Error enumerating network interfaces: %s", err)

    return interfaces


def filter_primary_interfaces(interfaces: List[Dict[str, Any]], max_count: int = 4) -> List[Dict[str, Any]]:
    """Filter out low-level virtual WFP filter drivers, QoS schedulers, and tunnel adapters,
    returning the top 3-4 most relevant physical and loopback network adapters for UI selection.

    Args:
        interfaces: Complete list of interfaces discovered by Scapy/Npcap.
        max_count: Maximum number of primary interfaces to return (default: 4).

    Returns:
        List[Dict[str, Any]]: Cleaned and prioritized interface list (3-4 options).
    """
    if not interfaces or len(interfaces) <= max_count:
        return interfaces

    virtual_filter_keywords = [
        "wfp",
        "qos",
        "filter driver",
        "lightweight",
        "kernel debug",
        "miniport",
        "teredo",
        "6to4",
        "ip-https",
        "wi-fi direct",
        "wifi direct",
        "packet scheduler",
        "network monitor",
        "native wifi",
        "virtual wifi",
    ]

    primary: List[Dict[str, Any]] = []
    secondary: List[Dict[str, Any]] = []

    for iface in interfaces:
        name = str(iface.get("name", "")).lower()
        desc = str(iface.get("description", "")).lower()
        full = f"{name} {desc}"

        if "loopback" in full or iface.get("is_loopback"):
            primary.append(iface)
        elif any(k in full for k in virtual_filter_keywords):
            secondary.append(iface)
        else:
            primary.append(iface)

    def interface_priority(item: Dict[str, Any]) -> Tuple[int, str]:
        name_desc = (str(item.get("name", "")) + " " + str(item.get("description", ""))).lower()
        has_routable_ip = bool(
            item.get("has_ip")
            and not str(item.get("ip", "")).startswith("169.254")
            and not str(item.get("ip", "")).startswith("127.")
        )
        if has_routable_ip:
            return (0, str(item.get("name", "")))
        if "ethernet" in name_desc and "loopback" not in name_desc:
            return (1, str(item.get("name", "")))
        if "wi-fi" in name_desc or "wireless" in name_desc:
            return (2, str(item.get("name", "")))
        if "loopback" in name_desc:
            return (3, str(item.get("name", "")))
        return (4, str(item.get("name", "")))

    primary.sort(key=interface_priority)

    # Backfill if primary list is shorter than max_count
    if len(primary) < max_count:
        for s in secondary:
            if s not in primary:
                primary.append(s)
            if len(primary) >= max_count:
                break

    return primary[:max_count]


def resolve_interface_name(name_or_query: str) -> str:
    """Resolve an adapter name, description, friendly label, or GUID to a Scapy interface.

    Args:
        name_or_query: Interface identifier or descriptive query.

    Returns:
        str: Validated Scapy interface name or device string.
    """
    if not name_or_query:
        return name_or_query

    query = name_or_query.strip()

    try:
        if hasattr(conf, "ifaces") and conf.ifaces:
            # 1. Exact match against iface.name
            for iface in conf.ifaces.values():
                if query == getattr(iface, "name", ""):
                    return query

            # 2. Exact match against network_name / pcap_name (GUID)
            for iface in conf.ifaces.values():
                guid = getattr(iface, "network_name", getattr(iface, "pcap_name", ""))
                if query == guid:
                    return getattr(iface, "name", query)

            # 3. Match against description
            for iface in conf.ifaces.values():
                desc = getattr(iface, "description", "")
                if query.lower() == desc.lower():
                    return getattr(iface, "name", query)

            # 4. Partial / Substring match (e.g. 'loopback' -> 'Loopback Pseudo-Interface 1')
            for iface in conf.ifaces.values():
                name = getattr(iface, "name", "")
                desc = getattr(iface, "description", "")
                if query.lower() in name.lower() or query.lower() in desc.lower():
                    return name

    except Exception:
        pass

    return query


class LiveCaptureWorker:
    """Thread-safe background worker that coordinates live packet capture.

    Decouples raw packet sniffing (via Scapy AsyncSniffer) from SQLite insertion
    and heuristic detection rules using an internal thread-safe queue.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        packet_limit: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        batch_size: int = 50,
        detectors: Optional[List[Any]] = None,
        enable_hybrid: bool = True,
        prevention_engine: Optional[Any] = None,
    ):
        self.interface = interface
        self.db_path = Path(db_path)
        self.packet_limit = packet_limit
        self.duration_seconds = duration_seconds
        self.batch_size = max(1, batch_size)
        self.custom_detectors = detectors
        self.enable_hybrid = enable_hybrid
        self.prevention_engine = prevention_engine

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sniffer: Optional[AsyncSniffer] = None
        self._queue: queue.Queue = queue.Queue(maxsize=10000)

        self._status = "IDLE"  # IDLE, RUNNING, STOPPED, COMPLETED, ERROR
        self._is_running = False
        self._packets_captured = 0
        self._packets_persisted = 0
        self._packets_dropped = 0
        self._alerts_generated = 0
        self._error_message: Optional[str] = None
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None

    @property
    def is_running(self) -> bool:
        """Return True if the background capture worker is currently active."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            return self._is_running

    def configure(
        self,
        interface: str,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        packet_limit: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        batch_size: int = 50,
        detectors: Optional[List[Any]] = None,
        enable_hybrid: bool = True,
        prevention_engine: Optional[Any] = None,
    ) -> None:
        """Update worker parameters when the worker is not currently running."""
        with self._lock:
            if (self._thread is not None and self._thread.is_alive()) or self._is_running:
                raise RuntimeError("Cannot reconfigure an active capture worker. Stop it first.")
            self.interface = interface
            self.db_path = Path(db_path)
            self.packet_limit = packet_limit
            self.duration_seconds = duration_seconds
            self.batch_size = max(1, batch_size)
            self.custom_detectors = detectors
            self.enable_hybrid = enable_hybrid
            self.prevention_engine = prevention_engine
            self._status = "IDLE"
            self._error_message = None

    def start(self) -> bool:
        """Start the live capture worker thread.

        Returns:
            bool: True if capture started successfully.

        Raises:
            RuntimeError: If a capture worker is already actively running or interface is unset.
        """
        with self._lock:
            if (self._thread is not None and self._thread.is_alive()) or self._is_running:
                raise RuntimeError("A live capture worker is already actively running.")
            if not self.interface:
                raise ValueError("No network interface specified for live capture.")

            self._stop_event.clear()
            self._queue = queue.Queue(maxsize=10000)
            self._packets_captured = 0
            self._packets_persisted = 0
            self._packets_dropped = 0
            self._alerts_generated = 0
            self._error_message = None
            self._started_at = time.time()
            self._stopped_at = None
            self._status = "RUNNING"
            self._is_running = True

            self._thread = threading.Thread(
                target=self._run_worker,
                name="MiniIDS-LiveCaptureWorker",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the capture worker to terminate and wait for resources to close."""
        self._stop_event.set()

        with self._lock:
            sniffer = self._sniffer

        if sniffer is not None:
            try:
                sniffer.stop()
            except Exception as ex:
                logger.debug("Error while stopping AsyncSniffer: %s", ex)

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        with self._lock:
            if thread is not None and thread.is_alive():
                # Worker thread has not finished within the timeout window
                self._is_running = True
                if self._status == "RUNNING":
                    self._status = "STOPPING"
            else:
                self._is_running = False
                if self._status in ("RUNNING", "STOPPING"):
                    self._status = "STOPPED"
                    if self._stopped_at is None:
                        self._stopped_at = time.time()

    def get_status(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot of the worker state."""
        with self._lock:
            # Synchronize state if thread has exited asynchronously
            if self._thread is not None and not self._thread.is_alive():
                self._is_running = False
                if self._status in ("RUNNING", "STOPPING"):
                    self._status = "STOPPED"
                    if self._stopped_at is None:
                        self._stopped_at = time.time()
            elif self._thread is not None and self._thread.is_alive():
                self._is_running = True

            elapsed = 0.0
            if self._started_at:
                end = self._stopped_at if self._stopped_at else time.time()
                elapsed = max(0.0, end - self._started_at)

            return {
                "status": self._status,
                "is_running": self._is_running,
                "interface": self.interface,
                "packets_captured": self._packets_captured,
                "packets_persisted": self._packets_persisted,
                "packets_dropped": self._packets_dropped,
                "alerts_generated": self._alerts_generated,
                "packet_limit": self.packet_limit,
                "duration_seconds": self.duration_seconds,
                "elapsed_seconds": elapsed,
                "error_message": self._error_message,
                "db_path": str(self.db_path),
            }

    def _run_worker(self) -> None:
        """Internal worker loop executing on the dedicated background thread."""
        from storage.database import DatabaseManager

        db: Optional[DatabaseManager] = None
        try:
            # 1. Initialize dedicated database connection for this thread
            db = DatabaseManager(self.db_path)
            db.init_db()

            # 2. Instantiate detection rules or hybrid detection engine
            if self.custom_detectors is not None:
                active_detectors = self.custom_detectors
            elif self.enable_hybrid:
                try:
                    from detection.hybrid_detection import HybridDetectionEngine
                    from ml.model_registry import ModelRegistry
                    from prevention.engine import PreventionEngine
                    from prevention.policy_engine import PreventionMode

                    pe = self.prevention_engine or PreventionEngine(db_manager=db, initial_mode=PreventionMode.SIMULATION)
                    registry = ModelRegistry(db_path=self.db_path)
                    sup_bundle = registry.load_active_bundle("supervised_binary") or registry.load_active_bundle()
                    anom_bundle = registry.load_active_bundle("anomaly_isolation_forest")
                    sup_m, sup_p, sup_meta = sup_bundle if sup_bundle else (None, None, None)
                    anom_m, anom_p, anom_meta = anom_bundle if anom_bundle else (None, None, None)

                    active_detectors = [
                        HybridDetectionEngine(
                            supervised_model=sup_m,
                            supervised_preprocessor=sup_p,
                            supervised_meta=sup_meta,
                            anomaly_model=anom_m,
                            anomaly_preprocessor=anom_p,
                            anomaly_meta=anom_meta,
                            prevention_engine=pe,
                            source_tag="live",
                        )
                    ]
                except Exception as err:
                    logger.warning("Could not initialize hybrid detection in live worker: %s. Using rules.", err)
                    from detection.port_scan import PortScanDetector
                    from detection.syn_rate import SynRateDetector
                    from detection.traffic_anomaly import TrafficAnomalyDetector
                    active_detectors = [
                        PortScanDetector(),
                        SynRateDetector(),
                        TrafficAnomalyDetector(),
                    ]
            else:
                from detection.port_scan import PortScanDetector
                from detection.syn_rate import SynRateDetector
                from detection.traffic_anomaly import TrafficAnomalyDetector
                active_detectors = [
                    PortScanDetector(),
                    SynRateDetector(),
                    TrafficAnomalyDetector(),
                ]

            # 3. Define raw packet callback for Scapy AsyncSniffer
            def _packet_callback(raw_pkt: Any) -> None:
                if self._stop_event.is_set():
                    return
                norm = normalize_packet(raw_pkt)
                if norm is not None:
                    try:
                        self._queue.put(norm, block=False)
                        with self._lock:
                            self._packets_captured += 1
                            if (
                                self.packet_limit is not None
                                and self._packets_captured >= self.packet_limit
                            ):
                                self._stop_event.set()
                    except queue.Full:
                        with self._lock:
                            self._packets_dropped += 1
                        logger.warning("Packet queue full; dropping packet to preserve memory.")

            # 4. Resolve interface name and start Scapy AsyncSniffer
            resolved_iface = resolve_interface_name(self.interface)
            try:
                self._sniffer = AsyncSniffer(
                    iface=resolved_iface,
                    prn=_packet_callback,
                    store=False,
                )
                self._sniffer.start()
            except Exception as sniff_err:
                with self._lock:
                    self._error_message = (
                        f"Failed to start sniffer on '{self.interface}' "
                        f"(resolved: '{resolved_iface}'): {sniff_err}"
                    )
                    self._status = "ERROR"
                    self._is_running = False
                    self._stopped_at = time.time()
                return

            # 5. Consumer processing loop: drain queue, run detectors, batch insert to SQLite
            batch: List[NormalizedPacket] = []
            while not self._stop_event.is_set():
                # Check duration limit
                if self.duration_seconds is not None:
                    elapsed = time.time() - (self._started_at or time.time())
                    if elapsed >= self.duration_seconds:
                        self._stop_event.set()
                        break

                try:
                    pkt = self._queue.get(timeout=0.25)
                    batch.append(pkt)

                    # Feed packet to each detection rule or hybrid engine
                    for detector in active_detectors:
                        if hasattr(detector, "process_packet_all"):
                            det_alerts, flows = detector.process_packet_all(pkt)
                            if flows:
                                db.insert_flows(flows)
                        else:
                            alt = detector.process_packet(pkt)
                            det_alerts = [alt] if alt is not None else []

                        for alert in det_alerts:
                            db.insert_alert(alert)
                            with self._lock:
                                self._alerts_generated += 1

                    # Periodic batch flush to database
                    if len(batch) >= self.batch_size:
                        db.insert_packets(batch)
                        with self._lock:
                            self._packets_persisted += len(batch)
                        batch.clear()

                except queue.Empty:
                    # Flush pending batch on idle timeout
                    if batch:
                        db.insert_packets(batch)
                        with self._lock:
                            self._packets_persisted += len(batch)
                        batch.clear()

            # 6. Stop sniffer and drain remaining queued packets
            if self._sniffer is not None:
                try:
                    self._sniffer.stop()
                except Exception:
                    pass

            # Drain any remaining packets in the queue
            while not self._queue.empty():
                try:
                    pkt = self._queue.get_nowait()
                    batch.append(pkt)
                    for detector in active_detectors:
                        if hasattr(detector, "process_packet_all"):
                            det_alerts, flows = detector.process_packet_all(pkt)
                            if flows:
                                db.insert_flows(flows)
                        else:
                            alt = detector.process_packet(pkt)
                            det_alerts = [alt] if alt is not None else []

                        for alert in det_alerts:
                            db.insert_alert(alert)
                            with self._lock:
                                self._alerts_generated += 1
                except queue.Empty:
                    break

            # Flush remaining flows from detectors
            for detector in active_detectors:
                if hasattr(detector, "flush"):
                    flush_alerts, flush_flows = detector.flush()
                    if flush_flows:
                        db.insert_flows(flush_flows)
                    for alert in flush_alerts:
                        db.insert_alert(alert)
                        with self._lock:
                            self._alerts_generated += 1

            if batch:
                db.insert_packets(batch)
                with self._lock:
                    self._packets_persisted += len(batch)
                batch.clear()

            with self._lock:
                if self._status != "ERROR":
                    if self.packet_limit and self._packets_captured >= self.packet_limit:
                        self._status = "COMPLETED"
                    elif self.duration_seconds and (time.time() - (self._started_at or 0)) >= self.duration_seconds:
                        self._status = "COMPLETED"
                    else:
                        self._status = "STOPPED"
                self._stopped_at = time.time()
                self._is_running = False

        except Exception as err:
            logger.exception("Unexpected error in live capture worker: %s", err)
            with self._lock:
                self._error_message = f"Live capture worker failure: {err}"
                self._status = "ERROR"
                self._is_running = False
                self._stopped_at = time.time()

        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass


# Global singleton instance for cross-session Streamlit state
_global_capture_worker: Optional[LiveCaptureWorker] = None
_global_worker_lock = threading.Lock()


def get_live_worker() -> LiveCaptureWorker:
    """Return the global singleton LiveCaptureWorker instance."""
    global _global_capture_worker
    with _global_worker_lock:
        if _global_capture_worker is None:
            _global_capture_worker = LiveCaptureWorker()
        return _global_capture_worker
