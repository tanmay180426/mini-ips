"""SQLite Database Management Module for Mini-IDS.

Handles initialization of database schemas, inserting normalized network
packets and detection alerts, and querying stored records for reporting
and dashboard visualization.

Database Tables:
1. `packets`: Stores normalized packet header metadata.
2. `alerts`: Stores rule-based detection alerts.

Design Principles:
- Uses Python's built-in `sqlite3` library (no external ORM dependencies).
- Fully parameterized SQL queries to prevent injection.
- Explicit transaction handling and clean resource closing.
- Configurable database path with default at `data/mini_ids.db`.
"""

import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Union

from capture.packet_capture import NormalizedPacket
from detection import Alert

logger = logging.getLogger(__name__)

# Default location for persistent project storage
DEFAULT_DB_PATH = Path("data") / "mini_ids.db"


class DatabaseManager:
    """Manages SQLite storage for normalized packets and security alerts.

    Attributes:
        db_path (Path): Path to the SQLite database file.
    """

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None

    def get_connection(self) -> sqlite3.Connection:
        """Get or establish an active connection to the SQLite database.

        Returns:
            sqlite3.Connection: Active database connection with Row factory enabled.
        """
        if self._connection is None:
            # Ensure parent directories exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
            # Enable Foreign Keys
            self._connection.execute("PRAGMA foreign_keys = ON;")
            self._connection.execute("PRAGMA busy_timeout = 5000;")
            # Enable Write-Ahead Logging (WAL) for concurrent read/write support
            try:
                cursor = self._connection.execute("PRAGMA journal_mode = WAL;")
                mode = cursor.fetchone()[0]
                if str(mode).lower() != "wal" and str(self.db_path) != ":memory:":
                    logger.warning("Requested WAL mode, but active journal_mode is '%s'", mode)
            except sqlite3.Error as err:
                logger.error("Failed to enable WAL journal mode: %s", err)
                raise
            self._run_migrations(self._connection)
        return self._connection

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run idempotent additive migrations for existing tables."""
        try:
            c = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts';")
            if c.fetchone():
                alert_cols = {row["name"] for row in conn.execute("PRAGMA table_info(alerts);").fetchall()}
                if "detection_method" not in alert_cols:
                    conn.execute("ALTER TABLE alerts ADD COLUMN detection_method TEXT DEFAULT 'HEURISTIC';")
                if "model_version" not in alert_cols:
                    conn.execute("ALTER TABLE alerts ADD COLUMN model_version TEXT;")
                if "confidence" not in alert_cols:
                    conn.execute("ALTER TABLE alerts ADD COLUMN confidence REAL;")
                if "anomaly_score" not in alert_cols:
                    conn.execute("ALTER TABLE alerts ADD COLUMN anomaly_score REAL;")
                if "threat_score" not in alert_cols:
                    conn.execute("ALTER TABLE alerts ADD COLUMN threat_score REAL;")
                if "prevention_action_id" not in alert_cols:
                    conn.execute("ALTER TABLE alerts ADD COLUMN prevention_action_id TEXT;")

            # Ensure prevention and audit tables exist
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prevention_actions (
                    action_id TEXT PRIMARY KEY,
                    alert_id TEXT,
                    timestamp REAL NOT NULL,
                    target_ip TEXT NOT NULL,
                    target_port INTEGER,
                    protocol TEXT,
                    action_type TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    threat_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL DEFAULT 300,
                    expires_at REAL,
                    approved_by TEXT,
                    executed_at REAL,
                    verified_at REAL,
                    rolled_back_at REAL,
                    details_json TEXT,
                    error_message TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    operator TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    details_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details_json TEXT
                );
                """
            )
        except Exception as e:
            logger.debug("Migration check skipped: %s", e)

    def init_db(self) -> None:
        """Create the database tables and indexes if they do not already exist.

        Safe to call multiple times (idempotent).
        """
        conn = self.get_connection()
        with conn:
            # 1. Packets Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS packets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    src_ip TEXT NOT NULL,
                    dst_ip TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    length INTEGER NOT NULL,
                    src_port INTEGER,
                    dst_port INTEGER,
                    tcp_flags TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_packets_timestamp
                ON packets(timestamp);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_packets_protocol
                ON packets(protocol);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_packets_src_dst
                ON packets(src_ip, dst_ip);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_packets_dst_port
                ON packets(dst_port);
                """
            )

            # 2. Alerts Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    rule_name TEXT NOT NULL,
                    src_ip TEXT,
                    dst_ip TEXT,
                    observed_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    window_seconds REAL NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
                ON alerts(timestamp);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_rule
                ON alerts(rule_name);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_severity
                ON alerts(severity);
                """
            )

            # 3. Additive Column Migrations for Existing Databases
            alert_cols = {row["name"] for row in conn.execute("PRAGMA table_info(alerts);").fetchall()}
            if "detection_method" not in alert_cols:
                conn.execute("ALTER TABLE alerts ADD COLUMN detection_method TEXT DEFAULT 'HEURISTIC';")
            if "model_version" not in alert_cols:
                conn.execute("ALTER TABLE alerts ADD COLUMN model_version TEXT;")
            if "confidence" not in alert_cols:
                conn.execute("ALTER TABLE alerts ADD COLUMN confidence REAL;")
            if "anomaly_score" not in alert_cols:
                conn.execute("ALTER TABLE alerts ADD COLUMN anomaly_score REAL;")
            if "threat_score" not in alert_cols:
                conn.execute("ALTER TABLE alerts ADD COLUMN threat_score REAL;")
            if "prevention_action_id" not in alert_cols:
                conn.execute("ALTER TABLE alerts ADD COLUMN prevention_action_id TEXT;")

            # 4. Models Table (Trained ML Artifact Registry)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL UNIQUE,
                    model_type TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    dataset_name TEXT,
                    task_type TEXT,
                    metrics_json TEXT,
                    feature_schema_json TEXT,
                    artifact_path TEXT,
                    created_at REAL NOT NULL,
                    is_active INTEGER DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_models_version
                ON models(version);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_models_active
                ON models(is_active);
                """
            )

            # 5. Flows Table (Network Flow Summaries)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    src_ip TEXT NOT NULL,
                    dst_ip TEXT NOT NULL,
                    src_port INTEGER,
                    dst_port INTEGER,
                    protocol TEXT NOT NULL,
                    packet_count INTEGER NOT NULL,
                    byte_count INTEGER NOT NULL,
                    fwd_packets INTEGER NOT NULL,
                    bwd_packets INTEGER NOT NULL,
                    fwd_bytes INTEGER NOT NULL,
                    bwd_bytes INTEGER NOT NULL,
                    flow_pkts_s REAL NOT NULL,
                    flow_bytes_s REAL NOT NULL,
                    features_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_flows_time
                ON flows(start_time, end_time);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_flows_src_dst
                ON flows(src_ip, dst_ip);
                """
            )

            # 6. Prevention Actions Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prevention_actions (
                    action_id TEXT PRIMARY KEY,
                    alert_id TEXT,
                    timestamp REAL NOT NULL,
                    target_ip TEXT NOT NULL,
                    target_port INTEGER,
                    protocol TEXT,
                    action_type TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    threat_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL DEFAULT 300,
                    expires_at REAL,
                    approved_by TEXT,
                    executed_at REAL,
                    verified_at REAL,
                    rolled_back_at REAL,
                    details_json TEXT,
                    error_message TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prevention_time
                ON prevention_actions(timestamp);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prevention_target
                ON prevention_actions(target_ip);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prevention_status
                ON prevention_actions(status);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prevention_mode
                ON prevention_actions(mode);
                """
            )

            # 7. Approval Events Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    operator TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    details_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_approval_action_id
                ON approval_events(action_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_approval_timestamp
                ON approval_events(timestamp);
                """
            )

            # 8. Audit Log Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_time
                ON audit_log(timestamp);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_actor
                ON audit_log(actor);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_log(action);
                """
            )

    def insert_packet(self, packet: NormalizedPacket) -> int:
        """Insert a single normalized packet into the database.

        Args:
            packet: A NormalizedPacket instance.

        Returns:
            int: The primary key ID of the inserted record.

        Raises:
            ValueError: If packet is invalid.
        """
        if not isinstance(packet, NormalizedPacket):
            raise ValueError(f"Expected NormalizedPacket instance, got {type(packet)}")

        conn = self.get_connection()
        query = """
            INSERT INTO packets (
                timestamp, src_ip, dst_ip, protocol, length,
                src_port, dst_port, tcp_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            float(packet.timestamp),
            str(packet.src_ip),
            str(packet.dst_ip),
            str(packet.protocol),
            int(packet.length),
            packet.src_port,
            packet.dst_port,
            packet.tcp_flags,
        )
        with conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid

    def insert_packets(self, packets: Iterable[NormalizedPacket]) -> int:
        """Insert multiple normalized packets within a single transaction.

        Args:
            packets: Iterable of NormalizedPacket instances.

        Returns:
            int: Total count of packets inserted.
        """
        conn = self.get_connection()
        query = """
            INSERT INTO packets (
                timestamp, src_ip, dst_ip, protocol, length,
                src_port, dst_port, tcp_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        param_list = [
            (
                float(p.timestamp),
                str(p.src_ip),
                str(p.dst_ip),
                str(p.protocol),
                int(p.length),
                p.src_port,
                p.dst_port,
                p.tcp_flags,
            )
            for p in packets
        ]
        if not param_list:
            return 0

        with conn:
            conn.executemany(query, param_list)
        return len(param_list)

    def insert_alert(self, alert: Alert) -> int:
        """Insert a single detection alert into the database.

        Args:
            alert: An Alert instance.

        Returns:
            int: The primary key ID of the inserted record.

        Raises:
            ValueError: If alert is invalid.
        """
        if not isinstance(alert, Alert):
            raise ValueError(f"Expected Alert instance, got {type(alert)}")

        conn = self.get_connection()
        query = """
            INSERT INTO alerts (
                timestamp, rule_name, src_ip, dst_ip,
                observed_value, threshold, window_seconds,
                severity, message, detection_method,
                model_version, confidence, anomaly_score,
                threat_score, prevention_action_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            float(alert.timestamp),
            str(alert.rule_name),
            alert.src_ip,
            alert.dst_ip,
            float(alert.observed_value),
            float(alert.threshold),
            float(alert.window_seconds),
            str(alert.severity),
            str(alert.message),
            getattr(alert, "detection_method", "HEURISTIC") or "HEURISTIC",
            getattr(alert, "model_version", None),
            getattr(alert, "confidence", None),
            getattr(alert, "anomaly_score", None),
            getattr(alert, "threat_score", None),
            getattr(alert, "prevention_action_id", None),
        )
        with conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid

    def insert_alerts(self, alerts: Iterable[Alert]) -> int:
        """Insert multiple detection alerts within a single transaction.

        Args:
            alerts: Iterable of Alert instances.

        Returns:
            int: Total count of alerts inserted.
        """
        conn = self.get_connection()
        query = """
            INSERT INTO alerts (
                timestamp, rule_name, src_ip, dst_ip,
                observed_value, threshold, window_seconds,
                severity, message, detection_method,
                model_version, confidence, anomaly_score,
                threat_score, prevention_action_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        param_list = [
            (
                float(a.timestamp),
                str(a.rule_name),
                a.src_ip,
                a.dst_ip,
                float(a.observed_value),
                float(a.threshold),
                float(a.window_seconds),
                str(a.severity),
                str(a.message),
                getattr(a, "detection_method", "HEURISTIC") or "HEURISTIC",
                getattr(a, "model_version", None),
                getattr(a, "confidence", None),
                getattr(a, "anomaly_score", None),
                getattr(a, "threat_score", None),
                getattr(a, "prevention_action_id", None),
            )
            for a in alerts
        ]
        if not param_list:
            return 0

        with conn:
            conn.executemany(query, param_list)
        return len(param_list)

    def get_packets(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "ASC",
    ) -> List[Dict[str, Any]]:
        """Retrieve stored packets in chronological order.

        Args:
            limit: Optional maximum number of records to return.
            offset: Number of initial records to skip.
            order: Sort direction by timestamp ('ASC' or 'DESC'). Default: 'ASC'.

        Returns:
            List[Dict[str, Any]]: List of dictionary representations of packets.
        """
        clean_order = "DESC" if order.upper() == "DESC" else "ASC"
        conn = self.get_connection()

        if limit is not None:
            query = f"""
                SELECT id, timestamp, src_ip, dst_ip, protocol, length,
                       src_port, dst_port, tcp_flags
                FROM packets
                ORDER BY timestamp {clean_order}, id {clean_order}
                LIMIT ? OFFSET ?;
            """
            cursor = conn.execute(query, (int(limit), int(offset)))
        else:
            query = f"""
                SELECT id, timestamp, src_ip, dst_ip, protocol, length,
                       src_port, dst_port, tcp_flags
                FROM packets
                ORDER BY timestamp {clean_order}, id {clean_order};
            """
            cursor = conn.execute(query)

        return [dict(row) for row in cursor.fetchall()]

    def get_alerts(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "ASC",
    ) -> List[Dict[str, Any]]:
        """Retrieve stored alerts in chronological order.

        Args:
            limit: Optional maximum number of records to return.
            offset: Number of initial records to skip.
            order: Sort direction by timestamp ('ASC' or 'DESC'). Default: 'ASC'.

        Returns:
            List[Dict[str, Any]]: List of dictionary representations of alerts.
        """
        clean_order = "DESC" if order.upper() == "DESC" else "ASC"
        conn = self.get_connection()

        if limit is not None:
            query = f"""
                SELECT id, timestamp, rule_name, src_ip, dst_ip,
                       observed_value, threshold, window_seconds,
                       severity, message, detection_method,
                       model_version, confidence, anomaly_score,
                       threat_score, prevention_action_id
                FROM alerts
                ORDER BY timestamp {clean_order}, id {clean_order}
                LIMIT ? OFFSET ?;
            """
            cursor = conn.execute(query, (int(limit), int(offset)))
        else:
            query = f"""
                SELECT id, timestamp, rule_name, src_ip, dst_ip,
                       observed_value, threshold, window_seconds,
                       severity, message, detection_method,
                       model_version, confidence, anomaly_score,
                       threat_score, prevention_action_id
                FROM alerts
                ORDER BY timestamp {clean_order}, id {clean_order};
            """
            cursor = conn.execute(query)

        return [dict(row) for row in cursor.fetchall()]

    def get_alert_by_id(self, alert_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single alert record by its primary key ID.

        Args:
            alert_id: Primary key ID of the alert.

        Returns:
            Optional[Dict[str, Any]]: Alert row as a dictionary, or None if not found.
        """
        conn = self.get_connection()
        query = """
            SELECT id, timestamp, rule_name, src_ip, dst_ip,
                   observed_value, threshold, window_seconds,
                   severity, message, detection_method,
                   model_version, confidence, anomaly_score,
                   threat_score, prevention_action_id
            FROM alerts
            WHERE id = ?;
        """
        cursor = conn.execute(query, (int(alert_id),))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_alert_investigation_states(
        self, alert_ids: Optional[Iterable[int]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Return latest triage state derived from append-only audit events.

        This uses the existing audit table, so investigation workflow support
        does not require a production schema migration.
        """
        import json
        conn = self.get_connection()
        ids = [int(value) for value in alert_ids] if alert_ids is not None else None
        if ids == []:
            return {}
        query = "SELECT log_id, timestamp, actor, target, result, details_json FROM audit_log WHERE action = 'UPDATE_ALERT_INVESTIGATION'"
        params: List[Any] = []
        if ids is not None:
            query += f" AND target IN ({','.join('?' for _ in ids)})"
            params.extend(str(value) for value in ids)
        query += " ORDER BY log_id ASC"
        states: Dict[int, Dict[str, Any]] = {}
        for row in conn.execute(query, tuple(params)).fetchall():
            try:
                details = json.loads(row["details_json"] or "{}")
                alert_id = int(details.get("alert_id", row["target"]))
                states[alert_id] = {
                    "alert_id": alert_id,
                    "status": str(details.get("status", row["result"])),
                    "notes": str(details.get("notes", "")),
                    "updated_at": float(row["timestamp"]),
                    "actor": str(row["actor"]),
                    "audit_log_id": int(row["log_id"]),
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return states

    def get_alert_investigation_state(self, alert_id: int) -> Dict[str, Any]:
        """Return one alert's current investigation state, defaulting to OPEN."""
        clean_id = int(alert_id)
        state = self.get_alert_investigation_states([clean_id]).get(clean_id)
        return state or {"alert_id": clean_id, "status": "OPEN", "notes": "", "updated_at": None, "actor": None}

    def update_alert_investigation(
        self, alert_id: int, status: str, notes: Optional[str] = None,
        actor: str = "ANALYST",
    ) -> bool:
        """Append a validated triage/note transition for an existing alert."""
        import time
        clean_id = int(alert_id)
        clean_status = str(status).strip().upper()
        if clean_status not in {"OPEN", "TRIAGED", "RESOLVED"}:
            raise ValueError("Investigation status must be OPEN, TRIAGED, or RESOLVED")
        clean_notes = str(notes or "")
        if len(clean_notes) > 10000:
            raise ValueError("Investigation notes cannot exceed 10,000 characters")
        alert = self.get_alert_by_id(clean_id)
        if alert is None:
            return False
        previous = self.get_alert_investigation_state(clean_id)
        self.insert_audit_event(
            actor=str(actor).strip() or "ANALYST",
            action="UPDATE_ALERT_INVESTIGATION",
            target=str(clean_id),
            mode="INVESTIGATION",
            result=clean_status,
            details={
                "alert_id": clean_id,
                "status": clean_status,
                "previous_status": previous["status"],
                "notes": clean_notes,
                "reason": "Analyst investigation update",
            },
            timestamp=time.time(),
        )
        return True

    def get_related_packets(
        self,
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        protocol: Optional[str] = None,
        limit: Optional[int] = 100,
        offset: int = 0,
        order: str = "ASC",
    ) -> List[Dict[str, Any]]:
        """Retrieve packet records matching given IP endpoints, time range, and protocol.

        Uses fully parameterized SQL queries to prevent injection.

        Args:
            src_ip: Optional source IP to match.
            dst_ip: Optional destination IP to match.
            start_time: Optional inclusive lower timestamp bound (epoch seconds).
            end_time: Optional inclusive upper timestamp bound (epoch seconds).
            protocol: Optional protocol string (e.g. 'TCP', 'UDP').
            limit: Maximum number of records to return. Default: 100.
            offset: Number of records to skip. Default: 0.
            order: Sort direction by timestamp ('ASC' or 'DESC'). Default: 'ASC'.

        Returns:
            List[Dict[str, Any]]: List of matching packet dictionaries.
        """
        conditions = []
        params: List[Any] = []

        if src_ip is not None:
            conditions.append("src_ip = ?")
            params.append(str(src_ip))

        if dst_ip is not None:
            conditions.append("dst_ip = ?")
            params.append(str(dst_ip))

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))

        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))

        if protocol is not None and str(protocol).upper() != "ALL":
            conditions.append("UPPER(protocol) = UPPER(?)")
            params.append(str(protocol))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        clean_order = "DESC" if str(order).upper() == "DESC" else "ASC"

        query = f"""
            SELECT id, timestamp, src_ip, dst_ip, protocol, length,
                   src_port, dst_port, tcp_flags
            FROM packets
            {where_clause}
            ORDER BY timestamp {clean_order}, id {clean_order}
        """

        if limit is not None:
            query += " LIMIT ? OFFSET ?;"
            params.extend([int(limit), int(offset)])
        else:
            query += ";"

        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_packet_count(self) -> int:
        """Return the total count of stored packets."""
        conn = self.get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM packets;")
        return cursor.fetchone()[0]

    def get_alert_count(self) -> int:
        """Return the total count of stored alerts."""
        conn = self.get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM alerts;")
        return cursor.fetchone()[0]

    def get_timestamp_bounds(self) -> Dict[str, Optional[float]]:
        """Return minimum and maximum timestamps across stored packets and alerts.

        Returns:
            Dict[str, Optional[float]]: Dict with 'min_timestamp' and 'max_timestamp'.
        """
        conn = self.get_connection()
        cursor = conn.execute("""
            SELECT MIN(min_ts), MAX(max_ts)
            FROM (
                SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts FROM packets
                UNION ALL
                SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts FROM alerts
            )
        """)
        row = cursor.fetchone()
        if row and (row[0] is not None or row[1] is not None):
            return {"min_timestamp": row[0], "max_timestamp": row[1]}
        return {"min_timestamp": None, "max_timestamp": None}

    def get_security_kpis(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, int]:
        """Calculate high-level security key performance indicators (KPIs).

        Args:
            start_time: Optional lower timestamp bound (epoch seconds).
            end_time: Optional upper timestamp bound (epoch seconds).

        Returns:
            Dict[str, int]: Dictionary containing:
                - 'total_packets': Total packet count.
                - 'total_alerts': Total alert count.
                - 'high_severity_alerts': Alerts with severity HIGH or CRITICAL.
                - 'distinct_alert_sources': Distinct non-null source IPs in alerts.
        """
        conn = self.get_connection()

        # 1. Packet Count
        pkt_conds = []
        pkt_params: List[Any] = []
        if start_time is not None:
            pkt_conds.append("timestamp >= ?")
            pkt_params.append(float(start_time))
        if end_time is not None:
            pkt_conds.append("timestamp <= ?")
            pkt_params.append(float(end_time))

        pkt_where = f"WHERE {' AND '.join(pkt_conds)}" if pkt_conds else ""
        pkt_query = f"SELECT COUNT(*) FROM packets {pkt_where};"
        total_packets = conn.execute(pkt_query, tuple(pkt_params)).fetchone()[0]

        # 2. Alert Metrics
        alert_conds = []
        alert_params: List[Any] = []
        if start_time is not None:
            alert_conds.append("timestamp >= ?")
            alert_params.append(float(start_time))
        if end_time is not None:
            alert_conds.append("timestamp <= ?")
            alert_params.append(float(end_time))

        alert_where = f"WHERE {' AND '.join(alert_conds)}" if alert_conds else ""
        alert_query = f"""
            SELECT
                COUNT(*) AS total_alerts,
                COALESCE(SUM(CASE WHEN UPPER(severity) IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END), 0) AS high_severity_alerts,
                COUNT(DISTINCT CASE WHEN src_ip IS NOT NULL AND TRIM(src_ip) != '' THEN src_ip END) AS distinct_alert_sources
            FROM alerts
            {alert_where};
        """
        alert_row = conn.execute(alert_query, tuple(alert_params)).fetchone()

        return {
            "total_packets": int(total_packets or 0),
            "total_alerts": int(alert_row["total_alerts"] or 0),
            "high_severity_alerts": int(alert_row["high_severity_alerts"] or 0),
            "distinct_alert_sources": int(alert_row["distinct_alert_sources"] or 0),
        }

    def get_alert_severity_distribution(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        rule: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate alert counts grouped by severity tier.

        Args:
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            rule: Optional detection rule filter.

        Returns:
            List[Dict[str, Any]]: List of dicts with 'severity' and 'count'.
        """
        conditions = []
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if rule is not None and rule.upper() != "ALL":
            conditions.append("rule_name = ?")
            params.append(str(rule))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT severity, COUNT(*) AS count
            FROM alerts
            {where_clause}
            GROUP BY severity
            ORDER BY count DESC;
        """
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_alerts_by_rule(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate alert counts grouped by detection rule name.

        Args:
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            severity: Optional severity filter.

        Returns:
            List[Dict[str, Any]]: List of dicts with 'rule_name' and 'count'.
        """
        conditions = []
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if severity is not None and severity.upper() != "ALL":
            conditions.append("severity = ?")
            params.append(str(severity))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT rule_name, COUNT(*) AS count
            FROM alerts
            {where_clause}
            GROUP BY rule_name
            ORDER BY count DESC;
        """
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_alert_timeline(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        rule: Optional[str] = None,
        severity: Optional[str] = None,
        bucket_resolution: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Aggregate alert counts across time buckets in UTC.

        Args:
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            rule: Optional rule filter.
            severity: Optional severity filter.
            bucket_resolution: Bucket granularity ('auto', 'day', 'hour', 'minute', 'second').

        Returns:
            List[Dict[str, Any]]: Chronologically ordered list of dicts with 'bucket' and 'count'.
        """
        conn = self.get_connection()

        conditions = []
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if rule is not None and rule.upper() != "ALL":
            conditions.append("rule_name = ?")
            params.append(str(rule))
        if severity is not None and severity.upper() != "ALL":
            conditions.append("severity = ?")
            params.append(str(severity))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Determine time span if auto
        res = bucket_resolution.lower()
        if res == "auto":
            range_query = f"SELECT MIN(timestamp), MAX(timestamp) FROM alerts {where_clause};"
            min_max = conn.execute(range_query, tuple(params)).fetchone()
            if min_max and min_max[0] is not None and min_max[1] is not None:
                span = float(min_max[1]) - float(min_max[0])
            elif start_time is not None and end_time is not None:
                span = float(end_time) - float(start_time)
            else:
                span = 0.0

            if span > 14 * 86400:
                strftime_fmt = "%Y-%m-%d UTC"
            elif span > 86400:
                strftime_fmt = "%Y-%m-%d %H:00 UTC"
            elif span > 3600:
                strftime_fmt = "%Y-%m-%d %H:%M UTC"
            else:
                strftime_fmt = "%Y-%m-%d %H:%M:%S UTC"
        elif res == "day":
            strftime_fmt = "%Y-%m-%d UTC"
        elif res == "hour":
            strftime_fmt = "%Y-%m-%d %H:00 UTC"
        elif res == "minute":
            strftime_fmt = "%Y-%m-%d %H:%M UTC"
        else:
            strftime_fmt = "%Y-%m-%d %H:%M:%S UTC"

        query = f"""
            SELECT
                strftime('{strftime_fmt}', timestamp, 'unixepoch') AS bucket,
                COUNT(*) AS count
            FROM alerts
            {where_clause}
            GROUP BY bucket
            ORDER BY MIN(timestamp) ASC;
        """
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_top_alert_sources(
        self,
        limit: int = 5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        rule: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top offending source IPs by alert count.

        Args:
            limit: Maximum number of entries.
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            rule: Optional rule filter.
            severity: Optional severity filter.

        Returns:
            List[Dict[str, Any]]: List of dicts with 'src_ip' and 'count'.
        """
        conditions = ["src_ip IS NOT NULL", "TRIM(src_ip) != ''"]
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if rule is not None and rule.upper() != "ALL":
            conditions.append("rule_name = ?")
            params.append(str(rule))
        if severity is not None and severity.upper() != "ALL":
            conditions.append("severity = ?")
            params.append(str(severity))

        where_clause = f"WHERE {' AND '.join(conditions)}"
        query = f"""
            SELECT src_ip, COUNT(*) AS count
            FROM alerts
            {where_clause}
            GROUP BY src_ip
            ORDER BY count DESC
            LIMIT ?;
        """
        params.append(int(limit))
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_top_alert_destinations(
        self,
        limit: int = 5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        rule: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top targeted destination IPs by alert count.

        Args:
            limit: Maximum number of entries.
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            rule: Optional rule filter.
            severity: Optional severity filter.

        Returns:
            List[Dict[str, Any]]: List of dicts with 'dst_ip' and 'count'.
        """
        conditions = ["dst_ip IS NOT NULL", "TRIM(dst_ip) != ''"]
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if rule is not None and rule.upper() != "ALL":
            conditions.append("rule_name = ?")
            params.append(str(rule))
        if severity is not None and severity.upper() != "ALL":
            conditions.append("severity = ?")
            params.append(str(severity))

        where_clause = f"WHERE {' AND '.join(conditions)}"
        query = f"""
            SELECT dst_ip, COUNT(*) AS count
            FROM alerts
            {where_clause}
            GROUP BY dst_ip
            ORDER BY count DESC
            LIMIT ?;
        """
        params.append(int(limit))
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_protocol_distribution(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        protocol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate stored packets grouped by transport/network protocol.

        Args:
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            protocol: Optional protocol filter (e.g. 'TCP', 'UDP', 'ICMP').

        Returns:
            List[Dict[str, Any]]: List of dicts with 'protocol', 'count', and 'total_bytes'.
        """
        conditions = []
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if protocol is not None and protocol.upper() != "ALL":
            conditions.append("UPPER(protocol) = UPPER(?)")
            params.append(str(protocol))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT
                protocol,
                COUNT(*) AS count,
                COALESCE(SUM(length), 0) AS total_bytes
            FROM packets
            {where_clause}
            GROUP BY protocol
            ORDER BY count DESC;
        """
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_traffic_timeline(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        protocol: Optional[str] = None,
        bucket_resolution: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Aggregate packet volume (packet counts and byte sums) across time buckets in UTC.

        Args:
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            protocol: Optional protocol filter.
            bucket_resolution: Granularity ('auto', 'day', 'hour', 'minute', 'second').

        Returns:
            List[Dict[str, Any]]: Chronologically ordered list of dicts with 'bucket',
                                  'packet_count', and 'total_bytes'.
        """
        conn = self.get_connection()

        conditions = []
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if protocol is not None and protocol.upper() != "ALL":
            conditions.append("UPPER(protocol) = UPPER(?)")
            params.append(str(protocol))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        res = bucket_resolution.lower()
        if res == "auto":
            range_query = f"SELECT MIN(timestamp), MAX(timestamp) FROM packets {where_clause};"
            min_max = conn.execute(range_query, tuple(params)).fetchone()
            if min_max and min_max[0] is not None and min_max[1] is not None:
                span = float(min_max[1]) - float(min_max[0])
            elif start_time is not None and end_time is not None:
                span = float(end_time) - float(start_time)
            else:
                span = 0.0

            if span > 14 * 86400:
                strftime_fmt = "%Y-%m-%d UTC"
            elif span > 86400:
                strftime_fmt = "%Y-%m-%d %H:00 UTC"
            elif span > 3600:
                strftime_fmt = "%Y-%m-%d %H:%M UTC"
            else:
                strftime_fmt = "%Y-%m-%d %H:%M:%S UTC"
        elif res == "day":
            strftime_fmt = "%Y-%m-%d UTC"
        elif res == "hour":
            strftime_fmt = "%Y-%m-%d %H:00 UTC"
        elif res == "minute":
            strftime_fmt = "%Y-%m-%d %H:%M UTC"
        else:
            strftime_fmt = "%Y-%m-%d %H:%M:%S UTC"

        query = f"""
            SELECT
                strftime('{strftime_fmt}', timestamp, 'unixepoch') AS bucket,
                COUNT(*) AS packet_count,
                COALESCE(SUM(length), 0) AS total_bytes
            FROM packets
            {where_clause}
            GROUP BY bucket
            ORDER BY MIN(timestamp) ASC;
        """
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_top_destination_ports(
        self,
        limit: int = 5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        protocol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most targeted destination ports by packet count.

        Args:
            limit: Maximum number of entries.
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            protocol: Optional protocol filter.

        Returns:
            List[Dict[str, Any]]: List of dicts with 'dst_port', 'packet_count', and 'total_bytes'.
        """
        conditions = ["dst_port IS NOT NULL"]
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if protocol is not None and protocol.upper() != "ALL":
            conditions.append("UPPER(protocol) = UPPER(?)")
            params.append(str(protocol))

        where_clause = f"WHERE {' AND '.join(conditions)}"
        query = f"""
            SELECT
                dst_port,
                COUNT(*) AS packet_count,
                COALESCE(SUM(length), 0) AS total_bytes
            FROM packets
            {where_clause}
            GROUP BY dst_port
            ORDER BY packet_count DESC
            LIMIT ?;
        """
        params.append(int(limit))
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_top_packet_destinations(
        self,
        limit: int = 5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        protocol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top destination IPs receiving network traffic by packet count.

        Args:
            limit: Maximum number of entries.
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            protocol: Optional protocol filter.

        Returns:
            List[Dict[str, Any]]: List of dicts with 'dst_ip', 'packet_count', and 'total_bytes'.
        """
        conditions = ["dst_ip IS NOT NULL", "TRIM(dst_ip) != ''"]
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))
        if protocol is not None and protocol.upper() != "ALL":
            conditions.append("UPPER(protocol) = UPPER(?)")
            params.append(str(protocol))

        where_clause = f"WHERE {' AND '.join(conditions)}"
        query = f"""
            SELECT
                dst_ip,
                COUNT(*) AS packet_count,
                COALESCE(SUM(length), 0) AS total_bytes
            FROM packets
            {where_clause}
            GROUP BY dst_ip
            ORDER BY packet_count DESC
            LIMIT ?;
        """
        params.append(int(limit))
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_detection_method_distribution(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate alert counts by detection method (HEURISTIC, ML_SUPERVISED, ML_ANOMALY, HYBRID)."""
        conditions = []
        params: List[Any] = []
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(float(end_time))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT COALESCE(detection_method, 'HEURISTIC') AS detection_method, COUNT(*) AS count
            FROM alerts
            {where_clause}
            GROUP BY detection_method
            ORDER BY count DESC;
        """
        conn = self.get_connection()
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    # --------------------------------------------------------------------------
    # Model Registry Methods
    # --------------------------------------------------------------------------

    def insert_model(self, model_meta: Dict[str, Any]) -> int:
        """Insert or replace a trained machine learning model record in the registry."""
        import json
        conn = self.get_connection()
        query = """
            INSERT OR REPLACE INTO models (
                name, version, model_type, algorithm, dataset_name,
                task_type, metrics_json, feature_schema_json,
                artifact_path, created_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        metrics_str = (
            json.dumps(model_meta.get("metrics", {}))
            if isinstance(model_meta.get("metrics"), dict)
            else str(model_meta.get("metrics_json", "{}"))
        )
        schema_str = (
            json.dumps(model_meta.get("feature_schema", []))
            if isinstance(model_meta.get("feature_schema"), (list, dict))
            else str(model_meta.get("feature_schema_json", "[]"))
        )
        params = (
            str(model_meta["name"]),
            str(model_meta["version"]),
            str(model_meta.get("model_type", "supervised")),
            str(model_meta.get("algorithm", "unknown")),
            model_meta.get("dataset_name"),
            model_meta.get("task_type", "binary"),
            metrics_str,
            schema_str,
            str(model_meta.get("artifact_path", "")),
            float(model_meta.get("created_at", 0.0)),
            1 if model_meta.get("is_active") else 0,
        )
        with conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid

    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieve all registered models ordered by creation time descending."""
        import json
        conn = self.get_connection()
        query = "SELECT * FROM models ORDER BY created_at DESC;"
        cursor = conn.execute(query)
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            try:
                d["metrics"] = json.loads(d.get("metrics_json") or "{}")
            except Exception:
                d["metrics"] = {}
            try:
                d["feature_schema"] = json.loads(d.get("feature_schema_json") or "[]")
            except Exception:
                d["feature_schema"] = []
            rows.append(d)
        return rows

    def get_active_model(self, model_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve the currently active model, optionally filtered by model_type."""
        import json
        conn = self.get_connection()
        if model_type:
            query = "SELECT * FROM models WHERE is_active = 1 AND model_type = ? ORDER BY created_at DESC LIMIT 1;"
            cursor = conn.execute(query, (str(model_type),))
        else:
            query = "SELECT * FROM models WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1;"
            cursor = conn.execute(query)
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["metrics"] = json.loads(d.get("metrics_json") or "{}")
        except Exception:
            d["metrics"] = {}
        try:
            d["feature_schema"] = json.loads(d.get("feature_schema_json") or "[]")
        except Exception:
            d["feature_schema"] = []
        return d

    def set_active_model(self, version: str, is_active: bool = True) -> bool:
        """Set a model version as active (deactivating others of the same model_type)."""
        conn = self.get_connection()
        with conn:
            c = conn.execute("SELECT model_type FROM models WHERE version = ?;", (version,))
            row = c.fetchone()
            if not row:
                return False
            mtype = row[0]
            if is_active:
                conn.execute("UPDATE models SET is_active = 0 WHERE model_type = ?;", (mtype,))
                conn.execute("UPDATE models SET is_active = 1 WHERE version = ?;", (version,))
            else:
                conn.execute("UPDATE models SET is_active = 0 WHERE version = ?;", (version,))
        return True

    def get_model_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Retrieve a registered model record by version string."""
        import json
        conn = self.get_connection()
        query = "SELECT * FROM models WHERE version = ?;"
        cursor = conn.execute(query, (str(version),))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["metrics"] = json.loads(d.get("metrics_json") or "{}")
        except Exception:
            d["metrics"] = {}
        try:
            d["feature_schema"] = json.loads(d.get("feature_schema_json") or "[]")
        except Exception:
            d["feature_schema"] = []
        return d

    # --------------------------------------------------------------------------
    # Flow Persistence Methods
    # --------------------------------------------------------------------------

    def insert_flows(self, flows: Iterable[Any]) -> int:
        """Insert multiple flow records in a single transaction."""
        import json
        conn = self.get_connection()
        query = """
            INSERT INTO flows (
                flow_id, start_time, end_time, duration,
                src_ip, dst_ip, src_port, dst_port, protocol,
                packet_count, byte_count, fwd_packets, bwd_packets,
                fwd_bytes, bwd_bytes, flow_pkts_s, flow_bytes_s, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        param_list = []
        for f in flows:
            feat_dict = getattr(f, "features", {}) if hasattr(f, "features") else (f.get("features", {}) if isinstance(f, dict) else {})
            param_list.append((
                str(getattr(f, "flow_id", f.get("flow_id") if isinstance(f, dict) else "")),
                float(getattr(f, "start_time", f.get("start_time", 0.0) if isinstance(f, dict) else 0.0)),
                float(getattr(f, "end_time", f.get("end_time", 0.0) if isinstance(f, dict) else 0.0)),
                float(getattr(f, "duration", f.get("duration", 0.0) if isinstance(f, dict) else 0.0)),
                str(getattr(f, "src_ip", f.get("src_ip", "") if isinstance(f, dict) else "")),
                str(getattr(f, "dst_ip", f.get("dst_ip", "") if isinstance(f, dict) else "")),
                getattr(f, "src_port", f.get("src_port") if isinstance(f, dict) else None),
                getattr(f, "dst_port", f.get("dst_port") if isinstance(f, dict) else None),
                str(getattr(f, "protocol", f.get("protocol", "TCP") if isinstance(f, dict) else "TCP")),
                int(getattr(f, "packet_count", f.get("packet_count", 0) if isinstance(f, dict) else 0)),
                int(getattr(f, "byte_count", f.get("byte_count", 0) if isinstance(f, dict) else 0)),
                int(getattr(f, "fwd_packets", f.get("fwd_packets", 0) if isinstance(f, dict) else 0)),
                int(getattr(f, "bwd_packets", f.get("bwd_packets", 0) if isinstance(f, dict) else 0)),
                int(getattr(f, "fwd_bytes", f.get("fwd_bytes", 0) if isinstance(f, dict) else 0)),
                int(getattr(f, "bwd_bytes", f.get("bwd_bytes", 0) if isinstance(f, dict) else 0)),
                float(getattr(f, "flow_pkts_s", f.get("flow_pkts_s", 0.0) if isinstance(f, dict) else 0.0)),
                float(getattr(f, "flow_bytes_s", f.get("flow_bytes_s", 0.0) if isinstance(f, dict) else 0.0)),
                json.dumps(feat_dict) if isinstance(feat_dict, dict) else "{}",
            ))
        if not param_list:
            return 0
        with conn:
            conn.executemany(query, param_list)
        return len(param_list)

    def get_flows(self, limit: Optional[int] = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve stored flow records."""
        conn = self.get_connection()
        if limit is not None:
            query = "SELECT * FROM flows ORDER BY start_time DESC LIMIT ? OFFSET ?;"
            cursor = conn.execute(query, (int(limit), int(offset)))
        else:
            query = "SELECT * FROM flows ORDER BY start_time DESC;"
            cursor = conn.execute(query)
        return [dict(r) for r in cursor.fetchall()]

    def get_flow_count(self) -> int:
        """Return total flow records count."""
        conn = self.get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM flows;")
        return cursor.fetchone()[0]

    # --------------------------------------------------------------------------
    # Prevention Actions, Approval Workflow & Audit Log Persistence
    # --------------------------------------------------------------------------

    def insert_prevention_action(self, action: Dict[str, Any]) -> str:
        """Insert a newly recommended, approved, or executed prevention action."""
        import json
        import time
        import uuid
        conn = self.get_connection()
        action_id = str(action.get("action_id") or f"act_{uuid.uuid4().hex[:12]}")
        details_val = action.get("details_json") or action.get("details", {})
        if isinstance(details_val, (dict, list)):
            details_json = json.dumps(details_val)
        else:
            details_json = str(details_val)

        query = """
            INSERT INTO prevention_actions (
                action_id, alert_id, timestamp, target_ip, target_port, protocol,
                action_type, mode, status, threat_score, risk_level, rule_name,
                duration_seconds, expires_at, approved_by, executed_at, verified_at,
                rolled_back_at, details_json, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with conn:
            conn.execute(
                query,
                (
                    action_id,
                    action.get("alert_id"),
                    float(action.get("timestamp", time.time())),
                    str(action.get("target_ip", "")),
                    action.get("target_port"),
                    action.get("protocol"),
                    str(action.get("action_type", "BLOCK_IP")),
                    str(action.get("mode", "SIMULATION")),
                    str(action.get("status", "RECOMMENDED")),
                    float(action.get("threat_score", 0.0)),
                    str(action.get("risk_level", "LOW")),
                    str(action.get("rule_name", "UNKNOWN")),
                    int(action.get("duration_seconds", 300)),
                    action.get("expires_at"),
                    action.get("approved_by"),
                    action.get("executed_at"),
                    action.get("verified_at"),
                    action.get("rolled_back_at"),
                    details_json,
                    action.get("error_message"),
                ),
            )
        return action_id

    def update_prevention_action_status(
        self,
        action_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        executed_at: Optional[float] = None,
        verified_at: Optional[float] = None,
        rolled_back_at: Optional[float] = None,
        error_message: Optional[str] = None,
        approved_by: Optional[str] = None,
        rule_name: Optional[str] = None,
    ) -> bool:
        """Update lifecycle status and execution timestamps for a prevention action."""
        import json
        conn = self.get_connection()
        fields = ["status = ?"]
        params: List[Any] = [str(status)]

        if details is not None:
            fields.append("details_json = ?")
            params.append(json.dumps(details) if isinstance(details, (dict, list)) else str(details))
        if executed_at is not None:
            fields.append("executed_at = ?")
            params.append(float(executed_at))
        if verified_at is not None:
            fields.append("verified_at = ?")
            params.append(float(verified_at))
        if rolled_back_at is not None:
            fields.append("rolled_back_at = ?")
            params.append(float(rolled_back_at))
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(str(error_message))
        if approved_by is not None:
            fields.append("approved_by = ?")
            params.append(str(approved_by))
        if rule_name is not None:
            fields.append("rule_name = ?")
            params.append(str(rule_name))

        params.append(str(action_id))
        query = f"UPDATE prevention_actions SET {', '.join(fields)} WHERE action_id = ?;"
        with conn:
            cursor = conn.execute(query, tuple(params))
            return cursor.rowcount > 0

    def get_prevention_actions(
        self,
        limit: Optional[int] = 100,
        offset: int = 0,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        target_ip: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query prevention actions with optional mode, status, and IP filters."""
        import json
        conn = self.get_connection()
        clauses = []
        params: List[Any] = []
        if mode:
            clauses.append("mode = ?")
            params.append(str(mode))
        if status:
            clauses.append("status = ?")
            params.append(str(status))
        if target_ip:
            clauses.append("target_ip = ?")
            params.append(str(target_ip))

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        if limit is not None:
            query = f"SELECT * FROM prevention_actions {where_str} ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
            params.extend([int(limit), int(offset)])
        else:
            query = f"SELECT * FROM prevention_actions {where_str} ORDER BY timestamp DESC;"

        cursor = conn.execute(query, tuple(params))
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            try:
                d["details"] = json.loads(d.get("details_json") or "{}")
            except Exception:
                d["details"] = {}
            rows.append(d)
        return rows

    def get_prevention_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single prevention action by ID."""
        import json
        conn = self.get_connection()
        cursor = conn.execute("SELECT * FROM prevention_actions WHERE action_id = ?;", (str(action_id),))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["details"] = json.loads(d.get("details_json") or "{}")
        except Exception:
            d["details"] = {}
        return d

    def insert_approval_event(
        self,
        action_id: str,
        operator: str,
        decision: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a human operator approval or rejection decision."""
        import json
        import time
        conn = self.get_connection()
        details_json = json.dumps(details or {})
        query = """
            INSERT INTO approval_events (action_id, timestamp, operator, decision, reason, details_json)
            VALUES (?, ?, ?, ?, ?, ?);
        """
        with conn:
            cursor = conn.execute(query, (str(action_id), time.time(), str(operator), str(decision), reason, details_json))
            return cursor.lastrowid

    def get_approval_events(self, action_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recorded operator approval decisions."""
        import json
        conn = self.get_connection()
        if action_id:
            cursor = conn.execute("SELECT * FROM approval_events WHERE action_id = ? ORDER BY timestamp DESC LIMIT ?;", (str(action_id), int(limit)))
        else:
            cursor = conn.execute("SELECT * FROM approval_events ORDER BY timestamp DESC LIMIT ?;", (int(limit),))
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            try:
                d["details"] = json.loads(d.get("details_json") or "{}")
            except Exception:
                d["details"] = {}
            rows.append(d)
        return rows

    def insert_audit_event(
        self,
        actor: str,
        action: str,
        target: str,
        mode: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert an immutable record into the audit log."""
        import json
        import time
        conn = self.get_connection()
        ts = float(timestamp if timestamp is not None else time.time())
        details_json = json.dumps(details or {})
        query = """
            INSERT INTO audit_log (timestamp, actor, action, target, mode, result, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with conn:
            cursor = conn.execute(query, (ts, str(actor), str(action), str(target), str(mode), str(result), details_json))
            return cursor.lastrowid

    def get_audit_logs(
        self,
        limit: Optional[int] = 100,
        offset: int = 0,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        mode: Optional[str] = None,
        result: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query immutable audit log entries with flexible filtering."""
        import json
        conn = self.get_connection()
        clauses = []
        params: List[Any] = []
        if actor:
            clauses.append("actor = ?")
            params.append(str(actor))
        if action:
            clauses.append("action = ?")
            params.append(str(action))
        if mode:
            clauses.append("mode = ?")
            params.append(str(mode))
        if result:
            clauses.append("result = ?")
            params.append(str(result))

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        if limit is not None:
            query = f"SELECT * FROM audit_log {where_str} ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
            params.extend([int(limit), int(offset)])
        else:
            query = f"SELECT * FROM audit_log {where_str} ORDER BY timestamp DESC;"

        cursor = conn.execute(query, tuple(params))
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            try:
                d["details"] = json.loads(d.get("details_json") or "{}")
            except Exception:
                d["details"] = {}
            rows.append(d)
        return rows

    def get_prevention_stats(self) -> Dict[str, Any]:
        """Aggregate prevention and audit KPIs."""
        conn = self.get_connection()
        total_actions = conn.execute("SELECT COUNT(*) FROM prevention_actions;").fetchone()[0]
        active_rules = conn.execute("SELECT COUNT(*) FROM prevention_actions WHERE status = 'SIMULATED';").fetchone()[0]
        pending_approvals = conn.execute("SELECT COUNT(*) FROM prevention_actions WHERE status = 'AWAITING_APPROVAL';").fetchone()[0]
        simulated_count = conn.execute("SELECT COUNT(*) FROM prevention_actions WHERE status = 'SIMULATED';").fetchone()[0]
        # Compatibility column retained for existing UI callers. Enforcement is disabled.
        enforced_count = 0
        total_audit_events = conn.execute("SELECT COUNT(*) FROM audit_log;").fetchone()[0]

        return {
            "total_prevention_actions": total_actions,
            "active_rules": active_rules,
            "pending_approvals": pending_approvals,
            "simulated_actions": simulated_count,
            "enforced_actions": enforced_count,
            "total_audit_events": total_audit_events,
        }

    def close(self) -> None:
        """Safely close the SQLite database connection if open."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as err:
                logger.debug("Error closing database connection: %s", err)
            finally:
                self._connection = None

    def __enter__(self) -> "DatabaseManager":
        self.get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Top-level module convenience functions


def init_db(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> DatabaseManager:
    """Convenience function to initialize the database schema and return a manager."""
    db = DatabaseManager(db_path)
    db.init_db()
    return db
