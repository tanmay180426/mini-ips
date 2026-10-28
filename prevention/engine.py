"""Master Prevention Engine Orchestrator for Mini-IDS.

Coordinates policy evaluation, manual approval queues, OS/simulation firewall execution,
verification checks, automatic rule rollback, and comprehensive audit logging.
"""

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from prevention.approval_manager import ApprovalManager
from prevention.base_adapter import BaseFirewallAdapter
from prevention.policy_engine import (
    ActionStatus,
    ActionType,
    PolicyEngine,
    PreventionAction,
    PreventionMode,
    RiskLevel,
    POLICY_VERSION,
)
from prevention.rollback import RollbackManager
from prevention.simulation_adapter import SimulationAdapter
from prevention.verification import VerificationManager
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class PreventionEngine:
    """End-to-end intrusion prevention orchestrator."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        initial_mode: PreventionMode = PreventionMode.SIMULATION,
        block_threshold: float = 0.70,
        rate_limit_threshold: float = 0.50,
        default_block_duration: int = 300,
        custom_allowlist: Optional[List[str]] = None,
        cooldown_seconds: int = 300,
        rate_limit_window_seconds: int = 60,
        max_actions_per_window: int = 20,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.db.init_db()
        if cooldown_seconds < 0 or rate_limit_window_seconds <= 0 or max_actions_per_window <= 0:
            raise ValueError("Cooldown must be non-negative and rate/action limits must be positive")
        self.cooldown_seconds = cooldown_seconds
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.max_actions_per_window = max_actions_per_window

        safe_initial_mode = PreventionMode.SIMULATION if PreventionMode(initial_mode) == PreventionMode.AUTHORIZED_ENFORCEMENT else PreventionMode(initial_mode)
        self.policy_engine = PolicyEngine(
            default_mode=safe_initial_mode,
            block_threshold=block_threshold,
            rate_limit_threshold=rate_limit_threshold,
            default_block_duration=default_block_duration,
            custom_allowlist=custom_allowlist,
        )

        self.approval_manager = ApprovalManager(self.db)
        self.verification_manager = VerificationManager(self.db)
        self.rollback_manager = RollbackManager(self.db)

        # Only the in-memory adapter is constructed or used by the engine.
        self.simulation_adapter = SimulationAdapter()

        # Threading for automatic timed rule expiration worker
        self._expiration_thread: Optional[threading.Thread] = None
        self._expiration_stop_event = threading.Event()

    @property
    def mode(self) -> PreventionMode:
        return self.policy_engine.mode

    @mode.setter
    def mode(self, new_mode: PreventionMode) -> None:
        self.set_mode(new_mode)

    def set_mode(self, new_mode: PreventionMode) -> None:
        """Switch operational prevention mode with audit trail."""
        old_mode = self.policy_engine.mode
        requested_mode = PreventionMode(new_mode)
        safe_mode = PreventionMode.SIMULATION if requested_mode == PreventionMode.AUTHORIZED_ENFORCEMENT else requested_mode
        self.policy_engine.mode = safe_mode
        self.db.insert_audit_event(
            actor="SYSTEM_OPERATOR",
            action="CHANGE_PREVENTION_MODE",
            target=f"{old_mode.value} -> {safe_mode.value}",
            mode=safe_mode.value,
            result="SUCCESS" if safe_mode == requested_mode else "DENIED",
            details={"previous_mode": old_mode.value, "requested_mode": requested_mode.value, "new_mode": safe_mode.value, "reason": "Simulation-only backend safety policy."},
        )
        logger.info("Prevention mode switched: %s -> %s", old_mode.value, safe_mode.value)

    def get_active_adapter(self, mode: Optional[PreventionMode] = None) -> BaseFirewallAdapter:
        """Select appropriate adapter based on requested mode and host OS.

        CRITICAL SIMULATION GUARD:
        In this Advanced ML-Assisted Mini-IPS project, simulation-only prevention
        is strictly enforced across all operational paths. Host operating system firewall
        commands (netsh/iptables) are blocked by design. All actions execute safely via SimulationAdapter.
        """
        return self.simulation_adapter

    def process_threat(
        self,
        src_ip: str,
        rule_name: str,
        alert_id: Optional[str] = None,
        dst_ip: Optional[str] = None,
        target_port: Optional[int] = None,
        protocol: Optional[str] = None,
        heuristic_severity: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        predicted_class: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> PreventionAction:
        """Process an alert or flow observation through policy evaluation and execution."""
        # Evaluation failures are converted to an auditable fail-closed result.
        try:
            action = self.policy_engine.evaluate(
            src_ip=src_ip,
            rule_name=rule_name,
            alert_id=alert_id,
            dst_ip=dst_ip,
            target_port=target_port,
            protocol=protocol,
            heuristic_severity=heuristic_severity,
            ml_confidence=ml_confidence,
            anomaly_score=anomaly_score,
            predicted_class=predicted_class,
            source_tag=source_tag,
            )
        except Exception as exc:
            logger.exception("Policy evaluation failed; refusing simulated action")
            action = PreventionAction(
                action_id=f"act_{uuid.uuid4().hex[:12]}", target_ip=str(src_ip or "UNKNOWN"),
                action_type=ActionType.LOG_ALERT, mode=PreventionMode.SIMULATION,
                status=ActionStatus.FAILED, threat_score=0.0, risk_level=RiskLevel.LOW,
                rule_name=str(rule_name or "UNKNOWN"), alert_id=alert_id,
                details={"policy_version": POLICY_VERSION, "reason": f"Policy evaluation failed: {exc}"},
            )
            action.error_message = f"Policy evaluation failed: {exc}"

        # 2. Persist initial action recommendation
        try:
            self.db.insert_prevention_action(action.to_dict())
        except Exception as exc:
            logger.exception("Could not persist prevention decision; no action will be executed")
            action.status = ActionStatus.FAILED
            action.error_message = f"Decision persistence failed: {exc}"
            return action

        # 3. Log initial recommendation audit event
        try:
            self.db.insert_audit_event(
                actor="POLICY_ENGINE", action="POLICY_EVALUATION", target=str(src_ip or "UNKNOWN"),
                mode=action.mode.value, result="SUCCESS" if action.status != ActionStatus.FAILED else "FAILURE",
                details={"action_id": action.action_id, "alert_id": action.alert_id, "action_type": action.action_type.value,
                         "status": action.status.value, "threat_score": action.threat_score, "risk_level": action.risk_level.value,
                         "rule_name": action.rule_name, "details": action.details, "policy_version": POLICY_VERSION},
            )
        except Exception as exc:
            logger.exception("Could not audit prevention decision; refusing action")
            action.status = ActionStatus.FAILED
            action.error_message = f"Decision audit persistence failed: {exc}"
            try:
                self.db.update_prevention_action_status(action.action_id, ActionStatus.FAILED.value, error_message=action.error_message)
            except Exception:
                logger.exception("Could not persist fail-closed audit error")
            return action

        if action.status == ActionStatus.FAILED:
            return action

        # Persistent limits are checked before any simulation side effect.
        try:
            history = self.db.get_prevention_actions(limit=None)
            now = time.time()
            recent = [item for item in history if item.get("action_id") != action.action_id and float(item.get("timestamp") or 0) >= now - self.rate_limit_window_seconds]
            if len(recent) >= self.max_actions_per_window:
                return self._suppress(action, "Maximum prevention actions per time window reached.", "RATE_LIMIT")
            target_history = [item for item in history if item.get("target_ip") == action.target_ip and item.get("status") in {ActionStatus.EXPIRED.value, ActionStatus.ROLLED_BACK.value}]
            if any(float(item.get("timestamp") or 0) >= now - self.cooldown_seconds for item in target_history):
                return self._suppress(action, f"Target cooldown active for {self.cooldown_seconds} seconds.", "COOLDOWN")
        except Exception as exc:
            action.status = ActionStatus.FAILED
            action.error_message = f"Policy limit lookup failed: {exc}"
            self.db.update_prevention_action_status(action.action_id, ActionStatus.FAILED.value, error_message=action.error_message)
            self.db.insert_audit_event(actor="PREVENTION_ENGINE", action="POLICY_LIMIT_FAILURE", target=src_ip, mode=action.mode.value, result="FAILURE", details={"action_id": action.action_id, "reason": action.error_message})
            return action

        # 4. Duplicate Action Suppression
        if action.action_type == ActionType.BLOCK_IP:
            active_rules = self.simulation_adapter.list_active_rules()
            has_active_rule = any(r.get("target_ip") == src_ip for r in active_rules)
            pending_queue = self.approval_manager.get_pending_queue()
            has_pending_approval = any(
                p.get("target_ip") == src_ip and p.get("action_id") != action.action_id
                for p in pending_queue
            )

            if has_active_rule or has_pending_approval:
                action.status = ActionStatus.SUPPRESSED
                existing_ref = "Active Simulation Record" if has_active_rule else "Pending Approval Queue"
                action.details["suppressed_duplicate"] = True
                action.details["existing_simulation_ref"] = existing_ref
                action.details["reason"] = (
                    f"Duplicate action suppressed: target IP '{src_ip}' already has an active or pending simulation ({existing_ref})."
                )
                self.db.update_prevention_action_status(
                    action_id=action.action_id,
                    status=ActionStatus.SUPPRESSED.value,
                    error_message=action.details["reason"],
                )
                self.db.insert_audit_event(
                    actor="PREVENTION_ENGINE",
                    action="SUPPRESS_DUPLICATE_ACTION",
                    target=src_ip,
                    mode=action.mode.value,
                    result="SUPPRESSED",
                    details={"action_id": action.action_id, "alert_id": action.alert_id, "target_ip": src_ip, "existing_ref": existing_ref, "reason": action.details["reason"]},
                )
                return action

        # 5. Execute or Queue according to mode and action type
        if action.action_type != ActionType.BLOCK_IP:
            # Low risk or non-blocking: remains LOG_ALERT / RATE_LIMIT
            return action

        if action.mode == PreventionMode.SIMULATION:
            self._execute_simulated(action)
        elif action.mode == PreventionMode.MANUAL_APPROVAL:
            logger.info("Action '%s' queued in Manual Approval Queue for %s", action.action_id, src_ip)
        elif action.mode == PreventionMode.AUTHORIZED_ENFORCEMENT:
            # Legacy persisted/configured values are fail-closed into simulation.
            self._execute_simulated(action)

        return action

    def _suppress(self, action: PreventionAction, reason: str, kind: str) -> PreventionAction:
        action.status = ActionStatus.SUPPRESSED
        action.details.update({"reason": reason, "suppression": kind, "policy_version": POLICY_VERSION})
        self.db.update_prevention_action_status(action.action_id, ActionStatus.SUPPRESSED.value, details=action.details, error_message=reason)
        self.db.insert_audit_event(actor="PREVENTION_ENGINE", action=f"SUPPRESS_{kind}", target=action.target_ip, mode=action.mode.value, result="SUPPRESSED", details={"action_id": action.action_id, "alert_id": action.alert_id, "reason": reason, "policy_version": POLICY_VERSION})
        return action

    def _execute_simulated(self, action: PreventionAction) -> None:
        """Execute simulated block without host modification."""
        if action.mode == PreventionMode.AUTHORIZED_ENFORCEMENT:
            action.status = ActionStatus.FAILED
            action.error_message = "Legacy enforcement mode is disabled."
            self.db.update_prevention_action_status(action.action_id, ActionStatus.FAILED.value, error_message=action.error_message)
            self.db.insert_audit_event(actor="SIMULATION_GUARD", action="DENY_ENFORCEMENT_REQUEST", target=action.target_ip, mode="SIMULATION", result="DENIED", details={"action_id": action.action_id, "reason": action.error_message, "traffic_prevented": False})
            return
        rule_name = f"MiniIDS_Block_{action.action_id}_{action.target_ip.replace('.', '_').replace(':', '_')}"
        adapter = self.simulation_adapter

        try:
            success, msg = adapter.apply_rule(
                rule_name=rule_name, target_ip=action.target_ip, protocol=action.protocol,
                port=action.target_port, duration_seconds=action.duration_seconds,
            )
        except Exception as exc:
            success, msg = False, f"Simulation adapter failure: {exc}"
        now = time.time()
        action.executed_at = now
        action.rule_name = rule_name

        if success:
            action.status = ActionStatus.SIMULATED
            action.details.update({"simulation_result": "Recorded in memory only; network traffic was not blocked.", "traffic_prevented": False})
            self.db.update_prevention_action_status(
                action_id=action.action_id,
                status=ActionStatus.SIMULATED.value,
                details=action.details,
                executed_at=now,
                rule_name=rule_name,
            )
            # Verify simulated presence
            if not self.verification_manager.verify_action(action.action_id, adapter, rule_name, action.target_ip):
                action.status = ActionStatus.FAILED
                action.error_message = "Simulation record verification failed."
                self.db.update_prevention_action_status(action.action_id, ActionStatus.FAILED.value, error_message=action.error_message)
            else:
                action.status = ActionStatus.SIMULATED
                action.verified_at = now

            self.db.insert_audit_event(
                actor="SIMULATION_ENGINE",
                action="RECORD_SIMULATION",
                target=action.target_ip,
                mode=PreventionMode.SIMULATION.value,
                result="SUCCESS",
                details={"action_id": action.action_id, "alert_id": action.alert_id, "rule_name": rule_name, "message": msg, "traffic_prevented": False},
                timestamp=now,
            )
        else:
            action.status = ActionStatus.FAILED
            action.error_message = msg
            self.db.update_prevention_action_status(
                action_id=action.action_id,
                status=ActionStatus.FAILED.value,
                error_message=msg,
            )
            self.db.insert_audit_event(actor="SIMULATION_ENGINE", action="SIMULATION_FAILURE", target=action.target_ip, mode="SIMULATION", result="FAILURE", details={"action_id": action.action_id, "alert_id": action.alert_id, "reason": msg, "traffic_prevented": False})

    def _execute_enforced(self, action: PreventionAction) -> None:
        """Compatibility denial; enforcement requests are never redirected to execution."""
        logger.warning(
            "Enforcement request denied by backend simulation-only guard for target '%s'.",
            action.target_ip,
        )
        action.status = ActionStatus.FAILED
        action.error_message = "Legacy enforcement mode is disabled; select simulation mode explicitly."
        self.db.update_prevention_action_status(action.action_id, ActionStatus.FAILED.value, error_message=action.error_message)
        self.db.insert_audit_event(actor="SIMULATION_GUARD", action="DENY_ENFORCEMENT_REQUEST", target=action.target_ip, mode=PreventionMode.SIMULATION.value, result="DENIED", details={"action_id": action.action_id, "reason": action.error_message, "traffic_prevented": False})

    def approve_and_execute(
        self,
        action_id: str,
        operator: str = "Admin_Operator",
        reason: Optional[str] = None,
    ) -> bool:
        """Approve an item in the queue and immediately execute the action."""
        if not self.approval_manager.approve_action(action_id, operator=operator, reason=reason):
            return False

        action_dict = self.db.get_prevention_action(action_id)
        if not action_dict:
            return False

        action = PreventionAction(
            action_id=action_dict["action_id"],
            target_ip=action_dict["target_ip"],
            action_type=ActionType(action_dict["action_type"]),
            mode=PreventionMode(action_dict["mode"]),
            status=ActionStatus(action_dict["status"]),
            threat_score=action_dict["threat_score"],
            risk_level=RiskLevel(action_dict["risk_level"]),
            rule_name=action_dict["rule_name"],
            duration_seconds=action_dict["duration_seconds"],
            alert_id=action_dict.get("alert_id"),
            target_port=action_dict.get("target_port"),
            protocol=action_dict.get("protocol"),
            details=action_dict.get("details", {}),
        )

        # Approved items execute via simulation
        self._execute_simulated(action)

        return action.status == ActionStatus.SIMULATED

    def reject(self, action_id: str, operator: str = "Admin_Operator", reason: Optional[str] = None) -> bool:
        """Reject a pending prevention action."""
        return self.approval_manager.reject_action(action_id, operator=operator, reason=reason)

    def unblock(self, action_id: str, operator: str = "Admin_Operator", reason: Optional[str] = None) -> Tuple[bool, str]:
        """Manually unblock an active rule."""
        action = self.db.get_prevention_action(action_id)
        if not action:
            return False, f"Action ID '{action_id}' not found."

        try:
            mode_str = action.get("mode", PreventionMode.SIMULATION.value)
            adapter = self.get_active_adapter(mode=PreventionMode(mode_str))
            return self.rollback_manager.unblock_action(action_id, adapter, operator=operator, reason=reason)
        except Exception as exc:
            logger.exception("Could not revoke simulated action %s", action_id)
            return False, f"Simulation revocation failed: {exc}"

    def expire_timed_rules(self) -> int:
        """Check all active rules and automatically roll back any expired entries."""
        adapter = self.get_active_adapter()
        return self.rollback_manager.check_and_expire_rules(adapter)

    def start_expiration_worker(self, interval_seconds: float = 5.0) -> None:
        """Start a daemon thread that periodically checks and expires timed simulation rules."""
        if self._expiration_thread is not None and self._expiration_thread.is_alive():
            return
        self._expiration_stop_event.clear()

        def _worker_loop() -> None:
            while not self._expiration_stop_event.is_set():
                try:
                    self.expire_timed_rules()
                except Exception as ex:
                    logger.debug("Error in prevention expiration worker: %s", ex)
                self._expiration_stop_event.wait(timeout=interval_seconds)

        self._expiration_thread = threading.Thread(
            target=_worker_loop,
            name="MiniIPS-RuleExpirationWorker",
            daemon=True,
        )
        self._expiration_thread.start()
        logger.info("Started background prevention rule expiration worker (interval: %.1fs)", interval_seconds)

    def stop_expiration_worker(self, timeout: float = 2.0) -> None:
        """Stop background rule expiration daemon safely."""
        self._expiration_stop_event.set()
        if self._expiration_thread is not None and self._expiration_thread.is_alive():
            self._expiration_thread.join(timeout=timeout)
            self._expiration_thread = None

    def shutdown(self) -> int:
        """Clean shutdown hook to stop worker and roll back all active simulation rules."""
        self.stop_expiration_worker()
        adapter = self.get_active_adapter()
        return self.rollback_manager.rollback_all_active(adapter, reason="System Shutdown")
