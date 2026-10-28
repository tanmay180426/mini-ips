"""Simulation-record revocation and expiration management for Mini-IDS.

Handles expiration, operator revocation, and shutdown cleanup for in-memory
simulation records. It never modifies host firewall state.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from prevention.base_adapter import BaseFirewallAdapter
from prevention.policy_engine import ActionStatus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class RollbackManager:
    """Manages active rule cleanup, timed expiration, and unblock workflows."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def unblock_action(
        self,
        action_id: str,
        adapter: BaseFirewallAdapter,
        operator: str = "Operator",
        reason: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Revoke an active simulation record without changing network behavior."""
        action = self.db.get_prevention_action(action_id)
        if not action:
            return False, f"Action ID '{action_id}' not found."

        if action.get("status") not in {ActionStatus.SIMULATED.value, ActionStatus.ACTIVE.value, ActionStatus.VERIFIED.value}:
            return False, "Only an active simulation can be revoked."
        rule_name = action.get("rule_name")
        target_ip = action.get("target_ip", "")
        try:
            success, msg = adapter.remove_rule(rule_name)
        except Exception as exc:
            success, msg = False, f"Simulation revocation failed: {exc}"
        now = time.time()

        if success or "not found" in msg.lower():
            self.db.update_prevention_action_status(
                action_id=action_id,
                status=ActionStatus.ROLLED_BACK.value,
                rolled_back_at=now,
            )
            self.db.insert_audit_event(
                actor=operator,
                action="REVOKE_SIMULATION",
                target=target_ip,
                mode=action.get("mode", ""),
                result="SUCCESS",
                details={"action_id": action_id, "rule_name": rule_name, "reason": reason or "Operator requested unblock."},
                timestamp=now,
            )
            logger.info("Simulation record revoked for '%s' (IP %s)", rule_name, target_ip)
            return True, f"Simulation revoked for {target_ip} (Rule: {rule_name}); no network block was removed."
        else:
            self.db.insert_audit_event(
                actor=operator,
                action="REVOKE_SIMULATION",
                target=target_ip,
                mode=action.get("mode", ""),
                result="FAILURE",
                details={"action_id": action_id, "rule_name": rule_name, "error": msg},
                timestamp=now,
            )
            return False, f"Failed to remove rule '{rule_name}': {msg}"

    def check_and_expire_rules(self, adapter: BaseFirewallAdapter) -> int:
        """Expire simulation records whose timestamps have passed."""
        now = time.time()
        active_actions = self.db.get_prevention_actions(limit=500)
        expired_count = 0

        for act in active_actions:
            status = act.get("status")
            if status not in [ActionStatus.SIMULATED.value, ActionStatus.ACTIVE.value, ActionStatus.VERIFIED.value]:
                continue

            expires_at = act.get("expires_at")
            if expires_at and now >= expires_at:
                action_id = act.get("action_id")
                rule_name = act.get("rule_name")
                target_ip = act.get("target_ip", "")

                try:
                    success, message = adapter.remove_rule(rule_name)
                except Exception as exc:
                    success, message = False, f"Simulation expiration cleanup failed: {exc}"
                if not success and "not found" not in message.lower():
                    self.db.update_prevention_action_status(action_id=action_id, status=ActionStatus.FAILED.value, error_message=f"Simulation expiration cleanup failed: {message}")
                    self.db.insert_audit_event(actor="EXPIRATION_MONITOR", action="EXPIRE_RULE_FAILURE", target=target_ip, mode="SIMULATION", result="FAILURE", details={"action_id": action_id, "error": message}, timestamp=now)
                    continue
                self.db.update_prevention_action_status(
                    action_id=action_id,
                    status=ActionStatus.EXPIRED.value,
                    rolled_back_at=now,
                )
                self.db.insert_audit_event(
                    actor="EXPIRATION_MONITOR",
                    action="EXPIRE_SIMULATION",
                    target=target_ip,
                    mode="SIMULATION",
                    result="SUCCESS",
                    details={"action_id": action_id, "rule_name": rule_name, "expired_at": now, "traffic_prevented": False},
                    timestamp=now,
                )
                expired_count += 1
                logger.info("Auto-expired prevention rule '%s' for IP %s", rule_name, target_ip)

        return expired_count

    def rollback_all_active(self, adapter: BaseFirewallAdapter, reason: str = "Clean Shutdown") -> int:
        """Emergency or graceful shutdown cleanup of all active rules."""
        active_actions = self.db.get_prevention_actions(limit=500)
        rolled_back_count = 0
        now = time.time()

        for act in active_actions:
            if act.get("status") in [ActionStatus.SIMULATED.value, ActionStatus.ACTIVE.value, ActionStatus.VERIFIED.value]:
                rule_name = act.get("rule_name")
                try:
                    success, message = adapter.remove_rule(rule_name)
                except Exception as exc:
                    success, message = False, f"Simulation shutdown cleanup failed: {exc}"
                if not success and "not found" not in message.lower():
                    self.db.update_prevention_action_status(action_id=act.get("action_id"), status=ActionStatus.FAILED.value, error_message=f"Simulation shutdown cleanup failed: {message}")
                    self.db.insert_audit_event(actor="SHUTDOWN_HANDLER", action="ROLLBACK_FAILURE", target=act.get("target_ip", ""), mode="SIMULATION", result="FAILURE", details={"action_id": act.get("action_id"), "error": message}, timestamp=now)
                    continue
                self.db.update_prevention_action_status(
                    action_id=act.get("action_id"),
                    status=ActionStatus.ROLLED_BACK.value,
                    rolled_back_at=now,
                )
                self.db.insert_audit_event(
                    actor="SHUTDOWN_HANDLER",
                    action="CLEAR_SIMULATION_RECORD",
                    target=act.get("target_ip", ""),
                    mode="SIMULATION",
                    result="SUCCESS",
                    details={"action_id": act.get("action_id"), "rule_name": rule_name, "reason": reason, "traffic_prevented": False},
                    timestamp=now,
                )
                rolled_back_count += 1

        return rolled_back_count
