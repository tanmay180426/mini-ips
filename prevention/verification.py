"""Simulation-record verification subsystem for Mini-IDS.

Verification confirms only that an in-memory simulation record exists. It never
asserts that network traffic was prevented.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

from prevention.base_adapter import BaseFirewallAdapter
from prevention.policy_engine import ActionStatus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class VerificationManager:
    """Verifies in-memory simulation records only."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def verify_action(
        self,
        action_id: str,
        adapter: BaseFirewallAdapter,
        rule_name: str,
        target_ip: str,
    ) -> bool:
        """Query the adapter to verify rule presence and update persistent status."""
        now = time.time()
        try:
            is_verified = adapter.verify_rule(rule_name, target_ip)
        except Exception as exc:
            is_verified = False
            error_message = f"Simulation record verification failed: {exc}"
        else:
            error_message = "Simulation record verification failed: no in-memory simulation record found."

        if is_verified:
            self.db.update_prevention_action_status(
                action_id=action_id,
                status=ActionStatus.SIMULATED.value,
                verified_at=now,
            )
            self.db.insert_audit_event(
                actor="VERIFICATION_ENGINE",
                action="VERIFY_SIMULATION_RECORD",
                target=target_ip,
                mode="SIMULATION",
                result="SUCCESS",
                details={"action_id": action_id, "rule_name": rule_name, "verified": True, "traffic_prevented": False},
                timestamp=now,
            )
            logger.info("Rule '%s' for IP %s verified successfully.", rule_name, target_ip)
            return True
        else:
            self.db.update_prevention_action_status(
                action_id=action_id,
                status=ActionStatus.FAILED.value,
                error_message=error_message,
            )
            self.db.insert_audit_event(
                actor="VERIFICATION_ENGINE",
                action="VERIFY_SIMULATION_RECORD",
                target=target_ip,
                mode="SIMULATION",
                result="FAILURE",
                details={"action_id": action_id, "rule_name": rule_name, "verified": False, "traffic_prevented": False, "error": error_message},
                timestamp=now,
            )
            logger.warning("Verification check failed for rule '%s' (target: %s).", rule_name, target_ip)
            return False
