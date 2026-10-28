"""Manual Approval Manager for Mini-IDS Prevention Engine.

Manages human-in-the-loop governance for response recommendations, tracking pending actions,
handling operator approvals and rejections, and persisting immutable audit trails.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from prevention.policy_engine import ActionStatus, PreventionAction, PreventionMode
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class ApprovalManager:
    """Manages the pending approval queue and approval lifecycle transitions."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def get_pending_queue(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve all actions currently awaiting operator approval."""
        return self.db.get_prevention_actions(
            limit=limit,
            status=ActionStatus.AWAITING_APPROVAL.value,
        )

    def approve_action(
        self,
        action_id: str,
        operator: str = "Admin_Operator",
        reason: Optional[str] = None,
    ) -> bool:
        """Approve a pending prevention action for execution."""
        action = self.db.get_prevention_action(action_id)
        if not action:
            logger.error("Cannot approve: Action ID '%s' not found.", action_id)
            return False

        if action.get("status") != ActionStatus.AWAITING_APPROVAL.value:
            logger.warning("Action '%s' is not in AWAITING_APPROVAL status (current: %s).", action_id, action.get("status"))
            return False

        if action.get("mode") != PreventionMode.MANUAL_APPROVAL.value or action.get("action_type") != "BLOCK_IP":
            logger.warning("Action '%s' does not meet approval execution requirements.", action_id)
            return False

        now = time.time()
        # Record approval event
        self.db.insert_approval_event(
            action_id=action_id,
            operator=operator,
            decision="APPROVED",
            reason=reason or "Operator approved prevention action.",
            details={"previous_status": action.get("status")},
        )

        # Update action status
        self.db.update_prevention_action_status(
            action_id=action_id,
            status=ActionStatus.APPROVED.value,
            approved_by=operator,
        )

        # Record audit log
        self.db.insert_audit_event(
            actor=operator,
            action="MANUAL_APPROVE",
            target=action.get("target_ip", ""),
            mode=action.get("mode", PreventionMode.MANUAL_APPROVAL.value),
            result="SUCCESS",
            details={"action_id": action_id, "rule_name": action.get("rule_name"), "reason": reason},
            timestamp=now,
        )
        logger.info("Operator '%s' approved action '%s' for target %s", operator, action_id, action.get("target_ip"))
        return True

    def reject_action(
        self,
        action_id: str,
        operator: str = "Admin_Operator",
        reason: Optional[str] = None,
    ) -> bool:
        """Reject a pending prevention action."""
        action = self.db.get_prevention_action(action_id)
        if not action:
            logger.error("Cannot reject: Action ID '%s' not found.", action_id)
            return False
        if action.get("status") != ActionStatus.AWAITING_APPROVAL.value:
            return False

        now = time.time()
        # Record approval event
        self.db.insert_approval_event(
            action_id=action_id,
            operator=operator,
            decision="REJECTED",
            reason=reason or "Operator rejected prevention recommendation.",
            details={"previous_status": action.get("status")},
        )

        # Update action status
        self.db.update_prevention_action_status(
            action_id=action_id,
            status=ActionStatus.REJECTED.value,
            approved_by=operator,
        )

        # Record audit log
        self.db.insert_audit_event(
            actor=operator,
            action="MANUAL_REJECT",
            target=action.get("target_ip", ""),
            mode=action.get("mode", PreventionMode.MANUAL_APPROVAL.value),
            result="REJECTED",
            details={"action_id": action_id, "rule_name": action.get("rule_name"), "reason": reason},
            timestamp=now,
        )
        logger.info("Operator '%s' rejected action '%s' for target %s", operator, action_id, action.get("target_ip"))
        return True

    def cancel_action(
        self,
        action_id: str,
        operator: str = "Admin_Operator",
        reason: Optional[str] = None,
    ) -> bool:
        """Cancel an action before execution."""
        action = self.db.get_prevention_action(action_id)
        if not action:
            return False
        if action.get("status") != ActionStatus.AWAITING_APPROVAL.value:
            return False

        now = time.time()
        self.db.insert_approval_event(
            action_id=action_id,
            operator=operator,
            decision="CANCELLED",
            reason=reason or "Operator cancelled prevention action.",
        )
        self.db.update_prevention_action_status(
            action_id=action_id,
            status=ActionStatus.CANCELLED.value,
        )
        self.db.insert_audit_event(
            actor=operator,
            action="CANCEL_ACTION",
            target=action.get("target_ip", ""),
            mode=action.get("mode", ""),
            result="SUCCESS",
            details={"action_id": action_id, "reason": reason},
            timestamp=now,
        )
        return True
