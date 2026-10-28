"""Mini-IDS Intrusion Prevention System (IPS) Subsystem.

Provides simulation-only prevention execution and manual approval for simulated actions,
composite threat scoring, and inert compatibility exports for legacy firewall adapters,
human approval workflows, state verification, automatic expiration rollback, and persistent audit logging.
"""

from prevention.approval_manager import ApprovalManager
from prevention.base_adapter import BaseFirewallAdapter
from prevention.engine import PreventionEngine
from prevention.linux_firewall_adapter import LinuxFirewallAdapter
from prevention.policy_engine import (
    ActionStatus,
    ActionType,
    PolicyEngine,
    PreventionAction,
    PreventionMode,
    RiskLevel,
)
from prevention.rollback import RollbackManager
from prevention.simulation_adapter import SimulationAdapter
from prevention.verification import VerificationManager
from prevention.windows_firewall_adapter import WindowsFirewallAdapter

__all__ = [
    "PreventionEngine",
    "PolicyEngine",
    "PreventionMode",
    "ActionType",
    "ActionStatus",
    "RiskLevel",
    "PreventionAction",
    "BaseFirewallAdapter",
    "SimulationAdapter",
    "WindowsFirewallAdapter",
    "LinuxFirewallAdapter",
    "ApprovalManager",
    "VerificationManager",
    "RollbackManager",
]
