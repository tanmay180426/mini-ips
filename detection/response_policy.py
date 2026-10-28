"""Response Policy and Threat Scoring Facade for Detection."""

from prevention.policy_engine import (
    ActionStatus,
    ActionType,
    PolicyEngine,
    PreventionAction,
    PreventionMode,
    RiskLevel,
)

# Alias for detection facade
ResponsePolicyEngine = PolicyEngine

__all__ = [
    "ActionStatus",
    "ActionType",
    "PolicyEngine",
    "PreventionAction",
    "PreventionMode",
    "ResponsePolicyEngine",
    "RiskLevel",
]
