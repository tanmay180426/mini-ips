"""Firewall Enforcement and Adapter Facade."""

from prevention.base_adapter import BaseFirewallAdapter, ExecutionResult
from prevention.engine import PreventionEngine
from prevention.linux_firewall_adapter import LinuxFirewallAdapter
from prevention.windows_firewall_adapter import WindowsFirewallAdapter

__all__ = [
    "BaseFirewallAdapter",
    "ExecutionResult",
    "PreventionEngine",
    "WindowsFirewallAdapter",
    "LinuxFirewallAdapter",
]
