"""Abstract Base Firewall Adapter for Mini-IDS Prevention Engine.

Defines the contract for host and network firewall enforcement adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseFirewallAdapter(ABC):
    """Abstract interface for OS-level or network-level firewall drivers."""

    RULE_PREFIX = "MiniIDS_Block_"

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """Check if the adapter is supported on this OS and has necessary privileges."""
        pass

    @abstractmethod
    def apply_rule(
        self,
        rule_name: str,
        target_ip: str,
        protocol: Optional[str] = None,
        port: Optional[int] = None,
        duration_seconds: int = 300,
    ) -> Tuple[bool, str]:
        """Apply a blocking rule for the target IP address.

        Returns:
            Tuple[bool, str]: (Success boolean, Status message or error description)
        """
        pass

    @abstractmethod
    def verify_rule(self, rule_name: str, target_ip: str) -> bool:
        """Verify that the rule is actively enforced in the firewall table."""
        pass

    @abstractmethod
    def remove_rule(self, rule_name: str) -> Tuple[bool, str]:
        """Remove a previously applied blocking rule.

        Returns:
            Tuple[bool, str]: (Success boolean, Status message)
        """
        pass

    @abstractmethod
    def list_active_rules(self) -> List[Dict[str, Any]]:
        """Query and return currently active Mini-IDS firewall rules."""
        pass
