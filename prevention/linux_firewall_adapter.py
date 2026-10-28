"""Linux Firewall Adapter (iptables / nftables) for Mini-IDS.

Provides fallback and integration on Linux systems. Checks root permissions
and executes scoped iptables or nftables commands.
"""

import logging
import os
import platform
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from prevention.base_adapter import BaseFirewallAdapter

logger = logging.getLogger(__name__)


class LinuxFirewallAdapter(BaseFirewallAdapter):
    """Manages iptables/nftables blocking rules on Linux systems."""

    SIMULATION_LOCKED: bool = True

    def __init__(self) -> None:
        self.is_linux = platform.system().lower() == "linux"
        self.iptables_path = shutil.which("iptables") if self.is_linux else None

    def is_available(self) -> Tuple[bool, str]:
        """Verify Linux host and root privileges."""
        return False, "Host firewall adapter is disabled: Mini-IPS is operating in Simulation-Only Mode."
        # Legacy implementation retained below for reference; unreachable.
        if not self.is_linux:
            return False, "Linux firewall adapter requires a Linux operating system."
        if os.geteuid() != 0:
            return False, "Insufficient privileges: iptables management requires root/sudo access."
        if not self.iptables_path:
            return False, "iptables command not found in system PATH."
        return True, "Linux iptables adapter is available."

    def apply_rule(
        self,
        rule_name: str,
        target_ip: str,
        protocol: Optional[str] = None,
        port: Optional[int] = None,
        duration_seconds: int = 300,
    ) -> Tuple[bool, str]:
        """Apply iptables DROP rule with comment tag."""
        raise RuntimeError("CRITICAL SAFETY GUARD: Host firewall modification is disabled in Simulation-Only Mode.")
        # Legacy implementation retained below for reference; unreachable.
        avail, reason = self.is_available()
        if not avail:
            return False, reason

        safe_comment = f"{self.RULE_PREFIX}{rule_name}"
        cmd = [self.iptables_path, "-I", "INPUT", "-s", target_ip, "-m", "comment", "--comment", safe_comment, "-j", "DROP"]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return True, f"iptables rule added for {target_ip} ({safe_comment})"
            return False, f"iptables error: {res.stderr.strip()}"
        except Exception as ex:
            return False, f"Execution exception: {str(ex)}"

    def verify_rule(self, rule_name: str, target_ip: str) -> bool:
        """Verify rule presence in iptables INPUT chain."""
        return False
        # Legacy implementation retained below for reference; unreachable.
        if not self.is_linux or not self.iptables_path:
            return False
        cmd = [self.iptables_path, "-L", "INPUT", "-v", "-n"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return res.returncode == 0 and target_ip in res.stdout
        except Exception:
            return False

    def remove_rule(self, rule_name: str) -> Tuple[bool, str]:
        """Remove iptables rule by comment tag or target IP."""
        raise RuntimeError("CRITICAL SAFETY GUARD: Host firewall modification is disabled in Simulation-Only Mode.")
        # Legacy implementation retained below for reference; unreachable.
        avail, reason = self.is_available()
        if not avail:
            return False, reason
        return True, "iptables rule removed"

    def list_active_rules(self) -> List[Dict[str, Any]]:
        """List active Mini-IDS iptables rules."""
        return []
