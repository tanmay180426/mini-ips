"""Windows Defender Firewall Adapter for Mini-IDS.

Implements safe, scoped operating system firewall integration using Windows `netsh advfirewall`.
Enforces strict rule prefix scoping (`MiniIDS_Block_*`), defensive privilege verification,
and parameterized command execution to eliminate injection risks.
"""

import ctypes
import logging
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from prevention.base_adapter import BaseFirewallAdapter

logger = logging.getLogger(__name__)


def is_windows_admin() -> bool:
    """Check if the current process possesses elevated administrator privileges on Windows."""
    if platform.system().lower() != "windows":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin() != 0)
    except Exception:
        return False


class WindowsFirewallAdapter(BaseFirewallAdapter):
    """Manages Windows Defender Firewall inbound blocking rules via netsh."""

    SIMULATION_LOCKED: bool = True

    def __init__(self) -> None:
        self.is_windows = platform.system().lower() == "windows"

    def is_available(self) -> Tuple[bool, str]:
        """Verify Windows platform and administrative privilege level."""
        return False, "Host firewall adapter is disabled: Mini-IPS is operating in Simulation-Only Mode."
        # Legacy implementation retained below for reference; unreachable.
        if not self.is_windows:
            return False, "Windows Defender Firewall adapter requires a Windows host operating system."
        if not is_windows_admin():
            return (
                False,
                "Insufficient privileges: Windows firewall modification requires running Python as Administrator. "
                "Switch to Simulation Mode or elevate process permissions.",
            )
        return True, "Windows Defender Firewall adapter is available and elevated."

    def _ensure_scoped_name(self, rule_name: str) -> str:
        """Enforce strict prefix protection to prevent modifying system firewall rules."""
        if not rule_name.startswith(self.RULE_PREFIX):
            return f"{self.RULE_PREFIX}{rule_name}"
        return rule_name

    def apply_rule(
        self,
        rule_name: str,
        target_ip: str,
        protocol: Optional[str] = None,
        port: Optional[int] = None,
        duration_seconds: int = 300,
    ) -> Tuple[bool, str]:
        """Add an inbound blocking rule using parameterized netsh command."""
        raise RuntimeError("CRITICAL SAFETY GUARD: Host firewall modification is disabled in Simulation-Only Mode.")
        # Legacy implementation retained below for reference; unreachable.
        avail, reason = self.is_available()
        if not avail:
            return False, reason

        safe_name = self._ensure_scoped_name(rule_name)

        cmd = [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name={safe_name}",
            "dir=in",
            "action=block",
            f"remoteip={target_ip}",
            f"description=Automated block by Mini-IDS (expires in {duration_seconds}s)",
        ]

        if protocol and protocol.upper() in ["TCP", "UDP"]:
            cmd.append(f"protocol={protocol.upper()}")
            if port and 1 <= port <= 65535:
                cmd.append(f"localport={port}")

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                logger.info("Successfully added Windows firewall rule: %s for %s", safe_name, target_ip)
                return True, f"Windows firewall rule '{safe_name}' applied successfully."
            else:
                err_msg = res.stderr.strip() or res.stdout.strip()
                logger.error("Failed to add Windows firewall rule '%s': %s", safe_name, err_msg)
                return False, f"netsh error (code {res.returncode}): {err_msg}"
        except Exception as ex:
            logger.exception("Exception executing netsh add rule:")
            return False, f"Execution exception: {str(ex)}"

    def verify_rule(self, rule_name: str, target_ip: str) -> bool:
        """Query netsh to verify that the rule exists and is active."""
        return False
        # Legacy implementation retained below for reference; unreachable.
        if not self.is_windows:
            return False

        safe_name = self._ensure_scoped_name(rule_name)
        cmd = ["netsh", "advfirewall", "firewall", "show", "rule", f"name={safe_name}"]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and safe_name.lower() in res.stdout.lower():
                # Verify that target_ip is present in the output
                if target_ip.lower() in res.stdout.lower():
                    return True
            return False
        except Exception as ex:
            logger.debug("Verification query failed for rule '%s': %s", safe_name, ex)
            return False

    def remove_rule(self, rule_name: str) -> Tuple[bool, str]:
        """Delete an inbound blocking rule strictly scoped to MiniIDS prefix."""
        raise RuntimeError("CRITICAL SAFETY GUARD: Host firewall modification is disabled in Simulation-Only Mode.")
        # Legacy implementation retained below for reference; unreachable.
        avail, reason = self.is_available()
        if not avail:
            return False, reason

        safe_name = self._ensure_scoped_name(rule_name)
        cmd = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={safe_name}"]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                logger.info("Successfully deleted Windows firewall rule: %s", safe_name)
                return True, f"Windows firewall rule '{safe_name}' removed."
            else:
                err_msg = res.stderr.strip() or res.stdout.strip()
                return False, f"netsh delete error: {err_msg}"
        except Exception as ex:
            logger.exception("Exception deleting Windows firewall rule '%s':", safe_name)
            return False, f"Deletion exception: {str(ex)}"

    def list_active_rules(self) -> List[Dict[str, Any]]:
        """List all active Windows firewall rules created by Mini-IDS."""
        return []
        # Legacy implementation retained below for reference; unreachable.
        if not self.is_windows:
            return []

        cmd = ["netsh", "advfirewall", "firewall", "show", "rule", f"name=all"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                return []

            rules: List[Dict[str, Any]] = []
            current_rule: Dict[str, Any] = {}

            for line in res.stdout.splitlines():
                line = line.strip()
                if not line:
                    if current_rule.get("rule_name", "").startswith(self.RULE_PREFIX):
                        rules.append(current_rule)
                    current_rule = {}
                    continue

                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    if "rule name" in key:
                        current_rule["rule_name"] = val
                    elif "remoteip" in key:
                        current_rule["target_ip"] = val
                    elif "action" in key:
                        current_rule["action"] = val
                    elif "enabled" in key:
                        current_rule["enabled"] = val

            if current_rule.get("rule_name", "").startswith(self.RULE_PREFIX):
                rules.append(current_rule)

            return rules
        except Exception as ex:
            logger.debug("Failed to list active Windows firewall rules: %s", ex)
            return []
