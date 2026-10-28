"""Response Policy Engine and Threat Scoring for Mini-IDS.

Calculates composite threat scores (0.0 to 1.0), maps risk levels, enforces strict
allowlists to prevent self-lockout, guards offline PCAP data from touching host firewalls,
and generates prioritized prevention recommendations.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import ipaddress
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)
POLICY_VERSION = "stage-c-1"
VALID_DETECTION_SOURCES = {"live", "pcap", "offline", "demo", "sample"}


class PreventionMode(str, Enum):
    """Simulation execution workflow; enforcement is disabled compatibility only."""
    SIMULATION = "SIMULATION"                      # Safe default: Zero OS changes, marks SIMULATED
    MANUAL_APPROVAL = "MANUAL_APPROVAL"            # Recommendation queue: human operator decides
    # Compatibility mode only. PreventionEngine routes it to simulation; no
    # real enforcement implementation is reachable from the application.
    AUTHORIZED_ENFORCEMENT = "AUTHORIZED_ENFORCEMENT"  # Disabled compatibility value.


class ActionType(str, Enum):
    """Supported prevention action types."""
    BLOCK_IP = "BLOCK_IP"
    RATE_LIMIT = "RATE_LIMIT"
    DROP_FLOW = "DROP_FLOW"
    QUARANTINE = "QUARANTINE"
    LOG_ALERT = "LOG_ALERT"


class ActionStatus(str, Enum):
    """Lifecycle statuses; ACTIVE/VERIFIED are retained for legacy rows only."""
    RECOMMENDED = "RECOMMENDED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    ACTIVE = "ACTIVE"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    SUPPRESSED = "SUPPRESSED"
    SIMULATED = "SIMULATED"


class RiskLevel(str, Enum):
    """Threat severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Default protected IP addresses and subnets that must never be blocked
DEFAULT_ALLOWLIST: Set[str] = {
    "127.0.0.1",
    "::1",
    "localhost",
    "0.0.0.0",
    "255.255.255.255",
    "8.8.8.8",
    "8.8.4.4",
    "1.1.1.1",
    "1.0.0.1",
    "9.9.9.9",
}


@dataclass
class PreventionAction:
    """Representation of an evaluated prevention action."""
    action_id: str
    target_ip: str
    action_type: ActionType
    mode: PreventionMode
    status: ActionStatus
    threat_score: float
    risk_level: RiskLevel
    rule_name: str
    duration_seconds: int = 300
    alert_id: Optional[str] = None
    target_port: Optional[int] = None
    protocol: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    approved_by: Optional[str] = None
    executed_at: Optional[float] = None
    verified_at: Optional[float] = None
    rolled_back_at: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        d["mode"] = self.mode.value
        d["status"] = self.status.value
        d["risk_level"] = self.risk_level.value
        return d


class PolicyEngine:
    """Evaluates security detections and computes risk-calibrated response recommendations."""

    def __init__(
        self,
        default_mode: PreventionMode = PreventionMode.SIMULATION,
        block_threshold: float = 0.70,
        rate_limit_threshold: float = 0.50,
        default_block_duration: int = 300,
        custom_allowlist: Optional[List[str]] = None,
    ) -> None:
        if not (0.0 <= float(rate_limit_threshold) <= float(block_threshold) <= 1.0):
            raise ValueError("Policy thresholds must satisfy 0 <= rate_limit_threshold <= block_threshold <= 1")
        if int(default_block_duration) <= 0:
            raise ValueError("default_block_duration must be a positive integer")
        self.mode = default_mode
        self.block_threshold = block_threshold
        self.rate_limit_threshold = rate_limit_threshold
        self.default_block_duration = default_block_duration
        self.allowlist: Set[str] = set(DEFAULT_ALLOWLIST)
        if custom_allowlist:
            for item in custom_allowlist:
                normalized = str(item).strip()
                if not self._valid_ip_or_network(normalized):
                    raise ValueError(f"Invalid allowlist IP or CIDR: {item!r}")
                self.allowlist.add(normalized)

    @staticmethod
    def _valid_ip_or_network(value: str) -> bool:
        if value.lower() == "localhost":
            return True
        try:
            ipaddress.ip_network(value, strict=False) if "/" in value else ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    def add_to_allowlist(self, ip_or_cidr: str) -> None:
        """Add an IP or network CIDR to the protected allowlist."""
        clean = str(ip_or_cidr).strip()
        if not clean or not self._valid_ip_or_network(clean):
            raise ValueError("Allowlist entry must be a valid IP address or CIDR")
        self.allowlist.add(clean)

    def remove_from_allowlist(self, ip_or_cidr: str) -> None:
        """Remove an item from the allowlist (ignoring critical defaults)."""
        clean = ip_or_cidr.strip()
        if clean not in DEFAULT_ALLOWLIST and clean in self.allowlist:
            self.allowlist.remove(clean)

    def is_allowlisted(self, ip_str: Optional[str]) -> Tuple[bool, str]:
        """Check if an IP address is protected by the allowlist."""
        if not ip_str:
            return True, "Empty IP address is protected by default."

        ip_clean = ip_str.strip()

        # Check IP address object properties first
        try:
            ip_obj = ipaddress.ip_address(ip_clean)
            if ip_obj.is_loopback:
                return True, "Loopback interface (127.0.0.1 / ::1) is strictly protected."
            if ip_obj.is_multicast or ip_obj.is_reserved:
                return True, "Multicast and reserved addresses are strictly protected."

            # Check CIDR networks in allowlist
            for entry in self.allowlist:
                if "/" in entry:
                    try:
                        net = ipaddress.ip_network(entry, strict=False)
                        if ip_obj in net:
                            return True, f"IP belongs to protected network subnet {entry}."
                    except ValueError:
                        continue
        except ValueError:
            pass

        if ip_clean in self.allowlist or ip_clean.lower() == "localhost":
            return True, f"IP '{ip_clean}' matches an exact entry in the protected allowlist."

        return False, ""

    def calculate_threat_score(
        self,
        rule_name: str,
        heuristic_severity: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        predicted_class: Optional[str] = None,
    ) -> Tuple[float, RiskLevel]:
        """Compute normalized composite threat score (0.0 to 1.0) and RiskLevel."""
        for name, value in (("ml_confidence", ml_confidence), ("anomaly_score", anomaly_score)):
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        weights = {"rule": 0.40, "ml": 0.45, "anomaly": 0.15}

        # 1. Rule component
        rule_score = 0.60
        sev = (heuristic_severity or "").upper()
        if sev == "CRITICAL":
            rule_score = 1.0
        elif sev == "HIGH":
            rule_score = 0.85
        elif sev == "MEDIUM":
            rule_score = 0.60
        elif sev == "LOW":
            rule_score = 0.30

        # Known aggressive rules boost
        rule_upper = rule_name.upper()
        if "DDOS" in rule_upper:
            rule_score = max(rule_score, 0.90)
        elif "PORTSCAN" in rule_upper or "SYN" in rule_upper or "ATTACK" in rule_upper:
            rule_score = max(rule_score, 0.80)

        # 2. ML Classifier component
        ml_score = 0.0
        if ml_confidence is not None:
            pred_cls = str(predicted_class or "").upper()
            if pred_cls not in ["BENIGN", "0"]:
                ml_score = float(ml_confidence)
            else:
                ml_score = 0.05
        else:
            # Rebalance weight to rule if ML unavailable
            weights["rule"] = 0.75
            weights["ml"] = 0.0
            weights["anomaly"] = 0.25

        # 3. Anomaly component
        anom_score = 0.0
        if anomaly_score is not None:
            # Calibrated anomaly score where 1.0 is highest outlier
            anom_score = min(1.0, max(0.0, float(anomaly_score)))

        composite = (
            weights["rule"] * rule_score
            + weights["ml"] * ml_score
            + weights["anomaly"] * anom_score
        )
        composite = round(min(1.0, max(0.0, composite)), 4)

        if composite >= 0.85:
            risk = RiskLevel.CRITICAL
        elif composite >= 0.70:
            risk = RiskLevel.HIGH
        elif composite >= 0.45:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        return composite, risk

    def explain_threat_score(
        self,
        rule_name: str,
        heuristic_severity: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        predicted_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate human-understandable, component-wise breakdown of hybrid risk scoring."""
        composite, risk = self.calculate_threat_score(
            rule_name=rule_name,
            heuristic_severity=heuristic_severity,
            ml_confidence=ml_confidence,
            anomaly_score=anomaly_score,
            predicted_class=predicted_class,
        )
        has_ml = ml_confidence is not None
        has_anom = anomaly_score is not None
        weights = {"rule": 0.40, "ml": 0.45, "anomaly": 0.15}
        if not has_ml:
            weights = {"rule": 0.75, "ml": 0.0, "anomaly": 0.25}

        # Component scores
        sev = (heuristic_severity or "").upper()
        rule_score = 0.60
        if sev == "CRITICAL":
            rule_score = 1.0
        elif sev == "HIGH":
            rule_score = 0.85
        elif sev == "MEDIUM":
            rule_score = 0.60
        elif sev == "LOW":
            rule_score = 0.30

        rule_upper = rule_name.upper()
        if "DDOS" in rule_upper:
            rule_score = max(rule_score, 0.90)
        elif "PORTSCAN" in rule_upper or "SYN" in rule_upper or "ATTACK" in rule_upper:
            rule_score = max(rule_score, 0.80)

        ml_score = 0.0
        if has_ml:
            pred_cls = str(predicted_class or "").upper()
            if pred_cls not in ["BENIGN", "0"]:
                ml_score = float(ml_confidence)
            else:
                ml_score = 0.05

        anom_score = min(1.0, max(0.0, float(anomaly_score))) if has_anom else 0.0

        return {
            "rule_name": rule_name,
            "heuristic_severity": heuristic_severity or "NONE",
            "rule_score": round(rule_score, 3),
            "rule_weight": weights["rule"],
            "rule_weighted_contribution": round(weights["rule"] * rule_score, 4),
            "ml_model_active": has_ml,
            "ml_predicted_class": predicted_class or "N/A",
            "ml_confidence": round(float(ml_confidence), 4) if has_ml else None,
            "ml_score": round(ml_score, 3),
            "ml_weight": weights["ml"],
            "ml_weighted_contribution": round(weights["ml"] * ml_score, 4),
            "anomaly_model_active": has_anom,
            "anomaly_score": round(float(anomaly_score), 4) if has_anom else None,
            "anomaly_weight": weights["anomaly"],
            "anomaly_weighted_contribution": round(weights["anomaly"] * anom_score, 4),
            "composite_score": composite,
            "risk_level": risk.value,
            "block_threshold": self.block_threshold,
            "rate_limit_threshold": self.rate_limit_threshold,
            "policy_decision": (
                "BLOCK_IP" if composite >= self.block_threshold
                else ("RATE_LIMIT" if composite >= self.rate_limit_threshold else "LOG_ALERT")
            ),
        }

    def evaluate(
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
        mode_override: Optional[PreventionMode] = None,
    ) -> PreventionAction:
        """Evaluate a threat observation and return a prioritized PreventionAction."""
        effective_mode = mode_override or self.mode

        # No execution recommendation is allowed without a valid target, rule,
        # recognized provenance, and a supported severity value.
        src_ip = str(src_ip or "").strip()
        rule_name = str(rule_name or "").strip()
        severity = str(heuristic_severity or "").strip().upper()
        source = str(source_tag or "").strip().lower()
        input_errors = []
        try:
            src_ip = ipaddress.ip_address(src_ip).compressed
        except ValueError:
            input_errors.append("A valid source IP address is required.")
        if not rule_name:
            input_errors.append("A detection rule name is required.")
        if source not in VALID_DETECTION_SOURCES:
            input_errors.append("A recognized detection source is required.")
        if not isinstance(alert_id, str) or not alert_id.strip():
            input_errors.append("A source detection event ID is required.")
        if severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            input_errors.append("A recognized detection severity is required.")
        if target_port is not None and (not isinstance(target_port, int) or isinstance(target_port, bool) or not 1 <= target_port <= 65535):
            input_errors.append("Target port must be an integer between 1 and 65535.")
        if protocol is not None and str(protocol).upper() not in {"TCP", "UDP", "ICMP", "ANY"}:
            input_errors.append("Protocol must be TCP, UDP, ICMP, or ANY.")

        action_id = f"act_{uuid.uuid4().hex[:12]}"
        now = time.time()
        if input_errors:
            return PreventionAction(
                action_id=action_id, alert_id=alert_id, timestamp=now, target_ip=src_ip or "UNKNOWN",
                target_port=target_port if isinstance(target_port, int) else None, protocol=protocol,
                action_type=ActionType.LOG_ALERT, mode=PreventionMode.SIMULATION,
                status=ActionStatus.FAILED, threat_score=0.0, risk_level=RiskLevel.LOW,
                rule_name=rule_name or "UNKNOWN", duration_seconds=self.default_block_duration,
                details={"policy_version": POLICY_VERSION, "reason": "; ".join(input_errors), "validation_errors": input_errors, "source_tag": source or None},
                error_message="Policy input validation failed.",
            )

        # Scoring validation errors are also fail-closed and auditable.
        try:
            threat_score, risk_level = self.calculate_threat_score(
                rule_name, heuristic_severity=severity, ml_confidence=ml_confidence,
                anomaly_score=anomaly_score, predicted_class=predicted_class,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return PreventionAction(
                action_id=action_id, alert_id=alert_id, timestamp=now, target_ip=src_ip,
                target_port=target_port, protocol=protocol, action_type=ActionType.LOG_ALERT,
                mode=PreventionMode.SIMULATION, status=ActionStatus.FAILED, threat_score=0.0,
                risk_level=RiskLevel.LOW, rule_name=rule_name, duration_seconds=self.default_block_duration,
                details={"policy_version": POLICY_VERSION, "reason": f"Invalid scoring input: {exc}", "source_tag": source},
                error_message=f"Invalid scoring input: {exc}",
            )

        # Check offline capture safeguard:
        # PCAP or Demo traffic is strictly prohibited from touching host firewalls
        is_offline_data = str(source_tag).lower() in ["pcap", "offline", "demo", "sample"]
        if effective_mode == PreventionMode.AUTHORIZED_ENFORCEMENT:
            logger.warning("Downgrading legacy enforcement mode to SIMULATION by safety policy.")
            effective_mode = PreventionMode.SIMULATION
        elif is_offline_data:
            effective_mode = PreventionMode.SIMULATION

        threat_score, risk_level = self.calculate_threat_score(
            rule_name=rule_name,
            heuristic_severity=heuristic_severity,
            ml_confidence=ml_confidence,
            anomaly_score=anomaly_score,
            predicted_class=predicted_class,
        )

        explanation = self.explain_threat_score(
            rule_name=rule_name,
            heuristic_severity=heuristic_severity,
            ml_confidence=ml_confidence,
            anomaly_score=anomaly_score,
            predicted_class=predicted_class,
        )

        # Allowlist Protection Check
        allowlisted, allowlist_reason = self.is_allowlisted(src_ip)

        duration = self.default_block_duration

        if allowlisted:
            # Must never block allowlisted addresses
            action_type = ActionType.LOG_ALERT
            initial_status = ActionStatus.RECOMMENDED
            reason = f"ALLOWLIST PROTECTED: {allowlist_reason}"
        elif severity in {"HIGH", "CRITICAL"} and bool(alert_id) and threat_score >= self.block_threshold:
            action_type = ActionType.BLOCK_IP
            if effective_mode == PreventionMode.MANUAL_APPROVAL:
                initial_status = ActionStatus.AWAITING_APPROVAL
            else:
                initial_status = ActionStatus.RECOMMENDED
            reason = f"Detection {alert_id} has severity {severity} and threat score ({threat_score:.2f}) meeting policy threshold ({self.block_threshold:.2f})."
        elif threat_score >= self.block_threshold:
            action_type = ActionType.RATE_LIMIT
            initial_status = ActionStatus.RECOMMENDED
            reason = f"Block criteria not satisfied (requires HIGH/CRITICAL severity and source event ID); downgraded to RATE_LIMIT."
        elif threat_score >= self.rate_limit_threshold:
            action_type = ActionType.RATE_LIMIT
            initial_status = ActionStatus.RECOMMENDED
            reason = f"Threat score ({threat_score:.2f}) warrants flow rate limiting."
        else:
            action_type = ActionType.LOG_ALERT
            initial_status = ActionStatus.RECOMMENDED
            reason = f"Threat score ({threat_score:.2f}) is below active prevention threshold."

        explanation["recommendation"] = reason
        explanation["policy_decision"] = action_type.value

        action = PreventionAction(
            action_id=action_id,
            alert_id=alert_id,
            timestamp=now,
            target_ip=src_ip,
            target_port=target_port,
            protocol=protocol,
            action_type=action_type,
            mode=effective_mode,
            status=initial_status,
            threat_score=threat_score,
            risk_level=risk_level,
            rule_name=rule_name,
            duration_seconds=duration,
            expires_at=now + duration if action_type == ActionType.BLOCK_IP else None,
            details={
                "policy_version": POLICY_VERSION,
                "dst_ip": dst_ip,
                "reason": reason,
                "ml_confidence": ml_confidence,
                "anomaly_score": anomaly_score,
                "predicted_class": predicted_class,
                "source_tag": source,
                "allowlisted": allowlisted,
                "explanation": explanation,
            },
        )
        return action
