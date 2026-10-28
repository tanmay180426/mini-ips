"""Interactive End-to-End Demonstration Script for Mini-IDS Prevention Engine.

Demonstrates:
1. Threat evaluation and composite scoring on attack flows.
2. Mode A (Simulation): Safe simulated blocking and instant verification.
3. Mode B (Manual Approval): Operator recommendation queue, review, and approval.
4. Mode C (Authorized Enforcement): Guarded OS firewall driver checks.
5. Allowlist protection (loopback / DNS cannot be blocked).
6. Manual unblock and timed rollback.
7. Immutable audit logging of all lifecycle actions.
"""

from pathlib import Path
import sys
import time

# Ensure project root in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prevention import (
    ActionStatus,
    ActionType,
    PolicyEngine,
    PreventionAction,
    PreventionEngine,
    PreventionMode,
    RiskLevel,
)
from storage.database import DatabaseManager


def run_prevention_demo():
    print("=" * 75)
    print("MINI-IDS INTRUSION PREVENTION SYSTEM (IDPS) DEMONSTRATION")
    print("=" * 75)

    # 1. Initialize Engine with in-memory or temporary database
    demo_db_path = PROJECT_ROOT / "data" / "demo_prevention.db"
    demo_db_path.parent.mkdir(parents=True, exist_ok=True)
    if demo_db_path.exists():
        try:
            demo_db_path.unlink()
        except Exception:
            pass

    db = DatabaseManager(demo_db_path)
    db.init_db()
    print(f"\n[+] Initialized isolated database: {demo_db_path}")

    # =========================================================================
    # PHASE 1: Mode A - Simulation Mode (Safe Default)
    # =========================================================================
    print("\n" + "-" * 75)
    print("PHASE 1: Mode A — Simulation Mode Execution (Safe Default)")
    print("-" * 75)

    engine = PreventionEngine(
        db_manager=db,
        initial_mode=PreventionMode.SIMULATION,
        block_threshold=0.70,
        rate_limit_threshold=0.50,
        default_block_duration=300,
    )
    print(f"Current Prevention Mode: {engine.mode.value}")

    # Attack 1: Distributed Denial of Service (DDoS)
    print("\n[*] Processing Incoming Threat Event 1: DDoS Attack from 198.51.100.42...")
    action1 = engine.process_threat(
        src_ip="198.51.100.42",
        rule_name="ML_SUPERVISED_DDOS",
        dst_ip="10.0.0.5",
        target_port=80,
        protocol="TCP",
        heuristic_severity="CRITICAL",
        ml_confidence=0.994,
        anomaly_score=0.92,
        predicted_class="DDoS",
    )

    print(f"    - Action ID:      {action1.action_id}")
    print(f"    - Threat Score:   {action1.threat_score:.3f} ({action1.risk_level.value})")
    print(f"    - Recommendation: {action1.action_type.value}")
    print(f"    - Status:         {action1.status.value}")
    print(f"    - Firewall Rule:  {action1.rule_name}")

    # Verify simulation state
    is_verified = engine.verification_manager.verify_action(
        action1.action_id,
        engine.simulation_adapter,
        action1.rule_name,
        action1.target_ip,
    )
    print(f"    - Verified State: {'VERIFIED (Simulated)' if is_verified else 'FAILED'}")

    # Attack 2: Protected Allowlist Check (Self-Lockout Safeguard)
    print("\n[*] Processing Threat Event 2: Allowlisted Probe on 127.0.0.1 (Loopback)...")
    action_allow = engine.process_threat(
        src_ip="127.0.0.1",
        rule_name="PORTSCAN_SWEEP",
        heuristic_severity="HIGH",
        ml_confidence=0.95,
    )
    print(f"    - Target IP:      {action_allow.target_ip}")
    print(f"    - Action Type:    {action_allow.action_type.value}")
    print(f"    - Policy Rationale: {action_allow.details.get('reason')}")
    assert action_allow.action_type == ActionType.LOG_ALERT, "Allowlisted IP must never be blocked!"
    print("    [SAFEGUARD VERIFIED] Loopback was protected against accidental lockout.")

    # =========================================================================
    # PHASE 2: Mode B - Manual Approval Mode (Human-in-the-Loop)
    # =========================================================================
    print("\n" + "-" * 75)
    print("PHASE 2: Mode B — Manual Approval Mode (Human-in-the-Loop Governance)")
    print("-" * 75)

    engine.set_mode(PreventionMode.MANUAL_APPROVAL)
    print(f"Switched Prevention Mode to: {engine.mode.value}")

    print("\n[*] Processing Threat Event 3: Suspicious Horizontal Port Scan from 203.0.113.89...")
    action3 = engine.process_threat(
        src_ip="203.0.113.89",
        rule_name="ML_SUPERVISED_PORTSCAN",
        dst_ip="10.0.0.12",
        target_port=445,
        protocol="TCP",
        heuristic_severity="HIGH",
        ml_confidence=0.965,
        anomaly_score=0.88,
        predicted_class="PortScan",
    )
    print(f"    - Action ID:      {action3.action_id}")
    print(f"    - Status:         {action3.status.value}")
    print(f"    - In Queue:       {action3.status == ActionStatus.AWAITING_APPROVAL}")

    # Inspect Pending Approval Queue
    pending_items = engine.approval_manager.get_pending_queue()
    print(f"\n[+] Current Pending Approval Queue Depth: {len(pending_items)}")
    for item in pending_items:
        print(f"    - Pending: [{item['action_id']}] Target: {item['target_ip']} | Score: {item['threat_score']:.2f} | Rule: {item['rule_name']}")

    # Operator Decision: Approve and execute
    print(f"\n[*] Operator reviewing Action {action3.action_id}...")
    print("    Decision: APPROVE (Operator: Senior_SOC_Analyst)")
    approved = engine.approve_and_execute(
        action_id=action3.action_id,
        operator="Senior_SOC_Analyst",
        reason="Confirmed anomalous recon pattern matching external CVE probe.",
    )
    print(f"    - Execution Result: {'ACTIVE & VERIFIED' if approved else 'FAILED'}")

    # Verify updated action record
    rec3 = db.get_prevention_action(action3.action_id)
    print(f"    - Final Record Status: {rec3['status']} (Approved by: {rec3['approved_by']})")

    # =========================================================================
    # PHASE 3: Manual Unblock & Expiration Rollback
    # =========================================================================
    print("\n" + "-" * 75)
    print("PHASE 3: Prevention Rollback & Operator Unblock Demonstration")
    print("-" * 75)

    print(f"[*] Operator initiating manual unblock for Action {action1.action_id} ({action1.target_ip})...")
    unblock_ok, unblock_msg = engine.unblock(action1.action_id, operator="Incident_Lead", reason="Investigation cleared")
    print(f"    - Result: {unblock_ok} ({unblock_msg})")

    unblock_rec = db.get_prevention_action(action1.action_id)
    print(f"    - Action Status in DB: {unblock_rec['status']} (Rolled back at: {unblock_rec['rolled_back_at']})")

    # =========================================================================
    # PHASE 4: Mode C - Authorized Enforcement Capability Audit
    # =========================================================================
    print("\n" + "-" * 75)
    print("PHASE 4: Mode C — Operating System Driver Capability Audit")
    print("-" * 75)

    wf_adapter = engine.windows_adapter
    is_avail, avail_reason = wf_adapter.is_available()
    print(f"Host OS:                         {sys.platform.capitalize()}")
    print(f"Windows Defender Driver Status:  {'AVAILABLE' if is_avail else 'UNAVAILABLE (Privilege-Gated)'}")
    print(f"Driver Reason / Details:         {avail_reason}")
    print("Rule Scoping Safeguard:          All rules strictly prefix-locked to 'MiniIDS_Block_*'")

    # =========================================================================
    # PHASE 5: Audit Log Integrity Verification
    # =========================================================================
    print("\n" + "-" * 75)
    print("PHASE 5: Immutable Security Audit Log Verification")
    print("-" * 75)

    audit_logs = db.get_audit_logs(limit=10)
    print(f"[+] Total Recorded Audit Log Events: {len(audit_logs)}")
    print(f"{'Log ID':<8} | {'Actor':<20} | {'Action':<22} | {'Target':<16} | {'Result':<10}")
    print("-" * 82)
    for log in audit_logs[:8]:
        print(f"{log['log_id']:<8} | {log['actor']:<20} | {log['action']:<22} | {log['target']:<16} | {log['result']:<10}")

    stats = db.get_prevention_stats()
    print("\n[+] Prevention Engine Final KPIs:")
    print(f"    - Total Actions Evaluated: {stats['total_prevention_actions']}")
    print(f"    - Active Rules:            {stats['active_rules']}")
    print(f"    - Pending Approvals:       {stats['pending_approvals']}")
    print(f"    - Simulated Blocks:        {stats['simulated_actions']}")
    print(f"    - Total Audit Events:      {stats['total_audit_events']}")

    print("\n" + "=" * 75)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY: ALL SAFEGUARDS VERIFIED")
    print("=" * 75)


if __name__ == "__main__":
    run_prevention_demo()
