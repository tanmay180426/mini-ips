"""Feature Importance, Prediction Explainability, and Defensive Guidance Module.

Provides global feature ranking for tree models and local anomaly/threat attribution
with contextual investigative recommendations for security operations analysts.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ModelExplainer:
    """Computes global feature importances and local flow threat attributions."""

    @staticmethod
    def get_global_feature_importance(model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Extract global feature importance from tree-based estimators."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            ranking = []
            for name, score in zip(feature_names, importances):
                ranking.append({
                    "feature": name,
                    "importance": float(score),
                    "importance_pct": round(float(score) * 100.0, 2),
                })
            ranking.sort(key=lambda x: x["importance"], reverse=True)
            return ranking
        return []

    @staticmethod
    def explain_flow_prediction(
        flow_features: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Identify key contributing features for a suspicious flow.

        Uses domain-specific heuristics and relative signal strengths to highlight
        the top influential metrics (e.g. abnormal SYN flag burst, tiny duration, rate surges).
        """
        explanations = []

        dur = flow_features.get("flow_duration", 0.0)
        syn = flow_features.get("syn_flag_count", 0.0)
        tot_pkts = flow_features.get("total_fwd_packets", 0.0) + flow_features.get("total_bwd_packets", 0.0)
        pkts_s = flow_features.get("flow_packets_s", 0.0)
        bytes_s = flow_features.get("flow_bytes_s", 0.0)
        avg_sz = flow_features.get("average_packet_size", 0.0)
        fwd_iat = flow_features.get("fwd_iat_mean", 0.0)

        # 1. SYN Flood / Rate Surge
        if syn > 10 and (syn / max(tot_pkts, 1)) > 0.6:
            explanations.append({
                "feature": "syn_flag_count",
                "observed": syn,
                "reason": f"Disproportionate SYN flag count ({int(syn)} SYNs, {syn/max(tot_pkts, 1)*100:.1f}% of packets). Characteristic of SYN Flood or TCP half-open scans.",
                "severity_weight": 0.9,
            })

        # 2. Port Scan / Reconnaissance indicator
        if dur < 0.1 and tot_pkts <= 4 and avg_sz < 70:
            explanations.append({
                "feature": "flow_duration_and_size",
                "observed": dur,
                "reason": f"Extremely brief connection ({dur*1000:.1f}ms) with minimal packet exchange ({int(tot_pkts)} pkts, {avg_sz:.1f} bytes). Typical port probe behavior.",
                "severity_weight": 0.8,
            })

        # 3. High packet rate volumetric surge
        if pkts_s > 100.0:
            explanations.append({
                "feature": "flow_packets_s",
                "observed": pkts_s,
                "reason": f"Elevated packet arrival rate ({pkts_s:.1f} pkts/sec). May indicate automated flood or stress testing.",
                "severity_weight": 0.85,
            })

        # 4. High byte rate bandwidth surge
        if bytes_s > 500000.0:
            explanations.append({
                "feature": "flow_bytes_s",
                "observed": bytes_s,
                "reason": f"Abnormally high data throughput ({bytes_s / 1024:.1f} KB/s). Indicator of bulk exfiltration or volumetric amplification.",
                "severity_weight": 0.75,
            })

        # 5. Low inter-arrival time
        if fwd_iat < 0.001 and tot_pkts > 10:
            explanations.append({
                "feature": "fwd_iat_mean",
                "observed": fwd_iat,
                "reason": f"Sub-millisecond inter-arrival times ({fwd_iat * 1000:.3f} ms). Typical of scripted automation rather than human interactive sessions.",
                "severity_weight": 0.7,
            })

        # Default fallback if no specific rule hit
        if not explanations:
            # Highlight top non-zero metrics
            for k, v in sorted(flow_features.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]:
                if v > 0:
                    explanations.append({
                        "feature": k,
                        "observed": v,
                        "reason": f"Observed feature value: {v:.2f}.",
                        "severity_weight": 0.5,
                    })

        explanations.sort(key=lambda x: x["severity_weight"], reverse=True)
        return explanations[:top_k]

    @staticmethod
    def get_defensive_recommendations(alert_rule_or_class: str) -> List[str]:
        """Provide prioritized defensive recommendations based on detection category."""
        target = alert_rule_or_class.upper()
        if "PORT" in target or "SCAN" in target:
            return [
                "Verify whether source IP is an authorized internal vulnerability scanner or monitoring sensor.",
                "Inspect host firewall logs to determine whether closed ports dropped or rejected probes (RST).",
                "Consider implementing dynamic rate limiting or blackholing via Fail2ban / iptables / Windows Defender Firewall.",
                "Audit target host for unnecessary listening services exposed to external networks.",
            ]
        elif "SYN" in target or "DOS" in target or "FLOOD" in target:
            return [
                "Verify whether target application or web server experienced service degradation or TCP pool exhaustion.",
                "Enable TCP SYN Cookies on target endpoints (`netsh int tcp set global rss=enabled` / `sysctl net.ipv4.tcp_syncookies=1`).",
                "Check for asymmetric spoofed source addresses; verify reverse path forwarding (uRPF) on border routers.",
                "Engage upstream DDoS scrubbing or web application firewall (WAF) rate limiters.",
            ]
        elif "BRUTE" in target:
            return [
                "Inspect authentication logs for target service (e.g. SSH, RDP, HTTP basic auth) for multiple consecutive 401/failure events.",
                "Enforce multi-factor authentication (MFA) and account lockout policies after repeated failures.",
                "Change default ports or restrict administrative interface access via VPN / IP whitelisting.",
            ]
        elif "VOLUME" in target or "ANOMALY" in target:
            return [
                "Correlate with scheduled backup jobs, database migrations, or large batch file transfers.",
                "Analyze destination endpoints and protocols to rule out data exfiltration channels.",
                "Verify network interface throughput baselines during equivalent time of day.",
            ]
        else:
            return [
                "Isolate affected endpoint to prevent lateral movement if malicious intent is suspected.",
                "Collect full packet capture and process memory dump for forensic analysis.",
                "Cross-reference source IP against threat intelligence reputation databases.",
            ]
