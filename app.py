"""Mini-IDS Intelligent Network Security Analytics Platform.

Streamlit web application combining heuristic intrusion detection, supervised
machine learning classification, unsupervised Isolation Forest anomaly detection,
and exploratory cybersecurity traffic analytics.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
import sys
import sklearn
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from analytics.exploratory_analysis import DatasetProfiler
from analytics.theme import SEVERITY_COLORS, THEME_COLORS, get_altair_theme, get_custom_css
from capture.live_capture import filter_primary_interfaces, get_available_interfaces, get_live_worker
from ml.dataset_manager import DATASET_METADATA, DatasetManager
from ml.evaluation import ModelEvaluator
from ml.model_registry import ModelRegistry
from ml.preprocessing import FlowDataPreprocessor
from ml.train_anomaly import AnomalyDetectorTrainer
from ml.train_supervised import SupervisedModelTrainer
from prevention import PreventionEngine, PreventionMode
from storage.database import DEFAULT_DB_PATH, DatabaseManager
from storage.demo_data import DEFAULT_DEMO_DB_PATH, get_or_create_demo_database

logger = logging.getLogger("mini_ids.dashboard")
PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: Union[str, Path]) -> Path:
    """Resolve configured relative paths against this repository, not the shell CWD."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_timestamp(ts: Union[float, int, str, None]) -> str:
    """Format UNIX epoch float to a readable datetime string (UTC)."""
    if ts is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


def build_severity_chart(severity_data: List[Dict[str, Any]]) -> Optional[alt.Chart]:
    """Construct an Altair horizontal bar chart for alert severity distribution."""
    if not severity_data:
        return None
    df = pd.DataFrame(severity_data)
    color_scale = alt.Scale(
        domain=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        range=[
            THEME_COLORS["critical"],
            THEME_COLORS["high"],
            THEME_COLORS["warning"],
            THEME_COLORS["low"],
        ],
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("count:Q", title="Alert Count"),
            y=alt.Y("severity:N", sort=["CRITICAL", "HIGH", "MEDIUM", "LOW"], title="Severity Tier"),
            color=alt.Color("severity:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("count:Q", title="Alert Count", format=","),
            ],
        )
        .properties(height=180)
    )
    return chart


def build_rule_chart(rule_data: List[Dict[str, Any]]) -> Optional[alt.Chart]:
    """Construct an Altair horizontal bar chart for alerts by detection rule."""
    if not rule_data:
        return None
    df = pd.DataFrame(rule_data)
    chart = (
        alt.Chart(df)
        .mark_bar(color=THEME_COLORS["primary_accent"], cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("count:Q", title="Alert Count"),
            y=alt.Y("rule_name:N", sort="-x", title="Detection Rule"),
            tooltip=[
                alt.Tooltip("rule_name:N", title="Rule"),
                alt.Tooltip("count:Q", title="Alert Count", format=","),
            ],
        )
        .properties(height=180)
    )
    return chart


def build_timeline_chart(
    timeline_data: List[Dict[str, Any]],
    color: str = THEME_COLORS["primary_accent"],
    fill_color: str = "#E0835D",
    y_field: str = "count",
    y_title: str = "Alert Count",
) -> Optional[alt.Chart]:
    """Construct an Altair discrete column bar chart for event timeline aggregation."""
    if not timeline_data:
        return None
    df = pd.DataFrame(timeline_data)
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("bucket:N", sort=None, title="Time Bucket (UTC)"),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("bucket:N", title="Interval (UTC)"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=","),
            ],
        )
        .properties(height=200)
    )
    return chart


def build_ranking_chart(
    data: List[Dict[str, Any]],
    category_field: str,
    value_field: str,
    category_title: str,
    value_title: str,
    bar_color: str = THEME_COLORS["secondary_accent"],
) -> Optional[alt.Chart]:
    """Construct a horizontal bar chart for ranked entities (IPs, Ports)."""
    if not data:
        return None
    df = pd.DataFrame(data)
    df[category_field] = df[category_field].astype(str)
    chart = (
        alt.Chart(df)
        .mark_bar(color=bar_color, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(f"{value_field}:Q", title=value_title),
            y=alt.Y(f"{category_field}:N", sort="-x", title=category_title),
            tooltip=[
                alt.Tooltip(f"{category_field}:N", title=category_title),
                alt.Tooltip(f"{value_field}:Q", title=value_title, format=","),
            ],
        )
        .properties(height=180)
    )
    return chart


def _detection_mode_from_records(alerts: List[Dict[str, Any]], model_probe_succeeded: bool = False) -> Tuple[str, str]:
    """Classify persisted detector evidence without inferring a hybrid mode from dependencies alone."""
    methods = [str(alert.get("detection_method") or "HEURISTIC").upper() for alert in alerts]
    has_ml_output = any(
        method.startswith("ML_") or method == "HYBRID"
        or alert.get("model_version") is not None
        or alert.get("confidence") is not None
        or alert.get("anomaly_score") is not None
        for alert, method in zip(alerts, methods)
    )
    has_rule_output = any(method == "HEURISTIC" or method == "HYBRID" for method in methods)
    if has_ml_output and has_rule_output:
        return "HYBRID", "Persisted alerts contain both heuristic and ML-attributed detections."
    if has_ml_output:
        return "ML_READY", "Persisted alerts include ML output attributed to a model or ML detection method."
    if has_rule_output:
        return "RULE_ONLY", "Persisted alerts are rule-based; no ML output is attributed to them."
    if model_probe_succeeded:
        return "ML_READY", "An active model completed inference on a persisted flow; no detections are stored yet."
    return "ML_UNAVAILABLE", "No persisted detector output or successful active-model inference is available."


def _probe_active_model(registry: ModelRegistry, model_type: str, latest_flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Load an active bundle and, when possible, infer on actual persisted flow features."""
    result: Dict[str, Any] = {
        "model_type": model_type,
        "bundle_status": "UNAVAILABLE",
        "inference_status": "NOT_RUN",
        "reason": "No active model bundle is available.",
        "version": None,
        "algorithm": None,
        "feature_count": None,
        "schema_compatible": False,
    }
    try:
        bundle = registry.load_active_bundle(model_type)
        if not bundle:
            return result
        model, preprocessor, metadata = bundle
        result.update({
            "bundle_status": "LOADED",
            "version": metadata.get("version"),
            "algorithm": metadata.get("algorithm"),
            "feature_count": len(metadata.get("feature_schema") or []),
        })
        expected = list(metadata.get("feature_schema") or [])
        fitted = list(getattr(preprocessor, "feature_names_", []) or [])
        if not expected or expected != fitted:
            result["bundle_status"] = "INCOMPATIBLE"
            result["reason"] = "Stored metadata and fitted preprocessor feature order do not match."
            return result
        result["schema_compatible"] = True
        if not latest_flow:
            result["reason"] = "Bundle loaded and schema matched; no persisted flow is available for inference."
            return result
        feature_payload = latest_flow.get("features_json")
        if not feature_payload:
            result["reason"] = "Bundle loaded; latest persisted flow has no feature payload."
            return result
        features = json.loads(feature_payload) if isinstance(feature_payload, str) else feature_payload
        transformed = preprocessor.transform_inference(features)
        expected_width = getattr(model, "n_features_in_", len(expected))
        if transformed.shape != (1, expected_width) or transformed.shape[1] != len(expected):
            raise ValueError(f"Feature shape {transformed.shape} does not match the fitted {expected_width}-input model.")
        if model_type.startswith("anomaly_"):
            if not callable(getattr(model, "score_samples", None)):
                raise TypeError("Active anomaly model does not expose score_samples().")
            model.score_samples(transformed)
        else:
            if not callable(getattr(model, "predict", None)):
                raise TypeError("Active supervised model does not expose predict().")
            model.predict(transformed)
        result["inference_status"] = "SUCCESS"
        result["reason"] = "Inference completed on the latest persisted flow; anomaly scores are not calibrated probabilities."
    except Exception as exc:
        result["inference_status"] = "FAILED"
        result["reason"] = f"Model load or inference failed: {exc}"
    return result


def _incident_counts(db: DatabaseManager, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    states = db.get_alert_investigation_states([a["id"] for a in alerts if a.get("id") is not None])
    open_count = sum(states.get(int(a["id"]), {}).get("status", "OPEN") != "RESOLVED" for a in alerts if a.get("id") is not None)
    return {"open": open_count, "states": states}


def render_alert_investigation_workspace(
    db: DatabaseManager, alerts: List[Dict[str, Any]], is_demo_mode: bool = False,
) -> None:
    """Render searchable, paginated incident triage using stored alerts and audit-backed notes."""
    st.title("🔍 Alert Investigation & Incident Triage")
    st.caption("Investigate persisted detections, packet evidence, model attribution, and audit-backed analyst notes.")
    if not alerts:
        st.info("No alerts exist in this database. Load the isolated synthetic demo database or ingest a PCAP to create persisted events.")
        return

    left, middle, right = st.columns([2, 1, 1])
    with left:
        query = st.text_input("Search alerts", placeholder="Alert ID, rule, host, or message", key="incident_search").strip().lower()
    with middle:
        severity_filter = st.selectbox("Severity", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"], key="incident_severity")
    with right:
        status_filter = st.selectbox("Incident status", ["All", "OPEN", "TRIAGED", "RESOLVED"], key="incident_status")

    states = db.get_alert_investigation_states([int(a["id"]) for a in alerts if a.get("id") is not None])
    filtered = []
    for alert in alerts:
        alert_id = int(alert["id"])
        state = states.get(alert_id, {"status": "OPEN", "notes": ""})
        haystack = " ".join(str(alert.get(key) or "") for key in ("id", "rule_name", "src_ip", "dst_ip", "message")).lower()
        if query and query not in haystack:
            continue
        if severity_filter != "All" and str(alert.get("severity", "")).upper() != severity_filter:
            continue
        if status_filter != "All" and state["status"] != status_filter:
            continue
        filtered.append(alert)

    sort_order = st.selectbox("Sort alerts", ["Newest first", "Oldest first", "Highest severity first"], key="incident_sort")
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    if sort_order == "Oldest first":
        filtered.sort(key=lambda a: (float(a.get("timestamp") or 0), int(a.get("id") or 0)))
    elif sort_order == "Highest severity first":
        filtered.sort(key=lambda a: (severity_rank.get(str(a.get("severity", "")).upper(), 4), -float(a.get("timestamp") or 0)))
    else:
        filtered.sort(key=lambda a: (float(a.get("timestamp") or 0), int(a.get("id") or 0)), reverse=True)

    page_size = st.selectbox("Rows per page", [10, 25, 50], index=1, key="incident_page_size")
    page_count = max(1, (len(filtered) + int(page_size) - 1) // int(page_size))
    page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1, key="incident_page")
    page_rows = filtered[(int(page) - 1) * int(page_size): int(page) * int(page_size)]
    st.caption(f"Showing {len(page_rows)} of {len(filtered)} matching alerts (page {int(page)} of {page_count}).")
    if not page_rows:
        st.info("No alerts match the selected search and filters.")
        return

    labels = {
        int(a["id"]): f"#{a['id']} · {a.get('severity', 'UNKNOWN')} · {a.get('rule_name', 'UNKNOWN')} · {a.get('src_ip') or 'N/A'} → {a.get('dst_ip') or 'N/A'} · {format_timestamp(a.get('timestamp'))}"
        for a in page_rows
    }
    selected_id = st.selectbox("Select alert", list(labels), format_func=lambda value: labels[value], key="incident_selected")
    alert = db.get_alert_by_id(int(selected_id))
    if alert is None:
        st.error("The selected alert no longer exists in the active database.")
        return

    state = db.get_alert_investigation_state(int(selected_id))
    st.markdown(f"### Alert #{selected_id} · {alert.get('rule_name', 'UNKNOWN')}")
    st.markdown(
        f"**Severity:** `{alert.get('severity', 'UNAVAILABLE')}` · **Status:** `{state['status']}` · "
        f"**Method:** `{alert.get('detection_method') or 'UNAVAILABLE'}` · **Time:** `{format_timestamp(alert.get('timestamp'))}`"
    )
    st.write(alert.get("message") or "No alert description was stored.")
    evidence_cols = st.columns(4)
    evidence_cols[0].metric("Source IP", alert.get("src_ip") or "Unavailable")
    evidence_cols[1].metric("Destination IP", alert.get("dst_ip") or "Unavailable")
    evidence_cols[2].metric("Observed / Threshold", f"{float(alert.get('observed_value') or 0):g} / {float(alert.get('threshold') or 0):g}")
    evidence_cols[3].metric("Window", f"{float(alert.get('window_seconds') or 0):g} s")
    st.caption(
        f"Model version: {alert.get('model_version') or 'Unavailable'} · "
        f"Confidence: {alert.get('confidence') if alert.get('confidence') is not None else 'Unavailable'} · "
        f"Anomaly score: {alert.get('anomaly_score') if alert.get('anomaly_score') is not None else 'Unavailable'} · "
        f"Threat score: {alert.get('threat_score') if alert.get('threat_score') is not None else 'Unavailable'}"
    )
    if alert.get("prevention_action_id"):
        st.caption(f"Linked prevention action: `{alert['prevention_action_id']}`")
    related_actions = [a for a in db.get_prevention_actions(limit=None) if str(a.get("alert_id")) == str(selected_id)]
    if related_actions:
        st.markdown("#### Prevention decisions for this alert")
        st.dataframe(pd.DataFrame([{
            "Action ID": a.get("action_id"), "Decision": a.get("action_type"), "Status": a.get("status"),
            "Reason": (a.get("details") or {}).get("reason", a.get("error_message") or "Unavailable"),
            "Expires": format_timestamp(a.get("expires_at")),
        } for a in related_actions]), width="stretch")

    alert_ts = float(alert.get("timestamp") or 0)
    window = float(alert.get("window_seconds") or 0)
    packets = db.get_related_packets(
        src_ip=alert.get("src_ip"), dst_ip=alert.get("dst_ip"),
        start_time=alert_ts - window - 1, end_time=alert_ts + 1, limit=200, order="ASC",
    )
    st.markdown("#### Stored packet evidence")
    if packets:
        st.dataframe(pd.DataFrame(packets), width="stretch")
    else:
        st.info("No matching packet evidence was found in the stored detection interval.")

    related = [a for a in alerts if int(a.get("id") or -1) != int(selected_id) and (
        (alert.get("src_ip") and a.get("src_ip") == alert.get("src_ip"))
        or (alert.get("dst_ip") and a.get("dst_ip") == alert.get("dst_ip"))
    )]
    st.markdown("#### Related stored alerts")
    if related:
        st.dataframe(pd.DataFrame([{
            "Alert ID": a.get("id"), "Time": format_timestamp(a.get("timestamp")), "Rule": a.get("rule_name"),
            "Severity": a.get("severity"), "Source IP": a.get("src_ip"), "Destination IP": a.get("dst_ip"),
        } for a in related[:20]]), width="stretch")
    else:
        st.caption("No other stored alerts share this source or destination address.")

    st.markdown("#### Analyst notes and disposition")
    notes = st.text_area("Investigation notes", value=state.get("notes", ""), max_chars=10000, key=f"incident_notes_{selected_id}")
    save_col, triage_col, resolve_col, reopen_col = st.columns(4)
    if save_col.button("Save notes", key=f"save_notes_{selected_id}"):
        if db.update_alert_investigation(int(selected_id), state["status"], notes, actor="DASHBOARD_ANALYST"):
            st.success("Notes saved to the audit history.")
            st.rerun()
    if triage_col.button("Mark triaged", disabled=state["status"] == "RESOLVED", key=f"triage_{selected_id}"):
        db.update_alert_investigation(int(selected_id), "TRIAGED", notes, actor="DASHBOARD_ANALYST")
        st.rerun()
    if resolve_col.button("Resolve incident", disabled=state["status"] == "RESOLVED", key=f"resolve_{selected_id}"):
        db.update_alert_investigation(int(selected_id), "RESOLVED", notes, actor="DASHBOARD_ANALYST")
        st.rerun()
    if reopen_col.button("Reopen", disabled=state["status"] != "RESOLVED", key=f"reopen_{selected_id}"):
        db.update_alert_investigation(int(selected_id), "OPEN", notes, actor="DASHBOARD_ANALYST")
        st.rerun()
    if state.get("updated_at") is not None:
        st.caption(f"Last investigation update: {format_timestamp(state['updated_at'])} by {state.get('actor') or 'Unknown'}")


def render_live_capture_section(target_db_path: Path, is_demo_mode: bool = False) -> None:
    """Render the Live Packet Capture control panel and real-time worker status."""
    st.markdown("### 📡 Live Packet Capture (Passive Sniffer)")

    if is_demo_mode:
        st.info(
            "☁️ **Live Packet Capture is disabled in Hosted Demonstration Mode.**\n\n"
            "Passive live packet sniffing requires direct physical network adapter access, administrative "
            "privileges, and local Npcap kernel drivers on a host workstation. In this hosted cloud "
            "demonstration, the dashboard operates in safe, read-only forensic mode.\n\n"
            "To perform real-time packet capture on your local Wi-Fi or Ethernet adapter, run Mini-IDS "
            "locally on Windows using:\n"
            "```powershell\n"
            "streamlit run app.py\n"
            "```"
        )
        st.divider()
        return

    st.caption(
        "Capture live traffic from local network adapters using Npcap. Packets are evaluated "
        "against detection rules and saved directly to the active SQLite database."
    )

    worker = get_live_worker()
    status = worker.get_status()

    # Capture status banner
    if status["status"] == "STOPPING":
        st.warning(
            f"⏳ **Stopping capture on `{status['interface']}`... finalizing queue and database writes.**\n\n"
            f"**Packets Captured:** {status['packets_captured']:,} | "
            f"**Persisted to DB:** {status['packets_persisted']:,} | "
            f"**Queue Drops:** {status['packets_dropped']:,} | "
            f"**Alerts Triggered:** {status['alerts_generated']:,}"
        )
    elif status["is_running"]:
        st.info(
            f"🟢 **Live Capture Active on `{status['interface']}`**\n\n"
            f"**Packets Captured:** {status['packets_captured']:,} | "
            f"**Persisted to DB:** {status['packets_persisted']:,} | "
            f"**Queue Drops:** {status['packets_dropped']:,} | "
            f"**Alerts Triggered:** {status['alerts_generated']:,} | "
            f"**Elapsed Time:** {status['elapsed_seconds']:.1f}s"
        )
    elif status["status"] == "COMPLETED":
        st.success(
            f"✅ **Capture Completed on `{status['interface']}`**\n\n"
            f"Captured: **{status['packets_captured']:,}** packets | "
            f"Persisted: **{status['packets_persisted']:,}** | "
            f"Queue Drops: **{status['packets_dropped']:,}** | "
            f"Alerts Triggered: **{status['alerts_generated']:,}** | "
            f"Duration: **{status['elapsed_seconds']:.1f}s**"
        )
    elif status["status"] == "STOPPED":
        st.warning(
            f"⏹️ **Capture Stopped on `{status['interface']}`**\n\n"
            f"Captured: **{status['packets_captured']:,}** packets | "
            f"Persisted: **{status['packets_persisted']:,}** | "
            f"Queue Drops: **{status['packets_dropped']:,}** | "
            f"Alerts Triggered: **{status['alerts_generated']:,}** | "
            f"Duration: **{status['elapsed_seconds']:.1f}s**"
        )
    elif status["status"] == "ERROR":
        st.error(f"❌ **Capture Error:** {status.get('error_message')}")
    elif status["status"] == "IDLE":
        st.caption("ℹ️ **Sniffer Status:** `IDLE` — Ready to capture. Select an interface in the configuration panel below and click **Start Capture**.")

    with st.expander("⚙️ Capture Configuration & Controls", expanded=status["is_running"] or status["status"] == "IDLE"):
        raw_ifaces = get_available_interfaces()

        if not raw_ifaces:
            st.warning(
                "⚠️ **No network interfaces discovered.**\n\n"
                "Ensure Npcap is installed on Windows with 'WinPcap API-compatible Mode' enabled. "
                "Administrative privileges may be required to access raw network adapters."
            )
            return

        # Keep 3-4 clean primary options by default, with optional checkbox for virtual filter drivers
        col_iface_title, col_iface_toggle = st.columns([3, 1])
        with col_iface_toggle:
            show_all_ifaces = st.checkbox("Show all virtual adapters", value=False, help="Display all virtual WFP filters and tunnel adapters")

        available_ifaces = raw_ifaces if show_all_ifaces else filter_primary_interfaces(raw_ifaces, max_count=4)

        iface_labels = [i["display_name"] for i in available_ifaces]
        selected_label = st.radio(
            "Select Network Interface",
            options=iface_labels,
            index=0,
            help="Choose the network adapter to monitor. Passive sniffing only; traffic is not injected or altered.",
            key="live_capture_iface_select",
        )
        selected_iface = next((i for i in available_ifaces if i["display_name"] == selected_label), available_ifaces[0])

        col_cond, col_val = st.columns(2)
        with col_cond:
            stop_condition = st.radio(
                "Stop Condition",
                options=["Manual Stop", "Duration Timeout", "Packet Count Limit"],
                index=0,
                horizontal=True,
                key="live_capture_condition_type",
            )
        with col_val:
            duration_limit: Optional[float] = None
            packet_limit: Optional[int] = None
            if stop_condition == "Duration Timeout":
                duration_limit = float(
                    st.number_input(
                        "Timeout (Seconds)",
                        min_value=1.0,
                        max_value=3600.0,
                        value=15.0,
                        step=5.0,
                        key="live_capture_duration_val",
                    )
                )
            elif stop_condition == "Packet Count Limit":
                packet_limit = int(
                    st.number_input(
                        "Max Packets",
                        min_value=5,
                        max_value=100000,
                        value=100,
                        step=25,
                        key="live_capture_packet_val",
                    )
                )

        btn_start, btn_stop, btn_refresh = st.columns([1, 1, 2])
        with btn_start:
            can_start = not status["is_running"]
            if st.button("▶️ Start Capture", disabled=not can_start, type="primary", key="live_btn_start"):
                iface_id = selected_iface["name"] if selected_iface["name"] else selected_iface["guid"]
                try:
                    worker.configure(
                        interface=iface_id,
                        db_path=target_db_path,
                        packet_limit=packet_limit,
                        duration_seconds=duration_limit,
                    )
                    worker.start()
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed to start live capture: {err}")
        with btn_stop:
            can_stop = status["is_running"]
            if st.button("⏹️ Stop Capture", disabled=not can_stop, key="live_btn_stop"):
                worker.stop()
                st.rerun()
        with btn_refresh:
            if st.button("🔄 Refresh Sniffer Status", key="live_btn_refresh"):
                st.rerun()


def render_prevention_center(db: DatabaseManager, is_demo_mode: bool = False) -> None:
    """Render the Intrusion Prevention Center workspace."""
    st.title("🛡️ Intrusion Prevention Center (IDPS)")
    st.caption("Detection response workflow with policy review and simulated actions. No host firewall rules are changed.")
    st.warning("Simulation only: every action is a recorded recommendation. No network traffic is blocked and no host firewall is changed.")
    previous_result = st.session_state.pop("last_demo_prevention_result", None)
    if previous_result:
        st.info(previous_result)

    if is_demo_mode:
        demo_alerts = db.get_alerts(limit=100, order="DESC")
        if demo_alerts:
            st.markdown("### Evaluate a stored demonstration detection")
            alert_labels = {
                str(a["id"]): f"#{a['id']} · {a.get('severity')} · {a.get('rule_name')} · {a.get('src_ip') or 'N/A'}"
                for a in demo_alerts if a.get("id") is not None
            }
            if alert_labels:
                chosen_demo_alert = st.selectbox("Stored demo alert", list(alert_labels), format_func=lambda key: alert_labels[key])
                if st.button("Evaluate simulation proposal", key="evaluate_demo_prevention"):
                    selected = next(a for a in demo_alerts if str(a.get("id")) == chosen_demo_alert)
                    engine = PreventionEngine(db_manager=db, initial_mode=PreventionMode.SIMULATION)
                    result = engine.process_threat(
                        src_ip=selected.get("src_ip") or "",
                        dst_ip=selected.get("dst_ip"),
                        rule_name=selected.get("rule_name") or "",
                        alert_id=str(selected["id"]),
                        heuristic_severity=selected.get("severity"),
                        ml_confidence=selected.get("confidence"),
                        anomaly_score=selected.get("anomaly_score"),
                        source_tag="demo",
                    )
                    st.session_state["last_demo_prevention_result"] = (
                        f"Policy evaluation persisted for alert {selected['id']}: {result.status.value} · {result.action_type.value}. "
                        "This is a simulation-only decision; it does not mean traffic was blocked."
                    )
                    st.rerun()

    # Manage session prevention mode
    if "prevention_mode" not in st.session_state:
        st.session_state["prevention_mode"] = PreventionMode.SIMULATION.value
    if st.session_state["prevention_mode"] == PreventionMode.AUTHORIZED_ENFORCEMENT.value:
        st.session_state["prevention_mode"] = PreventionMode.SIMULATION.value

    current_mode_val = st.session_state["prevention_mode"]
    current_mode = PreventionMode(current_mode_val)
    engine = PreventionEngine(db_manager=db, initial_mode=current_mode)

    # 1. Prevention Mode Switcher & Operational Posture
    st.markdown("### 🎛️ Prevention Operational Posture")
    mode_options = [
        PreventionMode.SIMULATION.value,
        PreventionMode.MANUAL_APPROVAL.value,
    ]
    mode_labels = {
        PreventionMode.SIMULATION.value: "🟢 Mode A: Simulation (Zero OS modifications, Safe Default)",
        PreventionMode.MANUAL_APPROVAL.value: "🟡 Mode B: Manual Approval (Operator Queue & Human-in-the-Loop, Simulated Action)",
    }

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        new_mode_selection = st.radio(
            "Select Prevention Execution Mode:",
            options=mode_options,
            format_func=lambda m: mode_labels[m],
            index=mode_options.index(current_mode_val),
            horizontal=False,
            key="prevention_mode_radio",
        )
        if new_mode_selection != current_mode_val:
            st.session_state["prevention_mode"] = new_mode_selection
            engine.set_mode(PreventionMode(new_mode_selection))
            st.rerun()

    with col_m2:
        adapter = engine.get_active_adapter()
        avail, reason = adapter.is_available()
        st.markdown("**Active Driver:**")
        if avail:
            st.success(f"**{adapter.__class__.__name__}**\n\n{reason}")
        else:
            st.warning(f"**{adapter.__class__.__name__}**\n\n{reason}")

    # Mode Context Banner
    if current_mode == PreventionMode.SIMULATION:
        st.info(
            "ℹ️ **Simulation Mode Active:** Detections generate response recommendations and in-memory simulation records. "
            "Host firewall tables are **never** altered. Safe for demonstrations, student evaluations, and non-admin systems."
        )
    elif current_mode == PreventionMode.MANUAL_APPROVAL:
        st.warning(
            "⚠️ **Manual Approval Mode Active:** High-risk threats enter the pending approval queue. An authorized security operator "
            "must explicitly review and approve before an in-memory simulation record is created. Network traffic is never blocked."
        )

    # 2. Prevention Metrics KPIs
    stats = db.get_prevention_stats()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Active Simulations", f"{stats['active_rules']:,}")
    k2.metric("Pending Approvals", f"{stats['pending_approvals']:,}")
    k3.metric("Simulated Actions", f"{stats['simulated_actions']:,}")
    k4.metric("Host Firewall", "Disabled")
    k5.metric("Audit Events", f"{stats['total_audit_events']:,}")

    st.markdown("---")

    # 3. Pending Approvals Queue
    st.markdown("### 📋 Pending Approval Queue (Human-in-the-Loop)")
    pending_queue = engine.approval_manager.get_pending_queue()
    if not pending_queue:
        st.success("✅ No pending approval requests in queue. All detection policies are current.")
    else:
        st.caption(f"{len(pending_queue)} item(s) awaiting security operator review.")
        for item in pending_queue:
            with st.container(border=True):
                c_info, c_action = st.columns([3, 1])
                with c_info:
                    st.markdown(
                        f"**Target IP:** `{item['target_ip']}` | "
                        f"**Rule:** `{item['rule_name']}` | "
                        f"**Threat Score:** `{item['threat_score']:.2f}` ({item['risk_level']}) | "
                        f"**Time:** {format_timestamp(item['timestamp'])}"
                    )
                    details = item.get("details", {})
                    reason = details.get("reason", "Threshold exceeded")
                    st.caption(f"**Rationale:** {reason} | Protocol: {item.get('protocol') or 'ANY'} | Port: {item.get('target_port') or 'ALL'}")
                with c_action:
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("Approve", key=f"app_{item['action_id']}", type="primary"):
                            engine.approve_and_execute(item["action_id"], operator="SOC_Operator")
                            st.toast(f"Approved simulated response for {item['target_ip']}")
                            st.rerun()
                    with col_b2:
                        if st.button("Reject", key=f"rej_{item['action_id']}"):
                            engine.reject(item["action_id"], operator="SOC_Operator", reason="Operator dismissed")
                            st.toast(f"Rejected recommendation for {item['target_ip']}")
                            st.rerun()

    st.markdown("---")

    # 4. Active Prevention Rules Table
    st.markdown("### 🧪 Active Simulations & Revoke Controls")
    active_actions = db.get_prevention_actions(limit=100)
    if not active_actions:
        st.info("No prevention actions currently recorded in the database.")
    else:
        table_rows = []
        for a in active_actions:
            table_rows.append({
                "Action ID": a["action_id"],
                "Target IP": a["target_ip"],
                "Action": a["action_type"],
                "Mode": a["mode"],
                "Status": a["status"],
                "Source Alert": a.get("alert_id") or "Unavailable",
                "Policy Reason": (a.get("details") or {}).get("reason", "Unavailable"),
                "Failure": a.get("error_message") or "",
                "Threat Score": f"{a['threat_score']:.2f}",
                "Risk": a["risk_level"],
                "Trigger Rule": a["rule_name"],
                "Timestamp": format_timestamp(a["timestamp"]),
                "Expires": format_timestamp(a.get("expires_at")),
            })
        st.dataframe(pd.DataFrame(table_rows), width="stretch")

        active_block_ids = [a["action_id"] for a in active_actions if a["status"] in ["SIMULATED", "ACTIVE", "VERIFIED"]]
        if active_block_ids:
            with st.expander("↩️ Revoke Simulation", expanded=False):
                col_sel, col_unb = st.columns([3, 1])
                with col_sel:
                    selected_unblock_id = st.selectbox("Select Simulation:", options=active_block_ids)
                with col_unb:
                    st.write("")
                    st.write("")
                    if st.button("Revoke Simulation", type="secondary"):
                        ok, msg = engine.unblock(selected_unblock_id, operator="SOC_Operator")
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

    st.markdown("---")

    # 5. Security Safeguards & Allowlist Panel
    with st.expander("🛡️ Security Safeguards & Protected Allowlists", expanded=False):
        st.markdown(
            "**Self-Lockout Prevention:** The Mini-IDS policy engine strictly verifies that the following IP categories "
            "can never receive a prevention simulation, regardless of threat scores or alerts:\n\n"
            "- **Loopback:** `127.0.0.1`, `::1`, `localhost`\n"
            "- **Network Infrastructure:** `0.0.0.0`, `255.255.255.255`\n"
            "- **Core DNS Resolvers:** `8.8.8.8`, `8.8.4.4`, `1.1.1.1`, `1.0.0.1`, `9.9.9.9`\n"
            "- **Configured Subnets:** Local gateway and designated operator addresses\n\n"
            "**Offline Traffic Containment:** All uploaded PCAP captures and synthetic demonstration datasets are strictly "
            "prohibited from triggering host firewall modifications. They run exclusively in Simulation mode."
        )
        if st.button("Clear All Active Simulations"):
            count = engine.shutdown()
            st.success(f"Cleared {count} in-memory simulation records. No network traffic was blocked.")
            st.rerun()


def render_audit_log_page(db: DatabaseManager) -> None:
    """Render SQLite security and prevention audit records."""
    st.title("📜 Security & Prevention Audit Log")
    st.caption("SQLite audit records for security decisions, policy evaluations, approval workflows, and simulated action state changes.")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        actor_filter = st.selectbox("Filter by Actor:", options=["ALL", "POLICY_ENGINE", "SIMULATION_ENGINE", "VERIFICATION_ENGINE", "SYSTEM_OPERATOR", "EXPIRATION_MONITOR", "DASHBOARD_ANALYST"])
    with f_col2:
        action_filter = st.selectbox("Filter by Action:", options=["ALL", "POLICY_EVALUATION", "RECORD_SIMULATION", "VERIFY_SIMULATION_RECORD", "MANUAL_APPROVE", "MANUAL_REJECT", "REVOKE_SIMULATION", "EXPIRE_SIMULATION", "CLEAR_SIMULATION_RECORD", "CHANGE_PREVENTION_MODE", "UPDATE_ALERT_INVESTIGATION"])
    with f_col3:
        mode_filter = st.selectbox("Filter by Mode:", options=["ALL", "SIMULATION", "MANUAL_APPROVAL", "INVESTIGATION"])
    with f_col4:
        result_filter = st.selectbox("Filter by Result:", options=["ALL", "SUCCESS", "FAILURE", "REJECTED", "OPEN", "TRIAGED", "RESOLVED"])

    actor_val = None if actor_filter == "ALL" else actor_filter
    action_val = None if action_filter == "ALL" else action_filter
    mode_val = None if mode_filter == "ALL" else mode_filter
    result_val = None if result_filter == "ALL" else result_filter

    audit_records = db.get_audit_logs(
        limit=200,
        actor=actor_val,
        action=action_val,
        mode=mode_val,
        result=result_val,
    )

    if not audit_records:
        st.info("No audit log events found matching the specified filters.")
        return

    st.markdown(f"**Showing {len(audit_records)} audit event(s):**")
    table_data = []
    for r in audit_records:
        details = r.get("details") if isinstance(r.get("details"), dict) else {}
        nested = details.get("details") if isinstance(details.get("details"), dict) else {}
        table_data.append({
            "Log ID": r["log_id"],
            "Timestamp (UTC)": format_timestamp(r["timestamp"]),
            "Actor": r["actor"],
            "Action": r["action"],
            "Target": r["target"],
            "Mode": r["mode"],
            "Result": r["result"],
            "Alert ID": details.get("alert_id", nested.get("alert_id")),
            "Action ID": details.get("action_id", nested.get("action_id")),
            "Reason": details.get("reason", nested.get("reason", "")),
            "Policy Version": details.get("policy_version", nested.get("policy_version", "")),
        })
    st.dataframe(pd.DataFrame(table_data), width="stretch")

    with st.expander("🔍 Deep Forensic Inspection (JSON Payload)", expanded=False):
        log_ids = [r["log_id"] for r in audit_records]
        selected_log_id = st.selectbox("Select Log ID to inspect:", options=log_ids)
        selected_record = next((r for r in audit_records if r["log_id"] == selected_log_id), None)
        if selected_record:
            st.json(selected_record)


def render_dashboard() -> None:
    """Main Streamlit application entrypoint."""
    default_db_path = _project_path(os.environ.get("MINI_IDS_DB_PATH", str(DEFAULT_DB_PATH)))
    st.set_page_config(
        page_title="Mini-IDS: Security Analytics Platform",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply global custom CSS design tokens
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Sidebar Header
    st.sidebar.title("🛡️ MINI-IDS")
    st.sidebar.caption("Intelligent Network Intrusion Detection & Analytics Platform")

    # Primary Configuration Input (Must remain sidebar.text_input[0] for tests)
    db_path_input = st.sidebar.text_input(
        "Database Path",
        value=str(default_db_path),
        help="Path to SQLite database containing stored packets, alerts, and model records.",
    )

    # Refresh Button (Must remain sidebar.button[0] for tests)
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Synthetic telemetry is opt-in so an empty local database remains explicit.
    default_demo_active = os.environ.get("MINI_IDS_DEMO_MODE", "").lower() in ("1", "true", "yes")

    demo_toggle = st.sidebar.toggle(
        "Demonstration Mode (Synthetic)",
        value=default_demo_active,
        help="Enable sanitized synthetic demonstration dataset for evaluation without local capture files.",
        key="app_demo_toggle",
    )

    # Navigation Sections (Sidebar Radio [0])
    nav_sections = [
        "Overview",
        "Live Monitoring",
        "Prevention Center",
        "Audit Log",
        "Alert Investigation",
        "Traffic Intelligence",
        "AI/ML Detection",
        "Model Laboratory",
        "Data Explorer",
        "Detection Rules",
        "System & Settings",
    ]
    selected_page = st.sidebar.radio(
        "Navigation",
        options=nav_sections,
        index=0,
        help="Switch between security monitoring, investigation, machine learning, and data science workspaces.",
        key="app_navigation_radio",
    )

    # Resolve active database
    raw_path = _project_path(db_path_input.strip() or default_db_path)
    custom_path_provided = bool(raw_path != default_db_path)
    is_demo_mode = demo_toggle and not custom_path_provided

    if is_demo_mode:
        demo_path = _project_path(os.environ.get("MINI_IDS_DEMO_DB_PATH", str(DEFAULT_DEMO_DB_PATH)))
        target_db_path = get_or_create_demo_database(demo_path)
        st.sidebar.info("🧪 **Operational Mode:** Demo Mode (Synthetic)")
        st.sidebar.caption(
            "Using deterministic synthetic demonstration dataset with RFC 5737 documentation IPs. "
            "Real captures and local databases are not accessed."
        )
    else:
        target_db_path = raw_path
        st.sidebar.caption(f"📁 **Active Database:** `{target_db_path}`")

    # Check database existence BEFORE opening connection
    if not is_demo_mode and not target_db_path.is_file():
        st.warning(f"Database file not found: `{target_db_path}`")
        st.info(
            "**No database exists at the specified path.**\n\n"
            "The Mini-IDS dashboard does not automatically create empty database files. "
            "To generate data, run offline analysis from your terminal:\n\n"
            "```powershell\n"
            "python main.py --pcap data/sample.pcap\n"
            "```\n\n"
            "Or toggle **Demonstration Mode (Synthetic)** in the sidebar to explore simulated telemetry."
        )
        return

    # Demonstration Mode top notification banner
    if is_demo_mode:
        st.info(
            "🧪 **Demonstration Mode Active** — Displaying sanitized synthetic network telemetry and simulated intrusion "
            "alerts for evaluation and mentor review. All events are synthetic and explicitly labeled. "
            "No real network packets, private host data, or live packet captures are active in this mode."
        )

    # Database connection & retrieval
    try:
        with DatabaseManager(target_db_path) as db:
            packet_count = db.get_packet_count()
            alert_count = db.get_alert_count()

            # Global Model Resolution (Resolves active models before page dispatch)
            registry = ModelRegistry(models_dir=_project_path("data/models"), db_path=target_db_path)
            try:
                active_sup = db.get_active_model("supervised_binary") or db.get_active_model()
            except Exception:
                active_sup = None
            try:
                active_anom = db.get_active_model("anomaly_isolation_forest")
            except Exception:
                active_anom = None

            all_alerts = db.get_alerts(order="DESC") if alert_count > 0 else []

            # Sidebar Filters (Preserves exact selectbox & checkbox order for tests)
            st.sidebar.header("Filter Options")
            st.sidebar.subheader("Alert Filters")
            st.sidebar.caption("Applies to Alert Analytics and Security Alerts.")
            available_rules = sorted(list({a["rule_name"] for a in all_alerts if a.get("rule_name")}))
            available_severities = sorted(list({a["severity"] for a in all_alerts if a.get("severity")}))

            selected_rule = st.sidebar.selectbox(
                "Detection Rule",
                options=["All"] + available_rules,
                index=0,
            )
            selected_severity = st.sidebar.selectbox(
                "Severity",
                options=["All"] + available_severities,
                index=0,
            )
            alert_limit = st.sidebar.selectbox(
                "Alerts Display Limit",
                options=[25, 50, 100, 250, "All"],
                index=1,
            )

            st.sidebar.subheader("Traffic Filters")
            st.sidebar.caption("Applies to Network Traffic Analytics.")
            packet_limit = st.sidebar.selectbox(
                "Recent Packets Limit",
                options=[50, 100, 250, 500, 1000],
                index=1,
            )
            selected_protocol = st.sidebar.selectbox(
                "Protocol",
                options=["All", "TCP", "UDP", "ICMP"],
                index=0,
            )

            st.sidebar.subheader("Analysis Scope")
            bounds = db.get_timestamp_bounds()
            min_ts = bounds.get("min_timestamp")
            max_ts = bounds.get("max_timestamp")

            enable_time_filter = st.sidebar.checkbox(
                "Filter by Time Range",
                value=False,
                help="Isolate records to a specific capture interval without modifying the database.",
            )
            scope_start_ts: Optional[float] = None
            scope_end_ts: Optional[float] = None
            if enable_time_filter and min_ts is not None and max_ts is not None:
                st.sidebar.caption(f"Stored bounds: {format_timestamp(min_ts)} — {format_timestamp(max_ts)}")
                scope_start_ts = st.sidebar.number_input(
                    "Start Timestamp (Epoch s)",
                    value=float(min_ts),
                    step=1.0,
                    format="%.2f",
                )
                scope_end_ts = st.sidebar.number_input(
                    "End Timestamp (Epoch s)",
                    value=float(max_ts),
                    step=1.0,
                    format="%.2f",
                )

            # Apply scope filtering
            active_alerts = all_alerts
            if enable_time_filter and scope_start_ts is not None and scope_end_ts is not None:
                active_alerts = [
                    a for a in active_alerts
                    if scope_start_ts <= float(a.get("timestamp", 0.0)) <= scope_end_ts
                ]

            fetch_limit = int(packet_limit) * 2 if selected_protocol != "All" else int(packet_limit)
            if enable_time_filter and scope_start_ts is not None and scope_end_ts is not None:
                raw_packets = db.get_related_packets(
                    start_time=scope_start_ts,
                    end_time=scope_end_ts,
                    protocol=selected_protocol if selected_protocol != "All" else None,
                    limit=fetch_limit,
                    order="DESC",
                )
            else:
                raw_packets = db.get_packets(limit=fetch_limit, order="DESC")

            # Filtered alerts for table and investigation
            filtered_alerts = active_alerts
            if selected_rule != "All":
                filtered_alerts = [a for a in filtered_alerts if a.get("rule_name") == selected_rule]
            if selected_severity != "All":
                filtered_alerts = [a for a in filtered_alerts if a.get("severity") == selected_severity]
            if alert_limit != "All":
                filtered_alerts = filtered_alerts[: int(alert_limit)]

            kpis = db.get_security_kpis(
                start_time=scope_start_ts if enable_time_filter else None,
                end_time=scope_end_ts if enable_time_filter else None,
            )

            # ------------------------------------------------------------------
            # PAGE 1: OVERVIEW (Default Page)
            # ------------------------------------------------------------------
            if selected_page == "Overview":
                render_live_capture_section(target_db_path, is_demo_mode=is_demo_mode)

                if packet_count == 0 and alert_count == 0 and db.get_flow_count() == 0:
                    st.info(
                        f"Connected to database `{target_db_path}`, but it contains no stored packets or alerts.\n\n"
                        "Run an offline PCAP analysis to populate this database:\n\n"
                        "```powershell\n"
                        f"python main.py --pcap <path_to_pcap> --db \"{target_db_path}\"\n"
                        "```"
                    )
                else:
                    if enable_time_filter and scope_start_ts is not None and scope_end_ts is not None:
                        st.subheader("Security Overview (Scoped to Selected Interval)")
                        st.caption(
                            f"Showing metrics between `{format_timestamp(scope_start_ts)}` and `{format_timestamp(scope_end_ts)}`. "
                            "Uncheck 'Filter by Time Range' in sidebar to view cumulative totals across all stored history."
                        )
                    else:
                        st.subheader("Stored Totals (Cumulative Across All Analyses)")
                        st.caption(
                            "Cumulative metrics reflect the complete database state across all ingested PCAP runs and live captures. "
                            "Historical PCAP data and recent captures are indexed in SQLite."
                        )

                    # Primary KPI Row
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                    kpi_col1.metric("Total Stored Packets", f"{kpis['total_packets']:,}")
                    kpi_col2.metric("Total Stored Alerts", f"{kpis['total_alerts']:,}")
                    kpi_col3.metric("High-Severity Alerts", f"{kpis['high_severity_alerts']:,}")
                    kpi_col4.metric("Distinct Alert Sources", f"{kpis['distinct_alert_sources']:,}")

                    # Rule Counts Row (preserves compatibility with test assertions)
                    rule_counts = Counter(a["rule_name"] for a in all_alerts if a.get("rule_name"))
                    if rule_counts:
                        st.markdown("**Alert Counts Grouped by Rule / Type:**")
                        rule_columns = st.columns(max(len(rule_counts), 1))
                        for col, (rule_name, count) in zip(rule_columns, sorted(rule_counts.items())):
                            col.metric(rule_name, f"{count:,}")

                    # Model Readiness Banner
                    registry = ModelRegistry(models_dir=_project_path("data/models"), db_path=target_db_path)
                    active_sup = db.get_active_model("supervised_binary") or db.get_active_model()
                    active_anom = db.get_active_model("anomaly_isolation_forest")
                    flow_rows = db.get_flows(limit=1)
                    latest_flow = flow_rows[0] if flow_rows else None
                    sup_probe = _probe_active_model(registry, "supervised_binary", latest_flow)
                    anom_probe = _probe_active_model(registry, "anomaly_isolation_forest", latest_flow)
                    detection_mode, detection_reason = _detection_mode_from_records(
                        all_alerts,
                        sup_probe["inference_status"] == "SUCCESS" or anom_probe["inference_status"] == "SUCCESS",
                    )

                    st.markdown("---")
                    mod_c1, mod_c2 = st.columns(2)
                    with mod_c1:
                        if sup_probe["inference_status"] == "SUCCESS":
                            st.success(
                                f"🤖 **Supervised inference verified:** `{sup_probe['algorithm']}` ({sup_probe['version']})"
                            )
                        else:
                            st.warning(f"ℹ️ **Supervised ML unavailable:** {sup_probe['reason']}")
                    with mod_c2:
                        if anom_probe["inference_status"] == "SUCCESS":
                            st.success(
                                f"🌲 **Anomaly inference verified:** `{anom_probe['algorithm']}` ({anom_probe['version']})"
                            )
                        else:
                            st.warning(f"ℹ️ **Anomaly ML unavailable:** {anom_probe['reason']}")
                    st.caption(f"Runtime mode: **{detection_mode}** · {detection_reason}")
                    st.markdown("### Processing and prevention snapshot")
                    overview_flows = db.get_flow_count()
                    overview_incidents = _incident_counts(db, all_alerts)
                    prevention_stats = db.get_prevention_stats()
                    actions = db.get_prevention_actions(limit=None)
                    proposed_count = sum(a.get("status") in {"RECOMMENDED", "AWAITING_APPROVAL"} for a in actions)
                    overview_cols = st.columns(5)
                    overview_cols[0].metric("Flows Processed", f"{overview_flows:,}")
                    overview_cols[1].metric("Open / Triaged Incidents", f"{overview_incidents['open']:,}")
                    overview_cols[2].metric("Prevention Proposals", f"{proposed_count:,}")
                    overview_cols[3].metric("Simulated Actions", f"{prevention_stats['simulated_actions']:,}")
                    capture_status = get_live_worker().get_status()
                    overview_cols[4].metric("Capture", "RUNNING" if capture_status.get("is_running") else "STOPPED")

                    source_distribution = db.get_detection_method_distribution()
                    if source_distribution:
                        st.markdown("#### Detection source distribution")
                        source_frame = pd.DataFrame(source_distribution).set_index("detection_method")
                        st.bar_chart(source_frame["count"])
                    recent_events = all_alerts[:5]
                    st.markdown("#### Recent security events")
                    if recent_events:
                        for event in recent_events:
                            st.markdown(
                                f"- `{format_timestamp(event.get('timestamp'))}` · **{event.get('severity', 'UNKNOWN')}** · "
                                f"{event.get('rule_name', 'UNKNOWN')} · {event.get('src_ip') or 'N/A'} → "
                                f"{event.get('dst_ip') or 'N/A'} · {event.get('detection_method') or 'HEURISTIC'}"
                            )
                    else:
                        st.caption("No persisted alert events are available.")

                    st.divider()

                    # Analytics Tabs
                    st.subheader("📊 Security Analytics")
                    tab_alerts, tab_traffic = st.tabs(["🚨 Alert Analytics", "🌐 Network Traffic Analytics"])

                    with tab_alerts:
                        st.caption(f"Showing alert analytics filtered by Rule: **{selected_rule}** | Severity: **{selected_severity}**.")
                        dist_col1, dist_col2 = st.columns(2)
                        with dist_col1:
                            st.markdown("##### Alert Severity Distribution")
                            sev_data = db.get_alert_severity_distribution(
                                start_time=scope_start_ts if enable_time_filter else None,
                                end_time=scope_end_ts if enable_time_filter else None,
                                rule=selected_rule,
                            )
                            sev_chart = build_severity_chart(sev_data)
                            if sev_chart is not None:
                                st.altair_chart(sev_chart, width="stretch")
                            else:
                                st.info("No alert severity records available.")

                        with dist_col2:
                            st.markdown("##### Alerts by Detection Rule")
                            rule_data = db.get_alerts_by_rule(
                                start_time=scope_start_ts if enable_time_filter else None,
                                end_time=scope_end_ts if enable_time_filter else None,
                                severity=selected_severity,
                            )
                            rule_chart = build_rule_chart(rule_data)
                            if rule_chart is not None:
                                st.altair_chart(rule_chart, width="stretch")
                            else:
                                st.info("No detection rule records available.")

                        st.markdown("##### Alert Timeline (UTC)")
                        alert_timeline_data = db.get_alert_timeline(
                            start_time=scope_start_ts if enable_time_filter else None,
                            end_time=scope_end_ts if enable_time_filter else None,
                            rule=selected_rule,
                            severity=selected_severity,
                        )
                        alert_tl_chart = build_timeline_chart(alert_timeline_data)
                        if alert_tl_chart is not None:
                            st.altair_chart(alert_tl_chart, width="stretch")

                        rank_col1, rank_col2 = st.columns(2)
                        with rank_col1:
                            st.markdown("##### Top Offending Source IPs")
                            top_sources = db.get_top_alert_sources(
                                start_time=scope_start_ts if enable_time_filter else None,
                                end_time=scope_end_ts if enable_time_filter else None,
                                rule=selected_rule,
                                severity=selected_severity,
                            )
                            src_chart = build_ranking_chart(
                                top_sources,
                                category_field="src_ip",
                                value_field="count",
                                category_title="Source IP",
                                value_title="Alert Count",
                                bar_color=THEME_COLORS["critical"],
                            )
                            if src_chart is not None:
                                st.altair_chart(src_chart, width="stretch")

                        with rank_col2:
                            st.markdown("##### Top Targeted Destination IPs")
                            top_dests = db.get_top_alert_destinations(
                                start_time=scope_start_ts if enable_time_filter else None,
                                end_time=scope_end_ts if enable_time_filter else None,
                                rule=selected_rule,
                                severity=selected_severity,
                            )
                            dst_chart = build_ranking_chart(
                                top_dests,
                                category_field="dst_ip",
                                value_field="count",
                                category_title="Destination IP",
                                value_title="Alert Count",
                                bar_color=THEME_COLORS["primary_accent"],
                            )
                            if dst_chart is not None:
                                st.altair_chart(dst_chart, width="stretch")

                    with tab_traffic:
                        st.caption(f"Showing raw packet traffic filtered by Protocol: **{selected_protocol}**.")
                        pcol1, pcol2 = st.columns(2)
                        with pcol1:
                            st.markdown("##### Protocol Distribution")
                            proto_data = db.get_protocol_distribution(
                                start_time=scope_start_ts if enable_time_filter else None,
                                end_time=scope_end_ts if enable_time_filter else None,
                                protocol=selected_protocol if selected_protocol != "All" else None,
                            )
                            p_chart = build_ranking_chart(
                                proto_data,
                                category_field="protocol",
                                value_field="count",
                                category_title="Protocol",
                                value_title="Packet Count",
                                bar_color=THEME_COLORS["secondary_accent"],
                            )
                            if p_chart is not None:
                                st.altair_chart(p_chart, width="stretch")

                        with pcol2:
                            st.markdown("##### Top Destination Ports")
                            port_data = db.get_top_destination_ports(
                                start_time=scope_start_ts if enable_time_filter else None,
                                end_time=scope_end_ts if enable_time_filter else None,
                                protocol=selected_protocol if selected_protocol != "All" else None,
                            )
                            port_chart = build_ranking_chart(
                                port_data,
                                category_field="dst_port",
                                value_field="packet_count",
                                category_title="Destination Port",
                                value_title="Packet Count",
                                bar_color=THEME_COLORS["primary_accent"],
                            )
                            if port_chart is not None:
                                st.altair_chart(port_chart, width="stretch")

                    st.divider()

                    # Recent Security Alerts Table (Must be dataframe[0] for tests)
                    st.subheader("Recent Security Alerts")
                    if not filtered_alerts:
                        st.info("No alerts match the selected filter criteria.")
                    else:
                        alert_rows = []
                        for a in filtered_alerts:
                            alert_rows.append({
                                "Alert ID": a.get("id"),
                                "Timestamp": format_timestamp(a.get("timestamp")),
                                "Rule": a.get("rule_name", "UNKNOWN"),
                                "Severity": a.get("severity", "UNKNOWN"),
                                "Detection Method": a.get("detection_method", "HEURISTIC"),
                                "Source IP": a.get("src_ip") or "N/A",
                                "Destination IP": a.get("dst_ip") or "N/A",
                                "Observed": round(float(a.get("observed_value", 0.0)), 2),
                                "Threshold": round(float(a.get("threshold", 0.0)), 2),
                                "Window (s)": round(float(a.get("window_seconds", 0.0)), 1),
                                "Description / Reason": a.get("message", ""),
                            })
                        df_alerts = pd.DataFrame(alert_rows)
                        st.dataframe(df_alerts, width="stretch")

                    st.divider()

                    # Alert Investigation Workflow (Must have selectbox[0] and dataframe[1] for tests)
                    st.subheader("🔍 Alert Investigation")
                    if not filtered_alerts:
                        st.info("No alerts available to investigate with the current filter criteria.")
                    else:
                        alert_id_to_label = {
                            a["id"]: (
                                f"Alert #{a['id']} | {a.get('rule_name', 'UNKNOWN')} [{a.get('severity', 'UNKNOWN')}] | "
                                f"{a.get('src_ip') or 'N/A'} -> {a.get('dst_ip') or 'N/A'} | "
                                f"{format_timestamp(a.get('timestamp'))}"
                            )
                            for a in filtered_alerts
                        }
                        selected_alert_id = st.selectbox(
                            "Select Alert to Investigate",
                            options=list(alert_id_to_label.keys()),
                            format_func=lambda aid: alert_id_to_label[aid],
                            key="overview_alert_select",
                        )
                        selected_alert = db.get_alert_by_id(selected_alert_id)
                        if selected_alert:
                            st.markdown(f"#### Alert Details — ID #{selected_alert['id']}")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Detection Rule", selected_alert.get("rule_name", "UNKNOWN"))
                            c2.metric("Severity", selected_alert.get("severity", "UNKNOWN"))
                            c3.metric("Observed / Threshold", f"{selected_alert.get('observed_value', 0):.2f} / {selected_alert.get('threshold', 0):.2f}")
                            c4.metric("Window Duration", f"{selected_alert.get('window_seconds', 0):.1f} s")

                            # Explainable Hybrid Risk Scoring Breakdown
                            t_score = selected_alert.get("threat_score")
                            prev_id = selected_alert.get("prevention_action_id")
                            if t_score is not None or prev_id:
                                action_rec = db.get_prevention_action(prev_id) if prev_id else None
                                expl = (action_rec.get("details", {}).get("explanation")) if action_rec else None
                                with st.expander("🧠 Explainable Threat Scoring & Prevention Rationale", expanded=True):
                                    if t_score is not None:
                                        st.markdown(f"**Composite Hybrid Threat Score:** `{t_score:.2f}` / 1.00")
                                    if action_rec:
                                        st.markdown(f"**Simulated Prevention Action:** `{action_rec.get('action_type')}` on `{action_rec.get('target_ip')}` (Status: `{action_rec.get('status')}`) — *SIMULATION ONLY*")
                                    if expl:
                                        st.markdown(f"**Summary Rationale:** {expl.get('summary', 'Threat evaluated against security policy.')}")
                                        e_cols = st.columns(3)
                                        e_cols[0].metric("Rule Component", f"{expl.get('rule_score', 0):.2f} (weight {expl.get('rule_weight', 0):.2f})")
                                        ml_status = f"{expl.get('ml_confidence', 0):.2f} [{expl.get('ml_class', 'N/A')}]" if expl.get("ml_active") else "Model Offline (0.00)"
                                        e_cols[1].metric("ML Component", ml_status)
                                        anom_status = f"{expl.get('anomaly_score', 0):.2f}" if expl.get("anomaly_active") else "Anomaly Offline (0.00)"
                                        e_cols[2].metric("Anomaly Component", anom_status)
                                        st.caption(f"**Policy Decision:** `{expl.get('decision')}` | **Risk Tier:** `{expl.get('risk_level')}` | **Recommendation:** {expl.get('recommendation')}")

                            alert_ts = float(selected_alert.get("timestamp", 0.0))
                            win_sec = float(selected_alert.get("window_seconds", 0.0))
                            effective_start = alert_ts - win_sec
                            effective_end = alert_ts

                            evidence_packets = db.get_related_packets(
                                src_ip=selected_alert.get("src_ip"),
                                dst_ip=selected_alert.get("dst_ip"),
                                start_time=effective_start,
                                end_time=effective_end,
                                limit=500,
                                order="ASC",
                            )

                            st.markdown("#### Related Packet Evidence")
                            if not evidence_packets:
                                st.warning(
                                    f"No matching packet records found for "
                                    f"{selected_alert.get('src_ip') or 'any'} -> {selected_alert.get('dst_ip') or 'any'} "
                                    f"within the detection interval `[{format_timestamp(effective_start)} — {format_timestamp(effective_end)}]`."
                                )
                            else:
                                st.success(f"Found **{len(evidence_packets)}** related packet record(s) matching detection window.")
                                evidence_rows = []
                                for p in evidence_packets:
                                    evidence_rows.append({
                                        "Timestamp": format_timestamp(p.get("timestamp")),
                                        "Epoch": f"{p.get('timestamp', 0.0):.6f}",
                                        "Protocol": p.get("protocol", ""),
                                        "Source IP": p.get("src_ip", ""),
                                        "Destination IP": p.get("dst_ip", ""),
                                        "Source Port": str(p["src_port"]) if p.get("src_port") is not None else "-",
                                        "Destination Port": str(p["dst_port"]) if p.get("dst_port") is not None else "-",
                                        "Length (Bytes)": p.get("length", 0),
                                        "TCP Flags": p.get("tcp_flags") if p.get("tcp_flags") is not None else "-",
                                    })
                                df_evidence = pd.DataFrame(evidence_rows)
                                st.dataframe(df_evidence, width="stretch")

                    st.divider()

                    # Recent Network Traffic Table (Must be dataframe[2] for tests)
                    st.subheader("Recent Network Traffic")
                    filtered_packets = raw_packets
                    if selected_protocol != "All":
                        filtered_packets = [
                            p for p in filtered_packets
                            if p.get("protocol", "").upper() == selected_protocol.upper()
                        ]
                    filtered_packets = filtered_packets[: int(packet_limit)]

                    if not filtered_packets:
                        st.info("No packets match the selected traffic filters.")
                    else:
                        packet_rows = []
                        for p in filtered_packets:
                            packet_rows.append({
                                "Timestamp": format_timestamp(p.get("timestamp")),
                                "Source IP": p.get("src_ip", ""),
                                "Destination IP": p.get("dst_ip", ""),
                                "Protocol": p.get("protocol", ""),
                                "Length (Bytes)": p.get("length", 0),
                                "Source Port": str(p["src_port"]) if p.get("src_port") is not None else "-",
                                "Destination Port": str(p["dst_port"]) if p.get("dst_port") is not None else "-",
                                "TCP Flags": p.get("tcp_flags") if p.get("tcp_flags") is not None else "-",
                            })
                        df_packets = pd.DataFrame(packet_rows)
                        st.dataframe(df_packets, width="stretch")

            # ------------------------------------------------------------------
            # PAGE 2: LIVE MONITORING
            # ------------------------------------------------------------------
            elif selected_page == "Live Monitoring":
                render_live_capture_section(target_db_path, is_demo_mode=is_demo_mode)

                st.subheader("Real-Time Event Stream")
                recent_live_alerts = db.get_alerts(limit=25, order="DESC")
                if not recent_live_alerts:
                    st.info("No security alerts triggered during current live capture session.")
                else:
                    st.dataframe(pd.DataFrame(recent_live_alerts), width="stretch")

            # ------------------------------------------------------------------
            # PAGE 3: PREVENTION CENTER (IDPS Prevention & Governance)
            # ------------------------------------------------------------------
            elif selected_page == "Prevention Center":
                render_prevention_center(db, is_demo_mode=is_demo_mode)

            # ------------------------------------------------------------------
            # PAGE 4: AUDIT LOG (Persisted Event Records)
            # ------------------------------------------------------------------
            elif selected_page == "Audit Log":
                render_audit_log_page(db)

            # ------------------------------------------------------------------
            # PAGE 3: ALERT INVESTIGATION (Dedicated Forensic Workspace)
            # ------------------------------------------------------------------
            elif selected_page == "Alert Investigation":
                render_alert_investigation_workspace(db, all_alerts, is_demo_mode=is_demo_mode)

            # ------------------------------------------------------------------
            # PAGE 4: TRAFFIC INTELLIGENCE
            # ------------------------------------------------------------------
            elif selected_page == "Traffic Intelligence":
                st.title("🌐 Network Traffic Intelligence & Flow Analytics")
                st.caption("Inspect discrete time-bucket volumes, flow distributions, and protocol proportions.")

                capture_state = get_live_worker().get_status()
                live_state = "RUNNING" if capture_state.get("is_running") else "STOPPED"
                st.info(
                    f"Capture worker: **{live_state}**. Active database: `{target_db_path}`. "
                    "Stored packet records may come from live capture or offline PCAP ingestion; source provenance is unavailable unless recorded by the ingest workflow."
                )

                ti_col1, ti_col2, ti_col3 = st.columns(3)
                ti_col1.metric("Total Observed Packets", f"{packet_count:,}")
                ti_col2.metric("Total Stored Alerts", f"{alert_count:,}")
                ti_col3.metric("Indexed Network Flows", f"{db.get_flow_count():,}")

                # Alerts are not packet-level labels, so do not estimate a packet ratio.
                st.markdown("---")
                st.markdown("### Observed traffic and detections")
                st.caption("Packet and alert totals are separate persisted records. An alert is not a packet-level classification.")

                # Protocol and Port Breakdown
                st.markdown("---")
                st.markdown("### 🔌 Protocol & Port Breakdown")
                p1, p2 = st.columns(2)
                with p1:
                    p_data = db.get_protocol_distribution()
                    p_chart = build_ranking_chart(p_data, "protocol", "count", "Protocol", "Packets", THEME_COLORS["secondary_accent"])
                    if p_chart:
                        st.altair_chart(p_chart, width="stretch")
                with p2:
                    pt_data = db.get_top_destination_ports(limit=8)
                    pt_chart = build_ranking_chart(pt_data, "dst_port", "packet_count", "Port", "Packets", THEME_COLORS["primary_accent"])
                    if pt_chart:
                        st.altair_chart(pt_chart, width="stretch")

                # Flows Table if available
                st.markdown("---")
                st.markdown("### 🌊 Indexed Bidirectional Network Flows")
                flows = db.get_flows(limit=50)
                if flows:
                    df_flows = pd.DataFrame(flows)
                    disp_cols = [c for c in ["flow_id", "src_ip", "dst_ip", "protocol", "duration", "packet_count", "byte_count", "flow_pkts_s"] if c in df_flows.columns]
                    st.dataframe(df_flows[disp_cols], width="stretch")
                else:
                    st.info("No flow summaries indexed yet. Run `python main.py --pcap <file> --hybrid` to aggregate and index stateful flows.")

            # ------------------------------------------------------------------
            # PAGE 5: AI/ML DETECTION
            # ------------------------------------------------------------------
            elif selected_page == "AI/ML Detection":
                st.title("🤖 AI/ML Detection Engine & Model Health")
                st.caption("Operational transparency for supervised classifiers and unsupervised Isolation Forest anomaly detectors.")

                registry = ModelRegistry(models_dir=_project_path("data/models"), db_path=target_db_path)
                active_sup = db.get_active_model("supervised_binary") or db.get_active_model()
                active_anom = db.get_active_model("anomaly_isolation_forest")
                flow_rows = db.get_flows(limit=1)
                latest_flow = flow_rows[0] if flow_rows else None
                sup_probe = _probe_active_model(registry, "supervised_binary", latest_flow)
                anom_probe = _probe_active_model(registry, "anomaly_isolation_forest", latest_flow)
                mode, mode_reason = _detection_mode_from_records(
                    all_alerts,
                    sup_probe["inference_status"] == "SUCCESS" or anom_probe["inference_status"] == "SUCCESS",
                )
                st.subheader(f"Runtime detection mode: {mode}")
                st.caption(mode_reason)

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown("### 🎯 Supervised Classifier Status")
                    if sup_probe["inference_status"] == "SUCCESS":
                        st.success(f"Bundle loaded and persisted-flow inference succeeded: **{sup_probe['algorithm']}** (`{sup_probe['version']}`)")
                        st.markdown(
                            f"- **Task:** `{active_sup.get('task_type', 'binary')}`\n"
                            f"- **Dataset:** `{active_sup.get('dataset_name', 'Unknown')}`\n"
                            f"- **Recorded test accuracy:** `{active_sup.get('metrics', {}).get('accuracy', 'Unavailable')}`\n"
                            f"- **Recorded weighted F1:** `{active_sup.get('metrics', {}).get('f1_weighted', 'Unavailable')}`\n"
                            "- **Inference input:** latest persisted flow features"
                        )
                    else:
                        st.warning(f"Supervised inference unavailable: {sup_probe['reason']}")

                with m_col2:
                    st.markdown("### 🌲 Unsupervised Anomaly Detector")
                    if anom_probe["inference_status"] == "SUCCESS":
                        st.success(f"Bundle loaded and persisted-flow anomaly inference succeeded: **{anom_probe['algorithm']}** (`{anom_probe['version']}`)")
                        st.markdown(
                            f"- **Recorded score threshold:** `{active_anom.get('metrics', {}).get('score_threshold', 'Unavailable')}`\n"
                            "- **Interpretation:** anomaly scores are not calibrated probabilities"
                        )
                    else:
                        st.warning(f"Anomaly inference unavailable: {anom_probe['reason']}")

                st.markdown("---")
                st.markdown("### 📐 Active Feature Schema")
                schema = active_sup.get("feature_schema", []) if active_sup else []
                if schema:
                    f_cols = st.columns(4)
                    for idx, feature in enumerate(schema):
                        f_cols[idx % 4].caption(f"{idx + 1}. `{feature}`")
                else:
                    st.info("No active supervised feature schema is recorded.")
                st.caption(
                    f"Bundle states — supervised: {sup_probe['bundle_status']} / {sup_probe['inference_status']}; "
                    f"anomaly: {anom_probe['bundle_status']} / {anom_probe['inference_status']}. "
                    "Inference probe results show runtime compatibility, not detector integration for live capture."
                )

                # Scientific Integrity Alert
                st.markdown("---")
                st.info(
                    "⚠️ **Scientific & Operational Notice:**\n\n"
                    "Unusual traffic patterns flagged by Isolation Forest do **NOT** automatically prove malicious compromise. "
                    "Network variance (e.g. software updates, large backup jobs, multi-stream media) frequently triggers high anomaly scores. "
                    "All AI/ML detections serve strictly as prioritized decision-support signals for security analysts."
                )

            # ------------------------------------------------------------------
            # PAGE 6: MODEL LABORATORY (Training & Evaluation Workspace)
            # ------------------------------------------------------------------
            elif selected_page == "Model Laboratory":
                st.title("🔬 Model Laboratory & Experimentation Workbench")
                st.caption("Train real classifiers, inspect verifiable metrics, compare confusion matrices, and activate validated model versions.")

                ds_manager = DatasetManager(base_dir=_project_path("data/datasets"))
                listing = ds_manager.list_datasets()
                registry = ModelRegistry(models_dir=_project_path("data/models"), db_path=target_db_path)

                with st.expander("🛠️ Train a New Model", expanded=True):
                    t_col1, t_col2, t_col3 = st.columns(3)
                    with t_col1:
                        target_ds = st.selectbox("Training Dataset", ["cicids2017", "unsw_nb15", "csecicids2018", "unified"], index=0)
                    with t_col2:
                        target_algo = st.selectbox(
                            "Algorithm",
                            ["Random Forest", "Extra Trees", "Logistic Regression", "HistGradientBoosting", "Isolation Forest"],
                            index=0,
                        )
                    with t_col3:
                        task_mode = st.selectbox("Classification Task", ["binary", "multiclass"], index=0)

                    auto_activate = st.checkbox("Automatically activate if validation succeeds", value=True)

                    if st.button("🚀 Train Model Now", type="primary"):
                        paths = ds_manager.get_dataset_paths(target_ds)
                        # Check raw file or create deterministic sample
                        raw_files = list(paths["raw"].glob("*.csv"))
                        if not raw_files:
                            st.info("Creating deterministic sample dataset fixture...")
                            ds_manager.generate_reproducible_sample_dataset(target_ds)
                            raw_files = list(paths["raw"].glob("*.csv"))

                        with st.spinner(f"Preprocessing data and training {target_algo}..."):
                            df = pd.read_csv(raw_files[0])
                            preprocessor = FlowDataPreprocessor(task_type=task_mode, split_method="stratified")
                            splits, prep_rep = preprocessor.fit_transform(df)

                            X_train, y_train = splits["X_train"], splits["y_train"]
                            X_test, y_test = splits["X_test"], splits["y_test"]

                            algo_slug = target_algo.lower().replace(" ", "_")
                            ver = f"{algo_slug}_{task_mode}_{int(time.time())}"

                            if algo_slug == "isolation_forest":
                                b_mask = y_train == 0
                                X_ben = X_train[b_mask] if b_mask.any() else X_train
                                trainer = AnomalyDetectorTrainer()
                                model, summary = trainer.train(X_ben)
                                is_anom, scores = trainer.score_samples(X_test)
                                acc = float(np.mean((is_anom.astype(int) == y_test)))
                                metrics = {
                                    "training_duration_seconds": summary.training_duration_seconds,
                                    "train_samples": summary.train_samples,
                                    "score_threshold": summary.score_threshold,
                                    "accuracy": acc,
                                }
                                m_type = "anomaly_isolation_forest"
                            else:
                                trainer = SupervisedModelTrainer(algorithm=algo_slug, task_type=task_mode)
                                model, summary = trainer.train(X_train, y_train)
                                class_names = [preprocessor.inverse_label_mapping_.get(i, f"Class_{i}") for i in sorted(list(set(y_train)))]
                                eval_res = ModelEvaluator.evaluate_classifier(
                                    model=model,
                                    X_test=X_test,
                                    y_test=y_test,
                                    model_name=target_algo,
                                    model_version=ver,
                                    dataset_name=target_ds,
                                    task_type=task_mode,
                                    split_method=prep_rep.split_method,
                                    class_names=class_names,
                                )
                                ModelEvaluator.save_reports(eval_res)
                                metrics = eval_res.to_dict()
                                m_type = f"supervised_{task_mode}"

                            meta = registry.save_model_bundle(
                                name=f"{target_algo}_{target_ds.upper()}",
                                version=ver,
                                model=model,
                                preprocessor=preprocessor,
                                model_type=m_type,
                                algorithm=target_algo,
                                dataset_name=target_ds,
                                task_type=task_mode,
                                feature_schema=preprocessor.feature_names_,
                                metrics=metrics,
                                set_active=auto_activate,
                            )
                            st.success(f"✅ Model `{ver}` trained successfully in {summary.training_duration_seconds:.2f}s!")
                            st.rerun()

                # Model Laboratory Tabs
                st.markdown("---")
                tab_registry, tab_cross_dataset, tab_benchmarks = st.tabs([
                    "📚 Model Artifact Registry",
                    "🌐 Cross-Dataset Generalization Matrix",
                    "📊 Multi-Dataset Benchmark Comparison",
                ])

                with tab_registry:
                    st.subheader("📚 Model Artifact Registry")
                    all_models = db.get_models()
                    if not all_models:
                        st.info("No models registered yet. Train a model using the form above.")
                    else:
                        for m in all_models:
                            is_act = bool(m.get("is_active"))
                            act_badge = "🟢 ACTIVE" if is_act else "⚪ INACTIVE"
                            with st.container():
                                st.markdown(f"#### {act_badge} `{m['name']}` (v: `{m['version']}`)")
                                mc1, mc2, mc3, mc4 = st.columns(4)
                                mc1.metric("Algorithm", m.get("algorithm", "-"))
                                mc2.metric("Task Type", m.get("task_type", "-"))
                                mc3.metric("Accuracy", f"{m['metrics'].get('accuracy', 0)*100:.2f}%")
                                mc4.metric("F1 Score", f"{m['metrics'].get('f1_macro', m['metrics'].get('f1_weighted', 0))*100:.2f}%")

                                if not is_act:
                                    if st.button(f"Activate {m['version']}", key=f"btn_act_{m['version']}"):
                                        db.set_active_model(m["version"], is_active=True)
                                        st.success(f"Activated model {m['version']}!")
                                        st.rerun()
                                st.caption(f"📁 Path: `{m.get('artifact_path')}` | Created: {format_timestamp(m.get('created_at'))}")
                                st.markdown("---")

                with tab_cross_dataset:
                    st.subheader("🌐 Cross-Dataset Generalization & Transfer Matrix")
                    st.caption("Empirical transferability results across CIC-IDS2017, UNSW-NB15, and CSE-CIC-IDS2018 using the 11-feature core.")
                    cross_report_file = Path("reports/metrics/cross_dataset_results.json")
                    if cross_report_file.exists():
                        with open(cross_report_file, "r", encoding="utf-8") as crf:
                            cr_data = json.load(crf)

                        experiments = cr_data.get("experiments", [])
                        tbl_rows = []
                        for exp in experiments:
                            m = exp["metrics"]
                            tbl_rows.append({
                                "Experiment": exp["name"],
                                "Train Source": "+".join(exp["train_sources"]).upper(),
                                "Test Source": exp["test_source"].upper(),
                                "Accuracy": f"{m.get('accuracy', 0.0)*100:.2f}%",
                                "Balanced Acc": f"{m.get('balanced_accuracy', 0.0)*100:.2f}%",
                                "Macro F1": f"{m.get('f1_macro', 0.0)*100:.2f}%",
                                "False Alarm Rate": f"{m.get('false_positive_rate', 0.0)*100:.2f}%",
                                "Transfer Gap": f"{exp.get('transfer_gap', 0.0):+.4f}",
                            })
                        st.dataframe(pd.DataFrame(tbl_rows), width="stretch")
                        st.info("💡 Key Research Finding: Cross-dataset domain shift causes significant transfer degradation when evaluating on disparate network architectures without multi-source domain adaptation. Hybrid detection (ML + Rules) provides essential defense-in-depth.")
                    else:
                        st.info("Run `python -m training.cross_dataset_evaluation` to populate the generalization matrix.")

                with tab_benchmarks:
                    st.subheader("📊 Multi-Dataset Benchmark Comparison")
                    master_rep_file = Path("reports/dataset_report.json")
                    if master_rep_file.exists():
                        with open(master_rep_file, "r", encoding="utf-8") as mrf:
                            m_rep = json.load(mrf)

                        bm_rows = []
                        for ds_k, tasks in m_rep.get("in_dataset_evaluations", {}).items():
                            for t_name, t_info in tasks.items():
                                for model_name, m_stats in t_info.get("models", {}).items():
                                    bm_rows.append({
                                        "Dataset": ds_k.upper(),
                                        "Task": t_name.upper(),
                                        "Model": model_name.split("_")[0].replace("-", " ").title(),
                                        "Accuracy": f"{m_stats.get('accuracy', 0.0)*100:.2f}%",
                                        "Balanced Acc": f"{m_stats.get('balanced_accuracy', 0.0)*100:.2f}%",
                                        "Macro F1": f"{m_stats.get('f1_macro', 0.0)*100:.2f}%",
                                        "False Alarm Rate": f"{m_stats.get('false_positive_rate', 0.0)*100:.2f}%",
                                        "Latency (ms/1k)": f"{m_stats.get('inference_latency_ms_per_1k', 0.0):.1f}ms",
                                    })
                        st.dataframe(pd.DataFrame(bm_rows), width="stretch")
                    else:
                        st.info("Run `python -m training.generate_report` to compile the multi-dataset benchmark comparison.")

            # ------------------------------------------------------------------
            # PAGE 7: DATA EXPLORER (Data Science EDA Workspace)
            # ------------------------------------------------------------------
            elif selected_page == "Data Explorer":
                st.title("📊 Data Explorer & Exploratory Analysis (EDA)")
                st.caption("Inspect raw dataset health, class distributions, missing value audits, and feature correlation matrices.")

                manager = DatasetManager(base_dir=_project_path("data/datasets"))
                target_ds = st.selectbox("Choose Dataset to Explore", ["cicids2017", "unsw_nb15", "csecicids2018", "unified"], index=0)
                paths = manager.get_dataset_paths(target_ds)
                raw_files = list(paths["raw"].glob("*.csv"))
                if not raw_files:
                    raw_sub = _project_path("data/raw") / target_ds
                    if raw_sub.exists():
                        raw_files = list(raw_sub.glob("*.csv"))

                if not raw_files:
                    st.info(f"No CSV files found in `{paths['raw']}`. Add a dataset file or use an explicit dataset preparation command.")

                if not raw_files:
                    st.stop()

                selected_csv = st.selectbox("Select CSV File", options=[f.name for f in raw_files], index=0)
                file_path = next((f for f in raw_files if f.name == selected_csv), paths["raw"] / selected_csv)
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)

                profile = DatasetProfiler.profile_dataframe(df)

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Total Records", f"{profile['total_rows']:,}")
                e2.metric("Total Features", f"{profile['total_cols']:,}")
                e3.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")
                e4.metric("Constant Columns", f"{len(profile['constant_columns'])}")

                # Class Distribution
                st.markdown("---")
                st.subheader("🎯 Ground-Truth Class Distribution")
                if profile["class_distribution"]:
                    cd_df = pd.DataFrame([
                        {"Class": k, "Count": v, "Percentage": round(v * 100.0 / profile['total_rows'], 2)}
                        for k, v in profile["class_distribution"].items()
                    ])
                    col_chart, col_tbl = st.columns([2, 1])
                    with col_chart:
                        c_chart = (
                            alt.Chart(cd_df)
                            .mark_bar(color=THEME_COLORS["primary_accent"], cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                            .encode(
                                x=alt.X("Count:Q", title="Samples"),
                                y=alt.Y("Class:N", sort="-x", title="Target Class"),
                                tooltip=["Class:N", "Count:Q", "Percentage:Q"],
                            )
                            .properties(height=200)
                        )
                        st.altair_chart(c_chart, width="stretch")
                    with col_tbl:
                        st.dataframe(cd_df, width="stretch")

                # Numerical Summaries
                st.markdown("---")
                st.subheader("📈 Statistical Feature Summaries")
                if profile["numeric_summaries"]:
                    summary_df = pd.DataFrame(profile["numeric_summaries"]).T
                    st.dataframe(summary_df.round(3), width="stretch")

                # Correlation Matrix
                st.markdown("---")
                st.subheader("🔥 Feature Correlation Analysis")
                cols, corr_matrix = DatasetProfiler.compute_correlation_matrix(df, max_features=12)
                if len(cols) > 1:
                    corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols).round(2)
                    st.dataframe(corr_df, width="stretch")

            # ------------------------------------------------------------------
            # PAGE 8: DETECTION RULES
            # ------------------------------------------------------------------
            elif selected_page == "Detection Rules":
                st.title("⚙️ Detection Rules & Heuristic Policies")
                st.caption("Inspect and manage threshold parameters for rule-based and machine-learning detectors.")

                rules_table = [
                    {
                        "Rule": "PORT_SCAN",
                        "Category": "Reconnaissance",
                        "Target": "TCP Probes across distinct destination ports",
                        "Threshold": "15 distinct ports",
                        "Window": "30.0 seconds",
                        "Severity": "HIGH",
                        "Status": "ACTIVE",
                    },
                    {
                        "Rule": "SYN_RATE",
                        "Category": "Denial of Service (DoS)",
                        "Target": "High-velocity TCP SYN packet bursts",
                        "Threshold": "20 SYN packets",
                        "Window": "10.0 seconds",
                        "Severity": "MEDIUM",
                        "Status": "ACTIVE",
                    },
                    {
                        "Rule": "TRAFFIC_VOLUME",
                        "Category": "Anomaly / Exfiltration",
                        "Target": "Sudden spikes in byte or packet volume",
                        "Threshold": "100,000 bytes or 500 packets",
                        "Window": "60.0 seconds",
                        "Severity": "MEDIUM",
                        "Status": "ACTIVE",
                    },
                    {
                        "Rule": "ML_SUPERVISED",
                        "Category": "Machine Learning Classifier",
                        "Target": "Multivariate flow vector matching known attack patterns",
                        "Threshold": "Confidence >= 0.50",
                        "Window": "Stateful Flow Duration",
                        "Severity": "CRITICAL / HIGH",
                        "Status": "ACTIVE" if active_sup else "STANDBY",
                    },
                    {
                        "Rule": "ML_ANOMALY",
                        "Category": "Unsupervised Outlier",
                        "Target": "Statistical deviation from benign traffic profile",
                        "Threshold": "Anomaly Score >= 0.50",
                        "Window": "Stateful Flow Duration",
                        "Severity": "MEDIUM",
                        "Status": "ACTIVE" if active_anom else "STANDBY",
                    },
                ]
                st.dataframe(pd.DataFrame(rules_table), width="stretch")

            # ------------------------------------------------------------------
            # PAGE 9: SYSTEM & SETTINGS
            # ------------------------------------------------------------------
            elif selected_page == "System & Settings":
                st.title("🖥️ System Diagnostics & Environment Status")
                st.caption("Verify runtime dependencies, database health, and evaluation reports.")
                with st.expander("Quick start, offline demo, and safe-operation guide", expanded=False):
                    st.markdown(
                        "1. Start from the repository root with `streamlit run app.py`.\n"
                        "2. Enable **Synthetic Demonstration Mode** in the sidebar to use the separate demo database.\n"
                        "3. Review stored detections in Alert Investigation, then evaluate a stored demo alert in Prevention Center.\n"
                        "4. The result is a simulation record; host firewall rules and network traffic are never changed.\n"
                        "5. For offline PCAP ingestion, run `python main.py --pcap <capture.pcap> --db <path-to-database>`.\n"
                        "6. `ML_READY` requires successful bundle loading and inference; it does not imply every detection path uses ML."
                    )

                s1, s2, s3 = st.columns(3)
                s1.metric("Python Interpreter", sys.version.split(" ")[0])
                s2.metric("Database Path", str(target_db_path.name))
                file_size_kb = target_db_path.stat().st_size / 1024 if target_db_path.exists() else 0
                s3.metric("Database Size", f"{file_size_kb:.1f} KB")

                st.markdown("---")
                st.subheader("📊 Stored Database Statistics")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Packets", f"{packet_count:,}")
                sc2.metric("Alerts", f"{alert_count:,}")
                sc3.metric("Indexed Flows", f"{db.get_flow_count():,}")
                all_registered_models = db.get_models()
                sc4.metric("Registered Models", f"{len(all_registered_models):,}")

                st.markdown("---")
                st.subheader("📦 Environment & Engine Status")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("SQLite Engine", sqlite3.sqlite_version)
                e2.metric("Streamlit Version", st.__version__)
                e3.metric("Scikit-Learn", sklearn.__version__)
                try:
                    import scapy
                    e4.metric("Scapy Engine", scapy.__version__)
                except Exception:
                    e4.metric("Scapy Engine", "Unavailable")

                st.markdown("---")
                st.markdown("### 📄 Evaluation Reports Browser")
                reports_dir = _project_path("data/reports")
                if reports_dir.exists():
                    md_reports = list(reports_dir.glob("*.md"))
                    if md_reports:
                        chosen_rep = st.selectbox("Select Evaluation Report", [r.name for r in md_reports])
                        rep_path = reports_dir / chosen_rep
                        st.markdown(rep_path.read_text(encoding="utf-8"))
                    else:
                        st.info("No evaluation reports generated yet. Run `python scripts/evaluate_models.py`.")
                else:
                    st.info("No evaluation reports generated yet.")

                st.markdown("---")
                st.markdown("### ⚖️ Defensive Cybersecurity & Privacy Principles")
                st.caption(
                    "Mini-IDS is an educational, passive network intrusion detection system. "
                    "It observes packet headers and flow metadata without altering, injecting, or terminating network sessions. "
                    "All captured network telemetry remains strictly on the local workstation and is never uploaded externally."
                )

    except sqlite3.Error as sql_err:
        st.error(f"Database error while reading `{target_db_path}`: {sql_err}")
        return
    except Exception as err:
        st.error(f"Failed to access database `{target_db_path}`: {err}")
        return


if __name__ == "__main__":
    render_dashboard()
