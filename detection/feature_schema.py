"""Central Network Flow Feature Schema and Validation Module.

Defines the canonical tabular flow feature schema used across training,
offline PCAP flow aggregation, and live inference. Provides defensive validation,
bounds-checking, infinite/NaN handling, and schema harmonization for CIC-IDS2017
and related benchmark datasets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

# Canonical list of 28 statistical flow features (strictly ordered)
CANONICAL_FLOW_FEATURES: List[str] = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "fwd_packet_length_min",
    "fwd_packet_length_max",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "bwd_packet_length_min",
    "bwd_packet_length_max",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_min",
    "flow_iat_max",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_mean",
    "bwd_iat_std",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
]

# Mapping from raw CIC-IDS2017 header variations to canonical feature names
CICIDS2017_HEADER_MAP: Dict[str, str] = {
    "flow duration": "flow_duration",
    "total fwd packets": "total_fwd_packets",
    "total backward packets": "total_bwd_packets",
    "total bwd packets": "total_bwd_packets",
    "total length of fwd packets": "total_length_of_fwd_packets",
    "total length of bwd packets": "total_length_of_bwd_packets",
    "fwd packet length mean": "fwd_packet_length_mean",
    "fwd packet length std": "fwd_packet_length_std",
    "fwd packet length min": "fwd_packet_length_min",
    "fwd packet length max": "fwd_packet_length_max",
    "bwd packet length mean": "bwd_packet_length_mean",
    "bwd packet length std": "bwd_packet_length_std",
    "bwd packet length min": "bwd_packet_length_min",
    "bwd packet length max": "bwd_packet_length_max",
    "flow bytes/s": "flow_bytes_s",
    "flow bytes_s": "flow_bytes_s",
    "flow packets/s": "flow_packets_s",
    "flow packets_s": "flow_packets_s",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "flow iat min": "flow_iat_min",
    "flow iat max": "flow_iat_max",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "fin flag count": "fin_flag_count",
    "syn flag count": "syn_flag_count",
    "rst flag count": "rst_flag_count",
    "psh flag count": "psh_flag_count",
    "ack flag count": "ack_flag_count",
}

# Features directly accessible from single packets vs stateful flow aggregations
PACKET_LEVEL_FIELDS: Set[str] = {
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "length",
    "timestamp",
    "tcp_flags",
}

LABEL_FIELDS: Set[str] = {"label", "attack_cat", "binary_label", "target"}


@dataclass
class SchemaValidationReport:
    """Detailed summary of feature schema validation and transformation."""
    is_valid: bool
    total_features: int
    missing_features: List[str]
    unexpected_features: List[str]
    nan_imputed_count: int
    inf_imputed_count: int
    clipped_negative_count: int


class FeatureSchemaValidator:
    """Enforces consistency, type safety, and value bounds across all data pipelines."""

    @staticmethod
    def normalize_column_name(col: str) -> str:
        """Strip whitespace, replace slashes and spaces with underscores, and lowercase."""
        clean = col.strip().lower()
        clean = clean.replace("/", "_").replace("-", "_")
        # Check explicit mapping
        return CICIDS2017_HEADER_MAP.get(clean, clean)

    @classmethod
    def validate_and_clean_dataframe(
        cls,
        df: pd.DataFrame,
        strict: bool = False,
    ) -> Tuple[pd.DataFrame, SchemaValidationReport]:
        """Align input DataFrame with CANONICAL_FLOW_FEATURES.
        
        Args:
            df: Input pandas DataFrame with flow data.
            strict: If True, raises ValueError if missing canonical columns.
            
        Returns:
            Tuple[pd.DataFrame, SchemaValidationReport]: Cleaned DataFrame containing
            exact canonical columns in order, plus validation report.
        """
        # 1. Normalize column names
        rename_map = {orig: cls.normalize_column_name(orig) for orig in df.columns}
        df_clean = df.rename(columns=rename_map).copy()

        missing = [f for f in CANONICAL_FLOW_FEATURES if f not in df_clean.columns]
        unexpected = [c for c in df_clean.columns if c not in CANONICAL_FLOW_FEATURES and c not in LABEL_FIELDS]

        if strict and missing:
            raise ValueError(f"Feature schema violation. Missing required canonical features: {missing}")

        # Fill missing features with 0.0
        for f in missing:
            df_clean[f] = 0.0

        # Select canonical columns in exact required order
        X = df_clean[CANONICAL_FLOW_FEATURES].copy()

        # 2. Count and sanitize infinite values
        inf_count = int(np.isinf(X.values).sum())
        X = X.replace([np.inf, -np.inf], np.nan)

        # 3. Count and sanitize NaN values
        nan_count = int(X.isna().sum().sum())
        X = X.fillna(0.0)

        # 4. Clip negative values for inherently non-negative physical quantities (counts, bytes, duration)
        negative_count = int((X < 0.0).sum().sum())
        X = X.clip(lower=0.0)

        # Cast to float64
        X = X.astype(np.float64)

        report = SchemaValidationReport(
            is_valid=len(missing) == 0,
            total_features=len(CANONICAL_FLOW_FEATURES),
            missing_features=missing,
            unexpected_features=unexpected,
            nan_imputed_count=nan_count,
            inf_imputed_count=inf_count,
            clipped_negative_count=negative_count,
        )
        return X, report

    @classmethod
    def validate_feature_dict(cls, feat_dict: Dict[str, Any]) -> Dict[str, float]:
        """Validate and sanitize a single flow's extracted feature dictionary."""
        sanitized: Dict[str, float] = {}
        for f in CANONICAL_FLOW_FEATURES:
            val = feat_dict.get(f, 0.0)
            try:
                flt_val = float(val)
                if np.isnan(flt_val) or np.isinf(flt_val):
                    flt_val = 0.0
                sanitized[f] = max(0.0, flt_val)
            except (ValueError, TypeError):
                sanitized[f] = 0.0
        return sanitized

    clean_feature_dict = validate_feature_dict


def validate_feature_dataframe(df: pd.DataFrame, strict: bool = False) -> SchemaValidationReport:
    """Validate a DataFrame against canonical flow feature schema."""
    _, report = FeatureSchemaValidator.validate_and_clean_dataframe(df, strict=strict)
    return report


def extract_canonical_features(
    df: pd.DataFrame, target_col: Optional[str] = "label"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Extract canonical feature matrix and target column from DataFrame."""
    y: Optional[pd.Series] = None
    target_clean = None
    if target_col:
        for c in df.columns:
            if c.strip().lower() == target_col.lower():
                target_clean = c
                break
        if target_clean is not None:
            y = df[target_clean].copy()

    X, _ = FeatureSchemaValidator.validate_and_clean_dataframe(df, strict=False)
    return X, y

