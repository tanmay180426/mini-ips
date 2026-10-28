"""Real-Time and Offline Flow Feature Preprocessing Pipeline for Detection.

Validates incoming flow dictionaries or DataFrames against either the
Canonical 28-Feature Schema (CIC-IDS2017) or the Unified 11-Feature Core,
applies median imputation and RobustScaler transformations, and prepares
feature matrices for inference.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from detection.feature_schema import (
    CANONICAL_FLOW_FEATURES,
    FeatureSchemaValidator,
    extract_canonical_features,
    validate_feature_dataframe,
)

# Canonical 11-feature common semantic core for unified multi-dataset evaluation
UNIFIED_CORE_FEATURES: List[str] = [
    "flow_duration_sec",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "total_packets",
    "total_bytes",
    "packet_rate",
    "byte_rate",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
]


class DetectionPreprocessor:
    """Handles flow schema validation, mapping, and scaling for live and offline inference."""

    def __init__(
        self,
        schema_type: str = "canonical",
        scaler: Optional[Any] = None,
        imputer: Optional[Any] = None,
    ) -> None:
        self.schema_type = schema_type
        self.scaler = scaler
        self.imputer = imputer
        self.feature_names = (
            UNIFIED_CORE_FEATURES if schema_type == "unified" else CANONICAL_FLOW_FEATURES
        )

    def prepare_flow(self, flow: Dict[str, Any]) -> np.ndarray:
        """Convert a single flow dictionary into a scaled 2D numpy array [1, n_features]."""
        if self.schema_type == "unified":
            from training.prepare_datasets import extract_unified_features_from_row
            series = pd.Series(flow)
            feat_dict = extract_unified_features_from_row(series, "cicids2017")
            row = np.array([[feat_dict[f] for f in UNIFIED_CORE_FEATURES]], dtype=np.float32)
        else:
            validator = FeatureSchemaValidator()
            cleaned = validator.clean_feature_dict(flow)
            row = np.array([[cleaned[f] for f in CANONICAL_FLOW_FEATURES]], dtype=np.float32)

        row[np.isinf(row)] = np.nan
        if self.imputer:
            row = self.imputer.transform(row)
        else:
            row = np.nan_to_num(row, nan=0.0)

        if self.scaler:
            row = self.scaler.transform(row)

        return row

    def prepare_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Clean and prepare a DataFrame batch for inference."""
        if self.schema_type == "unified":
            from training.prepare_datasets import extract_unified_features_from_row
            records = []
            for _, r in df.iterrows():
                feat_dict = extract_unified_features_from_row(r, "cicids2017")
                records.append([feat_dict[f] for f in UNIFIED_CORE_FEATURES])
            X = np.array(records, dtype=np.float32)
            f_names = UNIFIED_CORE_FEATURES
        else:
            X_df, _ = extract_canonical_features(df)
            X = X_df.values.astype(np.float32)
            f_names = CANONICAL_FLOW_FEATURES

        X[np.isinf(X)] = np.nan
        if self.imputer:
            X = self.imputer.transform(X)
        else:
            X = np.nan_to_num(X, nan=0.0)

        if self.scaler:
            X = self.scaler.transform(X)

        return X, f_names


__all__ = ["DetectionPreprocessor", "CANONICAL_FLOW_FEATURES", "UNIFIED_CORE_FEATURES"]
