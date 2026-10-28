"""Tabular Network-Flow Preprocessing and Leakage-Free Splitting Pipeline.

Implements robust data cleaning, numeric conversion, infinite-value handling,
leakage column exclusion, scikit-learn preprocessing pipelines, and leakage-free
splitting (temporal and stratified).
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Standard CICIDS / UNSW normalized target names
DEFAULT_LABEL_COLUMNS = ["label", "Label", "attack_cat"]

# Potential leakage features to exclude from model training
LEAKAGE_COLUMNS = [
    "timestamp",
    "flow_id",
    "src_ip",
    "source_ip",
    "dst_ip",
    "destination_ip",
    "id",
    "unnamed:_0",
    "unnamed: 0",
]

# Training data uses CICIDS header normalization, while live flow extraction
# emits the shorter ``total_bwd_packets`` spelling. Keep this translation at
# the training preprocessor boundary so saved model feature order stays intact.
INFERENCE_FEATURE_ALIASES = {
    "total_backward_packets": ("total_bwd_packets",),
}


@dataclass
class PreprocessingReport:
    """Summary metrics of dataset cleaning and splitting."""
    initial_rows: int
    initial_cols: int
    duplicate_rows_removed: int
    inf_values_handled: int
    missing_values_imputed: int
    constant_columns_dropped: List[str]
    leakage_columns_dropped: List[str]
    final_feature_count: int
    feature_names: List[str]
    split_method: str  # 'stratified' or 'temporal'
    train_rows: int
    val_rows: int
    test_rows: int
    label_distribution: Dict[str, int]
    target_type: str   # 'binary' or 'multiclass'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FlowDataPreprocessor:
    """Pipeline for cleaning, transforming, and partitioning tabular network flow datasets."""

    def __init__(
        self,
        target_col: Optional[str] = None,
        task_type: str = "binary",
        scale_features: bool = True,
        split_method: str = "stratified",
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.target_col = target_col
        self.task_type = task_type  # 'binary' or 'multiclass'
        self.scale_features = scale_features
        self.split_method = split_method
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # Fitted transformers
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        self.label_mapping_: Dict[str, int] = {}
        self.inverse_label_mapping_: Dict[int, str] = {}
        self.report_: Optional[PreprocessingReport] = None

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace and normalize column names."""
        df = df.copy()
        df.columns = [
            str(col).strip().lower().replace(" ", "_").replace("/", "_").replace(".", "_")
            for col in df.columns
        ]
        return df

    def find_target_column(self, df: pd.DataFrame) -> str:
        """Identify the ground-truth label column."""
        if self.target_col and self.target_col.lower() in df.columns:
            return self.target_col.lower()
        for cand in ["label", "attack_cat", "class"]:
            if cand in df.columns:
                return cand
        raise ValueError(f"Could not automatically locate label column in: {list(df.columns)}")

    def normalize_labels(self, labels: pd.Series) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """Map text or numeric labels to integer class IDs."""
        str_labels = labels.astype(str).str.strip()

        if self.task_type == "binary":
            # 0 = BENIGN, 1 = ATTACK
            y = np.where(str_labels.str.upper() == "BENIGN", 0, 1)
            mapping = {"BENIGN": 0, "ATTACK": 1}
            inv_mapping = {0: "BENIGN", 1: "ATTACK"}
            return y, mapping, inv_mapping

        # Multiclass: group into canonical attack families
        def map_multiclass(l: str) -> str:
            up = l.upper()
            if "BENIGN" in up:
                return "BENIGN"
            if "PORT" in up or "SCAN" in up:
                return "PortScan"
            if "DOS" in up or "DDOS" in up:
                return "DoS/DDoS"
            if "BRUTE" in up or "PATATOR" in up:
                return "BruteForce"
            if "BOT" in up:
                return "Botnet"
            if "INFILTR" in up:
                return "Infiltration"
            if "WEB" in up or "XSS" in up or "SQL" in up:
                return "WebAttack"
            return "Other_Attack"

        grouped = str_labels.apply(map_multiclass)
        unique_labels = sorted(list(grouped.unique()))
        # Ensure BENIGN is class 0
        if "BENIGN" in unique_labels:
            unique_labels.remove("BENIGN")
            unique_labels.insert(0, "BENIGN")

        mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        inv_mapping = {idx: lbl for lbl, idx in mapping.items()}
        y = grouped.map(mapping).to_numpy()
        return y, mapping, inv_mapping

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, np.ndarray], PreprocessingReport]:
        """Fit preprocessing pipeline on train split only and transform partitions.

        Prevents data leakage by computing imputations and scaling exclusively
        on training observations.
        """
        init_rows, init_cols = df.shape
        cleaned_df = self.clean_column_names(df)

        # 1. Identify target column
        target_name = self.find_target_column(cleaned_df)

        # 2. Separate labels before cleaning features
        raw_labels = cleaned_df[target_name]
        feature_df = cleaned_df.drop(columns=[target_name])

        # 3. Check and drop duplicate rows
        duplicates_mask = feature_df.duplicated()
        n_duplicates = int(duplicates_mask.sum())
        if n_duplicates > 0:
            feature_df = feature_df[~duplicates_mask]
            raw_labels = raw_labels[~duplicates_mask]

        # 4. Handle infinities in numeric columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        inf_count = 0
        for col in numeric_cols:
            inf_mask = np.isinf(feature_df[col])
            inf_count += int(inf_mask.sum())
            if inf_mask.any():
                feature_df[col] = feature_df[col].replace([np.inf, -np.inf], np.nan)

        # 5. Drop leakage columns
        leakage_dropped = []
        for col in list(feature_df.columns):
            for pattern in LEAKAGE_COLUMNS:
                if pattern in col.lower():
                    leakage_dropped.append(col)
                    feature_df = feature_df.drop(columns=[col])
                    break

        # 6. Drop constant variance columns (zero standard deviation)
        constant_dropped = []
        for col in feature_df.select_dtypes(include=[np.number]).columns:
            if feature_df[col].nunique(dropna=True) <= 1:
                constant_dropped.append(col)
                feature_df = feature_df.drop(columns=[col])

        # 7. Select remaining numeric features
        usable_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        X_raw = feature_df[usable_features]
        self.feature_names_ = usable_features

        # 8. Encode labels
        y_raw, self.label_mapping_, self.inverse_label_mapping_ = self.normalize_labels(raw_labels)

        # 9. Split partitions (Temporal or Stratified) to prevent data leakage
        if self.split_method == "temporal" and "timestamp" in cleaned_df.columns:
            # Sort chronologically by original timestamp
            sort_idx = np.argsort(cleaned_df.loc[X_raw.index, "timestamp"].to_numpy())
            X_sorted = X_raw.iloc[sort_idx]
            y_sorted = y_raw[sort_idx]

            n = len(X_sorted)
            test_n = int(n * self.test_size)
            val_n = int(n * self.val_size)
            train_n = n - test_n - val_n

            X_train_df = X_sorted.iloc[:train_n]
            y_train = y_sorted[:train_n]
            X_val_df = X_sorted.iloc[train_n : train_n + val_n]
            y_val = y_sorted[train_n : train_n + val_n]
            X_test_df = X_sorted.iloc[train_n + val_n :]
            y_test = y_sorted[train_n + val_n :]
        else:
            # Stratified random split
            X_train_df, X_temp_df, y_train, y_temp = train_test_split(
                X_raw,
                y_raw,
                test_size=(self.test_size + self.val_size),
                random_state=self.random_state,
                stratify=y_raw if len(np.unique(y_raw)) > 1 else None,
            )
            relative_test_ratio = self.test_size / (self.test_size + self.val_size)
            X_val_df, X_test_df, y_val, y_test = train_test_split(
                X_temp_df,
                y_temp,
                test_size=relative_test_ratio,
                random_state=self.random_state,
                stratify=y_temp if len(np.unique(y_temp)) > 1 else None,
            )

        # 10. Fit Imputer and Scaler STRICTLY on Training Split
        self.imputer = SimpleImputer(strategy="median")
        X_train_imp = self.imputer.fit_transform(X_train_df)
        X_val_imp = self.imputer.transform(X_val_df)
        X_test_imp = self.imputer.transform(X_test_df)

        missing_count = int(np.isnan(X_train_df.to_numpy()).sum())

        if self.scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train_imp)
            X_val = self.scaler.transform(X_val_imp)
            X_test = self.scaler.transform(X_test_imp)
        else:
            self.scaler = None
            X_train = X_train_imp
            X_val = X_val_imp
            X_test = X_test_imp

        # Class counts in raw data
        label_dist = {str(k): int(v) for k, v in raw_labels.value_counts().to_dict().items()}

        self.report_ = PreprocessingReport(
            initial_rows=init_rows,
            initial_cols=init_cols,
            duplicate_rows_removed=n_duplicates,
            inf_values_handled=inf_count,
            missing_values_imputed=missing_count,
            constant_columns_dropped=constant_dropped,
            leakage_columns_dropped=leakage_dropped,
            final_feature_count=len(self.feature_names_),
            feature_names=self.feature_names_,
            split_method=self.split_method,
            train_rows=len(X_train),
            val_rows=len(X_val),
            test_rows=len(X_test),
            label_distribution=label_dist,
            target_type=self.task_type,
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }, self.report_

    def transform_inference(self, feature_dict_or_df: Union[Dict[str, float], pd.DataFrame]) -> np.ndarray:
        """Transform incoming live or offline flow feature records for model inference.

        Ensures exact column ordering and applies fitted training imputation & scaling.
        """
        if self.imputer is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        if isinstance(feature_dict_or_df, dict):
            df = pd.DataFrame([feature_dict_or_df])
        else:
            df = feature_dict_or_df.copy()

        df = self.clean_column_names(df)

        # Resolve known runtime spellings before imputing genuinely unavailable
        # fields. Without this step, the backward packet count is silently
        # replaced by its training median for otherwise complete live flows.
        for expected_name, aliases in INFERENCE_FEATURE_ALIASES.items():
            if expected_name not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df[expected_name] = df[alias]
                        break

        # Align columns to fitted feature_names_
        aligned_df = pd.DataFrame(index=df.index)
        for col in self.feature_names_:
            if col in df.columns:
                aligned_df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                aligned_df[col] = np.nan

        X_imp = self.imputer.transform(aligned_df)
        if self.scaler is not None:
            return self.scaler.transform(X_imp)
        return X_imp

    def save(self, output_path: Union[str, Path]) -> Path:
        """Serialize preprocessor pipeline using joblib."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "FlowDataPreprocessor":
        """Load serialized preprocessor pipeline."""
        return joblib.load(input_path)
