"""Exploratory Data Analysis (EDA) and Tabular Profiling Module.

Computes dataset health checks, class distribution, missing/infinite value tallies,
summary statistics, and feature correlation matrices for tabular network datasets.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DatasetProfiler:
    """Profiles raw and cleaned network flow datasets for data science analysis."""

    @staticmethod
    def profile_dataframe(df: pd.DataFrame, sample_size: int = 5000) -> Dict[str, Any]:
        """Compute comprehensive statistical profile of a dataset."""
        total_rows = len(df)
        total_cols = len(df.columns)

        # Work on a sample if very large to keep UI responsive
        work_df = df.sample(min(total_rows, sample_size), random_state=42) if total_rows > sample_size else df

        # Column names and types
        col_types = {col: str(dtype) for col, dtype in work_df.dtypes.items()}

        # Missing & Infinite values
        missing_counts = work_df.isna().sum().to_dict()
        missing_pcts = {k: round(v * 100.0 / len(work_df), 2) for k, v in missing_counts.items()}

        numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        inf_counts = {}
        for col in numeric_cols:
            n_inf = int(np.isinf(work_df[col]).sum())
            if n_inf > 0:
                inf_counts[col] = n_inf

        # Duplicates
        duplicate_rows = int(work_df.duplicated().sum())

        # Constant columns
        constant_cols = [col for col in numeric_cols if work_df[col].nunique(dropna=True) <= 1]

        # Numerical summaries
        summaries = {}
        if numeric_cols:
            desc = work_df[numeric_cols].describe().T
            for col, row in desc.iterrows():
                summaries[col] = {
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "25%": float(row["25%"]),
                    "50%": float(row["50%"]),
                    "75%": float(row["75%"]),
                    "max": float(row["max"]),
                }

        # Label column analysis
        label_col = None
        for cand in ["label", "Label", "attack_cat", "class"]:
            if cand in work_df.columns:
                label_col = cand
                break

        class_distribution = {}
        if label_col:
            class_distribution = {
                str(k): int(v) for k, v in work_df[label_col].value_counts().to_dict().items()
            }

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "analyzed_rows": len(work_df),
            "column_types": col_types,
            "missing_counts": missing_counts,
            "missing_pcts": missing_pcts,
            "infinite_counts": inf_counts,
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_cols,
            "numeric_summaries": summaries,
            "label_column": label_col,
            "class_distribution": class_distribution,
        }

    @staticmethod
    def compute_correlation_matrix(
        df: pd.DataFrame,
        max_features: int = 15,
    ) -> Tuple[List[str], np.ndarray]:
        """Compute Pearson correlation matrix for top variance numeric features."""
        numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        if numeric_df.empty:
            return [], np.zeros((0, 0))

        # Select features with highest variance up to max_features
        variances = numeric_df.var()
        top_cols = variances.nlargest(min(len(variances), max_features)).index.tolist()

        corr_sub = numeric_df[top_cols].corr().fillna(0.0)
        return top_cols, corr_sub.to_numpy()
