"""Statistical Anomaly Detection Engine for Mini-IDS.

Leverages trained Isolation Forest estimators to identify novel zero-day
behavior and out-of-distribution traffic patterns without requiring
pre-existing attack signatures.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class AnomalyDetectionEngine:
    """Evaluates flow feature vectors against a trained Isolation Forest baseline."""

    def __init__(
        self,
        model: Any = None,
        threshold: float = 0.50,
    ) -> None:
        self.model = model
        self.threshold = threshold

    def score_vector(self, feature_vector: np.ndarray) -> Tuple[bool, float, str]:
        """Score a preprocessed 1D or 2D feature array.

        Returns:
            Tuple[is_anomaly: bool, anomaly_score: float, reason: str]
        """
        if self.model is None:
            return False, 0.0, "Anomaly model uninitialized"

        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector

        # Isolation forest scores: lower values mean more anomalous
        # Negate so higher means more anomalous
        raw_score = -float(self.model.score_samples(X)[0])
        score = round(raw_score, 4)
        is_anom = (score >= self.threshold)

        if is_anom:
            reason = f"Out-of-distribution anomaly score ({score:.3f} >= {self.threshold:.3f})"
        else:
            reason = f"Conforms to benign baseline distribution (score: {score:.3f})"

        return is_anom, score, reason


__all__ = ["AnomalyDetectionEngine"]
