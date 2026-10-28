"""Unsupervised Anomaly Detection Module (Isolation Forest).

Trains an Isolation Forest detector on documented benign network baseline flows,
computes calibrated anomaly scores, and provides decision thresholds.
"""

from dataclasses import asdict, dataclass
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class AnomalyModelSummary:
    """Execution summary for an unsupervised anomaly model."""
    algorithm: str
    training_population: str  # e.g., 'benign_only' or 'unlabeled'
    contamination: float
    n_estimators: int
    training_duration_seconds: float
    train_samples: int
    feature_count: int
    random_seed: int
    score_threshold: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnomalyDetectorTrainer:
    """Trains and manages Isolation Forest anomaly models."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_seed: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.model: Optional[IsolationForest] = None
        self.summary_: Optional[AnomalyModelSummary] = None
        self.score_threshold: float = 0.5  # Normalized threshold

    def train(
        self,
        X_train: np.ndarray,
        training_population: str = "benign_only",
    ) -> Tuple[IsolationForest, AnomalyModelSummary]:
        """Fit Isolation Forest on training features (ideally benign flows)."""
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty.")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_seed,
            n_jobs=-1,
        )

        logger.info(
            "Fitting Isolation Forest on %d samples (population: %s)...",
            len(X_train),
            training_population,
        )
        start = time.perf_counter()
        self.model.fit(X_train)
        duration = time.perf_counter() - start

        # Calculate score offset and calibrate threshold
        # scikit-learn decision_function: negative = anomaly, positive = inlier
        train_scores = self.model.decision_function(X_train)
        # 5th percentile of decision function represents the empirical decision boundary
        emp_threshold = float(np.percentile(train_scores, self.contamination * 100))

        self.summary_ = AnomalyModelSummary(
            algorithm="isolation_forest",
            training_population=training_population,
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            training_duration_seconds=duration,
            train_samples=int(len(X_train)),
            feature_count=int(X_train.shape[1]),
            random_seed=self.random_seed,
            score_threshold=emp_threshold,
            timestamp=time.time(),
        )
        return self.model, self.summary_

    def score_samples(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute anomaly predictions and normalized anomaly scores.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - is_anomaly (bool array, True if anomalous)
                - normalized_scores (float array [0.0, 1.0], higher = more anomalous)
        """
        if self.model is None:
            raise RuntimeError("Anomaly model is not trained yet.")

        # decision_function gives negative scores for outliers, positive for inliers
        raw_decisions = self.model.decision_function(X)
        # Predict: -1 for outlier, 1 for inlier
        raw_preds = self.model.predict(X)
        is_anomaly = raw_preds == -1

        # Normalize score into [0.0, 1.0] where 1.0 is extreme anomaly
        # Use sigmoid-like scaling centered at 0.0
        normalized = 1.0 / (1.0 + np.exp(raw_decisions * 5.0))
        return is_anomaly, normalized
