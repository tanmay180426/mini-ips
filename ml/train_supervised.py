"""Supervised Classification Model Training Module.

Implements real training routines for Logistic Regression, Random Forest,
and HistGradientBoosting classifiers for both binary and multiclass intrusion detection tasks.
"""

from dataclasses import asdict, dataclass
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class TrainingSummary:
    """Metadata and execution parameters of a model training run."""
    algorithm: str
    task_type: str
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float
    train_samples: int
    feature_count: int
    random_seed: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupervisedModelTrainer:
    """Trains scikit-learn classifiers on preprocessed network flow data."""

    SUPPORTED_ALGORITHMS = ["random_forest", "logistic_regression", "hist_gradient_boosting"]

    def __init__(
        self,
        algorithm: str = "random_forest",
        task_type: str = "binary",
        random_seed: int = 42,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        algo_clean = algorithm.lower().replace(" ", "_")
        if algo_clean not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from: {self.SUPPORTED_ALGORITHMS}")

        self.algorithm = algo_clean
        self.task_type = task_type
        self.random_seed = random_seed
        self.hyperparameters = hyperparameters or {}
        self.model: Optional[Any] = None
        self.summary_: Optional[TrainingSummary] = None

    def _build_estimator(self) -> Any:
        """Instantiate estimator with specified or default hyperparameters."""
        params = self.hyperparameters.copy()

        if self.algorithm == "random_forest":
            n_estimators = int(params.get("n_estimators", 100))
            max_depth = params.get("max_depth", 15)
            min_samples_split = int(params.get("min_samples_split", 2))
            class_weight = params.get("class_weight", "balanced")
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight=class_weight,
                random_state=self.random_seed,
                n_jobs=-1,
            )

        elif self.algorithm == "logistic_regression":
            max_iter = int(params.get("max_iter", 1000))
            c_val = float(params.get("C", 1.0))
            class_weight = params.get("class_weight", "balanced")
            return LogisticRegression(
                max_iter=max_iter,
                C=c_val,
                class_weight=class_weight,
                random_state=self.random_seed,
            )

        elif self.algorithm == "hist_gradient_boosting":
            max_iter = int(params.get("max_iter", 100))
            learning_rate = float(params.get("learning_rate", 0.1))
            max_depth = params.get("max_depth", 12)
            class_weight = params.get("class_weight", "balanced")
            return HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=self.random_seed,
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, TrainingSummary]:
        """Fit estimator on training data and record real execution timing."""
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty.")

        self.model = self._build_estimator()
        logger.info("Starting training of %s on %d samples...", self.algorithm, len(X_train))

        start_time = time.perf_counter()
        self.model.fit(X_train, y_train)
        duration = time.perf_counter() - start_time

        # Extract actual fitted params
        actual_params = self.model.get_params()
        # Keep only basic serializable params
        clean_params = {
            k: v for k, v in actual_params.items()
            if isinstance(v, (int, float, str, bool, list, type(None)))
        }

        self.summary_ = TrainingSummary(
            algorithm=self.algorithm,
            task_type=self.task_type,
            hyperparameters=clean_params,
            training_duration_seconds=duration,
            train_samples=int(len(X_train)),
            feature_count=int(X_train.shape[1]),
            random_seed=self.random_seed,
            timestamp=time.time(),
        )
        logger.info("Training complete in %.2f seconds.", duration)
        return self.model, self.summary_
