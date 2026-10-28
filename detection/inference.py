"""Central Inference Engine for Mini-IDS.

Provides unified inference across offline files, PCAPs, and real-time telemetry:
- Schema conformance validation
- Preprocessing and scaling pipeline transformation
- Supervised binary and multiclass predictions
- Calibrated prediction confidence scores
- Anomaly scoring and outlier flagging
- Safe fallback mode on missing or corrupt model artifacts.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

try:
    import joblib
    HAS_JOBLIB = True
    JOBLIB_ERROR = ""
except ImportError as err:
    HAS_JOBLIB = False
    JOBLIB_ERROR = str(err)

try:
    import sklearn
    HAS_SKLEARN = True
    SKLEARN_ERROR = ""
except ImportError as err:
    HAS_SKLEARN = False
    SKLEARN_ERROR = str(err)

import numpy as np
import pandas as pd

from detection.preprocessing import DetectionPreprocessor

logger = logging.getLogger("mini_ids.detection.inference")


def is_ml_available() -> Tuple[bool, str]:
    """Check whether optional ML dependencies (joblib, scikit-learn) are installed.

    Returns:
        Tuple[bool, str]: (is_available, status_or_diagnostic_reason)
    """
    if not HAS_JOBLIB:
        return False, f"Missing optional dependency 'joblib': {JOBLIB_ERROR}"
    if not HAS_SKLEARN:
        return False, f"Missing optional dependency 'scikit-learn': {SKLEARN_ERROR}"
    return True, "ML dependencies (joblib, scikit-learn) are available."


def get_ml_status() -> Dict[str, Any]:
    """Return an inspectable dictionary describing ML dependency status."""
    available, reason = is_ml_available()
    return {
        "available": available,
        "reason": reason,
        "joblib": HAS_JOBLIB,
        "sklearn": HAS_SKLEARN,
        "joblib_error": JOBLIB_ERROR,
        "sklearn_error": SKLEARN_ERROR,
    }


@dataclass
class DetectionResult:
    """Structured output from inference engine."""
    is_attack: bool
    predicted_label: str
    confidence: float
    is_anomaly: bool
    anomaly_score: float
    model_name: str
    model_version: str
    severity: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CentralInferenceEngine:
    """Manages model loading, input validation, and real-time flow scoring."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        anomaly_model_path: Optional[Union[str, Path]] = None,
        schema_type: str = "canonical",
        label_mapping: Optional[Dict[int, str]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        anomaly_preprocessor_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.schema_type = schema_type
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.anomaly_model: Optional[Any] = None
        self.model_preprocessor: Optional[Any] = None
        self.anomaly_preprocessor: Optional[Any] = None
        self.model_name = "Heuristic-Fallback"
        self.model_version = "v0.0.0"
        self.label_mapping = label_mapping or {0: "BENIGN", 1: "ATTACK"}
        self.dependencies_available, dependency_diagnostic = is_ml_available()
        self.ml_diagnostic = (
            "No model/preprocessor bundle has been configured."
            if self.dependencies_available
            else dependency_diagnostic
        )
        self.last_successful_inference: Optional[float] = None
        self.inference_failures = 0

        # Attempt to load model
        if model_path and Path(model_path).exists():
            if not HAS_JOBLIB:
                logger.warning("Cannot load model %s: joblib is not installed.", model_path)
                self.ml_diagnostic = "Cannot load model: joblib is not installed."
            else:
                try:
                    self.model = joblib.load(model_path)
                    self.model_name = Path(model_path).stem
                    self.model_version = Path(model_path).stem
                    logger.info("Loaded supervised model: %s", model_path)
                except Exception as err:
                    logger.warning("Failed to load model %s: %s", model_path, err)
                    self.ml_diagnostic = f"Failed to load model: {err}"

                fitted_preprocessor_path = (
                    Path(preprocessor_path) if preprocessor_path else Path(model_path).parent / "preprocessor.joblib"
                )
                if self.model is not None and fitted_preprocessor_path.is_file() and HAS_JOBLIB:
                    try:
                        self.model_preprocessor = joblib.load(fitted_preprocessor_path)
                        if not callable(getattr(self.model_preprocessor, "transform_inference", None)):
                            raise TypeError("fitted preprocessor has no transform_inference()")
                        self.ml_diagnostic = "Model and fitted preprocessor loaded; awaiting successful inference."
                    except Exception as err:
                        self.model_preprocessor = None
                        self.ml_diagnostic = f"Failed to load compatible supervised preprocessor: {err}"
                elif self.model is not None:
                    self.ml_diagnostic = "Model loaded without its fitted preprocessor; ML inference is unavailable."

        # Attempt to load scaler
        if scaler_path and Path(scaler_path).exists():
            if not HAS_JOBLIB:
                logger.warning("Cannot load scaler %s: joblib is not installed.", scaler_path)
            else:
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Loaded feature scaler: %s", scaler_path)
                except Exception as err:
                    logger.warning("Failed to load scaler %s: %s", scaler_path, err)

        # Attempt to load anomaly detector
        if anomaly_model_path and Path(anomaly_model_path).exists():
            if not HAS_JOBLIB:
                logger.warning("Cannot load anomaly detector %s: joblib is not installed.", anomaly_model_path)
            else:
                try:
                    self.anomaly_model = joblib.load(anomaly_model_path)
                    logger.info("Loaded anomaly detector: %s", anomaly_model_path)
                except Exception as err:
                    logger.warning("Failed to load anomaly detector %s: %s", anomaly_model_path, err)
            fitted_anomaly_preprocessor_path = (
                Path(anomaly_preprocessor_path)
                if anomaly_preprocessor_path
                else Path(anomaly_model_path).parent / "preprocessor.joblib"
            )
            if self.anomaly_model is not None and fitted_anomaly_preprocessor_path.is_file() and HAS_JOBLIB:
                try:
                    self.anomaly_preprocessor = joblib.load(fitted_anomaly_preprocessor_path)
                    if not callable(getattr(self.anomaly_preprocessor, "transform_inference", None)):
                        raise TypeError("fitted anomaly preprocessor has no transform_inference()")
                    self.ml_diagnostic = "Anomaly model and fitted preprocessor loaded; awaiting successful inference."
                except Exception as err:
                    self.anomaly_preprocessor = None
                    self.ml_diagnostic = f"Failed to load compatible anomaly preprocessor: {err}"
            elif self.anomaly_model is not None:
                self.ml_diagnostic = "Anomaly model loaded without its fitted preprocessor; anomaly inference is unavailable."

        self.preprocessor = DetectionPreprocessor(
            schema_type=self.schema_type,
            scaler=self.scaler,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return engine readiness and diagnostic status."""
        model_bundle_ready = (
            self.model is not None and self.model_preprocessor is not None
        ) or (
            self.anomaly_model is not None and self.anomaly_preprocessor is not None
        )
        ml_ready = model_bundle_ready and self.last_successful_inference is not None
        return {
            "ml_available": bool(ml_ready),
            "model_bundle_loaded": bool(self.dependencies_available and model_bundle_ready),
            "ml_status": "ML_READY" if ml_ready else "ML_UNAVAILABLE",
            "ml_diagnostic": (
                "A fitted model and preprocessor completed inference successfully."
                if ml_ready
                else self.ml_diagnostic or "No fitted model/preprocessor bundle has completed inference."
            ),
            "last_successful_inference": self.last_successful_inference,
            "inference_failures": self.inference_failures,
            "has_model": self.model is not None,
            "has_scaler": self.scaler is not None,
            "has_anomaly_model": self.anomaly_model is not None,
            "has_model_preprocessor": self.model_preprocessor is not None,
            "has_anomaly_preprocessor": self.anomaly_preprocessor is not None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "schema_type": self.schema_type,
        }

    def predict_flow(self, flow: Dict[str, Any]) -> DetectionResult:
        """Execute full inference pipeline on a single network flow."""
        has_fitted_preprocessor = self.model_preprocessor is not None or self.anomaly_preprocessor is not None
        if not has_fitted_preprocessor:
            try:
                # Rule/fallback preprocessing remains separate from the fitted
                # model path and is used only when no compatible bundle exists.
                self.preprocessor.prepare_flow(flow)
            except Exception as err:
                logger.error("Feature validation failed: %s", err)
                return DetectionResult(
                    is_attack=False,
                    predicted_label="ERROR_MALFORMED",
                    confidence=0.0,
                    is_anomaly=False,
                    anomaly_score=0.0,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    severity="LOW",
                    explanation=f"Malformed input schema: {err}",
                )

        # 1. Supervised Prediction
        pred_label_str = "BENIGN"
        is_attack = False
        confidence = 0.5

        if self.model is not None:
            if self.model_preprocessor is not None:
                try:
                    X_model = self.model_preprocessor.transform_inference(flow)
                    raw_pred = self.model.predict(X_model)[0]
                    label_mapping = getattr(self.model_preprocessor, "inverse_label_mapping_", None) or self.label_mapping
                    pred_label_str = label_mapping.get(int(raw_pred), str(raw_pred))
                    is_attack = (pred_label_str.upper() != "BENIGN" and raw_pred != 0)

                    if hasattr(self.model, "predict_proba"):
                        probas = self.model.predict_proba(X_model)[0]
                        confidence = float(np.max(probas))
                    else:
                        confidence = 0.90 if is_attack else 0.85
                    self.last_successful_inference = time.time()
                    self.ml_diagnostic = "Supervised model inference completed successfully."
                except Exception as err:
                    self.inference_failures += 1
                    self.ml_diagnostic = f"Supervised inference failed: {err}"
                    logger.warning("Supervised prediction error: %s", err)

        # 2. Anomaly Scoring
        is_anomaly = False
        anomaly_score = 0.0
        if self.anomaly_model is not None and self.anomaly_preprocessor is not None:
            try:
                X_anomaly = self.anomaly_preprocessor.transform_inference(flow)
                raw_score = -float(self.anomaly_model.score_samples(X_anomaly)[0])
                anomaly_score = round(raw_score, 4)
                is_anomaly = (anomaly_score >= 0.50)
                self.last_successful_inference = time.time()
                self.ml_diagnostic = "Anomaly model inference completed successfully."
            except Exception as err:
                self.inference_failures += 1
                self.ml_diagnostic = f"Anomaly inference failed: {err}"
                logger.warning("Anomaly evaluation error: %s", err)

        # Severity Assessment
        if is_attack and confidence >= 0.85:
            severity = "HIGH"
            explanation = f"Classified as {pred_label_str} with {confidence:.1%} confidence."
        elif is_attack:
            severity = "MEDIUM"
            explanation = f"Likely {pred_label_str} (confidence: {confidence:.1%})."
        elif is_anomaly:
            severity = "MEDIUM"
            explanation = f"Statistical flow anomaly detected (score: {anomaly_score:.2f})."
        else:
            severity = "LOW"
            explanation = "Normal traffic parameters within baseline limits."

        return DetectionResult(
            is_attack=is_attack,
            predicted_label=pred_label_str,
            confidence=round(confidence, 4),
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            model_name=self.model_name,
            model_version=self.model_version,
            severity=severity,
            explanation=explanation,
        )


__all__ = ["CentralInferenceEngine", "DetectionResult", "is_ml_available", "get_ml_status"]
