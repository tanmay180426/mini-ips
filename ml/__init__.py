"""Mini-IDS Machine Learning Package.

Contains modules for dataset acquisition, preprocessing, flow feature extraction,
supervised classification, unsupervised anomaly detection, model evaluation,
registry management, and explainability.
"""

from ml.feature_extraction import FlowFeatureExtractor, NetworkFlow
from ml.model_registry import ModelRegistry

__all__ = ["FlowFeatureExtractor", "NetworkFlow", "ModelRegistry"]
