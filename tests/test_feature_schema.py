"""Unit tests for Central Feature Schema and Validation Module."""

import numpy as np
import pandas as pd
import pytest

from detection.feature_schema import (
    CANONICAL_FLOW_FEATURES,
    FeatureSchemaValidator,
)


def test_canonical_feature_count_and_uniqueness():
    """Verify exactly 28 unique canonical flow features are defined."""
    assert len(CANONICAL_FLOW_FEATURES) == 28
    assert len(set(CANONICAL_FLOW_FEATURES)) == 28
    assert "flow_duration" in CANONICAL_FLOW_FEATURES
    assert "flow_bytes_s" in CANONICAL_FLOW_FEATURES
    assert "syn_flag_count" in CANONICAL_FLOW_FEATURES


def test_header_normalization():
    """Verify raw dataset headers with whitespace and punctuation normalize properly."""
    validator = FeatureSchemaValidator()
    assert validator.normalize_column_name(" Flow Duration ") == "flow_duration"
    assert validator.normalize_column_name("Flow Bytes/s") == "flow_bytes_s"
    assert validator.normalize_column_name("Total Fwd Packets") == "total_fwd_packets"
    assert validator.normalize_column_name("Total Backward Packets") == "total_bwd_packets"
    assert validator.normalize_column_name("SYN Flag Count") == "syn_flag_count"


def test_dataframe_cleaning_nan_inf_negative():
    """Verify NaN, inf, and negative numbers are cleaned and reported accurately."""
    raw_data = {
        " Flow Duration": [10.5, np.nan, 20.0],
        "Flow Bytes/s": [1000.0, np.inf, -50.0],
        "Total Fwd Packets": [5, 10, -2],
    }
    df = pd.DataFrame(raw_data)
    X, report = FeatureSchemaValidator.validate_and_clean_dataframe(df, strict=False)

    assert X.shape == (3, 28)
    assert list(X.columns) == CANONICAL_FLOW_FEATURES

    # Check that NaN and inf were cleaned to non-negative floats
    assert not np.isnan(X.values).any()
    assert not np.isinf(X.values).any()
    assert (X.values >= 0.0).all()

    # Verify counts in report
    assert report.nan_imputed_count >= 1
    assert report.inf_imputed_count >= 1
    assert report.clipped_negative_count >= 1
    assert len(report.missing_features) > 0


def test_dataframe_strict_mode_raises_on_missing():
    """Verify strict mode raises ValueError when required features are missing."""
    df = pd.DataFrame({"flow_duration": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing required canonical features"):
        FeatureSchemaValidator.validate_and_clean_dataframe(df, strict=True)


def test_feature_dict_validation():
    """Verify single flow dictionary validation."""
    raw_dict = {
        "flow_duration": 12.5,
        "total_fwd_packets": "10",
        "flow_bytes_s": np.inf,
        "syn_flag_count": -5,
        "unrelated_field": "test",
    }
    cleaned = FeatureSchemaValidator.validate_feature_dict(raw_dict)
    assert len(cleaned) == 28
    assert cleaned["flow_duration"] == 12.5
    assert cleaned["total_fwd_packets"] == 10.0
    assert cleaned["flow_bytes_s"] == 0.0  # inf replaced
    assert cleaned["syn_flag_count"] == 0.0  # negative clipped
    assert "unrelated_field" not in cleaned
