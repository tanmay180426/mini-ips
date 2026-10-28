"""Tests for Preprocessing Pipelines, Label Taxonomies, and Leakage Prevention."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from detection.preprocessing import DetectionPreprocessor
from training.prepare_datasets import (
    MULTICLASS_TAXONOMY,
    UNIFIED_CORE_FEATURES,
    extract_unified_features_from_row,
    map_label_to_taxonomy,
    prepare_dataset_specific,
)


def test_label_taxonomy_mappings():
    """Verify mapping of heterogeneous dataset labels into unified taxonomies."""
    # CIC-IDS2017
    b_int, b_str, m_int, m_str = map_label_to_taxonomy("BENIGN", "cicids2017")
    assert b_int == 0 and b_str == "BENIGN" and m_str == "BENIGN"

    b_int, b_str, m_int, m_str = map_label_to_taxonomy("PortScan", "cicids2017")
    assert b_int == 1 and b_str == "ATTACK" and m_str == "PORT_SCAN_RECONNAISSANCE"

    b_int, b_str, m_int, m_str = map_label_to_taxonomy("DDoS", "cicids2017")
    assert b_int == 1 and b_str == "ATTACK" and m_str == "DOS_DDOS"

    # UNSW-NB15
    b_int, b_str, m_int, m_str = map_label_to_taxonomy("Normal", "unsw_nb15")
    assert b_int == 0 and b_str == "BENIGN" and m_str == "BENIGN"

    b_int, b_str, m_int, m_str = map_label_to_taxonomy("Reconnaissance", "unsw_nb15")
    assert b_int == 1 and b_str == "ATTACK" and m_str == "PORT_SCAN_RECONNAISSANCE"

    b_int, b_str, m_int, m_str = map_label_to_taxonomy("Backdoor", "unsw_nb15")
    assert b_int == 1 and b_str == "ATTACK" and m_str == "BOTNET"

    # CSE-CIC-IDS2018
    b_int, b_str, m_int, m_str = map_label_to_taxonomy("FTP-BruteForce", "csecicids2018")
    assert b_int == 1 and b_str == "ATTACK" and m_str == "BRUTE_FORCE"


def test_unified_feature_extraction_units():
    """Verify flow duration in microseconds is properly transformed to seconds."""
    # CIC-IDS2017 row with 2,000,000 microseconds
    row_cic17 = pd.Series({
        "Flow Duration": 2_000_000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 5,
        "Total Length of Fwd Packets": 1000,
        "Total Length of Bwd Packets": 500,
        "Fwd Packet Length Mean": 100.0,
        "Bwd Packet Length Mean": 100.0,
        "Flow Packets/s": 7.5,
        "Flow Bytes/s": 750.0,
    })
    feat = extract_unified_features_from_row(row_cic17, "cicids2017")
    assert feat["flow_duration_sec"] == pytest.approx(2.0, abs=1e-3)
    assert feat["total_packets"] == 15
    assert feat["total_bytes"] == 1500

    # UNSW-NB15 row with 2.5 seconds
    row_unsw = pd.Series({
        "dur": 2.5,
        "spkts": 8,
        "dpkts": 4,
        "sbytes": 800,
        "dbytes": 400,
        "smean": 100.0,
        "dmean": 100.0,
        "rate": 4.8,
    })
    feat_unsw = extract_unified_features_from_row(row_unsw, "unsw_nb15")
    assert feat_unsw["flow_duration_sec"] == pytest.approx(2.5, abs=1e-3)
    assert feat_unsw["total_packets"] == 12
    assert feat_unsw["total_bytes"] == 1200


def test_detection_preprocessor_shape_and_order():
    """Verify DetectionPreprocessor transforms flow dictionary into aligned vector."""
    preprocessor = DetectionPreprocessor(schema_type="unified")
    flow = {
        "flow_duration": 1_000_000,
        "total_fwd_packets": 5,
        "total_backward_packets": 5,
        "total_length_of_fwd_packets": 500,
        "total_length_of_bwd_packets": 500,
        "fwd_packet_length_mean": 100.0,
        "bwd_packet_length_mean": 100.0,
        "flow_packets_s": 10.0,
        "flow_bytes_s": 1000.0,
    }
    vec = preprocessor.prepare_flow(flow)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, len(UNIFIED_CORE_FEATURES))
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_training_preprocessor_maps_live_backward_packet_alias():
    """A live spelling must supply the trained feature instead of its median."""
    from sklearn.impute import SimpleImputer
    from ml.preprocessing import FlowDataPreprocessor

    preprocessor = FlowDataPreprocessor()
    preprocessor.feature_names_ = ["total_backward_packets"]
    preprocessor.imputer = SimpleImputer(strategy="median").fit(
        pd.DataFrame({"total_backward_packets": [1.0, 5.0]})
    )
    preprocessor.scaler = None

    transformed = preprocessor.transform_inference({"total_bwd_packets": 4.0})

    assert transformed.shape == (1, 1)
    assert transformed[0, 0] == pytest.approx(4.0)
