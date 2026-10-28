"""Unit tests for Analytics, UI Theme Tokens, and Exploratory Data Analysis.

Tests cover:
- Design tokens and theme colors (analytics/theme.py).
- Custom CSS generation and color variable injection.
- Tabular dataset profiling and summary generation (analytics/exploratory_analysis.py).
- Feature correlation computation.
- Discrete interval traffic aggregation and benign/suspicious ratios (analytics/traffic_statistics.py).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest

from analytics.theme import THEME_COLORS, SEVERITY_COLORS, get_custom_css
from analytics.exploratory_analysis import DatasetProfiler
from analytics.traffic_statistics import TrafficStatistics
from capture.packet_capture import NormalizedPacket
from storage.database import DatabaseManager


def test_theme_color_tokens_and_palette():
    """Verify theme tokens match specified aesthetic color scheme."""
    required_keys = [
        "app_background",
        "card_background",
        "sidebar_background",
        "border",
        "text_primary",
        "text_secondary",
        "primary_accent",
        "secondary_accent",
        "critical",
        "warning",
    ]
    for k in required_keys:
        assert k in THEME_COLORS
        # Verify valid 7-character hex code #RRGGBB
        assert re.match(r"^#[0-9a-fA-F]{6}$", THEME_COLORS[k])

    # Severity color mapping
    assert "CRITICAL" in SEVERITY_COLORS
    assert "HIGH" in SEVERITY_COLORS
    assert "MEDIUM" in SEVERITY_COLORS
    assert "LOW" in SEVERITY_COLORS


def test_custom_css_generation():
    """Verify get_custom_css returns valid CSS referencing theme tokens."""
    css = get_custom_css()
    assert isinstance(css, str)
    assert "<style>" in css
    assert "</style>" in css
    assert THEME_COLORS["app_background"] in css
    assert THEME_COLORS["sidebar_background"] in css


def test_dataset_profiler_profile_dataframe():
    """Verify DatasetProfiler calculates column statistics, missingness, and types."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "feature_2": [10.0, 20.0, np.inf, 40.0, 50.0],
        "constant_col": [99, 99, 99, 99, 99],
        "Label": ["BENIGN", "BENIGN", "PortScan", "BENIGN", "PortScan"],
    })

    profile = DatasetProfiler.profile_dataframe(df)

    assert profile["total_rows"] == 5
    assert profile["total_cols"] == 4
    assert profile["missing_counts"]["feature_1"] == 1
    assert "feature_2" in profile["infinite_counts"]
    assert "constant_col" in profile["constant_columns"]
    assert profile["label_column"] == "Label"
    assert profile["class_distribution"]["BENIGN"] == 3
    assert profile["class_distribution"]["PortScan"] == 2


def test_dataset_profiler_correlation_matrix():
    """Verify correlation matrix calculation on numeric columns."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    df = pd.DataFrame({
        "x": x,
        "y": 2.0 * x + 1.0,  # Perfectly correlated
        "z": np.random.randn(50),
    })

    top_cols, corr_matrix = DatasetProfiler.compute_correlation_matrix(df)
    assert len(top_cols) == 3
    assert corr_matrix.shape == (3, 3)
    # x and y correlation is close to 1.0
    x_idx = top_cols.index("x")
    y_idx = top_cols.index("y")
    assert abs(corr_matrix[x_idx, y_idx] - 1.0) < 1e-4


def test_traffic_statistics_discrete_volume(tmp_path: Path):
    """Verify TrafficStatistics aggregates packets into discrete epoch buckets."""
    db_file = tmp_path / "traffic_stats.db"
    with DatabaseManager(db_file) as db:
        db.init_db()
        # Ingest packets across 2 distinct minute buckets
        db.insert_packet(NormalizedPacket(60.0, "10.0.0.1", "10.0.0.2", "TCP", 100, 1234, 80, "S"))
        db.insert_packet(NormalizedPacket(65.0, "10.0.0.1", "10.0.0.2", "TCP", 200, 1234, 80, "A"))
        db.insert_packet(NormalizedPacket(180.0, "10.0.0.3", "10.0.0.4", "UDP", 300, 5000, 53, ""))

        stats = TrafficStatistics(db)
        discrete_vol = stats.get_discrete_traffic_volume(bucket_seconds=60)

        assert len(discrete_vol) == 2
        # First bucket (epoch 60) has 2 packets, 300 bytes
        assert discrete_vol[0]["packet_count"] == 2
        assert discrete_vol[0]["byte_count"] == 300
        # Second bucket (epoch 180) has 1 packet, 300 bytes
        assert discrete_vol[1]["packet_count"] == 1
        assert discrete_vol[1]["byte_count"] == 300

        # Benign vs suspicious ratio
        ratios = stats.get_benign_vs_suspicious_ratio()
        assert ratios["total_packets"] == 3
        assert ratios["benign_packets"] == 3
        assert ratios["suspicious_packets"] == 0
