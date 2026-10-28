"""Mini-IDS Analytics Package.

Provides exploratory data analysis, traffic statistics aggregation,
charting helpers, and centralized visual theming.
"""

from analytics.theme import THEME_COLORS, get_altair_theme, get_custom_css

__all__ = ["THEME_COLORS", "get_altair_theme", "get_custom_css"]
