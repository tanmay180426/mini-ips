"""Centralized Visual Theme and Design Tokens for Mini-IDS.

Defines the project's warm neutral palette, font hierarchies,
Altair chart themes, and Streamlit custom CSS to deliver a consistent,
accessible, professional cybersecurity analytics UI.
"""

from typing import Any, Dict

# Centralized Color Palette
THEME_COLORS = {
    # Layout foundations
    "app_background": "#F6F5F0",
    "card_background": "#FFFFFF",
    "sidebar_background": "#202C3B",
    "border": "#E2E3DE",

    # Typography
    "text_primary": "#252A31",
    "text_secondary": "#737B84",
    "text_inverse": "#FFFFFF",

    # Accents & Branding
    "primary_accent": "#C76D45",    # Terracotta
    "secondary_accent": "#52877F",  # Sage / Teal

    # Alert & Status Tiers
    "critical": "#C65353",          # Crimson / Critical
    "high": "#D9534F",              # High
    "warning": "#D5A13B",           # Amber / Warning / Medium
    "low": "#52877F",               # Teal / Low
    "info": "#4A7C9B",              # Steel Blue / Informational
    "benign": "#4E8D7C",            # Calming Greenish-Teal
    "attack": "#C65353",            # Crimson

    # Surface & Highlight
    "surface_subtle": "#F0EFEA",
    "surface_highlight": "#FAF9F5",
}

SEVERITY_COLORS = {
    "CRITICAL": THEME_COLORS["critical"],
    "HIGH": THEME_COLORS["high"],
    "MEDIUM": THEME_COLORS["warning"],
    "LOW": THEME_COLORS["low"],
    "INFO": THEME_COLORS["info"],
}


def get_custom_css() -> str:
    """Return scoped custom CSS matching the Mini-IDS design specification.

    Targets standard Streamlit containers cleanly without brittle selectors.
    """
    return f"""
    <style>
    /* Global Application Shell */
    .stApp {{
        background-color: {THEME_COLORS['app_background']};
        color: {THEME_COLORS['text_primary']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    header[data-testid="stHeader"] {{
        background-color: {THEME_COLORS['app_background']} !important;
    }}

    /* Main Container & Typography */
    section.main {{
        background-color: {THEME_COLORS['app_background']};
        color: {THEME_COLORS['text_primary']};
    }}

    section.main p,
    section.main span,
    section.main label,
    section.main div[data-testid="stMarkdownContainer"] p,
    section.main div[data-testid="stMarkdownContainer"] span,
    section.main div[data-testid="stMarkdownContainer"] li {{
        color: {THEME_COLORS['text_primary']} !important;
    }}

    section.main .stCaption,
    section.main div[data-testid="stCaptionContainer"] p,
    section.main small {{
        color: {THEME_COLORS['text_secondary']} !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {THEME_COLORS['card_background']} !important;
        border: 1px solid {THEME_COLORS['border']} !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 1rem !important;
    }}

    [data-testid="stExpander"] summary {{
        background-color: #FAF9F5 !important;
        border-bottom: 1px solid {THEME_COLORS['border']} !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 0.65rem 1rem !important;
    }}

    [data-testid="stExpander"] summary:hover {{
        background-color: #F0EFEA !important;
    }}

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: {THEME_COLORS['text_primary']} !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }}

    [data-testid="stExpander"] summary svg {{
        fill: {THEME_COLORS['text_primary']} !important;
        color: {THEME_COLORS['text_primary']} !important;
    }}

    div[data-testid="stExpanderDetails"] {{
        background-color: {THEME_COLORS['card_background']} !important;
        border-radius: 0 0 6px 6px !important;
        padding: 1rem 1.25rem !important;
    }}

    /* Radio Buttons & Scrollable Adapter Selector */
    div[data-baseweb="radio"] label,
    div[data-baseweb="radio"] p,
    div[data-baseweb="radio"] span,
    div[data-baseweb="radio"] div {{
        color: {THEME_COLORS['text_primary']} !important;
        font-size: 0.88rem !important;
    }}

    div[data-baseweb="radio"] input:checked + div {{
        border-color: {THEME_COLORS['primary_accent']} !important;
        background-color: {THEME_COLORS['primary_accent']} !important;
    }}

    div[data-testid="stExpanderDetails"] div[data-testid="stRadio"] > div[role="radiogroup"] {{
        max-height: 240px;
        overflow-y: auto;
        border: 1px solid {THEME_COLORS['border']};
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        background-color: #FAF9F5;
    }}

    /* Form Inputs */
    section.main .stTextInput input,
    section.main .stNumberInput input {{
        background-color: #FFFFFF !important;
        color: {THEME_COLORS['text_primary']} !important;
        border: 1px solid {THEME_COLORS['border']} !important;
        border-radius: 4px !important;
    }}

    section.main div[data-baseweb="select"] {{
        background-color: #FFFFFF !important;
        color: {THEME_COLORS['text_primary']} !important;
        border: 1px solid {THEME_COLORS['border']} !important;
        border-radius: 4px !important;
    }}

    section.main div[data-baseweb="select"] * {{
        color: {THEME_COLORS['text_primary']} !important;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {THEME_COLORS['sidebar_background']} !important;
        color: #E2E8F0 !important;
        border-right: 1px solid {THEME_COLORS['border']};
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: #E2E8F0 !important;
    }}

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stNumberInput label {{
        color: #CBD5E1 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: {THEME_COLORS['text_primary']};
        font-weight: 600;
        letter-spacing: -0.015em;
    }}

    /* Cards / Surfaces */
    .ids-card {{
        background-color: {THEME_COLORS['card_background']};
        border: 1px solid {THEME_COLORS['border']};
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}

    .ids-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid {THEME_COLORS['border']};
        padding-bottom: 0.5rem;
    }}

    .ids-card-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {THEME_COLORS['text_primary']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Monospace for IPs, Ports, Timestamps */
    .ids-mono {{
        font-family: "JetBrains Mono", "SF Mono", "Consolas", "Courier New", monospace;
        font-size: 0.88rem;
    }}

    /* Severity Badges */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .badge-critical {{
        background-color: #FEE2E2;
        color: {THEME_COLORS['critical']};
        border: 1px solid #FCA5A5;
    }}
    .badge-high {{
        background-color: #FFEDD5;
        color: {THEME_COLORS['high']};
        border: 1px solid #FDBA74;
    }}
    .badge-warning {{
        background-color: #FEF3C7;
        color: {THEME_COLORS['warning']};
        border: 1px solid #FCD34D;
    }}
    .badge-low {{
        background-color: #E6FFFA;
        color: {THEME_COLORS['low']};
        border: 1px solid #99F6E4;
    }}
    .badge-benign {{
        background-color: #ECFDF5;
        color: {THEME_COLORS['benign']};
        border: 1px solid #A7F3D0;
    }}

    /* Metrics override */
    [data-testid="stMetricValue"] {{
        color: {THEME_COLORS['text_primary']} !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {THEME_COLORS['text_secondary']} !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}

    /* Buttons */
    button[kind="primary"] {{
        background-color: {THEME_COLORS['primary_accent']} !important;
        border-color: {THEME_COLORS['primary_accent']} !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
    }}
    button[kind="secondary"] {{
        background-color: #FFFFFF !important;
        border: 1px solid {THEME_COLORS['border']} !important;
        color: {THEME_COLORS['text_primary']} !important;
        border-radius: 4px !important;
    }}

    /* Dataframe tables */
    [data-testid="stDataFrame"] {{
        border: 1px solid {THEME_COLORS['border']};
        border-radius: 4px;
        background-color: {THEME_COLORS['card_background']};
    }}
    </style>
    """


def get_altair_theme() -> Dict[str, Any]:
    """Return Altair theme configuration dictionary using the project palette."""
    return {
        "config": {
            "background": THEME_COLORS["card_background"],
            "view": {
                "stroke": THEME_COLORS["border"],
                "strokeWidth": 1,
            },
            "title": {
                "color": THEME_COLORS["text_primary"],
                "fontSize": 13,
                "fontWeight": 600,
                "anchor": "start",
                "offset": 10,
            },
            "axis": {
                "domainColor": THEME_COLORS["border"],
                "gridColor": "#EDEDE8",
                "labelColor": THEME_COLORS["text_secondary"],
                "labelFontSize": 11,
                "titleColor": THEME_COLORS["text_primary"],
                "titleFontSize": 11,
                "titleFontWeight": 600,
                "tickColor": THEME_COLORS["border"],
            },
            "legend": {
                "labelColor": THEME_COLORS["text_secondary"],
                "titleColor": THEME_COLORS["text_primary"],
                "titleFontSize": 11,
                "labelFontSize": 10,
            },
        }
    }
