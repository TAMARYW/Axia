"""
config.py
---------
Central configuration for Axia.

Everything tunable lives here so no magic numbers are scattered
around the codebase:
  - News window (how many days back headlines are considered "fresh")
  - Recency half-life used to weight newer headlines more heavily
  - The credit-style rating scale (S&P-inspired) with its explanation
    dictionary, driven by the CALIBRATED stability probability.
"""

# ----------------------------------------------------------------------
# News settings
# ----------------------------------------------------------------------

# Only headlines published within the last N days are used.
NEWS_WINDOW_DAYS = 3

# Exponential-decay half-life (in days) for recency weighting.
# A headline this old counts half as much as one published right now.
RECENCY_HALF_LIFE_DAYS = 1.5

# Maximum number of headlines collected per company.
MAX_HEADLINES = 12

# ----------------------------------------------------------------------
# Model settings
# ----------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2

# The decision threshold is tuned automatically during training
# (maximizing F-beta with beta=2, i.e. recall-oriented) and stored in
# models/model_metrics.json. This value is only the fallback default.
DEFAULT_RISK_THRESHOLD = 0.20

# ----------------------------------------------------------------------
# Branding / credits (shown in the report footer)
# ----------------------------------------------------------------------

# Put the project authors here - shown as "Developed by ..." in the footer.
AUTHORS = "The Axia Team"
COPYRIGHT_HOLDER = "Axia"

# ----------------------------------------------------------------------
# Credit-style rating scale
# ----------------------------------------------------------------------
# Each entry: (min_stability_pct_inclusive, grade)
# Stability = calibrated probability that the company is financially
# stable, expressed 0-100. Evaluated top-down; first match wins.

RATING_SCALE = [
    (95, "A+"),
    (85, "A"),
    (75, "A-"),
    (60, "B+"),
    (45, "B"),
    (30, "B-"),
    (15, "C"),
    (0,  "D"),
]

# Human-readable dictionary shown in the UI and in the report legend.
RATING_DICTIONARY = {
    "A+": {
        "range":   "95-100%",
        "meaning": "Very high financial stability - minimal distress risk",
        "color":   "#16a34a",
    },
    "A": {
        "range":   "85-95%",
        "meaning": "High financial stability",
        "color":   "#22c55e",
    },
    "A-": {
        "range":   "75-85%",
        "meaning": "Good stability - low risk",
        "color":   "#4ade80",
    },
    "B+": {
        "range":   "60-75%",
        "meaning": "Reasonable stability - growth potential",
        "color":   "#facc15",
    },
    "B": {
        "range":   "45-60%",
        "meaning": "Grey zone - further review recommended",
        "color":   "#f59e0b",
    },
    "B-": {
        "range":   "30-45%",
        "meaning": "Signs of weakness - elevated risk",
        "color":   "#fb923c",
    },
    "C": {
        "range":   "15-30%",
        "meaning": "Significant financial risk",
        "color":   "#f87171",
    },
    "D": {
        "range":   "0-15%",
        "meaning": "Financial distress - very high risk",
        "color":   "#ef4444",
    },
}


def stability_to_rating(stability_pct: float) -> str:
    """Maps a calibrated stability probability (0-100) to a letter grade."""
    for threshold, grade in RATING_SCALE:
        if stability_pct >= threshold:
            return grade
    return "D"
