"""
utils.py
--------
Shared constants, type-safe helpers, and formatting functions.
No Streamlit imports — this module is UI-agnostic.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ── Bucket definitions ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Bucket:
    id: int           # 1–5
    stars: str        # display label, e.g. "★★☆☆☆"
    label: str        # e.g. "2-Star (26–50%)"
    range_label: str  # e.g. "26–50%"
    col_suffix: str   # suffix used in raw parquet columns, e.g. "2stars"
    exp_pct: float    # expected catch % (bucket midpoint)
    color: str        # hex for charts


BUCKETS: list[Bucket] = [
    Bucket(1, "★☆☆☆☆", "1-Star  (0–25%)",   "0–25%",   "1stars", 12.5, "#D85A30"),
    Bucket(2, "★★☆☆☆", "2-Star (26–50%)",   "26–50%",  "2stars", 37.5, "#E8974A"),
    Bucket(3, "★★★☆☆", "3-Star (51–75%)",   "51–75%",  "3stars", 62.5, "#C9A227"),
    Bucket(4, "★★★★☆", "4-Star (76–90%)",   "76–90%",  "4stars", 83.0, "#5A9E6F"),
    Bucket(5, "★★★★★", "5-Star (91–100%)",  "91–100%", "5stars", 95.5, "#1D9E75"),
]

BUCKET_BY_ID:    dict[int, Bucket] = {b.id: b for b in BUCKETS}
BUCKET_LABELS:   list[str]         = [b.label for b in BUCKETS]
BUCKET_COLORS:   dict[str, str]    = {b.label: b.color for b in BUCKETS}

SEASONS:         list[int] = [2023, 2024, 2025, 2026]
OF_POSITIONS:    list[str] = ["LF", "CF", "RF"]

# Approximate games played per season (used for per-game normalization)
SEASON_GAMES: dict[int, int] = {
    2023: 162,
    2024: 162,
    2025: 162,
    2026: 80,   # partial season
}


# ── Type-safe coercion ────────────────────────────────────────────────────────

def safe_float(value, default: float = np.nan) -> float:
    """Coerce any value to float; return *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Coerce any value to int; return *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_oaa(value) -> str:
    """Format OAA with sign and 2 decimal places."""
    if pd.isna(value):
        return "—"
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value):.2f}"


def fmt_pct(value) -> str:
    """Format a percentage value (already in 0–100 range)."""
    if pd.isna(value):
        return "—"
    return f"{float(value):.1f}%"


def fmt_speed(value) -> str:
    """Format sprint speed in ft/s."""
    if pd.isna(value):
        return "—"
    return f"{float(value):.1f} ft/s"


def fmt_int(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"


# ── Percentile helpers ────────────────────────────────────────────────────────

def add_percentile_column(
    df: pd.DataFrame,
    source_col: str,
    out_col: str,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Add a percentile rank column (0–100) for *source_col*.
    Higher percentile = better rank when ascending=False.
    """
    if source_col not in df.columns or df[source_col].isna().all():
        df[out_col] = np.nan
        return df
    df[out_col] = df[source_col].rank(pct=True, ascending=ascending) * 100
    return df


def percentile_label(pct: float) -> str:
    """Return a human-readable tier label for a percentile."""
    if pd.isna(pct):
        return "—"
    p = float(pct)
    if p >= 90:
        return "Elite"
    if p >= 75:
        return "Above avg"
    if p >= 40:
        return "Average"
    if p >= 20:
        return "Below avg"
    return "Poor"


# ── Name normalisation ────────────────────────────────────────────────────────

def normalise_name(raw: str) -> str:
    """
    Convert "Last, First" → "First Last".
    Handles suffixes like "Jr.", "Sr.", multi-word last names.
    Falls back to the original string if the comma is absent.
    """
    if not isinstance(raw, str) or "," not in raw:
        return raw
    parts = raw.split(",", 1)
    last  = parts[0].strip()
    first = parts[1].strip()
    return f"{first} {last}"
