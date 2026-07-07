"""
data_processor.py
-----------------
All data transformation logic.  No Streamlit, no I/O — pure pandas.

Public functions:
  build_dashboard_df(cp_raw, ss_raw)  → main entry point for app.py
  filter_df(df, seasons, positions, min_opps, player_query)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import (
    BUCKETS, SEASON_GAMES, OF_POSITIONS,
    safe_float, safe_int, normalise_name,
    add_percentile_column,
)

log = logging.getLogger(__name__)


# ── Column name mapping from raw parquet ─────────────────────────────────────
# Raw columns follow the pattern:  n_fieldout_Xstars, n_opp_Xstars, n_Xstar_percent
# We rename them to a clean internal schema:  bX_caught, bX_att, bX_catch_pct

def _bucket_col(star_suffix: str, kind: str) -> str:
    """Build raw parquet column name.  kind = 'fieldout' | 'opp' | 'percent'."""
    if kind == "percent":
        # e.g. n_1star_percent  (note: singular 'star' here)
        return f"n_{star_suffix[0]}star_percent"
    return f"n_{kind}_{star_suffix}"   # e.g. n_fieldout_1stars


# ── Main entry point ──────────────────────────────────────────────────────────

def build_dashboard_df(
    cp_raw: pd.DataFrame,
    ss_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform raw catch-prob and sprint-speed frames into a single,
    analysis-ready DataFrame — one row per player × season.

    Computed columns per bucket (X = 1..5):
      bX_att        int    — attempts (opportunities)
      bX_caught     int    — successful catches
      bX_catch_pct  float  — actual catch % (0–100)
      bX_exp        float  — expected catches  = att × (exp_pct/100)
      bX_oaa        float  — OAA for this bucket = caught − expected
      bX_avg_cp     float  — estimated avg CP inside bucket
                             (midpoint weighted by distance from centre)

    Top-level columns:
      player_name   str
      player_id     int
      season        int
      oaa           int    — total OAA (from source)
      total_opps    int    — sum of att across all buckets
      sprint_speed  float  — ft/s  (NaN if not available)
      position      str    — OF position from sprint-speed data (LF/CF/RF)
      team          str
      age           int
      bolts         float
      pct_oaa       float  — percentile rank (higher = better)
      pct_sprint    float  — percentile rank (higher = better)
    """
    if cp_raw.empty:
        log.warning("Empty catch-prob frame — returning empty dashboard df.")
        return pd.DataFrame()

    cp = _process_catch_prob(cp_raw)
    ss = _process_sprint(ss_raw)
    df = _merge(cp, ss)
    df = _add_percentiles(df)
    log.info("Dashboard df built: %d rows, %d columns.", len(df), len(df.columns))
    return df


# ── Filter ────────────────────────────────────────────────────────────────────

def filter_df(
    df: pd.DataFrame,
    seasons: Optional[list[int]] = None,
    positions: Optional[list[str]] = None,
    min_opps: int = 10,
    player_query: str = "",
) -> pd.DataFrame:
    """
    Apply sidebar filters.  Returns a filtered copy (never mutates input).
    """
    mask = pd.Series(True, index=df.index)

    if seasons:
        mask &= df["season"].isin(seasons)

    if positions:
        # position may be NaN for players not found in sprint-speed data
        mask &= df["position"].isin(positions) | df["position"].isna()

    mask &= df["total_opps"] >= min_opps

    if player_query.strip():
        q = player_query.strip().lower()
        mask &= df["player_name"].str.lower().str.contains(q, na=False)

    return df[mask].copy()


# ── Internal transformations ──────────────────────────────────────────────────

def _process_catch_prob(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and compute bucket-level columns from raw catch-prob frame."""
    df = raw.copy()

    # Normalise player name
    name_col = _find_col(df, ["last_name, first_name", "player_name", "name"])
    df["player_name"] = df[name_col].apply(normalise_name) if name_col else "Unknown"
    df["player_id"]   = df.get("player_id", pd.Series(np.nan, index=df.index))
    df["season"]      = df["season"].apply(safe_int)
    df["oaa"]         = df["oaa"].apply(safe_int) if "oaa" in df.columns else 0

    rows = []
    for _, row in df.iterrows():
        rec: dict = {
            "player_name": row["player_name"],
            "player_id":   safe_int(row.get("player_id", 0)),
            "season":      safe_int(row.get("season", 0)),
            "oaa":         safe_int(row.get("oaa", 0)),
        }

        total_att = 0
        for b in BUCKETS:
            sfx = b.col_suffix          # e.g. "1stars"
            att     = safe_int(row.get(f"n_opp_{sfx}", 0))
            caught  = safe_int(row.get(f"n_fieldout_{sfx}", 0))
            # raw pct column name quirk: n_1star_percent (singular)
            pct_key = f"n_{sfx[0]}star_percent"
            catch_pct = safe_float(row.get(pct_key, np.nan))

            expected = att * (b.exp_pct / 100)
            oaa_b    = (caught - expected) if att > 0 else np.nan

            # Estimate avg_cp within bucket using the actual catch% as a proxy.
            # The true avg_cp (per-play) is not in aggregated data.
            # We use a weighted blend: if catch_pct >> exp_pct → plays skewed high;
            # if catch_pct << exp_pct → plays skewed toward lower end of range.
            avg_cp = _estimate_avg_cp(b, catch_pct) if (att > 0 and not np.isnan(catch_pct)) else np.nan

            rec[f"b{b.id}_att"]       = att
            rec[f"b{b.id}_caught"]    = caught
            rec[f"b{b.id}_catch_pct"] = round(catch_pct, 1) if not np.isnan(catch_pct) else None
            rec[f"b{b.id}_avg_cp"]    = round(avg_cp, 1)    if not np.isnan(avg_cp)    else None
            rec[f"b{b.id}_exp"]       = round(expected, 2)
            rec[f"b{b.id}_oaa"]       = round(oaa_b, 2)     if not np.isnan(oaa_b)     else None

            total_att += att

        rec["total_opps"] = total_att
        rows.append(rec)

    return pd.DataFrame(rows)


def _estimate_avg_cp(bucket: object, actual_catch_pct: float) -> float:
    """
    Estimate the mean catch-probability of plays in this bucket.

    Because the aggregated leaderboard doesn't expose per-play CP values,
    we approximate using the relationship between actual catch% and the
    bucket's expected midpoint:

      avg_cp ≈ clamp(exp_pct + α × (catch_pct − exp_pct), b_min, b_max)

    where α = 0.5 dampens over-extrapolation.
    This is labelled "est." in the UI so users understand it's derived.
    """
    exp  = bucket.exp_pct
    b_min, b_max = _bucket_range(bucket.id)
    raw  = exp + 0.5 * (actual_catch_pct - exp)
    return float(np.clip(raw, b_min, b_max))


def _bucket_range(bucket_id: int) -> tuple[float, float]:
    ranges = {1: (0, 25), 2: (26, 50), 3: (51, 75), 4: (76, 90), 5: (91, 100)}
    return ranges.get(bucket_id, (0, 100))


def _process_sprint(raw: pd.DataFrame) -> pd.DataFrame:
    """Select and clean relevant sprint-speed columns."""
    if raw.empty:
        return pd.DataFrame(columns=["player_id", "season", "sprint_speed",
                                     "position", "team", "age", "bolts"])
    df = raw.copy()
    name_col = _find_col(df, ["last_name, first_name", "player_name", "name"])
    if name_col:
        df["player_name_ss"] = df[name_col].apply(normalise_name)

    keep = {
        "player_id":    "player_id",
        "sprint_speed": "sprint_speed",
        "position":     "position",
        "team":         "team",
        "age":          "age",
        "bolts":        "bolts",
        "season":       "season",
        "player_name_ss": "player_name_ss",
    }
    available = {k: v for k, v in keep.items() if k in df.columns}
    ss = df[list(available.keys())].rename(columns=available).copy()

    # Keep only outfield positions
    if "position" in ss.columns:
        ss = ss[ss["position"].isin(OF_POSITIONS)]

    ss["sprint_speed"] = ss["sprint_speed"].apply(safe_float)
    ss["age"]          = ss.get("age", pd.Series(dtype=float)).apply(safe_float)
    ss["bolts"]        = ss.get("bolts", pd.Series(dtype=float)).apply(safe_float)
    ss["player_id"]    = ss["player_id"].apply(safe_int)
    return ss


def _merge(cp: pd.DataFrame, ss: pd.DataFrame) -> pd.DataFrame:
    """Join catch-prob and sprint-speed on player_id × season."""
    if ss.empty:
        cp[["sprint_speed", "position", "team", "age", "bolts"]] = np.nan
        return cp

    ss_cols = ["player_id", "season", "sprint_speed", "position", "team", "age", "bolts"]
    ss_slim = ss[[c for c in ss_cols if c in ss.columns]].copy()
    # Deduplicate (a player can appear multiple times if traded)
    ss_slim = ss_slim.sort_values("sprint_speed", ascending=False).drop_duplicates(
        subset=["player_id", "season"]
    )

    merged = cp.merge(ss_slim, on=["player_id", "season"], how="left")
    return merged


def _add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add season-relative percentile columns for key metrics."""
    # We compute percentiles within each season group
    frames = []
    for season, grp in df.groupby("season"):
        grp = add_percentile_column(grp, "oaa",          "pct_oaa",    ascending=True)
        grp = add_percentile_column(grp, "sprint_speed", "pct_sprint", ascending=True)
        grp = add_percentile_column(grp, "total_opps",   "pct_opps",   ascending=True)
        frames.append(grp)
    return pd.concat(frames, ignore_index=True) if frames else df


# ── Utility ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
