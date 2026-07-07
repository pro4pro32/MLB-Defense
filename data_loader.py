"""
data_loader.py
--------------
Responsible for loading raw data — either from local parquet files
or by fetching from Baseball Savant via pybaseball.

Call hierarchy (from app.py perspective):
  get_combined_data()          → main entry point
    ├─ load_from_cache()       → reads data/processed/outfield_dashboard_combined.parquet
    └─ build_and_cache_data()  → reads raw parquets / fetches online → saves combined
         ├─ load_raw_season()
         │    ├─ _load_parquet()   (preferred)
         │    └─ _fetch_online()   (fallback)
         └─ load_sprint_season()
              ├─ _load_parquet()
              └─ _fetch_online()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Płaskie importy
from utils import SEASONS, safe_int

log = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
COMBINED_PATH = PROCESSED_DIR / "outfield_dashboard_combined.parquet"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def get_combined_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point called by app.py.

    Returns:
        (catch_prob_df, sprint_speed_df) — raw combined DataFrames
        ready to be processed by data_processor.py.
    """
    if use_cache and COMBINED_PATH.exists():
        return load_from_cache()
    return build_and_cache_data()


def load_from_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the pre-built combined parquet from data/processed/.
    Splits it back into catch_prob and sprint_speed sub-frames.
    """
    try:
        combined = pd.read_parquet(COMBINED_PATH)
        cp = combined[combined["_table"] == "catch_prob"].drop(columns=["_table"])
        ss = combined[combined["_table"] == "sprint_speed"].drop(columns=["_table"])
        log.info("Loaded %d catch-prob rows and %d sprint-speed rows from cache.", len(cp), len(ss))
        return cp, ss
    except Exception as exc:
        log.warning("Cache read failed (%s). Rebuilding from raw files.", exc)
        return build_and_cache_data()


def build_and_cache_data(
    seasons: Optional[list[int]] = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data for every season, concatenate, save combined parquet.

    Args:
        seasons: Override which seasons to load. Defaults to src.utils.SEASONS.
        progress_callback: Optional callable(fraction, message) for Streamlit progress.
    """
    seasons = seasons or SEASONS
    cp_frames: list[pd.DataFrame] = []
    ss_frames: list[pd.DataFrame] = []

    total_steps = len(seasons) * 2
    for step, yr in enumerate(seasons):
        _report(progress_callback, step / total_steps, f"Loading catch probability {yr}…")
        cp = _load_catch_prob_season(yr)
        if cp is not None:
            cp_frames.append(cp)

        _report(progress_callback, (step * 2 + 1) / total_steps, f"Loading sprint speed {yr}…")
        ss = _load_sprint_season(yr)
        if ss is not None:
            ss_frames.append(ss)

    _report(progress_callback, 1.0, "Saving combined cache…")

    cp_all = pd.concat(cp_frames, ignore_index=True) if cp_frames else pd.DataFrame()
    ss_all = pd.concat(ss_frames, ignore_index=True) if ss_frames else pd.DataFrame()

    _save_combined(cp_all, ss_all)
    return cp_all, ss_all


# ── Per-season loaders ────────────────────────────────────────────────────────

def _load_catch_prob_season(year: int) -> Optional[pd.DataFrame]:
    """Try local parquet first, fall back to pybaseball."""
    parquet_path = RAW_DIR / f"catch_prob_{year}.parquet"
    df = _load_parquet(parquet_path)
    if df is not None:
        df["season"] = year
        return df

    log.info("No local file for catch_prob %d — fetching online.", year)
    return _fetch_catch_prob_online(year)


def _load_sprint_season(year: int) -> Optional[pd.DataFrame]:
    """Try local parquet first, fall back to pybaseball."""
    parquet_path = RAW_DIR / f"sprint_speed_{year}.parquet"
    df = _load_parquet(parquet_path)
    if df is not None:
        df["season"] = year
        return df

    log.info("No local file for sprint_speed %d — fetching online.", year)
    return _fetch_sprint_online(year)


# ── Parquet I/O ───────────────────────────────────────────────────────────────

def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        log.debug("Loaded %s (%d rows).", path.name, len(df))
        return df
    except Exception as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return None


def _save_combined(cp: pd.DataFrame, ss: pd.DataFrame) -> None:
    """Tag and concatenate both frames into a single parquet."""
    if cp.empty and ss.empty:
        return
    cp_tagged = cp.copy(); cp_tagged["_table"] = "catch_prob"
    ss_tagged = ss.copy(); ss_tagged["_table"] = "sprint_speed"
    combined  = pd.concat([cp_tagged, ss_tagged], ignore_index=True)
    combined.to_parquet(COMBINED_PATH, index=False)
    log.info("Saved combined cache → %s", COMBINED_PATH)


# ── Online fetch (pybaseball) ─────────────────────────────────────────────────

def _fetch_catch_prob_online(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import statcast_outfield_catch_prob, cache as pb_cache
        pb_cache.enable()
        df = statcast_outfield_catch_prob(year=year, min_opp=1)
        df["season"] = year
        # Persist locally for future runs
        out = RAW_DIR / f"catch_prob_{year}.parquet"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        log.info("Fetched and saved catch_prob_%d (%d rows).", year, len(df))
        return df
    except Exception as exc:
        log.error("Online fetch failed for catch_prob %d: %s", year, exc)
        return None


def _fetch_sprint_online(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import statcast_sprint_speed, cache as pb_cache
        pb_cache.enable()
        df = statcast_sprint_speed(year=year, min_opp=10)
        df["season"] = year
        out = RAW_DIR / f"sprint_speed_{year}.parquet"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        log.info("Fetched and saved sprint_speed_%d (%d rows).", year, len(df))
        return df
    except Exception as exc:
        log.error("Online fetch failed for sprint_speed %d: %s", year, exc)
        return None


# ── Utility ───────────────────────────────────────────────────────────────────

def _report(callback, fraction: float, message: str) -> None:
    if callback is not None:
        callback(fraction, message)


def cache_age_message() -> Optional[str]:
    """Return a human-readable message about how old the combined cache is."""
    if not COMBINED_PATH.exists():
        return None
    import datetime
    mtime = COMBINED_PATH.stat().st_mtime
    age   = datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime)
    hours = int(age.total_seconds() // 3600)
    if hours < 1:
        return "Cache updated less than 1 hour ago."
    if hours < 24:
        return f"Cache last updated {hours}h ago."
    days = age.days
    return f"Cache last updated {days} day{'s' if days != 1 else ''} ago."
