"""
MLB_Defense.py
==============
MLB Outfield Catch Probability Dashboard — single-file Streamlit app.

Data:  data/raw/catch_prob_YYYY.parquet  &  data/raw/sprint_speed_YYYY.parquet
       (committed to the repo — 24 KB each, no LFS needed)

Run:
    streamlit run MLB_Defense.py
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Resolve paths relative to THIS file so the app works regardless of CWD.
# When Streamlit is launched from any directory the paths still resolve correctly.
_HERE = Path(__file__).parent.resolve()
RAW_DIR       = _HERE / "data" / "raw"
PROCESSED_DIR = _HERE / "data" / "processed"
COMBINED_PATH = PROCESSED_DIR / "outfield_dashboard_combined.parquet"

SEASONS:      list[int] = [2023, 2024, 2025, 2026]
OF_POSITIONS: list[str] = ["LF", "CF", "RF"]
SEASON_GAMES: dict[int, int] = {2023: 162, 2024: 162, 2025: 162, 2026: 80}


@dataclass(frozen=True)
class Bucket:
    id:          int    # 1–5  (1 = hardest, 5 = easiest)
    stars:       str    # e.g. "★★★★★"
    label:       str    # e.g. "5-Star (0-25%)"
    range_label: str    # e.g. "0-25%"
    col_suffix:  str    # raw parquet suffix, e.g. "5stars"
    exp_pct:     float  # expected catch % (bucket midpoint)
    color:       str    # hex for charts


# CRITICAL: Statcast names the HARDEST plays "5-star" and EASIEST "1-star".
# Confirmed empirically: 5stars → 6.5% avg catch rate; 1stars → 93.7%.
BUCKETS: list[Bucket] = [
    Bucket(1, "★★★★★", "5-Star (0-25%)",   "0-25%",   "5stars", 12.5, "#D85A30"),
    Bucket(2, "★★★★☆", "4-Star (26-50%)",  "26-50%",  "4stars", 37.5, "#E8974A"),
    Bucket(3, "★★★☆☆", "3-Star (51-75%)",  "51-75%",  "3stars", 62.5, "#C9A227"),
    Bucket(4, "★★☆☆☆", "2-Star (76-90%)",  "76-90%",  "2stars", 83.0, "#5A9E6F"),
    Bucket(5, "★☆☆☆☆", "1-Star (91-100%)", "91-100%", "1stars", 95.5, "#1D9E75"),
]
BUCKET_LABELS: list[str]      = [b.label for b in BUCKETS]
BUCKET_COLORS: dict[str, str] = {b.label: b.color for b in BUCKETS}
BUCKET_BY_ID:  dict[int, Bucket] = {b.id: b for b in BUCKETS}


# ── Type-safe coercions ───────────────────────────────────────────────────────

def safe_float(v, default: float = np.nan) -> float:
    try:    return float(v)
    except: return default

def safe_int(v, default: int = 0) -> int:
    try:    return int(v)
    except: return default


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_oaa(v) -> str:
    if pd.isna(v): return "—"
    return f"+{float(v):.2f}" if float(v) >= 0 else f"{float(v):.2f}"

def fmt_pct(v) -> str:
    return "—" if pd.isna(v) else f"{float(v):.1f}%"

def fmt_speed(v) -> str:
    return "—" if pd.isna(v) else f"{float(v):.1f} ft/s"

def fmt_int(v) -> str:
    return "—" if pd.isna(v) else f"{int(v):,}"


# ── Percentiles ───────────────────────────────────────────────────────────────

def add_percentile_col(df: pd.DataFrame, src: str, out: str,
                       ascending: bool = True) -> pd.DataFrame:
    if src not in df.columns or df[src].isna().all():
        df[out] = np.nan
        return df
    df[out] = df[src].rank(pct=True, ascending=ascending) * 100
    return df

def percentile_label(pct) -> str:
    if pd.isna(pct): return "—"
    p = float(pct)
    if p >= 90: return "Elite"
    if p >= 75: return "Above avg"
    if p >= 40: return "Average"
    if p >= 20: return "Below avg"
    return "Poor"


# ── Name normalisation ────────────────────────────────────────────────────────

def normalise_name(raw: str) -> str:
    """'Last, First' → 'First Last'. Handles Jr./Sr. suffixes."""
    if not isinstance(raw, str) or "," not in raw:
        return raw
    last, first = raw.split(",", 1)
    return f"{first.strip()} {last.strip()}"


# ── Cache age ─────────────────────────────────────────────────────────────────

def cache_age_message() -> Optional[str]:
    if not COMBINED_PATH.exists():
        return None
    age   = datetime.datetime.now() - datetime.datetime.fromtimestamp(
                COMBINED_PATH.stat().st_mtime)
    hours = int(age.total_seconds() // 3600)
    if hours < 1:  return "Cache updated less than 1 hour ago."
    if hours < 24: return f"Cache last updated {hours}h ago."
    return f"Cache last updated {age.days} day{'s' if age.days != 1 else ''} ago."


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def get_combined_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry point. Returns (catch_prob_df, sprint_speed_df)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and COMBINED_PATH.exists():
        return _load_from_cache()
    return _build_and_cache()


def _load_from_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        combined = pd.read_parquet(COMBINED_PATH)
        cp = combined[combined["_table"] == "catch_prob"].drop(columns=["_table"])
        ss = combined[combined["_table"] == "sprint_speed"].drop(columns=["_table"])
        return cp, ss
    except Exception as exc:
        log.warning("Cache read failed (%s) — rebuilding.", exc)
        return _build_and_cache()


def _build_and_cache(
    seasons: Optional[list[int]] = None,
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw parquets (or fetch online) → combine → save cache."""
    seasons = seasons or SEASONS
    cp_frames, ss_frames = [], []
    total = len(seasons) * 2

    for i, yr in enumerate(seasons):
        _report(progress_cb, i * 2 / total,       f"Loading catch probability {yr}…")
        cp = _load_cp_season(yr)
        if cp is not None: cp_frames.append(cp)

        _report(progress_cb, (i * 2 + 1) / total, f"Loading sprint speed {yr}…")
        ss = _load_ss_season(yr)
        if ss is not None: ss_frames.append(ss)

    _report(progress_cb, 1.0, "Saving cache…")
    cp_all = pd.concat(cp_frames, ignore_index=True) if cp_frames else pd.DataFrame()
    ss_all = pd.concat(ss_frames, ignore_index=True) if ss_frames else pd.DataFrame()
    _save_combined(cp_all, ss_all)
    return cp_all, ss_all


def _load_cp_season(year: int) -> Optional[pd.DataFrame]:
    path = RAW_DIR / f"catch_prob_{year}.parquet"
    df   = _read_parquet(path)
    if df is not None:
        df["season"] = year
        return df
    return _fetch_cp_online(year)


def _load_ss_season(year: int) -> Optional[pd.DataFrame]:
    path = RAW_DIR / f"sprint_speed_{year}.parquet"
    df   = _read_parquet(path)
    if df is not None:
        df["season"] = year
        return df
    return _fetch_ss_online(year)


def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return None


def _save_combined(cp: pd.DataFrame, ss: pd.DataFrame) -> None:
    if cp.empty and ss.empty:
        return
    cp_t = cp.copy(); cp_t["_table"] = "catch_prob"
    ss_t = ss.copy(); ss_t["_table"] = "sprint_speed"
    pd.concat([cp_t, ss_t], ignore_index=True).to_parquet(COMBINED_PATH, index=False)


def _fetch_cp_online(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import statcast_outfield_catch_prob, cache as pb
        pb.enable()
        df = statcast_outfield_catch_prob(year=year, min_opp=1)
        df["season"] = year
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_DIR / f"catch_prob_{year}.parquet", index=False)
        return df
    except Exception as exc:
        log.error("Online fetch failed catch_prob %d: %s", year, exc)
        return None


def _fetch_ss_online(year: int) -> Optional[pd.DataFrame]:
    try:
        from pybaseball import statcast_sprint_speed, cache as pb
        pb.enable()
        df = statcast_sprint_speed(year=year, min_opp=10)
        df["season"] = year
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_DIR / f"sprint_speed_{year}.parquet", index=False)
        return df
    except Exception as exc:
        log.error("Online fetch failed sprint_speed %d: %s", year, exc)
        return None


def _report(cb, frac: float, msg: str) -> None:
    if cb: cb(frac, msg)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def build_dashboard_df(cp_raw: pd.DataFrame, ss_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw frames → analysis-ready DataFrame (one row per player × season).

    Per-bucket columns (X = 1..5):
      bX_att       — opportunities
      bX_caught    — successful catches
      bX_catch_pct — actual catch %
      bX_exp       — expected catches (att × exp_pct/100)
      bX_oaa       — OAA contribution (caught − expected)
      bX_avg_cp    — estimated avg CP inside bucket (see _estimate_avg_cp)
    """
    if cp_raw.empty:
        return pd.DataFrame()
    cp = _process_catch_prob(cp_raw)
    ss = _process_sprint(ss_raw)
    df = _merge(cp, ss)
    df = _add_percentiles(df)
    return df


def filter_df(
    df: pd.DataFrame,
    seasons:      Optional[list[int]] = None,
    positions:    Optional[list[str]] = None,
    min_opps:     int  = 10,
    player_query: str  = "",
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if seasons:
        mask &= df["season"].isin(seasons)
    if positions:
        mask &= df["position"].isin(positions) | df["position"].isna()
    mask &= df["total_opps"] >= min_opps
    if player_query.strip():
        mask &= df["player_name"].str.lower().str.contains(
            player_query.strip().lower(), na=False)
    return df[mask].copy()


def _process_catch_prob(raw: pd.DataFrame) -> pd.DataFrame:
    df  = raw.copy()
    nc  = _find_col(df, ["last_name, first_name", "player_name", "name"])
    df["player_name"] = df[nc].apply(normalise_name) if nc else "Unknown"
    df["player_id"]   = df.get("player_id", pd.Series(np.nan, index=df.index))
    df["season"]      = df["season"].apply(safe_int)
    df["oaa"]         = df["oaa"].apply(safe_int) if "oaa" in df.columns else 0

    rows = []
    for _, row in df.iterrows():
        rec = {
            "player_name": row["player_name"],
            "player_id":   safe_int(row.get("player_id", 0)),
            "season":      safe_int(row.get("season", 0)),
            "oaa":         safe_int(row.get("oaa", 0)),
        }
        total_att = 0
        for b in BUCKETS:
            sfx      = b.col_suffix
            att      = safe_int(row.get(f"n_opp_{sfx}", 0))
            caught   = safe_int(row.get(f"n_fieldout_{sfx}", 0))
            pct_key  = f"n_{sfx[0]}star_percent"
            raw_pct  = safe_float(row.get(pct_key, np.nan))
            # Fallback: some rows have null pct but valid counts (Baseball Savant edge case)
            if np.isnan(raw_pct) and att > 0:
                raw_pct = caught / att * 100
            catch_pct = raw_pct
            expected  = att * (b.exp_pct / 100)
            oaa_b     = (caught - expected) if att > 0 else np.nan
            avg_cp    = _estimate_avg_cp(b, catch_pct) if (att > 0 and not np.isnan(catch_pct)) else np.nan

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


def _estimate_avg_cp(b: Bucket, catch_pct: float) -> float:
    """
    Estimate mean CP inside a bucket from the actual catch rate.
    avg_cp ≈ clamp(exp_pct + 0.5*(catch_pct - exp_pct), b_min, b_max)
    Labelled "est." in the UI — the aggregated data has no per-play CP.
    """
    ranges = {1:(0,25), 2:(26,50), 3:(51,75), 4:(76,90), 5:(91,100)}
    lo, hi = ranges[b.id]
    return float(np.clip(b.exp_pct + 0.5 * (catch_pct - b.exp_pct), lo, hi))


def _process_sprint(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["player_id","season","sprint_speed",
                                     "position","team","age","bolts"])
    df  = raw.copy()
    nc  = _find_col(df, ["last_name, first_name", "player_name", "name"])
    if nc:
        df["player_name_ss"] = df[nc].apply(normalise_name)
    keep = {"player_id":"player_id","sprint_speed":"sprint_speed",
            "position":"position","team":"team","age":"age","bolts":"bolts",
            "season":"season","player_name_ss":"player_name_ss"}
    avail = {k: v for k, v in keep.items() if k in df.columns}
    ss = df[list(avail.keys())].rename(columns=avail).copy()
    if "position" in ss.columns:
        ss = ss[ss["position"].isin(OF_POSITIONS)]
    for col, fn in [("sprint_speed", safe_float), ("age", safe_float),
                    ("bolts", safe_float), ("player_id", safe_int)]:
        if col in ss.columns:
            ss[col] = ss[col].apply(fn)
    return ss


def _merge(cp: pd.DataFrame, ss: pd.DataFrame) -> pd.DataFrame:
    if ss.empty:
        for c in ["sprint_speed","position","team","age","bolts"]:
            cp[c] = np.nan
        return cp
    keep = ["player_id","season","sprint_speed","position","team","age","bolts"]
    ss_s = ss[[c for c in keep if c in ss.columns]].copy()
    ss_s = (ss_s.sort_values("sprint_speed", ascending=False)
                .drop_duplicates(["player_id","season"]))
    return cp.merge(ss_s, on=["player_id","season"], how="left")


def _add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, grp in df.groupby("season"):
        grp = add_percentile_col(grp, "oaa",          "pct_oaa",    ascending=True)
        grp = add_percentile_col(grp, "sprint_speed", "pct_sprint", ascending=True)
        grp = add_percentile_col(grp, "total_opps",   "pct_opps",   ascending=True)
        frames.append(grp)
    return pd.concat(frames, ignore_index=True) if frames else df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLAYER DETAIL TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_player_detail_tab(df_all: pd.DataFrame) -> None:
    all_players = sorted(df_all["player_name"].dropna().unique().tolist())
    if not all_players:
        st.info("No player data available.")
        return

    col_sel, col_yr = st.columns([3, 1])
    with col_sel:
        try:
            latest = df_all["season"].max()
            best   = df_all[df_all["season"] == latest].sort_values("oaa", ascending=False).iloc[0]
            def_idx = all_players.index(best["player_name"])
        except Exception:
            def_idx = 0
        chosen_player = st.selectbox("Select player", all_players, index=def_idx, key="pd_player")

    with col_yr:
        player_seasons = sorted(
            df_all.loc[df_all["player_name"] == chosen_player, "season"]
            .dropna().unique().astype(int).tolist()
        )
        chosen_season = st.selectbox("Season", player_seasons,
                                     index=len(player_seasons)-1, key="pd_season")

    mask   = (df_all["player_name"] == chosen_player) & (df_all["season"] == chosen_season)
    subset = df_all[mask]
    if subset.empty:
        st.warning(f"No data for {chosen_player} in {chosen_season}.")
        return
    row = subset.iloc[0]

    # ── Header ────────────────────────────────────────────────────────────────
    pos   = row.get("position") or "OF"
    team  = row.get("team")     or "—"
    age   = int(row["age"]) if pd.notna(row.get("age")) else "—"
    oaa   = row.get("oaa", 0)
    bolts = row.get("bolts")
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:18px;flex-wrap:wrap;'>"
        f"<span style='font-size:26px;font-weight:700;'>{chosen_player}</span>"
        f"<span style='font-size:14px;color:#888;font-weight:500;'>"
        f"{pos} · {team} · Age {age} · {chosen_season}</span></div>",
        unsafe_allow_html=True,
    )
    total_att = sum(int(row.get(f"b{b.id}_att", 0) or 0) for b in BUCKETS)
    total_suc = sum(int(row.get(f"b{b.id}_caught", 0) or 0) for b in BUCKETS)
    total_exp = sum(float(row.get(f"b{b.id}_exp", 0) or 0) for b in BUCKETS)
    catch_pct = (total_suc / total_att * 100) if total_att > 0 else np.nan
    exp_pct   = (total_exp / total_att * 100) if total_att > 0 else np.nan

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    oaa_color = "#0F6E56" if oaa >= 0 else "#993C1D"
    oaa_sign  = "+" if oaa >= 0 else ""
    for col, label, val, sub, color in [
        (c1, "Total OAA",       f"{oaa_sign}{oaa}", f"{percentile_label(row.get('pct_oaa'))} ({fmt_pct(row.get('pct_oaa'))} pct)", oaa_color),
        (c2, "Opportunities",   fmt_int(total_att), "fly balls & line drives", None),
        (c3, "Sprint Speed",    fmt_speed(row.get("sprint_speed")), f"{percentile_label(row.get('pct_sprint'))} ({fmt_pct(row.get('pct_sprint'))} pct)", None),
        (c4, "Bolts",           f"{int(bolts)}" if pd.notna(bolts) else "—", "runs ≥ 30 ft/s", None),
        (c5, "Overall catch%",  fmt_pct(catch_pct), "all buckets", None),
        (c6, "Expected catch%", fmt_pct(exp_pct),   "model baseline", None),
    ]:
        col.markdown(
            f"<div class='kpi'><div class='k-label'>{label}</div>"
            f"<div class='k-value' style='{'color:'+color+';' if color else ''}'>{val}</div>"
            f"<div class='k-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Bucket table + OAA waterfall ─────────────────────────────────────────
    col_tbl, col_wf = st.columns(2)
    with col_tbl:
        st.markdown("<div class='section-head'>Per-bucket breakdown</div>", unsafe_allow_html=True)
        tbl_rows = []
        for b in BUCKETS:
            att    = int(row.get(f"b{b.id}_att", 0) or 0)
            caught = int(row.get(f"b{b.id}_caught", 0) or 0)
            cp_val = float(row.get(f"b{b.id}_catch_pct", np.nan) or np.nan)
            oaa_b  = row.get(f"b{b.id}_oaa")
            exp_t  = float(row.get(f"b{b.id}_exp", 0) or 0)
            delta  = cp_val - b.exp_pct if not np.isnan(cp_val) else np.nan
            tbl_rows.append({
                "Bracket":      b.label,
                "Opps":         att,
                "Caught":       caught,
                "Missed":       att - caught,
                "Catch%":       fmt_pct(cp_val),
                "Expected%":    fmt_pct(b.exp_pct),
                "Δ vs exp":     (f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%") if not np.isnan(delta) else "—",
                "Exp. catches": f"{exp_t:.1f}",
                "Bracket OAA":  fmt_oaa(oaa_b),
            })
        st.dataframe(pd.DataFrame(tbl_rows), hide_index=True, use_container_width=True, height=245)
        st.caption("Expected% = bucket midpoint. Bracket OAA = caught − expected.")

    with col_wf:
        st.markdown("<div class='section-head'>OAA waterfall by bracket</div>", unsafe_allow_html=True)
        wf_labels, wf_vals, wf_colors = [], [], []
        for b in BUCKETS:
            v = float(row.get(f"b{b.id}_oaa") or 0)
            wf_labels.append(b.label)
            wf_vals.append(v)
            wf_colors.append("#0F6E56" if v >= 0 else "#993C1D")
        wf_labels.append("Total OAA")
        wf_vals.append(float(oaa))
        wf_colors.append("#1D3557")
        fig_wf = go.Figure(go.Bar(
            x=wf_labels, y=wf_vals, marker_color=wf_colors,
            text=[f"{v:+.2f}" if i < len(BUCKETS) else f"{v:+.0f}" for i,v in enumerate(wf_vals)],
            textposition="outside", textfont=dict(size=11),
        ))
        fig_wf.add_hline(y=0, line_color="rgba(0,0,0,0.2)", line_width=1)
        fig_wf.update_layout(height=310, plot_bgcolor="white", paper_bgcolor="white",
                             font_family="system-ui", margin=dict(t=30,b=10,l=10,r=10),
                             yaxis=dict(showgrid=True, gridcolor="#eee", title="OAA"),
                             xaxis=dict(showgrid=False),
                             title=dict(text=f"{chosen_player} — {chosen_season}", font=dict(size=13)))
        st.plotly_chart(fig_wf, use_container_width=True)

    st.divider()

    # ── Catch% vs league ─────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Catch% vs league median — by bracket</div>", unsafe_allow_html=True)
    season_df = df_all[df_all["season"] == chosen_season]
    p_vals, l_vals, blabels = [], [], []
    for b in BUCKETS:
        pc = f"b{b.id}_catch_pct"
        p_vals.append(float(row.get(pc) or 0))
        l_vals.append(float(season_df[pc].median()) if pc in season_df.columns else 0)
        blabels.append(b.label)
    fig_vs = go.Figure()
    fig_vs.add_trace(go.Bar(name=chosen_player, x=blabels, y=p_vals,
                            marker_color="#1D9E75",
                            text=[f"{v:.1f}%" for v in p_vals],
                            textposition="outside", textfont=dict(size=10)))
    fig_vs.add_trace(go.Bar(name=f"League median ({chosen_season})", x=blabels, y=l_vals,
                            marker_color="#B4B2A9",
                            text=[f"{v:.1f}%" for v in l_vals],
                            textposition="outside", textfont=dict(size=10)))
    fig_vs.add_trace(go.Scatter(x=blabels, y=[b.exp_pct for b in BUCKETS],
                                mode="markers+lines", name="Expected",
                                marker=dict(symbol="diamond", size=8, color="#D85A30"),
                                line=dict(color="#D85A30", dash="dot", width=1.5)))
    fig_vs.update_layout(barmode="group", height=380, plot_bgcolor="white",
                         paper_bgcolor="white", font_family="system-ui",
                         yaxis=dict(title="Catch %", showgrid=True, gridcolor="#eee", range=[0,115]),
                         xaxis=dict(showgrid=False),
                         legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                     xanchor="right", x=1, font=dict(size=11)),
                         margin=dict(t=60,b=10))
    st.plotly_chart(fig_vs, use_container_width=True)
    st.caption("Green = player. Gray = league median. Red diamond = model expected. "
               "Above both gray and red = genuinely above-average in that bracket.")

    st.divider()

    # ── Simulated CP scatter ──────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Individual play simulation — CP distribution</div>",
                unsafe_allow_html=True)
    st.info(
        "⚠️ **Simulated** — the aggregated leaderboard has no per-play CP values. "
        "Points are sampled uniformly within each bracket's CP range, with outcomes "
        "proportional to the player's real bucket catch rate. "
        "Use the table above for precise numbers.",
        icon=None,
    )
    CP_RANGES = {1:(0,25), 2:(26,50), 3:(51,75), 4:(76,90), 5:(91,100)}
    rng = np.random.default_rng(seed=42)
    play_rows = []
    for b in BUCKETS:
        att    = int(row.get(f"b{b.id}_att",    0) or 0)
        caught = int(row.get(f"b{b.id}_caught", 0) or 0)
        if att == 0: continue
        lo, hi   = CP_RANGES[b.id]
        cp_vals  = np.sort(rng.uniform(lo, hi, att))[::-1]
        outcomes = ["Caught"] * caught + ["Missed"] * (att - caught)
        for cp_v, outcome in zip(cp_vals, outcomes):
            play_rows.append({"CP% (simulated)": round(cp_v, 1),
                              "Outcome": outcome, "Bracket": b.label})
    if play_rows:
        plays_df = pd.DataFrame(play_rows)
        fig_sim  = px.strip(plays_df, x="CP% (simulated)", y="Bracket", color="Outcome",
                            color_discrete_map={"Caught":"#1D9E75","Missed":"#D85A30"},
                            title=f"{chosen_player} {chosen_season} — simulated play distribution",
                            stripmode="overlay")
        for bnd in [25.5, 50.5, 75.5, 90.5]:
            fig_sim.add_vline(x=bnd, line_color="rgba(0,0,0,0.15)", line_dash="dash", line_width=1)
        fig_sim.update_traces(marker=dict(size=6, opacity=0.65))
        fig_sim.update_layout(height=max(280, len(BUCKETS)*55+80),
                              plot_bgcolor="white", paper_bgcolor="white",
                              font_family="system-ui",
                              xaxis=dict(title="Catch probability % (simulated)", range=[-2,102],
                                         showgrid=True, gridcolor="#eee"),
                              yaxis=dict(showgrid=False, categoryorder="array",
                                         categoryarray=[b.label for b in BUCKETS]),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          xanchor="right", x=1),
                              margin=dict(t=60,b=20))
        st.plotly_chart(fig_sim, use_container_width=True)
        # Summary table
        sim_rows = []
        for b in BUCKETS:
            att    = int(row.get(f"b{b.id}_att", 0) or 0)
            caught = int(row.get(f"b{b.id}_caught", 0) or 0)
            if att == 0: continue
            cp_r  = caught/att*100
            delta = cp_r - b.exp_pct
            sim_rows.append({"Bracket":b.label,"CP range":b.range_label,
                             "Plays":att,"Caught":caught,"Missed":att-caught,
                             "Actual%":f"{cp_r:.1f}%","Expected%":f"{b.exp_pct:.1f}%",
                             "Δ":f"{'+' if delta>=0 else ''}{delta:.1f}%"})
        if sim_rows:
            st.dataframe(pd.DataFrame(sim_rows), hide_index=True,
                         use_container_width=True, height=210)

    st.divider()

    # ── Season trend ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Season trend — OAA and sprint speed</div>",
                unsafe_allow_html=True)
    pdata = df_all[df_all["player_name"] == chosen_player].sort_values("season")
    if len(pdata) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_oaa = px.line(pdata, x="season", y="oaa", markers=True,
                              labels={"oaa":"Total OAA","season":"Season"},
                              title="OAA by season",
                              color_discrete_sequence=["#1D9E75"])
            fig_oaa.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
            fig_oaa.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                                  font_family="system-ui", margin=dict(t=40,b=20),
                                  xaxis=dict(tickmode="array", tickvals=pdata["season"].tolist()))
            fig_oaa.update_xaxes(showgrid=True, gridcolor="#eee")
            fig_oaa.update_yaxes(showgrid=True, gridcolor="#eee")
            st.plotly_chart(fig_oaa, use_container_width=True)
        with col_b:
            spd_data = pdata.dropna(subset=["sprint_speed"])
            if not spd_data.empty:
                fig_spd = px.line(spd_data, x="season", y="sprint_speed", markers=True,
                                  labels={"sprint_speed":"Sprint Speed (ft/s)","season":"Season"},
                                  title="Sprint speed by season",
                                  color_discrete_sequence=["#4A90D9"])
                fig_spd.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                                      font_family="system-ui", margin=dict(t=40,b=20),
                                      xaxis=dict(tickmode="array", tickvals=pdata["season"].tolist()))
                fig_spd.update_xaxes(showgrid=True, gridcolor="#eee")
                fig_spd.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_spd, use_container_width=True)
            else:
                st.info("No sprint speed data for trend.")
        # Per-bucket catch% trend
        bt_rows = []
        for _, sr in pdata.iterrows():
            for b in BUCKETS:
                pct = sr.get(f"b{b.id}_catch_pct")
                att = sr.get(f"b{b.id}_att", 0)
                if pd.notna(pct) and att and att > 0:
                    bt_rows.append({"Season":int(sr["season"]),"Bracket":b.label,
                                   "Catch%":float(pct),"Opps":int(att)})
        if bt_rows:
            btdf = pd.DataFrame(bt_rows)
            fig_bt = px.line(btdf, x="Season", y="Catch%", color="Bracket", markers=True,
                             color_discrete_map={b.label:b.color for b in BUCKETS},
                             title="Catch% per bracket over seasons", hover_data={"Opps":True})
            for b in BUCKETS:
                fig_bt.add_hline(y=b.exp_pct, line_dash="dot",
                                 line_color=b.color, opacity=0.35, line_width=1)
            fig_bt.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                                 font_family="system-ui", legend=dict(font=dict(size=10)),
                                 xaxis=dict(tickmode="array", tickvals=pdata["season"].tolist()),
                                 yaxis=dict(range=[0,110]), margin=dict(t=40))
            fig_bt.update_xaxes(showgrid=True, gridcolor="#eee")
            fig_bt.update_yaxes(showgrid=True, gridcolor="#eee")
            st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.caption(f"{chosen_player} only appears in one season — trend not available.")

    st.divider()

    # ── Peer comparison ───────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Peer comparison — same season</div>",
                unsafe_allow_html=True)
    min_peer_opps = max(10, int(row.get("total_opps", 10)) // 2)
    peers = df_all[(df_all["season"] == chosen_season) &
                   (df_all["total_opps"] >= min_peer_opps)].copy()
    if len(peers) >= 3:
        peers = peers.sort_values("oaa", ascending=False).reset_index(drop=True)
        peers["Rank"] = peers.index + 1
        pr = peers.loc[peers["player_name"] == chosen_player, "Rank"]
        if not pr.empty:
            st.markdown(f"**{chosen_player}** ranks **#{int(pr.iloc[0])}** "
                        f"of **{len(peers)}** outfielders by OAA in {chosen_season}.")
        top_peers = (pd.concat([peers.head(15),
                                peers[peers["player_name"] == chosen_player]])
                     .drop_duplicates("player_name")
                     .sort_values("oaa", ascending=True).tail(20))
        top_peers = top_peers.copy()
        top_peers["is_player"] = top_peers["player_name"] == chosen_player
        top_peers["color"]     = top_peers["is_player"].map({True:"#1D3557",False:"#B4B2A9"})
        top_peers["lbl"]       = top_peers["oaa"].apply(lambda v: f"{'+' if v>=0 else ''}{v}")
        fig_rank = go.Figure(go.Bar(
            x=top_peers["oaa"], y=top_peers["player_name"], orientation="h",
            marker_color=top_peers["color"].tolist(),
            text=top_peers["lbl"], textposition="outside", textfont=dict(size=10),
        ))
        fig_rank.add_vline(x=0, line_color="rgba(0,0,0,0.2)", line_width=1)
        fig_rank.update_layout(
            height=max(360, len(top_peers)*26+80),
            plot_bgcolor="white", paper_bgcolor="white", font_family="system-ui",
            xaxis=dict(title="OAA", showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=False),
            margin=dict(t=20,b=20,l=160,r=60),
            title=dict(text=f"OAA ranking — {chosen_season}", font=dict(size=13)),
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        # Speed vs OAA scatter
        ps = peers.dropna(subset=["sprint_speed"]).copy()
        if not ps.empty:
            ps["is_player"] = ps["player_name"] == chosen_player
            fig_ps = go.Figure()
            fig_ps.add_trace(go.Scatter(
                x=ps[~ps["is_player"]]["sprint_speed"],
                y=ps[~ps["is_player"]]["oaa"],
                mode="markers",
                marker=dict(size=7, color="#B4B2A9", opacity=0.6),
                text=ps[~ps["is_player"]]["player_name"],
                hovertemplate="<b>%{text}</b><br>Speed:%{x:.1f}<br>OAA:%{y}<extra></extra>",
                name="Peers",
            ))
            pl_s = ps[ps["is_player"]]
            fig_ps.add_trace(go.Scatter(
                x=pl_s["sprint_speed"], y=pl_s["oaa"],
                mode="markers+text",
                marker=dict(size=16, color="#1D3557", symbol="star"),
                text=[chosen_player], textposition="top center", textfont=dict(size=11),
                hovertemplate=f"<b>{chosen_player}</b><extra></extra>",
                name=chosen_player,
            ))
            med = ps["sprint_speed"].median()
            fig_ps.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.15)", line_width=1)
            fig_ps.add_vline(x=med, line_dash="dot", line_color="rgba(0,0,0,0.15)", line_width=1,
                             annotation_text=f"median {med:.1f}", annotation_position="top right",
                             annotation_font_size=9)
            fig_ps.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                                 font_family="system-ui", showlegend=False,
                                 xaxis=dict(title="Sprint Speed (ft/s)", showgrid=True, gridcolor="#eee"),
                                 yaxis=dict(title="OAA", showgrid=True, gridcolor="#eee"),
                                 margin=dict(t=30,b=20),
                                 title=dict(text=f"Speed vs OAA — {chosen_season} peer group",
                                            font=dict(size=13)))
            st.plotly_chart(fig_ps, use_container_width=True)
    else:
        st.info("Not enough peers to compare at current filters.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MLB Defense — Outfield Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:6px; }
.kpi { background:var(--background-secondary-color,#f5f4f0);
       border:0.5px solid var(--secondary-background-color,#e0ddd6);
       border-radius:10px; padding:14px 16px; }
.kpi .k-label { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:#888; margin-bottom:4px; }
.kpi .k-value { font-size:26px; font-weight:700; line-height:1.1; }
.kpi .k-sub   { font-size:11px; color:#888; margin-top:3px; }
.section-head { font-size:11px; font-weight:600; text-transform:uppercase;
                letter-spacing:.06em; color:#888; margin:18px 0 8px; }
.info-box { background:#EFF6FF; border-left:3px solid #3B82F6; border-radius:4px;
            padding:10px 14px; font-size:12px; color:#1E3A5F; margin:8px 0; }
[data-testid="stSidebar"] { min-width:270px; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600*12)
def _cached_raw(use_local: bool):
    return get_combined_data(use_cache=use_local)

@st.cache_data(show_spinner=False)
def _cached_dashboard(cp_len: int, ss_len: int, _cp, _ss):
    return build_dashboard_df(_cp, _ss)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚾ MLB Defense")
    st.caption("Outfield · Catch Probability · OAA · Sprint Speed")
    st.divider()

    st.markdown("**Data Source**")
    use_local = st.checkbox("Use local cache", value=True,
                            help="Loads data/processed/outfield_dashboard_combined.parquet")
    age_msg = cache_age_message()
    if age_msg:
        st.caption(f"🕐 {age_msg}")
    elif not COMBINED_PATH.exists():
        st.warning("No cache found — will load from raw files.")

    if st.button("🔄 Refresh from internet", use_container_width=True):
        with st.spinner("Fetching from Baseball Savant…"):
            prog = st.progress(0.0)
            _build_and_cache(progress_cb=lambda f, m: prog.progress(f, text=m))
            prog.empty()
        st.cache_data.clear()
        st.success("Refreshed!")
        st.rerun()

    st.divider()
    st.markdown("**Filters**")

    selected_seasons = st.multiselect("Seasons", SEASONS, default=SEASONS)
    selected_positions = st.multiselect("OF Position", OF_POSITIONS, default=OF_POSITIONS)
    min_opps = st.slider("Min. opportunities", 1, 120, 20,
                         help="Minimum fly-ball/line-drive opportunities.")
    player_search = st.text_input("🔍 Player search", placeholder="e.g. Trout, Bader…")

    st.divider()
    st.markdown("**Active Bracket** *(table & charts)*")
    active_label  = st.radio("Bracket", BUCKET_LABELS, index=0, label_visibility="collapsed")
    active_bucket = next(b for b in BUCKETS if b.label == active_label)

    st.divider()
    st.markdown("**Table Options**")
    metric_choice = st.radio("Primary metric", ["Total opportunities", "Avg opp / season"])
    top_n = st.selectbox("Show top N", [10, 20, 50, 0],
                         format_func=lambda x: "All" if x == 0 else str(x), index=1)


# ── Load data ─────────────────────────────────────────────────────────────────

if not selected_seasons:
    st.warning("Select at least one season in the sidebar.")
    st.stop()

with st.spinner("Loading data…"):
    try:
        cp_raw, ss_raw = _cached_raw(use_local)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if cp_raw.empty:
    st.error("No catch-probability data found. Check data/raw/ directory.")
    st.stop()

df_all = _cached_dashboard(len(cp_raw), len(ss_raw), cp_raw, ss_raw)
df     = filter_df(df_all,
                   seasons=selected_seasons,
                   positions=selected_positions or None,
                   min_opps=min_opps,
                   player_query=player_search)

if df.empty:
    st.warning("No players match the current filters.")
    st.stop()


# ── Header + KPI row ──────────────────────────────────────────────────────────

st.title("⚾ MLB Defense — Outfield Dashboard")
st.markdown(
    f"**{' · '.join(str(s) for s in sorted(selected_seasons))}**  "
    f"&nbsp;·&nbsp; {df['player_name'].nunique():,} players  "
    f"&nbsp;·&nbsp; {df['total_opps'].sum():,} total opportunities"
)

bid       = active_bucket.id
att_col   = f"b{bid}_att"
suc_col   = f"b{bid}_caught"
total_att = int(df[att_col].sum()) if att_col in df.columns else 0
total_suc = int(df[suc_col].sum()) if suc_col in df.columns else 0
ovr_pct   = (total_suc / total_att * 100) if total_att > 0 else 0.0
avg_oaa   = df["oaa"].mean()
avg_spr   = df["sprint_speed"].mean() if "sprint_speed" in df.columns else np.nan
best_row  = df.loc[df["oaa"].idxmax()] if not df["oaa"].isna().all() else None
best_name = best_row["player_name"] if best_row is not None else "—"
best_oaa  = best_row["oaa"]         if best_row is not None else 0.0
oaa_cls   = "color:#0F6E56" if avg_oaa >= 0 else "color:#993C1D"
oaa_sign  = "+" if avg_oaa >= 0 else ""
spd_str   = f"{avg_spr:.1f}" if not np.isnan(avg_spr) else "—"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="k-label">Opportunities ({active_bucket.stars})</div>
    <div class="k-value">{total_att:,}</div><div class="k-sub">bracket {active_bucket.range_label}</div></div>
  <div class="kpi"><div class="k-label">Catch rate</div>
    <div class="k-value">{ovr_pct:.1f}%</div><div class="k-sub">expected {active_bucket.exp_pct}%</div></div>
  <div class="kpi"><div class="k-label">Avg OAA</div>
    <div class="k-value" style="{oaa_cls}">{oaa_sign}{avg_oaa:.2f}</div>
    <div class="k-sub">outs above average</div></div>
  <div class="kpi"><div class="k-label">Best OAA</div>
    <div class="k-value" style="color:#0F6E56">+{best_oaa:.1f}</div>
    <div class="k-sub">{best_name}</div></div>
  <div class="kpi"><div class="k-label">Avg sprint speed</div>
    <div class="k-value">{spd_str}</div><div class="k-sub">ft / second</div></div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_table, tab_scatter, tab_trend, tab_buckets, tab_player = st.tabs([
    "📋 Leaderboard",
    "🔵 OAA vs Sprint Speed",
    "📈 Season Trend",
    "📊 Bucket Breakdown",
    "🔍 Player Detail",
])


# ── TAB 1: LEADERBOARD ────────────────────────────────────────────────────────

with tab_table:
    st.markdown(
        f"<div class='section-head'>Top players — {active_bucket.label} "
        f"({active_bucket.stars}) · expected {active_bucket.exp_pct}%</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='info-box'><b>Avg CP% (est.)</b> = estimated average catch probability "
        "<em>inside</em> this bracket. Two players in 5-Star (0-25%) might face 5% vs 22% plays "
        "— very different difficulty. Lower = harder. "
        "OAA = caught − Σ(individual catch probabilities).</div>",
        unsafe_allow_html=True,
    )
    att_c   = f"b{bid}_att"
    suc_c   = f"b{bid}_caught"
    pct_c   = f"b{bid}_catch_pct"
    avgcp_c = f"b{bid}_avg_cp"
    oaab_c  = f"b{bid}_oaa"

    df_sorted = df.sort_values(att_c, ascending=False)
    if top_n > 0:
        df_sorted = df_sorted.head(top_n)

    disp_rows = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        n_seas  = df[df["player_name"] == row["player_name"]]["season"].nunique()
        att_val = row.get(att_c, 0)
        if metric_choice == "Avg opp / season":
            att_val = round(att_val / max(n_seas, 1), 1)
        disp_rows.append({
            "#":            rank,
            "Player":       row.get("player_name", ""),
            "Pos":          row.get("position", "—") or "—",
            "Season":       str(row.get("season", "")),
            "Team":         row.get("team", "—") or "—",
            "Age":          int(row["age"]) if pd.notna(row.get("age")) else "—",
            "Opps":         int(att_val),
            "Caught":       int(row.get(suc_c, 0)),
            "Catch%":       fmt_pct(row.get(pct_c)),
            "Avg CP%(est)": fmt_pct(row.get(avgcp_c)),
            "Bracket OAA":  fmt_oaa(row.get(oaab_c)),
            "Total OAA":    fmt_oaa(row.get("oaa")),
            "OAA %ile":     percentile_label(row.get("pct_oaa")),
            "Sprint ft/s":  fmt_speed(row.get("sprint_speed")),
            "Speed %ile":   percentile_label(row.get("pct_sprint")),
        })
    st.dataframe(pd.DataFrame(disp_rows), use_container_width=True,
                 height=520, hide_index=True)
    st.caption(
        "**Avg CP% (est.)** — directional estimate, not a per-play mean. "
        "**Bracket OAA** = OAA within this bracket only. "
        "**Total OAA** = official Statcast across all brackets."
    )


# ── TAB 2: OAA vs SPRINT SPEED ────────────────────────────────────────────────

with tab_scatter:
    st.markdown("<div class='section-head'>OAA vs Sprint Speed</div>", unsafe_allow_html=True)
    sdf = df.dropna(subset=["oaa","sprint_speed"]).copy()
    if sdf.empty:
        st.info("No players with both OAA and sprint-speed data at current filters.")
    else:
        sdf["_sz"] = sdf[f"b{bid}_att"].fillna(2).clip(1, 200)
        fig_sc = px.scatter(sdf, x="sprint_speed", y="oaa", color="season", size="_sz",
                            hover_name="player_name",
                            hover_data={"season":True,"team":True,"position":True,
                                        "sprint_speed":":.1f","oaa":":.2f",
                                        f"b{bid}_att":True,"_sz":False},
                            labels={"sprint_speed":"Sprint Speed (ft/s)",
                                    "oaa":"OAA","season":"Season",
                                    f"b{bid}_att":f"Opps ({active_bucket.label})"},
                            color_discrete_sequence=["#1D9E75","#E8974A","#4A90D9","#A855F7"],
                            title=f"OAA vs Sprint Speed · bubble = opps in {active_bucket.label}")
        med = sdf["sprint_speed"].median()
        fig_sc.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,0.4)", line_width=1)
        fig_sc.add_vline(x=med, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1,
                         annotation_text=f"median {med:.1f}", annotation_position="top right",
                         annotation_font_size=10)
        for txt, ax, ay in [
            ("Fast & valuable",    sdf["sprint_speed"].max(), sdf["oaa"].max()),
            ("Slow but reads well",sdf["sprint_speed"].min(), sdf["oaa"].max()),
            ("Fast but struggles", sdf["sprint_speed"].max(), sdf["oaa"].min()),
            ("Below avg",          sdf["sprint_speed"].min(), sdf["oaa"].min()),
        ]:
            fig_sc.add_annotation(x=ax, y=ay, text=txt, showarrow=False,
                                  font=dict(size=9, color="rgba(128,128,128,0.7)"))
        fig_sc.update_layout(height=540, plot_bgcolor="white", paper_bgcolor="white",
                             font_family="system-ui",
                             legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                             margin=dict(t=60,b=40))
        fig_sc.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
        fig_sc.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Bubble = opps in active bracket. Top-right = fast & valuable. "
                   "Top-left = good OAA despite below-median speed.")


# ── TAB 3: SEASON TREND ───────────────────────────────────────────────────────

with tab_trend:
    st.markdown("<div class='section-head'>Season-over-season trend</div>", unsafe_allow_html=True)
    multi = df.groupby("player_name")["season"].nunique()
    multi = multi[multi >= 2].index.tolist()
    if not multi:
        st.info("No players with 2+ seasons at current filters.")
    else:
        defaults = (df[df["player_name"].isin(multi)].groupby("player_name")["oaa"]
                    .mean().abs().nlargest(6).index.tolist())
        chosen = st.multiselect("Select players", sorted(multi), default=defaults)
        if chosen:
            tdf = df[df["player_name"].isin(chosen)].sort_values("season")
            c1, c2 = st.columns(2)
            with c1:
                fig_t1 = px.line(tdf, x="season", y="oaa", color="player_name", markers=True,
                                 labels={"oaa":"OAA","season":"Season","player_name":"Player"},
                                 title="OAA over time",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig_t1.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,0.4)", line_width=1)
                fig_t1.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                                     font_family="system-ui",
                                     xaxis=dict(tickmode="array", tickvals=SEASONS),
                                     legend=dict(font=dict(size=10)), margin=dict(t=40))
                fig_t1.update_xaxes(showgrid=True, gridcolor="#eee")
                fig_t1.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_t1, use_container_width=True)
            with c2:
                spd_tdf = tdf.dropna(subset=["sprint_speed"])
                if not spd_tdf.empty:
                    fig_t2 = px.line(spd_tdf, x="season", y="sprint_speed", color="player_name",
                                     markers=True,
                                     labels={"sprint_speed":"Sprint Speed (ft/s)","season":"Season",
                                             "player_name":"Player"},
                                     title="Sprint Speed over time",
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_t2.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                                         font_family="system-ui",
                                         xaxis=dict(tickmode="array", tickvals=SEASONS),
                                         legend=dict(font=dict(size=10)), margin=dict(t=40))
                    fig_t2.update_xaxes(showgrid=True, gridcolor="#eee")
                    fig_t2.update_yaxes(showgrid=True, gridcolor="#eee")
                    st.plotly_chart(fig_t2, use_container_width=True)
                else:
                    st.info("No sprint-speed data for selected players.")
            oaab_col = f"b{bid}_oaa"
            if oaab_col in tdf.columns:
                fig_tb = px.line(tdf.dropna(subset=[oaab_col]),
                                 x="season", y=oaab_col, color="player_name", markers=True,
                                 labels={oaab_col:"Bracket OAA","season":"Season","player_name":"Player"},
                                 title=f"OAA in {active_bucket.label} only",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig_tb.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,0.4)", line_width=1)
                fig_tb.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                                     font_family="system-ui",
                                     xaxis=dict(tickmode="array", tickvals=SEASONS),
                                     legend=dict(font=dict(size=10)), margin=dict(t=40))
                fig_tb.update_xaxes(showgrid=True, gridcolor="#eee")
                fig_tb.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_tb, use_container_width=True)
            st.caption("Declining sprint speed + stable OAA → better reads compensating for age.")


# ── TAB 4: BUCKET BREAKDOWN ───────────────────────────────────────────────────

with tab_buckets:
    st.markdown("<div class='section-head'>Opportunity distribution by bracket</div>",
                unsafe_allow_html=True)
    ctrl, charts = st.columns([1, 3])
    with ctrl:
        n_plt    = st.slider("Players", 5, 25, 12, key="bp_n")
        sort_lbl = st.selectbox("Sort by bracket", BUCKET_LABELS, key="bp_sort")
        b_season = st.selectbox("Season", ["All"] + [str(s) for s in SEASONS], key="bp_yr")
    cdf = df.copy()
    if b_season != "All":
        cdf = cdf[cdf["season"] == int(b_season)]
    sbid     = next(b.id for b in BUCKETS if b.label == sort_lbl)
    sort_att = f"b{sbid}_att"
    if sort_att not in cdf.columns or cdf.empty:
        with charts: st.info("No data for selected filters.")
    else:
        top_pl = (cdf.sort_values(sort_att, ascending=False)
                     .drop_duplicates("player_name").head(n_plt)["player_name"].tolist())
        pdf = cdf[cdf["player_name"].isin(top_pl)]
        with charts:
            bar_r = []
            for b in BUCKETS:
                ac = f"b{b.id}_att"
                if ac not in pdf.columns: continue
                for _, r in pdf.iterrows():
                    bar_r.append({"Player":r["player_name"],"Bracket":b.label,"Opps":int(r[ac])})
            if bar_r:
                bdf = pd.DataFrame(bar_r)
                fig_bar = px.bar(bdf, x="Opps", y="Player", color="Bracket", orientation="h",
                                 color_discrete_map=BUCKET_COLORS,
                                 title=f"Opportunities by bracket — top {n_plt} sorted by {sort_lbl}",
                                 labels={"Opps":"Opportunities","Player":""})
                fig_bar.update_layout(height=max(380, n_plt*36+80),
                                      plot_bgcolor="white", paper_bgcolor="white",
                                      font_family="system-ui",
                                      yaxis=dict(categoryorder="total ascending"),
                                      legend=dict(orientation="h",yanchor="bottom",y=1.02,
                                                  xanchor="right",x=1,font=dict(size=10)),
                                      margin=dict(t=60))
                fig_bar.update_xaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Avg CP heatmap
            st.markdown("<div class='section-head'>Avg CP% (est.) — difficulty inside each bracket</div>",
                        unsafe_allow_html=True)
            heat_r = []
            for pn in top_pl:
                pr = pdf[pdf["player_name"] == pn]
                if pr.empty: continue
                pr = pr.iloc[0]
                for b in BUCKETS:
                    heat_r.append({"Player":pn,"Bracket":b.label,
                                   "Avg CP%":pr.get(f"b{b.id}_avg_cp", np.nan)})
            if heat_r:
                hdf   = pd.DataFrame(heat_r)
                pivot = hdf.pivot(index="Player", columns="Bracket", values="Avg CP%")
                pivot = pivot[[b.label for b in BUCKETS if b.label in pivot.columns]]
                z_txt = [[f"{v:.1f}%" if not np.isnan(v) else "—" for v in row]
                          for row in pivot.values]
                fig_h = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    colorscale="RdYlGn", text=z_txt, texttemplate="%{text}",
                    textfont={"size":11},
                    hovertemplate="Player:%{y}<br>Bracket:%{x}<br>Avg CP:%{z:.1f}%<extra></extra>",
                    colorbar=dict(title="Avg CP%", thickness=14, len=0.7),
                ))
                fig_h.update_layout(height=max(320, len(top_pl)*34+100),
                                    plot_bgcolor="white", paper_bgcolor="white",
                                    font_family="system-ui",
                                    xaxis=dict(side="top"), margin=dict(l=160,t=80),
                                    title="Lower = harder plays within that bracket")
                st.plotly_chart(fig_h, use_container_width=True)
                st.caption("Lower Avg CP% within a bracket means harder plays. "
                           "A 5-Star player at 5% avg CP faced near-impossible plays; "
                           "at 22% the plays were difficult but more achievable.")


# ── TAB 5: PLAYER DETAIL ──────────────────────────────────────────────────────

with tab_player:
    render_player_detail_tab(df_all)
