"""
app.py
------
MLB Outfield Dashboard — Streamlit entry point.

Run:
    streamlit run app.py

This file contains only UI logic.
All data loading  →  src/data_loader.py
All transformation →  src/data_processor.py
All helpers        →  src/utils.py
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader   import get_combined_data, build_and_cache_data, cache_age_message, COMBINED_PATH
from data_processor import build_dashboard_df, filter_df
from player_detail  import render_player_detail_tab
from utils import (
    BUCKETS, BUCKET_LABELS, BUCKET_COLORS, SEASONS, OF_POSITIONS,
    fmt_oaa, fmt_pct, fmt_speed, fmt_int, percentile_label,
)

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MLB Outfield Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 6px; }
.kpi { background: var(--background-secondary-color, #f5f4f0);
       border: 0.5px solid var(--secondary-background-color, #e0ddd6);
       border-radius: 10px; padding: 14px 16px; }
.kpi .k-label { font-size: 11px; text-transform: uppercase; letter-spacing:.05em;
                color: #888; margin-bottom: 4px; }
.kpi .k-value { font-size: 26px; font-weight: 700; line-height: 1.1; }
.kpi .k-sub   { font-size: 11px; color: #888; margin-top: 3px; }
.kpi .green   { color: #0F6E56; }
.kpi .red     { color: #993C1D; }
.kpi .neutral { color: #888; }

/* ── Section headers ── */
.section-head {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing:.06em; color: #888; margin: 18px 0 8px;
}

/* ── Percentile pill ── */
.pct-pill {
    display: inline-block; font-size: 10px; font-weight: 600;
    padding: 1px 6px; border-radius: 8px; white-space: nowrap;
}
.pct-elite    { background:#D1FAE5; color:#065F46; }
.pct-above    { background:#DBEAFE; color:#1E40AF; }
.pct-average  { background:#FEF3C7; color:#92400E; }
.pct-below    { background:#FEE2E2; color:#991B1B; }
.pct-poor     { background:#F3F4F6; color:#6B7280; }

/* ── Info box ── */
.info-box {
    background: #EFF6FF; border-left: 3px solid #3B82F6;
    border-radius: 4px; padding: 10px 14px;
    font-size: 12px; color: #1E3A5F; margin: 8px 0;
}

/* ── Tighten sidebar ── */
[data-testid="stSidebar"] { min-width: 270px; }
</style>
""", unsafe_allow_html=True)


# ── Cached data loading (Streamlit cache layer) ───────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600 * 12)
def _cached_raw(use_local: bool):
    return get_combined_data(use_cache=use_local)


@st.cache_data(show_spinner=False)
def _cached_dashboard(cp_hash: int, ss_hash: int, _cp, _ss):
    # _cp / _ss unhashed (leading underscore); hash via len proxy above
    return build_dashboard_df(_cp, _ss)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚾ MLB Outfield")
    st.caption("Catch Probability · OAA · Sprint Speed")
    st.divider()

    # ── Data source ────────────────────────────────────────────────────────
    st.markdown("**Data Source**")
    use_local = st.checkbox("Use local cache", value=True,
                            help="Load from data/processed/outfield_dashboard_combined.parquet")

    age_msg = cache_age_message()
    if age_msg:
        st.caption(f"🕐 {age_msg}")
    elif not COMBINED_PATH.exists():
        st.warning("No cache found — data will be loaded from raw files.")

    if st.button("🔄 Refresh from internet", use_container_width=True):
        with st.spinner("Fetching from Baseball Savant…"):
            prog = st.progress(0.0)
            def _cb(frac, msg):
                prog.progress(frac, text=msg)
            build_and_cache_data(progress_callback=_cb)
            prog.empty()
        st.cache_data.clear()
        st.success("Data refreshed!")
        st.rerun()

    st.divider()

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown("**Filters**")

    selected_seasons = st.multiselect(
        "Seasons", SEASONS, default=SEASONS,
        help="Select one or more seasons to include."
    )

    selected_positions = st.multiselect(
        "OF Position", OF_POSITIONS, default=OF_POSITIONS,
        help="Filter by outfield position (from sprint-speed data)."
    )

    min_opps = st.slider(
        "Min. opportunities", 1, 120, 20,
        help="Minimum total fly-ball/line-drive opportunities for a player to appear."
    )

    player_search = st.text_input(
        "🔍 Player search", placeholder="e.g. Trout, Bader…"
    )

    st.divider()

    # ── Active bucket ──────────────────────────────────────────────────────
    st.markdown("**Active Bracket** *(table & charts)*")
    active_label = st.radio(
        "Catch probability bracket",
        BUCKET_LABELS,
        index=0,
        label_visibility="collapsed",
    )
    active_bucket = next(b for b in BUCKETS if b.label == active_label)

    st.divider()

    # ── Table options ──────────────────────────────────────────────────────
    st.markdown("**Table Options**")
    metric_choice = st.radio(
        "Primary metric",
        ["Total opportunities", "Avg opp / season"],
        index=0,
    )
    top_n = st.selectbox(
        "Show top N", [10, 20, 50, 0],
        format_func=lambda x: "All" if x == 0 else str(x),
        index=1,
    )


# ── Load & process data ───────────────────────────────────────────────────────

if not selected_seasons:
    st.warning("Please select at least one season in the sidebar.")
    st.stop()

with st.spinner("Loading data…"):
    try:
        cp_raw, ss_raw = _cached_raw(use_local)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if cp_raw.empty:
    st.error("No catch-probability data found. Check your data/raw/ directory.")
    st.stop()

# Build dashboard df (cached by frame length as cheap proxy hash)
df_all = _cached_dashboard(len(cp_raw), len(ss_raw), cp_raw, ss_raw)

# Apply filters
df = filter_df(
    df_all,
    seasons=selected_seasons,
    positions=selected_positions if selected_positions else None,
    min_opps=min_opps,
    player_query=player_search,
)

if df.empty:
    st.warning("No players match the current filters. Try relaxing the criteria.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚾ MLB Outfield Dashboard")
st.markdown(
    f"**{' · '.join(str(s) for s in sorted(selected_seasons))}**  "
    f"&nbsp;·&nbsp; {df['player_name'].nunique():,} players  "
    f"&nbsp;·&nbsp; {df['total_opps'].sum():,} total opportunities",
    unsafe_allow_html=False,
)


# ── KPI row ───────────────────────────────────────────────────────────────────

bid       = active_bucket.id
att_col   = f"b{bid}_att"
suc_col   = f"b{bid}_caught"
oaa_b_col = f"b{bid}_oaa"

total_att  = int(df[att_col].sum())          if att_col  in df.columns else 0
total_suc  = int(df[suc_col].sum())          if suc_col  in df.columns else 0
overall_pct = (total_suc / total_att * 100)  if total_att > 0 else 0.0
avg_oaa     = df["oaa"].mean()               if "oaa"    in df.columns else 0.0
avg_sprint  = df["sprint_speed"].mean()      if "sprint_speed" in df.columns else np.nan

best_row  = df.loc[df["oaa"].idxmax()] if not df["oaa"].isna().all() else None
best_name = best_row["player_name"] if best_row is not None else "—"
best_oaa  = best_row["oaa"]         if best_row is not None else 0.0

oaa_cls    = "green" if avg_oaa >= 0 else "red"
oaa_sign   = "+" if avg_oaa >= 0 else ""
spd_str    = f"{avg_sprint:.1f}" if not np.isnan(avg_sprint) else "—"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="k-label">Opportunities ({active_bucket.stars})</div>
    <div class="k-value">{total_att:,}</div>
    <div class="k-sub">bracket {active_bucket.range_label}</div>
  </div>
  <div class="kpi">
    <div class="k-label">Catch rate</div>
    <div class="k-value">{overall_pct:.1f}%</div>
    <div class="k-sub">expected {active_bucket.exp_pct}%</div>
  </div>
  <div class="kpi">
    <div class="k-label">Avg OAA (total)</div>
    <div class="k-value {oaa_cls}">{oaa_sign}{avg_oaa:.2f}</div>
    <div class="k-sub">outs above average</div>
  </div>
  <div class="kpi">
    <div class="k-label">Best OAA</div>
    <div class="k-value green">+{best_oaa:.1f}</div>
    <div class="k-sub">{best_name}</div>
  </div>
  <div class="kpi">
    <div class="k-label">Avg sprint speed</div>
    <div class="k-value">{spd_str}</div>
    <div class="k-sub">ft / second</div>
  </div>
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


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ════════════════════════════════════════════════════════════════════════════

with tab_table:
    st.markdown(
        f"<div class='section-head'>Top players — {active_bucket.label} "
        f"({active_bucket.stars})  ·  expected catch rate: {active_bucket.exp_pct}%</div>",
        unsafe_allow_html=True,
    )

    # Info box explaining avg_cp
    st.markdown("""
    <div class="info-box">
    <b>Śr. CP% (est.)</b> = estimated average catch probability <em>inside</em> this bracket for each player.
    Two players in the same bracket can face very different difficulty —
    e.g. 5% vs 22% within 1-Star (0–25%). Lower = harder plays.
    OAA = caught − Σ(individual catch probabilities) = value above model expectation.
    </div>
    """, unsafe_allow_html=True)

    bid = active_bucket.id
    att_c    = f"b{bid}_att"
    suc_c    = f"b{bid}_caught"
    pct_c    = f"b{bid}_catch_pct"
    avgcp_c  = f"b{bid}_avg_cp"
    oaa_b_c  = f"b{bid}_oaa"

    sort_col = att_c if metric_choice == "Total opportunities" else att_c
    df_sorted = df.sort_values(att_c, ascending=False)
    if top_n > 0:
        df_sorted = df_sorted.head(top_n)

    # ── Build display table ────────────────────────────────────────────────
    def _pct_pill(pct_val) -> str:
        if pd.isna(pct_val):
            return ""
        label = percentile_label(pct_val)
        css   = {
            "Elite":     "pct-elite",
            "Above avg": "pct-above",
            "Average":   "pct-average",
            "Below avg": "pct-below",
            "Poor":      "pct-poor",
        }.get(label, "pct-poor")
        return f'<span class="pct-pill {css}">{label}</span>'

    display_rows = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        n_seasons = df[df["player_name"] == row["player_name"]]["season"].nunique()
        att_val   = row.get(att_c, 0)
        if metric_choice == "Avg opp / season":
            att_val = round(att_val / max(n_seasons, 1), 1)

        display_rows.append({
            "#":             rank,
            "Player":        row.get("player_name", ""),
            "Pos":           row.get("position", "—") or "—",
            "Season":        row.get("season", ""),
            "Team":          row.get("team", "—") or "—",
            "Age":           int(row["age"]) if pd.notna(row.get("age")) else "—",
            "Opps":          int(att_val),
            "Caught":        int(row.get(suc_c, 0)),
            "Catch%":        fmt_pct(row.get(pct_c)),
            "Avg CP% (est)": fmt_pct(row.get(avgcp_c)),
            "Bracket OAA":   fmt_oaa(row.get(oaa_b_c)),
            "Total OAA":     fmt_oaa(row.get("oaa")),
            "OAA %ile":      _pct_pill(row.get("pct_oaa")),
            "Sprint ft/s":   fmt_speed(row.get("sprint_speed")),
            "Speed %ile":    _pct_pill(row.get("pct_sprint")),
        })

    display_df = pd.DataFrame(display_rows)

    # Render with HTML for pill columns
    # We use st.dataframe for the numeric columns and show pills via markdown below
    # Streamlit 1.35+ supports st.dataframe column_config for HTML; use it if possible
    try:
        st.dataframe(
            display_df,
            use_container_width=True,
            height=520,
            hide_index=True,
            column_config={
                "OAA %ile":   st.column_config.TextColumn("OAA %ile"),
                "Speed %ile": st.column_config.TextColumn("Speed %ile"),
            },
        )
    except Exception:
        st.dataframe(display_df, use_container_width=True, height=520, hide_index=True)

    st.caption(
        "**Avg CP% (est.)** is derived from the bucket's catch rate vs. its expected midpoint — "
        "it is an *estimate*, not a per-play mean.  "
        "**Bracket OAA** = OAA within this star bracket only. "
        "**Total OAA** = official Statcast OAA across all brackets."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — OAA vs SPRINT SPEED SCATTER
# ════════════════════════════════════════════════════════════════════════════

with tab_scatter:
    st.markdown("<div class='section-head'>OAA vs Sprint Speed</div>", unsafe_allow_html=True)

    scatter_df = df.dropna(subset=["oaa", "sprint_speed"]).copy()

    if scatter_df.empty:
        st.info("No players with both OAA and sprint-speed data at current filters.")
    else:
        sz_col = f"b{bid}_att"
        scatter_df["_size"] = scatter_df[sz_col].fillna(2).clip(1, 200)

        hover_extras = {}
        if avgcp_c in scatter_df.columns:
            hover_extras[avgcp_c] = ":.1f"

        fig = px.scatter(
            scatter_df,
            x="sprint_speed",
            y="oaa",
            color="season",
            size="_size",
            hover_name="player_name",
            hover_data={
                "season":        True,
                "team":          True,
                "position":      True,
                "sprint_speed":  ":.1f",
                "oaa":           ":.2f",
                sz_col:          True,
                "_size":         False,
                **hover_extras,
            },
            labels={
                "sprint_speed": "Sprint Speed (ft/s)",
                "oaa":          "OAA (Outs Above Average)",
                "season":       "Season",
                sz_col:         f"Opps ({active_bucket.label})",
                avgcp_c:        "Avg CP% (est)",
            },
            color_discrete_sequence=["#1D9E75", "#E8974A", "#4A90D9", "#A855F7"],
            title=(
                f"OAA vs Sprint Speed  ·  "
                f"bubble = opps in {active_bucket.label}"
            ),
        )

        # Reference lines
        med_speed = scatter_df["sprint_speed"].median()
        fig.add_hline(y=0,          line_dash="dash",  line_color="rgba(128,128,128,0.4)", line_width=1)
        fig.add_vline(x=med_speed,  line_dash="dot",   line_color="rgba(128,128,128,0.4)", line_width=1,
                      annotation_text=f"median {med_speed:.1f}", annotation_position="top right",
                      annotation_font_size=10)

        # Quadrant annotations
        xmax = scatter_df["sprint_speed"].max()
        xmin = scatter_df["sprint_speed"].min()
        ymax = scatter_df["oaa"].max()
        ymin = scatter_df["oaa"].min()

        for text, ax, ay in [
            ("Fast & valuable", xmax, ymax),
            ("Slow but reads well", xmin, ymax),
            ("Fast but struggles", xmax, ymin),
            ("Below avg", xmin, ymin),
        ]:
            fig.add_annotation(x=ax, y=ay, text=text,
                               showarrow=False, font=dict(size=9, color="rgba(128,128,128,0.7)"),
                               xanchor="auto", yanchor="auto")

        fig.update_layout(
            height=540,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="system-ui",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Bubble size = opportunities in the active bracket.  "
            "Top-right quadrant = fast *and* above-average OAA.  "
            "Top-left = good OAA despite below-median speed (positioning / reads)."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEASON TREND
# ════════════════════════════════════════════════════════════════════════════

with tab_trend:
    st.markdown("<div class='section-head'>Season-over-season trend</div>", unsafe_allow_html=True)

    multi_season_players = (
        df.groupby("player_name")["season"].nunique()
    )
    multi_season_players = multi_season_players[multi_season_players >= 2].index.tolist()

    if not multi_season_players:
        st.info("No players appear in 2+ seasons at current filters.")
    else:
        # Default: top 6 by |OAA| across seasons
        default_players = (
            df[df["player_name"].isin(multi_season_players)]
            .groupby("player_name")["oaa"]
            .mean()
            .abs()
            .nlargest(6)
            .index.tolist()
        )

        chosen = st.multiselect(
            "Select players to compare",
            options=sorted(multi_season_players),
            default=default_players,
        )

        if chosen:
            trend_df = df[df["player_name"].isin(chosen)].sort_values("season")

            col_oaa, col_spd = st.columns(2)

            with col_oaa:
                fig_oaa = px.line(
                    trend_df, x="season", y="oaa", color="player_name",
                    markers=True,
                    labels={"oaa": "OAA", "season": "Season", "player_name": "Player"},
                    title="OAA over time",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_oaa.add_hline(y=0, line_dash="dash",
                                  line_color="rgba(128,128,128,0.4)", line_width=1)
                fig_oaa.update_layout(
                    height=380, plot_bgcolor="white", paper_bgcolor="white",
                    font_family="system-ui",
                    xaxis=dict(tickmode="array", tickvals=SEASONS, dtick=1),
                    legend=dict(font=dict(size=10)),
                    margin=dict(t=40),
                )
                fig_oaa.update_xaxes(showgrid=True, gridcolor="#eee")
                fig_oaa.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_oaa, use_container_width=True)

            with col_spd:
                spd_data = trend_df.dropna(subset=["sprint_speed"])
                if spd_data.empty:
                    st.info("No sprint-speed data for selected players.")
                else:
                    fig_spd = px.line(
                        spd_data, x="season", y="sprint_speed", color="player_name",
                        markers=True,
                        labels={"sprint_speed": "Sprint Speed (ft/s)",
                                "season": "Season", "player_name": "Player"},
                        title="Sprint Speed over time (aging curve)",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_spd.update_layout(
                        height=380, plot_bgcolor="white", paper_bgcolor="white",
                        font_family="system-ui",
                        xaxis=dict(tickmode="array", tickvals=SEASONS, dtick=1),
                        legend=dict(font=dict(size=10)),
                        margin=dict(t=40),
                    )
                    fig_spd.update_xaxes(showgrid=True, gridcolor="#eee")
                    fig_spd.update_yaxes(showgrid=True, gridcolor="#eee")
                    st.plotly_chart(fig_spd, use_container_width=True)

            st.caption(
                "Declining sprint speed + stable OAA → player compensating with better reads. "
                "Rising OAA + stable speed → improved positioning or route efficiency."
            )

            # Bucket-level OAA trend for active bucket
            st.markdown(
                f"<div class='section-head'>{active_bucket.label} bracket OAA over time</div>",
                unsafe_allow_html=True,
            )
            oaa_b_col_trend = f"b{bid}_oaa"
            if oaa_b_col_trend in trend_df.columns:
                fig_boaa = px.line(
                    trend_df.dropna(subset=[oaa_b_col_trend]),
                    x="season", y=oaa_b_col_trend, color="player_name",
                    markers=True,
                    labels={oaa_b_col_trend: "Bracket OAA", "season": "Season",
                            "player_name": "Player"},
                    title=f"OAA in {active_bucket.label} only",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_boaa.add_hline(y=0, line_dash="dash",
                                   line_color="rgba(128,128,128,0.4)", line_width=1)
                fig_boaa.update_layout(
                    height=340, plot_bgcolor="white", paper_bgcolor="white",
                    font_family="system-ui",
                    xaxis=dict(tickmode="array", tickvals=SEASONS, dtick=1),
                    legend=dict(font=dict(size=10)),
                    margin=dict(t=40),
                )
                fig_boaa.update_xaxes(showgrid=True, gridcolor="#eee")
                fig_boaa.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_boaa, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — BUCKET BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

with tab_buckets:
    st.markdown("<div class='section-head'>Opportunity distribution by bracket</div>",
                unsafe_allow_html=True)

    col_ctrl, col_charts = st.columns([1, 3])

    with col_ctrl:
        n_players_chart = st.slider("Players to show", 5, 25, 12, key="bp_n")
        sort_by_label   = st.selectbox("Sort by bracket", BUCKET_LABELS, key="bp_sort")
        chart_season    = st.selectbox(
            "Season", ["All"] + [str(s) for s in SEASONS], key="bp_yr"
        )

    chart_df = df.copy()
    if chart_season != "All":
        chart_df = chart_df[chart_df["season"] == int(chart_season)]

    sbid      = next(b.id for b in BUCKETS if b.label == sort_by_label)
    sort_att  = f"b{sbid}_att"

    if sort_att not in chart_df.columns or chart_df.empty:
        with col_charts:
            st.info("No data for the selected season/filters.")
    else:
        top_players = (
            chart_df.sort_values(sort_att, ascending=False)
            .drop_duplicates("player_name")
            .head(n_players_chart)
            ["player_name"]
            .tolist()
        )
        plot_df = chart_df[chart_df["player_name"].isin(top_players)]

        with col_charts:
            # ── Stacked bar ────────────────────────────────────────────────
            bar_rows = []
            for b in BUCKETS:
                ac = f"b{b.id}_att"
                if ac not in plot_df.columns:
                    continue
                for _, row in plot_df.iterrows():
                    bar_rows.append({
                        "Player":  row["player_name"],
                        "Bracket": b.label,
                        "Opps":    int(row[ac]),
                    })

            bar_df = pd.DataFrame(bar_rows)
            if not bar_df.empty:
                fig_bar = px.bar(
                    bar_df, x="Opps", y="Player", color="Bracket",
                    orientation="h",
                    color_discrete_map=BUCKET_COLORS,
                    title=f"Opportunities by bracket — top {n_players_chart} sorted by {sort_by_label}",
                    labels={"Opps": "Opportunities", "Player": ""},
                )
                fig_bar.update_layout(
                    height=max(380, n_players_chart * 36 + 80),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font_family="system-ui",
                    yaxis=dict(categoryorder="total ascending"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=10)),
                    margin=dict(t=60),
                )
                fig_bar.update_xaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Avg CP heatmap ─────────────────────────────────────────────
            st.markdown(
                "<div class='section-head'>"
                "Avg CP% (est.) per player × bracket — actual difficulty inside each bucket"
                "</div>",
                unsafe_allow_html=True,
            )

            heat_rows = []
            for pname in top_players:
                prow_df = plot_df[plot_df["player_name"] == pname]
                if prow_df.empty:
                    continue
                prow = prow_df.iloc[0]
                for b in BUCKETS:
                    avc = f"b{b.id}_avg_cp"
                    heat_rows.append({
                        "Player":  pname,
                        "Bracket": b.label,
                        "Avg CP%": prow.get(avc, np.nan),
                    })

            heat_df = pd.DataFrame(heat_rows)
            if not heat_df.empty:
                pivot = heat_df.pivot(index="Player", columns="Bracket", values="Avg CP%")
                # Reorder columns to match bucket order
                pivot = pivot[[b.label for b in BUCKETS if b.label in pivot.columns]]

                z_text = [
                    [f"{v:.1f}%" if not np.isnan(v) else "—" for v in row]
                    for row in pivot.values
                ]

                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale="RdYlGn",
                    zmid=None,
                    text=z_text,
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    hovertemplate=(
                        "Player: %{y}<br>Bracket: %{x}<br>"
                        "Avg CP (est): %{z:.1f}%<extra></extra>"
                    ),
                    colorbar=dict(title="Avg CP%", thickness=14, len=0.7),
                ))
                fig_heat.update_layout(
                    height=max(320, len(top_players) * 34 + 100),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font_family="system-ui",
                    xaxis=dict(side="top"),
                    margin=dict(l=160, t=80),
                    title=(
                        "Lower value within a bracket = harder plays faced. "
                        "Compare players in the same bracket fairly."
                    ),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                st.caption(
                    "**Avg CP% (est.)** is estimated from each player's catch rate vs. the bucket midpoint. "
                    "It is not a per-play mean — it is a directional signal. "
                    "A 1-Star player with 5% avg CP faced near-impossible plays; "
                    "one with 22% faced difficult but more achievable opportunities."
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PLAYER DETAIL
# ════════════════════════════════════════════════════════════════════════════

with tab_player:
    # Pass the FULL unfiltered dataframe so the player selector shows everyone
    # and trend/peer charts have full context. Sidebar filters only apply to
    # other tabs.
    render_player_detail_tab(df_all)
