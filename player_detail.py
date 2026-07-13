"""
player_detail.py
----------------
Renders the "Player Detail" tab in app.py.

Imported as:
    from src.player_detail import render_player_detail_tab

What this shows
---------------
Since the aggregated leaderboard data (catch_prob_YYYY.parquet) does NOT
contain per-play catch-probability values — only bucket-level totals — we
build the most honest and informative picture possible from what we have:

  1. Header card    — name, team, position, age, sprint speed, total OAA
  2. Bucket table   — for each of the 5 CP brackets:
                        opportunities, caught, actual catch%, expected catch%,
                        delta vs expected, OAA contribution
  3. Waterfall chart — OAA contribution per bucket (where the value comes from)
  4. Catch% vs League-avg bar chart — player vs median per bucket
  5. CP-range scatter (estimated) — one simulated point per opportunity,
       x = uniform random within bucket CP range, y = 0/1 (missed/caught),
       clearly labelled as a SIMULATION for illustrative purposes only.
  6. Season trend   — OAA and bucket-breakdown evolution over available seasons.
  7. Peer comparison — same season, same position, minimum same total_opps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    BUCKETS, fmt_oaa, fmt_pct, fmt_speed, fmt_int,
    percentile_label,
)


# ── Public entry point ────────────────────────────────────────────────────────

def render_player_detail_tab(df_all: pd.DataFrame) -> None:
    """
    Render the full Player Detail tab.
    df_all — the unfiltered dashboard DataFrame (all seasons, all players).
    """

    # ── Player selector ───────────────────────────────────────────────────────
    all_players = sorted(df_all["player_name"].dropna().unique().tolist())
    if not all_players:
        st.info("No player data available.")
        return

    col_sel, col_yr = st.columns([3, 1])
    with col_sel:
        chosen_player = st.selectbox(
            "Select player",
            all_players,
            index=_default_player_index(all_players, df_all),
            key="pd_player",
        )
    with col_yr:
        player_seasons = sorted(
            df_all.loc[df_all["player_name"] == chosen_player, "season"]
            .dropna().unique().astype(int).tolist()
        )
        chosen_season = st.selectbox(
            "Season",
            player_seasons,
            index=len(player_seasons) - 1,   # default to most recent
            key="pd_season",
        )

    # ── Fetch player row ──────────────────────────────────────────────────────
    mask   = (df_all["player_name"] == chosen_player) & (df_all["season"] == chosen_season)
    subset = df_all[mask]
    if subset.empty:
        st.warning(f"No data for {chosen_player} in {chosen_season}.")
        return
    row = subset.iloc[0]

    # ── Section 1: Header card ────────────────────────────────────────────────
    _render_header(row, chosen_season)
    st.divider()

    # ── Section 2: Bucket breakdown table + OAA waterfall ─────────────────────
    col_tbl, col_wf = st.columns([1, 1])
    with col_tbl:
        _render_bucket_table(row)
    with col_wf:
        _render_oaa_waterfall(row, chosen_player, chosen_season)

    st.divider()

    # ── Section 3: Catch% vs league average ───────────────────────────────────
    _render_vs_league(row, df_all, chosen_player, chosen_season)
    st.divider()

    # ── Section 4: Simulated CP scatter ───────────────────────────────────────
    _render_cp_scatter(row, chosen_player, chosen_season)
    st.divider()

    # ── Section 5: Season trend for this player ───────────────────────────────
    _render_player_trend(df_all, chosen_player)
    st.divider()

    # ── Section 6: Peer comparison ────────────────────────────────────────────
    _render_peer_comparison(df_all, row, chosen_player, chosen_season)


# ── Section renderers ─────────────────────────────────────────────────────────

def _render_header(row: pd.Series, season: int) -> None:
    """Big KPI cards at the top of the player page."""
    name     = row.get("player_name", "Unknown")
    pos      = row.get("position")   or "OF"
    team     = row.get("team")       or "—"
    age      = int(row["age"]) if pd.notna(row.get("age")) else "—"
    oaa      = row.get("oaa", 0)
    speed    = row.get("sprint_speed")
    tot_opp  = int(row.get("total_opps", 0))
    pct_oaa  = row.get("pct_oaa")
    pct_spd  = row.get("pct_sprint")
    bolts    = row.get("bolts")

    oaa_color = "#0F6E56" if oaa >= 0 else "#993C1D"
    oaa_sign  = "+" if oaa >= 0 else ""

    st.markdown(f"""
    <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:18px; flex-wrap:wrap;">
        <span style="font-size:26px; font-weight:700;">{name}</span>
        <span style="font-size:14px; color:#888; font-weight:500;">{pos} · {team} · Age {age} · {season}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _kpi(c1, "Total OAA", f"{oaa_sign}{oaa}", f"{percentile_label(pct_oaa)} ({fmt_pct(pct_oaa)} pct)",
         "#0F6E56" if oaa >= 0 else "#993C1D")
    _kpi(c2, "Opportunities", fmt_int(tot_opp), "fly balls & line drives")
    _kpi(c3, "Sprint Speed", fmt_speed(speed), f"{percentile_label(pct_spd)} ({fmt_pct(pct_spd)} pct)")
    _kpi(c4, "Bolts (season)", f"{int(bolts)}" if pd.notna(bolts) else "—",
         "runs ≥ 30 ft/s")
    # Derived: overall catch%
    total_att = sum(int(row.get(f"b{b.id}_att", 0)) for b in BUCKETS)
    total_suc = sum(int(row.get(f"b{b.id}_caught", 0)) for b in BUCKETS)
    catch_pct = (total_suc / total_att * 100) if total_att > 0 else np.nan
    _kpi(c5, "Overall catch%", fmt_pct(catch_pct), "all buckets combined")
    # Expected overall catch%
    total_exp = sum(float(row.get(f"b{b.id}_exp", 0) or 0) for b in BUCKETS)
    exp_pct   = (total_exp / total_att * 100) if total_att > 0 else np.nan
    _kpi(c6, "Expected catch%", fmt_pct(exp_pct), "model baseline")


def _render_bucket_table(row: pd.Series) -> None:
    """Detailed per-bucket table for one player."""
    st.markdown("<div class='section-head'>Per-bucket breakdown</div>", unsafe_allow_html=True)

    table_rows = []
    for b in BUCKETS:
        att      = int(row.get(f"b{b.id}_att",      0) or 0)
        caught   = int(row.get(f"b{b.id}_caught",   0) or 0)
        catch_p  = float(row.get(f"b{b.id}_catch_pct", np.nan) or np.nan)
        oaa_b    = row.get(f"b{b.id}_oaa")
        exp_tot  = float(row.get(f"b{b.id}_exp",    0) or 0)

        missed   = att - caught
        delta    = catch_p - b.exp_pct if not np.isnan(catch_p) else np.nan

        table_rows.append({
            "Bracket":        b.label,
            "CP range":       b.range_label,
            "Opps":           att,
            "Caught":         caught,
            "Missed":         missed,
            "Catch%":         fmt_pct(catch_p),
            "Expected%":      fmt_pct(b.exp_pct),
            "Δ vs exp":       (f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%")
                              if not np.isnan(delta) else "—",
            "Exp. catches":   f"{exp_tot:.1f}",
            "Bracket OAA":    fmt_oaa(oaa_b),
        })

    tbl_df = pd.DataFrame(table_rows)
    st.dataframe(tbl_df, hide_index=True, use_container_width=True, height=245)

    st.caption(
        "**Opps** = fly balls / line drives directed to this player in this CP bracket.  "
        "**Expected%** = the mid-point of the bracket (model baseline).  "
        "**Bracket OAA** = caught − Σ(individual catch probabilities, estimated from bucket midpoint)."
    )


def _render_oaa_waterfall(row: pd.Series, player: str, season: int) -> None:
    """Waterfall chart: OAA contribution by bucket."""
    st.markdown("<div class='section-head'>OAA contribution by bracket</div>",
                unsafe_allow_html=True)

    labels, values, colors = [], [], []
    running = 0.0
    for b in BUCKETS:
        oaa_b = row.get(f"b{b.id}_oaa")
        if oaa_b is None or (isinstance(oaa_b, float) and np.isnan(oaa_b)):
            oaa_b = 0.0
        oaa_b = float(oaa_b)
        labels.append(b.label)
        values.append(oaa_b)
        colors.append("#0F6E56" if oaa_b >= 0 else "#993C1D")
        running += oaa_b

    total_oaa = float(row.get("oaa", running))
    labels.append("Total OAA")
    values.append(total_oaa)
    colors.append("#1D3557")

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:+.2f}" if i < len(BUCKETS) else f"{v:+.0f}"
              for i, v in enumerate(values)],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.2)", line_width=1)
    fig.update_layout(
        height=310,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="system-ui",
        margin=dict(t=30, b=10, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False,
                   title="OAA contribution"),
        xaxis=dict(showgrid=False),
        title=dict(text=f"{player} — {season}", font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_vs_league(
    row: pd.Series,
    df_all: pd.DataFrame,
    player: str,
    season: int,
) -> None:
    """Grouped bar: player catch% vs league median per bucket."""
    st.markdown("<div class='section-head'>Catch% vs league median — by bracket</div>",
                unsafe_allow_html=True)

    season_df = df_all[df_all["season"] == season]
    player_vals, league_vals, bucket_labels = [], [], []

    for b in BUCKETS:
        pct_col = f"b{b.id}_catch_pct"
        player_pct = row.get(pct_col)
        league_med = season_df[pct_col].median() if pct_col in season_df.columns else np.nan

        player_vals.append(float(player_pct) if pd.notna(player_pct) else 0.0)
        league_vals.append(float(league_med) if pd.notna(league_med) else 0.0)
        bucket_labels.append(b.label)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=player, x=bucket_labels, y=player_vals,
        marker_color="#1D9E75",
        text=[f"{v:.1f}%" for v in player_vals],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.add_trace(go.Bar(
        name=f"League median ({season})", x=bucket_labels, y=league_vals,
        marker_color="#B4B2A9",
        text=[f"{v:.1f}%" for v in league_vals],
        textposition="outside", textfont=dict(size=10),
    ))
    # Expected lines as scatter
    fig.add_trace(go.Scatter(
        x=bucket_labels,
        y=[b.exp_pct for b in BUCKETS],
        mode="markers+lines",
        name="Expected (midpoint)",
        marker=dict(symbol="diamond", size=8, color="#D85A30"),
        line=dict(color="#D85A30", dash="dot", width=1.5),
    ))
    fig.update_layout(
        barmode="group",
        height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="system-ui",
        yaxis=dict(title="Catch %", showgrid=True, gridcolor="#eee", range=[0, 115]),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        margin=dict(t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Green = this player's catch rate per bracket. "
        "Gray = league median for the same season. "
        "Red diamond = model-expected catch rate (bucket midpoint). "
        "A player above gray AND above red = genuinely above-average defender in that bracket."
    )


def _render_cp_scatter(row: pd.Series, player: str, season: int) -> None:
    """
    Simulate individual plays within each bucket and plot them.

    Because the aggregated leaderboard does not expose per-play CP values,
    we SIMULATE them by drawing uniform random CP values within each bucket's
    CP range, then assigning outcomes (caught/missed) proportionally so that
    the simulated catch rate matches the player's actual bucket catch rate.

    This is clearly labelled as a simulation — it shows the *type* and
    *distribution* of plays faced, not the exact plays.
    """
    st.markdown(
        "<div class='section-head'>Individual play simulation — CP distribution</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "⚠️ **Simulated data** — the aggregated leaderboard does not contain per-play "
        "catch-probability values. The points below are randomly sampled within each "
        "bracket's CP range so that the simulated catch rate matches the player's real "
        "bucket catch rate. This is a *visual illustration* of the difficulty mix, "
        "not the actual plays. Use the bucket table above for precise numbers.",
        icon=None,
    )

    rng = np.random.default_rng(seed=42)
    CP_RANGES = {b.id: (b.exp_pct - 12.5, b.exp_pct + 12.5) for b in BUCKETS}
    # Actual ranges from Statcast definitions
    CP_RANGES = {1: (0, 25), 2: (26, 50), 3: (51, 75), 4: (76, 90), 5: (91, 100)}

    play_rows = []
    for b in BUCKETS:
        att     = int(row.get(f"b{b.id}_att",    0) or 0)
        caught  = int(row.get(f"b{b.id}_caught", 0) or 0)
        missed  = att - caught
        if att == 0:
            continue
        lo, hi = CP_RANGES[b.id]
        cp_vals = rng.uniform(lo, hi, att)
        # Sort so that highest CP plays are assigned "caught" first
        # (within a bucket, higher CP plays are slightly more likely to be caught)
        cp_vals_sorted = np.sort(cp_vals)[::-1]
        outcomes = np.array(["Caught"] * caught + ["Missed"] * missed)
        for cp, outcome in zip(cp_vals_sorted, outcomes):
            play_rows.append({
                "CP% (simulated)": round(cp, 1),
                "Outcome":         outcome,
                "Bracket":         b.label,
                "Bucket id":       b.id,
            })

    if not play_rows:
        st.info("No play data to display.")
        return

    plays_df = pd.DataFrame(play_rows)

    color_map = {"Caught": "#1D9E75", "Missed": "#D85A30"}

    fig = px.strip(
        plays_df,
        x="CP% (simulated)",
        y="Bracket",
        color="Outcome",
        color_discrete_map=color_map,
        hover_data={"CP% (simulated)": ":.1f", "Outcome": True, "Bracket": True},
        title=f"{player} {season} — simulated play distribution",
        stripmode="overlay",
    )
    # Add vertical lines for bucket boundaries
    for boundary in [25.5, 50.5, 75.5, 90.5]:
        fig.add_vline(x=boundary, line_color="rgba(0,0,0,0.15)",
                      line_dash="dash", line_width=1)

    fig.update_traces(marker=dict(size=6, opacity=0.65))
    fig.update_layout(
        height=max(280, len(BUCKETS) * 55 + 80),
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="system-ui",
        xaxis=dict(title="Catch probability % (simulated)", range=[-2, 102],
                   showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="", showgrid=False, categoryorder="array",
                   categoryarray=[b.label for b in BUCKETS]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bucket summary mini-table below the chart
    summary_rows = []
    for b in BUCKETS:
        att    = int(row.get(f"b{b.id}_att", 0) or 0)
        caught = int(row.get(f"b{b.id}_caught", 0) or 0)
        if att == 0:
            continue
        catch_p = caught / att * 100
        delta   = catch_p - b.exp_pct
        summary_rows.append({
            "Bracket":   b.label,
            "CP range":  b.range_label,
            "Plays":     att,
            "Caught":    caught,
            "Missed":    att - caught,
            "Actual%":   f"{catch_p:.1f}%",
            "Expected%": f"{b.exp_pct:.1f}%",
            "Δ":         f"{'+' if delta>=0 else ''}{delta:.1f}%",
        })

    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True, use_container_width=True, height=210,
        )


def _render_player_trend(df_all: pd.DataFrame, player: str) -> None:
    """OAA and bucket catch% trend for a single player across seasons."""
    st.markdown(
        "<div class='section-head'>Season trend — OAA and sprint speed</div>",
        unsafe_allow_html=True,
    )

    pdata = df_all[df_all["player_name"] == player].sort_values("season")
    if len(pdata) < 2:
        st.caption(f"{player} only appears in one season — trend not available.")
        return

    col_a, col_b = st.columns(2)

    # OAA trend
    with col_a:
        fig_oaa = px.line(
            pdata, x="season", y="oaa", markers=True,
            labels={"oaa": "Total OAA", "season": "Season"},
            title="OAA by season",
            color_discrete_sequence=["#1D9E75"],
        )
        fig_oaa.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
        fig_oaa.update_layout(
            height=300, plot_bgcolor="white", paper_bgcolor="white",
            font_family="system-ui", margin=dict(t=40, b=20),
            xaxis=dict(tickmode="array", tickvals=pdata["season"].tolist(), dtick=1),
        )
        fig_oaa.update_xaxes(showgrid=True, gridcolor="#eee")
        fig_oaa.update_yaxes(showgrid=True, gridcolor="#eee")
        st.plotly_chart(fig_oaa, use_container_width=True)

    # Sprint speed trend
    with col_b:
        spd_data = pdata.dropna(subset=["sprint_speed"])
        if spd_data.empty:
            st.info("No sprint speed data available for trend.")
        else:
            fig_spd = px.line(
                spd_data, x="season", y="sprint_speed", markers=True,
                labels={"sprint_speed": "Sprint Speed (ft/s)", "season": "Season"},
                title="Sprint speed by season",
                color_discrete_sequence=["#4A90D9"],
            )
            fig_spd.update_layout(
                height=300, plot_bgcolor="white", paper_bgcolor="white",
                font_family="system-ui", margin=dict(t=40, b=20),
                xaxis=dict(tickmode="array", tickvals=pdata["season"].tolist(), dtick=1),
            )
            fig_spd.update_xaxes(showgrid=True, gridcolor="#eee")
            fig_spd.update_yaxes(showgrid=True, gridcolor="#eee")
            st.plotly_chart(fig_spd, use_container_width=True)

    # Per-bucket catch% trend
    st.markdown(
        "<div class='section-head'>Catch% by bracket — season over season</div>",
        unsafe_allow_html=True,
    )
    bucket_trend_rows = []
    for _, season_row in pdata.iterrows():
        for b in BUCKETS:
            pct = season_row.get(f"b{b.id}_catch_pct")
            att = season_row.get(f"b{b.id}_att", 0)
            if pd.notna(pct) and att and att > 0:
                bucket_trend_rows.append({
                    "Season":  int(season_row["season"]),
                    "Bracket": b.label,
                    "Catch%":  float(pct),
                    "Opps":    int(att),
                })

    if bucket_trend_rows:
        btdf = pd.DataFrame(bucket_trend_rows)
        fig_bt = px.line(
            btdf, x="Season", y="Catch%", color="Bracket",
            markers=True,
            labels={"Catch%": "Catch %", "Season": "Season"},
            color_discrete_map={b.label: b.color for b in BUCKETS},
            title="Catch% per bracket over seasons",
            hover_data={"Opps": True},
        )
        # Add expected lines as dashed
        for b in BUCKETS:
            fig_bt.add_hline(
                y=b.exp_pct,
                line_dash="dot",
                line_color=b.color,
                opacity=0.35,
                line_width=1,
            )
        fig_bt.update_layout(
            height=360, plot_bgcolor="white", paper_bgcolor="white",
            font_family="system-ui",
            xaxis=dict(tickmode="array", tickvals=pdata["season"].tolist(), dtick=1),
            legend=dict(font=dict(size=10)),
            margin=dict(t=40),
        )
        fig_bt.update_xaxes(showgrid=True, gridcolor="#eee")
        fig_bt.update_yaxes(showgrid=True, gridcolor="#eee", range=[0, 110])
        st.plotly_chart(fig_bt, use_container_width=True)
        st.caption(
            "Dashed horizontal lines = expected catch% (bracket midpoint). "
            "Being consistently above your bracket's dashed line = genuine value above model."
        )


def _render_peer_comparison(
    df_all: pd.DataFrame,
    row: pd.Series,
    player: str,
    season: int,
) -> None:
    """Show where this player ranks among same-season OF peers."""
    st.markdown(
        "<div class='section-head'>Peer comparison — same season</div>",
        unsafe_allow_html=True,
    )

    peers = df_all[
        (df_all["season"] == season) &
        (df_all["total_opps"] >= max(10, int(row.get("total_opps", 10)) // 2))
    ].copy()

    if peers.empty or len(peers) < 3:
        st.info("Not enough peers to compare at current opportunity threshold.")
        return

    # Rank player within peers
    peers = peers.sort_values("oaa", ascending=False).reset_index(drop=True)
    peers["Rank"] = peers.index + 1
    player_rank = peers.loc[peers["player_name"] == player, "Rank"]
    player_rank = int(player_rank.iloc[0]) if not player_rank.empty else None

    if player_rank:
        st.markdown(
            f"**{player}** ranks **#{player_rank}** out of "
            f"**{len(peers)}** outfielders by OAA in {season} "
            f"(min {max(10, int(row.get('total_opps', 10)) // 2)} opportunities).",
        )

    # Horizontal OAA bar chart — highlight the player
    top_peers = pd.concat([
        peers.head(15),
        peers[peers["player_name"] == player],
    ]).drop_duplicates("player_name").sort_values("oaa", ascending=True).tail(20)

    top_peers["is_player"] = top_peers["player_name"] == player
    top_peers["color"]     = top_peers["is_player"].map({True: "#1D3557", False: "#B4B2A9"})
    top_peers["label"]     = top_peers["oaa"].apply(lambda v: f"{'+' if v>=0 else ''}{v}")

    fig = go.Figure(go.Bar(
        x=top_peers["oaa"],
        y=top_peers["player_name"],
        orientation="h",
        marker_color=top_peers["color"].tolist(),
        text=top_peers["label"],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate=(
            "<b>%{y}</b><br>OAA: %{x}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.2)", line_width=1)
    fig.update_layout(
        height=max(360, len(top_peers) * 26 + 80),
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="system-ui",
        xaxis=dict(title="OAA", showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=False),
        margin=dict(t=20, b=20, l=160, r=60),
        title=dict(text=f"OAA ranking among outfielders — {season}", font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sprint speed vs OAA quadrant for peers
    peer_scatter = peers.dropna(subset=["sprint_speed"])
    if not peer_scatter.empty:
        peer_scatter = peer_scatter.copy()
        peer_scatter["is_player"] = peer_scatter["player_name"] == player
        peer_scatter["size"]      = peer_scatter["is_player"].map({True: 18, False: 7})
        peer_scatter["symbol"]    = peer_scatter["is_player"].map({True: "star", False: "circle"})

        fig2 = go.Figure()
        # Peers
        non_player = peer_scatter[~peer_scatter["is_player"]]
        fig2.add_trace(go.Scatter(
            x=non_player["sprint_speed"], y=non_player["oaa"],
            mode="markers",
            marker=dict(size=7, color="#B4B2A9", opacity=0.6),
            text=non_player["player_name"],
            hovertemplate="<b>%{text}</b><br>Speed: %{x:.1f}<br>OAA: %{y}<extra></extra>",
            name="Peers",
        ))
        # Player
        pl_row = peer_scatter[peer_scatter["is_player"]]
        fig2.add_trace(go.Scatter(
            x=pl_row["sprint_speed"], y=pl_row["oaa"],
            mode="markers+text",
            marker=dict(size=16, color="#1D3557", symbol="star"),
            text=[player],
            textposition="top center",
            textfont=dict(size=11),
            hovertemplate=f"<b>{player}</b><br>Speed: %{{x:.1f}}<br>OAA: %{{y}}<extra></extra>",
            name=player,
        ))
        med_spd = peer_scatter["sprint_speed"].median()
        fig2.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.15)", line_width=1)
        fig2.add_vline(x=med_spd, line_dash="dot", line_color="rgba(0,0,0,0.15)", line_width=1,
                       annotation_text=f"median {med_spd:.1f}", annotation_position="top right",
                       annotation_font_size=9)
        fig2.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            font_family="system-ui",
            xaxis=dict(title="Sprint Speed (ft/s)", showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="OAA", showgrid=True, gridcolor="#eee"),
            margin=dict(t=30, b=20),
            title=dict(text=f"Speed vs OAA — {season} peer group", font=dict(size=13)),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kpi(col, label: str, value: str, sub: str = "", color: str | None = None) -> None:
    color_style = f"color:{color};" if color else ""
    col.markdown(f"""
    <div class="kpi">
        <div class="k-label">{label}</div>
        <div class="k-value" style="{color_style}">{value}</div>
        <div class="k-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def _default_player_index(players: list[str], df: pd.DataFrame) -> int:
    """Default to the player with the highest OAA in the most recent season."""
    try:
        latest = df["season"].max()
        best   = df[df["season"] == latest].sort_values("oaa", ascending=False).iloc[0]
        return players.index(best["player_name"])
    except Exception:
        return 0
