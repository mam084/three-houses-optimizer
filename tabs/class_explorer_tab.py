"""
tabs/class_explorer_tab.py

Class Explorer tab: browse any class's flat stat-boost line and its own
growth-rate modifiers, with an optional side-by-side second class for
comparison.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import (
    DLC_CLASS_TIER,
    STAT_COLS,
    TIER_ORDER,
    class_growth_axis_range,
    format_requirement,
    load_class_growth_lookup,
    load_weapon_requirements_lookup,
)




CLASS_CHART_STATS = STAT_COLS + ["Mov"]



def render_class_explorer_tab(stat_boosts_df: pd.DataFrame, weapon_req_df: pd.DataFrame, class_growth_df: pd.DataFrame):
    st.caption(
        "Browse a class's flat stat boost (the one-time bonus it adds on top of whatever character "
        "wears it, active only while that's their current class) alongside its own growth-RATE "
        "modifiers - a separate, real mechanic: every class also speeds up or slows down how fast "
        "specific stats climb on each level-up spent in it, stacking with the character's own "
        "personal growth rate. Both matter for a character's own path - see the Growth Rate "
        "Modifiers tab below - and are compounded across their whole class path (not just the final "
        "class) in the Character Optimizer tab's projection."
    )

    selectable = stat_boosts_df[~stat_boosts_df["name"].str.contains(r"\(", regex=True)]
    selectable = selectable[~selectable["tier"].isin(["NPC/Enemy"])]
    include_dlc_classes = st.checkbox(
        "Include DLC classes (Cindered Shadows)", value=False, key="explorer_include_dlc",
    )
    if not include_dlc_classes:
        selectable = selectable[selectable["tier"] != DLC_CLASS_TIER]

    tier_order_lookup = {tier: i for i, tier in enumerate(TIER_ORDER + ["Unique", DLC_CLASS_TIER])}
    tier_by_name = dict(zip(selectable["name"], selectable["tier"]))
    names = sorted(
        selectable["name"].tolist(),
        key=lambda n: (tier_order_lookup.get(tier_by_name.get(n), 99), n),
    )

    weapon_req_lookup = load_weapon_requirements_lookup(weapon_req_df)
    class_growth_lookup = load_class_growth_lookup(class_growth_df)

    col1, col2 = st.columns(2)
    with col1:
        class_a = st.selectbox("Class", options=names, key="explorer_class_a")
    with col2:
        class_b = st.selectbox("Compare with (optional)", options=["(none)"] + names, key="explorer_class_b")

    row_a = stat_boosts_df[stat_boosts_df["name"] == class_a].iloc[0]
    row_b = stat_boosts_df[stat_boosts_df["name"] == class_b].iloc[0] if class_b != "(none)" else None

    def class_caption(name, row):
        req = format_requirement(name, weapon_req_lookup)
        return f"**{name}** - {row['tier']} tier" + (f" - requires {req}" if req else "")

    st.caption(class_caption(class_a, row_a))
    if row_b is not None:
        st.caption(class_caption(class_b, row_b))

    tab_boost, tab_growth = st.tabs(["Stat Boosts (flat)", "Growth Rate Modifiers (per level-up)"])

    with tab_boost:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=CLASS_CHART_STATS, y=[row_a[s] for s in CLASS_CHART_STATS], name=class_a))
        if row_b is not None:
            fig.add_trace(go.Bar(x=CLASS_CHART_STATS, y=[row_b[s] for s in CLASS_CHART_STATS], name=class_b))

        all_values = [row_a[s] for s in CLASS_CHART_STATS] + ([row_b[s] for s in CLASS_CHART_STATS] if row_b is not None else [])
        y_min = min(min(all_values), 0) * 1.2 if min(all_values) < 0 else 0
        y_max = max(max(all_values), 1) * 1.25
        fig.update_layout(
            barmode="group",
            height=420,
            margin=dict(l=0, r=0, t=60, b=0),
            yaxis_title="Stat boost (flat, one-time)",
            yaxis=dict(range=[y_min, y_max]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"class_explorer_boost_{class_a}_{class_b}")

    with tab_growth:
        growth_a = class_growth_lookup.get(class_a, {})
        if not growth_a:
            st.caption(f"No growth-rate modifier data on file for {class_a}.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=STAT_COLS, y=[growth_a.get(s, 0) for s in STAT_COLS], name=class_a))
            if row_b is not None:
                growth_b = class_growth_lookup.get(class_b, {})
                if growth_b:
                    fig.add_trace(go.Bar(x=STAT_COLS, y=[growth_b.get(s, 0) for s in STAT_COLS], name=class_b))
            # Global range across EVERY class's growth-rate modifier (see
            # optimizer.class_growth_axis_range), not just whatever's
            # currently selected here - the same range the Character
            # Optimizer's own per-tier growth-rate mini chart uses, so a
            # bar's height means the same thing in both places rather than
            # each chart auto-scaling to its own pair of classes.
            axis_min, axis_max = class_growth_axis_range(class_growth_lookup)
            span = max(axis_max - axis_min, 1.0)
            padding = span * 0.15
            fig.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=0, r=0, t=60, b=0),
                yaxis_title="Growth-rate modifier (percentage points per level-up)",
                yaxis=dict(range=[axis_min - padding, axis_max + padding]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"class_explorer_growth_{class_a}_{class_b}")
            st.caption(
                "Added to the character's own personal growth rate on every level-up spent in this "
                "class - e.g. a class with +15% Str here means +0.15 expected Str per level, on top "
                "of whatever the character's own Str growth rate already contributes."
            )
