"""
tabs/build_comparer_tab.py

Build Comparer tab: two independent character/role/level picks, each its
own full recommend_for_character call, with final projected stats
overlaid on one chart for a direct comparison.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import (
    ROLE_PROFILES,
    STAT_COLS,
    format_combined_requirements,
    load_weapon_requirements_lookup,
    recommend_for_character,
)
from app_state import (
    DEFAULT_TARGET_LEVEL,
    display_name,
    resolve_character_gender,
)




def _render_build_comparer_side(
    prefix: str, base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
    weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df, class_base_stats_df,
    character_relics_df, playable_names, dlc_names, byleth_gender, default_character: str, role_options: list[str],
):
    """One side (A or B) of the Build Comparer tab - its own character/role/level picker plus a full recommend_for_character call, independent of the other side."""
    character = st.selectbox(
        "Character", options=playable_names, key=f"{prefix}_character",
        index=playable_names.index(default_character) if default_character in playable_names else 0,
        format_func=lambda name: display_name(name, dlc_names),
    )
    role_choice = st.selectbox("Target role", options=role_options, key=f"{prefix}_role")
    target_level = st.slider("Target level", min_value=5, max_value=40, value=DEFAULT_TARGET_LEVEL, key=f"{prefix}_level")
    role_name = None if role_choice == "Auto-detect from growth rates" else role_choice
    character_gender = resolve_character_gender(character, character_gender_df, byleth_gender)

    result = recommend_for_character(
        character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=role_name, target_level=target_level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
        starting_level_df=starting_level_df,
        class_growth_df=class_growth_df, class_base_stats_df=class_base_stats_df,
        character_relics_df=character_relics_df,
        character_gender_override=character_gender if character == "Protagonist" else None,
    )
    return character, result



def render_build_comparer_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                               weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df,
                               class_base_stats_df, character_relics_df, playable_names, dlc_names,
                               byleth_gender=None):
    """
    Compare two full class-PATH recommendations side by side - "I like our
    class comparer, but a way to directly compare two class paths would be
    really cool," and roadmap item P9 ("side-by-side build comparison, a
    natural extension of the Class Explorer's existing two-class
    comparison, applied to full builds or two team compositions"). Each
    side is its own independent character/role/level pick (so this covers
    "two different characters," "the same character under two roles," and
    "the same character at two target levels" all with the same UI), with
    the two final-stat projections overlaid on one bar chart for a direct
    read of the gap, not just two separate numbers to compare by eye.

    This compares RECOMMENDED paths (no mix-and-match here - that's what
    the Character Optimizer tab is for); it's meant for "which of these two
    approaches ends up stronger," not for fine-tuning one specific build.
    """
    st.caption(
        "Compare two full class paths side by side - two different characters, the same "
        "character built toward two different roles, or the same character at two target "
        "levels. Each side is an independent recommendation (use the Character Optimizer "
        "tab for mix-and-match fine-tuning of one specific build)."
    )
    role_options = ["Auto-detect from growth rates"] + list(ROLE_PROFILES.keys())
    default_a = playable_names[0] if playable_names else ""
    default_b = playable_names[1] if len(playable_names) > 1 else default_a

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Build A**")
        char_a, result_a = _render_build_comparer_side(
            "compare_a", base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df, class_base_stats_df,
            character_relics_df, playable_names, dlc_names, byleth_gender, default_a, role_options,
        )
    with col_b:
        st.markdown("**Build B**")
        char_b, result_b = _render_build_comparer_side(
            "compare_b", base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df, class_base_stats_df,
            character_relics_df, playable_names, dlc_names, byleth_gender, default_b, role_options,
        )

    weapon_req_lookup = load_weapon_requirements_lookup(weapon_req_df)

    def render_path_summary(container, character, result):
        with container:
            if not result["path"]:
                st.warning("No class path could be built at this level.")
                return
            st.write(f"**{character} -> {result['role_used']}**")
            for step in result["path"]:
                tag = " (unique class)" if step.get("is_unique_class") else ""
                st.caption(f"{step['tier']}: {step['class']}{tag} (fit score {step['score']})")
            combined_requirement = format_combined_requirements(
                [step["class"] for step in result["path"]], weapon_req_lookup,
            )
            if combined_requirement:
                st.caption(f"Skill ranks needed across this path: {combined_requirement}")

    col_a2, col_b2 = st.columns(2)
    render_path_summary(col_a2, char_a, result_a)
    render_path_summary(col_b2, char_b, result_b)

    if result_a["path"] and result_b["path"]:
        st.divider()
        st.subheader("Final Stat Comparison")
        final_a, final_b = result_a["expected_final_stats"], result_b["expected_final_stats"]
        label_a = f"{char_a} ({result_a['path'][-1]['class']}, Lv{result_a['expected_stats_at_level']})"
        label_b = f"{char_b} ({result_b['path'][-1]['class']}, Lv{result_b['expected_stats_at_level']})"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=STAT_COLS, y=[final_a[s] for s in STAT_COLS], name=label_a))
        fig.add_trace(go.Bar(x=STAT_COLS, y=[final_b[s] for s in STAT_COLS], name=label_b))
        max_value = max(list(final_a.values()) + list(final_b.values()) + [1])
        fig.update_layout(
            barmode="group",
            height=460,
            margin=dict(l=0, r=0, t=60, b=0),
            yaxis_title="Projected stat value",
            yaxis=dict(range=[0, max_value * 1.25]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"build_comparer_{char_a}_{char_b}")
