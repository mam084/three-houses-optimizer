"""
app.py

Streamlit dashboard for the Fire Emblem: Three Houses class-path optimizer.
Pick a character, either let the tool auto-detect their natural role from
growth rates or target a specific role yourself, and see the recommended
class path plus projected stats at a chosen level.

This file is just the entry point: it loads the data once, renders the
Byleth-gender selector shared across every tab, and dispatches to each
tab's own render function - see app_state.py for the helpers/state shared
across tabs, and tabs/ for each tab's actual rendering logic (round 7
split this out of what used to be a single ~1,850-line file; see the
round-7 project doc for the full before/after and why).

Run with:
    streamlit run app.py
"""

import streamlit as st

from app_state import BYLETH_PORTRAIT_SLUGS, DEFAULT_BYLETH_GENDER, get_dlc_names, get_playable_names, load_data
from tabs.build_comparer_tab import render_build_comparer_tab
from tabs.character_tab import render_character_tab
from tabs.class_explorer_tab import render_class_explorer_tab
from tabs.team_tab import render_team_tab

st.set_page_config(page_title="Three Houses Class Optimizer", layout="wide")


def main():
    (
        base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
        weapon_req_df, character_weapon_talent_df, recruitment_requirements_df, starting_level_df,
        class_growth_df, class_base_stats_df, character_relics_df,
    ) = load_data()
    playable_names = get_playable_names(base_stats_df)
    dlc_names = get_dlc_names(base_stats_df)

    title_col, byleth_gender_col = st.columns([5, 1])
    with title_col:
        st.title("Three Houses Class Optimizer")
    with byleth_gender_col:
        # Byleth's portrait is the one piece of art in this whole app that
        # depends on a choice the game itself hands the player, rather than
        # being fixed per character - see BYLETH_PORTRAIT_SLUGS. Rendered
        # once here (not per-tab) since Byleth is force-deployed on every
        # route and selectable directly in the Character Optimizer, so the
        # choice should carry across both tabs instead of being asked twice
        # or reset on every tab switch.
        byleth_gender = st.selectbox(
            "Byleth's gender", options=list(BYLETH_PORTRAIT_SLUGS.keys()),
            index=list(BYLETH_PORTRAIT_SLUGS.keys()).index(DEFAULT_BYLETH_GENDER),
            key="byleth_portrait_gender",
            help="The gender you'd pick for Byleth at the start of the game - affects which "
                 "portrait is shown (if you've added your own byleth_m/byleth_f art, see "
                 "assets/portraits/README.md) AND which gender-locked classes Byleth is "
                 "actually eligible for (e.g. War Master is male-only, Falcon Knight is "
                 "female-only) - exactly like every other character's fixed gender does.",
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Character Optimizer", "Team Builder", "Class Explorer", "Build Comparer"]
    )
    with tab1:
        render_character_tab(
            base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df,
            class_base_stats_df, character_relics_df, playable_names, dlc_names,
            byleth_gender=byleth_gender,
        )
    with tab2:
        render_team_tab(
            base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, recruitment_requirements_df, starting_level_df,
            class_growth_df, class_base_stats_df, character_relics_df, playable_names, dlc_names,
            byleth_gender=byleth_gender,
        )
    with tab3:
        render_class_explorer_tab(stat_boosts_df, weapon_req_df, class_growth_df)
    with tab4:
        render_build_comparer_tab(
            base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df,
            class_base_stats_df, character_relics_df, playable_names, dlc_names,
            byleth_gender=byleth_gender,
        )


if __name__ == "__main__":
    main()
