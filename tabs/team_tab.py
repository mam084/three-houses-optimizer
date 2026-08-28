"""
tabs/team_tab.py

Team Builder tab: route/pool selection, building a balanced team, the
Dancer single-slot assignment, and rendering each team member's card
(including reusing a locked-in Character Optimizer import via
st.session_state["imported_builds"] - see app_state and character_tab's
_import_build for the write side of that handoff).
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.optimizer import (
    WEAPON_SWITCH_WARNING_ENABLED,
    format_combined_requirements,
    load_weapon_requirements_lookup,
)
from src.team_builder import (
    REAL_ROUTES,
    build_team_with_paths,
    cross_house_names_in_pool,
    get_candidate_pool,
    load_recruitment_lookup,
    mandatory_names_for_route,
)
from app_state import (
    DEFAULT_TARGET_LEVEL,
    display_name,
    get_house_lookup,
    render_portrait,
)



# data/class_eligibility.csv's Dancer row documents "only one character can
# hold it per playthrough" and "excludes the Protagonist" in its unlock_note
# text, but (like every Unique-tier class not covered by
# UNIQUE_STORY_CLASS_TIER) that's not enforced anywhere in is_class_eligible
# - Dancer is open-eligibility data-wise, unlocked via the White Heron Cup
# event rather than a certification exam. Both real-game constraints are
# enforced here instead, at the one place Dancer is actually surfaced (see
# render_team_tab's Dancer selectbox): the single-select widget itself makes
# "only one character" structural, and this set excludes the Protagonist
# from the option list.
DANCER_INELIGIBLE_CHARACTERS = {"Protagonist"}



def render_team(
    team: list[dict], dlc_names: set[str], house_lookup: dict | None = None, dancer: str | None = None,
    byleth_gender: str | None = None, weapon_req_lookup: dict | None = None,
):
    """
    dancer: the character name (if any) assigned this team's single Dancer
    slot (see render_team_tab's Dancer selectbox / DANCER_INELIGIBLE_
    CHARACTERS) - shown as a badge on that one member's card, rather than
    Dancer being a class any/every member could be independently
    recommended into. Only one character can hold the Dancer class per
    playthrough (data/class_eligibility.csv's Dancer unlock_note), so this
    is deliberately a single roster-wide assignment, not a per-member
    option - the selectbox that produces this value in render_team_tab can
    only ever hold one name at a time.

    byleth_gender: passed straight through to render_portrait for whichever
    member is the Protagonist (Byleth is force-deployed on every route, so
    in practice this fires on almost every team) - see get_portrait_path.

    weapon_req_lookup (see optimizer.load_weapon_requirements_lookup), if
    given, is used to show each member's COMBINED skill-rank requirement
    across their whole path (optimizer.format_combined_requirements), not
    just the final tier's own requirement - "final class requirement
    display should show the skill rank needs for each class the character
    is going to use in the path, e.g. Paladin then War Master needs B
    Lances, B Riding, A Axes, A Brawling," not just War Master's own
    requirement string with no mention of what Paladin itself demanded on
    the way there.
    """
    st.subheader(f"Recommended Team ({len(team)})")

    role_counts = pd.Series([m["role"] for m in team]).value_counts()
    st.caption("Role coverage: " + ", ".join(f"{role} x{count}" for role, count in role_counts.items()))

    for member in team:
        path_str = " → ".join(
            f"{step['class']} (unique class)" if step.get("is_unique_class") else step["class"]
            for step in member["path"]
        )
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                render_portrait(
                    member["character"], house=(house_lookup or {}).get(member["character"]),
                    byleth_gender=byleth_gender,
                )
            with col2:
                st.markdown(f"**{display_name(member['character'], dlc_names)}**")
                st.caption(f"{member['role']} (score {member['score']})")
                if member["character"] == dancer:
                    st.caption("This team's Dancer (White Heron Cup) - swaps in instead of the path "
                               "at right when you need the Dance command; only one character on the "
                               "whole roster can hold it.")
            with col3:
                st.write(path_str)
                combined_requirement = format_combined_requirements(
                    [step["class"] for step in member["path"]], weapon_req_lookup,
                ) if weapon_req_lookup else None
                if combined_requirement:
                    st.caption("Skill ranks needed across this path: " + combined_requirement)
                if WEAPON_SWITCH_WARNING_ENABLED and member.get("weapon_switch_warning"):
                    st.caption(f"Warning: {member['weapon_switch_warning']}")
                st.caption(f"Why on the team: {member['why']}")



def render_team_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                     weapon_req_df, character_weapon_talent_df, recruitment_requirements_df, starting_level_df,
                     class_growth_df, class_base_stats_df, character_relics_df, playable_names, dlc_names,
                     byleth_gender=None):
    st.caption(
        "Builds a balanced team from a candidate pool by covering complementary "
        "roles, rather than just stacking the strongest individuals."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        route_choice = st.selectbox(
            "Route", options=["Full roster"] + REAL_ROUTES, key="team_route",
            help="A route includes that house's students plus the Protagonist and the Church/Knights of "
                 "Seiros staff, who are recruitable on any route. Black Eagles splits into Crimson Flower "
                 "(Edelgard stays) and Silver Snow (Edelgard and Hubert leave permanently and become "
                 "unrecruitable, around Chapter 11).",
        )
    with col2:
        team_size = st.slider("Team size", min_value=3, max_value=12, value=6, key="team_size")
    with col3:
        target_level = st.slider("Target level", min_value=5, max_value=40, value=DEFAULT_TARGET_LEVEL, key="team_level")

    col_dlc, col_dlc_classes, col_cross = st.columns(3)
    with col_dlc:
        include_dlc = st.checkbox(
            "Include DLC characters (Cindered Shadows)", value=False, key="team_include_dlc",
        )
    with col_dlc_classes:
        include_dlc_classes = st.checkbox(
            "Include DLC classes", value=False, key="team_include_dlc_classes",
            help="Trickster, War Monk/Cleric, Dark Flier and Valkyrie as Advanced-tier options for every "
                 "team member - a separate thing from DLC *characters* above, since any character can use "
                 "these classes if you own the Cindered Shadows DLC.",
        )
    with col_cross:
        include_cross_house = st.checkbox(
            "Model recruitment requirements (recruit from other houses)", value=False,
            key="team_include_cross_house",
            help="Also considers students from the route's other two houses, gated by their real in-game "
                 "recruitment requirement (Byleth's level - the target level above stands in for it - plus "
                 "a stat/skill-rank threshold, shown as a note but not enforced, since this tool doesn't "
                 "simulate Byleth's own stat growth). House leaders and their sworn retainers are never "
                 "recruitable this way. Has no effect on 'Full roster'.",
        )

    force_deployments = st.checkbox(
        "Force-deploy Byleth and the route's own lord", value=True, key="team_force_deployments",
        help="On a real route, the game doesn't let you leave Byleth or that route's house leader off the "
             "roster - on by default so the recommended team matches what's actually buildable. Turn off "
             "to build purely from fit scoring instead, ignoring who's actually forced onto the roster.",
    )

    recruitment_lookup = load_recruitment_lookup(recruitment_requirements_df)
    candidates = get_candidate_pool(
        base_stats_df, playable_names, route=None if route_choice == "Full roster" else route_choice,
        include_dlc=include_dlc, include_cross_house_recruits=include_cross_house,
        target_level=target_level, recruitment_lookup=recruitment_lookup,
    )
    cross_house_names = cross_house_names_in_pool(
        base_stats_df, candidates, None if route_choice == "Full roster" else route_choice,
    )

    # Byleth (always) and this route's own lord (on a real route) are
    # force-deployed story units - the game doesn't let you leave them off
    # the roster, so they're included and can't be excluded whenever
    # force_deployments is on (see mandatory_names_for_route). Turning the
    # toggle off drops this entirely, building from fit scoring alone.
    mandatory = [n for n in mandatory_names_for_route(route_choice) if n in candidates] if force_deployments else []
    if mandatory:
        st.caption(
            "Force-deployed, always included: "
            + ", ".join(display_name(n, dlc_names) for n in mandatory)
        )

    imported_builds = st.session_state.get("imported_builds", {})
    imported_in_pool = [n for n in imported_builds if n in candidates]
    if imported_builds:
        with st.container(border=True):
            st.caption("📥 Imported builds from the Character Optimizer tab (always included):")
            for name in list(imported_builds.keys()):
                col_name, col_remove = st.columns([5, 1])
                in_pool_note = "" if name in candidates else " (not in this pool - won't be added)"
                col_name.write(f"**{display_name(name, dlc_names)}**: {imported_builds[name]['final_class']}{in_pool_note}")
                if col_remove.button("Remove", key=f"remove_import_{name}"):
                    del st.session_state["imported_builds"][name]
                    st.rerun()

    with st.expander("Looking for something specific? (optional)"):
        col_a, col_b = st.columns(2)
        excludable_options = sorted(set(candidates) - set(mandatory) - set(imported_in_pool), key=str.lower)
        with col_a:
            must_include = st.multiselect(
                "Must include", options=excludable_options,
                format_func=lambda n: display_name(n, dlc_names), key="team_must_include",
                help="Build the rest of the team around these characters, in addition to whoever's "
                     "force-deployed or imported above.",
            )
        with col_b:
            exclude = st.multiselect(
                "Exclude", options=excludable_options,
                format_func=lambda n: display_name(n, dlc_names), key="team_exclude",
                help="Leave these characters out of consideration entirely. Force-deployed and imported "
                     "characters can't be excluded.",
            )

    effective_must_include = list(dict.fromkeys(mandatory + imported_in_pool + must_include))

    col_build, col_shuffle = st.columns([1, 1])
    build_clicked = col_build.button("Build Team", type="primary")
    shuffle_clicked = col_shuffle.button(
        "Different team, same pool",
        help="Same candidate pool and settings, but weighted-random picks instead of always the single "
             "top scorer per role - use this if you don't want the exact same team every time.",
    )

    if build_clicked:
        st.session_state["team_seed"] = None  # deterministic top-picks team
    if shuffle_clicked:
        st.session_state["team_seed"] = np.random.default_rng().integers(0, 2**31 - 1)

    if build_clicked or shuffle_clicked:
        seed = st.session_state.get("team_seed")
        rng = np.random.default_rng(seed) if seed is not None else None
        team = build_team_with_paths(
            candidates, base_stats_df, growth_rates_df, stat_boosts_df,
            team_size=team_size, target_level=target_level,
            eligibility_df=eligibility_df, character_gender_df=character_gender_df,
            must_include=effective_must_include, exclude=exclude, rng=rng,
            recruitment_lookup=recruitment_lookup, cross_house_names=cross_house_names,
            weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
            starting_level_df=starting_level_df, include_dlc_classes=include_dlc_classes,
            locked_builds=imported_builds, class_growth_df=class_growth_df,
            class_base_stats_df=class_base_stats_df,
            character_relics_df=character_relics_df,
            force_deployed=set(mandatory),
            byleth_gender=byleth_gender,
        )
        # Persisted in session_state rather than only a local variable: Streamlit reruns this whole
        # function on ANY widget interaction, not just the two build buttons above (e.g. touching the
        # team-size slider or the DLC checkbox) - without this, the just-built team would vanish the
        # instant the user touched anything else on the page, which was the reported "team results
        # disappearing on unrelated clicks" bug. Rendering from session_state below (outside this "was
        # a button just clicked" branch) means the last-built team survives every rerun until a build
        # button is pressed again.
        st.session_state["team_result"] = team
        st.session_state["team_result_requested_size"] = team_size

    if "team_result" in st.session_state:
        team = st.session_state["team_result"]
        if not team:
            st.warning("Couldn't build a team from this pool - try a larger candidate pool.")
        else:
            requested_size = st.session_state.get("team_result_requested_size", team_size)
            if len(team) < requested_size:
                st.caption(f"Only {len(team)} candidates were available in this pool.")

            dancer_options = ["None"] + [
                m["character"] for m in team if m["character"] not in DANCER_INELIGIBLE_CHARACTERS
            ]
            # Keying on this exact team's composition (rather than a fixed
            # key) means a newly-built team with a different roster always
            # starts this widget fresh at "None" instead of inheriting a
            # stale selection from a previous team that might not even
            # contain that character anymore - see DANCER_INELIGIBLE_
            # CHARACTERS docstring for why Dancer is single-select at all.
            dancer_widget_key = "team_dancer_select_" + ",".join(sorted(m["character"] for m in team))
            dancer_choice = st.selectbox(
                "Assign this team's Dancer (optional)",
                options=dancer_options, key=dancer_widget_key,
                format_func=lambda n: display_name(n, dlc_names) if n != "None" else "None",
                help="Only one character can hold the Dancer class per playthrough - it's unlocked via "
                     "the White Heron Cup event, not a certification exam, and equipping it replaces "
                     "whichever class that character is currently in rather than stacking with it. So "
                     "this is a single roster-wide slot, not something every team member can be "
                     "independently recommended into (the Protagonist can't hold it either).",
            )
            dancer = dancer_choice if dancer_choice != "None" else None

            render_team(
                team, dlc_names, house_lookup=get_house_lookup(base_stats_df), dancer=dancer,
                byleth_gender=byleth_gender, weapon_req_lookup=load_weapon_requirements_lookup(weapon_req_df),
            )
