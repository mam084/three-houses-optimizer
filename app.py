"""
app.py

Streamlit dashboard for the Fire Emblem: Three Houses class-path optimizer.
Pick a character, either let the tool auto-detect their natural role from
growth rates or target a specific role yourself, and see the recommended
class path plus projected stats at a chosen level.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import (
    DLC_CLASS_TIER,
    ROLE_PROFILES,
    STAT_COLS,
    TIER_ORDER,
    format_requirement,
    list_eligible_classes_at_tier,
    load_eligibility_lookup,
    load_weapon_requirements_lookup,
    recommend_for_character,
    score_class_for_role,
    stats_for_class_at_level,
)
from src.team_builder import (
    DLC_HOUSE,
    REAL_ROUTES,
    build_team_with_paths,
    cross_house_names_in_pool,
    get_candidate_pool,
    load_recruitment_lookup,
    mandatory_names_for_route,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
PORTRAIT_DIR = Path(__file__).resolve().parent / "assets" / "portraits"
DEFAULT_TARGET_LEVEL = 30

st.set_page_config(page_title="Three Houses Class Optimizer", page_icon="⚔️", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, ...]:
    required = [
        "character_base_stats.csv", "character_growth_rates.csv", "class_stat_boosts.csv",
        "class_eligibility.csv", "character_gender.csv", "class_weapon_requirements.csv",
        "character_weapon_talent.csv", "recruitment_requirements.csv", "character_starting_level.csv",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        st.error(
            f"Missing data file(s): {', '.join(missing)}. "
            f"Base stats/growth rates/stat boosts come from `python src/scrape_serenes.py`; "
            f"the rest are hand-maintained and checked into the repo directly (Serenes doesn't "
            f"have this data in scrapable tabular form)."
        )
        st.stop()

    return (
        pd.read_csv(DATA_DIR / "character_base_stats.csv"),
        pd.read_csv(DATA_DIR / "character_growth_rates.csv"),
        pd.read_csv(DATA_DIR / "class_stat_boosts.csv"),
        pd.read_csv(DATA_DIR / "class_eligibility.csv"),
        pd.read_csv(DATA_DIR / "character_gender.csv"),
        pd.read_csv(DATA_DIR / "class_weapon_requirements.csv"),
        pd.read_csv(DATA_DIR / "character_weapon_talent.csv"),
        pd.read_csv(DATA_DIR / "recruitment_requirements.csv"),
        pd.read_csv(DATA_DIR / "character_starting_level.csv"),
    )


def get_playable_names(base_stats_df: pd.DataFrame) -> list[str]:
    """NPCs (Sothis, Rhea, etc.) aren't recruitable/playable units - exclude them everywhere."""
    return sorted(n for n in base_stats_df["name"] if "(NPC)" not in n)


def get_dlc_names(base_stats_df: pd.DataFrame) -> set[str]:
    """Characters that require the Cindered Shadows DLC - flagged in the UI, never silently mixed in."""
    return set(base_stats_df[base_stats_df["house"] == DLC_HOUSE]["name"])


def display_name(name: str, dlc_names: set[str]) -> str:
    """Character label for dropdowns/rosters - tags DLC characters instead of listing them indistinguishably."""
    return f"{name} (DLC)" if name in dlc_names else name


def get_portrait_path(character_name: str) -> Path | None:
    """
    Look up a local portrait image for a character, if one has been placed
    in assets/portraits/ (see that folder's README - actual character art
    isn't shipped in this repo, since Three Houses portraits are Nintendo/
    Intelligent Systems IP and not this project's to redistribute; add your
    own from a source you have the rights to use).
    """
    if not PORTRAIT_DIR.exists():
        return None
    slug = character_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = PORTRAIT_DIR / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def render_portrait(character_name: str, width: int = 96):
    portrait = get_portrait_path(character_name)
    if portrait:
        st.image(str(portrait), width=width)
    else:
        st.markdown(
            f"<div style='width:{width}px;height:{width}px;border-radius:8px;background:#2b2b3a;"
            f"display:flex;align-items:center;justify-content:center;font-size:{width//3}px;'>"
            f"{character_name[0]}</div>",
            unsafe_allow_html=True,
        )


def render_path_with_mixmatch(
    path: list[dict],
    character: str,
    stat_boosts_df: pd.DataFrame,
    eligibility_lookup: dict | None,
    character_gender: str | None,
    role_weights: dict,
    weapon_req_lookup: dict,
    include_dlc_classes: bool = False,
) -> list[str]:
    """
    Render the recommended class path, letting the user swap in a
    different (still eligible) class at any tier instead of only ever
    seeing the tool's own top pick - "mix and match." Returns the list of
    actually-selected class names, tier order, for the caller to recompute
    stats from (see stats_for_class_at_level) - only the LAST entry
    matters for that, since only the current/final class's boost applies
    in-game, but every tier is independently overridable so the full path
    can be browsed hypothetically.

    A user's own pick (not the recommended class) still shows its fit
    score against role_weights and its certification weapon-rank
    requirement (weapon_req_lookup - see load_weapon_requirements_lookup),
    the same information the recommended pick gets, rather than just "your
    pick, overrides the recommendation" with no numbers to compare against.
    """
    st.subheader("Recommended Class Path")
    st.caption(
        "Swap in a different eligible class at any tier to compare - only the deepest tier's "
        "class feeds the projected stats below, since in-game a class boost doesn't stack across "
        "a career path, only your current class's boost counts."
    )
    cols = st.columns(len(path))
    selected = []
    for col, step in zip(cols, path):
        with col:
            options = list_eligible_classes_at_tier(
                step["tier"], stat_boosts_df, character_name=character,
                eligibility_lookup=eligibility_lookup, character_gender=character_gender,
                include_dlc_classes=include_dlc_classes,
            )
            if step["class"] not in options:
                options = sorted(set(options) | {step["class"]})
            choice = st.selectbox(
                step["tier"], options=options, index=options.index(step["class"]),
                key=f"mixmatch_{character}_{step['tier']}",
            )
            selected.append(choice)
            if choice == step["class"]:
                if step.get("is_unique_class"):
                    st.caption("⭐ Their own unique class")
                st.caption(f"fit score {step['score']}")
                if step.get("requirement"):
                    st.caption(f"Requires: {step['requirement']}")
                st.caption(step["why"])
            else:
                choice_row = stat_boosts_df[stat_boosts_df["name"] == choice]
                choice_score = (
                    round(float(score_class_for_role(choice_row.iloc[0], role_weights)), 2)
                    if not choice_row.empty else None
                )
                st.caption("Your pick - overrides the recommendation.")
                if choice_score is not None:
                    st.caption(f"fit score {choice_score}")
                requirement = format_requirement(choice, weapon_req_lookup)
                if requirement:
                    st.caption(f"Requires: {requirement}")
    return selected


def render_stat_charts(
    base_row: dict, final_stats: dict, character: str, final_class: str, base_level_label: str = "Level 1",
):
    st.subheader("Projected Stats")

    # Bar first: it's the more-used view (exact per-stat deltas are easier to
    # read off bars than a radar's overlapping fill shapes), so it's the tab
    # that's already active on load - st.tabs() activates whichever is
    # listed first, so the order below IS the default.
    tab_bar, tab_radar = st.tabs(["Bar", "Radar"])

    # Padding added above the tallest bar, and extra top margin, so a
    # hover label over one of the tallest bars has somewhere to draw
    # itself - Plotly clips a hover label at the plot's own boundary
    # rather than letting it overflow outside the chart area, which was
    # cutting off the top of the tooltip for exactly the stats worth
    # hovering over (the highest ones). A legend placed above the plot
    # (rather than Plotly's default top-right-inside-the-plot corner)
    # keeps it from sitting on top of the tallest bars and their tooltips
    # in the first place.
    max_value = max(list(base_row.values()) + list(final_stats.values()) + [1])
    y_range = [0, max_value * 1.25]

    with tab_bar:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=STAT_COLS,
            y=[base_row[s] for s in STAT_COLS],
            name=f"{character} ({base_level_label})",
            opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            x=STAT_COLS,
            y=[final_stats[s] for s in STAT_COLS],
            name=f"Projected as {final_class}",
        ))
        fig.update_layout(
            barmode="group",
            height=480,
            margin=dict(l=0, r=0, t=60, b=0),
            yaxis_title="Stat value",
            yaxis=dict(range=y_range),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hoverlabel=dict(namelength=-1),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{character}_{final_class}")

    with tab_radar:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[base_row[s] for s in STAT_COLS],
            theta=STAT_COLS,
            fill="toself",
            name=f"{character} ({base_level_label})",
            opacity=0.6,
        ))
        fig.add_trace(go.Scatterpolar(
            r=[final_stats[s] for s in STAT_COLS],
            theta=STAT_COLS,
            fill="toself",
            name=f"Projected as {final_class}",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=y_range)),
            height=480,
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"radar_{character}_{final_class}")

    st.caption(
        "Projection = base stats + expected level-up gains (growth rate used as an "
        "expected value, not a simulated playthrough) + the selected class's stat boost. "
        "Only the current class's boost applies - boosts don't stack across a career path."
    )


def render_team(team: list[dict], dlc_names: set[str]):
    st.subheader(f"Recommended Team ({len(team)})")

    role_counts = pd.Series([m["role"] for m in team]).value_counts()
    st.caption("Role coverage: " + ", ".join(f"{role} x{count}" for role, count in role_counts.items()))

    for member in team:
        path_str = " → ".join(
            f"{step['class']}⭐" if step.get("is_unique_class") else step["class"]
            for step in member["path"]
        )
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                render_portrait(member["character"])
            with col2:
                st.markdown(f"**{display_name(member['character'], dlc_names)}**")
                st.caption(f"{member['role']} (score {member['score']})")
            with col3:
                st.write(path_str)
                requirements = [step["requirement"] for step in member["path"] if step.get("requirement")]
                if requirements:
                    st.caption("Requires (final tier): " + requirements[-1])
                st.caption(f"Why on the team: {member['why']}")


CLASS_CHART_STATS = STAT_COLS + ["Mov"]


def render_class_explorer_tab(stat_boosts_df: pd.DataFrame, weapon_req_df: pd.DataFrame):
    st.caption(
        "Browse a class's own stat boosts - the flat bonus it adds on top of whatever character "
        "wears it. Unlike most Fire Emblem games, Three Houses growth rates belong to the "
        "*character*, not the class, and don't change based on what you're wearing - a class only "
        "contributes its fixed boost (shown here) plus, at Master tier and above, a faster-climbing "
        "stat cap. \"Class growth rates\" isn't really a thing in this game; this is the number that "
        "actually plays that role."
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

    col1, col2 = st.columns(2)
    with col1:
        class_a = st.selectbox("Class", options=names, key="explorer_class_a")
    with col2:
        class_b = st.selectbox("Compare with (optional)", options=["(none)"] + names, key="explorer_class_b")

    row_a = stat_boosts_df[stat_boosts_df["name"] == class_a].iloc[0]
    st.caption(
        f"**{class_a}** - {row_a['tier']} tier"
        + (f" - requires {format_requirement(class_a, weapon_req_lookup)}"
           if format_requirement(class_a, weapon_req_lookup) else "")
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(x=CLASS_CHART_STATS, y=[row_a[s] for s in CLASS_CHART_STATS], name=class_a))
    if class_b != "(none)":
        row_b = stat_boosts_df[stat_boosts_df["name"] == class_b].iloc[0]
        fig.add_trace(go.Bar(x=CLASS_CHART_STATS, y=[row_b[s] for s in CLASS_CHART_STATS], name=class_b))
        req_b = format_requirement(class_b, weapon_req_lookup)
        st.caption(f"**{class_b}** - {row_b['tier']} tier" + (f" - requires {req_b}" if req_b else ""))

    all_values = [row_a[s] for s in CLASS_CHART_STATS]
    if class_b != "(none)":
        all_values += [row_b[s] for s in CLASS_CHART_STATS]
    y_min = min(min(all_values), 0) * 1.2 if min(all_values) < 0 else 0
    y_max = max(max(all_values), 1) * 1.25
    fig.update_layout(
        barmode="group",
        height=420,
        margin=dict(l=0, r=0, t=60, b=0),
        yaxis_title="Stat boost",
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"class_explorer_{class_a}_{class_b}")


def render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                          weapon_req_df, character_weapon_talent_df, starting_level_df, playable_names, dlc_names):
    st.caption(
        "Pick a character and see a recommended class path - either toward their "
        "natural strengths, or a role you choose. Only classes that character can "
        "actually access (character/gender-locked classes included) are considered."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        character = st.selectbox(
            "Character", options=playable_names, key="char_select",
            format_func=lambda name: display_name(name, dlc_names),
        )
    with col2:
        role_choice = st.selectbox(
            "Target role",
            options=["Auto-detect from growth rates"] + list(ROLE_PROFILES.keys()),
            key="char_role_select",
        )
    with col3:
        target_level = st.slider("Target level", min_value=5, max_value=40, value=DEFAULT_TARGET_LEVEL, key="char_level")

    include_dlc_classes = st.checkbox(
        "Include DLC classes (Cindered Shadows certification classes)", value=False,
        key="char_include_dlc_classes",
        help="Also considers Trickster, War Monk/Cleric, Dark Flier and Valkyrie as Advanced-tier options - "
             "these need the Cindered Shadows DLC, same as the DLC-exclusive characters do.",
    )

    role_name = None if role_choice == "Auto-detect from growth rates" else role_choice

    result = recommend_for_character(
        character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=role_name, target_level=target_level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
        starting_level_df=starting_level_df, include_dlc_classes=include_dlc_classes,
    )

    col_portrait, col_info = st.columns([1, 5])
    with col_portrait:
        render_portrait(character, width=80)
    with col_info:
        if role_name is None:
            st.info(
                f"Auto-detected natural role: **{result['auto_detected_role']}** "
                f"(similarity {result['auto_detection_score']})"
            )
        else:
            st.caption(
                f"Their own natural role auto-detects as **{result['auto_detected_role']}** "
                f"(similarity {result['auto_detection_score']}) - you're targeting **{role_name}** instead."
            )
        if result["join_level"] > 1:
            st.caption(
                f"🕐 Joins the roster at level {result['join_level']} - base stats and the "
                f"projection below start from there, not level 1."
            )
        if result["expected_stats_at_level"] != result["requested_target_level"]:
            st.caption(
                f"Target level {result['requested_target_level']} is below their join level - "
                f"projecting to level {result['expected_stats_at_level']} instead."
            )

    if not result["path"]:
        st.warning("No class path could be built - check that class_stat_boosts.csv has data for all tiers.")
        return

    eligibility_lookup = load_eligibility_lookup(eligibility_df)
    character_gender = None
    gender_row = character_gender_df[character_gender_df["name"] == character]
    if not gender_row.empty:
        character_gender = gender_row.iloc[0]["gender"]

    weapon_req_lookup = load_weapon_requirements_lookup(weapon_req_df)
    role_weights = ROLE_PROFILES[result["role_used"]]

    selected_path = render_path_with_mixmatch(
        result["path"], character, stat_boosts_df, eligibility_lookup, character_gender,
        role_weights, weapon_req_lookup, include_dlc_classes=include_dlc_classes,
    )
    final_class = selected_path[-1]

    st.divider()
    effective_level = result["expected_stats_at_level"]
    if final_class == result["path"][-1]["class"]:
        final_stats = result["expected_final_stats"]
    else:
        final_stats = stats_for_class_at_level(
            character, final_class, base_stats_df, growth_rates_df, stat_boosts_df,
            effective_level, start_level=result["join_level"],
        )
    base_level_label = f"Join Level {result['join_level']}" if result["join_level"] > 1 else "Level 1"
    render_stat_charts(
        result["base_stats_at_join_level"], final_stats, character, final_class,
        base_level_label=base_level_label,
    )

    st.button(
        "📥 Import this build into Team Builder", key=f"import_{character}",
        help="Locks in this exact class path/stats for the Team Builder tab, instead of it "
             "recomputing a recommendation for this character from scratch.",
        on_click=_import_build,
        args=(character, result, selected_path, final_class, final_stats),
    )
    if character in st.session_state.get("imported_builds", {}):
        st.caption(f"✅ Imported - {character} will use this exact build in the Team Builder tab.")


def _import_build(character, result, selected_path, final_class, final_stats):
    """
    Callback for the "Import this build into Team Builder" button - stashes
    this character's currently-selected path/stats (including any mix-and-
    match overrides) into st.session_state so the Team Builder tab can pick
    it up as a locked_builds entry (see team_builder.build_team_with_paths)
    instead of recomputing its own recommendation for this character. Runs
    as an on_click callback (not inline in the render function) so the
    import happens before the rerun that follows the click, the standard
    Streamlit pattern for "this button press should affect what the rest of
    the app renders on the very next run."
    """
    path = result["path"]
    if selected_path != [step["class"] for step in path]:
        # A mix-and-match override was in play - only the final/deepest tier
        # actually feeds stats in-game (see render_path_with_mixmatch), so
        # rewrite just that last step to reflect the user's actual choice
        # rather than importing the tool's original recommendation instead
        # of what's on screen.
        path = path[:-1] + [{**path[-1], "class": final_class, "is_unique_class": False}]
    imported = st.session_state.setdefault("imported_builds", {})
    imported[character] = {
        "path": path,
        "final_class": final_class,
        "expected_final_stats": final_stats,
        "eligible_unique_classes": result["eligible_unique_classes"],
    }


def render_team_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                     weapon_req_df, character_weapon_talent_df, recruitment_requirements_df, starting_level_df,
                     playable_names, dlc_names):
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

    recruitment_lookup = load_recruitment_lookup(recruitment_requirements_df)
    candidates = get_candidate_pool(
        base_stats_df, playable_names, route=None if route_choice == "Full roster" else route_choice,
        include_dlc=include_dlc, include_cross_house_recruits=include_cross_house,
        target_level=target_level, recruitment_lookup=recruitment_lookup,
    )
    cross_house_names = cross_house_names_in_pool(
        base_stats_df, candidates, None if route_choice == "Full roster" else route_choice,
    )

    # Byleth and this route's own lord are force-deployed story units on
    # their route - the game doesn't let you leave them off the team, so
    # they're always included and can't be excluded (see
    # mandatory_names_for_route). No effect on "Full roster", which has no
    # single lord.
    mandatory = [n for n in mandatory_names_for_route(route_choice) if n in candidates]
    if mandatory:
        st.caption(
            "🔒 Force-deployed on this route, always included: "
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
        "🎲 Different team, same pool",
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
            locked_builds=imported_builds,
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
            render_team(team, dlc_names)


def main():
    (
        base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
        weapon_req_df, character_weapon_talent_df, recruitment_requirements_df, starting_level_df,
    ) = load_data()
    playable_names = get_playable_names(base_stats_df)
    dlc_names = get_dlc_names(base_stats_df)

    st.title("⚔️ Three Houses Class Optimizer")

    tab1, tab2, tab3 = st.tabs(["Character Optimizer", "Team Builder", "Class Explorer"])
    with tab1:
        render_character_tab(
            base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, starting_level_df, playable_names, dlc_names,
        )
    with tab2:
        render_team_tab(
            base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, recruitment_requirements_df, starting_level_df,
            playable_names, dlc_names,
        )
    with tab3:
        render_class_explorer_tab(stat_boosts_df, weapon_req_df)


if __name__ == "__main__":
    main()
