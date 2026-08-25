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
    ROLE_PROFILES,
    STAT_COLS,
    list_eligible_classes_at_tier,
    load_eligibility_lookup,
    recommend_for_character,
    stats_for_class_at_level,
)
from src.team_builder import (
    DLC_HOUSE,
    REAL_ROUTES,
    build_team_with_paths,
    cross_house_names_in_pool,
    get_candidate_pool,
    load_recruitment_lookup,
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
        "character_weapon_talent.csv", "recruitment_requirements.csv",
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
            )
            if step["class"] not in options:
                options = sorted(set(options) | {step["class"]})
            choice = st.selectbox(
                step["tier"], options=options, index=options.index(step["class"]),
                key=f"mixmatch_{character}_{step['tier']}",
            )
            selected.append(choice)
            if choice == step["class"]:
                st.caption(f"fit score {step['score']}")
                if step.get("requirement"):
                    st.caption(f"Requires: {step['requirement']}")
                st.caption(step["why"])
            else:
                st.caption("Your pick - overrides the recommendation.")
    return selected


def render_stat_charts(base_row: pd.Series, final_stats: dict, character: str, final_class: str):
    st.subheader("Projected Stats")

    tab_radar, tab_bar = st.tabs(["Radar", "Bar"])

    with tab_radar:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[base_row[s] for s in STAT_COLS],
            theta=STAT_COLS,
            fill="toself",
            name=f"{character} (Level 1)",
            opacity=0.6,
        ))
        fig.add_trace(go.Scatterpolar(
            r=[final_stats[s] for s in STAT_COLS],
            theta=STAT_COLS,
            fill="toself",
            name=f"Projected as {final_class}",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            height=450,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"radar_{character}_{final_class}")

    with tab_bar:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=STAT_COLS,
            y=[base_row[s] for s in STAT_COLS],
            name=f"{character} (Level 1)",
            opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            x=STAT_COLS,
            y=[final_stats[s] for s in STAT_COLS],
            name=f"Projected as {final_class}",
        ))
        fig.update_layout(
            barmode="group",
            height=450,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Stat value",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{character}_{final_class}")

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
        path_str = " → ".join(step["class"] for step in member["path"])
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


def render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                          weapon_req_df, character_weapon_talent_df, playable_names, dlc_names):
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

    role_name = None if role_choice == "Auto-detect from growth rates" else role_choice

    result = recommend_for_character(
        character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=role_name, target_level=target_level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
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

    if not result["path"]:
        st.warning("No class path could be built - check that class_stat_boosts.csv has data for all tiers.")
        return

    eligibility_lookup = load_eligibility_lookup(eligibility_df)
    character_gender = None
    gender_row = character_gender_df[character_gender_df["name"] == character]
    if not gender_row.empty:
        character_gender = gender_row.iloc[0]["gender"]

    selected_path = render_path_with_mixmatch(
        result["path"], character, stat_boosts_df, eligibility_lookup, character_gender,
    )
    final_class = selected_path[-1]

    st.divider()
    base_row = base_stats_df[base_stats_df["name"] == character].iloc[0]
    if final_class == result["path"][-1]["class"]:
        final_stats = result["expected_final_stats"]
    else:
        final_stats = stats_for_class_at_level(
            character, final_class, base_stats_df, growth_rates_df, stat_boosts_df, target_level,
        )
    render_stat_charts(base_row, final_stats, character, final_class)


def render_team_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                     weapon_req_df, character_weapon_talent_df, recruitment_requirements_df,
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

    col_dlc, col_cross = st.columns(2)
    with col_dlc:
        include_dlc = st.checkbox(
            "Include DLC characters (Cindered Shadows)", value=False, key="team_include_dlc",
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

    with st.expander("Looking for something specific? (optional)"):
        col_a, col_b = st.columns(2)
        with col_a:
            must_include = st.multiselect(
                "Must include", options=sorted(candidates, key=str.lower),
                format_func=lambda n: display_name(n, dlc_names), key="team_must_include",
                help="Build the rest of the team around these characters.",
            )
        with col_b:
            exclude = st.multiselect(
                "Exclude", options=sorted(candidates, key=str.lower),
                format_func=lambda n: display_name(n, dlc_names), key="team_exclude",
                help="Leave these characters out of consideration entirely.",
            )

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
            must_include=must_include, exclude=exclude, rng=rng,
            recruitment_lookup=recruitment_lookup, cross_house_names=cross_house_names,
            weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
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
        weapon_req_df, character_weapon_talent_df, recruitment_requirements_df,
    ) = load_data()
    playable_names = get_playable_names(base_stats_df)
    dlc_names = get_dlc_names(base_stats_df)

    st.title("⚔️ Three Houses Class Optimizer")

    tab1, tab2 = st.tabs(["Character Optimizer", "Team Builder"])
    with tab1:
        render_character_tab(
            base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, playable_names, dlc_names,
        )
    with tab2:
        render_team_tab(
            base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
            weapon_req_df, character_weapon_talent_df, recruitment_requirements_df,
            playable_names, dlc_names,
        )


if __name__ == "__main__":
    main()
