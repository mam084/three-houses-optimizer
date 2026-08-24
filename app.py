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
    TIER_ORDER,
    recommend_for_character,
)
from src.team_builder import DLC_HOUSE, REAL_ROUTES, build_team_with_paths, get_candidate_pool

DATA_DIR = Path(__file__).resolve().parent / "data"
PORTRAIT_DIR = Path(__file__).resolve().parent / "assets" / "portraits"

st.set_page_config(page_title="Three Houses Class Optimizer", page_icon="⚔️", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        "character_base_stats.csv", "character_growth_rates.csv", "class_stat_boosts.csv",
        "class_eligibility.csv", "character_gender.csv",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        st.error(
            f"Missing data file(s): {', '.join(missing)}. "
            f"Base stats/growth rates/stat boosts come from `python src/scrape_serenes.py`; "
            f"class_eligibility.csv and character_gender.csv are hand-maintained and checked "
            f"into the repo directly (Serenes doesn't have this data in scrapable form)."
        )
        st.stop()

    base_stats_df = pd.read_csv(DATA_DIR / "character_base_stats.csv")
    growth_rates_df = pd.read_csv(DATA_DIR / "character_growth_rates.csv")
    stat_boosts_df = pd.read_csv(DATA_DIR / "class_stat_boosts.csv")
    eligibility_df = pd.read_csv(DATA_DIR / "class_eligibility.csv")
    character_gender_df = pd.read_csv(DATA_DIR / "character_gender.csv")
    return base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df


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


def render_path(path: list[dict]):
    st.subheader("Recommended Class Path")
    cols = st.columns(len(path))
    for col, step in zip(cols, path):
        with col:
            st.metric(label=step["tier"], value=step["class"])
            st.caption(f"fit score {step['score']}")
            st.caption(step["why"])


def render_stat_radar(base_row: pd.Series, final_stats: dict, character: str, final_class: str):
    st.subheader("Projected Stats")

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
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Projection = base stats + expected level-up gains (growth rate used as an "
        "expected value, not a simulated playthrough) + the final class's stat boost. "
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
                st.caption(f"Why on the team: {member['why']}")


def render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df, playable_names, dlc_names):
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
        target_level = st.slider("Target level", min_value=5, max_value=40, value=20, key="char_level")

    role_name = None if role_choice == "Auto-detect from growth rates" else role_choice

    result = recommend_for_character(
        character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=role_name, target_level=target_level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
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

    render_path(result["path"])

    st.divider()
    base_row = base_stats_df[base_stats_df["name"] == character].iloc[0]
    final_class = result["path"][-1]["class"]
    render_stat_radar(base_row, result["expected_final_stats"], character, final_class)


def render_team_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df, playable_names, dlc_names):
    st.caption(
        "Builds a balanced team from a candidate pool by covering complementary "
        "roles, rather than just stacking the strongest individuals."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        route_choice = st.selectbox(
            "Route", options=["Full roster"] + REAL_ROUTES, key="team_route",
            help="A route includes that house's students plus the Protagonist and the Church/Knights of "
                 "Seiros staff, who are recruitable on any route.",
        )
    with col2:
        team_size = st.slider("Team size", min_value=3, max_value=12, value=6, key="team_size")
    with col3:
        target_level = st.slider("Target level", min_value=5, max_value=40, value=20, key="team_level")

    include_dlc = st.checkbox(
        "Include DLC characters (Cindered Shadows)", value=False, key="team_include_dlc",
    )

    candidates = get_candidate_pool(
        base_stats_df, playable_names, route=None if route_choice == "Full roster" else route_choice,
        include_dlc=include_dlc,
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
        )
        if not team:
            st.warning("Couldn't build a team from this pool - try a larger candidate pool.")
        else:
            if len(team) < team_size:
                st.caption(f"Only {len(team)} candidates were available in this pool.")
            render_team(team, dlc_names)


def main():
    base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df = load_data()
    playable_names = get_playable_names(base_stats_df)
    dlc_names = get_dlc_names(base_stats_df)

    st.title("⚔️ Three Houses Class Optimizer")

    tab1, tab2 = st.tabs(["Character Optimizer", "Team Builder"])
    with tab1:
        render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df, playable_names, dlc_names)
    with tab2:
        render_team_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df, playable_names, dlc_names)


if __name__ == "__main__":
    main()