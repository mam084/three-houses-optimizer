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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import (
    ROLE_PROFILES,
    STAT_COLS,
    TIER_ORDER,
    recommend_for_character,
)
from src.team_builder import build_team_with_paths

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="Three Houses Class Optimizer", page_icon="⚔️", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["character_base_stats.csv", "character_growth_rates.csv", "class_stat_boosts.csv"]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        st.error(
            f"Missing data file(s): {', '.join(missing)}. "
            f"Run `python src/scrape_serenes.py` first."
        )
        st.stop()

    base_stats_df = pd.read_csv(DATA_DIR / "character_base_stats.csv")
    growth_rates_df = pd.read_csv(DATA_DIR / "character_growth_rates.csv")
    stat_boosts_df = pd.read_csv(DATA_DIR / "class_stat_boosts.csv")
    return base_stats_df, growth_rates_df, stat_boosts_df


def get_playable_names(base_stats_df: pd.DataFrame) -> list[str]:
    """NPCs (Sothis, Rhea, etc.) aren't recruitable/playable units - exclude them everywhere."""
    return sorted(n for n in base_stats_df["name"] if "(NPC)" not in n)


def render_path(path: list[dict]):
    st.subheader("Recommended Class Path")
    cols = st.columns(len(path))
    for col, step in zip(cols, path):
        with col:
            st.metric(label=step["tier"], value=step["class"])
            st.caption(f"fit score {step['score']}")


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


def render_team(team: list[dict]):
    st.subheader(f"Recommended Team ({len(team)})")

    role_counts = pd.Series([m["role"] for m in team]).value_counts()
    st.caption("Role coverage: " + ", ".join(f"{role} x{count}" for role, count in role_counts.items()))

    for member in team:
        path_str = " → ".join(step["class"] for step in member["path"])
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{member['character']}**")
                st.caption(f"{member['role']} (score {member['score']})")
            with col2:
                st.write(path_str)


def render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, playable_names):
    st.caption(
        "Pick a character and see a recommended class path - either toward their "
        "natural strengths, or a role you choose."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        character = st.selectbox("Character", options=playable_names, key="char_select")
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
    )

    if role_name is None:
        st.info(
            f"Auto-detected natural role: **{result['auto_detected_role']}** "
            f"(similarity {result['auto_detection_score']})"
        )

    if not result["path"]:
        st.warning("No class path could be built - check that class_stat_boosts.csv has data for all tiers.")
        return

    render_path(result["path"])

    st.divider()
    base_row = base_stats_df[base_stats_df["name"] == character].iloc[0]
    final_class = result["path"][-1]["class"]
    render_stat_radar(base_row, result["expected_final_stats"], character, final_class)


def render_team_tab(base_stats_df, growth_rates_df, stat_boosts_df, playable_names):
    st.caption(
        "Builds a balanced team from a candidate pool by covering complementary "
        "roles, rather than just stacking the strongest individuals."
    )

    houses = sorted(base_stats_df[base_stats_df["name"].isin(playable_names)]["house"].unique())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        pool_choice = st.selectbox("Candidate pool", options=["Full roster"] + houses, key="team_pool")
    with col2:
        team_size = st.slider("Team size", min_value=3, max_value=12, value=6, key="team_size")
    with col3:
        target_level = st.slider("Target level", min_value=5, max_value=40, value=20, key="team_level")

    if pool_choice == "Full roster":
        candidates = playable_names
    else:
        candidates = base_stats_df[base_stats_df["house"] == pool_choice]["name"].tolist()

    if st.button("Build Team", type="primary"):
        team = build_team_with_paths(
            candidates, base_stats_df, growth_rates_df, stat_boosts_df,
            team_size=team_size, target_level=target_level,
        )
        if not team:
            st.warning("Couldn't build a team from this pool - try a larger candidate pool.")
        else:
            if len(team) < team_size:
                st.caption(f"Only {len(team)} candidates were available in this pool.")
            render_team(team)


def main():
    base_stats_df, growth_rates_df, stat_boosts_df = load_data()
    playable_names = get_playable_names(base_stats_df)

    st.title("⚔️ Three Houses Class Optimizer")

    tab1, tab2 = st.tabs(["Character Optimizer", "Team Builder"])
    with tab1:
        render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, playable_names)
    with tab2:
        render_team_tab(base_stats_df, growth_rates_df, stat_boosts_df, playable_names)


if __name__ == "__main__":
    main()