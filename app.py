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


def main():
    base_stats_df, growth_rates_df, stat_boosts_df = load_data()

    st.title("⚔️ Three Houses Class Optimizer")
    st.caption(
        "Pick a character and see a recommended class path - either toward their "
        "natural strengths, or a role you choose."
    )

    playable_names = sorted(
        n for n in base_stats_df["name"]
        if "(NPC)" not in n  # NPCs (Sothis, Rhea, etc.) aren't recruitable/playable units
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        character = st.selectbox("Character", options=playable_names)
    with col2:
        role_choice = st.selectbox(
            "Target role",
            options=["Auto-detect from growth rates"] + list(ROLE_PROFILES.keys()),
        )
    with col3:
        target_level = st.slider("Target level", min_value=5, max_value=40, value=20)

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


if __name__ == "__main__":
    main()