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
    load_character_proficiency_lookup,
    load_class_base_stats_lookup,
    load_class_growth_lookup,
    load_eligibility_lookup,
    load_weapon_requirements_lookup,
    recommend_for_character,
    score_class_for_role,
    score_growth_for_role,
    stats_for_selected_path,
    weapon_switch_penalty,
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

# Route-based color coding for the portrait tile/badge (see render_portrait) -
# each character's data/character_base_stats.csv "house" value maps to a
# color evoking that house/route's own game branding, so the fallback tile
# (still all anyone sees by default - see render_portrait's docstring) reads
# as "which house/route is this character from" at a glance instead of a
# single flat neutral box. A house absent from this map (there shouldn't be
# one - every character_base_stats.csv "house" value is covered) falls back
# to DEFAULT_PORTRAIT_COLOR.
HOUSE_COLORS = {
    "Black Eagles": "#7a1620",  # crimson/black, the house's own color scheme
    "Blue Lions": "#1d3f6e",  # royal blue
    "Golden Deer": "#8a5a12",  # gold/amber
    "Church of Seiros": "#7a6a2a",  # muted gold - the Church's own heraldry
    "Knights of Seiros": "#455163",  # steel gray - knights, not students
    "You and the Enigmatic Girl": "#1f6e5c",  # teal - Byleth/Sothis's own mint-teal color motif
    DLC_HOUSE: "#5a1f6e",  # purple - Cindered Shadows/Abyss's distinct visual identity
}
DEFAULT_PORTRAIT_COLOR = "#2b2b3a"

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

# Byleth (the Protagonist)'s gender is a player choice at the very start of
# the game - data/character_gender.csv already marks them "Any" so this
# choice never blocks a gender-locked class either way (see
# is_class_eligible's docstring). It has no effect on stats or eligibility,
# but it is the one thing that determines which portrait actually depicts
# them, for anyone who's added their own art per assets/portraits/README.md.
# So unlike every other character - whose single portrait file is just
# their lowercased name - Byleth's is resolved from one of these two slugs
# plus the selector rendered once in main() and threaded down through
# render_character_tab/render_team_tab/render_team into render_portrait.
BYLETH_PORTRAIT_SLUGS = {"Male": "byleth_m", "Female": "byleth_f"}
DEFAULT_BYLETH_GENDER = "Male"

st.set_page_config(page_title="Three Houses Class Optimizer", page_icon="⚔️", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, ...]:
    required = [
        "character_base_stats.csv", "character_growth_rates.csv", "class_stat_boosts.csv",
        "class_eligibility.csv", "character_gender.csv", "class_weapon_requirements.csv",
        "character_weapon_talent.csv", "recruitment_requirements.csv", "character_starting_level.csv",
        "class_growth_rates.csv", "class_base_stats.csv", "character_relics.csv",
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
        pd.read_csv(DATA_DIR / "class_growth_rates.csv"),
        pd.read_csv(DATA_DIR / "class_base_stats.csv"),
        pd.read_csv(DATA_DIR / "character_relics.csv"),
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


def get_portrait_path(character_name: str, byleth_gender: str | None = None) -> Path | None:
    """
    Look up a local portrait image for a character, if one has been placed
    in assets/portraits/ (see that folder's README - actual character art
    isn't shipped in this repo, since Three Houses portraits are Nintendo/
    Intelligent Systems IP and not this project's to redistribute; add your
    own from a source you have the rights to use).

    For the Protagonist specifically, byleth_gender ("Male"/"Female", from
    BYLETH_PORTRAIT_SLUGS - the selector in main() supplies this) is tried
    first as byleth_m/byleth_f, since Byleth's in-game portrait is a player
    choice rather than one fixed image the way every other character's is.
    A plain protagonist.* file - the ordinary lowercased-name slug every
    other character uses - is still checked as a fallback, for anyone who
    added one before splitting Byleth's art by gender, or who doesn't care
    to. byleth_gender is ignored for every other character.
    """
    if not PORTRAIT_DIR.exists():
        return None
    slugs = [character_name.lower().replace(" ", "_").replace("(", "").replace(")", "")]
    if character_name == "Protagonist":
        gender_slug = BYLETH_PORTRAIT_SLUGS.get(byleth_gender or DEFAULT_BYLETH_GENDER)
        slugs.insert(0, gender_slug)
    for slug in slugs:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = PORTRAIT_DIR / f"{slug}{ext}"
            if candidate.exists():
                return candidate
    return None


def get_house_lookup(base_stats_df: pd.DataFrame) -> dict:
    """Index data/character_base_stats.csv's "house" column by character name, for render_portrait's color coding."""
    return dict(zip(base_stats_df["name"], base_stats_df["house"]))


def render_portrait(character_name: str, house: str | None = None, width: int = 96, byleth_gender: str | None = None):
    """
    Render a character's portrait. If an image has been placed in
    assets/portraits/ for them (see get_portrait_path/that folder's README),
    it's used as-is - "bring your own art" stays supported for anyone who
    wants to add their own. Otherwise (the default for virtually everyone,
    since actual Three Houses character art isn't this project's to ship -
    see get_portrait_path), the fallback tile is color-coded by house/route
    (see HOUSE_COLORS) rather than a single flat neutral box, so portraits
    are still visually distinct and meaningful - which house/route a
    character belongs to - without redistributing anyone's IP. `house`
    should be that character's data/character_base_stats.csv "house" value
    (see get_house_lookup); omitted or unrecognized falls back to
    DEFAULT_PORTRAIT_COLOR. `byleth_gender` only matters when character_name
    is the Protagonist - see get_portrait_path.
    """
    portrait = get_portrait_path(character_name, byleth_gender=byleth_gender)
    if portrait:
        st.image(str(portrait), width=width)
    else:
        color = HOUSE_COLORS.get(house, DEFAULT_PORTRAIT_COLOR)
        title = house or ""
        st.markdown(
            f"<div title='{title}' style='width:{width}px;height:{width}px;border-radius:8px;"
            f"background:{color};color:#fff;display:flex;align-items:center;justify-content:center;"
            f"font-size:{width//3}px;font-weight:600;'>"
            f"{character_name[0]}</div>",
            unsafe_allow_html=True,
        )


def render_growth_rate_mini_chart(class_name: str, class_growth_lookup: dict, key: str) -> bool:
    """
    Compact per-stat growth-RATE modifier chart for one class (see
    optimizer.load_class_growth_lookup) - all of STAT_COLS at a glance, as
    a small horizontal bar chart, rather than a text summary of just the
    top couple of stats (the previous behavior here - see growth_caption
    in prior rounds). This is what makes growth-rate data visible on the
    Character Optimizer tab's own recommended-path view (render_path_with_
    mixmatch), not only when separately browsing a class in the Class
    Explorer tab (see render_class_explorer_tab's own, larger version of
    this same chart) - the Character Optimizer previously showed nothing
    beyond a 2-stat text caption here.

    Horizontal bars (stats on the y-axis) read better than vertical ones in
    the narrow per-tier columns render_path_with_mixmatch lays these out
    in - 9 rotated x-axis tick labels wouldn't fit, 9 horizontal bars do.
    Bars are colored by sign (a growth-rate modifier can be negative, e.g.
    Armored classes trading away Spd) so a negative modifier reads as a
    real cost, not just a smaller bar.

    Returns whether a chart was actually drawn - False (with a fallback
    caption, matching the Class Explorer tab's own wording) if class_name
    has no growth-rate data on file, so callers can render nothing further
    for that case.
    """
    mods = class_growth_lookup.get(class_name)
    if not mods:
        st.caption(f"No growth-rate modifier data on file for {class_name}.")
        return False

    values = [mods.get(s, 0) for s in STAT_COLS]
    colors = ["#d1453b" if v < 0 else "#2e8b57" for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=STAT_COLS, orientation="h", marker_color=colors,
        text=[f"{v:+g}%" for v in values], textposition="outside",
    ))
    max_abs = max((abs(v) for v in values), default=1) or 1
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis=dict(title=None, range=[-max_abs * 1.5, max_abs * 1.5], zeroline=True),
        yaxis=dict(autorange="reversed"),  # HP (STAT_COLS[0]) on top, reading order
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    return True


def render_path_with_mixmatch(
    path: list[dict],
    character: str,
    stat_boosts_df: pd.DataFrame,
    eligibility_lookup: dict | None,
    character_gender: str | None,
    role_weights: dict,
    role_used: str,
    weapon_req_lookup: dict,
    class_growth_lookup: dict,
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
    plus its own growth-rate modifiers (render_growth_rate_mini_chart), the
    same information the recommended pick gets, rather than just "your
    pick, overrides the recommendation" with no numbers to compare
    against.

    role_used is folded into every selectbox's widget key - switching the
    target role (or auto-detect landing on a different role) now resets
    each tier's selection back to that new role's own recommendation,
    instead of Streamlit silently keeping the previous role's picks
    selected under an unchanged key (the reported "switching the target
    role doesn't change the selected classes" bug).
    """
    st.subheader("Recommended Class Path")
    st.caption(
        "Swap in a different eligible class at any tier to compare - only the deepest tier's "
        "class feeds the projected stats below, since in-game a class boost doesn't stack across "
        "a career path, only your current class's boost counts. Growth-rate modifiers, unlike the "
        "flat boost, DO compound across every tier actually spent along the way - see the "
        "projection's caption below."
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
                key=f"mixmatch_{character}_{role_used}_{step['tier']}",
            )
            selected.append(choice)
            if choice == step["class"]:
                if step.get("is_unique_class"):
                    st.caption("⭐ Their own unique class")
                st.caption(f"fit score {step['score']}")
                if step.get("requirement"):
                    st.caption(f"Requires: {step['requirement']}")
                with st.expander("📈 Growth-rate modifiers"):
                    render_growth_rate_mini_chart(
                        choice, class_growth_lookup, key=f"growth_mini_{character}_{role_used}_{step['tier']}",
                    )
                if step.get("weapon_switch_warning"):
                    st.warning("Requires a weapon type never trained so far - a slow switch in practice.", icon="⚠️")
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
                with st.expander("📈 Growth-rate modifiers"):
                    render_growth_rate_mini_chart(
                        choice, class_growth_lookup, key=f"growth_mini_{character}_{role_used}_{step['tier']}",
                    )
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


def render_team(
    team: list[dict], dlc_names: set[str], house_lookup: dict | None = None, dancer: str | None = None,
    byleth_gender: str | None = None,
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
    """
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
                render_portrait(
                    member["character"], house=(house_lookup or {}).get(member["character"]),
                    byleth_gender=byleth_gender,
                )
            with col2:
                st.markdown(f"**{display_name(member['character'], dlc_names)}**")
                st.caption(f"{member['role']} (score {member['score']})")
                if member["character"] == dancer:
                    st.caption("💃 This team's Dancer (White Heron Cup) - swaps in instead of the path "
                               "at right when you need the Dance command; only one character on the "
                               "whole roster can hold it.")
            with col3:
                st.write(path_str)
                requirements = [step["requirement"] for step in member["path"] if step.get("requirement")]
                if requirements:
                    st.caption("Requires (final tier): " + requirements[-1])
                if member.get("weapon_switch_warning"):
                    st.caption(f"⚠️ {member['weapon_switch_warning']}")
                st.caption(f"Why on the team: {member['why']}")


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
            all_values = [growth_a.get(s, 0) for s in STAT_COLS]
            if row_b is not None:
                growth_b = class_growth_lookup.get(class_b, {})
                if growth_b:
                    fig.add_trace(go.Bar(x=STAT_COLS, y=[growth_b.get(s, 0) for s in STAT_COLS], name=class_b))
                    all_values += [growth_b.get(s, 0) for s in STAT_COLS]
            y_min = min(min(all_values), 0) * 1.2 if min(all_values) < 0 else 0
            y_max = max(max(all_values), 1) * 1.25
            fig.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=0, r=0, t=60, b=0),
                yaxis_title="Growth-rate modifier (percentage points per level-up)",
                yaxis=dict(range=[y_min, y_max]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"class_explorer_growth_{class_a}_{class_b}")
            st.caption(
                "Added to the character's own personal growth rate on every level-up spent in this "
                "class - e.g. a class with +15% Str here means +0.15 expected Str per level, on top "
                "of whatever the character's own Str growth rate already contributes."
            )


def render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                          weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df,
                          class_base_stats_df, character_relics_df, playable_names, dlc_names,
                          byleth_gender=None):
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
        class_growth_df=class_growth_df, class_base_stats_df=class_base_stats_df,
        character_relics_df=character_relics_df,
    )
    class_growth_lookup = load_class_growth_lookup(class_growth_df)
    class_base_stats_lookup = load_class_base_stats_lookup(class_base_stats_df)

    col_portrait, col_info = st.columns([1, 5])
    with col_portrait:
        character_house = base_stats_df.loc[base_stats_df["name"] == character, "house"].iloc[0] \
            if (base_stats_df["name"] == character).any() else None
        render_portrait(character, house=character_house, width=80, byleth_gender=byleth_gender)
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
        if result.get("weapon_switch_warning"):
            st.warning(result["weapon_switch_warning"], icon="⚠️")

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
        role_weights, result["role_used"], weapon_req_lookup, class_growth_lookup,
        include_dlc_classes=include_dlc_classes,
    )
    final_class = selected_path[-1]

    st.divider()
    effective_level = result["expected_stats_at_level"]
    # Always recomputed from the FULL actually-selected path (every tier,
    # not just the final one) - see stats_for_selected_path. A class
    # base-stat floor (see load_class_base_stats_lookup) can apply at any
    # tier, not just the last, so an earlier-tier mix-and-match override
    # can change the final numbers now, not just the path's flavor text -
    # this is what makes the projected-stats chart re-render on ANY tier
    # change, not only when the final tier itself was touched.
    selected_steps = [
        {"tier": step["tier"], "class": choice}
        for step, choice in zip(result["path"], selected_path)
    ]
    final_stats = stats_for_selected_path(
        character, selected_steps, base_stats_df, growth_rates_df, stat_boosts_df,
        effective_level, start_level=result["join_level"],
        class_growth_lookup=class_growth_lookup, class_base_stats_lookup=class_base_stats_lookup,
    )
    base_level_label = f"Join Level {result['join_level']}" if result["join_level"] > 1 else "Level 1"
    render_stat_charts(
        result["base_stats_at_join_level"], final_stats, character, final_class,
        base_level_label=base_level_label,
    )

    character_proficiency = load_character_proficiency_lookup(character_weapon_talent_df).get(character)

    st.button(
        "📥 Import this build into Team Builder", key=f"import_{character}",
        help="Locks in this exact class path/stats for the Team Builder tab, instead of it "
             "recomputing a recommendation for this character from scratch.",
        on_click=_import_build,
        args=(
            character, result, selected_path, final_class, final_stats,
            stat_boosts_df, role_weights, weapon_req_lookup, character_proficiency,
        ),
    )
    if character in st.session_state.get("imported_builds", {}):
        st.caption(f"✅ Imported - {character} will use this exact build in the Team Builder tab.")


def _import_build(
    character, result, selected_path, final_class, final_stats,
    stat_boosts_df, role_weights, weapon_req_lookup, character_proficiency,
):
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

    Every tier is rebuilt from the ACTUALLY-selected class (selected_path),
    not just copied from the tool's own recommendation (result["path"]) -
    this is what makes an earlier-tier override survive being imported at
    all (the original "imported build isn't the same as when it was
    imported" bug: importing only ever rewrote the final tier). But it's
    not enough to just swap in the new class name and leave every other
    field alone, either - "requirement", "score", and "weapon_switch_warning"
    are each specific to WHICH class occupies that tier and, for the
    warning, which skills were accumulated by that point in the ACTUAL
    selected path - so all three are recomputed here from the real
    selection, tier by tier, in order (accumulated_skills threaded forward
    exactly like recommend_path does). Leaving them copied from the
    original recommendation was the "now correctly imports changed classes
    but displays wrong final class requirements" bug: the class name
    updated, but its requirement/score/warning silently kept referring to
    whatever class the tool had originally recommended for that tier.
    """
    accumulated_skills = set(character_proficiency or [])
    path = []
    for step, choice in zip(result["path"], selected_path):
        if choice == step["class"]:
            # Unchanged from the recommendation - class/requirement/why/
            # is_unique_class stay as originally computed, but the warning
            # is still recomputed against accumulated_skills-so-far, since
            # an EARLIER tier's override can change what's "accumulated"
            # even when this tier's own pick didn't change.
            new_step = {
                **step,
                "weapon_switch_warning": weapon_switch_penalty(
                    choice, weapon_req_lookup, accumulated_skills, character_proficiency,
                ) > 0,
            }
        else:
            choice_row = stat_boosts_df[stat_boosts_df["name"] == choice]
            score = (
                round(float(score_class_for_role(choice_row.iloc[0], role_weights)), 2)
                if not choice_row.empty else None
            )
            new_step = {
                "tier": step["tier"],
                "class": choice,
                "score": score,
                "why": "Your pick - overrides the recommendation.",
                "requirement": format_requirement(choice, weapon_req_lookup),
                "is_unique_class": False,
                "weapon_switch_warning": weapon_switch_penalty(
                    choice, weapon_req_lookup, accumulated_skills, character_proficiency,
                ) > 0,
            }
        path.append(new_step)
        info = weapon_req_lookup.get(choice) if weapon_req_lookup else None
        if info:
            accumulated_skills |= {skill for skill, _ in info["requirements"]}

    imported = st.session_state.setdefault("imported_builds", {})
    imported[character] = {
        "path": path,
        "final_class": final_class,
        "expected_final_stats": final_stats,
        "eligible_unique_classes": result["eligible_unique_classes"],
    }


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
            "🔒 Force-deployed, always included: "
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
            locked_builds=imported_builds, class_growth_df=class_growth_df,
            class_base_stats_df=class_base_stats_df,
            character_relics_df=character_relics_df,
            force_deployed=set(mandatory),
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
                "💃 Assign this team's Dancer (optional)",
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
                byleth_gender=byleth_gender,
            )


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
        st.title("⚔️ Three Houses Class Optimizer")
    with byleth_gender_col:
        # Byleth's portrait is the one piece of art in this whole app that
        # depends on a choice the game itself hands the player, rather than
        # being fixed per character - see BYLETH_PORTRAIT_SLUGS. Rendered
        # once here (not per-tab) since Byleth is force-deployed on every
        # route and selectable directly in the Character Optimizer, so the
        # choice should carry across both tabs instead of being asked twice
        # or reset on every tab switch.
        byleth_gender = st.selectbox(
            "Byleth's portrait", options=list(BYLETH_PORTRAIT_SLUGS.keys()),
            index=list(BYLETH_PORTRAIT_SLUGS.keys()).index(DEFAULT_BYLETH_GENDER),
            key="byleth_portrait_gender",
            help="Only affects which portrait is shown if you've added your own "
                 "byleth_m/byleth_f art (see assets/portraits/README.md) - Byleth's "
                 "gender is a player choice in-game and never affects stats or class "
                 "eligibility either way.",
        )

    tab1, tab2, tab3 = st.tabs(["Character Optimizer", "Team Builder", "Class Explorer"])
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


if __name__ == "__main__":
    main()
