"""
tabs/character_tab.py

Character Optimizer tab: pick a character/role/level, show the
recommended class path with mix-and-match overrides, the projected-stats
charts, the per-role fit chart, and the growth-rate breakdown - plus the
"import this build into Team Builder" handoff (_import_build) that
team_tab.py's Team Builder tab picks up via st.session_state.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import (
    ROLE_PROFILES,
    STAT_COLS,
    WEAPON_SWITCH_WARNING_ENABLED,
    class_growth_axis_range,
    compute_roster_stat_stats,
    format_combined_requirements,
    format_requirement,
    growth_stack_axis_range,
    list_eligible_classes_at_tier,
    load_character_proficiency_lookup,
    load_character_relic_lookup,
    load_class_base_stats_lookup,
    load_class_growth_lookup,
    load_eligibility_lookup,
    load_weapon_requirements_lookup,
    path_weapon_switch_warning,
    recommend_for_character,
    role_relevant_stat_deltas,
    score_all_roles,
    score_class_for_role,
    stats_for_selected_path,
    weapon_switch_penalty,
)
from app_state import (
    DEFAULT_TARGET_LEVEL,
    display_name,
    query_param_int,
    query_param_str,
    render_portrait,
    resolve_character_gender,
    sync_share_query_params,
)




def render_growth_rate_mini_chart(class_name: str, class_growth_lookup: dict, key: str, axis_range: tuple[float, float]) -> bool:
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

    axis_range is the caller-supplied (min, max) from
    optimizer.class_growth_axis_range - a single global range shared by
    EVERY standalone growth-rate chart in the app (this one, and the Class
    Explorer's), so a bar's height means the same thing no matter which
    class or tier is being shown, rather than each chart auto-scaling to
    its own selection and silently making a middling modifier look as
    tall as a dramatically better/worse one elsewhere.

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
    axis_min, axis_max = axis_range
    # A little headroom beyond the exact global min/max so the outside
    # text labels (textposition="outside") never get clipped at the
    # chart's own edge - same fixed padding on every chart, so it doesn't
    # reintroduce the per-chart inconsistency this range is meant to fix.
    span = max(axis_max - axis_min, 1.0)
    padding = span * 0.15
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis=dict(title=None, range=[axis_min - padding, axis_max + padding], zeroline=True),
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
    growth_axis_range: tuple[float, float],
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

    Every choice - the recommended pick or an override - also shows the
    class's own flat stat boost narrowed to just the stat(s) role_weights
    actually cares about (see optimizer.role_relevant_stat_deltas), e.g.
    "Relevant to Magic Attacker: +3 Mag, +1 Dex" - so switching classes
    shows what that switch actually buys FOR THE ROLE being built toward,
    not just a fit-score number.
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
            choice_row = stat_boosts_df[stat_boosts_df["name"] == choice]
            relevant_deltas = (
                role_relevant_stat_deltas(choice_row.iloc[0], role_weights) if not choice_row.empty else []
            )
            if choice == step["class"]:
                if step.get("is_unique_class"):
                    st.caption("Their own unique class")
                st.caption(f"fit score {step['score']}")
                if relevant_deltas:
                    st.caption(f"Relevant to {role_used}: " + format_stat_deltas(relevant_deltas))
                if step.get("requirement"):
                    st.caption(f"Requires: {step['requirement']}")
                with st.expander("Growth-rate modifiers"):
                    render_growth_rate_mini_chart(
                        choice, class_growth_lookup, key=f"growth_mini_{character}_{role_used}_{step['tier']}",
                        axis_range=growth_axis_range,
                    )
                if WEAPON_SWITCH_WARNING_ENABLED and step.get("weapon_switch_warning"):
                    st.warning("Requires a weapon type never trained so far - a slow switch in practice.")
                st.caption(step["why"])
            else:
                choice_score = (
                    round(float(score_class_for_role(choice_row.iloc[0], role_weights)), 2)
                    if not choice_row.empty else None
                )
                st.caption("Your pick - overrides the recommendation.")
                if choice_score is not None:
                    st.caption(f"fit score {choice_score}")
                if relevant_deltas:
                    st.caption(f"Relevant to {role_used}: " + format_stat_deltas(relevant_deltas))
                requirement = format_requirement(choice, weapon_req_lookup)
                if requirement:
                    st.caption(f"Requires: {requirement}")
                with st.expander("Growth-rate modifiers"):
                    render_growth_rate_mini_chart(
                        choice, class_growth_lookup, key=f"growth_mini_{character}_{role_used}_{step['tier']}",
                        axis_range=growth_axis_range,
                    )
    return selected



def format_stat_deltas(deltas: list[tuple[str, float]]) -> str:
    """"+3 Str, +1 Spd" from [("Str", 3), ("Spd", 1)] - see role_relevant_stat_deltas."""
    return ", ".join(
        f"{'+' if value >= 0 else ''}{int(value) if float(value).is_integer() else value} {stat}"
        for stat, value in deltas
    )



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



def render_role_fit_chart(role_scores: dict, used_role: str, key: str):
    """
    Bar chart of a character's fit-similarity score (optimizer.score_all_roles)
    across EVERY role archetype, not just the single auto-detected winner -
    "when you select a character, all you can see is what is considered
    their best fit, but you can't see how well the program considers them
    to fit other roles." The role actually being targeted (used_role -
    either the auto-detected winner, or a manually-picked one) is colored
    differently so it stays easy to pick out among all five bars.

    These are the same standardized-growth-rate-cosine-similarity numbers
    behind auto-detection (plus the small natural_role_affinity_bonus
    nudge - see that function), not a fit score for any specific CLASS -
    "how naturally does this character's own growth lean toward each
    role," independent of which classes happen to be available at any
    given tier.
    """
    roles = list(role_scores.keys())
    values = [role_scores[r] for r in roles]
    order = sorted(range(len(roles)), key=lambda i: values[i])
    roles = [roles[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#2e8b57" if r == used_role else "#6c7a89" for r in roles]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=roles, orientation="h", marker_color=colors,
        text=[f"{v:+.2f}" for v in values], textposition="outside",
    ))
    max_abs = max((abs(v) for v in values), default=1) or 1
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis=dict(title="Growth-rate fit (higher = more natural)", range=[-max_abs * 1.3, max_abs * 1.6]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)



def render_growth_stack_chart(
    character: str, growth_row: pd.Series, selected_steps: list[dict], class_growth_lookup: dict,
    stack_axis_range: tuple[float, float],
):
    """
    Stacked bar chart of a character's own growth rate plus ONE tier's
    class growth-rate modifier (per stat), with a control to pick which
    tier in the actually-selected path to overlay - "an option to see
    character growths as well, perhaps a bar chart with character growths
    with class growths stacked on top, and click which of the classes in
    the path you want displayed." Defaults to the deepest/final tier,
    since that's the class actually driving the projected-stats chart
    above, but every tier in the path is selectable.

    stack_axis_range is the caller-supplied (min, max) from
    optimizer.growth_stack_axis_range - fixed across every character/tier
    this chart is ever shown for, so a stacked total's height is always
    comparable (e.g. a stat that's already near its practical ceiling
    under one class never LOOKS shorter than a genuinely smaller total
    for a different class/tier just because each chart auto-scaled to its
    own numbers). Deliberately a different, wider range than the
    standalone growth-rate charts (class_growth_axis_range) - see
    growth_stack_axis_range's own docstring for why the two shouldn't
    share one scale.
    """
    if not selected_steps:
        return
    st.subheader("Growth Breakdown")
    tier_labels = [f"{step['tier']} ({step['class']})" for step in selected_steps]
    chosen_label = st.radio(
        "Show growth breakdown for:", options=tier_labels, index=len(tier_labels) - 1,
        key=f"growth_stack_tier_{character}", horizontal=True,
    )
    chosen_step = selected_steps[tier_labels.index(chosen_label)]
    class_mods = class_growth_lookup.get(chosen_step["class"], {})

    character_values = [float(growth_row[s]) for s in STAT_COLS]
    class_values = [class_mods.get(s, 0) for s in STAT_COLS]

    axis_min, axis_max = stack_axis_range
    span = max(axis_max - axis_min, 1.0)
    padding = span * 0.1

    fig = go.Figure()
    fig.add_trace(go.Bar(x=STAT_COLS, y=character_values, name=f"{character}'s own growth"))
    fig.add_trace(go.Bar(x=STAT_COLS, y=class_values, name=f"{chosen_step['class']} modifier"))
    fig.update_layout(
        barmode="relative",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Growth rate (% per level-up)",
        yaxis=dict(range=[axis_min - padding, axis_max + padding]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"growth_stack_{character}_{chosen_step['tier']}")
    st.caption(
        "Character growth (bottom) plus this class's own growth-rate modifier (stacked on top, "
        "which can be negative for some stats/classes) - the combined rate actually used for "
        "that stat on every level-up spent in this class."
    )



def render_character_tab(base_stats_df, growth_rates_df, stat_boosts_df, eligibility_df, character_gender_df,
                          weapon_req_df, character_weapon_talent_df, starting_level_df, class_growth_df,
                          class_base_stats_df, character_relics_df, playable_names, dlc_names,
                          byleth_gender=None):
    st.caption(
        "Pick a character and see a recommended class path - either toward their "
        "natural strengths, or a role you choose. Only classes that character can "
        "actually access (character/gender-locked classes included) are considered. "
        "New here? Try Bernadetta as a Sniper, or leave the role on auto-detect and "
        "see what the tool thinks any character is naturally built for - the "
        "\"fit score\" on each step is just how well that class's own stat boosts "
        "line up with the role you're targeting; higher is a better match."
    )

    default_character = query_param_str("character")
    default_role = query_param_str("role")
    default_level = query_param_int("level", DEFAULT_TARGET_LEVEL)
    role_options = ["Auto-detect from growth rates"] + list(ROLE_PROFILES.keys())

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        character = st.selectbox(
            "Character", options=playable_names, key="char_select",
            index=playable_names.index(default_character) if default_character in playable_names else 0,
            format_func=lambda name: display_name(name, dlc_names),
        )
    with col2:
        role_choice = st.selectbox(
            "Target role", options=role_options, key="char_role_select",
            index=role_options.index(default_role) if default_role in role_options else 0,
        )
    with col3:
        target_level = st.slider(
            "Target level", min_value=5, max_value=40,
            value=default_level if 5 <= default_level <= 40 else DEFAULT_TARGET_LEVEL,
            key="char_level",
        )

    include_dlc_classes = st.checkbox(
        "Include DLC classes (Cindered Shadows certification classes)", value=False,
        key="char_include_dlc_classes",
        help="Also considers Trickster, War Monk/Cleric, Dark Flier and Valkyrie as Advanced-tier options - "
             "these need the Cindered Shadows DLC, same as the DLC-exclusive characters do.",
    )

    role_name = None if role_choice == "Auto-detect from growth rates" else role_choice
    sync_share_query_params(character, role_choice, target_level)

    character_gender = resolve_character_gender(character, character_gender_df, byleth_gender)

    result = recommend_for_character(
        character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=role_name, target_level=target_level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
        starting_level_df=starting_level_df, include_dlc_classes=include_dlc_classes,
        class_growth_df=class_growth_df, class_base_stats_df=class_base_stats_df,
        character_relics_df=character_relics_df,
        character_gender_override=character_gender if character == "Protagonist" else None,
    )
    class_growth_lookup = load_class_growth_lookup(class_growth_df)
    growth_axis_range = class_growth_axis_range(class_growth_lookup)
    growth_stack_range = growth_stack_axis_range(class_growth_lookup, growth_rates_df)
    class_base_stats_lookup = load_class_base_stats_lookup(class_base_stats_df)
    character_proficiency = load_character_proficiency_lookup(character_weapon_talent_df).get(character)
    character_relic_weapon_types = load_character_relic_lookup(character_relics_df).get(character)

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
                f"Joins the roster at level {result['join_level']} - base stats and the "
                f"projection below start from there, not level 1."
            )
        if result["expected_stats_at_level"] != result["requested_target_level"]:
            st.caption(
                f"Target level {result['requested_target_level']} is below their join level - "
                f"projecting to level {result['expected_stats_at_level']} instead."
            )

    with st.expander("How well does this character fit every role? (not just the best one)"):
        roster_means, roster_stds = compute_roster_stat_stats(growth_rates_df)
        growth_row = growth_rates_df[growth_rates_df["name"] == character].iloc[0]
        role_scores = score_all_roles(
            growth_row, roster_means, roster_stds,
            character_proficiency=character_proficiency,
            character_relic_weapon_types=character_relic_weapon_types,
        )
        render_role_fit_chart(role_scores, result["role_used"], key=f"role_fit_{character}")
        st.caption(
            "Every role's growth-rate fit for this character, not just the auto-detected best one - "
            "the highlighted bar is whichever role is currently being targeted above."
        )

    if not result["path"]:
        st.warning("No class path could be built - check that class_stat_boosts.csv has data for all tiers.")
        return

    eligibility_lookup = load_eligibility_lookup(eligibility_df)
    weapon_req_lookup = load_weapon_requirements_lookup(weapon_req_df)
    role_weights = ROLE_PROFILES[result["role_used"]]

    selected_path = render_path_with_mixmatch(
        result["path"], character, stat_boosts_df, eligibility_lookup, character_gender,
        role_weights, result["role_used"], weapon_req_lookup, class_growth_lookup,
        growth_axis_range, include_dlc_classes=include_dlc_classes,
    )
    final_class = selected_path[-1]

    # Recomputed from the FULL actually-selected path (every tier, not just
    # the final one) - see stats_for_selected_path. A class base-stat floor
    # (see load_class_base_stats_lookup) can apply at any tier, not just
    # the last, so an earlier-tier mix-and-match override can change the
    # final numbers now, not just the path's flavor text.
    selected_steps = [
        {"tier": step["tier"], "class": choice}
        for step, choice in zip(result["path"], selected_path)
    ]

    # Recomputed from selected_steps, NOT reused from
    # result["weapon_switch_warning"] (which only ever reflects the tool's
    # ORIGINAL recommendation) - this is the fix for "if a warning is
    # displayed and I change the class so there is no longer a class
    # warning, the overall warning is still displayed."
    combined_switch_warning = path_weapon_switch_warning(
        character, selected_steps, weapon_req_lookup, character_proficiency,
    )
    if WEAPON_SWITCH_WARNING_ENABLED and combined_switch_warning:
        st.warning(combined_switch_warning)

    combined_requirement = format_combined_requirements(
        [step["class"] for step in selected_steps], weapon_req_lookup,
    )
    if combined_requirement:
        st.caption(f"Skill ranks needed across this whole path: {combined_requirement}")

    st.divider()
    effective_level = result["expected_stats_at_level"]
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

    render_growth_stack_chart(character, growth_row, selected_steps, class_growth_lookup, growth_stack_range)

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
        st.caption(f"Imported - {character} will use this exact build in the Team Builder tab.")



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
