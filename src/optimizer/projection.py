"""
optimizer/projection.py

Expected-value stat projection: base stats at join level, a class's own
base-stat floor, and simulating growth either as a single flat-rate span
(expected_stats_at_level) or tier-by-tier along an actual class path
(expected_stats_along_path / path_level_bands), the mechanic that makes a
non-final mix-and-match override actually change the final numbers.
"""
import pandas as pd

from .constants import STAT_COLS, TIER_LEVEL_REQUIREMENTS



def apply_class_base_stat_floor(value: float, class_name: str, stat: str, class_base_stats_lookup: dict | None) -> float:
    """
    The floor step itself, in one place so expected_stats_at_level and
    expected_stats_along_path apply it identically: value snapped up to
    class_name's own base stat for `stat` (load_class_base_stats_lookup),
    if that's higher than value - otherwise value is returned unchanged.
    A class/stat absent from the lookup (no data on file, or
    class_base_stats_lookup itself is empty/None) is a no-op, not an error -
    same graceful-degradation precedent as every other optional lookup in
    this module.
    """
    if not class_base_stats_lookup:
        return value
    class_base = class_base_stats_lookup.get(class_name, {}).get(stat)
    if class_base is None:
        return value
    return max(value, float(class_base))



def base_stats_at_join_level(
    base_row: pd.Series,
    growth_row: pd.Series,
    join_level: int,
) -> dict:
    """
    A character's expected stats at their own join level, starting from
    Serenes Forest's Level-1-baseline base stats (see
    load_starting_level_lookup) and adding expected growth for the levels
    between 1 and join_level - the same expected-value approach
    expected_stats_at_level uses for level-ups in general (growth_rate% per
    level, no class boost - Noble/Commoner, the starting classes, both have
    an all-zero boost row, so there's nothing to add there). join_level=1
    (the default for most of the roster) returns the raw base stats
    unchanged, since there are no levels to bridge.
    """
    levels_gained = max(join_level - 1, 0)
    return {
        stat: round(base_row[stat] + (growth_row[stat] / 100) * levels_gained, 1)
        for stat in STAT_COLS
    }



def expected_stats_at_level(
    base_row: pd.Series,
    growth_row: pd.Series,
    final_class_boost_row: pd.Series,
    target_level: int = 30,
    start_level: int = 1,
    final_class_name: str | None = None,
    class_base_stats_lookup: dict | None = None,
) -> dict:
    """
    Estimate a character's expected stats at target_level in a given final
    class: base stats (still Serenes Forest's Level-1 baseline - see
    load_starting_level_lookup), snapped up to that class's own base-stat
    floor if the character's base is lower (see
    apply_class_base_stat_floor / load_class_base_stats_lookup - only
    applied when final_class_name and class_base_stats_lookup are both
    given, so old callers that don't pass them keep today's floor-free
    behavior) + expected level-up gains from start_level up to target_level
    (growth_rate% per level, used as an expected value - e.g. a 45% growth
    rate contributes 0.45 expected stat points per level, not a simulated
    coin flip) + the final class's stat boost.

    This is the flat single-class fallback (see expected_stats_along_path
    for the real, path-wide, per-tier-floor version) - it treats
    final_class_name as if it had been the character's class for the
    ENTIRE start_level..target_level span, floor applied once at
    start_level, same simplification the growth rate already made here
    before per-tier simulation existed.

    start_level defaults to 1 (today's behavior - every level from 1 to
    target_level counts) but should be a character's own join level when
    known (see load_starting_level_lookup / base_stats_at_join_level) - a
    character who doesn't actually join the roster until level 15 has
    already banked those 14 levels' worth of growth by the time you can
    recruit them, so target_level should never be projected as if they
    started from level 1 the way a Ch1 house student did. target_level
    below start_level is treated as target_level == start_level (0 levels
    gained) rather than producing a negative gain - callers should clamp
    the target-level input itself (see recommend_for_character) so this is
    just a safety floor, not the primary enforcement point.

    This is an expected-value calculation, not a single simulated
    playthrough - it answers "on average, how strong would this character
    be," which is what matters for comparing class paths against each
    other. A Monte Carlo version (simulating many individual playthroughs
    to show the range of outcomes, not just the average) is a natural
    upgrade for a later stage.
    """
    levels_gained = max(target_level - start_level, 0)
    result = {}
    for stat in STAT_COLS:
        base = float(base_row[stat])
        if final_class_name is not None:
            base = apply_class_base_stat_floor(base, final_class_name, stat, class_base_stats_lookup)
        expected_growth = (growth_row[stat] / 100) * levels_gained
        boost = final_class_boost_row[stat] if stat in final_class_boost_row.index else 0
        result[stat] = round(base + expected_growth + boost, 1)
    return result



def path_level_bands(
    path: list[dict], target_level: int, start_level: int = 1,
) -> list[tuple[str, str, int, int]]:
    """
    Split [start_level, target_level] into per-tier bands matching each
    path step's own reachable window, so growth can be simulated with
    each tier's own class along the way, rather than blended as if only
    the final class had ever applied (see expected_stats_along_path).

    A step's band starts at max(that tier's level requirement,
    start_level) - never earlier than the character's own join level, if
    later than the tier's normal unlock - and ends where the next step's
    band starts (or at target_level, for the last step). A band that
    would be empty after clamping (e.g. a character's join level already
    skips past an early tier entirely) is omitted rather than yielding a
    zero/negative-length entry.
    """
    if not path:
        return []
    bands = []
    for i, step in enumerate(path):
        band_start = max(TIER_LEVEL_REQUIREMENTS.get(step["tier"], 1), start_level)
        if band_start > target_level:
            continue
        if i + 1 < len(path):
            next_start = max(TIER_LEVEL_REQUIREMENTS.get(path[i + 1]["tier"], target_level), start_level)
            band_end = min(next_start, target_level)
        else:
            band_end = target_level
        if band_end <= band_start:
            continue
        bands.append((step["tier"], step["class"], band_start, band_end))
    return bands



def expected_stats_along_path(
    path: list[dict],
    base_row: dict,
    growth_row: pd.Series,
    final_class_boost_row: pd.Series,
    class_growth_lookup: dict,
    target_level: int = 30,
    start_level: int = 1,
    class_base_stats_lookup: dict | None = None,
) -> dict:
    """
    Like expected_stats_at_level, but applies each stat's growth using the
    combined character-growth-rate + PER-TIER class growth-rate modifier
    (see load_class_growth_lookup) for the levels actually spent in that
    tier's class along the path (see path_level_bands), instead of a
    single flat rate blended across the whole span. Classes really do
    modify how fast a stat grows on every level-up in this game, not just
    contribute a one-time flat boost - "tool says classes don't have
    growth rates - this is very wrong" was the original report, and this
    is the fix: growth gains are cumulative across every tier actually
    spent on the way to target_level, since those level-ups already
    happened, while the FINAL class's flat stat boost is still applied
    only once at the end (boosts don't stack across a career path - that
    part of the existing model is correct and unchanged).

    class_base_stats_lookup (see load_class_base_stats_lookup), if given,
    also applies each tier's own class base-stat FLOOR at the moment the
    path certifies into it - i.e. at the start of that tier's own band,
    before simulating that band's growth (see apply_class_base_stat_floor):
    a stat already below that class's own base snaps up to it; a stat
    already above is untouched. This is a genuinely different mechanic
    from the flat one-time class BOOST (which only ever applies once, from
    the final class, at the very end) - the floor can matter at EVERY tier
    along the path, not just the last one, which is also what makes an
    earlier, non-final tier's mix-and-match override actually change the
    final projected stats now (see app.py's render_character_tab).

    base_row should be the character's stats at start_level already (see
    base_stats_at_join_level) - this function only simulates growth
    forward from there, it doesn't re-derive a level-1 baseline.

    Falls back to expected_stats_at_level's flat-rate behavior (no
    per-class growth modeling, floor applied once for the final class only)
    if class_growth_lookup is empty/None or the path yields no bands - e.g.
    an empty path, or every tier already passed by start_level - so this
    stays backward compatible with unmodeled classes/DLC rows not present
    in data/class_growth_rates.csv.
    """
    bands = path_level_bands(path, target_level, start_level)
    if not class_growth_lookup or not bands:
        final_class_name = path[-1]["class"] if path else None
        return expected_stats_at_level(
            base_row, growth_row, final_class_boost_row, target_level, start_level,
            final_class_name=final_class_name, class_base_stats_lookup=class_base_stats_lookup,
        )

    # Levels between start_level and the first tier's own band (e.g. 1-4,
    # before Beginner's level-5 requirement) still happen - the character
    # just hasn't reached their first real class yet (still Noble/Commoner,
    # whose own growth-rate modifiers are ~0 - see data/class_growth_rates.csv),
    # so they're simulated with the character's own growth rate alone
    # rather than silently skipped (which would understate every stat by a
    # few levels' worth of growth and break the "levels gained sums to
    # target_level - start_level" invariant - see TestPathLevelBands). No
    # class-base-stat floor applies here either - the character hasn't
    # certified into anything yet.
    pre_band_levels = max(bands[0][2] - start_level, 0)

    result = {}
    for stat in STAT_COLS:
        value = float(base_row[stat]) + (float(growth_row[stat]) / 100) * pre_band_levels
        for _tier, class_name, band_start, band_end in bands:
            # The floor applies at the MOMENT of certifying into this
            # tier's class - before that tier's own growth is simulated,
            # not stacked on top of it.
            value = apply_class_base_stat_floor(value, class_name, stat, class_base_stats_lookup)
            levels = band_end - band_start
            if levels <= 0:
                continue
            class_growth = class_growth_lookup.get(class_name, {})
            combined_rate = (float(growth_row[stat]) + class_growth.get(stat, 0)) / 100
            value += combined_rate * levels
        boost = final_class_boost_row[stat] if stat in final_class_boost_row.index else 0
        result[stat] = round(value + boost, 1)
    return result



def stats_for_class_at_level(
    character_name: str,
    class_name: str,
    base_stats_df: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
    stat_boosts_df: pd.DataFrame,
    target_level: int = 30,
    start_level: int = 1,
    class_growth_lookup: dict | None = None,
    class_base_stats_lookup: dict | None = None,
) -> dict | None:
    """
    Expected stats for character_name at target_level, if their current
    class were class_name specifically - the same expected-value
    calculation recommend_for_character uses for its recommended final
    class, just callable for any class the user picks by hand.

    start_level should be the character's own join level when known (see
    load_starting_level_lookup). class_growth_lookup (see
    load_class_growth_lookup), if given, applies class_name's own
    growth-rate modifiers across the whole start_level..target_level span
    (treating it as "if this had been their class the entire time," the
    same assumption stats_for_class_at_level already makes about the flat
    boost) instead of the character's raw growth rate alone; omit for the
    old flat-rate-only behavior. class_base_stats_lookup (see
    load_class_base_stats_lookup), if given, applies class_name's own
    base-stat floor once, at start_level (same "as if this had been their
    class the whole time" simplification) - see
    apply_class_base_stat_floor.

    This treats class_name as the character's class for the WHOLE
    start_level..target_level span - a single-tier simplification, not the
    real per-tier-floor path simulation (see expected_stats_along_path /
    stats_for_selected_path, which is what the "mix and match" path
    override in the UI actually uses as of round 5, since a real path
    spends real levels in EARLIER tiers too, each with their own floor).
    Kept as a public, independently-testable building block and for
    single-class "what if" queries that aren't about a specific path.

    Returns None if class_name isn't in stat_boosts_df at all.
    """
    boost_rows = stat_boosts_df[stat_boosts_df["name"] == class_name]
    if boost_rows.empty:
        return None
    base_row = base_stats_df[base_stats_df["name"] == character_name].iloc[0]
    growth_row = growth_rates_df[growth_rates_df["name"] == character_name].iloc[0]
    if class_growth_lookup:
        class_growth = class_growth_lookup.get(class_name, {})
        combined_growth = growth_row[STAT_COLS].astype(float).copy()
        for stat in STAT_COLS:
            combined_growth[stat] = combined_growth[stat] + class_growth.get(stat, 0)
        return expected_stats_at_level(
            base_row, combined_growth, boost_rows.iloc[0], target_level, start_level,
            final_class_name=class_name, class_base_stats_lookup=class_base_stats_lookup,
        )
    return expected_stats_at_level(
        base_row, growth_row, boost_rows.iloc[0], target_level, start_level,
        final_class_name=class_name, class_base_stats_lookup=class_base_stats_lookup,
    )



def stats_for_selected_path(
    character_name: str,
    selected_steps: list[dict],
    base_stats_df: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
    stat_boosts_df: pd.DataFrame,
    target_level: int = 30,
    start_level: int = 1,
    class_growth_lookup: dict | None = None,
    class_base_stats_lookup: dict | None = None,
) -> dict | None:
    """
    Expected stats for character_name at target_level, along the ACTUAL
    sequence of classes in selected_steps ([{"tier": ..., "class": ...},
    ...], in path order) - the real, per-tier-floor path simulation (see
    expected_stats_along_path) behind the "mix and match" override in the
    UI, as of round 5.

    This matters, and is different from just calling stats_for_class_at_level
    on the final tier alone, because of the class base-stat floor (see
    load_class_base_stats_lookup): a floor applied by an EARLIER tier in
    the actually-selected path (not just the recommended one) can raise a
    stat that a later tier's growth then compounds on top of - so changing
    a non-final tier's mix-and-match dropdown can genuinely change the
    final projected stats now, not just the path's flavor text. Before
    round 5, only the final class's flat boost mattered for the numbers,
    so the projected-stats chart correctly didn't need to re-render on a
    non-final tier change; the floor mechanic is what makes every tier
    matter (see app.py's render_character_tab, which now always calls this
    instead of branching on "did only the final tier change").

    Returns None if character_name/growth data can't be found, or
    selected_steps is empty (nothing to project).
    """
    if not selected_steps:
        return None
    base_rows = base_stats_df[base_stats_df["name"] == character_name]
    growth_rows = growth_rates_df[growth_rates_df["name"] == character_name]
    if base_rows.empty or growth_rows.empty:
        return None
    final_class_name = selected_steps[-1]["class"]
    final_boost_rows = stat_boosts_df[stat_boosts_df["name"] == final_class_name]
    if final_boost_rows.empty:
        return None

    base_row = base_rows.iloc[0]
    growth_row = growth_rows.iloc[0]
    base_stats_at_start = base_stats_at_join_level(base_row, growth_row, start_level)

    return expected_stats_along_path(
        selected_steps, base_stats_at_start, growth_row, final_boost_rows.iloc[0],
        class_growth_lookup or {}, target_level, start_level=start_level,
        class_base_stats_lookup=class_base_stats_lookup,
    )
