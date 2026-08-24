"""
optimizer.py

v1 class-path recommender for Fire Emblem: Three Houses.

Given a character, recommends a class path (one class per tier: Beginner ->
Intermediate -> Advanced -> Master) either toward a specific target role
(e.g. "Tank") or auto-detected from the character's own growth rates (i.e.
"what is this character naturally good at?").

Known simplifications (v1):
  - Only Beginner/Intermediate/Advanced/Master/Unique tiers exist in the
    data; Unique classes (Emperor, Barbarossa, etc.) are no longer
    blanket-excluded - character/gender eligibility (data/class_eligibility.csv)
    is checked so a class only gets recommended to characters who can
    actually access it. Unique classes still aren't spliced into the
    Beginner->Master path itself (see eligible_unique_classes) since they
    don't unlock on that tier's level+seal system - they're surfaced as a
    separate "also eligible for" list instead. NPC/Enemy classes aren't
    player-selectable at all and remain excluded. DLC Exclusive classes are
    excluded from the standard path for the same "not always available"
    reasoning, though their eligibility rows still exist in
    class_eligibility.csv for when that changes.
  - Class unlocks are gated by tier level requirements (Beginner=5,
    Intermediate=10, Advanced=20, Master=30, verified against Serenes
    Forest's detailed-view page) but NOT by the game's real skill-
    proficiency requirements (e.g. minimum Sword/Faith/etc. rank), which
    we don't have data for yet. A path is truncated to whichever tiers are
    reachable at the requested target_level - e.g. targeting level 15 only
    considers Beginner and Intermediate, since Advanced requires level 20.
    This is still an approximation: it assumes proficiency ranks keep pace
    with level, which the in-game class requirements don't guarantee.
  - Only the FINAL class's stat boost is applied when estimating end-state
    stats - in the actual game, boosts don't stack across a career path,
    only your current class's boost counts. The recommended path's earlier
    tiers are shown for narrative/progression flavor, not because their
    boosts contribute to the final numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STAT_COLS = ["HP", "Str", "Mag", "Dex", "Spd", "Lck", "Def", "Res", "Cha"]
TIER_ORDER = ["Beginner", "Intermediate", "Advanced", "Master"]

# Minimum character level required to access each tier, per Serenes
# Forest's detailed-view page. (Unique classes are level 1 / start, but
# they're out of scope for v1 - see module docstring.)
TIER_LEVEL_REQUIREMENTS = {
    "Beginner": 5,
    "Intermediate": 10,
    "Advanced": 20,
    "Master": 30,
}


def reachable_tiers(target_level: int, tiers: list[str] = TIER_ORDER) -> list[str]:
    """
    Given a target level, return the subset of tiers (in order) whose level
    requirement is met at that level. E.g. target_level=15 with the default
    tier order returns ["Beginner", "Intermediate"], since Advanced needs
    level 20.
    """
    return [tier for tier in tiers if TIER_LEVEL_REQUIREMENTS.get(tier, 0) <= target_level]


def load_eligibility_lookup(eligibility_df: pd.DataFrame) -> dict:
    """
    Index data/class_eligibility.csv by class_name for fast lookup during
    recommendation. Each entry is {"characters": set|None, "gender":
    str|None} - None in either field means unrestricted on that axis.

    class_eligibility.csv itself is a small, hand-verified table (checked
    against Serenes Forest's classes page - which marks gender-locked
    classes with [M]/[F] tags - and cross-referenced against three other
    sources), hardcoded here rather than added to scrape_serenes.py. This
    mirrors the same call made for TIER_LEVEL_REQUIREMENTS: it's a small,
    stable dataset (23 rows) that doesn't change between game updates, so
    scraping it isn't worth a new page-parsing path. Classes not listed in
    the CSV are unrestricted by default.
    """
    lookup = {}
    for _, row in eligibility_df.iterrows():
        chars = row.get("locked_to_characters")
        gender = row.get("locked_to_gender")
        lookup[row["class_name"]] = {
            "characters": set(chars.split("|")) if isinstance(chars, str) and chars else None,
            "gender": gender if isinstance(gender, str) and gender else None,
        }
    return lookup


def is_class_eligible(
    character_name: str,
    class_name: str,
    eligibility_lookup: dict,
    character_gender: str | None = None,
) -> bool:
    """
    Whether character_name may access class_name, per eligibility_lookup.

    A class absent from eligibility_lookup is unrestricted (eligible to
    everyone). A restricted class requires the character to be in its
    locked_to_characters set (when one is specified) AND to match its
    locked_to_gender (when one is specified). A character_gender of "Any"
    (the Protagonist, whose gender is a player choice) or None (gender data
    wasn't supplied - degrade gracefully rather than block) always passes
    the gender check; character-lock checks still apply regardless.
    """
    restriction = eligibility_lookup.get(class_name)
    if restriction is None:
        return True

    allowed_characters = restriction["characters"]
    if allowed_characters is not None and character_name not in allowed_characters:
        return False

    required_gender = restriction["gender"]
    if required_gender is not None and character_gender not in (required_gender, "Any", None):
        return False

    return True


# Role archetypes as stat-weight profiles. Weights are relative importance,
# not required to sum to 1 - only relative magnitude matters for scoring.
#
# Note on "Speed/Precision" (previously named "Flier/Mobility"): Movement
# (Mov) isn't a growth stat - it's purely a class trait, not something a
# character has an innate growth rate for. That means natural role
# DETECTION (which only has growth rates to work with) can't actually tell
# whether a character has any flying affinity; a Mov weight there would
# silently contribute nothing. What the growth data CAN show is a Dex/Spd
# lean, which is better described honestly as speed/precision. Mov still
# matters when scoring CLASSES for this role (see score_class_for_role) -
# a real class does have a Mov stat - so a Speed/Precision character can
# still get steered toward flying classes if those score best, just without
# the role's name overclaiming what growth-rate data alone can detect.
ROLE_PROFILES = {
    "Physical Attacker": {"Str": 1.0, "Spd": 0.5},
    "Magic Attacker": {"Mag": 1.0, "Dex": 0.3},
    "Tank": {"HP": 1.0, "Def": 1.0},
    "Support": {"Res": 0.7, "Cha": 0.7},
    "Speed/Precision": {"Spd": 0.7, "Dex": 0.3, "Mov": 1.0},
}


def role_to_vector(role_weights: dict, stat_cols: list[str]) -> np.ndarray:
    """Turn a role's {stat: weight} dict into a vector aligned to stat_cols (0 for unlisted stats)."""
    return np.array([role_weights.get(stat, 0.0) for stat in stat_cols])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_roster_stat_stats(growth_rates_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Mean and standard deviation of each growth-rate stat across the roster.
    Used to standardize (z-score) a character's growth rates before role
    detection - see detect_natural_role for why this matters.
    """
    means = growth_rates_df[STAT_COLS].mean()
    stds = growth_rates_df[STAT_COLS].std().replace(0, 1)  # guard divide-by-zero
    return means, stds


def detect_natural_role(
    growth_row: pd.Series,
    roster_means: pd.Series | None = None,
    roster_stds: pd.Series | None = None,
) -> tuple[str, float]:
    """
    Given a character's growth-rate row, find which role archetype their
    growth rates most resemble.

    Growth rates are standardized (z-scored) against the roster before
    comparing, rather than compared as raw percentages. This matters because
    raw-percentage cosine similarity compares vector SHAPE across all 9
    stats at once, which under-weights a character's genuinely distinctive
    stat (e.g. unusually high Magic growth) if it happens to sit alongside
    moderately-high values in other stats shared with a different role's
    profile (e.g. Resistance, which Support also cares about) - two
    characters caught by this in testing, Hubert and Linhardt, were
    classified as Support instead of Magic Attacker before this fix, despite
    both being mage archetypes. Standardizing first measures "how unusual is
    this stat for this character, relative to everyone else" instead of raw
    magnitude, which better isolates each character's actual specialty.

    If roster_means/roster_stds aren't provided, falls back to raw growth
    rates (no standardization) - useful for one-off testing.
    """
    if roster_means is not None and roster_stds is not None:
        growth_vector = ((growth_row[STAT_COLS] - roster_means) / roster_stds).to_numpy(dtype=float)
    else:
        growth_vector = growth_row[STAT_COLS].to_numpy(dtype=float)

    best_role, best_score = None, -1.0
    for role_name, weights in ROLE_PROFILES.items():
        role_vector = role_to_vector(weights, STAT_COLS)
        score = cosine_similarity(growth_vector, role_vector)
        if score > best_score:
            best_role, best_score = role_name, score

    return best_role, best_score


def score_class_for_role(boost_row: pd.Series, role_weights: dict) -> float:
    """Dot product of a class's stat boosts with a role's weight profile - higher is a better fit."""
    stat_cols_present = [c for c in STAT_COLS if c in boost_row.index] + (
        ["Mov"] if "Mov" in boost_row.index else []
    )
    score = 0.0
    for stat in stat_cols_present:
        score += boost_row[stat] * role_weights.get(stat, 0.0)
    return score


def recommend_path(
    stat_boosts_df: pd.DataFrame,
    role_name: str,
    tiers: list[str] = TIER_ORDER,
    target_level: int | None = None,
    character_name: str | None = None,
    eligibility_lookup: dict | None = None,
    character_gender: str | None = None,
) -> list[dict]:
    """
    For a target role, pick the best-fitting class at each tier in order.
    Returns a list of {"tier": ..., "class": ..., "score": ...} dicts.

    If target_level is given, the path is truncated to only tiers reachable
    at that level (see TIER_LEVEL_REQUIREMENTS / reachable_tiers) - e.g. a
    level-15 target stops after Intermediate, since Advanced requires level
    20. If target_level is None (the default), no level-gating is applied
    and the full `tiers` list is used as-is, for callers that don't have a
    target level in mind (e.g. comparing class fit independent of pacing).

    If character_name and eligibility_lookup are both given, classes that
    character isn't eligible for (character-locked to someone else, or
    gender-locked against character_gender) are excluded from each tier's
    candidate pool before scoring - e.g. a male character won't have
    "Pegasus Knight" considered even if it would otherwise score best, and
    "Lord" is only considered for the three house leaders. If either is
    omitted, no eligibility filtering happens (backward compatible with
    callers that don't have this data on hand).
    """
    role_weights = ROLE_PROFILES[role_name]
    path = []

    reachable = reachable_tiers(target_level, tiers) if target_level is not None else tiers
    check_eligibility = character_name is not None and eligibility_lookup is not None

    for tier in reachable:
        tier_classes = stat_boosts_df[stat_boosts_df["tier"] == tier]

        # Exclude story/enemy-specific variant rows (e.g. "Lord (Judith)",
        # "Fortress Knight (Chapter 5 Gilbert)") - these are one-off NPC/boss
        # instances Serenes Forest documents alongside real playable classes,
        # not classes a player can actually select. They tend to have unusually
        # high stats and can otherwise out-score every legitimate option
        # regardless of role, which would make them get recommended for
        # basically everyone - a real bug caught by inspecting team-building
        # output, where the same variant kept appearing across every role.
        tier_classes = tier_classes[~tier_classes["name"].str.contains(r"\(", regex=True)]

        if check_eligibility:
            tier_classes = tier_classes[tier_classes["name"].apply(
                lambda name: is_class_eligible(character_name, name, eligibility_lookup, character_gender)
            )]

        if tier_classes.empty:
            continue

        scores = tier_classes.apply(lambda row: score_class_for_role(row, role_weights), axis=1)
        best_idx = scores.idxmax()
        best_row = tier_classes.loc[best_idx]

        path.append({
            "tier": tier,
            "class": best_row["name"],
            "score": round(float(scores.loc[best_idx]), 2),
        })

    return path


def eligible_unique_classes(
    character_name: str,
    stat_boosts_df: pd.DataFrame,
    role_weights: dict,
    eligibility_lookup: dict,
    character_gender: str | None = None,
) -> list[dict]:
    """
    List Unique-tier classes character_name is eligible for, scored against
    role_weights and sorted best-fit-first. Returns a list of {"class":
    ..., "score": ...} dicts (empty if none are eligible).

    This is informational, surfaced alongside recommend_path's output
    rather than spliced into it: Unique classes (Emperor, Enlightened One,
    etc.) don't map onto the Beginner->Master ladder the way regular
    classes do - they're unlocked by character-specific story beats, not by
    a uniform level + seal requirement (see TIER_LEVEL_REQUIREMENTS
    docstring), so we don't have a principled tier slot to insert them
    into. This function answers "what else can this character access" as a
    separate callout instead of pretending we know exactly when.
    """
    unique_classes = stat_boosts_df[stat_boosts_df["tier"] == "Unique"]
    unique_classes = unique_classes[~unique_classes["name"].str.contains(r"\(", regex=True)]

    results = []
    for _, row in unique_classes.iterrows():
        if not is_class_eligible(character_name, row["name"], eligibility_lookup, character_gender):
            continue
        results.append({
            "class": row["name"],
            "score": round(score_class_for_role(row, role_weights), 2),
        })

    return sorted(results, key=lambda r: r["score"], reverse=True)


def expected_stats_at_level(
    base_row: pd.Series,
    growth_row: pd.Series,
    final_class_boost_row: pd.Series,
    target_level: int = 20,
) -> dict:
    """
    Estimate a character's expected stats at target_level in a given final
    class: base stats + expected level-up gains (growth_rate% per level,
    used as an expected value - e.g. a 45% growth rate contributes 0.45
    expected stat points per level, not a simulated coin flip) + the final
    class's stat boost.

    This is an expected-value calculation, not a single simulated
    playthrough - it answers "on average, how strong would this character
    be," which is what matters for comparing class paths against each
    other. A Monte Carlo version (simulating many individual playthroughs
    to show the range of outcomes, not just the average) is a natural
    upgrade for a later stage.
    """
    levels_gained = target_level - 1
    result = {}
    for stat in STAT_COLS:
        base = base_row[stat]
        expected_growth = (growth_row[stat] / 100) * levels_gained
        boost = final_class_boost_row[stat] if stat in final_class_boost_row.index else 0
        result[stat] = round(base + expected_growth + boost, 1)
    return result


def recommend_for_character(
    character_name: str,
    base_stats_df: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
    stat_boosts_df: pd.DataFrame,
    role_name: str | None = None,
    target_level: int = 20,
    eligibility_df: pd.DataFrame | None = None,
    character_gender_df: pd.DataFrame | None = None,
) -> dict:
    """
    Full recommendation for one character: auto-detects a role if none is
    given, builds a class path toward it, and estimates expected stats at
    target_level in the recommended final class.

    If eligibility_df (data/class_eligibility.csv) is given, both the
    recommended path and the auxiliary "eligible_unique_classes" list
    respect character/gender restrictions (see recommend_path and
    eligible_unique_classes). character_gender_df (data/character_gender.csv)
    supplies the character's gender for that check; if omitted, gender-locked
    classes are treated as unrestricted rather than blocked (see
    is_class_eligible). Both are optional and default to None, which
    disables eligibility filtering entirely - existing callers that don't
    pass them keep getting today's behavior.
    """
    base_row = base_stats_df[base_stats_df["name"] == character_name].iloc[0]
    growth_row = growth_rates_df[growth_rates_df["name"] == character_name].iloc[0]

    roster_means, roster_stds = compute_roster_stat_stats(growth_rates_df)
    detected_role, detection_score = detect_natural_role(growth_row, roster_means, roster_stds)
    used_role = role_name or detected_role

    eligibility_lookup = load_eligibility_lookup(eligibility_df) if eligibility_df is not None else None
    character_gender = None
    if character_gender_df is not None:
        gender_row = character_gender_df[character_gender_df["name"] == character_name]
        if not gender_row.empty:
            character_gender = gender_row.iloc[0]["gender"]

    path = recommend_path(
        stat_boosts_df, used_role, target_level=target_level,
        character_name=character_name,
        eligibility_lookup=eligibility_lookup,
        character_gender=character_gender,
    )
    final_class_name = path[-1]["class"] if path else None

    final_stats = None
    if final_class_name:
        final_boost_row = stat_boosts_df[stat_boosts_df["name"] == final_class_name].iloc[0]
        final_stats = expected_stats_at_level(base_row, growth_row, final_boost_row, target_level)

    unique_options = None
    if eligibility_lookup is not None:
        unique_options = eligible_unique_classes(
            character_name, stat_boosts_df, ROLE_PROFILES[used_role],
            eligibility_lookup, character_gender,
        )

    return {
        "character": character_name,
        "role_used": used_role,
        "auto_detected_role": detected_role,
        "auto_detection_score": round(detection_score, 3),
        "path": path,
        "expected_stats_at_level": target_level,
        "expected_final_stats": final_stats,
        "eligible_unique_classes": unique_options,
    }


def main():
    base_stats_df = pd.read_csv(DATA_DIR / "character_base_stats.csv")
    growth_rates_df = pd.read_csv(DATA_DIR / "character_growth_rates.csv")
    stat_boosts_df = pd.read_csv(DATA_DIR / "class_stat_boosts.csv")
    eligibility_df = pd.read_csv(DATA_DIR / "class_eligibility.csv")
    character_gender_df = pd.read_csv(DATA_DIR / "character_gender.csv")

    import argparse
    parser = argparse.ArgumentParser(description="Recommend a Three Houses class path for a character.")
    parser.add_argument("character", type=str, help="Character name, e.g. 'Bernadetta'")
    parser.add_argument("--role", type=str, default=None,
                         choices=list(ROLE_PROFILES.keys()),
                         help="Target role (omit to auto-detect from growth rates)")
    parser.add_argument("--level", type=int, default=20, help="Target level for stat projection")
    args = parser.parse_args()

    result = recommend_for_character(
        args.character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=args.role, target_level=args.level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
    )

    print(f"\n{result['character']}")
    print(f"Auto-detected natural role: {result['auto_detected_role']} (similarity {result['auto_detection_score']})")
    print(f"Recommended path toward: {result['role_used']}")
    if not result["path"]:
        print(f"  (no tier reachable at level {args.level} - Beginner requires level "
              f"{TIER_LEVEL_REQUIREMENTS['Beginner']})")
    else:
        for step in result["path"]:
            print(f"  {step['tier']:>13}: {step['class']} (fit score {step['score']})")
        print(f"Expected stats at level {result['expected_stats_at_level']} as {result['path'][-1]['class']}:")
        print(f"  {result['expected_final_stats']}")

    if result["eligible_unique_classes"]:
        print("Also eligible for these Unique classes:")
        for option in result["eligible_unique_classes"]:
            print(f"  {option['class']} (fit score {option['score']})")


if __name__ == "__main__":
    main()