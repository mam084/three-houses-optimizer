"""
optimizer.py

v1 class-path recommender for Fire Emblem: Three Houses.

Given a character, recommends a class path (one class per tier: Beginner ->
Intermediate -> Advanced -> Master) either toward a specific target role
(e.g. "Tank") or auto-detected from the character's own growth rates (i.e.
"what is this character naturally good at?").

Known simplifications (v1):
  - Only Beginner/Intermediate/Advanced/Master tiers are considered. Unique
    classes (Emperor, Barbarossa, etc.) are character-locked to specific
    lords - we don't have that eligibility data yet, so recommending them
    generically would be wrong. NPC/Enemy classes aren't player-selectable
    at all. DLC Exclusive classes are excluded from v1 for the same
    "not always available" reasoning - a clean addition later.
  - Class unlocks are modeled by tier order only, not the game's real
    level + skill-proficiency requirements (we don't have skill requirement
    data yet). Tier order is a reasonable approximation of the real
    progression, just optimistic about exact timing.
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

# Role archetypes as stat-weight profiles. Weights are relative importance,
# not required to sum to 1 - only relative magnitude matters for scoring.
ROLE_PROFILES = {
    "Physical Attacker": {"Str": 1.0, "Spd": 0.5},
    "Magic Attacker": {"Mag": 1.0, "Dex": 0.3},
    "Tank": {"HP": 1.0, "Def": 1.0},
    "Support": {"Res": 0.7, "Cha": 0.7, "Mag": 0.3},
    "Flier/Mobility": {"Spd": 0.7, "Dex": 0.3, "Mov": 1.0},
}


def role_to_vector(role_weights: dict, stat_cols: list[str]) -> np.ndarray:
    """Turn a role's {stat: weight} dict into a vector aligned to stat_cols (0 for unlisted stats)."""
    return np.array([role_weights.get(stat, 0.0) for stat in stat_cols])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def detect_natural_role(growth_row: pd.Series) -> tuple[str, float]:
    """
    Given a character's growth-rate row, find which role archetype their
    growth rates most resemble (cosine similarity between the character's
    growth vector and each role's weight vector). Returns (role_name, score).
    """
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
) -> list[dict]:
    """
    For a target role, pick the best-fitting class at each tier in order.
    Returns a list of {"tier": ..., "class": ..., "score": ...} dicts.
    """
    role_weights = ROLE_PROFILES[role_name]
    path = []

    for tier in tiers:
        tier_classes = stat_boosts_df[stat_boosts_df["tier"] == tier]
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
) -> dict:
    """
    Full recommendation for one character: auto-detects a role if none is
    given, builds a class path toward it, and estimates expected stats at
    target_level in the recommended final class.
    """
    base_row = base_stats_df[base_stats_df["name"] == character_name].iloc[0]
    growth_row = growth_rates_df[growth_rates_df["name"] == character_name].iloc[0]

    detected_role, detection_score = detect_natural_role(growth_row)
    used_role = role_name or detected_role

    path = recommend_path(stat_boosts_df, used_role)
    final_class_name = path[-1]["class"] if path else None

    final_stats = None
    if final_class_name:
        final_boost_row = stat_boosts_df[stat_boosts_df["name"] == final_class_name].iloc[0]
        final_stats = expected_stats_at_level(base_row, growth_row, final_boost_row, target_level)

    return {
        "character": character_name,
        "role_used": used_role,
        "auto_detected_role": detected_role,
        "auto_detection_score": round(detection_score, 3),
        "path": path,
        "expected_stats_at_level": target_level,
        "expected_final_stats": final_stats,
    }


def main():
    base_stats_df = pd.read_csv(DATA_DIR / "character_base_stats.csv")
    growth_rates_df = pd.read_csv(DATA_DIR / "character_growth_rates.csv")
    stat_boosts_df = pd.read_csv(DATA_DIR / "class_stat_boosts.csv")

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
    )

    print(f"\n{result['character']}")
    print(f"Auto-detected natural role: {result['auto_detected_role']} (similarity {result['auto_detection_score']})")
    print(f"Recommended path toward: {result['role_used']}")
    for step in result["path"]:
        print(f"  {step['tier']:>13}: {step['class']} (fit score {step['score']})")
    print(f"Expected stats at level {result['expected_stats_at_level']} as {result['path'][-1]['class']}:")
    print(f"  {result['expected_final_stats']}")


if __name__ == "__main__":
    main()