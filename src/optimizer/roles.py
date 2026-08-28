"""
optimizer/roles.py

Role archetypes: turning a character's growth rates into a role fit
(auto-detection, score_all_roles) and a class's stat boosts/growth
modifiers into a role fit (score_class_for_role, score_growth_for_role),
plus the small shared vector-math helpers both directions use.
"""
import numpy as np
import pandas as pd

from .constants import (
    MAGIC_WEAPON_TYPES,
    NATURAL_ROLE_AFFINITY_WEIGHT,
    ROLE_NATURAL_WEAPON_CATEGORY,
    ROLE_PROFILES,
    STAT_COLS,
)

# Support used to also weight Cha (Charisma) at the same strength as Res, on
# the theory that "support" characters tend to be likeable/persuasive. In
# practice this conflated two unrelated things: Res/Faith-magic is what
# actually makes someone a good healer, while Cha governs battalion gambits
# and support-conversation unlocks - a "leadership" stat, not a "healing
# aptitude" one. Every one of Three Houses's three house leaders (Edelgard,
# Dimitri, Claude) has an unusually high Cha growth rate simply by virtue of
# being a lord, which - standardized against the roster (see
# detect_natural_role) - was enough to outscore their own much more
# on-theme Str/combat lean and get them auto-detected as "Support." That's
# the reported "why is Edelgard a Gremory??" bug: her natural-role
# auto-detection came back Support, so her recommended path ran Monk ->
# Priest -> Bishop -> Gremory, a healer/mage lineage with nothing to do
# with the axe-wielding Emperor she actually becomes. Dropping Cha from
# Support's weighting (verified against the full roster - Edelgard now
# correctly detects as Physical Attacker, and known healer archetypes like
# Mercedes/Flayn/Marianne/Linhardt still correctly detect as Support) fixes
# this without touching anything else; Cha still matters for eligibility
# (Dancer, the Lord line) and is still part of the projected stat block,
# it just no longer drives natural-role detection.


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



def natural_role_affinity_bonus(role_name: str, weapon_types: set | None) -> float:
    """
    NATURAL_ROLE_AFFINITY_WEIGHT if role_name's own weapon family
    (ROLE_NATURAL_WEAPON_CATEGORY) overlaps weapon_types (a character's
    starting weapon proficiency and/or Hero's Relic weapon type(s), unioned
    by the caller), else 0.0. A role absent from ROLE_NATURAL_WEAPON_CATEGORY
    (Tank, Speed/Precision) or an empty/missing weapon_types always returns
    0.0 - see both constants' docstrings for why this is deliberately
    narrow rather than a blanket "equipment always matters" rule.
    """
    category = ROLE_NATURAL_WEAPON_CATEGORY.get(role_name)
    if category is None or not weapon_types:
        return 0.0
    is_magic = bool(weapon_types & MAGIC_WEAPON_TYPES)
    is_physical = bool(weapon_types - MAGIC_WEAPON_TYPES)
    if category == "magic":
        return NATURAL_ROLE_AFFINITY_WEIGHT if is_magic else 0.0
    return NATURAL_ROLE_AFFINITY_WEIGHT if is_physical else 0.0



def score_all_roles(
    growth_row: pd.Series,
    roster_means: pd.Series | None = None,
    roster_stds: pd.Series | None = None,
    character_proficiency: set | None = None,
    character_relic_weapon_types: set | None = None,
) -> dict[str, float]:
    """
    Given a character's growth-rate row, score EVERY role archetype (not
    just the winner - see detect_natural_role, which is this function plus
    an argmax) via standardized-growth-rate cosine similarity, the same
    calculation for each of ROLE_PROFILES's entries.

    This is what lets a caller show "how well does this character fit
    every role," not just their single auto-detected best fit - "you can
    only see a character's best-fit role, not how well the tool considers
    them to fit the others" - see app.py's render_role_fit_chart.

    character_proficiency/character_relic_weapon_types (unioned together),
    if given, apply a small natural_role_affinity_bonus per role on top of
    the growth-rate similarity - see that function and
    NATURAL_ROLE_AFFINITY_WEIGHT for why this is bounded and only some
    roles are affected.
    """
    if roster_means is not None and roster_stds is not None:
        growth_vector = ((growth_row[STAT_COLS] - roster_means) / roster_stds).to_numpy(dtype=float)
    else:
        growth_vector = growth_row[STAT_COLS].to_numpy(dtype=float)

    combined_weapon_types = set(character_proficiency or set()) | set(character_relic_weapon_types or set())

    scores = {}
    for role_name, weights in ROLE_PROFILES.items():
        role_vector = role_to_vector(weights, STAT_COLS)
        score = cosine_similarity(growth_vector, role_vector)
        score += natural_role_affinity_bonus(role_name, combined_weapon_types)
        scores[role_name] = score
    return scores



def detect_natural_role(
    growth_row: pd.Series,
    roster_means: pd.Series | None = None,
    roster_stds: pd.Series | None = None,
    character_proficiency: set | None = None,
    character_relic_weapon_types: set | None = None,
) -> tuple[str, float]:
    """
    Given a character's growth-rate row, find which role archetype their
    growth rates most resemble (score_all_roles, then argmax).

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

    character_proficiency/character_relic_weapon_types: see score_all_roles
    - a small, bounded nudge toward whichever role matches the character's
    own starting weapon talent or Hero's Relic, so a character's own gear
    affinity gets SOME voice in which role is auto-detected, not just in
    which class gets picked once a role is already chosen.
    """
    scores = score_all_roles(
        growth_row, roster_means, roster_stds, character_proficiency, character_relic_weapon_types
    )
    best_role = max(scores, key=scores.get)
    return best_role, scores[best_role]



def score_class_for_role(boost_row: pd.Series, role_weights: dict) -> float:
    """Dot product of a class's stat boosts with a role's weight profile - higher is a better fit."""
    stat_cols_present = [c for c in STAT_COLS if c in boost_row.index] + (
        ["Mov"] if "Mov" in boost_row.index else []
    )
    score = 0.0
    for stat in stat_cols_present:
        score += boost_row[stat] * role_weights.get(stat, 0.0)
    return score



def score_growth_for_role(growth_mod_row: dict, role_weights: dict) -> float:
    """
    Dot product of a class's growth-RATE modifiers (percentage points, see
    load_class_growth_lookup) with a role's weight profile - the
    growth-rate analogue of score_class_for_role, letting a class's
    compounding stat growth, not just its flat one-time boost, factor
    into which class actually gets recommended. Scaled down by
    GROWTH_RATE_SCORE_WEIGHT wherever it's used - see that constant.
    """
    score = 0.0
    for stat, weight in role_weights.items():
        if stat in STAT_COLS:
            score += growth_mod_row.get(stat, 0) * weight
    return score



def role_relevant_stat_deltas(boost_row: pd.Series, role_weights: dict) -> list[tuple[str, float]]:
    """
    A class's own flat stat boosts (data/class_stat_boosts.csv), narrowed
    to just the stat(s) role_weights actually weighs (weight > 0), in
    weight-descending order - e.g. for Physical Attacker ({"Str": 1.0,
    "Spd": 0.5}) and a class boosting Str +3/Spd +1/Def +2, returns
    [("Str", 3), ("Spd", 1)] - Def is dropped since Physical Attacker
    doesn't weigh it at all.

    Used so switching to a different class in "mix and match" shows what
    that switch actually buys FOR THE ROLE being built toward ("+3 Str, +1
    Spd relevant to Physical Attacker"), rather than only a single fit
    score number the reader has to trust without seeing the stats behind
    it - "when changing the class to a different recommended one, we
    should display the bonus to stats that are relevant to the currently
    selected role."
    """
    stats_by_weight = sorted(
        (stat for stat, weight in role_weights.items() if weight > 0),
        key=lambda s: role_weights[s], reverse=True,
    )
    return [(stat, float(boost_row[stat])) for stat in stats_by_weight if stat in boost_row.index]



def primary_stats_for_role(role_weights: dict) -> list[str]:
    """The stat(s) tied for the highest weight in a role profile - e.g. ["HP", "Def"] for Tank."""
    top_weight = max(role_weights.values())
    return [stat for stat, weight in role_weights.items() if weight == top_weight]
