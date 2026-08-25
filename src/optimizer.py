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
    "Support": {"Res": 1.0},
    "Speed/Precision": {"Spd": 0.7, "Dex": 0.3, "Mov": 1.0},
}
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


def primary_stats_for_role(role_weights: dict) -> list[str]:
    """The stat(s) tied for the highest weight in a role profile - e.g. ["HP", "Def"] for Tank."""
    top_weight = max(role_weights.values())
    return [stat for stat, weight in role_weights.items() if weight == top_weight]


def load_weapon_requirements_lookup(weapon_req_df: pd.DataFrame | None) -> dict:
    """
    Index data/class_weapon_requirements.csv by class_name. Each entry is
    {"tier", "weapon_category" ("physical"/"magic"/"hybrid" - whether the
    class's certification needs a physical weapon skill, a magic skill, or
    both), "requirement_type" ("AND"/"OR" - whether every listed
    requirement is needed or just one of them), "requirements" (a list of
    (skill, rank) tuples, e.g. [("Sword", "B"), ("Axe", "C")])}.

    Hand-curated from Serenes Forest's certification-exam requirements
    (https://serenesforest.net/three-houses/) - the same precedent as
    class_eligibility.csv and character_gender.csv: a small, stable table
    for information the stat-boost data alone doesn't carry. A class
    absent from this CSV (Unique-tier classes, NPC/Enemy classes) has no
    certification-exam requirement to model - they unlock via story
    progression instead - and simply won't appear in the lookup.
    """
    if weapon_req_df is None:
        return {}
    lookup = {}
    for _, row in weapon_req_df.iterrows():
        requirements = [tuple(part.split(":")) for part in row["requirements"].split("|")]
        lookup[row["class_name"]] = {
            "tier": row["tier"],
            "weapon_category": row["weapon_category"],
            "requirement_type": row["requirement_type"],
            "requirements": requirements,
        }
    return lookup


def format_requirement(class_name: str, weapon_req_lookup: dict) -> str | None:
    """
    Human-readable certification requirement for a class, e.g. "Sword B or
    Axe C" (an OR - either satisfies it) or "Axe C and Heavy Armour D" (an
    AND - both are needed). Returns None if the class has no requirement
    data (see load_weapon_requirements_lookup).
    """
    info = weapon_req_lookup.get(class_name) if weapon_req_lookup else None
    if info is None:
        return None
    joiner = " or " if info["requirement_type"] == "OR" else " and "
    return joiner.join(f"{skill} {rank}" for skill, rank in info["requirements"])


def load_character_proficiency_lookup(character_weapon_talent_df: pd.DataFrame | None) -> dict:
    """
    Index data/character_weapon_talent.csv by character name -> the set of
    skill types that character starts with at their own personal-best
    starting rank (per Serenes Forest's skill-levels page - each
    character's Level 1 skill ranks are listed there; "natural
    proficiency" here means whichever skill(s) that character's own
    starting ranks are highest in, not a designer-labeled "Talent"). E.g.
    Felix's listed starting ranks are Sword D / Bow E+ / Brawling E+, so
    his natural proficiency is {"Sword"}. Several characters are tied
    across more than one skill (e.g. Sylvain: Lance/Axe/Riding all D) -
    every tied skill is included, not just one.
    """
    if character_weapon_talent_df is None:
        return {}
    lookup = {}
    for _, row in character_weapon_talent_df.iterrows():
        skills = row.get("natural_proficiencies")
        lookup[row["name"]] = set(skills.split("|")) if isinstance(skills, str) and skills else set()
    return lookup


def apply_weapon_affinity_fallback(
    tier_classes: pd.DataFrame,
    role_name: str,
    role_weights: dict,
    weapon_req_lookup: dict,
    character_proficiency: set | None = None,
) -> pd.DataFrame:
    """
    For the two Attacker roles specifically, if this tier's stat-boost data
    is completely uninformative about the role's primary stat (every
    candidate is at 0 - the exact situation restrict_to_primary_relevant
    can't resolve on its own, since even the "right" answer scores 0),
    fall back to each class's real certification weapon/magic requirement
    (data/class_weapon_requirements.csv - see load_weapon_requirements_lookup)
    instead of leaving the pick to whichever class happens to have an
    unrelated secondary-stat boost. This generalizes what used to be a
    Beginner-only hardcoded physical/magic table (BEGINNER_CLASS_AFFINITY,
    added to fix the original "why Soldier for mages?" bug - see
    restrict_to_primary_relevant's docstring) into real requirement data
    that applies at any tier; it's still a no-op almost everywhere else,
    since real Mag/Str stat boosts already differentiate classes fine past
    Beginner tier - Beginner is just the one tier sparse enough (a single
    +1 each, to four different stats) that this fallback usually has
    something to do.

    When more than one class matches the role's weapon/magic affinity
    (e.g. two magic-leaning classes tied on stats), further narrows to
    whichever also overlaps the character's own starting weapon
    proficiency (character_proficiency - see
    load_character_proficiency_lookup), when that narrows the field to at
    least one match; otherwise leaves every affinity match in place as an
    honest tie.
    """
    if role_name not in ("Physical Attacker", "Magic Attacker") or not weapon_req_lookup:
        return tier_classes

    primary_stats = [s for s in primary_stats_for_role(role_weights) if s in tier_classes.columns]
    stat_data_is_informative = bool(primary_stats) and (tier_classes[primary_stats] > 0).any(axis=1).any()
    if stat_data_is_informative:
        return tier_classes

    affinity_target = "magic" if role_name == "Magic Attacker" else "physical"

    def matches_affinity(name: str) -> bool:
        info = weapon_req_lookup.get(name)
        return info is not None and info["weapon_category"] in (affinity_target, "hybrid")

    affinity_matches = tier_classes[tier_classes["name"].map(matches_affinity)]
    if affinity_matches.empty:
        return tier_classes  # no known-affinity class available - fall back to unrestricted (honest tie)

    if character_proficiency and len(affinity_matches) > 1:
        def matches_own_proficiency(name: str) -> bool:
            required_skills = {skill for skill, _ in weapon_req_lookup[name]["requirements"]}
            return bool(required_skills & character_proficiency)

        proficiency_matches = affinity_matches[affinity_matches["name"].map(matches_own_proficiency)]
        if not proficiency_matches.empty:
            return proficiency_matches

    return affinity_matches


def restrict_to_primary_relevant(tier_classes: pd.DataFrame, role_weights: dict) -> pd.DataFrame:
    """
    Narrow a tier's candidate classes to those that contribute at least
    something to the role's most important stat(s), when at least one
    candidate does.

    Why this exists: Beginner-tier stat boosts are tiny (a single +1) and,
    per Serenes Forest, none of the four Beginner classes (Myrmidon,
    Soldier, Fighter, Monk) boosts Magic at all - only Monk has any magic
    proficiency (it's the lead-in to Mage/Priest), and its one boost is to
    Resistance. Without this guard, scoring "Magic Attacker" fit as a flat
    dot-product let Soldier's incidental +1 Dex (Magic Attacker's minor
    secondary weight) numerically outscore Monk's irrelevant +1 Res, so the
    tool recommended Soldier - a lance class with no magic proficiency at
    all - as the "best" Beginner step for a mage. (This was the reported
    "why Soldier for mages?" bug.)

    The fix: a class with zero contribution to the role's primary stat(s)
    should never out-rank a class that has some, purely on the strength of
    secondary stats. So if any candidate in the tier has a nonzero primary
    stat, candidates with zero across all primary stats are dropped before
    scoring; the remaining candidates are still ranked by the full weighted
    score (secondary stats still refine ordering among relevant options).
    If NO candidate has any primary-stat relevance (a genuine tie, e.g.
    Tank at Beginner tier, where nothing boosts HP or Def), no restriction
    is applied and the tier is scored as before - that's an honest gap in
    the data, not something a tie-break should paper over.
    """
    primary_stats = [s for s in primary_stats_for_role(role_weights) if s in tier_classes.columns]
    if not primary_stats:
        return tier_classes

    has_primary_relevance = (tier_classes[primary_stats] > 0).any(axis=1)
    if has_primary_relevance.any():
        return tier_classes[has_primary_relevance]
    return tier_classes


def explain_pick(
    boost_row: pd.Series,
    role_weights: dict,
    role_name: str | None = None,
    weapon_req_lookup: dict | None = None,
    character_proficiency: set | None = None,
) -> str:
    """
    One-sentence, human-readable reason a class was picked for a role, so
    the recommendation isn't just a bare class name and a score.

    Names the 1-2 stat boosts that drove the score, in plain terms (e.g.
    "boosts Mag +3 and Res +1"). If none of the role's weighted stats are
    boosted at all, checks whether the pick came from the weapon-affinity
    fallback (see apply_weapon_affinity_fallback) before falling back to a
    generic "no good option" explanation - e.g. Monk winning Magic Attacker
    at Beginner tier despite a 0 stat score, because it's the only
    magic-proficient class available, not because a stat says so. When the
    fallback fired AND the class's requirement overlaps the character's own
    starting weapon proficiency (character_proficiency - see
    load_character_proficiency_lookup), that gets named too.
    """
    contributions = [
        (stat, boost_row[stat], weight)
        for stat, weight in role_weights.items()
        if weight > 0 and stat in boost_row.index and boost_row[stat] > 0
    ]
    if contributions:
        contributions.sort(key=lambda c: c[1] * c[2], reverse=True)
        parts = [f"+{int(boost) if float(boost).is_integer() else boost} {stat}" for stat, boost, _ in contributions[:2]]
        return f"Best fit at this tier - boosts {' and '.join(parts)}, matching this role's priorities."

    class_name = boost_row.get("name")
    info = weapon_req_lookup.get(class_name) if weapon_req_lookup else None
    wants = "magic" if role_name == "Magic Attacker" else "physical" if role_name == "Physical Attacker" else None
    if info is not None and wants is not None and info["weapon_category"] in (wants, "hybrid"):
        requirement = format_requirement(class_name, weapon_req_lookup)
        required_skills = {skill for skill, _ in info["requirements"]}
        proficiency_note = ""
        if character_proficiency and (required_skills & character_proficiency):
            matched = ", ".join(sorted(required_skills & character_proficiency))
            proficiency_note = f" - and it's this character's own starting strength ({matched})"
        return (
            f"No class at this tier boosts a stat this role cares about, but this is a "
            f"{info['weapon_category']}-weapon class (requires {requirement}){proficiency_note}, "
            f"a closer thematic fit than the alternatives."
        )

    return "No class at this tier boosts a stat this role cares about - this is the least-irrelevant option available."


def recommend_path(
    stat_boosts_df: pd.DataFrame,
    role_name: str,
    tiers: list[str] = TIER_ORDER,
    target_level: int | None = None,
    character_name: str | None = None,
    eligibility_lookup: dict | None = None,
    character_gender: str | None = None,
    weapon_req_lookup: dict | None = None,
    character_proficiency: set | None = None,
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

        tier_classes = apply_weapon_affinity_fallback(
            tier_classes, role_name, role_weights, weapon_req_lookup or {}, character_proficiency
        )
        tier_classes = restrict_to_primary_relevant(tier_classes, role_weights)

        scores = tier_classes.apply(lambda row: score_class_for_role(row, role_weights), axis=1)
        best_idx = scores.idxmax()
        best_row = tier_classes.loc[best_idx]
        best_class = best_row["name"]

        path.append({
            "tier": tier,
            "class": best_class,
            "score": round(float(scores.loc[best_idx]), 2),
            "why": explain_pick(best_row, role_weights, role_name, weapon_req_lookup, character_proficiency),
            "requirement": format_requirement(best_class, weapon_req_lookup) if weapon_req_lookup else None,
        })

    return path


def list_eligible_classes_at_tier(
    tier: str,
    stat_boosts_df: pd.DataFrame,
    character_name: str | None = None,
    eligibility_lookup: dict | None = None,
    character_gender: str | None = None,
) -> list[str]:
    """
    Every player-selectable class name at `tier` that character_name is
    eligible for (same eligibility rules as recommend_path - see
    is_class_eligible), excluding story/enemy-specific variant rows (e.g.
    "Lord (Judith)"). If character_name/eligibility_lookup are omitted, no
    eligibility filtering happens and every class at that tier is
    returned.

    Used to let a user override the recommended pick at a given tier with
    any other class they were actually eligible for - the "mix and match"
    path - rather than only ever seeing the single top-scoring choice.
    """
    tier_classes = stat_boosts_df[stat_boosts_df["tier"] == tier]
    tier_classes = tier_classes[~tier_classes["name"].str.contains(r"\(", regex=True)]

    if character_name is not None and eligibility_lookup is not None:
        tier_classes = tier_classes[tier_classes["name"].apply(
            lambda name: is_class_eligible(character_name, name, eligibility_lookup, character_gender)
        )]

    return sorted(tier_classes["name"].tolist())


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
    unique_classes = restrict_to_primary_relevant(unique_classes, role_weights)

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
    target_level: int = 30,
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


def stats_for_class_at_level(
    character_name: str,
    class_name: str,
    base_stats_df: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
    stat_boosts_df: pd.DataFrame,
    target_level: int = 30,
) -> dict | None:
    """
    Expected stats for character_name at target_level, if their current
    class were class_name specifically - the same expected-value
    calculation recommend_for_character uses for its recommended final
    class (see expected_stats_at_level), just callable for any class the
    user picks by hand. Powers the "mix and match" path override in the
    UI: swap in a different class at whichever tier you'd actually end on,
    and see the resulting projection, not just the tool's own top pick.
    Returns None if class_name isn't in stat_boosts_df at all.
    """
    boost_rows = stat_boosts_df[stat_boosts_df["name"] == class_name]
    if boost_rows.empty:
        return None
    base_row = base_stats_df[base_stats_df["name"] == character_name].iloc[0]
    growth_row = growth_rates_df[growth_rates_df["name"] == character_name].iloc[0]
    return expected_stats_at_level(base_row, growth_row, boost_rows.iloc[0], target_level)


def recommend_for_character(
    character_name: str,
    base_stats_df: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
    stat_boosts_df: pd.DataFrame,
    role_name: str | None = None,
    target_level: int = 30,
    eligibility_df: pd.DataFrame | None = None,
    character_gender_df: pd.DataFrame | None = None,
    weapon_req_df: pd.DataFrame | None = None,
    character_weapon_talent_df: pd.DataFrame | None = None,
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
    is_class_eligible). weapon_req_df (data/class_weapon_requirements.csv)
    and character_weapon_talent_df (data/character_weapon_talent.csv), if
    given, attach a certification-requirement string to each path step and
    feed the character's own starting weapon proficiency into the
    weapon-affinity fallback (see apply_weapon_affinity_fallback). All four
    are optional and default to None, which disables the corresponding
    feature entirely - existing callers that don't pass them keep getting
    today's behavior.
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

    weapon_req_lookup = load_weapon_requirements_lookup(weapon_req_df) if weapon_req_df is not None else None
    character_proficiency_lookup = load_character_proficiency_lookup(character_weapon_talent_df) \
        if character_weapon_talent_df is not None else {}
    character_proficiency = character_proficiency_lookup.get(character_name)

    path = recommend_path(
        stat_boosts_df, used_role, target_level=target_level,
        character_name=character_name,
        eligibility_lookup=eligibility_lookup,
        character_gender=character_gender,
        weapon_req_lookup=weapon_req_lookup,
        character_proficiency=character_proficiency,
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
    weapon_req_df = pd.read_csv(DATA_DIR / "class_weapon_requirements.csv")
    character_weapon_talent_df = pd.read_csv(DATA_DIR / "character_weapon_talent.csv")

    import argparse
    parser = argparse.ArgumentParser(description="Recommend a Three Houses class path for a character.")
    parser.add_argument("character", type=str, help="Character name, e.g. 'Bernadetta'")
    parser.add_argument("--role", type=str, default=None,
                         choices=list(ROLE_PROFILES.keys()),
                         help="Target role (omit to auto-detect from growth rates)")
    parser.add_argument("--level", type=int, default=30, help="Target level for stat projection")
    args = parser.parse_args()

    result = recommend_for_character(
        args.character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=args.role, target_level=args.level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
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
            if step.get("requirement"):
                print(f"  {'':>13}     requires {step['requirement']}")
            print(f"  {'':>13}  -> {step['why']}")
        print(f"Expected stats at level {result['expected_stats_at_level']} as {result['path'][-1]['class']}:")
        print(f"  {result['expected_final_stats']}")

    if result["eligible_unique_classes"]:
        print("Also eligible for these Unique classes:")
        for option in result["eligible_unique_classes"]:
            print(f"  {option['class']} (fit score {option['score']})")


if __name__ == "__main__":
    main()