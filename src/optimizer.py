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

# class_stat_boosts.csv's tier value for the four Cindered Shadows
# certification classes (Trickster, War Monk/Cleric, Dark Flier, Valkyrie).
# These were sitting in the data entirely unused - not in TIER_ORDER, so
# never considered by recommend_path, and not "Unique" tier either, so
# never surfaced by eligible_unique_classes. Their real certification
# requirements (data/class_weapon_requirements.csv) are B-rank pairs, the
# same shape as Advanced-tier requirements (Hero: Sword B/Axe C, Swordmaster:
# Sword A) rather than Master's usual A-rank-plus, so when a caller opts in
# via include_dlc_classes, they're merged into the ADVANCED tier's candidate
# pool specifically (see recommend_path / list_eligible_classes_at_tier) -
# an approximation, not a sourced "this is officially an Advanced-tier
# class" claim, since Cindered Shadows classes don't fit the base game's
# tier ladder at all.
DLC_CLASS_TIER = "DLC Exclusive"
DLC_CLASS_MERGE_TIER = "Advanced"

# Character-locked Unique-tier classes that DO map onto a specific rung of
# the Beginner->Master ladder, per class_eligibility.csv's own unlock_note
# text ("Edelgard's unique Advanced-tier class", "Byleth's unique
# Master-tier class", etc.) - unlike Dancer/Noble/Commoner, which are Unique
# tier but not a personal story-class endpoint for one character. Splicing
# these into recommend_path (see below) answers "characters with unique
# classes should have those incorporated as options, weighted higher" -
# these seven are exactly the classes that are both (a) locked to one
# specific character and (b) documented as replacing a named tier, so they
# have a principled slot to go in, unlike Dancer (open to anyone, no fixed
# tier) or Noble/Commoner (starting classes, not endpoints).
UNIQUE_STORY_CLASS_TIER = {
    "Armored Lord": "Advanced",
    "High Lord": "Advanced",
    "Wyvern Master": "Advanced",
    "Emperor": "Master",
    "Great Lord": "Master",
    "Barbarossa": "Master",
    "Enlightened One": "Master",
}
# Flat bonus added to a spliced-in unique class's fit score (additive, not
# a multiplier - Armored Lord's stat line, all HP/Def and a NEGATIVE Spd,
# scores below zero for a Physical Attacker role's weighted dot product,
# and a multiplier only makes a negative score more negative). +8 clears
# the largest ordinary winning margins seen in practice (Edelgard's own
# Physical-Attacker path: Armored Lord/Emperor's negative-Spd, tank-leaning
# stat lines actually score a couple points worse than Swordmaster/Wyvern
# Lord on raw stats alone - a big enough real-world gap that a smaller
# bonus, e.g. +5, still lost to them) while still occasionally losing a
# dead-even tie to a tier's genuinely best-specialized class (e.g. Byleth's
# Great Knight for a Tank build over the more general-purpose Enlightened
# One) rather than mechanically overriding every role, every time,
# regardless of fit.
UNIQUE_CLASS_SCORE_BONUS = 8.0

# Small score bonus for a class whose certification weapon overlaps a
# character's own starting weapon proficiency (character_weapon_talent.csv)
# - "each class also helps a character grow in a specific weapon rank
# faster than others." This generalizes what apply_weapon_affinity_fallback
# already did (matching proficiency) from a Beginner-tier-only tie-break
# into a real, if modest, factor at every tier: kept deliberately small
# relative to a class's real stat-boost contribution (which is usually
# several points) so it refines close calls and rewards leaning into a
# character's own natural weapon strengths, without overriding a tier
# where the stat data already has a clear, better-fitting answer.
WEAPON_PROFICIENCY_BONUS = 0.5


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


def weapon_growth_bonus(
    class_name: str,
    weapon_req_lookup: dict | None,
    character_proficiency: set | None,
) -> float:
    """
    WEAPON_PROFICIENCY_BONUS if class_name's certification weapon(s)
    overlap character_proficiency (the character's own starting weapon
    talent - see load_character_proficiency_lookup), else 0.0.

    In the real game, training in a class that uses your character's
    already-strong weapon type is how that weapon rank actually climbs
    fastest - practicing Sword in a sword-using class raises Sword rank,
    practicing Reason in a magic class raises Reason rank, and so on. This
    doesn't simulate weapon-exp gain (see the module's Known Simplifications
    in README.md), but it does let "this class plays to a strength this
    character already has" nudge the recommendation, generalizing what
    apply_weapon_affinity_fallback already did as a Beginner-only
    tie-break into a small factor at every tier.
    """
    if not weapon_req_lookup or not character_proficiency:
        return 0.0
    info = weapon_req_lookup.get(class_name)
    if info is None:
        return 0.0
    required_skills = {skill for skill, _ in info["requirements"]}
    return WEAPON_PROFICIENCY_BONUS if required_skills & character_proficiency else 0.0


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


def load_starting_level_lookup(starting_level_df: pd.DataFrame | None) -> dict:
    """
    Index data/character_starting_level.csv by character name -> the level
    they actually join the player's roster at ("join level"), rather than
    the Level 1 baseline Serenes Forest's own base-stats table normalizes
    everyone to for easy side-by-side comparison ("these are the
    characters' stats when they are Level 1 in the Noble/Commoner class" -
    per that page's own header note). For the Protagonist and the 24 house
    students, join level is a verified 1 (playable from the Ch1 prologue).
    For Church/Knights of Seiros staff and DLC characters, who join
    progressively later, join level is an approximate placeholder derived
    from their recruitment timing (see the CSV's own notes column for each
    row's sourcing and confidence) rather than a directly-sourced figure -
    and for the Ashen Wolves specifically (Yuri/Balthus/Constance/Hapi),
    who canonically auto-scale to roughly match Byleth's own level when
    recruited rather than joining at one fixed level, no floor is enforced
    at all (join_level 1, same "doesn't simulate Byleth's own playthrough"
    gap already documented for cross-house recruitment in team_builder.py).

    Returns {} if starting_level_df is None (backward compatible - callers
    that don't pass it get today's behavior: everyone treated as level 1).
    """
    if starting_level_df is None:
        return {}
    lookup = {}
    for _, row in starting_level_df.iterrows():
        level = row.get("join_level")
        lookup[row["name"]] = int(level) if pd.notna(level) else 1
    return lookup


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


def restrict_support_to_magic_classes(
    tier_classes: pd.DataFrame,
    role_name: str,
    weapon_req_lookup: dict,
) -> pd.DataFrame:
    """
    For the Support role specifically, drop candidate classes that have no
    magic (Reason/Faith) access at all, per each class's real certification
    weapon_category (data/class_weapon_requirements.csv - "magic" or
    "hybrid" pass, "physical" is dropped).

    Why this exists: Support's fit score is driven by Resistance (see
    ROLE_PROFILES), which several purely physical/flying classes also boost
    generously - Falcon Knight boosts Res +4 with no Reason/Faith access at
    all, which could out-score every real healer class at Master tier on
    Res alone, landing a healer-archetype character (e.g. Mercedes) in a
    class that literally can't cast the healing magic that made her a
    Support pick in the first place. A class with no certification data
    (Unique-tier classes, whose requirements aren't in the CSV - see
    load_weapon_requirements_lookup) is left in rather than dropped, since
    "unknown" shouldn't be treated as "definitely non-magic"; if this hard
    filter would eliminate every candidate at a tier (an honest gap in the
    data, not a real one for this game - every tier has at least one magic
    or hybrid option), the unfiltered tier is returned instead of an empty
    one.
    """
    if role_name != "Support" or not weapon_req_lookup:
        return tier_classes

    def has_magic_access(name: str) -> bool:
        info = weapon_req_lookup.get(name)
        if info is None:
            return True  # no requirement data on file - don't block on an unknown
        return info["weapon_category"] in ("magic", "hybrid")

    magic_capable = tier_classes[tier_classes["name"].map(has_magic_access)]
    if magic_capable.empty:
        return tier_classes
    return magic_capable


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


def eligible_unique_story_class_by_tier(
    character_name: str,
    eligibility_lookup: dict,
    character_gender: str | None = None,
) -> dict:
    """
    tier -> class_name for each UNIQUE_STORY_CLASS_TIER entry character_name
    is eligible for (see is_class_eligible) - in practice non-empty only
    for the Protagonist and the three house leaders, since every entry in
    UNIQUE_STORY_CLASS_TIER is locked to exactly one character. Used by
    recommend_path and list_eligible_classes_at_tier to splice a
    character's own personal story class into their path at its documented
    tier, rather than only surfacing it via the separate
    eligible_unique_classes callout.
    """
    result = {}
    for class_name, tier in UNIQUE_STORY_CLASS_TIER.items():
        if is_class_eligible(character_name, class_name, eligibility_lookup, character_gender):
            result[tier] = class_name
    return result


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
    include_dlc_classes: bool = False,
) -> list[dict]:
    """
    For a target role, pick the best-fitting class at each tier in order.
    Returns a list of {"tier": ..., "class": ..., "score": ..., "why": ...,
    "requirement": ..., "is_unique_class": ...} dicts.

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
    callers that don't have this data on hand). The same two arguments also
    control unique-class splicing: if character_name has a personal Unique-
    tier story class mapped to a reachable tier (Emperor, Great Lord,
    Barbarossa, Enlightened One, and their Advanced-tier predecessors - see
    UNIQUE_STORY_CLASS_TIER), it's added to that tier's candidate pool and
    weighted to win it (UNIQUE_CLASS_SCORE_WEIGHT) whenever it's reasonably
    competitive for role_name, not only when it already tops the raw stat
    score - it's the character's own canonical class in the story.

    include_dlc_classes: when True, merges the four Cindered Shadows
    certification classes (Trickster, War Monk/Cleric, Dark Flier, Valkyrie
    - see DLC_CLASS_TIER/DLC_CLASS_MERGE_TIER) into the Advanced tier's
    candidate pool, same eligibility rules as everything else. Off by
    default - same "opt-in and separate from the base roster" precedent as
    team_builder's include_dlc for DLC characters, since these also require
    owning the Cindered Shadows DLC.
    """
    role_weights = ROLE_PROFILES[role_name]
    path = []

    reachable = reachable_tiers(target_level, tiers) if target_level is not None else tiers
    check_eligibility = character_name is not None and eligibility_lookup is not None

    unique_by_tier = (
        eligible_unique_story_class_by_tier(character_name, eligibility_lookup, character_gender)
        if check_eligibility else {}
    )

    for tier in reachable:
        tier_classes = stat_boosts_df[stat_boosts_df["tier"] == tier]
        if include_dlc_classes and tier == DLC_CLASS_MERGE_TIER:
            tier_classes = pd.concat([tier_classes, stat_boosts_df[stat_boosts_df["tier"] == DLC_CLASS_TIER]])

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

        unique_class_name = unique_by_tier.get(tier)
        if unique_class_name and unique_class_name not in tier_classes["name"].values:
            tier_classes = pd.concat([tier_classes, stat_boosts_df[stat_boosts_df["name"] == unique_class_name]])

        if tier_classes.empty:
            continue

        # The spliced-in unique class (if any) is exempted from the two
        # narrowing filters below: restrict_to_primary_relevant would
        # otherwise drop it outright whenever its stat line doesn't touch
        # the role's primary stat at all (e.g. Armored Lord's 0 Str would
        # get it dropped for a Physical Attacker role, before the scoring
        # bonus even gets a chance to weigh it) - it should always reach
        # scoring, where UNIQUE_CLASS_SCORE_BONUS decides whether it wins.
        is_unique_row = tier_classes["name"] == unique_class_name
        unique_row_df = tier_classes[is_unique_row]
        rest = tier_classes[~is_unique_row]

        rest = restrict_support_to_magic_classes(rest, role_name, weapon_req_lookup or {})
        rest = apply_weapon_affinity_fallback(
            rest, role_name, role_weights, weapon_req_lookup or {}, character_proficiency
        )
        rest = restrict_to_primary_relevant(rest, role_weights)
        tier_classes = pd.concat([rest, unique_row_df]) if not unique_row_df.empty else rest

        def score_row(row, unique_class_name=unique_class_name):
            score = score_class_for_role(row, role_weights)
            score += weapon_growth_bonus(row["name"], weapon_req_lookup, character_proficiency)
            if row["name"] == unique_class_name:
                score += UNIQUE_CLASS_SCORE_BONUS
            return score

        scores = tier_classes.apply(score_row, axis=1)
        best_idx = scores.idxmax()
        best_row = tier_classes.loc[best_idx]
        best_class = best_row["name"]
        is_unique_pick = unique_class_name is not None and best_class == unique_class_name

        why = (
            f"{character_name}'s own unique class at this tier - their canonical path in the story, "
            f"weighted above the generic options here."
            if is_unique_pick else
            explain_pick(best_row, role_weights, role_name, weapon_req_lookup, character_proficiency)
        )

        path.append({
            "tier": tier,
            "class": best_class,
            "score": round(float(scores.loc[best_idx]), 2),
            "why": why,
            "requirement": format_requirement(best_class, weapon_req_lookup) if weapon_req_lookup else None,
            "is_unique_class": is_unique_pick,
        })

    return path


def list_eligible_classes_at_tier(
    tier: str,
    stat_boosts_df: pd.DataFrame,
    character_name: str | None = None,
    eligibility_lookup: dict | None = None,
    character_gender: str | None = None,
    include_dlc_classes: bool = False,
) -> list[str]:
    """
    Every player-selectable class name at `tier` that character_name is
    eligible for (same eligibility rules as recommend_path - see
    is_class_eligible), excluding story/enemy-specific variant rows (e.g.
    "Lord (Judith)"). If character_name/eligibility_lookup are omitted, no
    eligibility filtering happens and every class at that tier is
    returned. If character_name has a personal Unique-tier story class
    mapped to `tier` (see UNIQUE_STORY_CLASS_TIER) and eligibility_lookup
    is given, it's included alongside the tier's regular classes - same
    "incorporated as an option" treatment recommend_path gives it.
    include_dlc_classes additionally merges the Cindered Shadows
    certification classes into the Advanced tier's option list (see
    DLC_CLASS_TIER/DLC_CLASS_MERGE_TIER).

    Used to let a user override the recommended pick at a given tier with
    any other class they were actually eligible for - the "mix and match"
    path - rather than only ever seeing the single top-scoring choice.
    """
    tier_classes = stat_boosts_df[stat_boosts_df["tier"] == tier]
    if include_dlc_classes and tier == DLC_CLASS_MERGE_TIER:
        tier_classes = pd.concat([tier_classes, stat_boosts_df[stat_boosts_df["tier"] == DLC_CLASS_TIER]])
    tier_classes = tier_classes[~tier_classes["name"].str.contains(r"\(", regex=True)]

    names = set(tier_classes["name"].tolist())

    if character_name is not None and eligibility_lookup is not None:
        names = {
            name for name in names
            if is_class_eligible(character_name, name, eligibility_lookup, character_gender)
        }
        unique_by_tier = eligible_unique_story_class_by_tier(character_name, eligibility_lookup, character_gender)
        if tier in unique_by_tier:
            names.add(unique_by_tier[tier])

    return sorted(names)


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
    start_level: int = 1,
) -> dict:
    """
    Estimate a character's expected stats at target_level in a given final
    class: base stats (still Serenes Forest's Level-1 baseline - see
    load_starting_level_lookup) + expected level-up gains from start_level
    up to target_level (growth_rate% per level, used as an expected value -
    e.g. a 45% growth rate contributes 0.45 expected stat points per level,
    not a simulated coin flip) + the final class's stat boost.

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
    start_level: int = 1,
) -> dict | None:
    """
    Expected stats for character_name at target_level, if their current
    class were class_name specifically - the same expected-value
    calculation recommend_for_character uses for its recommended final
    class (see expected_stats_at_level), just callable for any class the
    user picks by hand. Powers the "mix and match" path override in the
    UI: swap in a different class at whichever tier you'd actually end on,
    and see the resulting projection, not just the tool's own top pick.
    start_level should be the character's own join level when known (see
    load_starting_level_lookup) - passed through to expected_stats_at_level.
    Returns None if class_name isn't in stat_boosts_df at all.
    """
    boost_rows = stat_boosts_df[stat_boosts_df["name"] == class_name]
    if boost_rows.empty:
        return None
    base_row = base_stats_df[base_stats_df["name"] == character_name].iloc[0]
    growth_row = growth_rates_df[growth_rates_df["name"] == character_name].iloc[0]
    return expected_stats_at_level(base_row, growth_row, boost_rows.iloc[0], target_level, start_level)


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
    starting_level_df: pd.DataFrame | None = None,
    include_dlc_classes: bool = False,
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
    weapon-affinity fallback and the weapon-growth scoring bonus (see
    apply_weapon_affinity_fallback / weapon_growth_bonus). starting_level_df
    (data/character_starting_level.csv), if given, supplies the character's
    join level (see load_starting_level_lookup): target_level is floored to
    that join level (a character can never be projected below the level
    they actually join the roster at - see the "join_level" return value),
    and both "base_stats" and "expected_final_stats" are computed from that
    join level rather than assuming everyone starts from level 1. All these
    are optional and default to None/False, which disables the
    corresponding feature entirely - existing callers that don't pass them
    keep getting today's behavior (join_level 1, no DLC classes).
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

    starting_level_lookup = load_starting_level_lookup(starting_level_df)
    join_level = starting_level_lookup.get(character_name, 1)
    effective_target_level = max(target_level, join_level)
    base_stats_at_join = base_stats_at_join_level(base_row, growth_row, join_level)

    path = recommend_path(
        stat_boosts_df, used_role, target_level=effective_target_level,
        character_name=character_name,
        eligibility_lookup=eligibility_lookup,
        character_gender=character_gender,
        weapon_req_lookup=weapon_req_lookup,
        character_proficiency=character_proficiency,
        include_dlc_classes=include_dlc_classes,
    )
    final_class_name = path[-1]["class"] if path else None

    final_stats = None
    if final_class_name:
        final_boost_row = stat_boosts_df[stat_boosts_df["name"] == final_class_name].iloc[0]
        final_stats = expected_stats_at_level(
            base_row, growth_row, final_boost_row, effective_target_level, start_level=join_level,
        )

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
        "expected_stats_at_level": effective_target_level,
        "requested_target_level": target_level,
        "join_level": join_level,
        "base_stats_at_join_level": base_stats_at_join,
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
    starting_level_df = pd.read_csv(DATA_DIR / "character_starting_level.csv")

    import argparse
    parser = argparse.ArgumentParser(description="Recommend a Three Houses class path for a character.")
    parser.add_argument("character", type=str, help="Character name, e.g. 'Bernadetta'")
    parser.add_argument("--role", type=str, default=None,
                         choices=list(ROLE_PROFILES.keys()),
                         help="Target role (omit to auto-detect from growth rates)")
    parser.add_argument("--level", type=int, default=30, help="Target level for stat projection")
    parser.add_argument("--include-dlc-classes", action="store_true",
                         help="Also consider the Cindered Shadows certification classes (Trickster, "
                              "War Monk/Cleric, Dark Flier, Valkyrie) as Advanced-tier options.")
    args = parser.parse_args()

    result = recommend_for_character(
        args.character, base_stats_df, growth_rates_df, stat_boosts_df,
        role_name=args.role, target_level=args.level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
        starting_level_df=starting_level_df, include_dlc_classes=args.include_dlc_classes,
    )

    print(f"\n{result['character']}")
    if result["join_level"] > 1:
        print(f"Joins the roster at level {result['join_level']} - base stats and stat projection "
              f"start from there, not level 1.")
    if result["expected_stats_at_level"] != args.level:
        print(f"Target level {args.level} is below join level {result['join_level']} - "
              f"using {result['expected_stats_at_level']} instead.")
    print(f"Auto-detected natural role: {result['auto_detected_role']} (similarity {result['auto_detection_score']})")
    print(f"Recommended path toward: {result['role_used']}")
    if not result["path"]:
        print(f"  (no tier reachable at level {result['expected_stats_at_level']} - Beginner requires level "
              f"{TIER_LEVEL_REQUIREMENTS['Beginner']})")
    else:
        for step in result["path"]:
            tag = " [unique class]" if step.get("is_unique_class") else ""
            print(f"  {step['tier']:>13}: {step['class']}{tag} (fit score {step['score']})")
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