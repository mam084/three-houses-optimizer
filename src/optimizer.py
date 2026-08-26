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

# Every class data/class_eligibility.csv itself documents as reached WITHOUT
# a certification exam - its own unlock_note says so for each of these
# ("... unlocked via story progression rather than a certification exam",
# "... not selectable via certification", Dancer's White Heron Cup event
# note) - unlike every ordinary class, which is entered by clearing a real
# in-game certification exam with a real weapon/skill-rank requirement (see
# data/class_weapon_requirements.csv). These eleven should never show a
# certification-requirement line: the seven UNIQUE_STORY_CLASS_TIER classes
# above, plus Lord (the house-leader class, story-unlocked, not one of the
# seven since it isn't itself a splice target - Armored Lord/High Lord/
# Wyvern Master are its own upgrades), and Noble/Commoner (starting
# classes) and Dancer (White Heron Cup, not a certification exam).
#
# This used to rely entirely on data/class_weapon_requirements.csv simply
# having no row for these classes - true for ten of the eleven, but Lord
# had a stray row (tier mislabeled "Intermediate", requirement "Sword D+
# and Authority C") left over from before unique-class handling existed,
# which surfaced as a real, wrong "Requires: ..." line on Lord specifically
# (the same class of bug reported for Enlightened One) - see
# load_weapon_requirements_lookup, which now filters this set out
# regardless of what the CSV happens to contain, so a future stray row for
# any of these eleven can't reintroduce the bug.
NO_CERTIFICATION_CLASSES = set(UNIQUE_STORY_CLASS_TIER.keys()) | {"Lord", "Noble", "Commoner", "Dancer"}
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

# Small score bonus for a class whose certification weapon matches the
# weapon type of a character's own Hero's Relic (data/character_relics.csv)
# - the same precedent and magnitude as WEAPON_PROFICIENCY_BONUS above,
# just keyed off a relic instead of a starting proficiency. In the real
# game a Hero's Relic only unlocks its special combat art for a character
# who bears the matching Crest, but its raw Might/Hit/Crit line is strong
# enough that a relic-bearing character is generally steered toward
# classes that can actually equip its weapon type regardless - "a
# character's relic nudges scoring toward classes that use the relic's
# weapon type." Not every character has a relic (see
# load_character_relic_lookup); this is a no-op for the ones who don't.
RELIC_AFFINITY_BONUS = 0.5

# Small score bonus/penalty for how a class's growth-RATE modifiers (not
# just its flat one-time stat boost - see data/class_growth_rates.csv and
# load_class_growth_lookup) line up with a role's priorities, and a
# separate penalty for asking a character to adopt a weapon type they've
# never trained anywhere in the path so far or at the start (see
# weapon_switch_penalty). Both intentionally small relative to a class's
# own stat-boost score (usually several points) and to
# UNIQUE_CLASS_SCORE_BONUS above - calibrated (see
# TestGrowthRateScoringDoesNotDestabilizeExistingPicks in
# tests/test_optimizer.py) so they refine which class wins a genuinely
# close tier and surface a real in-game cost, without overriding a tier
# where the stat-boost data already gives a clear, better-fitting answer.
GROWTH_RATE_SCORE_WEIGHT = 0.05
WEAPON_SWITCH_PENALTY = 0.6

# Mount and heavy-armor training are more forgiving than a standard weapon-
# type switch (e.g. Sword -> Axe) - in practice a character who's ridden
# ANY mount (Cavalier's Riding, Pegasus Knight's Flying, an early Armored
# Knight's Heavy Armour, ...) picks up a different mount/armor skill fairly
# readily, since a lot of what a real player is actually building - map
# awareness, positioning around a bulkier unit, fighting from horseback or
# in the air - carries over between them, unlike starting a weapon type
# from zero. So weapon_switch_penalty treats these three skills specially:
# they're only flagged as a real switch when BOTH (a) the class asks for
# them at a genuinely high rank (see HIGH_CERTIFICATION_RANKS - A is the
# highest rank in this dataset) AND (b) the character has zero practice in
# ANY of the three so far (not just the literal one being asked for), per
# "the warning should only fire for these categories on a jump to a high
# rank with zero prior practice in any related skill." Ordinary weapon
# skills (Sword, Axe, Lance, Bow, Brawling, Reason, Faith, Authority)
# aren't touched by this - they keep the original strict "only the exact
# skill counts" rule.
#
# This is deliberately narrower than it sounds: Catherine's Swordmaster ->
# Wyvern Lord jump (Lance C / Axe A / Flying A) stays flagged even after
# this relaxation, since Axe A is an ORDINARY weapon requirement she has
# zero practice in - that half of the AND requirement alone still trips
# the switch, regardless of how Flying A (the mount half) is scored. See
# test_catherine_swordmaster_to_wyvern_lord_flags_the_real_complaint,
# kept passing by this change on purpose - that transition is genuinely
# unrealistic in-game and the spec asks for it to stay flagged as-is.
MOUNT_ARMOR_SKILLS = {"Riding", "Flying", "Heavy Armour"}
HIGH_CERTIFICATION_RANKS = {"A"}


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
# (Mov) used to be weighted here too, on the theory that a fast character
# should also be steered toward flying/mounted classes. In practice this
# double-counted: Mov isn't a growth stat at all (it's purely a class
# trait, so it never affected natural-role DETECTION - see the paragraph
# below), and weighting it in CLASS scoring just meant "Speed/Precision"
# picks skewed toward whichever class happened to have the biggest Mov
# stat, even when that class's actual Spd/Dex boosts were mediocre - a
# character who's fast doesn't need a role that also chases mobility for
# its own sake, that's a separate axis a player can already see and choose
# directly (e.g. by eligibility or the class explorer). Dropped entirely -
# this role now scores purely on Speed (primary) and Dex (secondary,
# "precision"), the same two-stat shape as every other role profile below.
#
# Movement isn't a growth stat - it's purely a class trait, not something a
# character has an innate growth rate for. That means natural role
# DETECTION (which only has growth rates to work with) never could tell
# whether a character has any flying affinity in the first place; what the
# growth data CAN show is a Dex/Spd lean, which is what "speed/precision"
# actually names.
ROLE_PROFILES = {
    "Physical Attacker": {"Str": 1.0, "Spd": 0.5},
    "Magic Attacker": {"Mag": 1.0, "Dex": 0.3},
    "Tank": {"HP": 1.0, "Def": 1.0},
    "Support": {"Res": 1.0},
    "Speed/Precision": {"Spd": 1.0, "Dex": 0.3},
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


def load_character_relic_lookup(character_relics_df: pd.DataFrame | None) -> dict:
    """
    Index data/character_relics.csv by character name -> the set of weapon
    types (e.g. {"Sword"}, or {"Sword", "Reason"} for a character with more
    than one relic/crest - see Lysithea) their own Hero's Relic(s) use.

    Hand-curated from Serenes Forest's per-weapon-type pages (each
    Hero's-Relic-tier weapon's own certification requirement text names the
    Crest that powers its combat art) cross-referenced against each
    Crest's documented bearer(s) - the same sourcing precedent as
    class_eligibility.csv and class_weapon_requirements.csv. Scoped to
    actual Hero's Relic WEAPONS only (a class's certification requirement
    is a weapon-type concept) - non-weapon relics (Aegis Shield, Rafail
    Gem, Fetters of Dromi - shields/accessories, no weapon-type
    certification to nudge toward) aren't included. Not every playable
    character bears a relic; a character absent from the CSV simply has no
    relic-affinity bonus to apply (see relic_affinity_bonus).

    Returns {} if character_relics_df is None - callers degrade to
    today's "relics aren't modeled" behavior rather than failing.
    """
    if character_relics_df is None:
        return {}
    lookup: dict[str, set] = {}
    for _, row in character_relics_df.iterrows():
        lookup.setdefault(row["character"], set()).add(row["weapon_type"])
    return lookup


def relic_affinity_bonus(
    class_name: str,
    weapon_req_lookup: dict | None,
    character_relic_weapon_types: set | None,
) -> float:
    """
    RELIC_AFFINITY_BONUS if class_name's certification weapon(s) overlap
    character_relic_weapon_types (a character's own Hero's Relic weapon
    type(s) - see load_character_relic_lookup), else 0.0. Same shape and
    magnitude as weapon_growth_bonus, just keyed off a relic instead of a
    starting proficiency - "a character's relic nudges scoring toward
    classes that use the relic's weapon type."
    """
    if not weapon_req_lookup or not character_relic_weapon_types:
        return 0.0
    info = weapon_req_lookup.get(class_name)
    if info is None:
        return 0.0
    required_skills = {skill for skill, _ in info["requirements"]}
    return RELIC_AFFINITY_BONUS if required_skills & character_relic_weapon_types else 0.0


def weapon_switch_penalty(
    class_name: str,
    weapon_req_lookup: dict | None,
    accumulated_skills: set,
    original_proficiency: set | None,
) -> float:
    """
    WEAPON_SWITCH_PENALTY if class_name's certification asks for a skill
    type neither the character started with (original_proficiency) NOR
    appeared anywhere earlier in the path so far (accumulated_skills),
    else 0.0. Respects the requirement's AND/OR shape
    (weapon_req_lookup's "requirement_type" - see
    load_weapon_requirements_lookup): an OR requirement (e.g. "Sword B or
    Axe C") only needs ONE listed skill already known to count as a
    non-switch, but an AND requirement (e.g. Wyvern Lord's "Lance C and
    Axe A and Flying A") needs EVERY listed skill already known - already
    knowing Axe from an earlier tier doesn't make picking up Lance AND
    Flying, at A rank, for the first time at Master tier a free switch.
    Getting this AND/OR distinction right is exactly what "weapon
    requirements aren't weighted well - Catherine going Swordmaster to
    Wyvern Lord is very difficult in practice" needed: an earlier version
    of this check treated "any overlap at all" as safe for every class
    regardless of AND/OR, so Catherine's genuinely rough Sword ->
    Lance/Axe/Flying jump at Wyvern Lord went unflagged the moment an
    earlier, unrelated tier happened to touch Axe (see
    test_catherine_swordmaster_to_wyvern_lord_flags_the_real_complaint).

    Checking BOTH accumulated_skills and original_proficiency (not just
    accumulated_skills) matters too: an early, unrelated detour picked
    purely for its stats shouldn't count as "already knows this weapon
    type" any more than never having touched it at all would.

    In the real game, training a brand-new weapon proficiency this late
    is a real practical cost the flat per-tier stat-boost score doesn't
    otherwise reflect at all.

    Mount/heavy-armor skills (MOUNT_ARMOR_SKILLS) are graded on a more
    forgiving rule than ordinary weapon skills - see that constant's
    docstring: a mount/armor requirement only counts as "still unmet" when
    it's at a genuinely high rank (HIGH_CERTIFICATION_RANKS) AND the
    character has zero practice in any of the three mount/armor skills so
    far, not just the one literally being asked for. This can make an OR
    requirement satisfied (or an AND requirement's mount/armor half
    harmless) even without the exact skill in `known`.
    """
    if not weapon_req_lookup:
        return 0.0
    info = weapon_req_lookup.get(class_name)
    if info is None:
        return 0.0
    requirements = info["requirements"]
    if not requirements:
        return 0.0
    known = set(accumulated_skills) | set(original_proficiency or [])

    def skill_unmet(skill: str, rank: str) -> bool:
        if skill in MOUNT_ARMOR_SKILLS:
            if rank not in HIGH_CERTIFICATION_RANKS:
                return False  # low/mid-rank mount or armor training - never a flagged switch
            return not (known & MOUNT_ARMOR_SKILLS)  # high rank - only unmet if truly no related practice
        return skill not in known

    unmet_flags = [skill_unmet(skill, rank) for skill, rank in requirements]
    if info.get("requirement_type") == "AND":
        is_switch = any(unmet_flags)
    else:
        is_switch = all(unmet_flags)
    return WEAPON_SWITCH_PENALTY if is_switch else 0.0


def load_class_growth_lookup(class_growth_df: pd.DataFrame | None) -> dict:
    """
    Index data/class_growth_rates.csv by class name -> {stat: modifier}
    (percentage points, additive on top of the character's own personal
    growth rate on every level-up - a real, separate Three Houses
    mechanic from a class's one-time flat certification stat boost in
    class_stat_boosts.csv; see that CSV's own header and
    https://serenesforest.net/three-houses/classes/growth-rates/).

    Returns {} if class_growth_df is None - callers degrade to today's
    "growth rates aren't modeled per-class" behavior rather than failing.
    """
    if class_growth_df is None:
        return {}
    lookup = {}
    for _, row in class_growth_df.iterrows():
        lookup[row["name"]] = {stat: row[stat] for stat in STAT_COLS if stat in row.index}
    return lookup


def load_class_base_stats_lookup(class_base_stats_df: pd.DataFrame | None) -> dict:
    """
    Index data/class_base_stats.csv by class name -> {stat: value} - each
    class's OWN base-stat line (hand-curated from Serenes Forest's
    per-class base-stats page), a third, separate mechanic from both a
    class's one-time flat certification boost (class_stat_boosts.csv) and
    its growth-RATE modifiers (class_growth_rates.csv, load_class_growth_lookup).

    This is a FLOOR, not a bonus: on certifying into a class, if the
    character's current stat is already below that class's own base stat,
    it snaps up to the class's base value; if the character's stat is
    already higher (e.g. a heavily-leveled character moving into an early
    class), nothing happens - the class base never subtracts, and it never
    stacks additively with the character's own stat either. See
    expected_stats_along_path, which applies this at the start of every
    tier actually spent along a path (the moment of certifying into that
    tier's class), not just the final one.

    Returns {} if class_base_stats_df is None - callers degrade to no
    floor being applied (today's pre-round-5 behavior) rather than failing.
    """
    if class_base_stats_df is None:
        return {}
    lookup = {}
    for _, row in class_base_stats_df.iterrows():
        lookup[row["name"]] = {stat: row[stat] for stat in STAT_COLS if stat in row.index}
    return lookup


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

    Any class in NO_CERTIFICATION_CLASSES is skipped even if the CSV
    happens to contain a row for it - see that constant's docstring for
    why (a stray "Lord" row was exactly this bug in practice). This is the
    single choke point every caller (format_requirement,
    weapon_switch_penalty, weapon_growth_bonus,
    apply_weapon_affinity_fallback, ...) goes through, so the guard applies
    everywhere at once rather than needing a check at each call site.
    """
    if weapon_req_df is None:
        return {}
    lookup = {}
    for _, row in weapon_req_df.iterrows():
        if row["class_name"] in NO_CERTIFICATION_CLASSES:
            continue
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
    class_growth_lookup: dict | None = None,
    character_relic_weapon_types: set | None = None,
) -> list[dict]:
    """
    For a target role, pick the best-fitting class at each tier in order.
    Returns a list of {"tier": ..., "class": ..., "score": ..., "why": ...,
    "requirement": ..., "is_unique_class": ..., "weapon_switch_warning":
    ...} dicts.

    class_growth_lookup (see load_class_growth_lookup), if given, factors
    each candidate's growth-RATE modifiers into its score (see
    score_growth_for_role/GROWTH_RATE_SCORE_WEIGHT) alongside its flat
    stat boost, and every tier's certification requirement is checked
    against a running "skills used so far in this path" set (seeded from
    character_proficiency): a class that needs a weapon type foreign to
    both the character's own starting proficiency and everything picked
    earlier in the path takes a WEAPON_SWITCH_PENALTY and that step's
    "weapon_switch_warning" is set True (see weapon_switch_penalty) -
    "weapon requirements aren't weighted well" (e.g. Swordmaster into
    Wyvern Lord) is exactly this case.

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

    character_relic_weapon_types (see load_character_relic_lookup), if
    given, is the set of weapon types the character's own Hero's Relic(s)
    use (e.g. {"Sword"} for the Sword of the Creator). Each tier's
    candidates get a RELIC_AFFINITY_BONUS toward classes whose weapon
    certification uses one of those types (see relic_affinity_bonus) - same
    precedent and magnitude as weapon_growth_bonus, on the theory that a
    character canonically wielding a specific relic is nudged toward
    classes that let them actually use it. Characters with no relic (most
    of the roster) get no bonus and score exactly as before.
    """
    role_weights = ROLE_PROFILES[role_name]
    path = []

    reachable = reachable_tiers(target_level, tiers) if target_level is not None else tiers
    check_eligibility = character_name is not None and eligibility_lookup is not None

    unique_by_tier = (
        eligible_unique_story_class_by_tier(character_name, eligibility_lookup, character_gender)
        if check_eligibility else {}
    )

    # Skills required (by weapon-type certification) by whatever's actually
    # been picked so far in this path, plus the character's own starting
    # proficiency - see weapon_switch_penalty. Updated after each tier's
    # pick, in path order, so a later tier's penalty check reflects the
    # path actually built, not a hypothetical.
    accumulated_skills = set(character_proficiency or [])

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

        def score_row(row, unique_class_name=unique_class_name, accumulated_skills=frozenset(accumulated_skills)):
            score = score_class_for_role(row, role_weights)
            score += weapon_growth_bonus(row["name"], weapon_req_lookup, character_proficiency)
            score += relic_affinity_bonus(row["name"], weapon_req_lookup, character_relic_weapon_types)
            if class_growth_lookup:
                growth_mod = class_growth_lookup.get(row["name"], {})
                score += score_growth_for_role(growth_mod, role_weights) * GROWTH_RATE_SCORE_WEIGHT
            score -= weapon_switch_penalty(row["name"], weapon_req_lookup, accumulated_skills, character_proficiency)
            if row["name"] == unique_class_name:
                score += UNIQUE_CLASS_SCORE_BONUS
            return score

        scores = tier_classes.apply(score_row, axis=1)
        best_idx = scores.idxmax()
        best_row = tier_classes.loc[best_idx]
        best_class = best_row["name"]
        is_unique_pick = unique_class_name is not None and best_class == unique_class_name
        switch_warning = weapon_switch_penalty(
            best_class, weapon_req_lookup, accumulated_skills, character_proficiency
        ) > 0

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
            "weapon_switch_warning": switch_warning,
        })

        best_info = weapon_req_lookup.get(best_class) if weapon_req_lookup else None
        if best_info:
            accumulated_skills |= {skill for skill, _ in best_info["requirements"]}

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
    class_growth_df: pd.DataFrame | None = None,
    class_base_stats_df: pd.DataFrame | None = None,
    character_relics_df: pd.DataFrame | None = None,
) -> dict:
    """
    Full recommendation for one character: auto-detects a role if none is
    given, builds a class path toward it, and estimates expected stats at
    target_level by simulating growth tier-by-tier along that whole path
    (see expected_stats_along_path), not just from the final class alone.

    class_growth_df (data/class_growth_rates.csv), if given, supplies each
    class's own growth-RATE modifiers (see load_class_growth_lookup) -
    both to nudge which class gets recommended at each tier (see
    recommend_path's class_growth_lookup) and to the stat projection
    itself. Omit for the old flat-character-growth-only behavior.

    class_base_stats_df (data/class_base_stats.csv), if given, supplies
    each class's own base-stat FLOOR (see load_class_base_stats_lookup) -
    applied at every tier the path actually certifies into, not just the
    final one (see expected_stats_along_path). Omit for the old
    floor-free behavior.

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
    join level rather than assuming everyone starts from level 1.
    character_relics_df (data/character_relics.csv), if given, supplies
    the character's own Hero's Relic weapon type(s) (see
    load_character_relic_lookup) and nudges path scoring toward classes
    that use them (see recommend_path's character_relic_weapon_types /
    relic_affinity_bonus). All these are optional and default to
    None/False, which disables the corresponding feature entirely -
    existing callers that don't pass them keep getting today's behavior
    (join_level 1, no DLC classes, no relic affinity).
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

    class_growth_lookup = load_class_growth_lookup(class_growth_df)
    class_base_stats_lookup = load_class_base_stats_lookup(class_base_stats_df)
    character_relic_lookup = load_character_relic_lookup(character_relics_df)
    character_relic_weapon_types = character_relic_lookup.get(character_name)

    path = recommend_path(
        stat_boosts_df, used_role, target_level=effective_target_level,
        character_name=character_name,
        eligibility_lookup=eligibility_lookup,
        character_gender=character_gender,
        weapon_req_lookup=weapon_req_lookup,
        character_proficiency=character_proficiency,
        include_dlc_classes=include_dlc_classes,
        class_growth_lookup=class_growth_lookup,
        character_relic_weapon_types=character_relic_weapon_types,
    )
    final_class_name = path[-1]["class"] if path else None

    final_stats = None
    if final_class_name:
        final_boost_row = stat_boosts_df[stat_boosts_df["name"] == final_class_name].iloc[0]
        final_stats = expected_stats_along_path(
            path, base_stats_at_join, growth_row, final_boost_row, class_growth_lookup,
            effective_target_level, start_level=join_level,
            class_base_stats_lookup=class_base_stats_lookup,
        )

    weapon_switch_warning = None
    flagged_steps = [step for step in path if step.get("weapon_switch_warning")]
    if flagged_steps:
        names = ", ".join(f"{step['tier']} ({step['class']})" for step in flagged_steps)
        weapon_switch_warning = (
            f"This path asks {character_name} to pick up a weapon type they've never trained - "
            f"at {names} - which is a slow, costly switch in practice, not the free one the stat "
            f"numbers alone would suggest."
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
        "weapon_switch_warning": weapon_switch_warning,
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
    class_growth_df = pd.read_csv(DATA_DIR / "class_growth_rates.csv")
    class_base_stats_df = pd.read_csv(DATA_DIR / "class_base_stats.csv")
    character_relics_df = pd.read_csv(DATA_DIR / "character_relics.csv")

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
        class_growth_df=class_growth_df, class_base_stats_df=class_base_stats_df,
        character_relics_df=character_relics_df,
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
        if result["weapon_switch_warning"]:
            print(f"⚠ {result['weapon_switch_warning']}")

    if result["eligible_unique_classes"]:
        print("Also eligible for these Unique classes:")
        for option in result["eligible_unique_classes"]:
            print(f"  {option['class']} (fit score {option['score']})")


if __name__ == "__main__":
    main()