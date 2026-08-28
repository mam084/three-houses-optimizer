"""
optimizer/constants.py

Tunable weights/bonuses/penalties, tier and rank tables, and role-profile
definitions shared across the optimizer package - the pure data half of
the recommendation engine, with no pandas/numpy logic of its own beyond
reachable_tiers (kept here since it operates directly on
TIER_LEVEL_REQUIREMENTS/TIER_ORDER and nothing else).
"""
from pathlib import Path


# src/optimizer/constants.py -> .parent (src/optimizer/) -> .parent (src/) ->
# .parent (repo root) / "data" - one level deeper than the old single-file
# src/optimizer.py needed, since this module now lives inside a package
# subdirectory rather than directly under src/.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


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
NO_CERTIFICATION_CLASSES = set(UNIQUE_STORY_CLASS_TIER.keys()) | {"Lord", "Noble", "Commoner", "Dancer", "Death Knight"}


# Personal DLC-exclusive classes: locked to exactly one character in
# class_eligibility.csv (Death Knight -> Jeritza is currently the only one -
# Dark Flier/Valkyrie are DLC-exclusive too but only gender-locked, not
# character-locked, so they aren't in this dict), but - unlike the
# UNIQUE_STORY_CLASS_TIER classes above - NOT spliced into a fixed rung of
# that character's Beginner->Master ladder, since Death Knight isn't a
# replacement for one of Jeritza's own tiers the way Armored Lord is for
# Edelgard's Advanced tier; it's just one candidate among the usual
# DLC_CLASS_MERGE_TIER pool once DLC classes are opted into, competing for
# that tier's slot on stats like any other class. Maps class_name -> the
# one character it's locked to, and which of ROLE_PROFILES' roles that
# class's own stat line actually reads as a fit for - Death Knight's boosts
# (HP+5/Str+3/Def+2/Res+3, negative-leaning Spd growth) are a real, if not
# Fortress-Knight-beating, bruiser line: "physical" in the non-magic sense
# (Physical Attacker, Tank), not a dedicated glass-cannon striker or caster.
# See PERSONAL_DLC_CLASS_ROLE_BONUS/personal_dlc_class_role_bonus (used by
# recommend_path) for how this is actually scored.
PERSONAL_DLC_CLASS_ROLES = {
    "Death Knight": {"character": "Jeritza", "roles": {"Physical Attacker", "Tank"}},
}

# Small score bonus for a personal DLC-exclusive class (see
# PERSONAL_DLC_CLASS_ROLES) scored for one of its own suited roles for its
# one eligible character - the same small, additive, tie-breaking-not-
# overriding precedent as WEAPON_PROFICIENCY_BONUS/RELIC_AFFINITY_BONUS
# below, deliberately far short of UNIQUE_CLASS_SCORE_BONUS above: Death
# Knight already out-scores every other Advanced-tier class on raw stats
# alone for Physical Attacker (so this bonus is mostly insurance there), but
# for Tank it trails Fortress Knight's dedicated +10 Def boost by a wide
# margin that a "slight" bonus deliberately doesn't try to close - Fortress
# Knight is a legitimately better Tank pick, DLC or not, and this constant
# is sized to refine a close call, not force Death Knight into every
# physical-flavored role regardless of fit. is_class_eligible (via
# class_eligibility.csv) is what actually keeps this class off every other
# character's list; this only matters once it's already a legitimate,
# eligible candidate.
PERSONAL_DLC_CLASS_ROLE_BONUS = 1.5

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


# Certification-exam skill ranks, worst to best, as they actually appear in
# data/class_weapon_requirements.csv (D, C, C+, B, B+, A) - E/E+/D+/A+/S are
# included too since character_weapon_talent.csv's "top_rank" column and the
# game's own rank scale use them, even though no CLASS requirement currently
# asks for one. Used to compare two ranks for the same skill (see
# combined_requirements_for_classes) - a plain string compare would put
# "B+" before "B" alphabetically-wrong-ways and wouldn't order "C+" between
# "C" and "B" at all.
RANK_ORDER = ["E", "E+", "D", "D+", "C", "C+", "B", "B+", "A", "A+", "S"]

RANK_INDEX = {rank: i for i, rank in enumerate(RANK_ORDER)}


# A role's real-game "does this class let the character actually contribute
# to this role" check, used to gate which unique/story class gets
# UNIQUE_CLASS_SCORE_BONUS (see recommend_path) - the same
# physical/magic/hybrid vocabulary class_weapon_requirements.csv's own
# weapon_category column uses. Tank and Speed/Precision aren't listed
# because, unlike Physical/Magic Attacker and Support, this dataset's Tank
# and Speed/Precision candidate pools were never observed to include a
# unique class whose weapon category actually conflicts with the role (a
# tank or a speedster doesn't need a specific weapon type the way an
# Attacker or a healer does) - a role absent from this dict is simply never
# gated, same as ROLE_REQUIRED_WEAPON_CATEGORY's own "role doesn't care"
# no-op case.
ROLE_REQUIRED_WEAPON_CATEGORY = {
    "Physical Attacker": {"physical", "hybrid"},
    "Magic Attacker": {"magic", "hybrid"},
    "Support": {"magic", "hybrid"},
}


# Small, bounded nudge (on a cosine-similarity scale of -1..1) toward a role
# that matches a character's own starting weapon proficiency or Hero's
# Relic weapon type - see natural_role_affinity_bonus. Deliberately much
# smaller than a real growth-rate-driven similarity gap (usually several
# tenths) so it only breaks a genuinely close natural-role call, or nudges
# an otherwise-ambiguous character toward the role their own equipment
# already points at, without overriding a role a character's growth rates
# clearly favor. This was previously nonexistent - natural-role
# auto-detection looked at growth rates ALONE, with proficiency/relics only
# ever factoring into which CLASS gets picked within an already-decided
# role (see weapon_growth_bonus/relic_affinity_bonus) - "how much are a
# character's natural proficiencies being taken into account for their
# default role" and "Lorenz's relic only supports mages, but he's still
# suggested as a Tank" are both this same gap: a character's own gear
# affinity had no voice at all in which role got auto-detected.
NATURAL_ROLE_AFFINITY_WEIGHT = 0.05


# Which of the two broad weapon families (data/character_weapon_talent.csv,
# data/character_relics.csv) each auto-detectable role is "about," for
# natural_role_affinity_bonus. Reason/Faith are the only magic weapon types
# in this dataset; every other skill type (Sword/Axe/Lance/Bow/Brawling/
# Riding/Flying/Heavy Armour/Authority) is physical. Tank and Speed/
# Precision are deliberately left unmapped (None) here too - same reasoning
# as ROLE_REQUIRED_WEAPON_CATEGORY above; a role with no entry gets no
# affinity nudge either direction, rather than an invented one.
MAGIC_WEAPON_TYPES = {"Reason", "Faith"}

ROLE_NATURAL_WEAPON_CATEGORY = {
    "Physical Attacker": "physical",
    "Magic Attacker": "magic",
    "Support": "magic",
}



def reachable_tiers(target_level: int, tiers: list[str] = TIER_ORDER) -> list[str]:
    """
    Given a target level, return the subset of tiers (in order) whose level
    requirement is met at that level. E.g. target_level=15 with the default
    tier order returns ["Beginner", "Intermediate"], since Advanced needs
    level 20.
    """
    return [tier for tier in tiers if TIER_LEVEL_REQUIREMENTS.get(tier, 0) <= target_level]



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
