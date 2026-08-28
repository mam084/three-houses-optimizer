"""
optimizer/requirements.py

Certification-exam weapon/skill-rank requirements: formatting them for
display, scoring bonuses/penalties for how a class's requirement lines up
with a character's own proficiency/relic/path-so-far, and classifying a
unique class's weapon category from its stat data (since unique classes
have no certification-exam row of their own to read a category from).
"""
import pandas as pd

from .constants import (
    HIGH_CERTIFICATION_RANKS,
    MOUNT_ARMOR_SKILLS,
    RANK_INDEX,
    RELIC_AFFINITY_BONUS,
    ROLE_REQUIRED_WEAPON_CATEGORY,
    WEAPON_PROFICIENCY_BONUS,
    WEAPON_SWITCH_PENALTY,
)



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



def path_weapon_switch_warning(
    character_name: str,
    selected_steps: list[dict],
    weapon_req_lookup: dict | None,
    character_proficiency: set | None = None,
) -> str | None:
    """
    The same "this path asks <character> to pick up a weapon type never
    trained" summary recommend_for_character's own "weapon_switch_warning"
    key produces, but recomputed from an arbitrary ACTUAL path
    (selected_steps - a list of {"tier", "class"} dicts, in path order) -
    e.g. after "mix and match" overrides - rather than only ever reflecting
    the tool's own originally-recommended path.

    This is the fix for "if a warning is displayed and I change the class
    so there is no longer a class warning, the overall warning is still
    displayed": the recommended-path warning was computed once, before the
    user could override anything, and never recomputed afterward - so
    clearing the flagged step via mix-and-match left the stale summary
    banner on screen. Callers (see app.py's render_character_tab) should
    call this AFTER mix-and-match selections are known, using the actually-
    selected path, and use ITS return value for the summary banner instead
    of recommend_for_character's original one.

    Threads accumulated_skills forward tier-by-tier exactly like
    recommend_path/weapon_switch_penalty do, seeded from
    character_proficiency. Returns None if no step in the actual path is
    flagged.
    """
    accumulated_skills = set(character_proficiency or [])
    flagged = []
    for step in selected_steps:
        class_name = step["class"]
        if weapon_switch_penalty(class_name, weapon_req_lookup, accumulated_skills, character_proficiency) > 0:
            flagged.append(f"{step['tier']} ({class_name})")
        info = weapon_req_lookup.get(class_name) if weapon_req_lookup else None
        if info:
            accumulated_skills |= {skill for skill, _ in info["requirements"]}
    if not flagged:
        return None
    names = ", ".join(flagged)
    return (
        f"This path asks {character_name} to pick up a weapon type they've never trained - "
        f"at {names} - which is a slow, costly switch in practice, not the free one the stat "
        f"numbers alone would suggest."
    )



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



def combined_requirements_for_classes(
    class_names: list[str], weapon_req_lookup: dict | None,
) -> list[tuple[str, str]]:
    """
    Merge every class's own certification requirement (see
    load_weapon_requirements_lookup) across a character's WHOLE path
    (class_names, in path order - e.g. ["Paladin", "War Master"]) into one
    skill -> highest-required-rank mapping, e.g. Paladin's own "Lance C and
    Riding B" plus War Master's own "Axe A and Brawling A" merges into
    [("Axe","A"), ("Brawling","A"), ("Lance","C"), ("Riding","B")] - "what
    skill ranks will this character actually need across the whole path
    they're taking," not just the final tier's own requirement, since a
    real player has to clear every earlier tier's requirement on the way
    there too. If the same skill is asked for at two different tiers (rare,
    but possible), the higher of the two ranks wins (see RANK_INDEX) -
    whichever rank was reached first is still held once the character moves
    on, so the higher ask is the one that actually matters for "can this
    character finish this path."

    Classes with no requirement on file (Unique/story classes locked out of
    class_weapon_requirements.csv entirely - see NO_CERTIFICATION_CLASSES)
    are silently skipped, not treated as if they had a "no requirement at
    all" entry that would somehow blank out an earlier tier's real one.

    Returns a list of (skill, rank) tuples sorted by skill name for a
    stable, readable display order - empty if no class in class_names has
    any requirement data (weapon_req_lookup is empty/None, or the whole
    path is unique/story classes).
    """
    if not weapon_req_lookup:
        return []
    best: dict[str, str] = {}
    for class_name in class_names:
        info = weapon_req_lookup.get(class_name)
        if info is None:
            continue
        for skill, rank in info["requirements"]:
            if skill not in best or RANK_INDEX.get(rank, -1) > RANK_INDEX.get(best[skill], -1):
                best[skill] = rank
    return sorted(best.items())



def format_combined_requirements(class_names: list[str], weapon_req_lookup: dict | None) -> str | None:
    """
    Human-readable combined-path requirement, e.g. "Axe A, Brawling A,
    Lance C, Riding B" (see combined_requirements_for_classes) - the
    whole-path analogue of format_requirement's single-class string. None
    if the path has no requirement data at all to show.
    """
    pairs = combined_requirements_for_classes(class_names, weapon_req_lookup)
    if not pairs:
        return None
    return ", ".join(f"{skill} {rank}" for skill, rank in pairs)



def unique_class_weapon_category(
    class_name: str, stat_boosts_df: pd.DataFrame, class_growth_lookup: dict | None,
) -> str | None:
    """
    Approximate "physical" vs "hybrid" weapon_category for a
    UNIQUE_STORY_CLASS_TIER class, derived from the class's own Mag stat
    boost (data/class_stat_boosts.csv) and Mag growth-rate modifier
    (data/class_growth_rates.csv) rather than a fresh hand-curated table -
    these classes are excluded from class_weapon_requirements.csv entirely
    (see NO_CERTIFICATION_CLASSES), so they have no certification-based
    weapon_category the ordinary way, but the stat data already answers the
    practical question this project's weapon_category concept exists for:
    does this class actually grow a character's magic.

    Checked against the real data for every UNIQUE_STORY_CLASS_TIER class:
    Armored Lord/Emperor (Edelgard), High Lord/Great Lord (Dimitri), and
    Wyvern Master/Barbarossa (Claude) all show a flat 0 Mag boost AND 0 Mag
    growth modifier - "physical" - while Enlightened One (Byleth) shows a
    real +3 Mag boost and +10% Mag growth, comparable to its own Str line -
    "hybrid," matching Byleth's own genuinely dual-focus canon class. This
    is the fix behind "Dimitri/Claude's unique class gets suggested for a
    magic-attacking role even though it has no magic access" - see
    recommend_path, which uses this to decide whether an incompatible
    unique class should be exempted from the role's narrowing filters and
    get UNIQUE_CLASS_SCORE_BONUS, or scored (and filtered) like any other
    candidate.

    Returns None (unknown - callers should not gate on an unknown) if
    class_name isn't in stat_boosts_df at all.
    """
    rows = stat_boosts_df[stat_boosts_df["name"] == class_name]
    if rows.empty:
        return None
    boost_row = rows.iloc[0]
    mag_boost = float(boost_row["Mag"]) if "Mag" in boost_row.index else 0.0
    mag_growth = 0.0
    if class_growth_lookup:
        mag_growth = float(class_growth_lookup.get(class_name, {}).get("Mag", 0) or 0)
    return "hybrid" if (mag_boost > 0 or mag_growth > 0) else "physical"



def role_compatible_with_weapon_category(role_name: str, category: str | None) -> bool:
    """
    Whether a class of the given weapon_category ("physical"/"magic"/
    "hybrid", or None if unknown) can meaningfully serve role_name, per
    ROLE_REQUIRED_WEAPON_CATEGORY. A role absent from that dict, or an
    unknown category, is always compatible (no gate) - see that constant's
    docstring for why only some roles are gated at all.
    """
    required = ROLE_REQUIRED_WEAPON_CATEGORY.get(role_name)
    if required is None or category is None:
        return True
    return category in required
