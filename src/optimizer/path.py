"""
optimizer/path.py

Class-path selection: which candidate classes are even in play at a tier
(eligibility, DLC merge, unique-class splice, role-appropriate filters),
how a candidate is scored, and the public recommend_path/
list_eligible_classes_at_tier/eligible_unique_classes entry points that
tie eligibility.py/roles.py/requirements.py together into one pick per
tier.
"""
import pandas as pd

from .constants import (
    DLC_CLASS_MERGE_TIER,
    DLC_CLASS_TIER,
    GROWTH_RATE_SCORE_WEIGHT,
    ROLE_PROFILES,
    TIER_ORDER,
    UNIQUE_CLASS_SCORE_BONUS,
    reachable_tiers,
)
from .eligibility import eligible_unique_story_class_by_tier, is_class_eligible
from .requirements import (
    format_requirement,
    relic_affinity_bonus,
    role_compatible_with_weapon_category,
    unique_class_weapon_category,
    weapon_growth_bonus,
    weapon_switch_penalty,
)
from .roles import primary_stats_for_role, score_class_for_role, score_growth_for_role



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

        # The spliced-in unique class (if any) is normally exempted from
        # the two narrowing filters below: restrict_to_primary_relevant
        # would otherwise drop it outright whenever its stat line doesn't
        # touch the role's primary stat at all (e.g. Armored Lord's 0 Str
        # would get it dropped for a Physical Attacker role, before the
        # scoring bonus even gets a chance to weigh it) - it should reach
        # scoring, where UNIQUE_CLASS_SCORE_BONUS decides whether it wins.
        #
        # But that exemption is itself gated on the unique class actually
        # being ABLE to serve this role at all (see
        # unique_class_weapon_category/role_compatible_with_weapon_category):
        # a lord's Master-tier unique class with zero Mag boost and zero
        # Mag growth (Great Lord, Emperor, Barbarossa - every lord line
        # except Byleth's own hybrid Enlightened One) has no business being
        # exempted from the Magic Attacker filters and force-scored with a
        # +8 bonus regardless of fit - that was the reported "Dimitri/
        # Claude's unique class gets suggested for a magic role despite
        # having no magic access" bug. An incompatible unique class still
        # gets ADDED to the tier's candidate pool (so mix-and-match can
        # still surface it, and it can still win an honest tie the normal
        # filters don't resolve), it just doesn't skip the filters or earn
        # the bonus.
        unique_gets_bonus = unique_class_name is not None and role_compatible_with_weapon_category(
            role_name, unique_class_weapon_category(unique_class_name, stat_boosts_df, class_growth_lookup)
        )
        if unique_gets_bonus:
            is_unique_row = tier_classes["name"] == unique_class_name
        else:
            is_unique_row = pd.Series(False, index=tier_classes.index)
        unique_row_df = tier_classes[is_unique_row]
        rest = tier_classes[~is_unique_row]

        rest = restrict_support_to_magic_classes(rest, role_name, weapon_req_lookup or {})
        rest = apply_weapon_affinity_fallback(
            rest, role_name, role_weights, weapon_req_lookup or {}, character_proficiency
        )
        rest = restrict_to_primary_relevant(rest, role_weights)
        tier_classes = pd.concat([rest, unique_row_df]) if not unique_row_df.empty else rest

        if tier_classes.empty:
            continue

        def score_row(
            row, unique_class_name=unique_class_name, unique_gets_bonus=unique_gets_bonus,
            accumulated_skills=frozenset(accumulated_skills),
        ):
            score = score_class_for_role(row, role_weights)
            score += weapon_growth_bonus(row["name"], weapon_req_lookup, character_proficiency)
            score += relic_affinity_bonus(row["name"], weapon_req_lookup, character_relic_weapon_types)
            if class_growth_lookup:
                growth_mod = class_growth_lookup.get(row["name"], {})
                score += score_growth_for_role(growth_mod, role_weights) * GROWTH_RATE_SCORE_WEIGHT
            score -= weapon_switch_penalty(row["name"], weapon_req_lookup, accumulated_skills, character_proficiency)
            if unique_gets_bonus and row["name"] == unique_class_name:
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
