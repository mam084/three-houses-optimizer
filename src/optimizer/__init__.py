"""
optimizer package (src/optimizer/__init__.py)

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

from .constants import (
    DATA_DIR,
    DLC_CLASS_MERGE_TIER,
    DLC_CLASS_TIER,
    GROWTH_RATE_SCORE_WEIGHT,
    HIGH_CERTIFICATION_RANKS,
    MAGIC_WEAPON_TYPES,
    MOUNT_ARMOR_SKILLS,
    NATURAL_ROLE_AFFINITY_WEIGHT,
    NO_CERTIFICATION_CLASSES,
    PERSONAL_DLC_CLASS_ROLE_BONUS,
    PERSONAL_DLC_CLASS_ROLES,
    RANK_INDEX,
    RANK_ORDER,
    RELIC_AFFINITY_BONUS,
    ROLE_NATURAL_WEAPON_CATEGORY,
    ROLE_PROFILES,
    ROLE_REQUIRED_WEAPON_CATEGORY,
    STAT_COLS,
    TIER_LEVEL_REQUIREMENTS,
    TIER_ORDER,
    UNIQUE_CLASS_SCORE_BONUS,
    UNIQUE_STORY_CLASS_TIER,
    WEAPON_PROFICIENCY_BONUS,
    WEAPON_SWITCH_PENALTY,
    reachable_tiers,
)

from .data_loading import (
    load_character_proficiency_lookup,
    load_character_relic_lookup,
    load_class_base_stats_lookup,
    load_class_growth_lookup,
    load_eligibility_lookup,
    load_starting_level_lookup,
    load_weapon_requirements_lookup,
)

from .eligibility import (
    eligible_unique_story_class_by_tier,
    is_class_eligible,
)

from .roles import (
    compute_roster_stat_stats,
    cosine_similarity,
    detect_natural_role,
    natural_role_affinity_bonus,
    primary_stats_for_role,
    role_relevant_stat_deltas,
    role_to_vector,
    score_all_roles,
    score_class_for_role,
    score_growth_for_role,
)

from .requirements import (
    combined_requirements_for_classes,
    format_combined_requirements,
    format_requirement,
    path_weapon_switch_warning,
    personal_dlc_class_role_bonus,
    relic_affinity_bonus,
    role_compatible_with_weapon_category,
    unique_class_weapon_category,
    weapon_growth_bonus,
    weapon_switch_penalty,
)

from .projection import (
    apply_class_base_stat_floor,
    base_stats_at_join_level,
    expected_stats_along_path,
    expected_stats_at_level,
    path_level_bands,
    stats_for_class_at_level,
    stats_for_selected_path,
)

from .path import (
    apply_weapon_affinity_fallback,
    eligible_unique_classes,
    explain_pick,
    list_eligible_classes_at_tier,
    recommend_path,
    restrict_support_to_magic_classes,
    restrict_to_primary_relevant,
)

from .recommend import (
    main,
    recommend_for_character,
)

__all__ = [
    "DATA_DIR",
    "DLC_CLASS_MERGE_TIER",
    "DLC_CLASS_TIER",
    "GROWTH_RATE_SCORE_WEIGHT",
    "HIGH_CERTIFICATION_RANKS",
    "MAGIC_WEAPON_TYPES",
    "MOUNT_ARMOR_SKILLS",
    "NATURAL_ROLE_AFFINITY_WEIGHT",
    "NO_CERTIFICATION_CLASSES",
    "PERSONAL_DLC_CLASS_ROLE_BONUS",
    "PERSONAL_DLC_CLASS_ROLES",
    "RANK_INDEX",
    "RANK_ORDER",
    "RELIC_AFFINITY_BONUS",
    "ROLE_NATURAL_WEAPON_CATEGORY",
    "ROLE_PROFILES",
    "ROLE_REQUIRED_WEAPON_CATEGORY",
    "STAT_COLS",
    "TIER_LEVEL_REQUIREMENTS",
    "TIER_ORDER",
    "UNIQUE_CLASS_SCORE_BONUS",
    "UNIQUE_STORY_CLASS_TIER",
    "WEAPON_PROFICIENCY_BONUS",
    "WEAPON_SWITCH_PENALTY",
    "apply_class_base_stat_floor",
    "apply_weapon_affinity_fallback",
    "base_stats_at_join_level",
    "combined_requirements_for_classes",
    "compute_roster_stat_stats",
    "cosine_similarity",
    "detect_natural_role",
    "eligible_unique_classes",
    "eligible_unique_story_class_by_tier",
    "expected_stats_along_path",
    "expected_stats_at_level",
    "explain_pick",
    "format_combined_requirements",
    "format_requirement",
    "is_class_eligible",
    "list_eligible_classes_at_tier",
    "load_character_proficiency_lookup",
    "load_character_relic_lookup",
    "load_class_base_stats_lookup",
    "load_class_growth_lookup",
    "load_eligibility_lookup",
    "load_starting_level_lookup",
    "load_weapon_requirements_lookup",
    "main",
    "natural_role_affinity_bonus",
    "path_level_bands",
    "path_weapon_switch_warning",
    "personal_dlc_class_role_bonus",
    "primary_stats_for_role",
    "reachable_tiers",
    "recommend_for_character",
    "recommend_path",
    "relic_affinity_bonus",
    "restrict_support_to_magic_classes",
    "restrict_to_primary_relevant",
    "role_compatible_with_weapon_category",
    "role_relevant_stat_deltas",
    "role_to_vector",
    "score_all_roles",
    "score_class_for_role",
    "score_growth_for_role",
    "stats_for_class_at_level",
    "stats_for_selected_path",
    "unique_class_weapon_category",
    "weapon_growth_bonus",
    "weapon_switch_penalty",
]
