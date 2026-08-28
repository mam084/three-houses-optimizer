"""
optimizer/recommend.py

Top-level orchestration: recommend_for_character (auto-detect or target a
role, build the path, project stats, assemble the full result dict) plus
the CLI entry point (python -m src.optimizer) - the module that actually
wires data_loading.py/roles.py/path.py/projection.py together for a
caller who just wants "the recommendation for this character."
"""
import pandas as pd

from .constants import DATA_DIR, ROLE_PROFILES, TIER_LEVEL_REQUIREMENTS
from .data_loading import (
    load_character_proficiency_lookup,
    load_character_relic_lookup,
    load_class_base_stats_lookup,
    load_class_growth_lookup,
    load_eligibility_lookup,
    load_starting_level_lookup,
    load_weapon_requirements_lookup,
)
from .path import eligible_unique_classes, recommend_path
from .projection import base_stats_at_join_level, expected_stats_along_path
from .roles import compute_roster_stat_stats, detect_natural_role



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
    character_gender_override: str | None = None,
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

    character_gender_override, if given, replaces whatever character_gender_df
    itself says for this character - the one real use case is the
    Protagonist: Byleth's gender is recorded as "Any" in
    data/character_gender.csv (a genuine, permanent fact - Byleth has no
    fixed gender the way every other character does), which correctly
    means Byleth is never gender-locked OUT of anything by that CSV alone.
    But in the actual game the player DOES pick Byleth's gender at the very
    start, and that choice is exactly as real a gender-lock determinant for
    Byleth as any other character's fixed gender is for them - a
    Male-Byleth playthrough genuinely cannot take Falcon Knight, and a
    Female-Byleth playthrough genuinely cannot take War Master, the same
    way any other character can't cross a gender-locked class. Without this
    override, is_class_eligible's "Any" passthrough meant BOTH were always
    offered to Byleth regardless of which gender the player actually chose -
    the reported "War Master/Falcon Knight should depend on Byleth's
    gender" bug. Passing the app's own Byleth-gender selection here
    (app.py's byleth_gender, translated to "Male"/"Female") makes Byleth's
    gender-locked-class eligibility behave exactly like everyone else's -
    every other character's character_gender_override is expected to stay
    None, since character_gender_df already has their real, fixed gender.
    """
    base_row = base_stats_df[base_stats_df["name"] == character_name].iloc[0]
    growth_row = growth_rates_df[growth_rates_df["name"] == character_name].iloc[0]

    eligibility_lookup = load_eligibility_lookup(eligibility_df) if eligibility_df is not None else None
    character_gender = None
    if character_gender_df is not None:
        gender_row = character_gender_df[character_gender_df["name"] == character_name]
        if not gender_row.empty:
            character_gender = gender_row.iloc[0]["gender"]
    if character_gender_override is not None:
        character_gender = character_gender_override

    weapon_req_lookup = load_weapon_requirements_lookup(weapon_req_df) if weapon_req_df is not None else None
    character_proficiency_lookup = load_character_proficiency_lookup(character_weapon_talent_df) \
        if character_weapon_talent_df is not None else {}
    character_proficiency = character_proficiency_lookup.get(character_name)

    character_relic_lookup = load_character_relic_lookup(character_relics_df)
    character_relic_weapon_types = character_relic_lookup.get(character_name)

    roster_means, roster_stds = compute_roster_stat_stats(growth_rates_df)
    detected_role, detection_score = detect_natural_role(
        growth_row, roster_means, roster_stds,
        character_proficiency=character_proficiency,
        character_relic_weapon_types=character_relic_weapon_types,
    )
    used_role = role_name or detected_role

    starting_level_lookup = load_starting_level_lookup(starting_level_df)
    join_level = starting_level_lookup.get(character_name, 1)
    effective_target_level = max(target_level, join_level)
    base_stats_at_join = base_stats_at_join_level(base_row, growth_row, join_level)

    class_growth_lookup = load_class_growth_lookup(class_growth_df)
    class_base_stats_lookup = load_class_base_stats_lookup(class_base_stats_df)

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
