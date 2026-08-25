"""
tests/test_optimizer.py

Unit tests for src/optimizer.py, using Python's stdlib unittest (no pytest
available in every environment this project's been developed in - see
README's Testing section). Run with:

    python -m unittest discover -s tests -v

Loads the real data/ CSVs once at import time rather than fabricated
fixtures, since a lot of what's being tested here (the Edelgard/Gremory
regression, the Beginner-tier weapon-affinity fallback, eligibility rules)
is specifically about this project's real data producing sane results,
not just the algorithm being internally consistent.
"""

import inspect
import unittest
from pathlib import Path

import pandas as pd

from src.optimizer import (
    DLC_CLASS_MERGE_TIER,
    DLC_CLASS_TIER,
    ROLE_PROFILES,
    STAT_COLS,
    TIER_LEVEL_REQUIREMENTS,
    TIER_ORDER,
    UNIQUE_CLASS_SCORE_BONUS,
    UNIQUE_STORY_CLASS_TIER,
    apply_weapon_affinity_fallback,
    base_stats_at_join_level,
    compute_roster_stat_stats,
    detect_natural_role,
    eligible_unique_story_class_by_tier,
    format_requirement,
    is_class_eligible,
    list_eligible_classes_at_tier,
    load_character_proficiency_lookup,
    load_eligibility_lookup,
    load_starting_level_lookup,
    load_weapon_requirements_lookup,
    reachable_tiers,
    recommend_for_character,
    recommend_path,
    restrict_support_to_magic_classes,
    stats_for_class_at_level,
    weapon_growth_bonus,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Loaded once at import time and shared read-only across every test below -
# deliberately module-level rather than a TestCase mixin's setUpClass:
# mixing a plain class into `class Foo(unittest.TestCase, Mixin)` puts
# TestCase ahead of Mixin in the MRO, and TestCase already defines
# setUpClass (as a no-op) - so `Mixin.setUpClass` silently never runs via
# inheritance. Module-level globals sidestep that trap entirely.
BASE_STATS_DF = pd.read_csv(DATA_DIR / "character_base_stats.csv")
GROWTH_RATES_DF = pd.read_csv(DATA_DIR / "character_growth_rates.csv")
STAT_BOOSTS_DF = pd.read_csv(DATA_DIR / "class_stat_boosts.csv")
ELIGIBILITY_DF = pd.read_csv(DATA_DIR / "class_eligibility.csv")
CHARACTER_GENDER_DF = pd.read_csv(DATA_DIR / "character_gender.csv")
WEAPON_REQ_DF = pd.read_csv(DATA_DIR / "class_weapon_requirements.csv")
CHARACTER_WEAPON_TALENT_DF = pd.read_csv(DATA_DIR / "character_weapon_talent.csv")
PLAYABLE_NAMES = sorted(n for n in BASE_STATS_DF["name"] if "(NPC)" not in n)

STARTING_LEVEL_DF = pd.read_csv(DATA_DIR / "character_starting_level.csv")
WEAPON_REQ_LOOKUP = load_weapon_requirements_lookup(WEAPON_REQ_DF)
PROFICIENCY_LOOKUP = load_character_proficiency_lookup(CHARACTER_WEAPON_TALENT_DF)
ELIGIBILITY_LOOKUP = load_eligibility_lookup(ELIGIBILITY_DF)
STARTING_LEVEL_LOOKUP = load_starting_level_lookup(STARTING_LEVEL_DF)
ROSTER_MEANS, ROSTER_STDS = compute_roster_stat_stats(GROWTH_RATES_DF)


def detect(name: str):
    row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == name].iloc[0]
    return detect_natural_role(row, ROSTER_MEANS, ROSTER_STDS)


class TestReachableTiers(unittest.TestCase):
    def test_high_level_reaches_every_tier(self):
        self.assertEqual(reachable_tiers(100), TIER_ORDER)

    def test_level_15_stops_after_intermediate(self):
        self.assertEqual(reachable_tiers(15), ["Beginner", "Intermediate"])

    def test_level_5_only_beginner(self):
        self.assertEqual(reachable_tiers(5), ["Beginner"])

    def test_level_4_nothing_reachable(self):
        self.assertEqual(reachable_tiers(4), [])

    def test_level_30_all_tiers(self):
        self.assertEqual(reachable_tiers(30), TIER_ORDER)

    def test_matches_tier_level_requirements_table(self):
        for tier, required_level in TIER_LEVEL_REQUIREMENTS.items():
            self.assertIn(tier, reachable_tiers(required_level))
            self.assertNotIn(tier, reachable_tiers(required_level - 1))


class TestEligibility(unittest.TestCase):
    def test_unrestricted_class_is_eligible_for_anyone(self):
        self.assertTrue(is_class_eligible("Bernadetta", "Mercenary", ELIGIBILITY_LOOKUP))
        self.assertTrue(is_class_eligible("Dedue", "Mercenary", ELIGIBILITY_LOOKUP))

    def test_character_locked_class_blocks_others(self):
        self.assertTrue(is_class_eligible("Edelgard", "Emperor", ELIGIBILITY_LOOKUP))
        self.assertFalse(is_class_eligible("Dimitri", "Emperor", ELIGIBILITY_LOOKUP))
        self.assertFalse(is_class_eligible("Bernadetta", "Emperor", ELIGIBILITY_LOOKUP))

    def test_gender_locked_class(self):
        self.assertTrue(is_class_eligible("Bernadetta", "Pegasus Knight", ELIGIBILITY_LOOKUP, "Female"))
        self.assertFalse(is_class_eligible("Caspar", "Pegasus Knight", ELIGIBILITY_LOOKUP, "Male"))

    def test_gender_locked_class_degrades_gracefully_without_gender_data(self):
        # No gender supplied at all -> don't block on the gender axis (see is_class_eligible docstring).
        self.assertTrue(is_class_eligible("Caspar", "Pegasus Knight", ELIGIBILITY_LOOKUP, None))

    def test_protagonist_gender_any_passes_every_gender_lock(self):
        self.assertTrue(is_class_eligible("Protagonist", "Pegasus Knight", ELIGIBILITY_LOOKUP, "Any"))
        self.assertTrue(is_class_eligible("Protagonist", "War Master", ELIGIBILITY_LOOKUP, "Any"))

    def test_lord_only_the_three_house_leaders(self):
        for leader in ("Edelgard", "Dimitri", "Claude"):
            self.assertTrue(is_class_eligible(leader, "Lord", ELIGIBILITY_LOOKUP))
        self.assertFalse(is_class_eligible("Hubert", "Lord", ELIGIBILITY_LOOKUP))


class TestWeaponRequirements(unittest.TestCase):
    def test_every_weapon_requirement_class_exists_in_stat_boosts(self):
        # class_weapon_requirements.csv should only ever describe real, certifiable classes.
        boost_class_names = set(STAT_BOOSTS_DF["name"])
        for class_name in WEAPON_REQ_LOOKUP:
            self.assertIn(class_name, boost_class_names, f"{class_name} not in class_stat_boosts.csv")

    def test_format_requirement_or_join(self):
        self.assertEqual(format_requirement("Fighter", WEAPON_REQ_LOOKUP), "Axe D or Bow D or Brawling D")

    def test_format_requirement_and_join(self):
        self.assertEqual(format_requirement("Hero", WEAPON_REQ_LOOKUP), "Sword B and Axe C")

    def test_format_requirement_unknown_class_returns_none(self):
        self.assertIsNone(format_requirement("Emperor", WEAPON_REQ_LOOKUP))  # Unique tier, no cert exam

    def test_every_character_has_proficiency_data(self):
        for name in PLAYABLE_NAMES:
            self.assertIn(name, PROFICIENCY_LOOKUP, f"{name} missing from character_weapon_talent.csv")
            self.assertTrue(PROFICIENCY_LOOKUP[name], f"{name} has an empty proficiency set")

    def test_beginner_magic_attacker_fallback_prefers_monk(self):
        # This is the mechanism that fixes "why Soldier for mages?" - see optimizer.py docstrings.
        tier_classes = STAT_BOOSTS_DF[STAT_BOOSTS_DF["tier"] == "Beginner"]
        tier_classes = tier_classes[~tier_classes["name"].str.contains(r"\(")]
        result = apply_weapon_affinity_fallback(
            tier_classes, "Magic Attacker", ROLE_PROFILES["Magic Attacker"], WEAPON_REQ_LOOKUP,
        )
        self.assertEqual(set(result["name"]), {"Monk"})

    def test_beginner_physical_attacker_fallback_is_a_noop(self):
        # Unlike Magic Attacker, Physical Attacker's primary stat (Str) IS boosted by a
        # Beginner class (Fighter +1 Str) - the fallback only ever activates when stat data
        # is fully uninformative (see its docstring), so here it should leave every candidate
        # untouched; narrowing to Str-relevant classes is restrict_to_primary_relevant's job,
        # exercised below via the full recommend_path pipeline instead.
        tier_classes = STAT_BOOSTS_DF[STAT_BOOSTS_DF["tier"] == "Beginner"]
        tier_classes = tier_classes[~tier_classes["name"].str.contains(r"\(")]
        result = apply_weapon_affinity_fallback(
            tier_classes, "Physical Attacker", ROLE_PROFILES["Physical Attacker"], WEAPON_REQ_LOOKUP,
        )
        self.assertEqual(set(result["name"]), set(tier_classes["name"]))

    def test_recommend_path_beginner_physical_attacker_never_picks_monk(self):
        path = recommend_path(STAT_BOOSTS_DF, "Physical Attacker", target_level=5)
        self.assertEqual(path[0]["class"], "Fighter")

    def test_recommend_path_beginner_magic_attacker_picks_monk(self):
        path = recommend_path(
            STAT_BOOSTS_DF, "Magic Attacker", target_level=5, weapon_req_lookup=WEAPON_REQ_LOOKUP,
        )
        self.assertEqual(path[0]["class"], "Monk")

    def test_fallback_is_a_noop_when_stat_data_is_informative(self):
        # At Intermediate tier, Mag is a real differentiator (Mage +1 Mag) - the fallback shouldn't touch it.
        tier_classes = STAT_BOOSTS_DF[STAT_BOOSTS_DF["tier"] == "Intermediate"]
        tier_classes = tier_classes[~tier_classes["name"].str.contains(r"\(")]
        result = apply_weapon_affinity_fallback(
            tier_classes, "Magic Attacker", ROLE_PROFILES["Magic Attacker"], WEAPON_REQ_LOOKUP,
        )
        self.assertEqual(len(result), len(tier_classes))


class TestRoleDetectionRegression(unittest.TestCase):
    """
    Regression coverage for the "Edelgard is a Gremory??" bug: Support's
    role-detection weighting used to include Cha at the same strength as
    Res, which let the three house leaders' inflated Cha growth (a
    leadership stat, not a healing one) outscore their own much more
    on-theme combat stats. See ROLE_PROFILES's comment in optimizer.py.
    """

    def test_support_profile_does_not_weight_charisma(self):
        self.assertNotIn("Cha", ROLE_PROFILES["Support"])

    def test_edelgard_detects_as_physical_not_support(self):
        role, _ = detect("Edelgard")
        self.assertEqual(role, "Physical Attacker")

    def test_edelgard_recommended_path_has_no_gremory(self):
        result = recommend_for_character(
            "Edelgard", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=30, eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
        )
        classes_in_path = [step["class"] for step in result["path"]]
        self.assertNotIn("Gremory", classes_in_path)
        self.assertEqual(result["auto_detected_role"], "Physical Attacker")

    def test_known_healer_archetypes_still_detect_as_support(self):
        # Guards against overcorrecting - dropping Cha shouldn't break characters who ARE healers.
        for name in ("Mercedes", "Flayn", "Marianne"):
            role, _ = detect(name)
            self.assertEqual(role, "Support", f"{name} should still auto-detect as Support")


class TestListEligibleClassesAtTier(unittest.TestCase):
    def test_returns_sorted_unique_classes(self):
        classes = list_eligible_classes_at_tier("Beginner", STAT_BOOSTS_DF)
        self.assertEqual(classes, sorted(classes))
        self.assertIn("Monk", classes)

    def test_excludes_story_variant_rows(self):
        classes = list_eligible_classes_at_tier("Intermediate", STAT_BOOSTS_DF)
        self.assertNotIn("Lord (Judith)", classes)

    def test_gender_locked_class_excluded_for_wrong_gender(self):
        classes = list_eligible_classes_at_tier(
            "Intermediate", STAT_BOOSTS_DF, character_name="Caspar",
            eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender="Male",
        )
        self.assertNotIn("Pegasus Knight", classes)


class TestStatsForClassAtLevel(unittest.TestCase):
    def test_matches_recommend_for_character_for_the_same_final_class(self):
        result = recommend_for_character(
            "Felix", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=30, eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
        )
        final_class = result["path"][-1]["class"]
        recomputed = stats_for_class_at_level(
            "Felix", final_class, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF, 30,
        )
        self.assertEqual(recomputed, result["expected_final_stats"])

    def test_unknown_class_returns_none(self):
        self.assertIsNone(stats_for_class_at_level(
            "Felix", "Not A Real Class", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF, 30,
        ))


class TestRecommendPathAndStats(unittest.TestCase):
    def test_recommend_path_directly_respects_target_level_gating(self):
        path = recommend_path(STAT_BOOSTS_DF, "Tank", target_level=15)
        self.assertEqual([step["tier"] for step in path], ["Beginner", "Intermediate"])

    def test_expected_final_stats_has_every_stat_column(self):
        result = recommend_for_character(
            "Dedue", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=30, eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
        )
        self.assertEqual(set(result["expected_final_stats"].keys()), set(STAT_COLS))


class TestDefaults(unittest.TestCase):
    def test_target_level_defaults_are_30(self):
        for func in (recommend_for_character,):
            self.assertEqual(
                inspect.signature(func).parameters["target_level"].default, 30,
                f"{func.__name__} should default target_level to 30",
            )


class TestFullRosterSweep(unittest.TestCase):
    """Every character, every role mode, every requirement dataset supplied - must never raise."""

    def test_sweep_no_exceptions(self):
        role_modes = [None] + list(ROLE_PROFILES.keys())
        for name in PLAYABLE_NAMES:
            for role in role_modes:
                for include_dlc_classes in (False, True):
                    with self.subTest(character=name, role=role, include_dlc_classes=include_dlc_classes):
                        result = recommend_for_character(
                            name, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
                            role_name=role, target_level=30,
                            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
                            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
                            starting_level_df=STARTING_LEVEL_DF, include_dlc_classes=include_dlc_classes,
                        )
                        self.assertIsInstance(result["path"], list)

    def test_support_role_never_lands_in_a_physical_only_class(self):
        # Every Support-role path step, across the whole roster, must be a class with real magic
        # (Reason/Faith) access - see restrict_support_to_magic_classes. This is the direct
        # regression test for "Mercedes' final tier is Falcon Knight for Support."
        weapon_req_lookup = WEAPON_REQ_LOOKUP
        for name in PLAYABLE_NAMES:
            with self.subTest(character=name):
                result = recommend_for_character(
                    name, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
                    role_name="Support", target_level=30,
                    eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
                    weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
                    starting_level_df=STARTING_LEVEL_DF,
                )
                for step in result["path"]:
                    info = weapon_req_lookup.get(step["class"])
                    if info is not None:
                        self.assertNotEqual(
                            info["weapon_category"], "physical",
                            f"{name}'s Support path picked {step['class']} at {step['tier']}, "
                            f"which has no magic access at all",
                        )

    def test_mercedes_support_final_class_is_magic_capable(self):
        # The exact reported case: Mercedes targeting Support should never end in Falcon Knight.
        result = recommend_for_character(
            "Mercedes", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Support", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF,
        )
        final_class = result["path"][-1]["class"]
        self.assertNotEqual(final_class, "Falcon Knight")
        self.assertIn(WEAPON_REQ_LOOKUP[final_class]["weapon_category"], ("magic", "hybrid"))


class TestRestrictSupportToMagicClasses(unittest.TestCase):
    def test_drops_physical_only_classes_for_support(self):
        master_tier = STAT_BOOSTS_DF[STAT_BOOSTS_DF["tier"] == "Master"]
        master_tier = master_tier[~master_tier["name"].str.contains(r"\(", regex=True)]
        filtered = restrict_support_to_magic_classes(master_tier, "Support", WEAPON_REQ_LOOKUP)
        self.assertNotIn("Falcon Knight", filtered["name"].tolist())
        self.assertIn("Gremory", filtered["name"].tolist())

    def test_noop_for_non_support_roles(self):
        master_tier = STAT_BOOSTS_DF[STAT_BOOSTS_DF["tier"] == "Master"]
        filtered = restrict_support_to_magic_classes(master_tier, "Physical Attacker", WEAPON_REQ_LOOKUP)
        self.assertEqual(len(filtered), len(master_tier))

    def test_noop_without_weapon_req_lookup(self):
        master_tier = STAT_BOOSTS_DF[STAT_BOOSTS_DF["tier"] == "Master"]
        filtered = restrict_support_to_magic_classes(master_tier, "Support", {})
        self.assertEqual(len(filtered), len(master_tier))


class TestUniqueClassSplicing(unittest.TestCase):
    def test_eligible_unique_story_class_by_tier_only_for_locked_characters(self):
        self.assertEqual(
            eligible_unique_story_class_by_tier("Edelgard", ELIGIBILITY_LOOKUP),
            {"Advanced": "Armored Lord", "Master": "Emperor"},
        )
        self.assertEqual(eligible_unique_story_class_by_tier("Bernadetta", ELIGIBILITY_LOOKUP), {})

    def test_edelgard_physical_attacker_path_uses_her_own_unique_classes(self):
        result = recommend_for_character(
            "Edelgard", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF,
        )
        by_tier = {step["tier"]: step["class"] for step in result["path"]}
        self.assertEqual(by_tier["Advanced"], "Armored Lord")
        self.assertEqual(by_tier["Master"], "Emperor")
        self.assertTrue(
            next(s for s in result["path"] if s["tier"] == "Master")["is_unique_class"]
        )

    def test_unique_classes_never_leak_to_ineligible_characters(self):
        for name in PLAYABLE_NAMES:
            if name in ("Protagonist", "Edelgard", "Dimitri", "Claude"):
                continue
            for role in ROLE_PROFILES:
                with self.subTest(character=name, role=role):
                    result = recommend_for_character(
                        name, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
                        role_name=role, target_level=30,
                        eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
                        weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
                        starting_level_df=STARTING_LEVEL_DF,
                    )
                    picked_classes = {step["class"] for step in result["path"]}
                    self.assertTrue(picked_classes.isdisjoint(UNIQUE_STORY_CLASS_TIER))

    def test_bonus_is_large_enough_to_win_a_negative_raw_score(self):
        # Armored Lord scores negative for Physical Attacker (its Spd boost is -3) - the additive
        # bonus, not a multiplier, is what has to carry it past Swordmaster/Brigand etc.
        self.assertGreater(UNIQUE_CLASS_SCORE_BONUS, 5.0)


class TestWeaponGrowthBonus(unittest.TestCase):
    def test_bonus_when_class_matches_character_proficiency(self):
        # Felix's own starting proficiency is Sword.
        bonus = weapon_growth_bonus("Myrmidon", WEAPON_REQ_LOOKUP, PROFICIENCY_LOOKUP["Felix"])
        self.assertGreater(bonus, 0.0)

    def test_no_bonus_when_no_overlap(self):
        bonus = weapon_growth_bonus("Monk", WEAPON_REQ_LOOKUP, PROFICIENCY_LOOKUP["Felix"])
        self.assertEqual(bonus, 0.0)

    def test_no_bonus_without_proficiency_data(self):
        self.assertEqual(weapon_growth_bonus("Myrmidon", WEAPON_REQ_LOOKUP, None), 0.0)

    def test_no_bonus_for_unknown_class(self):
        self.assertEqual(weapon_growth_bonus("Not A Real Class", WEAPON_REQ_LOOKUP, {"Sword"}), 0.0)


class TestJoinLevel(unittest.TestCase):
    def test_house_students_join_at_level_one(self):
        for name in ("Protagonist", "Edelgard", "Dimitri", "Claude", "Mercedes"):
            self.assertEqual(STARTING_LEVEL_LOOKUP[name], 1)

    def test_late_recruit_has_a_higher_join_level(self):
        self.assertGreater(STARTING_LEVEL_LOOKUP["Catherine"], 1)

    def test_base_stats_at_join_level_one_is_unchanged(self):
        base_row = BASE_STATS_DF[BASE_STATS_DF["name"] == "Mercedes"].iloc[0]
        growth_row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Mercedes"].iloc[0]
        result = base_stats_at_join_level(base_row, growth_row, 1)
        for stat in STAT_COLS:
            self.assertEqual(result[stat], round(float(base_row[stat]), 1))

    def test_base_stats_at_higher_join_level_adds_expected_growth(self):
        base_row = BASE_STATS_DF[BASE_STATS_DF["name"] == "Catherine"].iloc[0]
        growth_row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Catherine"].iloc[0]
        result = base_stats_at_join_level(base_row, growth_row, 15)
        expected_str = round(float(base_row["Str"]) + (float(growth_row["Str"]) / 100) * 14, 1)
        self.assertEqual(result["Str"], expected_str)
        self.assertGreaterEqual(result["Str"], base_row["Str"])

    def test_target_level_below_join_level_is_floored(self):
        result = recommend_for_character(
            "Catherine", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=5,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF,
        )
        self.assertEqual(result["requested_target_level"], 5)
        self.assertEqual(result["join_level"], STARTING_LEVEL_LOOKUP["Catherine"])
        self.assertEqual(result["expected_stats_at_level"], STARTING_LEVEL_LOOKUP["Catherine"])

    def test_without_starting_level_df_everyone_is_level_one(self):
        result = recommend_for_character(
            "Catherine", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF, target_level=30,
        )
        self.assertEqual(result["join_level"], 1)


class TestDlcClasses(unittest.TestCase):
    def test_dlc_classes_absent_by_default(self):
        path = recommend_path(
            STAT_BOOSTS_DF, "Magic Attacker", target_level=30,
            character_name="Lysithea", eligibility_lookup=ELIGIBILITY_LOOKUP,
            character_gender="Female", weapon_req_lookup=WEAPON_REQ_LOOKUP,
        )
        dlc_class_names = set(STAT_BOOSTS_DF[STAT_BOOSTS_DF["tier"] == DLC_CLASS_TIER]["name"])
        picked = {step["class"] for step in path}
        self.assertTrue(picked.isdisjoint(dlc_class_names))

    def test_dlc_classes_available_at_advanced_when_opted_in(self):
        options = list_eligible_classes_at_tier(
            DLC_CLASS_MERGE_TIER, STAT_BOOSTS_DF,
            character_name="Lysithea", eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender="Female",
            include_dlc_classes=True,
        )
        self.assertIn("Valkyrie", options)

    def test_dlc_classes_respect_gender_lock(self):
        options = list_eligible_classes_at_tier(
            DLC_CLASS_MERGE_TIER, STAT_BOOSTS_DF,
            character_name="Caspar", eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender="Male",
            include_dlc_classes=True,
        )
        self.assertNotIn("Valkyrie", options)  # Female-locked
        self.assertIn("Trickster", options)  # unrestricted


if __name__ == "__main__":
    unittest.main()
