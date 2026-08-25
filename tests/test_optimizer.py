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
    ROLE_PROFILES,
    STAT_COLS,
    TIER_LEVEL_REQUIREMENTS,
    TIER_ORDER,
    apply_weapon_affinity_fallback,
    compute_roster_stat_stats,
    detect_natural_role,
    format_requirement,
    is_class_eligible,
    list_eligible_classes_at_tier,
    load_character_proficiency_lookup,
    load_eligibility_lookup,
    load_weapon_requirements_lookup,
    reachable_tiers,
    recommend_for_character,
    recommend_path,
    stats_for_class_at_level,
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

WEAPON_REQ_LOOKUP = load_weapon_requirements_lookup(WEAPON_REQ_DF)
PROFICIENCY_LOOKUP = load_character_proficiency_lookup(CHARACTER_WEAPON_TALENT_DF)
ELIGIBILITY_LOOKUP = load_eligibility_lookup(ELIGIBILITY_DF)
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
                with self.subTest(character=name, role=role):
                    result = recommend_for_character(
                        name, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
                        role_name=role, target_level=30,
                        eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
                        weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
                    )
                    self.assertIsInstance(result["path"], list)


if __name__ == "__main__":
    unittest.main()
