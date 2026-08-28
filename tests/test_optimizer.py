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
    NO_CERTIFICATION_CLASSES,
    PERSONAL_DLC_CLASS_ROLE_BONUS,
    PERSONAL_DLC_CLASS_ROLES,
    RELIC_AFFINITY_BONUS,
    ROLE_PROFILES,
    STAT_COLS,
    TIER_LEVEL_REQUIREMENTS,
    TIER_ORDER,
    UNIQUE_CLASS_SCORE_BONUS,
    UNIQUE_STORY_CLASS_TIER,
    WEAPON_SWITCH_PENALTY,
    apply_class_base_stat_floor,
    apply_weapon_affinity_fallback,
    base_stats_at_join_level,
    combined_requirements_for_classes,
    compute_roster_stat_stats,
    detect_natural_role,
    eligible_unique_story_class_by_tier,
    expected_stats_along_path,
    format_combined_requirements,
    format_requirement,
    is_class_eligible,
    list_eligible_classes_at_tier,
    load_character_proficiency_lookup,
    load_character_relic_lookup,
    load_class_base_stats_lookup,
    load_class_growth_lookup,
    load_eligibility_lookup,
    load_starting_level_lookup,
    load_weapon_requirements_lookup,
    natural_role_affinity_bonus,
    path_level_bands,
    path_weapon_switch_warning,
    personal_dlc_class_role_bonus,
    reachable_tiers,
    recommend_for_character,
    recommend_path,
    relic_affinity_bonus,
    restrict_support_to_magic_classes,
    role_compatible_with_weapon_category,
    role_relevant_stat_deltas,
    score_all_roles,
    score_growth_for_role,
    stats_for_class_at_level,
    stats_for_selected_path,
    unique_class_weapon_category,
    weapon_growth_bonus,
    weapon_switch_penalty,
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
CLASS_GROWTH_DF = pd.read_csv(DATA_DIR / "class_growth_rates.csv")
CLASS_BASE_STATS_DF = pd.read_csv(DATA_DIR / "class_base_stats.csv")
CHARACTER_RELICS_DF = pd.read_csv(DATA_DIR / "character_relics.csv")
WEAPON_REQ_LOOKUP = load_weapon_requirements_lookup(WEAPON_REQ_DF)
PROFICIENCY_LOOKUP = load_character_proficiency_lookup(CHARACTER_WEAPON_TALENT_DF)
ELIGIBILITY_LOOKUP = load_eligibility_lookup(ELIGIBILITY_DF)
STARTING_LEVEL_LOOKUP = load_starting_level_lookup(STARTING_LEVEL_DF)
CLASS_GROWTH_LOOKUP = load_class_growth_lookup(CLASS_GROWTH_DF)
CLASS_BASE_STATS_LOOKUP = load_class_base_stats_lookup(CLASS_BASE_STATS_DF)
CHARACTER_RELIC_LOOKUP = load_character_relic_lookup(CHARACTER_RELICS_DF)
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

    def test_no_certification_classes_never_show_a_requirement(self):
        # Item 4 audit: none of the eleven story/starting classes that
        # class_eligibility.csv itself documents as reached WITHOUT a
        # certification exam should ever surface a requirement line -
        # "Enlightened One says it has a requirement of A rank swords" was
        # the reported symptom; "Lord" (a stray row in
        # class_weapon_requirements.csv, since removed) was a confirmed
        # live instance of the same bug.
        for class_name in NO_CERTIFICATION_CLASSES:
            self.assertIsNone(
                format_requirement(class_name, WEAPON_REQ_LOOKUP),
                f"{class_name} should never show a certification requirement",
            )

    def test_load_weapon_requirements_lookup_ignores_stray_rows_for_no_cert_classes(self):
        # Even if class_weapon_requirements.csv regains a stray row for one
        # of these (as "Lord" once had), the loader must still drop it -
        # this is the actual regression guard, independent of what the
        # checked-in CSV currently contains.
        stray_df = pd.DataFrame([
            {"class_name": "Lord", "tier": "Intermediate", "weapon_category": "physical",
             "requirement_type": "AND", "requirements": "Sword:D+|Authority:C"},
            {"class_name": "Fighter", "tier": "Beginner", "weapon_category": "physical",
             "requirement_type": "OR", "requirements": "Axe:D|Bow:D|Brawling:D"},
        ])
        lookup = load_weapon_requirements_lookup(stray_df)
        self.assertNotIn("Lord", lookup)
        self.assertIn("Fighter", lookup)

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
                            class_growth_df=CLASS_GROWTH_DF, class_base_stats_df=CLASS_BASE_STATS_DF,
                            character_relics_df=CHARACTER_RELICS_DF,
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


class TestLoadCharacterRelicLookup(unittest.TestCase):
    def test_protagonist_wields_sword_of_the_creator(self):
        self.assertEqual(CHARACTER_RELIC_LOOKUP["Protagonist"], {"Sword"})

    def test_lysithea_has_two_relics_across_two_weapon_types(self):
        # Lysithea bears two minor crests (Charon and Gloucester) and has a
        # relic for each - Thunderbrand (Sword) and Thyrsus (Reason) - so
        # her entry should union both weapon types, not just the last row
        # read from the CSV.
        self.assertEqual(CHARACTER_RELIC_LOOKUP["Lysithea"], {"Sword", "Reason"})

    def test_character_with_no_relic_is_absent_from_the_lookup(self):
        self.assertNotIn("Bernadetta", CHARACTER_RELIC_LOOKUP)

    def test_returns_empty_dict_for_none(self):
        self.assertEqual(load_character_relic_lookup(None), {})


class TestRelicAffinityBonus(unittest.TestCase):
    def test_bonus_when_class_certification_uses_the_relic_weapon_type(self):
        # The Protagonist's Sword of the Creator is a Sword relic - Myrmidon
        # certifies on Sword, so it should get the bonus.
        bonus = relic_affinity_bonus("Myrmidon", WEAPON_REQ_LOOKUP, CHARACTER_RELIC_LOOKUP["Protagonist"])
        self.assertEqual(bonus, RELIC_AFFINITY_BONUS)

    def test_no_bonus_when_no_overlap(self):
        # Monk certifies on Reason/Faith, not Sword.
        bonus = relic_affinity_bonus("Monk", WEAPON_REQ_LOOKUP, CHARACTER_RELIC_LOOKUP["Protagonist"])
        self.assertEqual(bonus, 0.0)

    def test_no_bonus_without_relic_data(self):
        self.assertEqual(relic_affinity_bonus("Myrmidon", WEAPON_REQ_LOOKUP, None), 0.0)

    def test_no_bonus_for_character_with_no_relic(self):
        self.assertEqual(
            relic_affinity_bonus("Myrmidon", WEAPON_REQ_LOOKUP, CHARACTER_RELIC_LOOKUP.get("Bernadetta")), 0.0,
        )

    def test_no_bonus_for_unknown_class(self):
        self.assertEqual(relic_affinity_bonus("Not A Real Class", WEAPON_REQ_LOOKUP, {"Sword"}), 0.0)


class TestRelicAffinityFeedsIntoScoring(unittest.TestCase):
    """
    Integration-level: a relic-bearing character's own Hero's Relic weapon
    type should be able to move recommend_path's pick, the same way
    weapon_growth_bonus already does for starting proficiency - not just
    exercise relic_affinity_bonus in isolation.
    """

    def test_recommend_path_prefers_the_relic_weapon_type_in_a_tied_role_matchup(self):
        # Two single-row "classes" tied on every stat the role cares about,
        # differing only in their weapon certification - Sword vs. Reason.
        # Without a relic, score_class_for_role ties them (order/whichever
        # pandas idxmax picks first is not something to pin down); with a
        # Sword relic, the Sword option should win outright.
        tier_rows = pd.DataFrame([
            {"name": "Sword Option", "tier": "Beginner", "HP": 5, "Str": 5, "Mag": 0,
             "Dex": 5, "Spd": 5, "Lck": 5, "Def": 5, "Res": 0, "Cha": 5, "Mov": 4},
            {"name": "Reason Option", "tier": "Beginner", "HP": 5, "Str": 5, "Mag": 0,
             "Dex": 5, "Spd": 5, "Lck": 5, "Def": 5, "Res": 0, "Cha": 5, "Mov": 4},
        ])
        weapon_req_lookup = {
            "Sword Option": {"weapon_category": "physical", "requirement_type": "AND", "requirements": [("Sword", "D")]},
            "Reason Option": {"weapon_category": "magic", "requirement_type": "AND", "requirements": [("Reason", "D")]},
        }
        path = recommend_path(
            tier_rows, "Physical Attacker", tiers=["Beginner"],
            weapon_req_lookup=weapon_req_lookup,
            character_relic_weapon_types={"Sword"},
        )
        self.assertEqual(path[0]["class"], "Sword Option")

    def test_recommend_for_character_accepts_character_relics_df_without_raising(self):
        # Full-pipeline smoke check for a relic-bearing character - the
        # detailed scoring effect is covered by the tied-matchup test above;
        # this just confirms the plumbing (recommend_for_character ->
        # recommend_path -> score_row) doesn't break for real data.
        result = recommend_for_character(
            "Claude", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            class_base_stats_df=CLASS_BASE_STATS_DF, character_relics_df=CHARACTER_RELICS_DF,
        )
        self.assertIsInstance(result["path"], list)


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


class TestDeathKnightEligibility(unittest.TestCase):
    """
    "Jeritza is the only character that can use the Death Knight class" -
    unlike Trickster/War Monk/Cleric (unrestricted DLC Exclusive classes)
    or Dark Flier/Valkyrie (gender-locked DLC Exclusive classes), Death
    Knight is a DLC Exclusive class locked to exactly one character, per
    the class_eligibility.csv row added alongside this test. Covers the
    eligibility half of the report; TestPersonalDlcClassRoleBonus below
    covers the "weighted slightly higher for him in physical roles" half.
    """

    def test_death_knight_eligible_for_jeritza(self):
        self.assertTrue(is_class_eligible("Jeritza", "Death Knight", ELIGIBILITY_LOOKUP, "Male"))

    def test_death_knight_not_eligible_for_other_characters(self):
        for name, gender in (("Sylvain", "Male"), ("Dorothea", "Female"), ("Petra", "Female")):
            with self.subTest(name=name):
                self.assertFalse(is_class_eligible(name, "Death Knight", ELIGIBILITY_LOOKUP, gender))

    def test_death_knight_absent_from_other_characters_advanced_tier_options_even_with_dlc_on(self):
        options = list_eligible_classes_at_tier(
            DLC_CLASS_MERGE_TIER, STAT_BOOSTS_DF,
            character_name="Sylvain", eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender="Male",
            include_dlc_classes=True,
        )
        self.assertNotIn("Death Knight", options)

    def test_death_knight_present_in_jeritzas_advanced_tier_options_when_dlc_on(self):
        options = list_eligible_classes_at_tier(
            DLC_CLASS_MERGE_TIER, STAT_BOOSTS_DF,
            character_name="Jeritza", eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender="Male",
            include_dlc_classes=True,
        )
        self.assertIn("Death Knight", options)

    def test_death_knight_wins_jeritzas_physical_attacker_path_at_advanced_when_dlc_on(self):
        path = recommend_path(
            STAT_BOOSTS_DF, "Physical Attacker", target_level=30,
            character_name="Jeritza", eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender="Male",
            weapon_req_lookup=WEAPON_REQ_LOOKUP, class_growth_lookup=CLASS_GROWTH_LOOKUP,
            include_dlc_classes=True,
        )
        advanced = next(step for step in path if step["tier"] == "Advanced")
        self.assertEqual(advanced["class"], "Death Knight")

    def test_death_knight_never_wins_a_non_jeritza_physical_attacker_path_even_with_dlc_on(self):
        # Nobody else is eligible for it (see the eligibility tests above),
        # so it should never show up as a *pick*, DLC opted in or not -
        # belt-and-suspenders alongside the eligibility-list tests, at the
        # level a user actually sees (a recommended path, not a raw option
        # list).
        for name, gender in (("Sylvain", "Male"), ("Dorothea", "Female")):
            with self.subTest(name=name):
                path = recommend_path(
                    STAT_BOOSTS_DF, "Physical Attacker", target_level=30,
                    character_name=name, eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender=gender,
                    weapon_req_lookup=WEAPON_REQ_LOOKUP, class_growth_lookup=CLASS_GROWTH_LOOKUP,
                    include_dlc_classes=True,
                )
                picked = {step["class"] for step in path}
                self.assertNotIn("Death Knight", picked)

    def test_death_knight_has_no_certification_weapon_requirement(self):
        # Story-unlocked, like Lord/Dancer/Noble/Commoner - not something
        # any character (Jeritza included) certifies into via a seal exam.
        self.assertIn("Death Knight", NO_CERTIFICATION_CLASSES)
        self.assertIsNone(format_requirement("Death Knight", WEAPON_REQ_LOOKUP))


class TestPersonalDlcClassRoleBonus(unittest.TestCase):
    """
    The "should maybe be weighted slightly higher for him in physical
    roles" half of the Death Knight report - a small, additive nudge (see
    PERSONAL_DLC_CLASS_ROLE_BONUS's own comment for why it's sized well
    below UNIQUE_CLASS_SCORE_BONUS), not something that overrides a
    genuinely better-fitting class like Fortress Knight for Tank.
    """

    def test_bonus_applies_for_jeritza_in_his_listed_roles(self):
        for role in PERSONAL_DLC_CLASS_ROLES["Death Knight"]["roles"]:
            with self.subTest(role=role):
                self.assertEqual(
                    personal_dlc_class_role_bonus("Death Knight", "Jeritza", role),
                    PERSONAL_DLC_CLASS_ROLE_BONUS,
                )

    def test_bonus_is_zero_for_jeritza_in_an_unlisted_role(self):
        self.assertEqual(personal_dlc_class_role_bonus("Death Knight", "Jeritza", "Magic Attacker"), 0.0)

    def test_bonus_is_zero_for_any_other_character(self):
        self.assertEqual(personal_dlc_class_role_bonus("Death Knight", "Sylvain", "Physical Attacker"), 0.0)

    def test_bonus_is_zero_for_a_class_with_no_entry(self):
        self.assertEqual(personal_dlc_class_role_bonus("Trickster", "Jeritza", "Physical Attacker"), 0.0)

    def test_bonus_does_not_override_a_genuinely_better_tank(self):
        # Fortress Knight's dedicated +10 Def boost should still win Tank
        # for Jeritza even with Death Knight's own small bonus applied -
        # the bonus is a tie-breaker, not a splice/override like
        # UNIQUE_CLASS_SCORE_BONUS.
        path = recommend_path(
            STAT_BOOSTS_DF, "Tank", target_level=30,
            character_name="Jeritza", eligibility_lookup=ELIGIBILITY_LOOKUP, character_gender="Male",
            weapon_req_lookup=WEAPON_REQ_LOOKUP, class_growth_lookup=CLASS_GROWTH_LOOKUP,
            include_dlc_classes=True,
        )
        advanced = next(step for step in path if step["tier"] == "Advanced")
        self.assertEqual(advanced["class"], "Fortress Knight")


class TestClassGrowthRates(unittest.TestCase):
    """
    Coverage for data/class_growth_rates.csv and load_class_growth_lookup -
    the "tool says classes don't have growth rates - this is very wrong"
    report. Classes DO have their own growth-rate modifiers in Three
    Houses, a separate mechanic from a class's one-time flat stat boost
    (class_stat_boosts.csv) - see
    https://serenesforest.net/three-houses/classes/growth-rates/.
    """

    def test_every_playable_class_has_growth_rate_data(self):
        playable = STAT_BOOSTS_DF[~STAT_BOOSTS_DF["name"].str.contains(r"\(", regex=True)]
        playable = playable[playable["tier"] != "NPC/Enemy"]
        missing = sorted(set(playable["name"]) - set(CLASS_GROWTH_LOOKUP.keys()))
        self.assertEqual(missing, [], f"missing growth-rate data for: {missing}")

    def test_warrior_has_a_meaningful_positive_str_modifier(self):
        # Warrior is a well-known Str/HP powerhouse class - a real, sourced
        # modifier should show up here, not a placeholder zero.
        self.assertGreater(CLASS_GROWTH_LOOKUP["Warrior"]["Str"], 0)

    def test_score_growth_for_role_is_a_weighted_dot_product(self):
        growth_mod = {"Str": 20, "Spd": 10, "Mag": -5}
        score = score_growth_for_role(growth_mod, ROLE_PROFILES["Physical Attacker"])
        # Physical Attacker weights Str 1.0, Spd 0.5 - Mag isn't weighted at all.
        self.assertAlmostEqual(score, 20 * 1.0 + 10 * 0.5)

    def test_score_growth_for_role_ignores_unweighted_stats(self):
        growth_mod = {"Cha": 50}  # Cha isn't in any ROLE_PROFILES weight dict
        self.assertEqual(score_growth_for_role(growth_mod, ROLE_PROFILES["Tank"]), 0.0)


class TestPathLevelBands(unittest.TestCase):
    def test_empty_path_returns_no_bands(self):
        self.assertEqual(path_level_bands([], target_level=30), [])

    def test_bands_cover_target_level_minus_start_level_total_levels(self):
        # Regression test for an off-by-one bug where the last band's end
        # used target_level + 1, double-counting a level-up. start_level=5
        # matches Beginner's own level requirement, so there's no
        # pre-first-tier gap to account for separately here (see
        # TestExpectedStatsAlongPath for that gap-filling behavior).
        path = [
            {"tier": "Beginner", "class": "Fighter"},
            {"tier": "Intermediate", "class": "Brigand"},
            {"tier": "Advanced", "class": "Warrior"},
            {"tier": "Master", "class": "War Master"},
        ]
        bands = path_level_bands(path, target_level=30, start_level=5)
        total_levels = sum(end - start for _tier, _cls, start, end in bands)
        self.assertEqual(total_levels, 30 - 5)

    def test_bands_clamp_to_start_level_for_a_late_join(self):
        # A character who joins at level 15 has already banked their
        # Beginner-tier levels before they're recruitable - that band
        # should be skipped entirely (clamped away). Intermediate absorbs
        # levels 15-20 (Advanced isn't reachable until 20), so every level
        # from start_level to target_level is still covered by exactly one
        # tier's band - no gaps, no double-counting.
        path = [
            {"tier": "Beginner", "class": "Fighter"},
            {"tier": "Intermediate", "class": "Brigand"},
            {"tier": "Advanced", "class": "Warrior"},
        ]
        bands = path_level_bands(path, target_level=30, start_level=15)
        tiers_covered = {tier for tier, _cls, _s, _e in bands}
        self.assertNotIn("Beginner", tiers_covered)
        self.assertIn("Intermediate", tiers_covered)
        self.assertIn("Advanced", tiers_covered)
        total_levels = sum(end - start for _tier, _cls, start, end in bands)
        self.assertEqual(total_levels, 30 - 15)


class TestExpectedStatsAlongPath(unittest.TestCase):
    def test_falls_back_to_flat_rate_without_a_growth_lookup(self):
        result = recommend_for_character(
            "Bernadetta", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
        )
        without_growth = stats_for_class_at_level(
            "Bernadetta", result["path"][-1]["class"], BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=30,
        )
        self.assertEqual(result["expected_final_stats"], without_growth)

    def test_per_tier_class_growth_changes_the_projection(self):
        result_flat = recommend_for_character(
            "Bernadetta", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
        )
        result_with_growth = recommend_for_character(
            "Bernadetta", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            class_growth_df=CLASS_GROWTH_DF,
        )
        # Simulating each tier's own class growth modifier along the whole
        # path (instead of flat character-growth-only) should change at
        # least one projected stat - otherwise the projection isn't
        # actually using the new data.
        self.assertNotEqual(result_flat["expected_final_stats"], result_with_growth["expected_final_stats"])

    def test_pre_first_tier_levels_still_accrue_the_characters_own_growth(self):
        # Levels 1-4 (before Beginner's level-5 requirement) aren't covered
        # by any path_level_bands entry, but they still happened - a
        # character's own growth rate should still apply there, not be
        # silently dropped (which would understate every projected stat
        # and break the "total levels gained == target_level - start_level"
        # invariant).
        path = [{"tier": "Beginner", "class": "Fighter"}, {"tier": "Intermediate", "class": "Brigand"}]
        base_row = BASE_STATS_DF[BASE_STATS_DF["name"] == "Caspar"].iloc[0]
        growth_row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Caspar"].iloc[0]
        boost_row = STAT_BOOSTS_DF[STAT_BOOSTS_DF["name"] == "Brigand"].iloc[0]
        stats_from_level_1 = expected_stats_along_path(
            path, base_row, growth_row, boost_row, CLASS_GROWTH_LOOKUP, target_level=15, start_level=1,
        )
        stats_from_level_5 = expected_stats_along_path(
            path, base_row.copy() if hasattr(base_row, "copy") else dict(base_row),
            growth_row, boost_row, CLASS_GROWTH_LOOKUP, target_level=15, start_level=5,
        )
        # Starting the simulation from level 1 (four extra levels of the
        # character's own growth before Beginner even unlocks) should
        # project HIGHER stats than starting from level 5 with the same
        # base_row - if the pre-band gap were silently dropped, the two
        # would come out identical.
        self.assertNotEqual(stats_from_level_1, stats_from_level_5)

    def test_every_stat_column_present_in_a_path_wide_projection(self):
        path = [{"tier": "Beginner", "class": "Fighter"}, {"tier": "Intermediate", "class": "Brigand"}]
        base_row = BASE_STATS_DF[BASE_STATS_DF["name"] == "Caspar"].iloc[0]
        growth_row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Caspar"].iloc[0]
        boost_row = STAT_BOOSTS_DF[STAT_BOOSTS_DF["name"] == "Brigand"].iloc[0]
        stats = expected_stats_along_path(
            path, base_row, growth_row, boost_row, CLASS_GROWTH_LOOKUP, target_level=15, start_level=1,
        )
        self.assertEqual(set(stats.keys()), set(STAT_COLS))

    def test_class_base_stat_floor_snaps_up_at_each_tier_before_that_tiers_growth(self):
        # Item 6: a class base-stat floor applies at the MOMENT of
        # certifying into each tier - before that tier's own growth is
        # simulated, not stacked on top of it - and it applies at EVERY
        # tier along the path, not just the final one. Hand-computed
        # exactly: a character with every stat at 1 and zero personal
        # growth, walking Fighter (Beginner, HP base 20, HP growth +10%)
        # -> Brigand (Intermediate, HP base 28, HP growth +30%), levels
        # 1->15 (pre-band 4 levels, Fighter band 5-10, Brigand band 10-15).
        zero_growth = pd.Series({s: 0 for s in STAT_COLS})
        zero_boost = pd.Series({s: 0 for s in STAT_COLS})
        low_base = pd.Series({s: 1 for s in STAT_COLS})
        path = [{"tier": "Beginner", "class": "Fighter"}, {"tier": "Intermediate", "class": "Brigand"}]

        floored = expected_stats_along_path(
            path, low_base, zero_growth, zero_boost, CLASS_GROWTH_LOOKUP, target_level=15, start_level=1,
            class_base_stats_lookup=CLASS_BASE_STATS_LOOKUP,
        )
        # 1 (pre-band, 4 levels @ 0%) -> floor to Fighter's HP base (20) ->
        # +5 levels @ Fighter's 10% HP growth (+0.5) = 20.5 -> floor to
        # Brigand's HP base (28, higher than 20.5) -> +5 levels @
        # Brigand's 30% HP growth (+1.5) = 29.5.
        self.assertEqual(floored["HP"], 29.5)

        unfloored = expected_stats_along_path(
            path, low_base, zero_growth, zero_boost, CLASS_GROWTH_LOOKUP, target_level=15, start_level=1,
        )
        # Without the floor: 1 + 5*0.10 + 5*0.30 = 3.0 - confirms the gap
        # is entirely the floor's doing, not a side effect of the growth math.
        self.assertEqual(unfloored["HP"], 3.0)

    def test_class_base_stat_floor_is_a_ceiling_free_no_op_when_already_higher(self):
        # A stat already ABOVE every relevant class's base is untouched -
        # the floor never subtracts, and never adds on top either (it's a
        # floor, not a bonus).
        zero_growth = pd.Series({s: 0 for s in STAT_COLS})
        zero_boost = pd.Series({s: 0 for s in STAT_COLS})
        high_base = pd.Series({s: 100 for s in STAT_COLS})
        path = [{"tier": "Beginner", "class": "Fighter"}, {"tier": "Intermediate", "class": "Brigand"}]

        floored = expected_stats_along_path(
            path, high_base, zero_growth, zero_boost, CLASS_GROWTH_LOOKUP, target_level=15, start_level=1,
            class_base_stats_lookup=CLASS_BASE_STATS_LOOKUP,
        )
        unfloored = expected_stats_along_path(
            path, high_base, zero_growth, zero_boost, CLASS_GROWTH_LOOKUP, target_level=15, start_level=1,
        )
        self.assertEqual(floored, unfloored)


class TestApplyClassBaseStatFloor(unittest.TestCase):
    def test_snaps_up_when_below(self):
        self.assertEqual(
            apply_class_base_stat_floor(5, "Warrior", "HP", CLASS_BASE_STATS_LOOKUP),
            float(CLASS_BASE_STATS_LOOKUP["Warrior"]["HP"]),
        )

    def test_leaves_higher_value_unchanged(self):
        self.assertEqual(apply_class_base_stat_floor(500, "Warrior", "HP", CLASS_BASE_STATS_LOOKUP), 500)

    def test_missing_lookup_is_a_no_op(self):
        self.assertEqual(apply_class_base_stat_floor(5, "Warrior", "HP", None), 5)
        self.assertEqual(apply_class_base_stat_floor(5, "Warrior", "HP", {}), 5)

    def test_missing_class_is_a_no_op(self):
        self.assertEqual(apply_class_base_stat_floor(5, "Not A Real Class", "HP", CLASS_BASE_STATS_LOOKUP), 5)


class TestLoadClassBaseStatsLookup(unittest.TestCase):
    def test_returns_empty_dict_for_none(self):
        self.assertEqual(load_class_base_stats_lookup(None), {})

    def test_covers_the_same_real_classes_as_class_growth_rates(self):
        # data/class_base_stats.csv should mirror data/class_growth_rates.csv's
        # own class list exactly - both describe the same "real, certifiable
        # or story-unlocked classes" set (no NPC/Enemy rows, no
        # story/enemy-variant rows like "Lord (Judith)").
        self.assertEqual(set(CLASS_BASE_STATS_LOOKUP.keys()), set(CLASS_GROWTH_LOOKUP.keys()))


class TestStatsForSelectedPath(unittest.TestCase):
    def test_matches_recommend_for_character_when_selection_equals_recommendation(self):
        result = recommend_for_character(
            "Bernadetta", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            class_growth_df=CLASS_GROWTH_DF, class_base_stats_df=CLASS_BASE_STATS_DF,
        )
        selected_steps = [{"tier": s["tier"], "class": s["class"]} for s in result["path"]]
        recomputed = stats_for_selected_path(
            "Bernadetta", selected_steps, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=result["expected_stats_at_level"], start_level=result["join_level"],
            class_growth_lookup=CLASS_GROWTH_LOOKUP, class_base_stats_lookup=CLASS_BASE_STATS_LOOKUP,
        )
        self.assertEqual(recomputed, result["expected_final_stats"])

    def test_overriding_a_non_final_tier_can_change_the_final_projection(self):
        # Item 1 regression guard: before round 5, only the final tier's
        # class affected the projected stats. With the per-tier floor,
        # swapping an EARLIER tier's class can change the numbers too -
        # the whole reason the projected-stats chart now needs to
        # re-render on any tier change, not just the final one.
        result = recommend_for_character(
            "Bernadetta", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            class_growth_df=CLASS_GROWTH_DF, class_base_stats_df=CLASS_BASE_STATS_DF,
        )
        original_steps = [{"tier": s["tier"], "class": s["class"]} for s in result["path"]]
        # Override the FIRST tier only, to whichever Beginner class isn't
        # already the recommendation (their base stats differ enough from
        # each other - e.g. Monk's Mag/Res-leaning line vs. Myrmidon/
        # Soldier/Fighter's Str-leaning ones - to move the floor) while
        # leaving every later tier (including the final one) alone.
        overridden_steps = [dict(s) for s in original_steps]
        alternative = next(
            c for c in ["Myrmidon", "Soldier", "Fighter", "Monk"] if c != overridden_steps[0]["class"]
        )
        overridden_steps[0]["class"] = alternative

        original_stats = stats_for_selected_path(
            "Bernadetta", original_steps, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=result["expected_stats_at_level"], start_level=result["join_level"],
            class_growth_lookup=CLASS_GROWTH_LOOKUP, class_base_stats_lookup=CLASS_BASE_STATS_LOOKUP,
        )
        overridden_stats = stats_for_selected_path(
            "Bernadetta", overridden_steps, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=result["expected_stats_at_level"], start_level=result["join_level"],
            class_growth_lookup=CLASS_GROWTH_LOOKUP, class_base_stats_lookup=CLASS_BASE_STATS_LOOKUP,
        )
        self.assertNotEqual(original_stats, overridden_stats)

    def test_returns_none_for_empty_selection(self):
        self.assertIsNone(stats_for_selected_path(
            "Bernadetta", [], BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
        ))

    def test_returns_none_for_unknown_final_class(self):
        self.assertIsNone(stats_for_selected_path(
            "Bernadetta", [{"tier": "Beginner", "class": "Not A Real Class"}],
            BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
        ))


class TestWeaponSwitchPenalty(unittest.TestCase):
    def test_or_requirement_is_satisfied_by_any_one_known_skill(self):
        # Fighter's Beginner requirement is "Axe D or Bow D or Brawling D" -
        # an OR - so already knowing just Axe should be enough.
        penalty = weapon_switch_penalty(
            "Fighter", WEAPON_REQ_LOOKUP, accumulated_skills=set(), original_proficiency={"Axe"},
        )
        self.assertEqual(penalty, 0.0)

    def test_and_requirement_needs_every_listed_skill_known(self):
        # Wyvern Lord requires Lance C AND Axe A AND Flying A - knowing only
        # Axe from earlier in the path isn't enough to call this a
        # non-switch; Lance and Flying are still both brand new.
        penalty = weapon_switch_penalty(
            "Wyvern Lord", WEAPON_REQ_LOOKUP, accumulated_skills={"Axe"}, original_proficiency={"Sword"},
        )
        self.assertEqual(penalty, WEAPON_SWITCH_PENALTY)

    def test_and_requirement_satisfied_once_every_skill_is_known(self):
        penalty = weapon_switch_penalty(
            "Wyvern Lord", WEAPON_REQ_LOOKUP,
            accumulated_skills={"Axe", "Lance", "Flying"}, original_proficiency=None,
        )
        self.assertEqual(penalty, 0.0)

    def test_unknown_class_has_no_penalty(self):
        self.assertEqual(
            weapon_switch_penalty("Not A Real Class", WEAPON_REQ_LOOKUP, set(), None), 0.0,
        )

    def test_low_rank_mount_requirement_never_flags(self):
        # Item 5: Cavalier's Riding D is a low-rank mount requirement -
        # never a flagged switch, even with zero prior riding practice at all.
        penalty = weapon_switch_penalty(
            "Cavalier", WEAPON_REQ_LOOKUP, accumulated_skills=set(), original_proficiency={"Lance"},
        )
        self.assertEqual(penalty, 0.0)

    def test_high_rank_mount_requirement_flags_with_zero_related_practice(self):
        # Great Knight's Heavy Armour A is a genuinely high-rank ask - with
        # zero practice in ANY mount/armor skill (only Axe known), this
        # should still flag, per "the warning should only fire for these
        # categories on a jump to a high rank with zero prior practice."
        penalty = weapon_switch_penalty(
            "Great Knight", WEAPON_REQ_LOOKUP, accumulated_skills={"Axe"}, original_proficiency=None,
        )
        self.assertEqual(penalty, WEAPON_SWITCH_PENALTY)

    def test_high_rank_mount_requirement_forgiven_by_related_mount_practice(self):
        # Same Great Knight ask, but the character already has SOME
        # mount/armor practice (Riding, from an earlier Cavalier/Paladin
        # tier) even though they've never specifically trained Heavy
        # Armour - that should be enough to excuse the high-rank ask,
        # since mount and armor training are graded as one forgiving
        # category, not skill-by-skill.
        penalty = weapon_switch_penalty(
            "Great Knight", WEAPON_REQ_LOOKUP, accumulated_skills={"Axe", "Riding"}, original_proficiency=None,
        )
        self.assertEqual(penalty, 0.0)

    def test_ordinary_weapon_skills_are_unaffected_by_the_mount_armor_relaxation(self):
        # Falcon Knight needs Sword C AND Lance A AND Flying A - even
        # though Flying A (mount, high rank, zero related practice) would
        # normally flag, Sword and Lance are ORDINARY weapon skills and
        # still use the strict exact-skill rule - a character who's never
        # touched Sword or Lance still gets flagged regardless of how
        # Flying is scored.
        penalty = weapon_switch_penalty(
            "Falcon Knight", WEAPON_REQ_LOOKUP, accumulated_skills={"Riding"}, original_proficiency=None,
        )
        self.assertEqual(penalty, WEAPON_SWITCH_PENALTY)

    def test_catherine_swordmaster_to_wyvern_lord_flags_the_real_complaint(self):
        # The original report: "weapon requirements aren't weighted well -
        # Catherine going Swordmaster to Wyvern Lord is very difficult in
        # practice". Catherine's own starting proficiency is Sword only -
        # Wyvern Lord's Lance/Axe/Flying (AND) requirement should be
        # flagged as a real weapon-type switch, whatever detour the rest
        # of her path took first.
        result = recommend_for_character(
            "Catherine", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            class_growth_df=CLASS_GROWTH_DF,
        )
        master_step = next(s for s in result["path"] if s["tier"] == "Master")
        self.assertEqual(master_step["class"], "Wyvern Lord")
        self.assertTrue(master_step["weapon_switch_warning"])
        self.assertIsNotNone(result["weapon_switch_warning"])
        self.assertIn("Wyvern Lord", result["weapon_switch_warning"])


class TestGrowthRateScoringDoesNotDestabilizeExistingPicks(unittest.TestCase):
    """
    GROWTH_RATE_SCORE_WEIGHT is deliberately small so factoring growth
    rates into which class gets recommended (see recommend_path's
    class_growth_lookup) refines close calls without overriding a tier
    where the stat-boost data already gives a clear answer - these guard
    the specific recommendations other tests already pin down.
    """

    def test_edelgard_physical_attacker_still_uses_her_own_unique_classes_with_growth_scoring(self):
        result = recommend_for_character(
            "Edelgard", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
        )
        by_tier = {step["tier"]: step["class"] for step in result["path"]}
        self.assertEqual(by_tier["Advanced"], "Armored Lord")
        self.assertEqual(by_tier["Master"], "Emperor")

    def test_edelgard_recommended_path_still_has_no_gremory_with_growth_scoring(self):
        result = recommend_for_character(
            "Edelgard", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            target_level=30, eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            class_growth_df=CLASS_GROWTH_DF,
        )
        classes_in_path = [step["class"] for step in result["path"]]
        self.assertNotIn("Gremory", classes_in_path)

    def test_full_roster_every_role_sweep_with_growth_scoring_raises_nothing(self):
        for name in PLAYABLE_NAMES:
            for role_name in list(ROLE_PROFILES.keys()) + [None]:
                with self.subTest(character=name, role=role_name):
                    result = recommend_for_character(
                        name, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
                        role_name=role_name, target_level=30,
                        eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
                        weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
                        starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
                    )
                    self.assertIsInstance(result["path"], list)


class TestUniqueClassRoleGate(unittest.TestCase):
    """
    Round 6: "Dimitri/Claude's unique class gets suggested for a magic
    role even though it has no magic access" - see
    unique_class_weapon_category/role_compatible_with_weapon_category and
    recommend_path's unique_gets_bonus gate.
    """

    def test_great_lord_has_no_magic_access(self):
        self.assertEqual(
            unique_class_weapon_category("Great Lord", STAT_BOOSTS_DF, CLASS_GROWTH_LOOKUP), "physical",
        )

    def test_enlightened_one_is_hybrid(self):
        # Byleth's own master class shows a real +Mag boost and +Mag
        # growth alongside a comparable Str line - genuinely dual-focus,
        # unlike every lord line's own master/advanced unique class.
        self.assertEqual(
            unique_class_weapon_category("Enlightened One", STAT_BOOSTS_DF, CLASS_GROWTH_LOOKUP), "hybrid",
        )

    def test_role_compatibility_gate(self):
        self.assertFalse(role_compatible_with_weapon_category("Magic Attacker", "physical"))
        self.assertTrue(role_compatible_with_weapon_category("Magic Attacker", "hybrid"))
        self.assertTrue(role_compatible_with_weapon_category("Physical Attacker", "physical"))
        self.assertTrue(role_compatible_with_weapon_category("Tank", "physical"))  # Tank isn't gated at all
        self.assertTrue(role_compatible_with_weapon_category("Magic Attacker", None))  # unknown never blocks

    def test_dimitri_magic_attacker_path_never_uses_his_physical_only_unique_class(self):
        result = recommend_for_character(
            "Dimitri", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Magic Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            class_base_stats_df=CLASS_BASE_STATS_DF, character_relics_df=CHARACTER_RELICS_DF,
        )
        picked = {step["class"] for step in result["path"]}
        self.assertNotIn("High Lord", picked)
        self.assertNotIn("Great Lord", picked)
        # And the path should actually reach a real magic-capable class instead.
        final_class = result["path"][-1]["class"]
        info = WEAPON_REQ_LOOKUP.get(final_class)
        self.assertIsNotNone(info, f"expected {final_class} to have real requirement data")
        self.assertIn(info["weapon_category"], ("magic", "hybrid"))

    def test_claude_magic_attacker_path_never_uses_his_physical_only_unique_class(self):
        result = recommend_for_character(
            "Claude", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Magic Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            class_base_stats_df=CLASS_BASE_STATS_DF, character_relics_df=CHARACTER_RELICS_DF,
        )
        picked = {step["class"] for step in result["path"]}
        self.assertNotIn("Wyvern Master", picked)
        self.assertNotIn("Barbarossa", picked)

    def test_byleth_enlightened_one_still_wins_for_magic_attacker_when_competitive(self):
        # The Protagonist's own unique class IS magic-capable (hybrid) - it
        # should still be exempted from the narrowing filters and eligible
        # for its bonus, unlike the three lords' physical-only ones.
        result = recommend_for_character(
            "Protagonist", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Magic Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            class_base_stats_df=CLASS_BASE_STATS_DF, character_relics_df=CHARACTER_RELICS_DF,
        )
        master_step = next(step for step in result["path"] if step["tier"] == "Master")
        self.assertEqual(master_step["class"], "Enlightened One")
        self.assertTrue(master_step["is_unique_class"])

    def test_edelgard_physical_attacker_path_still_unaffected(self):
        # Physical Attacker is exactly what Armored Lord/Emperor ARE built
        # for - the gate should be a no-op here, same as before round 6.
        result = recommend_for_character(
            "Edelgard", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            class_base_stats_df=CLASS_BASE_STATS_DF, character_relics_df=CHARACTER_RELICS_DF,
        )
        master_step = next(step for step in result["path"] if step["tier"] == "Master")
        self.assertEqual(master_step["class"], "Emperor")


class TestBylethGenderEligibility(unittest.TestCase):
    """
    Round 6: "War Master is male-only, Falcon Knight is female-only, so it
    should depend on the gender the player picked for Byleth" - see
    recommend_for_character's character_gender_override.
    """

    def test_male_byleth_is_ineligible_for_falcon_knight(self):
        self.assertFalse(is_class_eligible("Protagonist", "Falcon Knight", ELIGIBILITY_LOOKUP, "Male"))

    def test_female_byleth_is_ineligible_for_war_master(self):
        self.assertFalse(is_class_eligible("Protagonist", "War Master", ELIGIBILITY_LOOKUP, "Female"))

    def test_male_byleth_is_eligible_for_war_master(self):
        self.assertTrue(is_class_eligible("Protagonist", "War Master", ELIGIBILITY_LOOKUP, "Male"))

    def test_female_byleth_is_eligible_for_falcon_knight(self):
        self.assertTrue(is_class_eligible("Protagonist", "Falcon Knight", ELIGIBILITY_LOOKUP, "Female"))

    def test_without_override_any_gender_is_never_blocked(self):
        # The CSV's own "Any" value, with no override applied - documents
        # the pre-override degrade-gracefully behavior is unchanged.
        self.assertTrue(is_class_eligible("Protagonist", "Falcon Knight", ELIGIBILITY_LOOKUP, "Any"))
        self.assertTrue(is_class_eligible("Protagonist", "War Master", ELIGIBILITY_LOOKUP, "Any"))

    def test_recommend_for_character_override_replaces_the_csvs_any(self):
        male_result = recommend_for_character(
            "Protagonist", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Speed/Precision", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            character_gender_override="Male",
        )
        picked = {step["class"] for step in male_result["path"]}
        self.assertNotIn("Falcon Knight", picked)

        female_result = recommend_for_character(
            "Protagonist", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            character_gender_override="Female",
        )
        picked = {step["class"] for step in female_result["path"]}
        self.assertNotIn("War Master", picked)

    def test_no_override_means_character_gender_df_still_used(self):
        # A regular character's override stays None in every real caller -
        # passing None here should behave exactly like before this feature
        # existed (their own fixed gender from the CSV, untouched).
        result_a = recommend_for_character(
            "Lysithea", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Magic Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
        )
        result_b = recommend_for_character(
            "Lysithea", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Magic Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
            character_gender_override=None,
        )
        self.assertEqual(result_a["path"], result_b["path"])


class TestNaturalRoleAffinity(unittest.TestCase):
    """
    Round 6: "how much are a character's natural proficiencies being
    taken into account for their default role" / "Lorenz's relic only
    supports mages, but he's still suggested as a Tank" - see
    natural_role_affinity_bonus/score_all_roles.
    """

    def test_magic_weapon_types_boost_magic_leaning_roles(self):
        self.assertGreater(natural_role_affinity_bonus("Magic Attacker", {"Reason"}), 0.0)
        self.assertGreater(natural_role_affinity_bonus("Support", {"Faith"}), 0.0)

    def test_physical_weapon_types_boost_physical_leaning_roles(self):
        self.assertGreater(natural_role_affinity_bonus("Physical Attacker", {"Sword"}), 0.0)

    def test_no_cross_contamination_between_categories(self):
        self.assertEqual(natural_role_affinity_bonus("Magic Attacker", {"Sword", "Lance"}), 0.0)
        self.assertEqual(natural_role_affinity_bonus("Physical Attacker", {"Reason", "Faith"}), 0.0)

    def test_unmapped_roles_are_never_nudged(self):
        self.assertEqual(natural_role_affinity_bonus("Tank", {"Reason"}), 0.0)
        self.assertEqual(natural_role_affinity_bonus("Speed/Precision", {"Sword"}), 0.0)

    def test_empty_or_missing_weapon_types_is_a_noop(self):
        self.assertEqual(natural_role_affinity_bonus("Magic Attacker", None), 0.0)
        self.assertEqual(natural_role_affinity_bonus("Magic Attacker", set()), 0.0)

    def test_score_all_roles_covers_every_role_profile(self):
        growth_row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Bernadetta"].iloc[0]
        scores = score_all_roles(growth_row, ROSTER_MEANS, ROSTER_STDS)
        self.assertEqual(set(scores.keys()), set(ROLE_PROFILES.keys()))

    def test_detect_natural_role_matches_score_all_roles_argmax(self):
        growth_row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Bernadetta"].iloc[0]
        role, score = detect_natural_role(growth_row, ROSTER_MEANS, ROSTER_STDS)
        scores = score_all_roles(growth_row, ROSTER_MEANS, ROSTER_STDS)
        self.assertEqual(role, max(scores, key=scores.get))
        self.assertAlmostEqual(score, scores[role])

    def test_lorenz_relic_affinity_flips_a_genuinely_close_tank_vs_support_call(self):
        # Documents the real-data regression this feature fixes: Lorenz's
        # own relic (Thyrsus, Reason) plus his physical starting
        # proficiency (Lance/Riding) makes Tank vs Support close enough on
        # growth rates alone that the relic's magic affinity should be
        # enough to tip Support over Tank - see the module's own
        # NATURAL_ROLE_AFFINITY_WEIGHT docstring for the calibration intent.
        growth_row = GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Lorenz"].iloc[0]
        without_affinity = score_all_roles(growth_row, ROSTER_MEANS, ROSTER_STDS)
        self.assertGreater(without_affinity["Tank"], without_affinity["Support"])

        proficiency = PROFICIENCY_LOOKUP.get("Lorenz")
        relic_types = CHARACTER_RELIC_LOOKUP.get("Lorenz")
        self.assertEqual(relic_types, {"Reason"})
        with_affinity = score_all_roles(
            growth_row, ROSTER_MEANS, ROSTER_STDS,
            character_proficiency=proficiency, character_relic_weapon_types=relic_types,
        )
        self.assertGreater(with_affinity["Support"], with_affinity["Tank"])

    def test_affinity_nudge_never_overrides_a_clear_growth_rate_signal(self):
        # A character with an overwhelming, unambiguous growth-rate lean
        # shouldn't flip roles just because their proficiency happens to
        # point elsewhere - the nudge is calibrated to break close ties,
        # not override real signal. Edelgard's own Physical Attacker lean
        # is a large, well-established margin (see
        # TestRoleDetectionRegression) - a physical-weapon proficiency
        # nudge toward the SAME role she already wins shouldn't be needed
        # to keep her a Physical Attacker either way.
        role, _ = detect_natural_role(
            GROWTH_RATES_DF[GROWTH_RATES_DF["name"] == "Edelgard"].iloc[0], ROSTER_MEANS, ROSTER_STDS,
            character_proficiency=PROFICIENCY_LOOKUP.get("Edelgard"),
            character_relic_weapon_types=CHARACTER_RELIC_LOOKUP.get("Edelgard"),
        )
        self.assertEqual(role, "Physical Attacker")


class TestCombinedPathRequirements(unittest.TestCase):
    """
    Round 6: "final class requirement display should show the skill rank
    needs for each class the character is going to use in the path" - see
    combined_requirements_for_classes/format_combined_requirements.
    """

    def test_merges_requirements_across_the_whole_path(self):
        pairs = combined_requirements_for_classes(["Cavalier", "Paladin"], WEAPON_REQ_LOOKUP)
        skills = dict(pairs)
        self.assertIn("Lance", skills)
        self.assertIn("Riding", skills)

    def test_higher_rank_wins_when_the_same_skill_repeats(self):
        # Fabricate a tiny lookup where the same skill appears at two
        # different ranks across two "tiers" - the higher one should win.
        lookup = {
            "StepA": {"tier": "Beginner", "weapon_category": "physical", "requirement_type": "AND",
                      "requirements": [("Sword", "D")]},
            "StepB": {"tier": "Advanced", "weapon_category": "physical", "requirement_type": "AND",
                      "requirements": [("Sword", "B")]},
        }
        pairs = combined_requirements_for_classes(["StepA", "StepB"], lookup)
        self.assertEqual(dict(pairs)["Sword"], "B")

    def test_classes_with_no_requirement_data_are_skipped_not_blanking(self):
        pairs = combined_requirements_for_classes(["Great Lord", "Paladin"], WEAPON_REQ_LOOKUP)
        skills = dict(pairs)
        self.assertIn("Lance", skills)  # Paladin's own requirement survives

    def test_empty_lookup_or_path_returns_empty(self):
        self.assertEqual(combined_requirements_for_classes([], WEAPON_REQ_LOOKUP), [])
        self.assertEqual(combined_requirements_for_classes(["Paladin"], {}), [])
        self.assertEqual(combined_requirements_for_classes(["Paladin"], None), [])

    def test_format_combined_requirements_returns_a_readable_string(self):
        text = format_combined_requirements(["Cavalier", "Paladin"], WEAPON_REQ_LOOKUP)
        self.assertIsInstance(text, str)
        self.assertIn("Lance", text)

    def test_format_combined_requirements_none_when_empty(self):
        self.assertIsNone(format_combined_requirements(["Great Lord"], WEAPON_REQ_LOOKUP))


class TestPathWeaponSwitchWarning(unittest.TestCase):
    """
    Round 6: "if a warning is displayed and I change the class so there is
    no longer a class warning, the overall warning is still displayed" -
    see path_weapon_switch_warning, which recomputes the summary from the
    ACTUAL selected path instead of reusing recommend_for_character's
    original one.
    """

    def test_flags_the_same_steps_recommend_path_would(self):
        result = recommend_for_character(
            "Catherine", BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            role_name="Physical Attacker", target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF, class_growth_df=CLASS_GROWTH_DF,
        )
        self.assertIsNotNone(result["weapon_switch_warning"])
        selected_steps = [{"tier": s["tier"], "class": s["class"]} for s in result["path"]]
        recomputed = path_weapon_switch_warning(
            "Catherine", selected_steps, WEAPON_REQ_LOOKUP, PROFICIENCY_LOOKUP.get("Catherine"),
        )
        self.assertIsNotNone(recomputed)

    def test_overriding_the_flagged_tier_clears_the_warning(self):
        # Catherine's Swordmaster -> Wyvern Lord jump is the documented
        # real flag (test_catherine_swordmaster_to_wyvern_lord_flags_the_
        # real_complaint) - overriding Master away from Wyvern Lord to
        # something Sword-based should clear the warning entirely, not
        # leave a stale one behind.
        selected_steps = [
            {"tier": "Beginner", "class": "Myrmidon"},
            {"tier": "Intermediate", "class": "Mercenary"},
            {"tier": "Advanced", "class": "Swordmaster"},
            {"tier": "Master", "class": "Swordmaster"},  # not a real Master class, but exercises "no switch"
        ]
        recomputed = path_weapon_switch_warning(
            "Catherine", selected_steps, WEAPON_REQ_LOOKUP, PROFICIENCY_LOOKUP.get("Catherine"),
        )
        self.assertIsNone(recomputed)

    def test_no_flagged_steps_returns_none(self):
        selected_steps = [{"tier": "Beginner", "class": "Myrmidon"}]
        self.assertIsNone(
            path_weapon_switch_warning("Catherine", selected_steps, WEAPON_REQ_LOOKUP, {"Sword"})
        )


class TestRoleRelevantStatDeltas(unittest.TestCase):
    """
    Round 6: "when changing the class to a different recommended one, we
    should display the bonus to stats that are relevant to the currently
    selected role" - see role_relevant_stat_deltas.
    """

    def test_narrows_to_only_the_roles_weighted_stats(self):
        row = STAT_BOOSTS_DF[STAT_BOOSTS_DF["name"] == "Warrior"].iloc[0]
        deltas = role_relevant_stat_deltas(row, ROLE_PROFILES["Physical Attacker"])
        stats = [stat for stat, _ in deltas]
        self.assertEqual(set(stats), {"Str", "Spd"})

    def test_sorted_by_role_weight_descending(self):
        row = STAT_BOOSTS_DF[STAT_BOOSTS_DF["name"] == "Warrior"].iloc[0]
        deltas = role_relevant_stat_deltas(row, ROLE_PROFILES["Physical Attacker"])
        self.assertEqual([stat for stat, _ in deltas], ["Str", "Spd"])  # Str weight 1.0 > Spd weight 0.5

    def test_empty_when_role_weights_stat_absent_from_row(self):
        row = pd.Series({"HP": 1})
        self.assertEqual(role_relevant_stat_deltas(row, {"Str": 1.0}), [])


if __name__ == "__main__":
    unittest.main()
