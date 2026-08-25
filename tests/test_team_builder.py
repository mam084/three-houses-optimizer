"""
tests/test_team_builder.py

Unit tests for src/team_builder.py. See test_optimizer.py's module
docstring for why this uses stdlib unittest, real data/ CSVs, and
module-level fixtures instead of a TestCase mixin's setUpClass. Run with:

    python -m unittest discover -s tests -v
"""

import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.team_builder import (
    BLACK_EAGLES_CRIMSON_FLOWER,
    BLACK_EAGLES_SILVER_SNOW,
    DLC_HOUSE,
    REAL_ROUTES,
    ROUTE_LORD,
    SILVER_SNOW_LOST_CHARACTERS,
    build_balanced_team,
    build_team_with_paths,
    cross_house_names_in_pool,
    get_candidate_pool,
    is_cross_house_recruitable,
    load_recruitment_lookup,
    mandatory_names_for_route,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_STATS_DF = pd.read_csv(DATA_DIR / "character_base_stats.csv")
GROWTH_RATES_DF = pd.read_csv(DATA_DIR / "character_growth_rates.csv")
STAT_BOOSTS_DF = pd.read_csv(DATA_DIR / "class_stat_boosts.csv")
ELIGIBILITY_DF = pd.read_csv(DATA_DIR / "class_eligibility.csv")
CHARACTER_GENDER_DF = pd.read_csv(DATA_DIR / "character_gender.csv")
WEAPON_REQ_DF = pd.read_csv(DATA_DIR / "class_weapon_requirements.csv")
CHARACTER_WEAPON_TALENT_DF = pd.read_csv(DATA_DIR / "character_weapon_talent.csv")
RECRUITMENT_REQUIREMENTS_DF = pd.read_csv(DATA_DIR / "recruitment_requirements.csv")
STARTING_LEVEL_DF = pd.read_csv(DATA_DIR / "character_starting_level.csv")
RECRUITMENT_LOOKUP = load_recruitment_lookup(RECRUITMENT_REQUIREMENTS_DF)
PLAYABLE_NAMES = sorted(n for n in BASE_STATS_DF["name"] if "(NPC)" not in n)


class TestSilverSnow(unittest.TestCase):
    """
    Regression coverage for the "Silver Snow route" request: Edelgard and
    Hubert both leave permanently and become unrecruitable on Silver Snow,
    while Crimson Flower keeps the Black Eagles roster intact.
    """

    def test_silver_snow_excludes_edelgard_and_hubert(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=BLACK_EAGLES_SILVER_SNOW)
        self.assertNotIn("Edelgard", pool)
        self.assertNotIn("Hubert", pool)

    def test_silver_snow_keeps_the_rest_of_black_eagles(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=BLACK_EAGLES_SILVER_SNOW)
        for name in ("Ferdinand", "Linhardt", "Caspar", "Bernadetta", "Dorothea", "Petra"):
            self.assertIn(name, pool)

    def test_crimson_flower_keeps_edelgard_and_hubert(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=BLACK_EAGLES_CRIMSON_FLOWER)
        self.assertIn("Edelgard", pool)
        self.assertIn("Hubert", pool)

    def test_both_black_eagles_routes_include_church_and_knights_staff(self):
        for route in (BLACK_EAGLES_CRIMSON_FLOWER, BLACK_EAGLES_SILVER_SNOW):
            pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=route)
            self.assertIn("Seteth", pool)  # Church of Seiros
            self.assertIn("Catherine", pool)  # Knights of Seiros
            self.assertIn("Protagonist", pool)

    def test_silver_snow_pool_is_exactly_crimson_flower_minus_two(self):
        crimson = set(get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=BLACK_EAGLES_CRIMSON_FLOWER))
        silver = set(get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=BLACK_EAGLES_SILVER_SNOW))
        self.assertEqual(crimson - silver, SILVER_SNOW_LOST_CHARACTERS)

    def test_all_four_routes_are_distinct_selectable_options(self):
        self.assertEqual(len(REAL_ROUTES), 4)
        self.assertEqual(len(set(REAL_ROUTES)), 4)


class TestCandidatePool(unittest.TestCase):
    def test_full_roster_excludes_dlc_by_default(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=None, include_dlc=False)
        self.assertNotIn("Yuri", pool)

    def test_full_roster_includes_dlc_when_requested(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=None, include_dlc=True)
        self.assertIn("Yuri", pool)

    def test_route_without_dlc_flag_never_includes_dlc(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions", include_dlc=False)
        dlc_names = set(BASE_STATS_DF[BASE_STATS_DF["house"] == DLC_HOUSE]["name"])
        self.assertEqual(set(pool) & dlc_names, set())

    def test_route_pool_excludes_other_houses_students_by_default(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions")
        self.assertNotIn("Claude", pool)
        self.assertNotIn("Edelgard", pool)


class TestCrossHouseRecruitment(unittest.TestCase):
    def test_house_leaders_and_retainers_never_cross_house_recruitable(self):
        for name in ("Edelgard", "Hubert", "Dimitri", "Dedue", "Claude"):
            self.assertFalse(is_cross_house_recruitable(name, RECRUITMENT_LOOKUP, target_level=40))

    def test_students_gated_by_byleth_level_requirement(self):
        # Felix requires Byleth level 15+.
        self.assertFalse(is_cross_house_recruitable("Felix", RECRUITMENT_LOOKUP, target_level=10))
        self.assertTrue(is_cross_house_recruitable("Felix", RECRUITMENT_LOOKUP, target_level=15))
        self.assertTrue(is_cross_house_recruitable("Felix", RECRUITMENT_LOOKUP, target_level=30))

    def test_disabled_by_default(self):
        pool = get_candidate_pool(
            BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions",
            recruitment_lookup=RECRUITMENT_LOOKUP, target_level=40,
        )
        self.assertNotIn("Claude", pool)  # include_cross_house_recruits defaults False

    def test_enabling_adds_eligible_cross_house_students_at_high_level(self):
        pool = get_candidate_pool(
            BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions",
            include_cross_house_recruits=True, target_level=40, recruitment_lookup=RECRUITMENT_LOOKUP,
        )
        self.assertIn("Lysithea", pool)  # Golden Deer, requires level 15
        self.assertIn("Ferdinand", pool)  # Black Eagles, requires level 10

    def test_still_excludes_locked_lords_even_when_enabled(self):
        pool = get_candidate_pool(
            BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions",
            include_cross_house_recruits=True, target_level=40, recruitment_lookup=RECRUITMENT_LOOKUP,
        )
        self.assertNotIn("Claude", pool)
        self.assertNotIn("Edelgard", pool)
        self.assertNotIn("Hubert", pool)

    def test_low_target_level_excludes_high_requirement_recruits(self):
        pool = get_candidate_pool(
            BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions",
            include_cross_house_recruits=True, target_level=5, recruitment_lookup=RECRUITMENT_LOOKUP,
        )
        # Every cross-house requirement is level 10+, so at target_level 5 nothing new should be added
        # beyond Blue Lions' own pool.
        own_pool = set(get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions"))
        self.assertEqual(set(pool), own_pool)

    def test_full_roster_ignores_cross_house_flag(self):
        with_flag = get_candidate_pool(
            BASE_STATS_DF, PLAYABLE_NAMES, route="Full roster",
            include_cross_house_recruits=True, target_level=40, recruitment_lookup=RECRUITMENT_LOOKUP,
        )
        without_flag = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Full roster")
        self.assertEqual(set(with_flag), set(without_flag))

    def test_cross_house_names_in_pool_excludes_native_students(self):
        pool = get_candidate_pool(
            BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions",
            include_cross_house_recruits=True, target_level=40, recruitment_lookup=RECRUITMENT_LOOKUP,
        )
        cross_names = cross_house_names_in_pool(BASE_STATS_DF, pool, "Blue Lions")
        # Dimitri/Dedue/Felix etc. are native Blue Lions - never "cross-house" even though they're in the pool.
        self.assertNotIn("Dimitri", cross_names)
        self.assertNotIn("Felix", cross_names)
        # Lysithea (Golden Deer) genuinely is a cross-house addition here.
        self.assertIn("Lysithea", cross_names)

    def test_cross_house_names_empty_for_full_roster(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Full roster")
        self.assertEqual(cross_house_names_in_pool(BASE_STATS_DF, pool, "Full roster"), set())

    def test_recruitment_note_only_attached_to_actual_cross_house_picks(self):
        # Regression test for a bug caught during development: recruitment_requirements.csv has a row
        # for every student (since each is recruitable onto the OTHER two routes), so a naive
        # "is this character in the lookup" check wrongly annotated a route's OWN students (e.g. Black
        # Eagles' Petra/Linhardt) as "recruited from another house." The note must only ever attach to
        # names cross_house_names_in_pool actually flagged as cross-house for this route.
        pool = get_candidate_pool(
            BASE_STATS_DF, PLAYABLE_NAMES, route=BLACK_EAGLES_CRIMSON_FLOWER,
            include_cross_house_recruits=True, target_level=40, recruitment_lookup=RECRUITMENT_LOOKUP,
        )
        cross_names = cross_house_names_in_pool(BASE_STATS_DF, pool, BLACK_EAGLES_CRIMSON_FLOWER)
        team = build_team_with_paths(
            pool, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF, team_size=len(pool),
            target_level=40, eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            recruitment_lookup=RECRUITMENT_LOOKUP, cross_house_names=cross_names,
        )
        for member in team:
            has_note = "Recruited from another house" in member["why"]
            if member["character"] in ("Petra", "Linhardt", "Ferdinand", "Caspar", "Bernadetta", "Dorothea"):
                self.assertFalse(has_note, f"{member['character']} is native Black Eagles, shouldn't get a note")
            elif member["character"] in cross_names:
                self.assertTrue(has_note, f"{member['character']} is cross-house, should get a note")


class TestBuildBalancedTeam(unittest.TestCase):
    def test_deterministic_without_rng(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Golden Deer")
        team_a = build_balanced_team(pool, GROWTH_RATES_DF, team_size=6)
        team_b = build_balanced_team(pool, GROWTH_RATES_DF, team_size=6)
        self.assertEqual([m["character"] for m in team_a], [m["character"] for m in team_b])

    def test_must_include_forces_membership(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Golden Deer")
        team = build_balanced_team(pool, GROWTH_RATES_DF, team_size=6, must_include=["Raphael"])
        self.assertIn("Raphael", [m["character"] for m in team])

    def test_exclude_prevents_membership(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Full roster")
        team = build_balanced_team(pool, GROWTH_RATES_DF, team_size=30, exclude=["Dimitri"])
        self.assertNotIn("Dimitri", [m["character"] for m in team])

    def test_must_include_wins_over_exclude(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Golden Deer")
        team = build_balanced_team(
            pool, GROWTH_RATES_DF, team_size=6, must_include=["Raphael"], exclude=["Raphael"],
        )
        self.assertIn("Raphael", [m["character"] for m in team])

    def test_seeded_rng_is_reproducible(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Full roster")
        team_a = build_balanced_team(pool, GROWTH_RATES_DF, team_size=8, rng=np.random.default_rng(42))
        team_b = build_balanced_team(pool, GROWTH_RATES_DF, team_size=8, rng=np.random.default_rng(42))
        self.assertEqual([m["character"] for m in team_a], [m["character"] for m in team_b])

    def test_team_size_larger_than_pool_returns_whole_pool(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Golden Deer")
        team = build_balanced_team(pool, GROWTH_RATES_DF, team_size=999)
        self.assertEqual(len(team), len(pool))


class TestBuildTeamWithPaths(unittest.TestCase):
    def test_every_member_has_a_path_and_requirement_field(self):
        pool = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route="Blue Lions")
        team = build_team_with_paths(
            pool, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF, team_size=6, target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
        )
        for member in team:
            self.assertTrue(member["path"])
            self.assertIn("requirement", member["path"][0])

    def test_default_target_level_is_30(self):
        self.assertEqual(inspect.signature(build_team_with_paths).parameters["target_level"].default, 30)


class TestRouteSweep(unittest.TestCase):
    def test_every_route_dlc_cross_house_combo_builds_without_error(self):
        for route in [None, "Full roster"] + REAL_ROUTES:
            for include_dlc in (False, True):
                for cross in (False, True):
                    with self.subTest(route=route, include_dlc=include_dlc, cross=cross):
                        pool = get_candidate_pool(
                            BASE_STATS_DF, PLAYABLE_NAMES, route=route, include_dlc=include_dlc,
                            include_cross_house_recruits=cross, target_level=30,
                            recruitment_lookup=RECRUITMENT_LOOKUP,
                        )
                        cross_names = cross_house_names_in_pool(BASE_STATS_DF, pool, route)
                        team = build_team_with_paths(
                            pool, BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
                            team_size=6, target_level=30,
                            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
                            recruitment_lookup=RECRUITMENT_LOOKUP, cross_house_names=cross_names,
                            weapon_req_df=WEAPON_REQ_DF,
                            character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
                        )
                        self.assertIsInstance(team, list)


class TestMandatoryNames(unittest.TestCase):
    def test_full_roster_still_forces_protagonist_but_no_lord(self):
        # Byleth (the Protagonist) is force-deployed on every route, including
        # "Full roster"/None - there's no route on which they're actually
        # benchable - but a mixed-route roster has no single lord to force.
        self.assertEqual(mandatory_names_for_route("Full roster"), ["Protagonist"])
        self.assertEqual(mandatory_names_for_route(None), ["Protagonist"])

    def test_each_real_route_forces_protagonist_and_its_own_lord(self):
        for route in REAL_ROUTES:
            with self.subTest(route=route):
                mandatory = mandatory_names_for_route(route)
                self.assertEqual(mandatory, ["Protagonist", ROUTE_LORD[route]])

    def test_both_black_eagles_routes_share_edelgard_as_lord(self):
        self.assertEqual(
            mandatory_names_for_route(BLACK_EAGLES_CRIMSON_FLOWER),
            mandatory_names_for_route(BLACK_EAGLES_SILVER_SNOW),
        )

    def test_team_built_with_mandatory_names_always_includes_them(self):
        route = "Golden Deer"
        candidates = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=route)
        mandatory = mandatory_names_for_route(route)
        team = build_balanced_team(candidates, GROWTH_RATES_DF, team_size=4, must_include=mandatory)
        team_names = [m["character"] for m in team]
        for name in mandatory:
            self.assertIn(name, team_names)

    def test_force_deployed_names_get_force_deployed_why_text_not_by_request(self):
        # "Force deployments currently say they're on the team 'by request'
        # - should say they're force deployed" - the original report.
        route = "Golden Deer"
        candidates = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=route)
        mandatory = mandatory_names_for_route(route)
        team = build_balanced_team(
            candidates, GROWTH_RATES_DF, team_size=4, must_include=mandatory, force_deployed=set(mandatory),
        )
        by_name = {m["character"]: m for m in team}
        for name in mandatory:
            self.assertIn("Force-deployed", by_name[name]["why"])
            self.assertNotIn("Included by request", by_name[name]["why"])

    def test_a_plain_must_include_name_still_says_by_request(self):
        route = "Golden Deer"
        candidates = get_candidate_pool(BASE_STATS_DF, PLAYABLE_NAMES, route=route)
        team = build_balanced_team(
            candidates, GROWTH_RATES_DF, team_size=4, must_include=["Lysithea"], force_deployed=set(),
        )
        by_name = {m["character"]: m for m in team}
        self.assertIn("Included by request", by_name["Lysithea"]["why"])
        self.assertNotIn("Force-deployed", by_name["Lysithea"]["why"])


class TestLockedBuilds(unittest.TestCase):
    def test_locked_build_is_used_instead_of_a_fresh_recommendation(self):
        locked_path = [{"tier": "Beginner", "class": "Fighter", "score": 1.0, "why": "test",
                         "requirement": None, "is_unique_class": False}]
        locked_stats = {stat: 99.9 for stat in ("HP", "Str", "Mag", "Dex", "Spd", "Lck", "Def", "Res", "Cha")}
        team = build_team_with_paths(
            ["Bernadetta", "Ashe", "Ignatz"], BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            team_size=3, target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF,
            locked_builds={"Bernadetta": {
                "path": locked_path, "final_class": "Fighter", "expected_final_stats": locked_stats,
                "eligible_unique_classes": [],
            }},
        )
        member = next(m for m in team if m["character"] == "Bernadetta")
        self.assertEqual(member["final_class"], "Fighter")
        self.assertEqual(member["expected_final_stats"], locked_stats)
        self.assertIn("imported build", member["why"])

    def test_characters_without_a_locked_build_are_unaffected(self):
        team = build_team_with_paths(
            ["Bernadetta", "Ashe", "Ignatz"], BASE_STATS_DF, GROWTH_RATES_DF, STAT_BOOSTS_DF,
            team_size=3, target_level=30,
            eligibility_df=ELIGIBILITY_DF, character_gender_df=CHARACTER_GENDER_DF,
            weapon_req_df=WEAPON_REQ_DF, character_weapon_talent_df=CHARACTER_WEAPON_TALENT_DF,
            starting_level_df=STARTING_LEVEL_DF,
            locked_builds={"Bernadetta": {
                "path": [{"tier": "Beginner", "class": "Fighter", "score": 1.0, "why": "test",
                          "requirement": None, "is_unique_class": False}],
                "final_class": "Fighter", "expected_final_stats": {}, "eligible_unique_classes": [],
            }},
        )
        others = [m for m in team if m["character"] != "Bernadetta"]
        for member in others:
            self.assertNotIn("imported build", member["why"])
            self.assertTrue(member["path"])  # recomputed normally


if __name__ == "__main__":
    unittest.main()
