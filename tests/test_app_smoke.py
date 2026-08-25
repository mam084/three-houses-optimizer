"""
tests/test_app_smoke.py

Structural smoke tests for app.py: does each render function run to
completion, without raising, across a handful of representative widget
inputs? This is NOT a substitute for actually running
`streamlit run app.py` and looking at it - see tests/stubs/streamlit.py's
docstring - but it does catch the class of bug an import-and-read review
alone can miss (a typo'd variable name, a wrong argument, a KeyError that
only shows up once real data flows through a particular branch), which
matters here since this project's development environment doesn't always
have a real Streamlit/Plotly install available to run the app directly.

Run with:

    python -m unittest discover -s tests -v

(tests/stubs/ is added to sys.path automatically, below, before app.py is
imported - real Streamlit/Plotly are still preferred if actually
installed, since sys.path.append puts the stub after any already-installed
real package on sys.path.)
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STUBS_DIR = Path(__file__).resolve().parent / "stubs"
for path in (str(REPO_ROOT), str(STUBS_DIR)):
    if path not in sys.path:
        sys.path.append(path)

try:
    import streamlit as st  # real install if present, else tests/stubs/streamlit.py
    import app
except Exception as exc:  # pragma: no cover - environment-dependent
    st = None
    app = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(app is None, f"could not import app.py / streamlit: {IMPORT_ERROR}")
class TestAppSmoke(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()
        st.WIDGET_OVERRIDES = {}
        (
            self.base_stats_df, self.growth_rates_df, self.stat_boosts_df, self.eligibility_df,
            self.character_gender_df, self.weapon_req_df, self.character_weapon_talent_df,
            self.recruitment_requirements_df, self.starting_level_df,
        ) = app.load_data()
        self.playable_names = app.get_playable_names(self.base_stats_df)
        self.dlc_names = app.get_dlc_names(self.base_stats_df)

    def _render_character_tab(self):
        app.render_character_tab(
            self.base_stats_df, self.growth_rates_df, self.stat_boosts_df, self.eligibility_df,
            self.character_gender_df, self.weapon_req_df, self.character_weapon_talent_df,
            self.starting_level_df, self.playable_names, self.dlc_names,
        )

    def _render_team_tab(self):
        app.render_team_tab(
            self.base_stats_df, self.growth_rates_df, self.stat_boosts_df, self.eligibility_df,
            self.character_gender_df, self.weapon_req_df, self.character_weapon_talent_df,
            self.recruitment_requirements_df, self.starting_level_df, self.playable_names, self.dlc_names,
        )

    def test_character_tab_default_widgets(self):
        st.WIDGET_OVERRIDES = {"char_select": "Byleth" if "Byleth" in self.playable_names else self.playable_names[0]}
        self._render_character_tab()

    def test_character_tab_manual_role_and_mixmatch_override(self):
        st.WIDGET_OVERRIDES = {
            "char_select": "Bernadetta",
            "char_role_select": "Magic Attacker",
            "mixmatch_Bernadetta_Beginner": "Fighter",
        }
        self._render_character_tab()

    def test_character_tab_every_playable_character(self):
        # Cheap insurance against a KeyError/IndexError that only fires for one specific character's data.
        for name in self.playable_names:
            with self.subTest(character=name):
                st.session_state.clear()
                st.WIDGET_OVERRIDES = {"char_select": name}
                self._render_character_tab()

    def test_character_tab_join_level_note_for_late_recruit(self):
        # Catherine joins at level 15 per data/character_starting_level.csv - a target level below
        # that should get silently floored, not crash or produce a negative "levels gained".
        st.WIDGET_OVERRIDES = {"char_select": "Catherine", "char_level": 5}
        self._render_character_tab()

    def test_character_tab_dlc_classes_toggle(self):
        st.WIDGET_OVERRIDES = {"char_select": "Mercedes", "char_include_dlc_classes": True}
        self._render_character_tab()

    def test_character_tab_import_build_button_populates_session_state(self):
        st.WIDGET_OVERRIDES = {"char_select": "Bernadetta", f"import_Bernadetta": True}
        self._render_character_tab()
        self.assertIn("Bernadetta", st.session_state.get("imported_builds", {}))
        imported = st.session_state["imported_builds"]["Bernadetta"]
        self.assertIn("final_class", imported)
        self.assertIn("expected_final_stats", imported)
        self.assertIn("path", imported)

    def test_team_tab_build_then_survives_unrelated_rerun(self):
        st.WIDGET_OVERRIDES = {
            "team_route": "Black Eagles (Silver Snow)", "team_size": 6, "team_level": 30,
            "Build Team": True,
        }
        self._render_team_tab()
        self.assertIn("team_result", st.session_state)
        first_team = st.session_state["team_result"]
        self.assertEqual(len(first_team), 6)
        self.assertNotIn("Edelgard", [m["character"] for m in first_team])
        self.assertNotIn("Hubert", [m["character"] for m in first_team])

        # Regression test for the "team results disappearing on unrelated clicks" bug: a rerun where
        # neither button was pressed (simulating the user touching some other widget) must NOT clear
        # the previously built team.
        st.WIDGET_OVERRIDES = {
            "team_route": "Black Eagles (Silver Snow)", "team_size": 6, "team_level": 30,
            "Build Team": False, "\U0001F3B2 Different team, same pool": False,
        }
        self._render_team_tab()
        self.assertEqual(st.session_state["team_result"], first_team)

    def test_team_tab_cross_house_recruitment_and_dlc(self):
        st.WIDGET_OVERRIDES = {
            "team_route": "Blue Lions", "team_size": 8, "team_level": 20,
            "team_include_dlc": True, "team_include_cross_house": True, "team_include_dlc_classes": True,
            "Build Team": True,
        }
        self._render_team_tab()
        self.assertEqual(len(st.session_state["team_result"]), 8)

    def test_team_tab_shuffle_button(self):
        st.WIDGET_OVERRIDES = {
            "team_route": "Full roster", "team_size": 10, "team_level": 30,
            "\U0001F3B2 Different team, same pool": True,
        }
        self._render_team_tab()
        self.assertEqual(len(st.session_state["team_result"]), 10)

    def test_team_tab_no_button_pressed_renders_nothing_but_does_not_crash(self):
        st.WIDGET_OVERRIDES = {"team_route": "Full roster"}
        self._render_team_tab()
        self.assertNotIn("team_result", st.session_state)

    def test_team_tab_route_forces_byleth_and_lord(self):
        # Byleth and the route's own lord are force-deployed - always on the built team for a
        # real route (team_size large enough that round-robin alone wouldn't guarantee it).
        st.WIDGET_OVERRIDES = {
            "team_route": "Golden Deer", "team_size": 4, "team_level": 30,
            "Build Team": True,
        }
        self._render_team_tab()
        members = [m["character"] for m in st.session_state["team_result"]]
        self.assertIn("Protagonist", members)
        self.assertIn("Claude", members)

    def test_team_tab_full_roster_has_no_mandatory_names(self):
        st.WIDGET_OVERRIDES = {
            "team_route": "Full roster", "team_size": 3, "team_level": 30,
            "Build Team": True,
        }
        self._render_team_tab()
        # Just needs to not crash and produce a team - "Full roster" has no single lord to force.
        self.assertEqual(len(st.session_state["team_result"]), 3)

    def test_team_tab_uses_imported_build(self):
        st.session_state["imported_builds"] = {
            "Bernadetta": {
                "path": [{"tier": "Beginner", "class": "Fighter", "score": 1.0, "why": "test",
                          "requirement": None, "is_unique_class": False}],
                "final_class": "Fighter",
                "expected_final_stats": {s: 10 for s in app.STAT_COLS},
                "eligible_unique_classes": [],
            }
        }
        st.WIDGET_OVERRIDES = {
            "team_route": "Black Eagles (Crimson Flower)", "team_size": 6, "team_level": 30,
            "Build Team": True,
        }
        self._render_team_tab()
        members = {m["character"]: m for m in st.session_state["team_result"]}
        self.assertIn("Bernadetta", members)
        self.assertEqual(members["Bernadetta"]["final_class"], "Fighter")

    def test_class_explorer_tab_default(self):
        app.render_class_explorer_tab(self.stat_boosts_df, self.weapon_req_df)

    def test_class_explorer_tab_compare_two_classes(self):
        st.WIDGET_OVERRIDES = {
            "explorer_class_a": "Gremory", "explorer_class_b": "Bishop", "explorer_include_dlc": True,
        }
        app.render_class_explorer_tab(self.stat_boosts_df, self.weapon_req_df)


if __name__ == "__main__":
    unittest.main()
