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
        if hasattr(st, "MARKDOWN_CALLS"):
            st.MARKDOWN_CALLS.clear()
        if hasattr(st, "SELECTBOX_CALLS"):
            st.SELECTBOX_CALLS.clear()
        if hasattr(st, "CAPTION_CALLS"):
            st.CAPTION_CALLS.clear()
        (
            self.base_stats_df, self.growth_rates_df, self.stat_boosts_df, self.eligibility_df,
            self.character_gender_df, self.weapon_req_df, self.character_weapon_talent_df,
            self.recruitment_requirements_df, self.starting_level_df, self.class_growth_df,
            self.class_base_stats_df, self.character_relics_df,
        ) = app.load_data()
        self.playable_names = app.get_playable_names(self.base_stats_df)
        self.dlc_names = app.get_dlc_names(self.base_stats_df)

    def _render_character_tab(self):
        app.render_character_tab(
            self.base_stats_df, self.growth_rates_df, self.stat_boosts_df, self.eligibility_df,
            self.character_gender_df, self.weapon_req_df, self.character_weapon_talent_df,
            self.starting_level_df, self.class_growth_df, self.class_base_stats_df,
            self.character_relics_df, self.playable_names, self.dlc_names,
        )

    def _render_team_tab(self):
        app.render_team_tab(
            self.base_stats_df, self.growth_rates_df, self.stat_boosts_df, self.eligibility_df,
            self.character_gender_df, self.weapon_req_df, self.character_weapon_talent_df,
            self.recruitment_requirements_df, self.starting_level_df, self.class_growth_df,
            self.class_base_stats_df, self.character_relics_df, self.playable_names, self.dlc_names,
        )

    def test_character_tab_default_widgets(self):
        st.WIDGET_OVERRIDES = {"char_select": "Byleth" if "Byleth" in self.playable_names else self.playable_names[0]}
        self._render_character_tab()

    def test_character_tab_portrait_is_color_coded_by_house(self):
        # Item 8 (route-based color coding): the fallback portrait tile
        # should be tinted per the character's own house/route (see
        # app.HOUSE_COLORS), not a single flat neutral box for everyone.
        if not hasattr(st, "MARKDOWN_CALLS"):
            self.skipTest("stub-only assertion - MARKDOWN_CALLS isn't part of the real streamlit API")
        st.WIDGET_OVERRIDES = {"char_select": "Bernadetta"}  # Black Eagles
        self._render_character_tab()
        portrait_html = next(html for html in st.MARKDOWN_CALLS if "div title=" in html)
        self.assertIn(app.HOUSE_COLORS["Black Eagles"], portrait_html)
        self.assertIn("Black Eagles", portrait_html)

    def test_team_tab_portraits_are_color_coded_by_house(self):
        if not hasattr(st, "MARKDOWN_CALLS"):
            self.skipTest("stub-only assertion - MARKDOWN_CALLS isn't part of the real streamlit API")
        st.WIDGET_OVERRIDES = {"team_route": "Blue Lions", "Build Team": True}
        self._render_team_tab()
        portrait_htmls = [html for html in st.MARKDOWN_CALLS if "div title=" in html]
        self.assertTrue(portrait_htmls)
        # Every team member on a Blue Lions build is either a Blue Lions
        # student or Church/Knights of Seiros/Protagonist staff (see
        # get_candidate_pool) - whichever house each one's tile reports,
        # it should be a real, colored house, not the neutral fallback.
        for html in portrait_htmls:
            self.assertNotIn(app.DEFAULT_PORTRAIT_COLOR, html)

    def test_character_tab_shows_growth_rate_mini_chart_for_every_tier(self):
        # Item 8: growth-rate visualization on the Character Optimizer
        # tab's own recommended-path view, not just the Class Explorer tab
        # - and showing more than the old 2-stat text summary (all of
        # STAT_COLS, per tier - see render_growth_rate_mini_chart).
        if not hasattr(st, "PLOTLY_CALLS"):
            self.skipTest("stub-only assertion - PLOTLY_CALLS isn't part of the real streamlit API")
        st.WIDGET_OVERRIDES = {"char_select": "Bernadetta", "char_role_select": "Physical Attacker"}
        st.PLOTLY_CALLS.clear()
        self._render_character_tab()

        growth_charts = [(key, fig) for key, fig in st.PLOTLY_CALLS if key.startswith("growth_mini_")]
        self.assertTrue(growth_charts)
        # Every mini chart shows all of STAT_COLS (more than the old 2-stat
        # text caption), one bar trace, horizontal.
        for _, fig in growth_charts:
            self.assertEqual(len(fig.traces), 1)
            trace = fig.traces[0]
            self.assertEqual(list(trace.kwargs["y"]), app.STAT_COLS)
            self.assertEqual(trace.kwargs["orientation"], "h")

    def test_character_tab_manual_role_and_mixmatch_override(self):
        st.WIDGET_OVERRIDES = {
            "char_select": "Bernadetta",
            "char_role_select": "Magic Attacker",
            "mixmatch_Bernadetta_Magic Attacker_Beginner": "Fighter",
        }
        self._render_character_tab()

    def test_character_tab_non_final_tier_override_changes_projected_stats_chart(self):
        # Item 1 regression guard, exercised through the real render path
        # (not just the optimizer function directly): overriding a
        # NON-FINAL mix-and-match tier must change the projected-stats bar
        # chart, not be silently ignored until the final tier also changes
        # - the class base-stat floor (item 6) is what makes this true now.
        if not hasattr(st, "PLOTLY_CALLS"):
            self.skipTest("stub-only assertion - PLOTLY_CALLS isn't part of the real streamlit API")

        st.WIDGET_OVERRIDES = {"char_select": "Bernadetta", "char_role_select": "Magic Attacker"}
        st.PLOTLY_CALLS.clear()
        self._render_character_tab()
        baseline_fig = next(fig for key, fig in st.PLOTLY_CALLS if key.startswith("bar_"))
        baseline_projected = next(t for t in baseline_fig.traces if t.kwargs["name"].startswith("Projected"))

        st.session_state.clear()
        st.WIDGET_OVERRIDES = {
            "char_select": "Bernadetta", "char_role_select": "Magic Attacker",
            "mixmatch_Bernadetta_Magic Attacker_Beginner": "Fighter",
        }
        st.PLOTLY_CALLS.clear()
        self._render_character_tab()
        overridden_fig = next(fig for key, fig in st.PLOTLY_CALLS if key.startswith("bar_"))
        overridden_projected = next(t for t in overridden_fig.traces if t.kwargs["name"].startswith("Projected"))

        self.assertNotEqual(baseline_projected.kwargs["y"], overridden_projected.kwargs["y"])

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

    def test_character_tab_import_build_uses_overridden_tier_requirement(self):
        # Regression test for the "now correctly imports changed classes
        # but displays wrong final class requirements" bug (spec item B):
        # overriding a tier's class via mix-and-match, then importing,
        # must carry that class's OWN requirement/score/warning - not the
        # ones cached from whichever class the tool originally recommended
        # for that tier.
        st.WIDGET_OVERRIDES = {
            "char_select": "Bernadetta",
            "char_role_select": "Magic Attacker",
            "mixmatch_Bernadetta_Magic Attacker_Beginner": "Fighter",
            "import_Bernadetta": True,
        }
        self._render_character_tab()
        imported = st.session_state["imported_builds"]["Bernadetta"]
        beginner_step = next(s for s in imported["path"] if s["tier"] == "Beginner")
        self.assertEqual(beginner_step["class"], "Fighter")
        # Fighter's real requirement, not Monk's (the Magic Attacker recommendation
        # this tier's dropdown was overridden away from).
        self.assertEqual(beginner_step["requirement"], "Axe D or Bow D or Brawling D")
        self.assertFalse(beginner_step["is_unique_class"])

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

    def test_team_tab_dancer_assignment_excludes_protagonist_and_is_single_select(self):
        # Item 9: Dancer is a single roster-wide slot, not something every
        # team member can be independently assigned - enforced here by (a)
        # the Protagonist never appearing in the option list at all (see
        # app.DANCER_INELIGIBLE_CHARACTERS) and (b) the control being one
        # selectbox (options list, index-picks-one), never a multiselect.
        st.WIDGET_OVERRIDES = {
            "team_route": "Blue Lions", "team_size": 6, "team_level": 30,
            "Build Team": True,
        }
        self._render_team_tab()
        team = st.session_state["team_result"]
        self.assertIn("Protagonist", [m["character"] for m in team])  # force-deployed, so on this team

        if not hasattr(st, "SELECTBOX_CALLS"):
            self.skipTest("stub-only assertion - SELECTBOX_CALLS isn't part of the real streamlit API")
        dancer_call = next(
            (key, options) for key, options in st.SELECTBOX_CALLS if key.startswith("team_dancer_select_")
        )
        _, dancer_options = dancer_call
        self.assertNotIn("Protagonist", dancer_options)
        self.assertIn("None", dancer_options)
        # Every other team member IS offered.
        for member in team:
            if member["character"] != "Protagonist":
                self.assertIn(member["character"], dancer_options)

    def test_team_tab_assigned_dancer_is_reflected_on_their_card(self):
        st.WIDGET_OVERRIDES = {
            "team_route": "Blue Lions", "team_size": 6, "team_level": 30,
            "Build Team": True,
        }
        self._render_team_tab()
        team = st.session_state["team_result"]
        non_protagonist = next(m["character"] for m in team if m["character"] != "Protagonist")
        dancer_widget_key = "team_dancer_select_" + ",".join(sorted(m["character"] for m in team))

        # Rerun without pressing Build again (same pattern as the
        # unrelated-rerun regression test above) - persisting a team while
        # assigning its Dancer, the way a real user would.
        st.WIDGET_OVERRIDES = {
            "team_route": "Blue Lions", "team_size": 6, "team_level": 30,
            "Build Team": False, "\U0001F3B2 Different team, same pool": False,
            dancer_widget_key: non_protagonist,
        }
        if hasattr(st, "CAPTION_CALLS"):
            st.CAPTION_CALLS.clear()
        self._render_team_tab()
        self.assertEqual(st.session_state["team_result"], team)  # unchanged, same as the rerun-survival test

        if not hasattr(st, "CAPTION_CALLS"):
            self.skipTest("stub-only assertion - CAPTION_CALLS isn't part of the real streamlit API")
        dancer_badges = [c for c in st.CAPTION_CALLS if "This team's Dancer" in c]
        # Exactly one member's card shows the badge - the single roster-wide slot, not a per-member option.
        self.assertEqual(len(dancer_badges), 1)

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

    def test_team_tab_full_roster_forces_byleth_but_no_lord(self):
        st.WIDGET_OVERRIDES = {
            "team_route": "Full roster", "team_size": 3, "team_level": 30,
            "Build Team": True,
        }
        self._render_team_tab()
        # "Full roster" has no single lord to force, but Byleth (the Protagonist) is always
        # force-deployed regardless of route.
        team = st.session_state["team_result"]
        self.assertEqual(len(team), 3)
        self.assertIn("Protagonist", [m["character"] for m in team])

    def test_team_tab_no_force_deployments_toggle_lets_byleth_be_omitted(self):
        st.WIDGET_OVERRIDES = {
            "team_route": "Golden Deer", "team_size": 4, "team_level": 30,
            "team_force_deployments": False,
            "Build Team": True,
        }
        self._render_team_tab()
        # Just needs to not crash - the toggle removes the forced inclusion, it doesn't guarantee
        # Byleth/Claude are actually excluded (they may still win their role on fit alone).
        self.assertEqual(len(st.session_state["team_result"]), 4)

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
        app.render_class_explorer_tab(self.stat_boosts_df, self.weapon_req_df, self.class_growth_df)

    def test_class_explorer_tab_compare_two_classes(self):
        st.WIDGET_OVERRIDES = {
            "explorer_class_a": "Gremory", "explorer_class_b": "Bishop", "explorer_include_dlc": True,
        }
        app.render_class_explorer_tab(self.stat_boosts_df, self.weapon_req_df, self.class_growth_df)


if __name__ == "__main__":
    unittest.main()
