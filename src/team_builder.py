"""
team_builder.py

Recommends a balanced team from a pool of candidate characters, covering
complementary roles rather than optimizing one character in isolation
(that's what optimizer.py already does).

Approach: auto-detect each candidate's natural role (reusing
optimizer.detect_natural_role), group by role, then build the team via
round-robin selection across role groups - each round, pick the
highest-scoring not-yet-picked character from the next role in rotation
(or a weighted-random pick among them, if variety is requested - see
build_balanced_team's rng parameter). This favors role diversity over just
stacking the single "best" characters, which would otherwise tend to
produce redundant teams (e.g. five Physical Attackers and no Tank).

This is a greedy heuristic, not a global optimum over all possible team
combinations - deliberately so, since brute-force search over ~40 choose 6
candidates is both slow and hard to explain. Round-robin-by-role is fast,
its default (no rng) behavior is deterministic, and its reasoning is easy
to state plainly: "balance role coverage first, favor strength within
that."
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer import ROLE_PROFILES, detect_natural_role, recommend_for_character

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The three houses you actually pick a route through in the base game.
# "Church of Seiros" and "Knights of Seiros" aren't routes - they're the
# Officers Academy staff and knights who are recruitable as playable units
# regardless of which house route you chose, so a candidate pool for "the
# Black Eagles route" should include them too, not just Black Eagles
# students. "You and the Enigmatic Girl" is the Protagonist's own
# placeholder house from the base-stats data (Sothis, its only other
# member, is an NPC and already excluded elsewhere) - folded in here so a
# route pool includes the Protagonist without surfacing that odd label as
# its own selectable pool.
REAL_ROUTES = ["Black Eagles", "Blue Lions", "Golden Deer"]
ROUTE_INCLUDES_HOUSES = {
    route: [route, "Church of Seiros", "Knights of Seiros", "You and the Enigmatic Girl"]
    for route in REAL_ROUTES
}
DLC_HOUSE = "DLC-exclusive"


def get_candidate_pool(
    base_stats_df: pd.DataFrame,
    playable_names: list[str],
    route: str | None = None,
    include_dlc: bool = False,
) -> list[str]:
    """
    Build a candidate pool reflecting who's actually available to recruit.

    route=None (or "Full roster") returns every playable character. A
    specific route (e.g. "Black Eagles") returns that house's students PLUS
    the Protagonist and the Church/Knights of Seiros staff, who are
    recruitable on any route - narrowing to just the named house was the
    "team builder should access all characters available in that route"
    gap. DLC-exclusive characters (Cindered Shadows) are opt-in via
    include_dlc, since they require separately-owned DLC and are otherwise
    not "available" the way base-roster characters are - this applies to
    both the full roster and a specific route.

    playable_names should already exclude NPCs (see get_playable_names in
    app.py) - this function doesn't re-check that.
    """
    playable_df = base_stats_df[base_stats_df["name"].isin(playable_names)]

    if route is None or route == "Full roster":
        pool_df = playable_df
    else:
        houses = ROUTE_INCLUDES_HOUSES.get(route, [route])
        pool_df = playable_df[playable_df["house"].isin(houses)]

    if not include_dlc:
        pool_df = pool_df[pool_df["house"] != DLC_HOUSE]

    return pool_df["name"].tolist()


def assign_roles(candidates: list[str], growth_rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each candidate character, auto-detect their best-fitting role.
    Returns a DataFrame: character, role, score.

    Growth rates are standardized against the full roster passed in
    (growth_rates_df), not just the candidate subset - this keeps role
    detection consistent with optimizer.recommend_for_character, which
    does the same. Pass the full roster's growth_rates_df here even if
    candidates is a smaller pool (e.g. one route), so "unusual" is measured
    against the whole cast, not just whoever's in the pool.
    """
    from src.optimizer import compute_roster_stat_stats

    roster_means, roster_stds = compute_roster_stat_stats(growth_rates_df)

    rows = []
    for name in candidates:
        growth_row = growth_rates_df[growth_rates_df["name"] == name]
        if growth_row.empty:
            continue
        role, score = detect_natural_role(growth_row.iloc[0], roster_means, roster_stds)
        rows.append({"character": name, "role": role, "score": score})

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _weighted_choice(rng: np.random.Generator, remaining: list) -> int:
    """
    Pick an index into `remaining` (a list of assign_roles row-tuples),
    weighted toward higher-scoring candidates rather than always taking the
    top one. Scores can be negative (cosine similarity ranges -1..1), so
    weights are shifted to be non-negative before sampling; a small floor
    keeps every candidate a nonzero chance rather than making the lowest
    scorer unreachable.
    """
    scores = np.array([max(r.score, 0.0) for r in remaining])
    weights = scores - scores.min() + 0.05
    weights = weights / weights.sum()
    return int(rng.choice(len(remaining), p=weights))


def build_balanced_team(
    candidates: list[str],
    growth_rates_df: pd.DataFrame,
    team_size: int = 6,
    must_include: list[str] | None = None,
    exclude: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Build a team of team_size characters from the candidate pool, balancing
    role coverage via round-robin selection across each character's
    auto-detected best role.

    must_include: characters to force onto the team before round-robin
    fills the rest (e.g. "I'm building around Lysithea"). Each still gets a
    role assignment and its normal "why" explanation, just picked first.
    Silently ignored if a name isn't in candidates/growth_rates_df.

    exclude: characters to drop from the candidate pool before building -
    "give me a team, but not with Raphael" - applied after must_include, so
    excluding someone you also must_include is a no-op for them (must_include
    wins) rather than a silent contradiction.

    rng: if given, each round-robin pick is a weighted-random choice among
    the remaining candidates for that role (weighted toward higher scores)
    instead of always the single top scorer - this is what makes repeated
    calls (e.g. a "regenerate" button) return different teams instead of
    the same one every time, while still respecting role balance and each
    character's actual fit. Omit (the default) for the original
    deterministic top-score behavior.

    Returns a list of {"character", "role", "score", "why"} dicts, in the
    order they were picked.
    """
    must_include = must_include or []
    exclude = set(exclude or [])

    working_candidates = [c for c in candidates if c not in exclude]
    assignments = assign_roles(working_candidates, growth_rates_df)
    if assignments.empty:
        return []

    by_character = assignments.set_index("character")
    team: list[dict] = []
    picked = set()

    for name in must_include:
        if name in picked or name not in by_character.index or len(team) >= team_size:
            continue
        row = by_character.loc[name]
        team.append({
            "character": name,
            "role": row["role"],
            "score": round(float(row["score"]), 3),
            "why": "Included by request.",
        })
        picked.add(name)

    remaining_assignments = assignments[~assignments["character"].isin(picked)]
    role_groups = {
        role: list(group.itertuples(index=False))
        for role, group in remaining_assignments.groupby("role")
    }
    role_order = sorted(role_groups.keys())
    role_cursor = {role: 0 for role in role_order}
    role_pick_count = {role: 0 for role in role_order}

    while len(team) < team_size:
        picked_this_pass = False

        for role in role_order:
            if len(team) >= team_size:
                break

            group = role_groups[role]
            cursor = role_cursor[role]
            remaining = group[cursor:]
            if not remaining:
                continue  # this role's candidates are exhausted

            if rng is not None and len(remaining) > 1:
                choice_idx = _weighted_choice(rng, remaining)
                candidate = remaining[choice_idx]
                # swap the picked one to the front of the unconsumed slice
                # so the cursor still just advances by one each time
                group[cursor], group[cursor + choice_idx] = group[cursor + choice_idx], group[cursor]
                candidate = group[cursor]
                variety_note = " (weighted-random pick among remaining candidates for variety, not the single top score)"
            else:
                candidate = remaining[0]
                variety_note = ""

            role_cursor[role] += 1
            role_pick_count[role] += 1
            if rng is not None:
                rank_note = f"Filled the {'first' if role_pick_count[role] == 1 else f'{role_pick_count[role]}{_ordinal_suffix(role_pick_count[role])}'} {role} slot"
            else:
                rank_note = (
                    f"Top-scoring {role} candidate not yet on the team"
                    if role_pick_count[role] == 1
                    else f"{role_pick_count[role]}{_ordinal_suffix(role_pick_count[role])} {role} slot filled "
                         f"(role rotation covers every role once before repeating one)"
                )
            team.append({
                "character": candidate.character,
                "role": candidate.role,
                "score": round(candidate.score, 3),
                "why": rank_note + variety_note + ".",
            })
            picked_this_pass = True

        if not picked_this_pass:
            break  # every role group exhausted, fewer candidates than team_size

    return team


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def build_team_with_paths(
    candidates: list[str],
    base_stats_df: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
    stat_boosts_df: pd.DataFrame,
    team_size: int = 6,
    target_level: int = 20,
    eligibility_df: pd.DataFrame | None = None,
    character_gender_df: pd.DataFrame | None = None,
    must_include: list[str] | None = None,
    exclude: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Like build_balanced_team, but also attaches each member's full class-path
    recommendation (reusing optimizer.recommend_for_character), so the team
    output is immediately useful rather than just a role assignment.

    eligibility_df and character_gender_df are passed straight through to
    recommend_for_character - see its docstring. Both optional and default
    to None (no eligibility filtering), for backward compatibility with
    existing callers. must_include/exclude/rng are passed straight through
    to build_balanced_team - see its docstring.
    """
    team = build_balanced_team(
        candidates, growth_rates_df, team_size,
        must_include=must_include, exclude=exclude, rng=rng,
    )

    for member in team:
        full_rec = recommend_for_character(
            member["character"], base_stats_df, growth_rates_df, stat_boosts_df,
            role_name=member["role"], target_level=target_level,
            eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        )
        member["path"] = full_rec["path"]
        member["final_class"] = full_rec["path"][-1]["class"] if full_rec["path"] else None
        member["expected_final_stats"] = full_rec["expected_final_stats"]
        member["eligible_unique_classes"] = full_rec["eligible_unique_classes"]

    return team


def main():
    base_stats_df = pd.read_csv(DATA_DIR / "character_base_stats.csv")
    growth_rates_df = pd.read_csv(DATA_DIR / "character_growth_rates.csv")
    stat_boosts_df = pd.read_csv(DATA_DIR / "class_stat_boosts.csv")
    eligibility_df = pd.read_csv(DATA_DIR / "class_eligibility.csv")
    character_gender_df = pd.read_csv(DATA_DIR / "character_gender.csv")

    import argparse
    parser = argparse.ArgumentParser(description="Recommend a balanced Three Houses team.")
    parser.add_argument("--house", type=str, default=None,
                         help="Limit candidates to one route (e.g. 'Black Eagles') - includes that house's "
                              "students plus the Protagonist and Church/Knights of Seiros staff, who are "
                              "recruitable on any route. Omit for the full roster.")
    parser.add_argument("--size", type=int, default=6, help="Team size (default: 6)")
    parser.add_argument("--include-dlc", action="store_true",
                         help="Include DLC-exclusive characters (Cindered Shadows) in the candidate pool.")
    parser.add_argument("--must-include", type=str, default=None,
                         help="Comma-separated characters to force onto the team, e.g. 'Lysithea,Dedue'.")
    parser.add_argument("--exclude", type=str, default=None,
                         help="Comma-separated characters to drop from consideration.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for weighted-random variety instead of always the top-scoring pick "
                              "per role. Omit for the original deterministic behavior.")
    args = parser.parse_args()

    playable_names = base_stats_df[~base_stats_df["name"].str.contains(r"\(NPC\)")]["name"].tolist()
    candidates = get_candidate_pool(base_stats_df, playable_names, route=args.house, include_dlc=args.include_dlc)
    must_include = [n.strip() for n in args.must_include.split(",")] if args.must_include else None
    exclude = [n.strip() for n in args.exclude.split(",")] if args.exclude else None
    rng = np.random.default_rng(args.seed) if args.seed is not None else None

    team = build_team_with_paths(
        candidates, base_stats_df, growth_rates_df, stat_boosts_df, team_size=args.size,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        must_include=must_include, exclude=exclude, rng=rng,
    )

    dlc_names = set(base_stats_df[base_stats_df["house"] == DLC_HOUSE]["name"])

    print(f"\nRecommended team ({len(team)}/{args.size}):\n")
    for member in team:
        path_str = " -> ".join(step["class"] for step in member["path"])
        tag = " (DLC)" if member["character"] in dlc_names else ""
        print(f"{member['character']:15}{tag:6} | {member['role']:18} | {path_str}")
        print(f"{'':15}{'':6}   -> {member['why']}")


if __name__ == "__main__":
    main()
