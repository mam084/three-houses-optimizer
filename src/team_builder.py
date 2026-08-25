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

# The three houses you actually pick a route through in the base game -
# except Black Eagles splits into two story routes depending on whether
# you side with Edelgard (Crimson Flower) or the Church against her
# (Silver Snow), and that split has real roster consequences: on Silver
# Snow, Edelgard and Hubert both leave permanently around Chapter 11 and
# become unrecruitable final-chapter enemies, while the rest of the Black
# Eagles cast stays normally recruitable (confirmed against Fire Emblem
# Wiki's and Serenes Forest's Silver Snow coverage - see
# SILVER_SNOW_LOST_CHARACTERS below). Blue Lions and Golden Deer don't
# have an equivalent story-route split, so they're each just one pool.
#
# "Church of Seiros" and "Knights of Seiros" aren't routes - they're the
# Officers Academy staff and knights who are recruitable as playable units
# regardless of which route you chose, so a candidate pool for any route
# should include them too, not just that route's own house students. "You
# and the Enigmatic Girl" is the Protagonist's own placeholder house from
# the base-stats data (Sothis, its only other member, is an NPC and
# already excluded elsewhere) - folded in here so a route pool includes
# the Protagonist without surfacing that odd label as its own selectable
# pool.
BLACK_EAGLES_CRIMSON_FLOWER = "Black Eagles (Crimson Flower)"
BLACK_EAGLES_SILVER_SNOW = "Black Eagles (Silver Snow)"
REAL_ROUTES = [BLACK_EAGLES_CRIMSON_FLOWER, BLACK_EAGLES_SILVER_SNOW, "Blue Lions", "Golden Deer"]

# Which data/character_base_stats.csv "house" value each route's own
# students come from - both Black Eagles routes pull from the same house.
ROUTE_HOME_HOUSE = {
    BLACK_EAGLES_CRIMSON_FLOWER: "Black Eagles",
    BLACK_EAGLES_SILVER_SNOW: "Black Eagles",
    "Blue Lions": "Blue Lions",
    "Golden Deer": "Golden Deer",
}
ROUTE_INCLUDES_HOUSES = {
    route: [house, "Church of Seiros", "Knights of Seiros", "You and the Enigmatic Girl"]
    for route, house in ROUTE_HOME_HOUSE.items()
}
REAL_HOUSES = ["Black Eagles", "Blue Lions", "Golden Deer"]
SILVER_SNOW_LOST_CHARACTERS = {"Edelgard", "Hubert"}
DLC_HOUSE = "DLC-exclusive"


def get_candidate_pool(
    base_stats_df: pd.DataFrame,
    playable_names: list[str],
    route: str | None = None,
    include_dlc: bool = False,
    include_cross_house_recruits: bool = False,
    target_level: int | None = None,
    recruitment_lookup: dict | None = None,
) -> list[str]:
    """
    Build a candidate pool reflecting who's actually available to recruit.

    route=None (or "Full roster") returns every playable character. A
    specific route (e.g. "Blue Lions") returns that route's own house
    students PLUS the Protagonist and the Church/Knights of Seiros staff,
    who are recruitable on any route - narrowing to just the named house
    was the original "team builder should access all characters available
    in that route" gap. On the Silver Snow route specifically, Edelgard
    and Hubert are dropped from the pool - see SILVER_SNOW_LOST_CHARACTERS
    - since they're permanently unrecruitable enemies on that route, unlike
    Crimson Flower where they're normal roster members. DLC-exclusive
    characters (Cindered Shadows) are opt-in via include_dlc, since they
    require separately-owned DLC and are otherwise not "available" the way
    base-roster characters are - this applies to both the full roster and
    a specific route.

    include_cross_house_recruits (requires recruitment_lookup - see
    load_recruitment_lookup) additionally pulls in students from the
    route's OTHER two houses who are actually recruitable per the game's
    real recruitment requirements (data/recruitment_requirements.csv):
    house leaders and their sworn retainers (Edelgard/Hubert,
    Dimitri/Dedue, Claude) are never included, since the game doesn't let
    you recruit them away from their own route regardless of stats; every
    other cross-house student is included once target_level meets their
    listed Byleth-level requirement (see is_cross_house_recruitable) - an
    approximation, since it assumes Byleth's own level roughly tracks
    target_level, rather than simulating Byleth's actual playthrough
    level. The stat/skill-rank half of each requirement (e.g. "high Magic,
    Faith B") is carried along as a note (see format_recruitment_note) but
    not enforced, since this tool doesn't simulate Byleth's own stat
    growth to check it. Ignored for "Full roster", since there's no "other
    house" to recruit from there.

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

    pool = pool_df["name"].tolist()

    if route == BLACK_EAGLES_SILVER_SNOW:
        pool = [name for name in pool if name not in SILVER_SNOW_LOST_CHARACTERS]

    if include_cross_house_recruits and route not in (None, "Full roster") and recruitment_lookup:
        own_house = ROUTE_HOME_HOUSE.get(route)
        cross_house_df = playable_df[
            playable_df["house"].isin(REAL_HOUSES) & (playable_df["house"] != own_house)
        ]
        for name in cross_house_df["name"]:
            if name in pool:
                continue
            if is_cross_house_recruitable(name, recruitment_lookup, target_level):
                pool.append(name)

    return pool


def load_recruitment_lookup(recruitment_requirements_df: pd.DataFrame | None) -> dict:
    """
    Index data/recruitment_requirements.csv by character name. Each entry
    is {"home_house", "byleth_level_required" (int, or None if the
    character can't be recruited outside their own route at all - the
    three house leaders and their sworn retainers Hubert/Dedue),
    "stat_required", "skill_required", "skill_rank_required", "notes"}.

    Hand-curated from Serenes Forest's recruitment-requirements page - the
    same precedent as class_eligibility.csv, character_gender.csv and
    class_weapon_requirements.csv.
    """
    if recruitment_requirements_df is None:
        return {}
    lookup = {}
    for _, row in recruitment_requirements_df.iterrows():
        level = row.get("byleth_level_required")
        lookup[row["character"]] = {
            "home_house": row.get("home_house"),
            "byleth_level_required": None if pd.isna(level) else int(level),
            "stat_required": row.get("stat_required") if isinstance(row.get("stat_required"), str) else None,
            "skill_required": row.get("skill_required") if isinstance(row.get("skill_required"), str) else None,
            "skill_rank_required": row.get("skill_rank_required") if isinstance(row.get("skill_rank_required"), str) else None,
            "notes": row.get("notes") if isinstance(row.get("notes"), str) else None,
        }
    return lookup


def is_cross_house_recruitable(
    character_name: str, recruitment_lookup: dict, target_level: int | None = None
) -> bool:
    """
    Whether character_name can be recruited away from their own house's
    route, per recruitment_lookup (see load_recruitment_lookup). A
    character absent from the lookup (Church/Knights of Seiros staff, the
    Protagonist) isn't a cross-house recruitment case at all - always
    True, since get_candidate_pool already includes them unconditionally.
    A character with no byleth_level_required (the house leaders and
    their retainers) can never be recruited cross-house - always False. A
    target_level below the requirement means "not yet feasible at the
    level you're targeting" - False; omitting target_level skips that
    check (True whenever a requirement exists at all).
    """
    info = recruitment_lookup.get(character_name)
    if info is None:
        return True
    if info["byleth_level_required"] is None:
        return False
    if target_level is not None and info["byleth_level_required"] > target_level:
        return False
    return True


def cross_house_names_in_pool(base_stats_df: pd.DataFrame, pool: list[str], route: str | None) -> set[str]:
    """
    Which names in `pool` (as returned by get_candidate_pool) are actually
    cross-house recruits for `route` - i.e. their native house isn't part
    of that route's own pool (see ROUTE_INCLUDES_HOUSES). Used to scope
    the "Recruited from another house" note (see format_recruitment_note)
    to characters who actually needed recruiting, rather than every
    character recruitment_requirements.csv happens to have a row for -
    Black Eagles' own students (e.g. Petra, Linhardt) have rows too, since
    they're each recruitable onto the OTHER two routes, but that's
    irrelevant when they're already native to the route being built.
    Returns an empty set for "Full roster" (or no route), since there's no
    "own house" to be cross of.
    """
    if route is None or route == "Full roster":
        return set()
    own_pool_houses = set(ROUTE_INCLUDES_HOUSES.get(route, [route]))
    pool_set = set(pool)
    cross = base_stats_df[
        base_stats_df["name"].isin(pool_set) & ~base_stats_df["house"].isin(own_pool_houses)
    ]["name"]
    return set(cross)


def format_recruitment_note(character_name: str, recruitment_lookup: dict) -> str | None:
    """
    Human-readable recruitment requirement for a cross-house pick, e.g.
    "Byleth level 15+, high Magic, Faith B (B support with Byleth lowers
    requirements)". Returns None if character_name isn't in the lookup
    (not a cross-house case, or the lookup wasn't supplied).
    """
    info = recruitment_lookup.get(character_name) if recruitment_lookup else None
    if info is None or info["byleth_level_required"] is None:
        return None
    parts = [f"Byleth level {info['byleth_level_required']}+"]
    if info["stat_required"] and info["skill_required"] and info["skill_rank_required"]:
        parts.append(f"high {info['stat_required']}, {info['skill_required']} {info['skill_rank_required']}")
    note = ", ".join(parts)
    if info["notes"]:
        note += f" ({info['notes']})"
    return note


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
    recruitment_lookup: dict | None = None,
    cross_house_names: set | None = None,
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
    # must_include beats exclude for a name in both (see docstring above) - previously exclude was
    # applied to the candidate pool BEFORE must_include ever got a chance to add its names back, so a
    # must_include name that also happened to be excluded silently vanished instead of being force-added
    # (caught by tests/test_team_builder.py's test_must_include_wins_over_exclude).
    effective_exclude = exclude - set(must_include)

    working_candidates = [c for c in candidates if c not in effective_exclude]
    assignments = assign_roles(working_candidates, growth_rates_df)
    if assignments.empty:
        return []

    by_character = assignments.set_index("character")
    team: list[dict] = []
    picked = set()

    def with_recruitment_note(why: str, character_name: str) -> str:
        if not cross_house_names or character_name not in cross_house_names or not recruitment_lookup:
            return why
        note = format_recruitment_note(character_name, recruitment_lookup)
        return f"{why} Recruited from another house - {note}." if note else why

    for name in must_include:
        if name in picked or name not in by_character.index or len(team) >= team_size:
            continue
        row = by_character.loc[name]
        team.append({
            "character": name,
            "role": row["role"],
            "score": round(float(row["score"]), 3),
            "why": with_recruitment_note("Included by request.", name),
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
                "why": with_recruitment_note(rank_note + variety_note + ".", candidate.character),
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
    target_level: int = 30,
    eligibility_df: pd.DataFrame | None = None,
    character_gender_df: pd.DataFrame | None = None,
    must_include: list[str] | None = None,
    exclude: list[str] | None = None,
    rng: np.random.Generator | None = None,
    recruitment_lookup: dict | None = None,
    cross_house_names: set | None = None,
    weapon_req_df: pd.DataFrame | None = None,
    character_weapon_talent_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Like build_balanced_team, but also attaches each member's full class-path
    recommendation (reusing optimizer.recommend_for_character), so the team
    output is immediately useful rather than just a role assignment.

    eligibility_df, character_gender_df, weapon_req_df and
    character_weapon_talent_df are passed straight through to
    recommend_for_character - see its docstring. All optional and default
    to None (the corresponding feature disabled), for backward
    compatibility with existing callers. must_include/exclude/rng/
    recruitment_lookup/cross_house_names are passed straight through to
    build_balanced_team - see its docstring (and cross_house_names_in_pool
    for how to compute that last one from a get_candidate_pool result).
    """
    team = build_balanced_team(
        candidates, growth_rates_df, team_size,
        must_include=must_include, exclude=exclude, rng=rng,
        recruitment_lookup=recruitment_lookup, cross_house_names=cross_house_names,
    )

    for member in team:
        full_rec = recommend_for_character(
            member["character"], base_stats_df, growth_rates_df, stat_boosts_df,
            role_name=member["role"], target_level=target_level,
            eligibility_df=eligibility_df, character_gender_df=character_gender_df,
            weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
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
    weapon_req_df = pd.read_csv(DATA_DIR / "class_weapon_requirements.csv")
    character_weapon_talent_df = pd.read_csv(DATA_DIR / "character_weapon_talent.csv")
    recruitment_requirements_df = pd.read_csv(DATA_DIR / "recruitment_requirements.csv")

    import argparse
    parser = argparse.ArgumentParser(description="Recommend a balanced Three Houses team.")
    parser.add_argument("--house", type=str, default=None,
                         help="Limit candidates to one route, e.g. 'Blue Lions' or "
                              f"'{BLACK_EAGLES_SILVER_SNOW}' - includes that route's own house students "
                              "plus the Protagonist and Church/Knights of Seiros staff, who are recruitable "
                              "on any route. Omit for the full roster. Choices: "
                              + ", ".join(f"'{r}'" for r in REAL_ROUTES))
    parser.add_argument("--size", type=int, default=6, help="Team size (default: 6)")
    parser.add_argument("--target-level", type=int, default=30, help="Target level for stat projection (default: 30)")
    parser.add_argument("--include-dlc", action="store_true",
                         help="Include DLC-exclusive characters (Cindered Shadows) in the candidate pool.")
    parser.add_argument("--include-cross-house-recruits", action="store_true",
                         help="Also consider students from the route's other two houses, gated by their real "
                              "in-game recruitment requirements (data/recruitment_requirements.csv) - see "
                              "get_candidate_pool's docstring.")
    parser.add_argument("--must-include", type=str, default=None,
                         help="Comma-separated characters to force onto the team, e.g. 'Lysithea,Dedue'.")
    parser.add_argument("--exclude", type=str, default=None,
                         help="Comma-separated characters to drop from consideration.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for weighted-random variety instead of always the top-scoring pick "
                              "per role. Omit for the original deterministic behavior.")
    args = parser.parse_args()

    recruitment_lookup = load_recruitment_lookup(recruitment_requirements_df)
    playable_names = base_stats_df[~base_stats_df["name"].str.contains(r"\(NPC\)")]["name"].tolist()
    candidates = get_candidate_pool(
        base_stats_df, playable_names, route=args.house, include_dlc=args.include_dlc,
        include_cross_house_recruits=args.include_cross_house_recruits,
        target_level=args.target_level, recruitment_lookup=recruitment_lookup,
    )
    must_include = [n.strip() for n in args.must_include.split(",")] if args.must_include else None
    exclude = [n.strip() for n in args.exclude.split(",")] if args.exclude else None
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    cross_house_names = cross_house_names_in_pool(base_stats_df, candidates, args.house)

    team = build_team_with_paths(
        candidates, base_stats_df, growth_rates_df, stat_boosts_df, team_size=args.size,
        target_level=args.target_level,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
        must_include=must_include, exclude=exclude, rng=rng,
        recruitment_lookup=recruitment_lookup, cross_house_names=cross_house_names,
        weapon_req_df=weapon_req_df, character_weapon_talent_df=character_weapon_talent_df,
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
