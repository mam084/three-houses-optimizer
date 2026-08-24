"""
team_builder.py

Recommends a balanced team from a pool of candidate characters, covering
complementary roles rather than optimizing one character in isolation
(that's what optimizer.py already does).

Approach: auto-detect each candidate's natural role (reusing
optimizer.detect_natural_role), group by role, then build the team via
round-robin selection across role groups - each round, pick the
highest-scoring not-yet-picked character from the next role in rotation.
This favors role diversity over just stacking the single "best" characters,
which would otherwise tend to produce redundant teams (e.g. five Physical
Attackers and no Tank).

This is a greedy heuristic, not a global optimum over all possible team
combinations - deliberately so, since brute-force search over ~40 choose 6
candidates is both slow and hard to explain. Round-robin-by-role is fast,
deterministic, and its reasoning is easy to state plainly: "balance role
coverage first, favor strength within that."
"""

from pathlib import Path

import pandas as pd

from src.optimizer import ROLE_PROFILES, detect_natural_role, recommend_for_character

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def assign_roles(candidates: list[str], growth_rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each candidate character, auto-detect their best-fitting role.
    Returns a DataFrame: character, role, score.

    Growth rates are standardized against the full roster passed in
    (growth_rates_df), not just the candidate subset - this keeps role
    detection consistent with optimizer.recommend_for_character, which
    does the same. Pass the full roster's growth_rates_df here even if
    candidates is a smaller pool (e.g. one house), so "unusual" is measured
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


def build_balanced_team(
    candidates: list[str],
    growth_rates_df: pd.DataFrame,
    team_size: int = 6,
) -> list[dict]:
    """
    Build a team of team_size characters from the candidate pool, balancing
    role coverage via round-robin selection across each character's
    auto-detected best role.

    Returns a list of {"character": ..., "role": ..., "score": ...} dicts,
    in the order they were picked.
    """
    assignments = assign_roles(candidates, growth_rates_df)
    if assignments.empty:
        return []

    # Group candidates by role, each group sorted strongest-fit first
    role_groups = {
        role: list(group.itertuples(index=False))
        for role, group in assignments.groupby("role")
    }
    # Rotate roles in a stable order so results are deterministic
    role_order = sorted(role_groups.keys())

    team = []
    role_cursor = {role: 0 for role in role_order}

    while len(team) < team_size:
        picked_this_pass = False

        for role in role_order:
            if len(team) >= team_size:
                break

            group = role_groups[role]
            cursor = role_cursor[role]
            if cursor >= len(group):
                continue  # this role's candidates are exhausted

            candidate = group[cursor]
            role_cursor[role] += 1
            team.append({
                "character": candidate.character,
                "role": candidate.role,
                "score": round(candidate.score, 3),
            })
            picked_this_pass = True

        if not picked_this_pass:
            break  # every role group exhausted, fewer candidates than team_size

    return team


def build_team_with_paths(
    candidates: list[str],
    base_stats_df: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
    stat_boosts_df: pd.DataFrame,
    team_size: int = 6,
    target_level: int = 20,
    eligibility_df: pd.DataFrame | None = None,
    character_gender_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Like build_balanced_team, but also attaches each member's full class-path
    recommendation (reusing optimizer.recommend_for_character), so the team
    output is immediately useful rather than just a role assignment.

    eligibility_df and character_gender_df are passed straight through to
    recommend_for_character - see its docstring. Both optional and default
    to None (no eligibility filtering), for backward compatibility with
    existing callers.
    """
    team = build_balanced_team(candidates, growth_rates_df, team_size)

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
                         help="Limit candidates to one house (e.g. 'Black Eagles'). Omit for full roster.")
    parser.add_argument("--size", type=int, default=6, help="Team size (default: 6)")
    args = parser.parse_args()

    if args.house:
        candidates = base_stats_df[base_stats_df["house"] == args.house]["name"].tolist()
    else:
        candidates = base_stats_df[~base_stats_df["name"].str.contains(r"\(NPC\)")]["name"].tolist()

    team = build_team_with_paths(
        candidates, base_stats_df, growth_rates_df, stat_boosts_df, team_size=args.size,
        eligibility_df=eligibility_df, character_gender_df=character_gender_df,
    )

    print(f"\nRecommended team ({len(team)}/{args.size}):\n")
    for member in team:
        path_str = " -> ".join(step["class"] for step in member["path"])
        print(f"{member['character']:15} | {member['role']:18} | {path_str}")


if __name__ == "__main__":
    main()