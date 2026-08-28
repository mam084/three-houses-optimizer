"""
optimizer/data_loading.py

CSV-backed DataFrame -> lookup-dict indexers. Every function here takes a
raw data/*.csv DataFrame (already loaded by the caller - this module never
touches disk itself) and returns a plain dict keyed for fast lookup during
recommendation, degrading gracefully (returns {}) when the DataFrame is
None, so callers that don't have a given CSV on hand keep working.
"""
import pandas as pd

from .constants import NO_CERTIFICATION_CLASSES, STAT_COLS



def load_eligibility_lookup(eligibility_df: pd.DataFrame) -> dict:
    """
    Index data/class_eligibility.csv by class_name for fast lookup during
    recommendation. Each entry is {"characters": set|None, "gender":
    str|None} - None in either field means unrestricted on that axis.

    class_eligibility.csv itself is a small, hand-verified table (checked
    against Serenes Forest's classes page - which marks gender-locked
    classes with [M]/[F] tags - and cross-referenced against three other
    sources), hardcoded here rather than added to scrape_serenes.py. This
    mirrors the same call made for TIER_LEVEL_REQUIREMENTS: it's a small,
    stable dataset (23 rows) that doesn't change between game updates, so
    scraping it isn't worth a new page-parsing path. Classes not listed in
    the CSV are unrestricted by default.
    """
    lookup = {}
    for _, row in eligibility_df.iterrows():
        chars = row.get("locked_to_characters")
        gender = row.get("locked_to_gender")
        lookup[row["class_name"]] = {
            "characters": set(chars.split("|")) if isinstance(chars, str) and chars else None,
            "gender": gender if isinstance(gender, str) and gender else None,
        }
    return lookup



def load_character_relic_lookup(character_relics_df: pd.DataFrame | None) -> dict:
    """
    Index data/character_relics.csv by character name -> the set of weapon
    types (e.g. {"Sword"}, or {"Sword", "Reason"} for a character with more
    than one relic/crest - see Lysithea) their own Hero's Relic(s) use.

    Hand-curated from Serenes Forest's per-weapon-type pages (each
    Hero's-Relic-tier weapon's own certification requirement text names the
    Crest that powers its combat art) cross-referenced against each
    Crest's documented bearer(s) - the same sourcing precedent as
    class_eligibility.csv and class_weapon_requirements.csv. Scoped to
    actual Hero's Relic WEAPONS only (a class's certification requirement
    is a weapon-type concept) - non-weapon relics (Aegis Shield, Rafail
    Gem, Fetters of Dromi - shields/accessories, no weapon-type
    certification to nudge toward) aren't included. Not every playable
    character bears a relic; a character absent from the CSV simply has no
    relic-affinity bonus to apply (see relic_affinity_bonus).

    Returns {} if character_relics_df is None - callers degrade to
    today's "relics aren't modeled" behavior rather than failing.
    """
    if character_relics_df is None:
        return {}
    lookup: dict[str, set] = {}
    for _, row in character_relics_df.iterrows():
        lookup.setdefault(row["character"], set()).add(row["weapon_type"])
    return lookup



def load_class_growth_lookup(class_growth_df: pd.DataFrame | None) -> dict:
    """
    Index data/class_growth_rates.csv by class name -> {stat: modifier}
    (percentage points, additive on top of the character's own personal
    growth rate on every level-up - a real, separate Three Houses
    mechanic from a class's one-time flat certification stat boost in
    class_stat_boosts.csv; see that CSV's own header and
    https://serenesforest.net/three-houses/classes/growth-rates/).

    Returns {} if class_growth_df is None - callers degrade to today's
    "growth rates aren't modeled per-class" behavior rather than failing.
    """
    if class_growth_df is None:
        return {}
    lookup = {}
    for _, row in class_growth_df.iterrows():
        lookup[row["name"]] = {stat: row[stat] for stat in STAT_COLS if stat in row.index}
    return lookup



def class_growth_axis_range(class_growth_lookup: dict) -> tuple[float, float]:
    """
    Global (min, max) across every class's own growth-rate modifier, for
    every stat - the one fixed range every STANDALONE growth-rate chart
    (a single class, or two classes side by side in the Class Explorer -
    never stacked with a character's own growth, see
    growth_stack_axis_range for that) should share.

    Without this, each chart auto-scaled to whatever class(es) happened
    to be on screen, so a modest +15% modifier could render just as tall
    as another class's +40% simply because the two were never shown next
    to each other - misleading in exactly the way a shared axis fixes: a
    bar's height means the same thing everywhere it appears.

    Returns (-1.0, 1.0) if class_growth_lookup is empty, so callers never
    divide by (or plot against) a zero-width range.
    """
    values = [v for mods in class_growth_lookup.values() for v in mods.values()]
    if not values:
        return (-1.0, 1.0)
    return (float(min(values)), float(max(values)))



def growth_stack_axis_range(class_growth_lookup: dict, growth_rates_df: pd.DataFrame | None) -> tuple[float, float]:
    """
    Global (min, max) for the STACKED growth chart only (a character's own
    per-stat growth rate plus one class's own growth-rate modifier, summed
    - see render_growth_stack_chart) - deliberately a SEPARATE range from
    class_growth_axis_range's standalone one. A stacked total can run
    well above any single class's own modifier (a character's own growth
    rate alone is usually 10-70%), so sharing one axis across both chart
    types would needlessly compress the standalone charts to make room
    for totals they never actually show.

    Bounded by the real worst case across the whole roster and class
    list - the single highest and lowest per-stat growth rate anywhere in
    growth_rates_df, plus the single highest and lowest per-stat modifier
    anywhere in class_growth_lookup - so a stacked bar can never exceed
    what this range shows, no matter which character/class/stat is being
    displayed.

    Returns (-1.0, 1.0) if either input has no usable data.
    """
    class_values = [v for mods in class_growth_lookup.values() for v in mods.values()]
    if growth_rates_df is None or growth_rates_df.empty or not class_values:
        return (-1.0, 1.0)
    stat_cols_present = [c for c in STAT_COLS if c in growth_rates_df.columns]
    if not stat_cols_present:
        return (-1.0, 1.0)
    char_values = growth_rates_df[stat_cols_present].to_numpy().flatten()
    if len(char_values) == 0:
        return (-1.0, 1.0)
    stack_min = float(char_values.min()) + float(min(class_values))
    stack_max = float(char_values.max()) + float(max(class_values))
    return (stack_min, stack_max)



def load_class_base_stats_lookup(class_base_stats_df: pd.DataFrame | None) -> dict:
    """
    Index data/class_base_stats.csv by class name -> {stat: value} - each
    class's OWN base-stat line (hand-curated from Serenes Forest's
    per-class base-stats page), a third, separate mechanic from both a
    class's one-time flat certification boost (class_stat_boosts.csv) and
    its growth-RATE modifiers (class_growth_rates.csv, load_class_growth_lookup).

    This is a FLOOR, not a bonus: on certifying into a class, if the
    character's current stat is already below that class's own base stat,
    it snaps up to the class's base value; if the character's stat is
    already higher (e.g. a heavily-leveled character moving into an early
    class), nothing happens - the class base never subtracts, and it never
    stacks additively with the character's own stat either. See
    expected_stats_along_path, which applies this at the start of every
    tier actually spent along a path (the moment of certifying into that
    tier's class), not just the final one.

    Returns {} if class_base_stats_df is None - callers degrade to no
    floor being applied (today's pre-round-5 behavior) rather than failing.
    """
    if class_base_stats_df is None:
        return {}
    lookup = {}
    for _, row in class_base_stats_df.iterrows():
        lookup[row["name"]] = {stat: row[stat] for stat in STAT_COLS if stat in row.index}
    return lookup



def load_weapon_requirements_lookup(weapon_req_df: pd.DataFrame | None) -> dict:
    """
    Index data/class_weapon_requirements.csv by class_name. Each entry is
    {"tier", "weapon_category" ("physical"/"magic"/"hybrid" - whether the
    class's certification needs a physical weapon skill, a magic skill, or
    both), "requirement_type" ("AND"/"OR" - whether every listed
    requirement is needed or just one of them), "requirements" (a list of
    (skill, rank) tuples, e.g. [("Sword", "B"), ("Axe", "C")])}.

    Hand-curated from Serenes Forest's certification-exam requirements
    (https://serenesforest.net/three-houses/) - the same precedent as
    class_eligibility.csv and character_gender.csv: a small, stable table
    for information the stat-boost data alone doesn't carry. A class
    absent from this CSV (Unique-tier classes, NPC/Enemy classes) has no
    certification-exam requirement to model - they unlock via story
    progression instead - and simply won't appear in the lookup.

    Any class in NO_CERTIFICATION_CLASSES is skipped even if the CSV
    happens to contain a row for it - see that constant's docstring for
    why (a stray "Lord" row was exactly this bug in practice). This is the
    single choke point every caller (format_requirement,
    weapon_switch_penalty, weapon_growth_bonus,
    apply_weapon_affinity_fallback, ...) goes through, so the guard applies
    everywhere at once rather than needing a check at each call site.
    """
    if weapon_req_df is None:
        return {}
    lookup = {}
    for _, row in weapon_req_df.iterrows():
        if row["class_name"] in NO_CERTIFICATION_CLASSES:
            continue
        requirements = [tuple(part.split(":")) for part in row["requirements"].split("|")]
        lookup[row["class_name"]] = {
            "tier": row["tier"],
            "weapon_category": row["weapon_category"],
            "requirement_type": row["requirement_type"],
            "requirements": requirements,
        }
    return lookup



def load_character_proficiency_lookup(character_weapon_talent_df: pd.DataFrame | None) -> dict:
    """
    Index data/character_weapon_talent.csv by character name -> the set of
    skill types that character starts with at their own personal-best
    starting rank (per Serenes Forest's skill-levels page - each
    character's Level 1 skill ranks are listed there; "natural
    proficiency" here means whichever skill(s) that character's own
    starting ranks are highest in, not a designer-labeled "Talent"). E.g.
    Felix's listed starting ranks are Sword D / Bow E+ / Brawling E+, so
    his natural proficiency is {"Sword"}. Several characters are tied
    across more than one skill (e.g. Sylvain: Lance/Axe/Riding all D) -
    every tied skill is included, not just one.
    """
    if character_weapon_talent_df is None:
        return {}
    lookup = {}
    for _, row in character_weapon_talent_df.iterrows():
        skills = row.get("natural_proficiencies")
        lookup[row["name"]] = set(skills.split("|")) if isinstance(skills, str) and skills else set()
    return lookup



def load_starting_level_lookup(starting_level_df: pd.DataFrame | None) -> dict:
    """
    Index data/character_starting_level.csv by character name -> the level
    they actually join the player's roster at ("join level"), rather than
    the Level 1 baseline Serenes Forest's own base-stats table normalizes
    everyone to for easy side-by-side comparison ("these are the
    characters' stats when they are Level 1 in the Noble/Commoner class" -
    per that page's own header note). For the Protagonist and the 24 house
    students, join level is a verified 1 (playable from the Ch1 prologue).
    For Church/Knights of Seiros staff and DLC characters, who join
    progressively later, join level is an approximate placeholder derived
    from their recruitment timing (see the CSV's own notes column for each
    row's sourcing and confidence) rather than a directly-sourced figure -
    and for the Ashen Wolves specifically (Yuri/Balthus/Constance/Hapi),
    who canonically auto-scale to roughly match Byleth's own level when
    recruited rather than joining at one fixed level, no floor is enforced
    at all (join_level 1, same "doesn't simulate Byleth's own playthrough"
    gap already documented for cross-house recruitment in team_builder.py).

    Returns {} if starting_level_df is None (backward compatible - callers
    that don't pass it get today's behavior: everyone treated as level 1).
    """
    if starting_level_df is None:
        return {}
    lookup = {}
    for _, row in starting_level_df.iterrows():
        level = row.get("join_level")
        lookup[row["name"]] = int(level) if pd.notna(level) else 1
    return lookup
