# Design Notes

Detailed modeling decisions behind the optimizer, the game mechanics it does and doesn't capture, and why. The [README](../README.md) has the overview.

## Role detection

- **Standardized growth rates.** Growth rates are z-scored against the full roster before computing cosine similarity to each role archetype (`score_all_roles` in `src/optimizer/roles.py`). Raw percentages compare the overall shape of all nine stats at once, which under-weights a character's truly distinctive stat. Hubert and Linhardt were classified as Support rather than Magic Attacker until this change.
- **Weapon talent and Hero's Relics nudge close calls.** `natural_role_affinity_bonus` adds a small, bounded bonus toward the role whose weapon category matches a character's starting talent or relic. It flips Lorenz from Tank to Support on a close growth-rate call while leaving decisive reads (like Edelgard's Physical Attacker) untouched.
- **Fit across all roles is visible.** The Character Optimizer shows the similarity score for every archetype, not just the winner, so a close Tank/Support call is visible as such.

## Class scoring

Implemented in `recommend_path` (`src/optimizer/path.py`).

| Component | Constant | Purpose |
|---|---|---|
| Flat stat boost · role weights | — | Primary fit signal |
| Class growth modifiers · role weights | `GROWTH_RATE_SCORE_WEIGHT` | Rewards classes that compound the right stats |
| Starting weapon talent match | `WEAPON_PROFICIENCY_BONUS` | Prefers classes using weapons the character already favors |
| Hero's Relic match | `RELIC_AFFINITY_BONUS` | Prefers classes that let a relic-bearer use their relic |
| Untrained weapon type required | `WEAPON_SWITCH_PENALTY` | Flags costly switches (e.g. a sword user jumping to Wyvern Lord) |
| Character-unique class | `UNIQUE_CLASS_SCORE_BONUS` | Prefers a character's own canonical class when it fits the role |

All nudges are deliberately small relative to the stat-boost score, so they refine close tiers without overriding a clearly better fit. A weapon switch can still win a tier if its stats are much better; the warning reports the cost rather than hiding the pick.

**Eligibility rules**
- Character-locked and gender-locked classes are enforced from `class_eligibility.csv` and `character_gender.csv`. Byleth's gender is a player choice, so the gender selector in the app also drives Byleth's eligibility for War Master and Falcon Knight.
- Tiers are level-gated (Beginner 5, Intermediate 10, Advanced 20, Master 30), and paths stop at the last tier reachable at the target level.
- Support only considers classes with real healing-magic access at every tier, since Resistance alone would otherwise pull in classes like Falcon Knight.
- Unique classes are only spliced into a path when their weapon category suits the target role (`role_compatible_with_weapon_category`), so a lord's physical unique class is never recommended for Magic Attacker.
- Death Knight is locked to Jeritza. The four Cindered Shadows certification classes are opt-in and placed at Advanced tier, which best matches their B-rank requirements (the game doesn't assign them a tier).
- Beginner classes each boost a single unrelated stat, so stat data alone can't separate physical from magic paths there. `apply_weapon_affinity_fallback` breaks that tie using certification weapon type and starting talent, and only activates when the stat data is uninformative.

**Weapon-switch warnings** respect AND/OR certification requirements: an AND requirement like Wyvern Lord's (Lance/Axe/Flying) needs every listed skill already trained to count as familiar. Riding, Flying, and Heavy Armour transitions are only flagged for a jump to rank A with no prior training, since they're a smaller practical cost than a new weapon type. The path-level warning is recomputed from whatever classes are currently selected, so it clears when a mix-and-match override removes the switch.

## Stat projection

Implemented in `src/optimizer/projection.py`.

1. Start from the character's base stats at their join level (`character_starting_level.csv`).
2. For each tier in the path, on certifying into that class, raise any stat below the class's own base stat up to it (a floor, not a bonus).
3. Accrue that stretch of levels using the character's growth rates plus the class's growth modifiers.
4. Apply only the final class's one-time flat boost at the end. Boosts don't stack across a career in the game.

Because of the per-tier floor, changing a non-final tier in the mix-and-match picker can change the final numbers, so the projection is always recomputed from the full selected path.

## Team building

Implemented in `src/team_builder.py`.

- **Route pools.** Each route includes its own house plus church and Knights of Seiros staff. Black Eagles splits into Crimson Flower and Silver Snow; Edelgard and Hubert leave permanently on Silver Snow. DLC characters are opt-in and tagged.
- **Cross-house recruitment (optional).** Gated by a Byleth-level threshold, using the target-level slider as a stand-in for progress. The stat or skill requirement is shown as a note, not checked. House leaders and their retainers are never recruitable.
- **Forced deployments.** Byleth and the route's lord are added first, labeled as required by the game, with a toggle to turn this off.
- **Role rotation.** Remaining slots go to whichever role is least represented on the team so far, counting forced members, so every role is covered once before any repeats.
- **Variety mode.** Weighted-random picks within each role return a different reasonable team on each click. It can repeat a team by chance and isn't searching for a globally second-best team.
- **Dancer** is a single roster-wide slot assigned after the team is built, matching the White Heron Cup unlock. The tournament itself isn't simulated.

## Known simplifications

- **Weapon experience isn't simulated.** The tool knows whether a weapon type has been trained anywhere in the path, not whether the required rank is reachable by the needed level. A real model would need per-battle weapon-exp data that isn't published in a usable form.
- **One weight vector per role.** "Speed/Precision" scores Petra's Wyvern Lord and Pegasus Knight against the same weights, even though one trades speed for bulk. A hand-curated sub-archetype per class would be subjective in exactly the cases it's meant to help with. The planned direction is letting players adjust each role's stat weights directly.
- **Expected values, not distributions.** A 45% growth rate adds 0.45 points per level. That's the right quantity for comparing paths on average but doesn't show the spread of real playthroughs.
- **Approximate join levels for late joiners.** Exact (level 1) for Byleth and the 24 house students. Church staff and DLC characters use levels mapped from their recruitment month. The Ashen Wolves scale to Byleth's level in-game, so no floor is applied to them.
- **Hand-curated data.** Eligibility, certification requirements, weapon talents, relics, join levels, class base stats, and recruitment requirements were built by hand from Serenes Forest and cross-checked where noted in the loaders. Each is small and stable, so a scraper wasn't worth the complexity.
- **Weapon relics only.** `character_relics.csv` lists relics that are weapons tied to a certification weapon type. Non-weapon relics (Aegis Shield, Rafail Gem, Fetters of Dromi) don't affect class scoring and aren't included.

## Portraits

No game art is bundled. A character's portrait is shown if added to `assets/portraits/`; otherwise a tile colored by house is used. Portraits are decoded with Pillow (which checks actual file contents rather than the extension), center-cropped, resized with LANCZOS, and embedded as data URIs, so a corrupt or mislabeled file falls back to the tile instead of breaking the page. `byleth_m.png` and `byleth_f.png` let the gender selector pick Byleth's portrait.

## Code organization

`src/optimizer/` is split by concern (`constants`, `data_loading`, `eligibility`, `roles`, `requirements`, `projection`, `path`, `recommend`), with `__init__.py` re-exporting the public API so `from src.optimizer import X` and `python -m src.optimizer` both work. `app.py` is a thin entry point; shared state lives in `app_state.py` and each tab has its own module in `tabs/`.
