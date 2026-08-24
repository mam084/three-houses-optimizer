# Three Houses Class Optimizer

An interactive tool for Fire Emblem: Three Houses that recommends a class path for any character - either toward their natural strengths (auto-detected from growth rates) or a role you choose - and projects their stats at a target level.

Built as a portfolio project. Started as a Pokemon competitive team-builder; pivoted here to dig into a genuinely different kind of problem: simulating stat growth and class progression rather than looking up static competitive data.

![Dashboard screenshot](docs/screenshot.png)

## Features

- Scrapes character base stats, growth rates, and class stat boosts directly from [Serenes Forest](https://serenesforest.net/three-houses/)
- **Auto-detected role** - infers what a character is naturally good at from their growth rates, standardized against the roster and compared via cosine similarity against 5 role archetypes: Physical Attacker, Magic Attacker, Tank, Support, Speed/Precision - no input needed
- **Manual role targeting** - override the auto-detection to ask "what if I built this character as a mage instead?"
- **Class path recommendation** - best-fitting class at each tier (Beginner -> Intermediate -> Advanced -> Master) toward the target role, truncated to whichever tiers are actually reachable at the target level (e.g. targeting level 15 stops after Intermediate, since Advanced requires level 20)
- **Eligibility-aware** - recommendations respect character-locked and gender-locked classes (e.g. "Lord" is only ever recommended to the three house leaders, "Falcon Knight" only to characters who can access it), so the tool never suggests a class that character couldn't actually take in-game
- **Stat projection** - expected stats at a chosen level, shown as a before/after radar chart against the character's Level 1 base stats

## Known Simplifications

Worth being upfront about, since these are real gaps between this tool and the actual game mechanics:

- **Unique classes aren't spliced into the recommended path.** Classes like Emperor and Barbarossa (character-locked to specific lords) and Falcon Knight (gender-locked) are now eligibility-checked - recommendations never suggest a class a character can't access. But Unique classes specifically still don't slot into the Beginner -> Master path itself, since they're unlocked by character-specific story beats rather than a uniform level + seal requirement, so there's no principled tier to insert them at.
- **Class progression is level-gated but not skill-proficiency-gated.** The real game requires both a character level *and* a skill-proficiency rank to change class (e.g. "Swordmaster needs Sword rank A and character level 20"). We model the level half exactly (Beginner=5, Intermediate=10, Advanced=20, Master=30, verified against Serenes Forest) and truncate the path accordingly, but still assume proficiency ranks keep pace with level - we don't have skill-level data yet, so a recommended path can be more optimistic about timing than a real playthrough would allow.
- **Only the final class's stat boost is applied.** In the real game, boosts don't stack across a career path - only your current class's boost counts. The earlier tiers in a recommended path are shown for progression flavor, not because their boosts contribute to the final projected stats.
- **Stat projections are expected values, not simulations.** A 45% growth rate contributes 0.45 expected points per level, not a simulated coin-flip per level-up. This is the right number for comparing paths against each other on average, but doesn't show the range of outcomes a real playthrough could produce.
- **Character/class eligibility data is hand-curated, not scraped.** `class_eligibility.csv` and `character_gender.csv` were built by hand from Serenes Forest's classes page (which marks gender-locked classes with `[M]`/`[F]` tags) and cross-checked against other sources, rather than added to the scraper - it's a small, stable ~65-row dataset that doesn't change between game updates, so a dedicated scraping path wasn't worth the complexity.

## Tech Stack

- **Data:** scraped from [Serenes Forest](https://serenesforest.net/three-houses/) with `requests` + `BeautifulSoup`
- **Processing:** pandas
- **Recommendation logic:** NumPy (cosine similarity for role detection, weighted scoring for class fit)
- **Dashboard:** Streamlit
- **Charts:** Plotly

## Setup

```bash
# clone and enter the repo
git clone https://github.com/<your-username>/three-houses-optimizer.git
cd three-houses-optimizer

# install dependencies
pip install -r requirements.txt

# scrape character and class data from Serenes Forest
python src/scrape_serenes.py

# launch the dashboard
streamlit run app.py
```

You can also use the recommender from the command line directly. Run these
as modules (`-m`) from the project root, not as direct file paths - some
scripts import from others, and `-m` mode resolves those imports correctly:
```bash
python -m src.optimizer Bernadetta
python -m src.optimizer Dedue --role "Magic Attacker" --level 30
python -m src.team_builder --house "Black Eagles" --size 6
python -m src.team_builder --size 8
```

## Project Structure

```
three-houses-optimizer/
├── data/
│   ├── character_base_stats.csv
│   ├── character_growth_rates.csv
│   ├── class_stat_boosts.csv
│   ├── class_eligibility.csv        # hand-curated, not scraped - see Known Simplifications
│   └── character_gender.csv         # hand-curated, not scraped - see Known Simplifications
├── docs/
│   └── screenshot.png
├── src/
│   ├── scrape_serenes.py
│   ├── optimizer.py
│   └── team_builder.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

**`src/`**
- `scrape_serenes.py` - pulls character and class data from Serenes Forest, with manual BeautifulSoup row parsing (not `pandas.read_html()`) to correctly handle the site's spoiler-toggle rows
- `optimizer.py` - the recommendation logic: role detection, level-gated and eligibility-aware class path selection, stat projection
- `team_builder.py` - builds a balanced, eligibility-aware team from a candidate pool via round-robin selection across auto-detected roles

**`app.py`** - the Streamlit dashboard.

## Roadmap

- [x] **Data pipeline:** scrape character stats, growth rates, and class stat boosts
- [x] **v1 recommender:** auto-detected or manually-targeted role, class path recommendation, stat projection
- [x] **Dashboard:** interactive character/role/level picker with path and radar chart
- [x] **Team composition layer:** recommend a full squad covering complementary roles, not just one character at a time
- [x] **Level-gating:** recommended paths are truncated to tiers actually reachable at the target level, replacing the old "always assume the full 4-tier path" approximation *(this release)*
- [x] **Unique-class eligibility:** character-locked and gender-locked classes are now respected across both the class-path recommender and the team builder, so recommendations never suggest a class that character couldn't access *(this release)*
- [ ] **Skill-proficiency requirements:** the level half of class-change requirements is modeled; the skill-proficiency-rank half (e.g. "Sword rank A") still isn't
- [ ] **Deploy publicly**

## Data Source

Character and class data via [Serenes Forest](https://serenesforest.net/three-houses/), a fan-maintained Fire Emblem reference site. Base stats, growth rates, and class stat boosts are scraped directly (`src/scrape_serenes.py`); class eligibility and character gender are hand-curated from the same site (see Known Simplifications) since that data isn't in a cleanly scrapable tabular form.