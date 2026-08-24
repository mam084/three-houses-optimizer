# Three Houses Class Optimizer

An interactive tool for Fire Emblem: Three Houses that recommends a class path for any character - either toward their natural strengths (auto-detected from growth rates) or a role you choose - and projects their stats at a target level.

Built as a portfolio project. Started as a Pokemon competitive team-builder; pivoted here to dig into a genuinely different kind of problem: simulating stat growth and class progression rather than looking up static competitive data.

![Dashboard screenshot](docs/screenshot.png)

## Features

- Scrapes character base stats, growth rates, and class stat boosts directly from [Serenes Forest](https://serenesforest.net/three-houses/)
- **Auto-detected role** - infers what a character is naturally good at from their growth rates (cosine similarity against 5 role archetypes: Physical Attacker, Magic Attacker, Tank, Support, Flier/Mobility), no input needed
- **Manual role targeting** - override the auto-detection to ask "what if I built this character as a mage instead?"
- **Class path recommendation** - best-fitting class at each tier (Beginner -> Intermediate -> Advanced -> Master) toward the target role
- **Stat projection** - expected stats at a chosen level, shown as a before/after radar chart against the character's Level 1 base stats

## Known Simplifications

Worth being upfront about, since these are real gaps between this tool and the actual game mechanics:

- **Unique-tier classes are excluded.** Classes like Emperor and Barbarossa are character-locked to specific lords (Edelgard, Claude, etc.). We don't yet have that eligibility data, so recommending them generically would be wrong - they're left out rather than mis-recommended.
- **Class progression uses tier order, not real unlock requirements.** The actual game gates classes by character level + skill proficiency rank (e.g. "Swordmaster needs Sword rank B+ and character level 10"), which we don't have data for yet. Tier order (Beginner -> Master) is a reasonable approximation of the real pacing, just optimistic about exact timing.
- **Only the final class's stat boost is applied.** In the real game, boosts don't stack across a career path - only your current class's boost counts. The earlier tiers in a recommended path are shown for progression flavor, not because their boosts contribute to the final projected stats.
- **Stat projections are expected values, not simulations.** A 45% growth rate contributes 0.45 expected points per level, not a simulated coin-flip per level-up. This is the right number for comparing paths against each other on average, but doesn't show the range of outcomes a real playthrough could produce.

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

You can also use the recommender from the command line directly:
```bash
python src/optimizer.py Bernadetta
python src/optimizer.py Dedue --role "Magic Attacker" --level 30
```

## Project Structure

```
three-houses-optimizer/
├── data/
│   ├── character_base_stats.csv
│   ├── character_growth_rates.csv
│   └── class_stat_boosts.csv
├── docs/
│   └── screenshot.png
├── src/
│   ├── scrape_serenes.py
│   └── optimizer.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

**`src/`**
- `scrape_serenes.py` - pulls character and class data from Serenes Forest, with manual BeautifulSoup row parsing (not `pandas.read_html()`) to correctly handle the site's spoiler-toggle rows
- `optimizer.py` - the recommendation logic: role detection, class path selection, stat projection

**`app.py`** - the Streamlit dashboard.

## Roadmap

- [x] **Data pipeline:** scrape character stats, growth rates, and class stat boosts
- [x] **v1 recommender:** auto-detected or manually-targeted role, class path recommendation, stat projection
- [x] **Dashboard:** interactive character/role/level picker with path and radar chart *(this release)*
- [ ] **Team composition layer:** recommend a full squad covering complementary roles, not just one character at a time
- [ ] **Skill requirement data:** replace the tier-order approximation with the game's real level + skill-proficiency unlock conditions
- [ ] **Unique-class eligibility:** properly model which lord classes are available to which characters
- [ ] **Deploy:** ship as a public web app (Streamlit Community Cloud)

## Data Source

Character and class data via [Serenes Forest](https://serenesforest.net/three-houses/), a fan-maintained Fire Emblem reference site.