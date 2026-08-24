"""
scrape_serenes.py

Scrapes Fire Emblem: Three Houses character and class data from Serenes
Forest (https://serenesforest.net/three-houses/), a well-maintained fan
reference site with clean, well-organized data tables.

Pulls three datasets:
  - Character base stats (Level 1 stats in the starting Noble/Commoner class)
  - Character growth rates (per-stat % chance of a stat increasing on level up)
  - Class stat boosts (automatic stat bonuses granted by each class, plus
    which tier - Beginner/Intermediate/Advanced/Master - a class belongs to)

Each page is a series of tables, one per section (character pages are
grouped by house; the stat-boosts page is grouped by class tier). We walk
the page's headings and tables together so each row can be tagged with its
section, parsing each table's rows manually via BeautifulSoup rather than
pandas.read_html().

That manual approach matters for the stat-boosts page specifically: it has
two rows that act as spoiler-toggle links on the live site (e.g. "click to
reveal the Protagonist's final class"), each rendered as a single wide cell
spanning the whole table. The underlying data for the classes they "reveal"
(Enlightened One, Armored Lord, etc.) is actually already present in the
static HTML unconditionally - the click just toggles CSS visibility on the
live page, nothing is loaded dynamically. But pandas.read_html() choked on
the toggle rows' irregular cell count and silently dropped several
subsequent real class rows in the same table as a result. Parsing rows
manually avoids that: a toggle row has 1 cell where a real row has 10, so
it's skipped precisely because its shape doesn't match the header - not
because of what it says - and every real row after it parses normally.

The stat-boosts page also needs numeric cleanup the other two pages don't:
  - Blank cells mean a 0 boost, not missing data
  - Some cells show a mounted bonus in parentheses, e.g. "2 (+2)" - we keep
    the base (unmounted) value and drop the parenthetical for now, since
    modeling mount state is out of scope for this stage

Usage:
    python src/scrape_serenes.py
"""

import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_STATS_URL = "https://serenesforest.net/three-houses/characters/base-stats/"
GROWTH_RATES_URL = "https://serenesforest.net/three-houses/characters/growth-rates/"
STAT_BOOSTS_URL = "https://serenesforest.net/three-houses/classes/stat-boosts/"

HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio project data collection)"}


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.text


def iter_sections_and_tables(html: str):
    """
    Walk the page in document order, yielding (section_title, table_soup)
    for every table, where section_title is the text of the nearest
    preceding heading (h3 or h4 - Serenes Forest uses h4 for these).
    """
    soup = BeautifulSoup(html, "lxml")
    current_section = None

    for element in soup.find_all(["h3", "h4", "table"]):
        if element.name in ("h3", "h4"):
            current_section = element.get_text(strip=True)
        elif element.name == "table":
            yield current_section, element


def extract_table(table_soup) -> tuple[list[str], list[list[str]]]:
    """
    Manually parse a <table> element's rows via BeautifulSoup rather than
    pandas.read_html(). This gives us full control over row handling -
    notably, footnote/note rows (which span the whole table as a single
    wide cell, e.g. a colspan cell) have a different cell count than real
    data rows and can be reliably skipped on that basis, structurally,
    rather than guessing from the text. This also avoids a real failure
    mode we hit with pandas.read_html() on this site: a colspan-based note
    row confused its row alignment and caused it to silently drop several
    subsequent real data rows in the same table.

    Returns (headers, rows) where rows is a list of cell-text lists, one
    per real data row (rows that don't match the header's cell count are
    treated as footnotes/notes and dropped).
    """
    trs = table_soup.find_all("tr")
    if not trs:
        return [], []

    header_cells = trs[0].find_all(["th", "td"])
    headers = [c.get_text(strip=True) for c in header_cells]
    if headers and headers[0] == "":
        headers[0] = "Name"  # some tables (e.g. DLC Exclusive) omit the first header
    n_cols = len(headers)

    data_rows = []
    for tr in trs[1:]:
        cells = tr.find_all(["td", "th"])
        texts = [c.get_text(strip=True) for c in cells]
        if len(texts) != n_cols:
            # Cell count doesn't match the header - almost always a footnote/
            # note row using a single spanning cell, not real data. Skip it.
            continue
        data_rows.append(texts)

    return headers, data_rows


def parse_character_page(html: str, value_label: str) -> pd.DataFrame:
    """
    Parse a character stat page (base stats or growth rates) into a tidy
    DataFrame: one row per character, columns for house/section and each stat.
    """
    frames = []
    for section, table in iter_sections_and_tables(html):
        headers, rows = extract_table(table)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=headers)
        df["house"] = section
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"Name": "name"})

    stat_cols = [c for c in combined.columns if c not in ("name", "house")]
    for col in stat_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.dropna(subset=["name"]).reset_index(drop=True)
    return combined


def parse_numeric_cell(value) -> int:
    """
    Parse a stat-boost cell into its base (unmounted) integer value.
    Handles blanks (-> 0), plain integers, and mounted-bonus notation like
    '2 (+2)' or '(-2)' (base value only; the parenthetical mount bonus is
    dropped for this stage).
    """
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    match = re.match(r"^(-?\d+)", text)
    return int(match.group(1)) if match else 0


def parse_stat_boosts_page(html: str) -> pd.DataFrame:
    """
    Parse the class stat-boosts page into a tidy DataFrame: one row per class.

    Uses extract_table() (manual BeautifulSoup row parsing) rather than
    pandas.read_html(), which turned out to be essential here: this page has
    two rows that act as spoiler-toggle links, each rendered as a single
    wide cell spanning the whole table. pandas.read_html() choked on these
    and silently dropped several subsequent real class rows in the same
    table (7 classes went missing in early testing). Walking rows manually
    and skipping only rows whose cell count doesn't match the header avoids
    that failure mode entirely, since the toggle rows are structurally
    distinct (1 cell) from real data rows (10 cells), regardless of content.
    """
    frames = []
    for section, table in iter_sections_and_tables(html):
        headers, rows = extract_table(table)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=headers)
        df["tier"] = section
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"Name": "name"})
    combined = combined.dropna(subset=["name"]).reset_index(drop=True)

    stat_cols = [c for c in combined.columns if c not in ("name", "tier")]
    for col in stat_cols:
        combined[col] = combined[col].apply(parse_numeric_cell)

    return combined


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching base stats from {BASE_STATS_URL} ...")
    base_stats_html = fetch_html(BASE_STATS_URL)
    base_stats_df = parse_character_page(base_stats_html, "base_stat")
    base_stats_path = DATA_DIR / "character_base_stats.csv"
    base_stats_df.to_csv(base_stats_path, index=False)
    print(f"Saved {len(base_stats_df)} characters to {base_stats_path}")

    print(f"\nFetching growth rates from {GROWTH_RATES_URL} ...")
    growth_rates_html = fetch_html(GROWTH_RATES_URL)
    growth_rates_df = parse_character_page(growth_rates_html, "growth_rate")
    growth_rates_path = DATA_DIR / "character_growth_rates.csv"
    growth_rates_df.to_csv(growth_rates_path, index=False)
    print(f"Saved {len(growth_rates_df)} characters to {growth_rates_path}")

    print(f"\nFetching class stat boosts from {STAT_BOOSTS_URL} ...")
    stat_boosts_html = fetch_html(STAT_BOOSTS_URL)
    stat_boosts_df = parse_stat_boosts_page(stat_boosts_html)
    stat_boosts_path = DATA_DIR / "class_stat_boosts.csv"
    stat_boosts_df.to_csv(stat_boosts_path, index=False)
    print(f"Saved {len(stat_boosts_df)} classes to {stat_boosts_path}")

    print("\nDone.")
    print("\nBase stats preview:")
    print(base_stats_df.head())
    print("\nClass stat boosts preview:")
    print(stat_boosts_df.head())


if __name__ == "__main__":
    main()