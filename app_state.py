"""
app_state.py

Shared infrastructure every tab in tabs/ draws on: loading the CSVs
once (load_data), resolving a character's display name/portrait/gender
(including Byleth's player-chosen gender - see resolve_character_gender),
the shareable-URL query-param helpers, and the house-color-coded
portrait renderer (including the "portraits looked blurry" fix - see
_load_portrait_data_uri). Nothing here renders a tab on its own; app.py
and tabs/*.py both import from this module, never the other way around.
"""

import base64
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.team_builder import DLC_HOUSE

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow ships with Streamlit itself, but degrade gracefully
    Image = None



DATA_DIR = Path(__file__).resolve().parent / "data"

PORTRAIT_DIR = Path(__file__).resolve().parent / "assets" / "portraits"

DEFAULT_TARGET_LEVEL = 30


# Route-based color coding for the portrait tile/badge (see render_portrait) -
# each character's data/character_base_stats.csv "house" value maps to a
# color evoking that house/route's own game branding, so the fallback tile
# (still all anyone sees by default - see render_portrait's docstring) reads
# as "which house/route is this character from" at a glance instead of a
# single flat neutral box. A house absent from this map (there shouldn't be
# one - every character_base_stats.csv "house" value is covered) falls back
# to DEFAULT_PORTRAIT_COLOR.
HOUSE_COLORS = {
    "Black Eagles": "#7a1620",  # crimson/black, the house's own color scheme
    "Blue Lions": "#1d3f6e",  # royal blue
    "Golden Deer": "#8a5a12",  # gold/amber
    "Church of Seiros": "#7a6a2a",  # muted gold - the Church's own heraldry
    "Knights of Seiros": "#455163",  # steel gray - knights, not students
    "You and the Enigmatic Girl": "#1f6e5c",  # teal - Byleth/Sothis's own mint-teal color motif
    DLC_HOUSE: "#5a1f6e",  # purple - Cindered Shadows/Abyss's distinct visual identity
}

DEFAULT_PORTRAIT_COLOR = "#2b2b3a"


# Byleth (the Protagonist)'s gender is a player choice at the very start of
# the game - data/character_gender.csv already marks them "Any" so this
# choice never blocks a gender-locked class either way (see
# is_class_eligible's docstring). It has no effect on stats or eligibility,
# but it is the one thing that determines which portrait actually depicts
# them, for anyone who's added their own art per assets/portraits/README.md.
# So unlike every other character - whose single portrait file is just
# their lowercased name - Byleth's is resolved from one of these two slugs
# plus the selector rendered once in main() and threaded down through
# render_character_tab/render_team_tab/render_team into render_portrait.
BYLETH_PORTRAIT_SLUGS = {"Male": "byleth_m", "Female": "byleth_f"}

DEFAULT_BYLETH_GENDER = "Male"



def resolve_character_gender(character_name: str, character_gender_df: pd.DataFrame, byleth_gender: str | None = None) -> str | None:
    """
    A character's gender for ELIGIBILITY purposes - character_gender_df's
    own recorded value for everyone, EXCEPT the Protagonist, where the
    actual player-chosen gender (byleth_gender, "Male"/"Female" - the same
    selector that already picks Byleth's portrait) is used instead of the
    CSV's "Any".

    This matters because "Any" alone (is_class_eligible's own graceful-
    degradation value, correct as a permanent FACT about Byleth - there is
    no fixed gender the way every other character has one) also happens to
    mean "never blocked by a gender-locked class either way," which is
    right for "does this class exist for Byleth at all" but wrong for "is
    THIS Byleth, after the player's actual gender choice, eligible for it" -
    the reported bug ("War Master is male-only, Falcon Knight is
    female-only, so it should depend on the gender the player picked")
    was exactly this: Byleth was being offered (and even recommended) both,
    regardless of which portrait/gender was actually selected. Threading
    the real choice through here makes Byleth's gender-locked-class
    eligibility behave exactly like everyone else's fixed gender does.

    Used everywhere in this file that needs a character's gender for
    eligibility (the call into recommend_for_character, the mix-and-match
    option list, import, Team Builder), not just one call site, so every
    gender-aware check in the UI agrees with what was actually recommended.
    """
    if character_name == "Protagonist":
        return byleth_gender or DEFAULT_BYLETH_GENDER
    gender_row = character_gender_df[character_gender_df["name"] == character_name]
    if not gender_row.empty:
        return gender_row.iloc[0]["gender"]
    return None



@st.cache_data
def load_data() -> tuple[pd.DataFrame, ...]:
    required = [
        "character_base_stats.csv", "character_growth_rates.csv", "class_stat_boosts.csv",
        "class_eligibility.csv", "character_gender.csv", "class_weapon_requirements.csv",
        "character_weapon_talent.csv", "recruitment_requirements.csv", "character_starting_level.csv",
        "class_growth_rates.csv", "class_base_stats.csv", "character_relics.csv",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        st.error(
            f"Missing data file(s): {', '.join(missing)}. "
            f"Base stats/growth rates/stat boosts come from `python src/scrape_serenes.py`; "
            f"the rest are hand-maintained and checked into the repo directly (Serenes doesn't "
            f"have this data in scrapable tabular form)."
        )
        st.stop()

    return (
        pd.read_csv(DATA_DIR / "character_base_stats.csv"),
        pd.read_csv(DATA_DIR / "character_growth_rates.csv"),
        pd.read_csv(DATA_DIR / "class_stat_boosts.csv"),
        pd.read_csv(DATA_DIR / "class_eligibility.csv"),
        pd.read_csv(DATA_DIR / "character_gender.csv"),
        pd.read_csv(DATA_DIR / "class_weapon_requirements.csv"),
        pd.read_csv(DATA_DIR / "character_weapon_talent.csv"),
        pd.read_csv(DATA_DIR / "recruitment_requirements.csv"),
        pd.read_csv(DATA_DIR / "character_starting_level.csv"),
        pd.read_csv(DATA_DIR / "class_growth_rates.csv"),
        pd.read_csv(DATA_DIR / "class_base_stats.csv"),
        pd.read_csv(DATA_DIR / "character_relics.csv"),
    )



def query_param_str(key: str, default: str | None = None) -> str | None:
    """
    Read one query-string param (e.g. `?character=Bernadetta` -> "Bernadetta")
    via st.query_params, for prefilling a widget's default on first load -
    see render_character_tab's use of this for "make results shareable ...
    encode a build in the URL." Returns default if st.query_params isn't
    available at all (an older Streamlit, or the test stub - this feature
    degrades to "no shareable link," not a crash) or the key is absent.
    """
    try:
        params = st.query_params
    except Exception:
        return default
    value = params.get(key, default)
    if isinstance(value, list):  # a couple of Streamlit versions return a list for a repeated param
        value = value[0] if value else default
    return value



def query_param_int(key: str, default: int) -> int:
    """Integer counterpart to query_param_str (e.g. `?level=25` -> 25) - falls back to default on anything unparsable."""
    raw = query_param_str(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default



def sync_share_query_params(character: str, role_choice: str, target_level: int) -> None:
    """
    Mirror the Character Optimizer tab's current character/role/level
    selection into the browser's own URL query string via st.query_params,
    so the address bar itself becomes a shareable/bookmarkable link back to
    this exact selection (Streamlit reruns the whole script on every widget
    change, so this runs - and stays current - on every interaction, not
    just once at load). Mix-and-match tier overrides aren't included here -
    see the README's Known Simplifications for why that's a deliberately
    scoped-down version of "shareable," not an oversight.

    A no-op (not a crash) wherever st.query_params isn't available - see
    query_param_str.
    """
    try:
        st.query_params["character"] = character
        st.query_params["role"] = role_choice
        st.query_params["level"] = str(target_level)
    except Exception:
        pass



def get_playable_names(base_stats_df: pd.DataFrame) -> list[str]:
    """NPCs (Sothis, Rhea, etc.) aren't recruitable/playable units - exclude them everywhere."""
    return sorted(n for n in base_stats_df["name"] if "(NPC)" not in n)



def get_dlc_names(base_stats_df: pd.DataFrame) -> set[str]:
    """Characters that require the Cindered Shadows DLC - flagged in the UI, never silently mixed in."""
    return set(base_stats_df[base_stats_df["house"] == DLC_HOUSE]["name"])



def display_name(name: str, dlc_names: set[str]) -> str:
    """Character label for dropdowns/rosters - tags DLC characters instead of listing them indistinguishably."""
    return f"{name} (DLC)" if name in dlc_names else name



def get_portrait_path(character_name: str, byleth_gender: str | None = None) -> Path | None:
    """
    Look up a local portrait image for a character, if one has been placed
    in assets/portraits/ (see that folder's README - actual character art
    isn't shipped in this repo, since Three Houses portraits are Nintendo/
    Intelligent Systems IP and not this project's to redistribute; add your
    own from a source you have the rights to use).

    For the Protagonist specifically, byleth_gender ("Male"/"Female", from
    BYLETH_PORTRAIT_SLUGS - the selector in main() supplies this) is tried
    first as byleth_m/byleth_f, since Byleth's in-game portrait is a player
    choice rather than one fixed image the way every other character's is.
    A plain protagonist.* file - the ordinary lowercased-name slug every
    other character uses - is still checked as a fallback, for anyone who
    added one before splitting Byleth's art by gender, or who doesn't care
    to. byleth_gender is ignored for every other character.
    """
    if not PORTRAIT_DIR.exists():
        return None
    slugs = [character_name.lower().replace(" ", "_").replace("(", "").replace(")", "")]
    if character_name == "Protagonist":
        gender_slug = BYLETH_PORTRAIT_SLUGS.get(byleth_gender or DEFAULT_BYLETH_GENDER)
        slugs.insert(0, gender_slug)
    for slug in slugs:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = PORTRAIT_DIR / f"{slug}{ext}"
            if candidate.exists():
                return candidate
    return None



def get_house_lookup(base_stats_df: pd.DataFrame) -> dict:
    """Index data/character_base_stats.csv's "house" column by character name, for render_portrait's color coding."""
    return dict(zip(base_stats_df["name"], base_stats_df["house"]))



# How much bigger than its CSS display width a portrait is rendered at
# before being embedded, so it looks crisp on a high-DPI/retina display
# instead of blurry - a portrait shown at width=96 is actually encoded at
# 96 * PORTRAIT_RENDER_SCALE px, then constrained back down to 96px via
# CSS, letting the BROWSER's own high-quality downscaling do the final
# sizing rather than whatever resize path an image-serving library takes
# internally. See render_portrait/_load_portrait_data_uri for the "some
# portraits looked blurry" fix this is part of.
PORTRAIT_RENDER_SCALE = 3

# Upper bound on the encoded/source side length, regardless of scale - the
# real portrait files added under assets/portraits/ run from ~1200px up to
# ~2500px per side, so without a cap a team roster's worth of portraits
# would embed several megabytes of base64 image data into the page for no
# visible benefit at these display sizes.
MAX_PORTRAIT_SOURCE_PX = 480



@st.cache_data(show_spinner=False)
def _load_portrait_data_uri(path_str: str, mtime: float, size_px: int) -> str | None:
    """
    Base64 PNG data URI for the portrait at path_str, resized to a
    size_px x size_px square (center-cropped first if the source isn't
    already square), or None if the file can't actually be decoded as an
    image at all.

    Why this exists, not just `st.image(path, width=...)`: auditing
    assets/portraits/ turned up two real problems with the files placed
    there, neither about the art itself. First, seven files (anna.png,
    balthus.png, constance.png, cyril.png, hapi.png, jeritza.png,
    yuri.png) turned out to be non-image content entirely - an HTML error
    page or a JS file saved with a ".png" name, almost certainly a failed/
    blocked download saved without checking what actually came back -
    which would render as a broken-image icon (or worse, an exception) if
    handed to an image widget as-is; these seven have since been removed
    from assets/portraits/ (the character just shows the color-coded tile
    now, same as any other character with no portrait file at all - see
    render_portrait). Second, the remaining real portraits were valid,
    high-resolution (1200px-2500px) WebP images that happened to be named
    ".png" - not literally broken, but an extension that lies about the
    actual format is exactly the kind of mismatch that makes "does this
    decode/display correctly everywhere" a coin flip rather than a
    guarantee, which is a plausible contributor to "portraits are blurry"
    (a decoder that trusted the extension over the content could fail or
    silently mis-render, and any browser-side fallback path is unlikely
    to be a high-quality one); these have since been re-saved on disk as
    genuine PNGs (downscaled to a 900px-per-side cap - still comfortably
    above MAX_PORTRAIT_SOURCE_PX below - so the repo isn't carrying
    multi-megabyte full-res source art no display size here actually uses).

    This function's own decoding/resizing is still worth keeping even
    with clean assets on disk now: it's what guards against the NEXT
    mislabeled or corrupt file someone adds, not just the ones already
    found and fixed. Explicitly decoding here via Pillow (which sniffs
    actual file content, never trusting the extension), re-encoding to a
    real, correctly-sized PNG, and embedding that as a data URI means a
    file that can't actually be decoded as an image cleanly returns None
    here instead of reaching the browser at all, so render_portrait falls
    back to the color-coded tile for exactly that character instead of
    showing a broken icon or raising; and every genuinely valid portrait
    is guaranteed to reach the page as an actual, correctly-labeled,
    appropriately-sized PNG, with the RESIZE done here (Pillow's own
    high-quality LANCZOS filter, not whatever a generic image-serving
    path might use) rather than left to chance.

    @st.cache_data keyed on (path_str, mtime, size_px) - mtime (the
    source file's own modification time) busts the cache automatically if
    a portrait file is ever replaced, without needing any manual cache-
    clearing.
    """
    if Image is None:
        return None
    try:
        with Image.open(path_str) as im:
            im.load()  # force full decode now, while we can still catch a decode error
            im = im.convert("RGBA")
            w, h = im.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            encode_px = min(size_px, MAX_PORTRAIT_SOURCE_PX)
            im = im.resize((encode_px, encode_px), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
    except Exception:
        return None
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"



def render_portrait(character_name: str, house: str | None = None, width: int = 96, byleth_gender: str | None = None):
    """
    Render a character's portrait. If an image has been placed in
    assets/portraits/ for them (see get_portrait_path/that folder's README)
    AND it can actually be decoded as an image (see
    _load_portrait_data_uri), it's shown, resized and embedded at a
    display-crisp resolution - "bring your own art" stays supported for
    anyone who wants to add their own. Otherwise (no file at all, or one
    that can't be decoded - see _load_portrait_data_uri's docstring for why
    that second case can happen), the fallback tile is color-coded by
    house/route (see HOUSE_COLORS) rather than a single flat neutral box,
    so portraits are still visually distinct and meaningful - which house/
    route a character belongs to - without redistributing anyone's IP or
    ever showing a broken-image icon. `house` should be that character's
    data/character_base_stats.csv "house" value (see get_house_lookup);
    omitted or unrecognized falls back to DEFAULT_PORTRAIT_COLOR.
    `byleth_gender` only matters when character_name is the Protagonist -
    see get_portrait_path.
    """
    portrait = get_portrait_path(character_name, byleth_gender=byleth_gender)
    data_uri = None
    if portrait:
        data_uri = _load_portrait_data_uri(
            str(portrait), portrait.stat().st_mtime, width * PORTRAIT_RENDER_SCALE,
        )
    if data_uri:
        st.markdown(
            f"<img data-portrait-file='{portrait.name}' src='{data_uri}' width='{width}' height='{width}' "
            f"style='border-radius:8px;object-fit:cover;display:block;' />",
            unsafe_allow_html=True,
        )
    else:
        color = HOUSE_COLORS.get(house, DEFAULT_PORTRAIT_COLOR)
        title = house or ""
        st.markdown(
            f"<div title='{title}' style='width:{width}px;height:{width}px;border-radius:8px;"
            f"background:{color};color:#fff;display:flex;align-items:center;justify-content:center;"
            f"font-size:{width//3}px;font-weight:600;'>"
            f"{character_name[0]}</div>",
            unsafe_allow_html=True,
        )
