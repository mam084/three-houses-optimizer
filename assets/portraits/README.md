# Character portraits

`app.py` looks for a portrait image here for every character it displays
(character picker, team roster cards). If a file's here, it's shown; if
not - the default for virtually everyone, since no real art is bundled
(see below) - the UI falls back to a placeholder tile with the character's
initial, colored by their own house/route (Black Eagles, Blue Lions,
Golden Deer, Church of Seiros, Knights of Seiros, the Protagonist/Sothis's
own house, or DLC-exclusive - see `HOUSE_COLORS` in `app.py`), so
portraits stay visually meaningful without needing any actual art at all.
It never breaks or leaves a gap.

## Why there aren't any checked in

Three Houses character portraits and official art are owned by Nintendo /
Intelligent Systems / Koei Tecmo, not this project. This repo scrapes
*stats* (numbers - base stats, growth rates, class boosts) from Serenes
Forest, which is a defensible fair-use-ish case for a portfolio project;
redistributing character *art* is a meaningfully different and riskier
claim, so none is bundled here. You'll need to add your own, from a source
you actually have the rights to use - a few realistic options:

- Screenshots you take yourself from your own copy of the game (personal/
  portfolio use, not redistributed at scale, is the safest lane).
- Officially licensed art you've separately obtained the rights to use.
- Simple placeholder art you draw or generate yourself, if you just want
  visual distinction between characters rather than "official" portraits.

If you ever deploy this publicly (not just run it locally / show it in a
portfolio interview), it's worth revisiting whether bundling real character
art alongside the app is something you're comfortable with - that's a
judgment call for you to make, not something the code enforces.

## How to add one

Drop an image named after the character, lowercased, spaces replaced with
underscores, parentheses stripped - e.g.:

```
edelgard.png
dedue.jpg
protagonist.png
```

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`. See
`get_portrait_path()` in `app.py` for the exact slugging logic if a name
isn't matching.
