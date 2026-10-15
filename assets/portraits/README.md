# Character portraits

Character portraits shown in the app (character picker, team roster cards).
Portraits are included for the playable cast. If a character's image is
missing, the UI falls back to a placeholder tile with the character's
initial, colored by their house/route (see `HOUSE_COLORS` in `app.py`), so
it never breaks or leaves a gap.

## Credit

Character portraits are official *Fire Emblem: Three Houses* art, ©
Nintendo / Intelligent Systems / Koei Tecmo. They're included for
non-commercial fan and portfolio use only and are not covered by this
repository's MIT license. If you're a rights holder and want them removed,
please open an issue.

## Adding or replacing a portrait

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

### Byleth is the one exception

Byleth's gender is a choice the player makes at the start of the game, not
a fixed trait the way it is for every other character - so a plain
`protagonist.png` can only ever show one of them. Add both instead, named:

```
byleth_m.png
byleth_f.png
```

and a "Byleth's portrait" selector next to the app's title (see `main()`
in `app.py`) picks between them - it's purely cosmetic, the same as every
other portrait, and has no effect on stats or class eligibility either
way (Byleth's gender is already recorded as "Any" in
`data/character_gender.csv`, so it never blocks a gender-locked class in
either direction - see `is_class_eligible`). A plain `protagonist.png` is
still honored as a fallback if you'd rather use one image for both.
