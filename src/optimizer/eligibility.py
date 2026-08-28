"""
optimizer/eligibility.py

Character/gender/unique-class eligibility checks - "may this character
actually access this class," independent of how well it scores for any
role.
"""
from .constants import UNIQUE_STORY_CLASS_TIER



def is_class_eligible(
    character_name: str,
    class_name: str,
    eligibility_lookup: dict,
    character_gender: str | None = None,
) -> bool:
    """
    Whether character_name may access class_name, per eligibility_lookup.

    A class absent from eligibility_lookup is unrestricted (eligible to
    everyone). A restricted class requires the character to be in its
    locked_to_characters set (when one is specified) AND to match its
    locked_to_gender (when one is specified). A character_gender of "Any"
    (the Protagonist, whose gender is a player choice) or None (gender data
    wasn't supplied - degrade gracefully rather than block) always passes
    the gender check; character-lock checks still apply regardless.
    """
    restriction = eligibility_lookup.get(class_name)
    if restriction is None:
        return True

    allowed_characters = restriction["characters"]
    if allowed_characters is not None and character_name not in allowed_characters:
        return False

    required_gender = restriction["gender"]
    if required_gender is not None and character_gender not in (required_gender, "Any", None):
        return False

    return True



def eligible_unique_story_class_by_tier(
    character_name: str,
    eligibility_lookup: dict,
    character_gender: str | None = None,
) -> dict:
    """
    tier -> class_name for each UNIQUE_STORY_CLASS_TIER entry character_name
    is eligible for (see is_class_eligible) - in practice non-empty only
    for the Protagonist and the three house leaders, since every entry in
    UNIQUE_STORY_CLASS_TIER is locked to exactly one character. Used by
    recommend_path and list_eligible_classes_at_tier to splice a
    character's own personal story class into their path at its documented
    tier, rather than only surfacing it via the separate
    eligible_unique_classes callout.
    """
    result = {}
    for class_name, tier in UNIQUE_STORY_CLASS_TIER.items():
        if is_class_eligible(character_name, class_name, eligibility_lookup, character_gender):
            result[tier] = class_name
    return result
