"""Brand normalisation must not match on arbitrary substrings.

`_normalize_entity` fell back to:

    for brand_key, brand_id in BRAND_NORMALIZATION.items():
        if brand_key in key or key in brand_key:
            return brand_id

Both halves are wrong, and short keys make it severe. The table contains `fb`
(Meta), `aws` (Amazon) and `gpt` (OpenAI):

  - `brand_key in key` maps **"FBI" -> brand:meta** ("fb" is inside "fbi").
  - `key in brand_key` maps **"Al" -> brand:google** ("al" is inside "alphabet"),
    and any 2-3 letter fragment of a brand name to that brand.

Dict iteration order then decides which wrong answer you get. A sentiment
dashboard aggregating by `canonical_id` silently attributes FBI mentions to
Meta -- the kind of error that is invisible in aggregate and indefensible when
someone checks one row.
"""
from __future__ import annotations

import pytest

from configs.config import BRAND_NORMALIZATION
from src.ner.pipeline import _normalize_entity


# ---------------------------------------------------------------------------
# The specific false positives that shipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface",
    [
        "FBI",  # contains "fb" -> was brand:meta
        "Al",  # inside "alphabet" -> was brand:google
        "AWSome",  # contains "aws" -> was brand:amazon
        "Gptool",  # contains "gpt" -> was brand:openai
        "GP",  # inside "gpt"
        "A",  # inside almost everything
        "Metabolism",  # contains "meta" -> was brand:meta
        "Metallica",  # contains "meta" -> was brand:meta
        "Googleplex",  # contains "google" -> was brand:google
        "Amazonia",  # contains "amazon" -> was brand:amazon
        "Twitterature",  # contains "twitter" -> was brand:twitter
        "Applebee's",  # contains "apple"
    ],
)
def test_unrelated_names_are_not_mapped_to_a_brand(surface):
    assert _normalize_entity(surface, "ORG") is None, (
        f"{surface!r} was mapped to a brand by substring matching"
    )


def test_fbi_is_not_meta():
    """Named separately because it is the clearest demonstration of the bug."""
    assert _normalize_entity("FBI", "ORG") != "brand:meta"
    assert _normalize_entity("FBI", "ORG") is None


@pytest.mark.parametrize(
    "surface, expected",
    [
        ("GOOG", "brand:google"),  # Alphabet class-C ticker
        ("Insta", "brand:meta"),
        ("Elon", "brand:twitter"),
    ],
)
def test_real_mentions_that_the_fix_initially_broke_still_resolve(surface, expected):
    """Removing the fragment direction was right, but it cost real recall.

    These three resolved under the old substring rule and stopped resolving
    under word-boundary matching — an adversarial review caught it. The fix is
    to list them as table entries, not to restore a rule that also mapped "FBI"
    to Meta. Precision and recall have to hold at the same time, which is what
    this test and `test_unrelated_names_are_not_mapped_to_a_brand` assert
    together.
    """
    assert _normalize_entity(surface, "ORG") == expected


# ---------------------------------------------------------------------------
# The real mappings must still work
# ---------------------------------------------------------------------------


def test_every_table_entry_still_resolves_to_itself():
    """The fix must not break the lookups the table exists for."""
    for key, expected in BRAND_NORMALIZATION.items():
        assert _normalize_entity(key, "ORG") == expected, f"{key!r} stopped resolving"
        assert _normalize_entity(key.upper(), "ORG") == expected
        assert _normalize_entity(f"  {key}  ", "ORG") == expected


@pytest.mark.parametrize("prefix", ["$", "#", "@"])
def test_ticker_and_handle_prefixes_are_stripped(prefix):
    for key, expected in list(BRAND_NORMALIZATION.items())[:5]:
        assert _normalize_entity(f"{prefix}{key}", "ORG") == expected


def test_multiword_brands_match_on_whole_words_not_fragments():
    """A multi-word brand should match when the phrase is present, but a single
    fragment of it should not."""
    multiword = [k for k in BRAND_NORMALIZATION if " " in k]
    for key in multiword:
        assert _normalize_entity(key, "ORG") == BRAND_NORMALIZATION[key]
        first = key.split()[0]
        # A bare fragment only resolves if it is itself a table entry.
        if first not in BRAND_NORMALIZATION:
            assert _normalize_entity(first, "ORG") is None, (
                f"fragment {first!r} of {key!r} resolved on its own"
            )


# ---------------------------------------------------------------------------
# Entity-type gating, unchanged behaviour worth pinning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("etype", ["PER", "LOC"])
def test_person_and_location_entities_are_never_normalised(etype):
    first_brand = next(iter(BRAND_NORMALIZATION))
    assert _normalize_entity(first_brand, etype) is None


def test_empty_surface_is_none():
    assert _normalize_entity("", "ORG") is None
    assert _normalize_entity(None, "ORG") is None
