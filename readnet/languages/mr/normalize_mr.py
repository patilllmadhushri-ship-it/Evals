"""Marathi normalisation. Every rule here is explained for non-programmers in
RULES.md next to this file — change the two together."""

from __future__ import annotations

from .. import LanguageProfile
from .. import devanagari as dv
from ..hi.normalize_hi import letter_variants

CONFIG = dv.NormalizationConfig(
    # Marathi has no native nukta sounds; every dot is dropped.
    keep_nukta_on=frozenset(),
    chandrabindu_to_anusvara=True,
    # Deliberately empty until a Marathi linguist supplies real minimal pairs.
    protected_words=frozenset(),
    spelling_variants={},
    # ॲ and ऍ both write the English "a" in बॅट/ॲप; the eyelash ऱ in ऱ्या is
    # the same र sound. ळ is NOT unified with ल — they are different sounds.
    char_unifications={"ॲ": "ऍ", "ऱ": "र"},
)


def normalize_mr(text: str) -> tuple[str, dv.Trace]:
    return dv.normalize(text, CONFIG)


PROFILE = LanguageProfile(
    code="mr",
    name="Marathi",
    normalize=normalize_mr,
    fillers=frozenset({"अ", "अं", "आं", "उम", "उम्म", "हम्म", "हं", "एं"}),
    forgiven_pairs=frozenset({frozenset("शष")}),
    letter_variants=letter_variants,
)
