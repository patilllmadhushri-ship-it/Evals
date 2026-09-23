"""Hindi normalisation. Every rule here is explained for non-programmers in
RULES.md next to this file — change the two together."""

from __future__ import annotations

from .. import LanguageProfile
from .. import devanagari as dv

CONFIG = dv.NormalizationConfig(
    # ड़ and ढ़ are native Hindi sounds (पढ़ना, सड़क): the dot is kept. The dot on
    # क़ ख़ ग़ ज़ फ़ marks an Urdu/Persian sound most speakers do not make, so
    # it is dropped — ज़रा and जरा are one word.
    keep_nukta_on=frozenset("डढ"),
    chandrabindu_to_anusvara=True,
    # Words where chandrabindu vs anusvara is the difference between two words.
    protected_words=dv.protected("हँस", "हंस", "हँसी", "हंसी"),
    spelling_variants=dv.variants(
        {
            "गयी": "गई",
            "गये": "गए",
            "नयी": "नई",
            "आये": "आए",
            "लिये": "लिए",
            "दिये": "दिए",
            "किये": "किए",
            "हुये": "हुए",
            "चाहिये": "चाहिए",
            "जायेगा": "जाएगा",
            "जायेगी": "जाएगी",
        }
    ),
    char_unifications={"ॲ": "ऍ", "ऱ": "र"},  # ॲ -> ऍ, ऱ -> र
)


def normalize_hi(text: str) -> tuple[str, dv.Trace]:
    return dv.normalize(text, CONFIG)


def letter_variants(letter: str) -> set[str]:
    """A child reading क says "ka"; engines write that as क, का or कअ."""
    if len(letter) == 1 and letter in dv.CONSONANTS:
        return {letter, letter + "ा", letter + "अ"}
    return {letter}


PROFILE = LanguageProfile(
    code="hi",
    name="Hindi",
    normalize=normalize_hi,
    fillers=frozenset({"अ", "अं", "आं", "उम", "उम्म", "हम्म", "हं", "एं", "ऊं"}),
    forgiven_pairs=frozenset({frozenset("शष")}),
    letter_variants=letter_variants,
)
