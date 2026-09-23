"""Marathi rulebook, executable copy.

One function per rule in RULES.md, named after the rule id. Read the two side
by side, and change them in the same commit. Deterministic; no model calls.
"""

from __future__ import annotations

from .. import Forgiveness, LanguageProfile, Rule, Trace
from .. import devanagari as dv


def mr_01_unicode_nfc(text: str, trace: Trace) -> str:
    return dv.nfc(text)


def mr_02_romanised_to_devanagari(text: str, trace: Trace) -> str:
    return dv.romanised_to_devanagari(text, trace.romanised_words)


def mr_03_one_digit_set(text: str, trace: Trace) -> str:
    return dv.digits_to_ascii(text)


def mr_04_strip_invisible_and_punctuation(text: str, trace: Trace) -> str:
    return dv.strip_invisible_and_punctuation(text)


def mr_05_nukta_is_silent(text: str, trace: Trace) -> str:
    # Marathi has no native sounds written with a nukta.
    return dv.per_word(text, dv.drop_nukta)


def mr_06_chandrabindu_equals_anusvara(text: str, trace: Trace) -> str:
    return dv.per_word(text, dv.chandrabindu_to_anusvara)


def mr_08_class_nasal_equals_anusvara(text: str, trace: Trace) -> str:
    return dv.per_word(text, dv.class_nasal_to_anusvara)


def mr_09_one_letter_one_code(text: str, trace: Trace) -> str:
    # ॲ/ऍ are one letter; eyelash ऱ is the same र sound. ळ is NOT ल.
    return dv.unify_chars(text, {"ॲ": "ऍ", "ऱ": "र"})


RULES = (
    Rule("MR-01", mr_01_unicode_nfc, neutral=True),
    Rule("MR-02", mr_02_romanised_to_devanagari, neutral=True),
    Rule("MR-03", mr_03_one_digit_set, neutral=True),
    Rule("MR-04", mr_04_strip_invisible_and_punctuation, neutral=True),
    Rule("MR-05", mr_05_nukta_is_silent, neutral=True),
    Rule("MR-06", mr_06_chandrabindu_equals_anusvara, neutral=True),
    Rule("MR-08", mr_08_class_nasal_equals_anusvara, neutral=True),
    Rule("MR-09", mr_09_one_letter_one_code, neutral=True),
)


def mr_22_sha_ssa_same_sound(expected, heard, diffs, labels) -> bool:
    return bool(diffs) and all(frozenset((r, h)) == frozenset("शष") for r, h in diffs)


PROFILE = LanguageProfile(
    code="mr",
    name="Marathi",
    rules=RULES,
    tables=dv.TABLES,
    # No dropped-nasal rule: in Marathi the anusvara often carries grammar
    # (plural/oblique), so that decision waits for a Marathi linguist.
    forgiveness=(Forgiveness("MR-22", mr_22_sha_ssa_same_sound),),
    fillers=frozenset({"अ", "अं", "आं", "उम", "उम्म", "हम्म", "हं", "एं"}),
    letter_variants=dv.consonant_letter_variants,
    letter_rule="MR-23",
    romanised_rule="MR-24",
)
