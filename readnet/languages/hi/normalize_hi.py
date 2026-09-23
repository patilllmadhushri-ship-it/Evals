"""Hindi rulebook, executable copy.

One function per rule in RULES.md, named after the rule id. Read the two side
by side, and change them in the same commit. Deterministic; no model calls.
"""

from __future__ import annotations

from .. import Forgiveness, LanguageProfile, Rule, Trace
from .. import devanagari as dv

# -- text rules, in pipeline order ---------------------------------------------------


def hi_01_unicode_nfc(text: str, trace: Trace) -> str:
    return dv.nfc(text)


def hi_02_romanised_to_devanagari(text: str, trace: Trace) -> str:
    return dv.romanised_to_devanagari(text, trace.romanised_words)


def hi_03_one_digit_set(text: str, trace: Trace) -> str:
    return dv.digits_to_ascii(text)


def hi_04_strip_invisible_and_punctuation(text: str, trace: Trace) -> str:
    return dv.strip_invisible_and_punctuation(text)


def hi_05_urdu_nukta_is_silent(text: str, trace: Trace) -> str:
    # ड़ and ढ़ are native Hindi sounds (पढ़ना, सड़क): their dot is kept.
    return dv.per_word(text, lambda w: dv.drop_nukta(w, keep_on=frozenset("डढ")))


#: HI-07: words where chandrabindu vs anusvara is the difference between two words.
HI_07_CHANDRABINDU_PROTECTED = frozenset(dv.nfc(w) for w in ("हँस", "हँसी", "हँसना"))


def hi_06_chandrabindu_equals_anusvara(text: str, trace: Trace) -> str:
    return dv.per_word(text, dv.chandrabindu_to_anusvara, skip=HI_07_CHANDRABINDU_PROTECTED)


def hi_08_class_nasal_equals_anusvara(text: str, trace: Trace) -> str:
    return dv.per_word(text, dv.class_nasal_to_anusvara)


def hi_09_one_letter_one_code(text: str, trace: Trace) -> str:
    return dv.unify_chars(text, {"ॲ": "ऍ", "ऱ": "र"})  # ॲ -> ऍ, ऱ -> र


HI_10_SPELLING_VARIANTS = {
    dv.nfc(k): dv.nfc(v)
    for k, v in {
        "गयी": "गई", "गये": "गए", "नयी": "नई", "आये": "आए",
        "लिये": "लिए", "दिये": "दिए", "किये": "किए", "हुये": "हुए",
        "चाहिये": "चाहिए", "जायेगा": "जाएगा", "जायेगी": "जाएगी",
    }.items()
}


def hi_10_spelling_variants(text: str, trace: Trace) -> str:
    return dv.per_word(text, lambda w: HI_10_SPELLING_VARIANTS.get(w, w))


RULES = (
    Rule("HI-01", hi_01_unicode_nfc, neutral=True),
    Rule("HI-02", hi_02_romanised_to_devanagari, neutral=True),
    Rule("HI-03", hi_03_one_digit_set, neutral=True),
    Rule("HI-04", hi_04_strip_invisible_and_punctuation, neutral=True),
    Rule("HI-05", hi_05_urdu_nukta_is_silent, neutral=True),
    Rule("HI-06", hi_06_chandrabindu_equals_anusvara, neutral=True),
    Rule("HI-08", hi_08_class_nasal_equals_anusvara, neutral=True),
    Rule("HI-09", hi_09_one_letter_one_code, neutral=True),
    Rule("HI-10", hi_10_spelling_variants, neutral=True),
)


def _normalized(*words: str) -> frozenset[str]:
    from .. import run_rules

    return frozenset(run_rules(w, RULES, neutral=False)[0] for w in words)


# -- forgiving rules that need the expected word (run inside score.py) -----------------

#: HI-21: dropping the nasal destroys these words, so HI-20 does not apply.
HI_21_NASAL_CRITICAL = _normalized("चाँद", "हँस")


def hi_20_dropped_nasal_not_assessed(expected, heard, diffs, labels) -> bool:
    return (
        expected not in HI_21_NASAL_CRITICAL
        and bool(diffs)
        and all(r in dv.TABLES.nasal_marks and h == "" for r, h in diffs)
    )


def hi_22_sha_ssa_same_sound(expected, heard, diffs, labels) -> bool:
    return bool(diffs) and all(frozenset((r, h)) == frozenset("शष") for r, h in diffs)


PROFILE = LanguageProfile(
    code="hi",
    name="Hindi",
    rules=RULES,
    tables=dv.TABLES,
    forgiveness=(
        Forgiveness("HI-20", hi_20_dropped_nasal_not_assessed),
        Forgiveness("HI-22", hi_22_sha_ssa_same_sound),
    ),
    fillers=frozenset({"अ", "अं", "आं", "उम", "उम्म", "हम्म", "हं", "एं", "ऊं"}),
    letter_variants=dv.consonant_letter_variants,
    letter_rule="HI-23",
    romanised_rule="HI-24",
    exception_rules=("HI-07", "HI-21"),
)
