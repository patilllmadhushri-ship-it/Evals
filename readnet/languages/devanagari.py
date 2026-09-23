"""Devanagari building blocks shared by the Hindi and Marathi rulebooks.

Nothing here is a rule by itself. Each language's ``normalize_xx.py`` has one
function per rule, named after the rule in its RULES.md, and those functions
call the helpers below. So a rule is read in RULES.md, found by its id in
``normalize_xx.py``, and tested by the rows of ``tests/field_cases.csv`` that
carry that id.
"""

from __future__ import annotations

import re
import unicodedata

from . import ScriptTables, pairs

# -- code points ---------------------------------------------------------------

NUKTA = "\u093c"
VIRAMA = "\u094d"
ANUSVARA = "\u0902"
CHANDRABINDU = "\u0901"
VISARGA = "\u0903"
ZWJ = "\u200d"
ZWNJ = "\u200c"

#: Format characters that change rendering and carry no sound.
INVISIBLE = {ZWJ, ZWNJ, "\u200b", "\u2060", "\ufeff", "\u00ad"}

CONSONANTS = set(chr(c) for c in range(0x0915, 0x093A)) | {"\u0931", "\u0933", "\u0934"}
MATRAS = set(chr(c) for c in range(0x093E, 0x094D)) | {"\u0962", "\u0963"}

_DEVANAGARI_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

#: Nasal + virama before a stop of its own class is the same sound as an
#: anusvara: संत and सन्त, अंक and अङ्क. Keyed by nasal, valued by its class.
NASAL_CLASSES = {
    "ङ": set("कखगघ"),
    "ञ": set("चछजझ"),
    "ण": set("टठडढ"),
    "न": set("तथदध"),
    "म": set("पफबभ"),
}

#: The hand-made confusion tables. The mature version is estimated from field
#: data — see readnet/confusion.py — and should replace these once it exists.
TABLES = ScriptTables(
    phonetic={
        "aspiration": pairs("कख", "गघ", "चछ", "जझ", "टठ", "डढ", "तथ", "दध", "पफ", "बभ"),
        "voicing": pairs("कग", "खघ", "चज", "छझ", "टड", "ठढ", "तद", "थध", "पब", "फभ"),
        "retroflex_dental": pairs("टत", "ठथ", "डद", "ढध", "णन", "ळल"),
        "sibilant": pairs("शष", "शस", "षस"),
        "vowel_length": frozenset(
            frozenset(p)
            for p in (("ि", "ी"), ("ु", "ू"), ("अ", "आ"), ("इ", "ई"), ("उ", "ऊ"), ("े", "ै"), ("ो", "ौ"))
        ),
    },
    # Look alike, sound different: घ/ध and भ/म differ by one stroke.
    visual=pairs("घध", "भम", "बव", "पष", "थय", "ङड", "मस", "नल"),
    vowel_signs=frozenset(MATRAS),
    nasal_marks=frozenset({ANUSVARA, CHANDRABINDU}),
    joiners=frozenset({VIRAMA}),
    #: Adding or dropping ा is a length error, not a missing vowel.
    length_marks=frozenset({"ा"}),
)

# -- helpers the rule functions call -------------------------------------------------


def nfc(text: str) -> str:
    """NFC also decomposes the precomposed nukta letters (क़ U+0958 -> क + ़),
    which is what lets one rule treat every nukta the same way."""
    return unicodedata.normalize("NFC", text)


# Romanised Hindi/Marathi, longest match first. A fallback for an engine that
# answers in Latin script; it cannot resolve every ambiguity (t is त or ट,
# i is ि or ी), which is why score.py forgives exactly those differences on
# words that came through here.
_LATIN_CONSONANTS = [
    ("chh", "छ"), ("ksh", "क्ष"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"),
    ("th", "थ"), ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"),
    ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"), ("t", "त"), ("d", "द"), ("n", "न"),
    ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"),
    ("w", "व"), ("s", "स"), ("h", "ह"), ("f", "फ"), ("z", "ज"), ("q", "क"), ("x", "क्स"),
]
_LATIN_VOWELS = [
    ("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("ee", "ई", "ी"), ("ii", "ई", "ी"),
    ("oo", "ऊ", "ू"), ("uu", "ऊ", "ू"), ("a", "अ", ""), ("i", "इ", "ि"), ("u", "उ", "ु"),
    ("e", "ए", "े"), ("o", "ओ", "ो"),
]


def _match(table, word: str, i: int):
    for entry in table:
        if word.startswith(entry[0], i):
            return entry
    return None


def transliterate_latin_word(word: str) -> str:
    word = word.lower()
    out: list[str] = []
    i = 0
    while i < len(word):
        consonant = _match(_LATIN_CONSONANTS, word, i)
        if consonant:
            i += len(consonant[0])
            out.append(consonant[1])
            vowel = _match(_LATIN_VOWELS, word, i)
            if vowel:
                i += len(vowel[0])
                # Word-final "a" in romanised Hindi is nearly always आ (sita, kamla).
                if vowel[0] == "a" and i == len(word) and len(out) > 1:
                    out.append("ा")
                else:
                    out.append(vowel[2])
            else:
                following = _match(_LATIN_CONSONANTS, word, i)
                if following and following[1] == consonant[1]:  # pakka -> पक्का
                    out.append(VIRAMA)
                    i += len(following[0])
                    out.append(following[1])
                    vowel = _match(_LATIN_VOWELS, word, i)
                    if vowel:
                        i += len(vowel[0])
                        out.append("ा" if vowel[0] == "a" and i == len(word) else vowel[2])
            continue
        vowel = _match(_LATIN_VOWELS, word, i)
        if vowel:
            i += len(vowel[0])
            out.append(vowel[1])
            continue
        i += 1  # anything unmapped is dropped
    return "".join(out)


_LATIN_RUN = re.compile(r"[A-Za-z]+")


def romanised_to_devanagari(text: str, converted: list[str]) -> str:
    """Latin words -> Devanagari; appends each converted word to `converted`."""

    def swap(match: re.Match) -> str:
        word = transliterate_latin_word(match.group(0))
        converted.append(word)
        return word

    return _LATIN_RUN.sub(swap, text)


def digits_to_ascii(text: str) -> str:
    return "".join(_DEVANAGARI_DIGITS.get(ch, ch) for ch in text)


def _is_word_char(ch: str) -> bool:
    return ("\u0900" <= ch <= "\u097f" and ch not in "\u0964\u0965\u0970") or ch.isdigit()


def strip_invisible_and_punctuation(text: str) -> str:
    text = "".join(ch for ch in text if ch not in INVISIBLE)
    text = "".join(ch if _is_word_char(ch) else " " for ch in text)
    return " ".join(text.split())


def per_word(text: str, fn, skip: frozenset[str] = frozenset()) -> str:
    return " ".join(word if word in skip else fn(word) for word in text.split())


def drop_nukta(word: str, keep_on: frozenset[str] = frozenset()) -> str:
    out: list[str] = []
    for ch in word:
        if ch == NUKTA and not (out and out[-1] in keep_on):
            continue
        out.append(ch)
    return "".join(out)


def chandrabindu_to_anusvara(word: str) -> str:
    return word.replace(CHANDRABINDU, ANUSVARA)


def class_nasal_to_anusvara(word: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(word):
        if (
            ch == VIRAMA
            and out
            and out[-1] in NASAL_CLASSES
            and i + 1 < len(word)
            and word[i + 1] in NASAL_CLASSES[out[-1]]
        ):
            out[-1] = ANUSVARA
            continue
        out.append(ch)
    return "".join(out)


def unify_chars(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def consonant_letter_variants(letter: str) -> set[str]:
    """A child reading क says "ka"; engines write that as क, का or कअ."""
    if len(letter) == 1 and letter in CONSONANTS:
        return {letter, letter + "ा", letter + "अ"}
    return {letter}
