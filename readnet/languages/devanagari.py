"""Script-level machinery shared by every Devanagari language module.

Each language's ``normalize_xx.py`` is a thin configuration over the steps here:
the steps themselves (NFC, transliteration, control-character removal, harmless
diacritics, lexicon lookups) are the same for Hindi and Marathi, and what
differs — which diacritics are harmless, which words must keep theirs, which
spellings are interchangeable — is passed in as data.

The order of the steps is fixed and not reversible. RULES.md in each language
folder explains every step in plain English; this file is the executable copy.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

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
INDEPENDENT_VOWELS = set(chr(c) for c in range(0x0904, 0x0915)) | {"\u0960", "\u0961", "\u0972"}
MATRAS = set(chr(c) for c in range(0x093E, 0x094D)) | {"\u0962", "\u0963"}

_DEVANAGARI_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

#: Nasal + virama before a stop of its own class is the same sound as an
#: anusvara: संत and सन्त, अंक and अङ्क. Keyed by nasal, valued by its class.
_NASAL_CLASSES = {
    "ङ": set("कखगघ"),
    "ञ": set("चछजझ"),
    "ण": set("टठडढ"),
    "न": set("तथदध"),
    "म": set("पफबभ"),
}

# -- phonetic and visual confusion tables (used by score.py) ------------------


def _pairs(*pairs: str) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(p) for p in pairs)


ASPIRATION = _pairs("कख", "गघ", "चछ", "जझ", "टठ", "डढ", "तथ", "दध", "पफ", "बभ")
VOICING = _pairs("कग", "खघ", "चज", "छझ", "टड", "ठढ", "तद", "थध", "पब", "फभ")
RETROFLEX_DENTAL = _pairs("टत", "ठथ", "डद", "ढध", "णन", "ळल")
SIBILANT = _pairs("शष", "शस", "षस")
VOWEL_LENGTH = frozenset(
    frozenset(p)
    for p in (("ि", "ी"), ("ु", "ू"), ("अ", "आ"), ("इ", "ई"), ("उ", "ऊ"), ("े", "ै"), ("ो", "ौ"))
)
#: Letters children (and adults) confuse because they look alike, not because
#: they sound alike: घ/ध and भ/म differ by one stroke, ख is written like रव.
VISUAL = _pairs("घध", "भम", "बव", "पष", "थय", "ङड", "मस", "नल")


# -- normalisation --------------------------------------------------------------


@dataclass
class Trace:
    """What each normalisation step changed, for debugging a disputed verdict."""

    steps: list[tuple[str, str, str]] = field(default_factory=list)
    transliterated: list[str] = field(default_factory=list)
    #: The Devanagari words that came from Latin script. Romanisation cannot
    #: show vowel length or dental vs retroflex, so the scorer forgives those
    #: differences on these words rather than blame the child for them.
    romanised_words: set[str] = field(default_factory=set)

    def record(self, step: str, before: str, after: str) -> None:
        if before != after:
            self.steps.append((step, before, after))


@dataclass(frozen=True)
class NormalizationConfig:
    """What a language decides; the steps are shared."""

    #: Consonants whose nukta is a real phoneme and must be kept (Hindi ड़/ढ़).
    keep_nukta_on: frozenset[str] = frozenset()
    #: Treat chandrabindu as anusvara (except on protected words).
    chandrabindu_to_anusvara: bool = True
    #: Words where a diacritic changes the word: never stripped.
    protected_words: frozenset[str] = frozenset()
    #: Interchangeable spellings of the same spoken word -> one canonical form.
    spelling_variants: dict[str, str] = field(default_factory=dict)
    #: Extra single-character unifications applied everywhere (ॲ -> ऍ, ऱ -> र).
    char_unifications: dict[str, str] = field(default_factory=dict)


def step_nfc(text: str) -> str:
    """Step 1. One byte sequence per visible character.

    NFC also *decomposes* the precomposed nukta letters (क़ U+0958 becomes
    क + ़), which is what lets step 4 treat every nukta the same way.
    """
    return unicodedata.normalize("NFC", text)


# Romanised Hindi/Marathi, longest match first. This is a fallback for an ASR
# engine that answers in Latin script; it cannot resolve every ambiguity
# (t is त or ट, final a is अ or आ), so transliterated words are logged in the
# trace and a disputed verdict on one should be read with that in mind.
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


def step_transliterate(text: str, trace: Trace) -> str:
    """Step 2. Latin-script words back to Devanagari; Devanagari digits to ASCII."""

    def swap(match: re.Match) -> str:
        converted = transliterate_latin_word(match.group(0))
        trace.transliterated.append(f"{match.group(0)}->{converted}")
        trace.romanised_words.add(converted)
        return converted

    text = _LATIN_RUN.sub(swap, text)
    return "".join(_DEVANAGARI_DIGITS.get(ch, ch) for ch in text)


def _is_word_char(ch: str) -> bool:
    return ("\u0900" <= ch <= "\u097f" and ch not in "\u0964\u0965\u0970") or ch.isdigit()


def step_strip_controls(text: str) -> str:
    """Step 3. Drop ZWJ/ZWNJ and other invisible marks; punctuation becomes space."""
    text = "".join(ch for ch in text if ch not in INVISIBLE)
    text = "".join(ch if _is_word_char(ch) else " " for ch in text)
    return " ".join(text.split())


def _strip_word(word: str, config: NormalizationConfig) -> str:
    out: list[str] = []
    for i, ch in enumerate(word):
        ch = config.char_unifications.get(ch, ch)
        if ch == NUKTA:
            if out and out[-1] in config.keep_nukta_on:
                out.append(ch)
            continue
        if ch == CHANDRABINDU and config.chandrabindu_to_anusvara:
            ch = ANUSVARA
        # Class nasal + virama before a same-class stop -> anusvara.
        if (
            ch == VIRAMA
            and out
            and out[-1] in _NASAL_CLASSES
            and i + 1 < len(word)
            and word[i + 1] in _NASAL_CLASSES[out[-1]]
        ):
            out[-1] = ANUSVARA
            continue
        out.append(ch)
    return "".join(out)


def step_harmless_diacritics(text: str, config: NormalizationConfig) -> str:
    """Step 4 (guarded by step 5's protected-word list, checked first per word)."""
    return " ".join(
        word if word in config.protected_words else _strip_word(word, config)
        for word in text.split()
    )


def step_lexicon(text: str, config: NormalizationConfig) -> str:
    """Step 5. Word-level lookups: fold interchangeable spellings to one form."""
    return " ".join(config.spelling_variants.get(word, word) for word in text.split())


def normalize(text: str, config: NormalizationConfig) -> tuple[str, Trace]:
    trace = Trace()
    for name, fn in (
        ("nfc", step_nfc),
        ("transliterate", lambda t: step_transliterate(t, trace)),
        ("strip_controls", step_strip_controls),
        ("harmless_diacritics", lambda t: step_harmless_diacritics(t, config)),
        ("lexicon", lambda t: step_lexicon(t, config)),
    ):
        before = text
        text = fn(text)
        trace.record(name, before, text)
    return text, trace


def protected(*words: str) -> frozenset[str]:
    """Protected words are matched after NFC, so store them that way."""
    return frozenset(unicodedata.normalize("NFC", w) for w in words)


def variants(mapping: dict[str, str]) -> dict[str, str]:
    return {unicodedata.normalize("NFC", k): unicodedata.normalize("NFC", v) for k, v in mapping.items()}
