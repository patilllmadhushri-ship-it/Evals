"""Grapheme-to-phoneme for Hindi and Marathi: text → the sounds a reader says.

घर is written with two letters but spoken as three sounds, /ɡʰ ə ɾ/: every
consonant carries an unwritten vowel (schwa, /ə/) unless something removes
it. Pronunciation scoring has to check sounds, not letters — the /ə/ in घर is
never written, so a letter-level check cannot score it.

The rules, in order:

1. Letters to sounds (tables below). A consonant with no vowel sign gets /ə/;
   a vowel sign replaces it; a virama (्) removes it.
2. Nasal marks: anusvara before a stop becomes that stop's nasal (अंक →
   /ə ŋ k/); elsewhere, and chandrabindu always, nasalises the vowel.
3. Schwa deletion (what makes Hindi spelling not match speech):
   - word-final /ə/ is dropped (घर → /ɡʰ ə ɾ/), except in a one-letter word
     (a letter read alone, क → /k ə/);
   - a medial /ə/ between vowel+consonant and consonant+vowel is dropped,
     scanning right to left (कमला → /k ə m l aː/, not /k ə m ə l aː/).

The symbols are IPA as used by eSpeak, which is what the phoneme model was
trained on. Where the model has no symbol for a Hindi sound, the nearest one
it has is used and listed in APPROXIMATIONS, so a disputed score can be read
with that in mind. The rules are the standard textbook account and will be
wrong for some words (compounds, loan words); exceptions belong in the
language's rulebook, like the other rules.
"""

from __future__ import annotations

from .languages import devanagari as dv

CONSONANTS = {
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʰ", "ङ": "ŋ",
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "ɟʰ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʰ", "ण": "ɳ",
    "त": "t", "थ": "tʰ", "द": "d", "ध": "dʰ", "न": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʰ", "म": "m",
    "य": "j", "र": "ɾ", "ल": "l", "व": "ʋ", "ळ": "ɭ",
    "श": "ʃ", "ष": "ʃ", "स": "s", "ह": "h",
}
#: Consonant + nukta: the sounds that survive normalisation (ड़, ढ़).
NUKTA_CONSONANTS = {"ड": "ɽ", "ढ": "ɽ", "क": "k", "ख": "x", "ग": "ɣ", "ज": "z", "फ": "f"}

VOWELS = {  # independent vowel letter -> sound
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː", "ऋ": "ɾɪ",
    "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː", "ऍ": "æ", "ऑ": "ɔ",
}
MATRAS = {  # dependent vowel sign -> sound
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː", "ृ": "ɾɪ",
    "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː", "ॅ": "æ", "ॉ": "ɔ",
}
#: Nasal of each stop's place of articulation (for anusvara before a stop).
HOMORGANIC = {
    **dict.fromkeys(["k", "kʰ", "ɡ", "ɡʰ"], "ŋ"),
    **dict.fromkeys(["tʃ", "tʃʰ", "dʒ", "ɟʰ"], "ɲ"),
    **dict.fromkeys(["ʈ", "ʈʰ", "ɖ", "ɖʰ"], "ɳ"),
    **dict.fromkeys(["t", "tʰ", "d", "dʰ"], "n"),
    **dict.fromkeys(["p", "pʰ", "b", "bʰ"], "m"),
}
NASAL_VOWEL = {"aː": "ɑ̃", "ə": "ə̃", "ɪ": "ĩ", "iː": "ĩ", "ʊ": "ũ", "uː": "ũ", "eː": "ẽ", "ɛː": "ɛ̃", "oː": "õ", "ɔː": "ɔ̃"}
VOWEL_SOUNDS = set(VOWELS.values()) | set(MATRAS.values()) | set(NASAL_VOWEL.values()) | {"ə"}

#: Hindi sounds the phoneme model has no symbol for, and what stands in.
APPROXIMATIONS = {"झ": "ɟʰ for /dʒʱ/", "थ": "tʰ (dental)", "ढ़": "ɽ (unaspirated)", "ए/ओ": "eː/oː"}

SCHWA = "ə"


def word_to_phonemes(word: str) -> list[str]:
    """One (already normalised) word → its phonemes."""
    return [p for p, _ in _phones_with_sources(word)]


def letter_sounds(word: str) -> list[str]:
    """For each character of `word`, the sounds it stands for, after schwa
    deletion: घर -> ["ɡʰ ə", "ɾ"]; a vowel sign gives its vowel; a virama or
    nukta gives "" (its sound is already on the consonant)."""
    sounds = [[] for _ in word]
    for phone, source in _phones_with_sources(word):
        sounds[source].append(phone)
    return [" ".join(s) for s in sounds]


def _phones_with_sources(word: str) -> list[tuple[str, int]]:
    phones: list[str] = []
    sources: list[int] = []
    #: index in `phones` of each inherent schwa, for the deletion pass.
    inherent: list[int] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        start = i
        if ch in CONSONANTS:
            if nxt == dv.NUKTA:
                phones.append(NUKTA_CONSONANTS.get(ch, CONSONANTS[ch]))
                i += 1
                nxt = chars[i + 1] if i + 1 < len(chars) else ""
            else:
                phones.append(CONSONANTS[ch])
            sources.append(start)
            if nxt in MATRAS:
                phones.append(MATRAS[nxt])
                sources.append(i + 1)
                i += 1
            elif nxt == dv.VIRAMA:
                i += 1
            else:
                inherent.append(len(phones))
                phones.append(SCHWA)
                sources.append(start)
        elif ch in VOWELS:
            phones.append(VOWELS[ch])
            sources.append(start)
        elif ch in (dv.ANUSVARA, dv.CHANDRABINDU) and phones:
            following = CONSONANTS.get(nxt)
            if ch == dv.ANUSVARA and following in HOMORGANIC:
                phones.append(HOMORGANIC[following])
                sources.append(start)
            elif phones[-1] in NASAL_VOWEL:
                phones[-1] = NASAL_VOWEL[phones[-1]]
            else:
                phones.append("n")
                sources.append(start)
        elif ch == dv.VISARGA:
            phones.append("h")
            sources.append(start)
        i += 1
    kept = _delete_schwas(phones, inherent)
    return [(phones[k], sources[k]) for k in kept]


def _delete_schwas(phones: list[str], inherent: list[int]) -> list[int]:
    """Indices of the phones that survive schwa deletion."""
    drop: set[int] = set()
    consonant_count = sum(1 for p in phones if p not in VOWEL_SOUNDS)
    # Word-final: घर -> /ɡʰ ə ɾ/. A lone consonant (a letter read aloud) keeps it.
    if inherent and inherent[-1] == len(phones) - 1 and consonant_count > 1:
        drop.add(inherent[-1])

    def is_vowel(k: int) -> bool:
        return 0 <= k < len(phones) and k not in drop and phones[k] in VOWEL_SOUNDS

    def is_consonant(k: int) -> bool:
        return 0 <= k < len(phones) and k not in drop and phones[k] not in VOWEL_SOUNDS

    # Medial, right to left: V C [ə] C V -> V C C V (कमला -> /kəmlaː/).
    for k in reversed(inherent):
        if k in drop:
            continue
        if is_consonant(k - 1) and is_vowel(k - 2) and is_consonant(k + 1) and is_vowel(k + 2):
            drop.add(k)
    return [k for k in range(len(phones)) if k not in drop]


def text_to_phonemes(normalized_text: str) -> list[list[str]]:
    """Normalised text → one phoneme list per word."""
    return [word_to_phonemes(w) for w in normalized_text.split()]


def show(phones: list[str]) -> str:
    return "/" + " ".join(phones) + "/"
