"""Which letters and sounds the child got wrong in a reading.

A score says how many mistakes; a teacher needs to know which ones — "the
child cannot yet say घ" is what tells them what to practise. Each scored
reading is broken into one entry per letter the child tried, marked right or
wrong, by the strongest evidence available:

* ``letter_check`` — a letter task decided from the audio, letter by letter;
* ``gop`` — each aligned sound's GOP against the threshold in use;
* ``text`` — no audio model: the letters of a word the transcript shows
  misread (from the same alignment the scorer uses) are wrong, the rest right.

Differences the rulebook forgives (a dropped nasal in गाँव, HI-20) are not
held against the child. Nothing is stored.
"""

from __future__ import annotations

from .. import g2p, languages
from ..score import classify_substitution

#: Marks with no sound of their own: never listed as a "letter".
SILENT = {"\u094d", "\u093c"}  # virama, nukta


def wrong_letters(result: dict, language: str, threshold: float) -> list[dict]:
    """The letters the child got wrong in one scored reading, in reading order."""
    return [e for e in sound_events(result, language, threshold) if not e["ok"]]


def sound_events(result: dict, language: str, threshold: float):
    """One row per letter/sound the reader attempted in a scored reading."""
    if result.get("letter_check"):
        for c in result["letter_check"]:
            yield {"word": c["letter"], "letter": c["letter"], "sound": " ".join(g2p.word_to_phonemes(c["letter"])),
                   "gop": c["margin"], "ok": c["heard"] == c["letter"], "heard": c["heard"], "source": "letter_check"}
        return
    tables = languages.get(language).tables
    for op in result.get("ops", []):
        word = op.get("ref")
        if not word or op.get("kind") == "deletion":
            continue  # a skipped word says nothing about which sounds the child can make
        if op.get("units"):
            for u in op["units"]:
                if u.get("sounds") and u.get("letter") not in SILENT:
                    yield {"word": word, "letter": u.get("letter") or u["sounds"], "sound": u["sounds"],
                           "gop": u["gop"], "ok": u["gop"] >= threshold, "heard": u.get("heard", ""),
                           "source": "gop"}
            continue
        sounds = g2p.letter_sounds(word)
        wrong: set[int] = set()
        if op.get("kind") == "substitution" and op.get("counts_as_mistake"):
            hyp = op.get("hyp") or ""
            diffs = classify_substitution(word, hyp, tables).char_diffs
            wrong_letters = {r for r, _ in diffs if r}
            wrong = {i for i, ch in enumerate(word) if ch in wrong_letters}
        for i, ch in enumerate(word):
            if ch in SILENT or not sounds[i]:
                continue
            yield {"word": word, "letter": ch, "sound": sounds[i], "gop": None, "ok": i not in wrong,
                   "heard": "", "source": "text"}
