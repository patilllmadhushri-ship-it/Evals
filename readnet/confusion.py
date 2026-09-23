"""Letter confusions estimated from data, not written by hand.

The visual/phonetic split in each language's ``ScriptTables`` is two hand-made
lists. This module estimates the same thing from real pairs: which letter was
expected and which one was heard, counted over word-level substitutions. The
result has two uses:

* **Checking the hand-made tables.** Frequent confusions missing from them, and
  listed pairs that never occur, are both worth a conversation with a linguist.
* **Candidates for a closed-set decision.** When the screen shows घ, the
  question should be "does the audio match घ or one of its likely
  confusions?" — see ``acoustic.closed_set_decision``. This module supplies
  "likely confusions".

Feed it (expected, heard) pairs. Use human transcripts of children's readings
to learn how *children* misread; use (human transcript, model output) to learn
how the *model* mishears. They are different matrices, so keep them apart.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from . import languages
from .languages import ScriptTables
from .score import align, classify_substitution, word_sub_cost


@dataclass
class ConfusionMatrix:
    #: (expected char, heard char) -> count; "" on one side is a drop/insert.
    counts: Counter = field(default_factory=Counter)
    #: expected char -> times it appeared in an aligned expected word.
    seen: Counter = field(default_factory=Counter)

    def rate(self, expected: str, heard: str) -> float:
        return self.counts[(expected, heard)] / self.seen[expected] if self.seen[expected] else 0.0

    def top(self, n: int = 20) -> list[tuple[str, str, int, float]]:
        return [(e, h, c, self.rate(e, h)) for (e, h), c in self.counts.most_common(n)]

    def confusions_of(self, letter: str, *, min_count: int = 2, k: int = 4) -> list[str]:
        ranked = sorted(
            ((h, c) for (e, h), c in self.counts.items() if e == letter and h and c >= min_count),
            key=lambda item: -item[1],
        )
        return [h for h, _ in ranked[:k]]


def estimate(pairs: Iterable[tuple[str, str]], language: str) -> ConfusionMatrix:
    profile = languages.get(language)
    matrix = ConfusionMatrix()
    for expected, heard in pairs:
        ref = profile.normalize_neutral(expected)[0].split()
        hyp = profile.normalize_neutral(heard)[0].split()
        for kind, i, j in align(ref, hyp, word_sub_cost):
            if i is None:
                continue
            matrix.seen.update(ref[i])
            if kind == "substitution":
                c = classify_substitution(ref[i], hyp[j], profile.tables)
                if c.category != "different_word":  # unrelated words teach nothing about letters
                    matrix.counts.update(c.char_diffs)
    return matrix


def hand_made_confusions(letter: str, tables: ScriptTables) -> list[str]:
    """The partners of `letter` in the hand-made phonetic and visual tables."""
    partners: list[str] = []
    for table in (*tables.phonetic.values(), tables.visual):
        for pair in table:
            if letter in pair:
                partners.extend(ch for ch in pair if ch != letter and ch not in partners)
    return partners


def audit(matrix: ConfusionMatrix, tables: ScriptTables, *, min_count: int = 3) -> dict[str, list]:
    """Where the data and the hand-made tables disagree."""
    listed = set().union(*tables.phonetic.values(), tables.visual)
    frequent = {frozenset((e, h)) for (e, h), c in matrix.counts.items() if e and h and c >= min_count}
    return {
        "frequent_but_unlisted": sorted("/".join(sorted(p)) for p in frequent - listed),
        "listed_but_never_seen": sorted(
            "/".join(sorted(p)) for p in listed if not any(frozenset(k) == p for k in matrix.counts)
        ),
    }
