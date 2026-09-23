"""How good is the ASR model? The neutral benchmark, split by sample type.

This is the other normaliser's job. It answers a different question from
scoring children, and must not borrow its leniency:

* **Neutral normaliser only.** It removes differences that are not in the
  speech at all (bytes, script, invisible marks, spelling conventions) and
  nothing else. It does not forgive dropped nasals, श/ष, letter-name spellings,
  repeats or fillers. Those are pedagogical choices, and applying them here
  flatters the model.
* **Split by level, every time.** Letters, words, paragraphs and stories are
  reported separately. A single pooled figure is dominated by the long
  paragraph and story samples and hides weak letter and word recognition.

Input rows: ``level``, ``reference`` (what a human heard the child say — not
the text on screen), ``hypothesis`` (the model's transcript). The forgiving
figures are reported alongside only to show how much a forgiving benchmark
would have overstated accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from . import languages
from .aser import Level
from .score import edit_count, score


@dataclass
class LevelBenchmark:
    level: str
    samples: int
    wer: float
    cer: float
    #: Share of samples transcribed exactly (after neutral normalisation).
    exact: float
    #: Words "wrong" after child-scoring forgiveness — what you must NOT report
    #: as model accuracy. The gap to `wer` is how much a forgiving benchmark flatters.
    forgiving_wer: float


def _rates(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    word_edits = word_total = char_edits = char_total = exact = 0
    for ref, hyp in pairs:
        r_words, h_words = ref.split(), hyp.split()
        word_edits += edit_count(r_words, h_words)
        word_total += len(r_words)
        r_chars, h_chars = ref.replace(" ", ""), hyp.replace(" ", "")
        char_edits += edit_count(r_chars, h_chars)
        char_total += len(r_chars)
        exact += ref == hyp
    return (
        word_edits / max(word_total, 1),
        char_edits / max(char_total, 1),
        exact / max(len(pairs), 1),
    )


def benchmark(rows: Iterable[dict], language: str) -> list[LevelBenchmark]:
    """Pooled WER/CER per level (summed edits over summed reference length)."""
    profile = languages.get(language)
    neutral: dict[str, list] = defaultdict(list)
    forgiven_errors: dict[str, int] = defaultdict(int)
    for row in rows:
        parsed = Level.parse(row["level"])
        level = parsed.label
        neutral[level].append(
            (profile.normalize_neutral(row["reference"])[0], profile.normalize_neutral(row["hypothesis"])[0])
        )
        ref, ref_trace = profile.normalize(row["reference"])
        hyp, hyp_trace = profile.normalize(row["hypothesis"])
        forgiven_errors[level] += score(
            ref.split(), hyp.split(), profile, letter_task=parsed == Level.LETTER, romanised=hyp_trace.romanised_words
        ).mistakes

    order = [lvl.label for lvl in Level if lvl.label in neutral]
    out: list[LevelBenchmark] = []
    for level in order:
        wer, cer, exact = _rates(neutral[level])
        words = sum(len(ref.split()) for ref, _ in neutral[level])
        out.append(LevelBenchmark(level, len(neutral[level]), wer, cer, exact, forgiven_errors[level] / max(words, 1)))
    return out


def format_table(results: list[LevelBenchmark]) -> str:
    lines = [f"{'level':<10} {'n':>5} {'WER':>7} {'CER':>7} {'exact':>7}   (forgiving WER, do not report)"]
    for r in results:
        lines.append(
            f"{r.level:<10} {r.samples:>5} {r.wer:>7.1%} {r.cer:>7.1%} {r.exact:>7.1%}   ({r.forgiving_wer:.1%})"
        )
    missing = [lvl.label for lvl in Level if lvl != Level.BEGINNER and lvl.label not in {r.level for r in results}]
    if missing:
        lines.append(f"no samples for: {', '.join(missing)} — this benchmark says nothing about them")
    return "\n".join(lines)
