"""The three blueprint stages joined up: normalise → align and classify → judge.

    assess_item   one screen of text + what the child said -> scored item
    assess_task   the items for one ASER level -> pass/fail for that level
    aser.place    the tasks given -> the child's level

The ASR transcript is an input, not something this module produces, so any
engine — the providers in stt_eval, or an on-device model — can feed it.
Acoustic evidence (forced alignment + GOP) is optional and additive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import languages
from .acoustic import AcousticEvidence
from .aser import Level, Rules, TaskOutcome, judge_task
from .languages.devanagari import Trace
from .score import ScoreResult, score


@dataclass
class ItemAssessment:
    canonical: str
    transcript: str
    canonical_normalized: str
    transcript_normalized: str
    result: ScoreResult
    canonical_trace: Trace
    transcript_trace: Trace
    #: Words the transcript says were read right but the audio disagrees with.
    acoustic_doubts: list[str] = field(default_factory=list)
    acoustic: AcousticEvidence | None = None

    def as_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "transcript": self.transcript,
            "canonical_normalized": self.canonical_normalized,
            "transcript_normalized": self.transcript_normalized,
            "transcript_changes": self.transcript_trace.steps,
            "transliterated": self.transcript_trace.transliterated,
            "acoustic_doubts": self.acoustic_doubts,
            **self.result.as_dict(),
        }


def assess_item(
    canonical: str,
    transcript: str,
    *,
    language: str = "hi",
    level: Level | str = Level.PARAGRAPH,
    rules: Rules | None = None,
    acoustic: AcousticEvidence | None = None,
    gop_threshold: float | None = None,
    count_acoustic_doubts: bool = False,
) -> ItemAssessment:
    """Score one item.

    `gop_threshold` is an LLR cut-off (see acoustic.py). A word the transcript
    marks correct but whose LLR falls below it is listed in `acoustic_doubts`;
    it becomes a mistake only with `count_acoustic_doubts=True`, which should
    wait until the threshold is calibrated against human-labelled audio.
    """
    rules = rules or Rules()
    level = Level.parse(level)
    profile = languages.get(language)
    canon_norm, canon_trace = profile.normalize(canonical)
    hyp_norm, hyp_trace = profile.normalize(transcript)
    result = score(
        canon_norm.split(),
        hyp_norm.split(),
        profile,
        letter_task=level == Level.LETTER,
        count_insertions=rules.count_insertions,
        romanised=hyp_trace.romanised_words,
    )

    doubts: list[str] = []
    if acoustic is not None and gop_threshold is not None:
        for op in result.ops:
            if op.ref_index is None or op.counts_as_mistake or op.ref_index >= len(acoustic.words):
                continue
            evidence = acoustic.words[op.ref_index]
            if evidence.llr is not None and evidence.llr < gop_threshold:
                doubts.append(op.ref)
                op.note = f"audio disagrees (GOP-LLR {evidence.llr:.2f} < {gop_threshold:.2f})"
                if count_acoustic_doubts:
                    op.counts_as_mistake = True
                    op.category = "acoustic"

    return ItemAssessment(
        canonical, transcript, canon_norm, hyp_norm, result, canon_trace, hyp_trace, doubts, acoustic
    )


def assess_task(
    level: Level | str,
    items: Sequence[tuple[str, str]],
    *,
    language: str = "hi",
    rules: Rules | None = None,
    acoustic: Sequence[AcousticEvidence | None] | None = None,
    gop_threshold: float | None = None,
    count_acoustic_doubts: bool = False,
) -> tuple[TaskOutcome, list[ItemAssessment]]:
    """Judge one ASER level from its items.

    Letter and word tasks are usually five separate recordings (one per item) or
    one recording of the whole list; either way the unit counted is a reference
    token read correctly. Paragraph and story tasks sum mistakes over the text.
    """
    rules = rules or Rules()
    level = Level.parse(level)
    acoustic = list(acoustic) if acoustic else [None] * len(items)
    assessed = [
        assess_item(
            canonical,
            transcript,
            language=language,
            level=level,
            rules=rules,
            acoustic=evidence,
            gop_threshold=gop_threshold,
            count_acoustic_doubts=count_acoustic_doubts,
        )
        for (canonical, transcript), evidence in zip(items, acoustic)
    ]
    mistakes = sum(a.result.mistakes for a in assessed)
    correct = sum(a.result.correct for a in assessed)
    total = sum(a.result.ref_len for a in assessed)

    wpm = longest_pause = None
    timed = [a.acoustic for a in assessed if a.acoustic is not None]
    if timed:
        rates = [e.words_per_minute() for e in timed if e.words_per_minute() is not None]
        wpm = sum(rates) / len(rates) if rates else None
        pauses = [p for e in timed for p in e.pauses()]
        longest_pause = max(pauses) if pauses else None

    outcome = judge_task(
        level, mistakes=mistakes, correct=correct, total=total, rules=rules, wpm=wpm, longest_pause=longest_pause
    )
    return outcome, assessed
