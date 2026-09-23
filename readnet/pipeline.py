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
from .languages import Trace
from .score import ScoreResult, score


def heard_from_audio(word: str, units, threshold: float) -> str | None:
    """`word` as the audio has it: each unit GOP rejects is replaced by what
    the audio holds there ("" if nothing). None when nothing was rejected or
    the units are not letters of `word` (a phoneme model's IPA symbols)."""
    chars = list(word)
    pos, changed = 0, False
    for u in units:
        while pos < len(chars) and chars[pos] != u.unit:
            pos += 1
        if pos == len(chars):
            return None
        if u.llr < threshold:
            chars[pos] = u.heard
            changed = True
        pos += 1
    return "".join(chars) if changed else None


def _rehear(result: ScoreResult, acoustic: AcousticEvidence, threshold: float, profile) -> None:
    for i, op in enumerate(result.ops):
        if op.kind != "match" or op.ref_index is None or op.ref_index >= len(acoustic.words):
            continue
        heard = heard_from_audio(op.ref, acoustic.words[op.ref_index].units, threshold)
        if heard is None:
            continue
        # The rebuilt word is text like any transcript, so it goes through the
        # same normalisation: a model that splits a nasal between ं and ँ has
        # still heard the one nasal sound (HI-06).
        heard = profile.normalize(heard)[0].replace(" ", "")
        if heard == op.ref:
            continue
        redo = score([op.ref], [heard] if heard else [], profile).ops
        new = next(o for o in redo if o.ref_index is not None)
        new.ref_index = op.ref_index
        new.note = (f"the audio check heard {heard or 'nothing'}; the transcript said {op.hyp}"
                    + (f" · {new.note}" if new.note else ""))
        result.ops[i] = new


def is_letter_text(normalized: str) -> bool:
    """Single letters (क ख ग), whatever task type they were entered under."""
    tokens = normalized.split()
    return bool(tokens) and all(len(t) == 1 for t in tokens)


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
            "canonical_changes": self.canonical_trace.steps,
            "romanised_words": self.transcript_trace.romanised_words,
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
    audio_decides: bool = False,
) -> ItemAssessment:
    """Score one item.

    `audio_decides`: where the transcript says a word was read right but GOP
    says some of its sounds were not in the audio, rebuild the word from what
    the audio holds (the missing sound dropped, a different sound swapped in)
    and put *that* through the same scorer and rulebook. Speech engines
    write the dictionary word — चाद comes back as चाँद — so without this a
    dropped sound is invisible. The rulebook still decides: चाँद -> चाद counts
    (HI-21), गाँव -> गाव stays forgiven (HI-20).

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
    letter_task = level == Level.LETTER or is_letter_text(canon_norm)
    result = score(
        canon_norm.split(),
        hyp_norm.split(),
        profile,
        letter_task=letter_task,
        count_insertions=rules.count_insertions,
        romanised=hyp_trace.romanised_words,
    )

    if audio_decides and acoustic is not None and gop_threshold is not None and not letter_task:
        _rehear(result, acoustic, gop_threshold, profile)

    doubts: list[str] = []
    if acoustic is not None and gop_threshold is not None:
        for op in result.ops:
            if op.ref_index is None or op.counts_as_mistake or op.ref_index >= len(acoustic.words):
                continue
            if op.note.startswith("the audio check"):
                continue  # already re-scored from the audio
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
    audio_decides: bool = False,
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
            audio_decides=audio_decides,
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
