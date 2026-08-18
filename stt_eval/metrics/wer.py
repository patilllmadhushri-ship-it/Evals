"""Deterministic metrics: word and character error rate.

These are always computed and need no API beyond the STT provider itself. Both
are reported per clip as a rate, and pooled across the dataset by summing errors
and reference lengths — not by averaging per-clip rates, which would let a
three-word clip outweigh a three-minute one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .align import Alignment, align, tokenize_chars, tokenize_words


@dataclass(frozen=True)
class DeterministicScores:
    wer: float
    cer: float
    word_errors: int
    word_length: int
    char_errors: int
    char_length: int
    word_alignment: Alignment
    char_alignment: Alignment

    @property
    def substitutions(self) -> int:
        return self.word_alignment.substitutions

    @property
    def deletions(self) -> int:
        return self.word_alignment.deletions

    @property
    def insertions(self) -> int:
        return self.word_alignment.insertions


def score_deterministic(ground_truth: str, prediction: str) -> DeterministicScores:
    word_alignment = align(tokenize_words(ground_truth), tokenize_words(prediction))
    char_alignment = align(tokenize_chars(ground_truth), tokenize_chars(prediction))
    return DeterministicScores(
        wer=word_alignment.error_rate,
        cer=char_alignment.error_rate,
        word_errors=word_alignment.errors,
        word_length=word_alignment.ref_length,
        char_errors=char_alignment.errors,
        char_length=char_alignment.ref_length,
        word_alignment=word_alignment,
        char_alignment=char_alignment,
    )


def pooled_rate(total_errors: float, total_length: float) -> float | None:
    """Dataset-level rate: summed errors over summed reference length."""
    if total_length <= 0:
        return None
    return total_errors / total_length
