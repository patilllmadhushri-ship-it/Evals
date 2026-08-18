"""Normalisation and edit-distance alignment shared by every metric.

The alignment is the substrate for the whole ladder: plain WER/CER read the
operation counts, and the LLM-WER / LLM-CER metrics read the differing segments
so a judge can decide which of them actually changed the meaning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Sequence

Op = Literal["equal", "sub", "del", "ins"]

_PUNCTUATION = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Case-fold, strip punctuation, collapse whitespace, NFKC-normalise."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def tokenize_chars(text: str) -> list[str]:
    """Characters of the normalised text, with spaces removed.

    Dropping spaces is what makes CER forgiving of segmentation differences,
    which matters for languages with ambiguous word boundaries.
    """
    return [char for char in normalize_text(text) if not char.isspace()]


@dataclass(frozen=True)
class Operation:
    op: Op
    ref: list[str]
    hyp: list[str]

    @property
    def ref_text(self) -> str:
        return " ".join(self.ref)

    @property
    def hyp_text(self) -> str:
        return " ".join(self.hyp)


@dataclass(frozen=True)
class Alignment:
    operations: list[Operation]
    ref_length: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        if self.ref_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_length


def _edit_matrix(ref: Sequence[str], hyp: Sequence[str]) -> list[list[int]]:
    rows, cols = len(ref) + 1, len(hyp) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j
    for i in range(1, rows):
        ref_token = ref[i - 1]
        row, previous_row = matrix[i], matrix[i - 1]
        for j in range(1, cols):
            if ref_token == hyp[j - 1]:
                row[j] = previous_row[j - 1]
            else:
                row[j] = 1 + min(previous_row[j - 1], previous_row[j], row[j - 1])
    return matrix


def align(ref: Sequence[str], hyp: Sequence[str]) -> Alignment:
    """Levenshtein-align two token sequences and group the ops into runs."""
    matrix = _edit_matrix(ref, hyp)
    i, j = len(ref), len(hyp)
    raw: list[tuple[Op, str | None, str | None]] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and matrix[i][j] == matrix[i - 1][j - 1]:
            raw.append(("equal", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and matrix[i][j] == matrix[i - 1][j - 1] + 1:
            raw.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and matrix[i][j] == matrix[i - 1][j] + 1:
            raw.append(("del", ref[i - 1], None))
            i -= 1
        else:
            raw.append(("ins", None, hyp[j - 1]))
            j -= 1

    raw.reverse()

    substitutions = sum(1 for op, _, _ in raw if op == "sub")
    deletions = sum(1 for op, _, _ in raw if op == "del")
    insertions = sum(1 for op, _, _ in raw if op == "ins")

    # Collapse adjacent non-equal ops into a single differing segment, so an LLM
    # judge sees "five hundred rupees" vs "500" as one decision, not three.
    grouped: list[Operation] = []
    for op, ref_token, hyp_token in raw:
        kind: Op = "equal" if op == "equal" else "sub"
        if grouped and grouped[-1].op == kind:
            last = grouped[-1]
            if ref_token is not None:
                last.ref.append(ref_token)
            if hyp_token is not None:
                last.hyp.append(hyp_token)
            continue
        grouped.append(
            Operation(
                op=kind,
                ref=[ref_token] if ref_token is not None else [],
                hyp=[hyp_token] if hyp_token is not None else [],
            )
        )

    return Alignment(
        operations=grouped,
        ref_length=len(ref),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def differing_segments(alignment: Alignment) -> list[Operation]:
    return [operation for operation in alignment.operations if operation.op != "equal"]
