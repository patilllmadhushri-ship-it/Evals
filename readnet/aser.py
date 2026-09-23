"""ASER levels and the adaptive order the test is given in.

The ladder and the pass rules are Pratham's, from the ASER reading tool — the
app automates the judging, it does not invent a new test:

* The child starts at the **paragraph** (Grade 1 text).
  - Passes (≤ 3 mistakes) → given the **story** (Grade 2 text).
    Passes (≤ 3 mistakes) → Story level; fails → Paragraph level.
  - Fails → given **words**. Reads 4 of 5 → Word level.
    Fails → given **letters**. Reads 4 of 5 → Letter level; else Beginner.

The thresholds live in `Rules` so a programme can change them without code.
Fluency (words per minute, long pauses) is measured whenever word timings are
available but only *enforced* when a programme sets a threshold, because ASER's
fluency judgement ("reads like a sentence, not word by word") has no official
number and an invented one would create false fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Level(IntEnum):
    BEGINNER = 0
    LETTER = 1
    WORD = 2
    PARAGRAPH = 3
    STORY = 4

    @classmethod
    def parse(cls, value: "str | int | Level") -> "Level":
        if isinstance(value, Level):
            return value
        if isinstance(value, int) or str(value).isdigit():
            return cls(int(value))
        return cls[str(value).strip().upper()]

    @property
    def label(self) -> str:
        return self.name.title()


#: The tasks a child can be given (Beginner is an outcome, not a task).
TASKS = (Level.LETTER, Level.WORD, Level.PARAGRAPH, Level.STORY)
START = Level.PARAGRAPH


@dataclass(frozen=True)
class Rules:
    items_required: dict[Level, int] = field(
        default_factory=lambda: {Level.LETTER: 4, Level.WORD: 4}
    )
    max_mistakes: dict[Level, int] = field(
        default_factory=lambda: {Level.PARAGRAPH: 3, Level.STORY: 3}
    )
    #: Optional fluency gates; None means measured and reported, not enforced.
    min_wpm: dict[Level, float] = field(default_factory=dict)
    max_pause_seconds: float | None = None
    #: Count extra (non-repeat, non-filler) words as mistakes. See score.py.
    count_insertions: bool = False


@dataclass
class TaskOutcome:
    level: Level
    passed: bool
    mistakes: int
    correct: int
    total: int
    wpm: float | None = None
    longest_pause: float | None = None
    reasons: list[str] = field(default_factory=list)


def judge_task(
    level: Level,
    *,
    mistakes: int,
    correct: int,
    total: int,
    rules: Rules,
    wpm: float | None = None,
    longest_pause: float | None = None,
) -> TaskOutcome:
    reasons: list[str] = []
    if level in rules.items_required:
        need = rules.items_required[level]
        passed = correct >= need
        reasons.append(f"read {correct} of {total} {level.label.lower()}s correctly (need {need})")
    else:
        limit = rules.max_mistakes[level]
        passed = mistakes <= limit
        reasons.append(f"{mistakes} mistake(s) (at most {limit} allowed)")

    if level in rules.min_wpm and wpm is not None and wpm < rules.min_wpm[level]:
        passed = False
        reasons.append(f"{wpm:.0f} words/min, below the {rules.min_wpm[level]:.0f} required")
    if (
        rules.max_pause_seconds is not None
        and longest_pause is not None
        and longest_pause > rules.max_pause_seconds
    ):
        passed = False
        reasons.append(f"paused {longest_pause:.1f}s (limit {rules.max_pause_seconds:.1f}s)")
    return TaskOutcome(level, passed, mistakes, correct, total, wpm, longest_pause, reasons)


def next_task(outcomes: dict[Level, TaskOutcome]) -> Level | None:
    """The task to give next, or None when the child can be placed."""
    if Level.PARAGRAPH not in outcomes:
        return Level.PARAGRAPH
    if outcomes[Level.PARAGRAPH].passed:
        return None if Level.STORY in outcomes else Level.STORY
    if Level.WORD not in outcomes:
        return Level.WORD
    if outcomes[Level.WORD].passed:
        return None
    return None if Level.LETTER in outcomes else Level.LETTER


@dataclass
class Placement:
    level: Level
    path: list[str]


def place(outcomes: dict[Level, TaskOutcome]) -> Placement:
    """Walk the ASER order over the tasks given and return the child's level."""
    path: list[str] = []
    level: Level | None = None
    current: Level | None = START
    while current is not None:
        if current not in outcomes:
            raise ValueError(f"Cannot place yet: the {current.label} task has not been given")
        outcome = outcomes[current]
        path.append(f"{current.label}: {'pass' if outcome.passed else 'fail'} — {'; '.join(outcome.reasons)}")
        if current == Level.PARAGRAPH:
            current = Level.STORY if outcome.passed else Level.WORD
        elif current == Level.STORY:
            level, current = (Level.STORY if outcome.passed else Level.PARAGRAPH), None
        elif current == Level.WORD:
            if outcome.passed:
                level, current = Level.WORD, None
            else:
                current = Level.LETTER
        else:  # LETTER
            level, current = (Level.LETTER if outcome.passed else Level.BEGINNER), None
    return Placement(level, path)
