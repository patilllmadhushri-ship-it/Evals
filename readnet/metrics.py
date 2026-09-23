"""How well the system agrees with expert human assessors.

* **Accuracy** and **quadratic weighted kappa** on the ASER level — kappa
  because levels are ordered: calling a Story reader "Paragraph" is a smaller
  miss than calling them "Beginner".
* **False fail rate** — the system placed the child *below* the human verdict.
  The operational metric: every false fail is a child sent back to work they
  can already do, and a teacher who stops trusting the app.
* **Precision and recall on mistakes**, word by word, against human marking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .aser import Level


def quadratic_weighted_kappa(human: Sequence[int], system: Sequence[int], n_levels: int = len(Level)) -> float:
    human = np.asarray(human, dtype=int)
    system = np.asarray(system, dtype=int)
    observed = np.zeros((n_levels, n_levels))
    for h, s in zip(human, system):
        observed[h, s] += 1
    grid = np.arange(n_levels)
    weights = (grid[:, None] - grid[None, :]) ** 2 / (n_levels - 1) ** 2
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(observed.sum(), 1)
    denominator = (weights * expected).sum()
    return 1.0 if denominator == 0 else float(1 - (weights * observed).sum() / denominator)


@dataclass
class LevelAgreement:
    n: int
    accuracy: float
    weighted_kappa: float
    false_fail_rate: float
    false_pass_rate: float
    confusion: list[list[int]]  # rows human, columns system

    def as_dict(self) -> dict:
        return self.__dict__


def level_agreement(pairs: Iterable[tuple[Level | str | int, Level | str | int]]) -> LevelAgreement:
    """`pairs` of (human level, system level)."""
    human, system = [], []
    for h, s in pairs:
        human.append(int(Level.parse(h)))
        system.append(int(Level.parse(s)))
    n = len(human)
    if n == 0:
        raise ValueError("No pairs to evaluate")
    confusion = [[0] * len(Level) for _ in Level]
    for h, s in zip(human, system):
        confusion[h][s] += 1
    return LevelAgreement(
        n=n,
        accuracy=sum(h == s for h, s in zip(human, system)) / n,
        weighted_kappa=quadratic_weighted_kappa(human, system),
        false_fail_rate=sum(s < h for h, s in zip(human, system)) / n,
        false_pass_rate=sum(s > h for h, s in zip(human, system)) / n,
        confusion=confusion,
    )


def mistake_precision_recall(items: Iterable[tuple[set[int], set[int]]]) -> dict[str, float]:
    """`items` of (human-marked mistake word indices, system-marked ones)."""
    tp = fp = fn = 0
    for human, system in items:
        tp += len(human & system)
        fp += len(system - human)
        fn += len(human - system)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
