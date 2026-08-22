"""Stratified sampling for the expensive metrics.

Deterministic metrics are free once the transcript exists, so they run on
everything. LLM metrics cost money and time per clip, and on a large dataset
most of that spend is confirming what WER already told you: clips that matched
exactly are equivalent, clips that diverged wildly are not.

Sampling a stratified slice instead — a share drawn from each confidence band —
keeps the interesting middle of the distribution well covered while cutting the
judge bill. Stratifying (rather than sampling uniformly at random) is what stops
a 10% sample from consisting entirely of easy clips.

The confidence signal is the deterministic error rate: a pair with WER 0 is a
high-confidence agreement, a pair with WER 0.8 is a high-confidence failure, and
the band between them is where meaning-aware judgment actually changes the
verdict.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

#: Bands of deterministic error rate, from "certainly fine" to "certainly bad".
#: Upper bound is exclusive except for the last band.
CONFIDENCE_BANDS: list[tuple[str, float, float]] = [
    ("exact match", 0.0, 0.0001),
    ("near match", 0.0001, 0.10),
    ("diverging", 0.10, 0.30),
    ("poor", 0.30, 1.01),
]


def band_for(error_rate: float | None) -> str:
    """Which confidence band a deterministic error rate falls in."""
    if error_rate is None:
        return "unknown"
    for name, low, high in CONFIDENCE_BANDS:
        if low <= error_rate < high:
            return name
    return CONFIDENCE_BANDS[-1][0]


@dataclass(frozen=True)
class SamplePlan:
    """Which pairs get the expensive metrics, and why."""

    selected: set[tuple[str, str]]
    per_band: dict[str, tuple[int, int]]  # band -> (sampled, total)
    rate: float

    @property
    def total_selected(self) -> int:
        return len(self.selected)

    @property
    def total_candidates(self) -> int:
        return sum(total for _, total in self.per_band.values())

    def describe(self) -> str:
        parts = [
            f"{name}: {sampled}/{total}"
            for name, (sampled, total) in self.per_band.items()
            if total
        ]
        return " · ".join(parts)


def plan(
    candidates: list[tuple[tuple[str, str], float | None]],
    *,
    rate: float,
    seed: int = 0,
    minimum_per_band: int = 1,
) -> SamplePlan:
    """Choose which (clip, provider) pairs to judge.

    `candidates` pairs a (clip_id, provider) key with its deterministic error
    rate. `rate` of 1.0 selects everything, which is the default — sampling is
    an opt-in cost control, not something that quietly happens to a small run.

    Every non-empty band contributes at least `minimum_per_band` pairs, so no
    band vanishes from the results at low sampling rates. Selection is seeded,
    so re-running the same dataset judges the same pairs.
    """
    if rate >= 1.0:
        return SamplePlan(
            selected={key for key, _ in candidates},
            per_band=_band_counts(candidates, {key for key, _ in candidates}),
            rate=1.0,
        )

    grouped: dict[str, list[tuple[str, str]]] = {}
    for key, error_rate in candidates:
        grouped.setdefault(band_for(error_rate), []).append(key)

    rng = random.Random(seed)
    selected: set[tuple[str, str]] = set()
    for band, keys in grouped.items():
        target = max(minimum_per_band, round(len(keys) * rate)) if keys else 0
        target = min(target, len(keys))
        ordered = sorted(keys)  # deterministic before shuffling
        rng.shuffle(ordered)
        selected.update(ordered[:target])

    return SamplePlan(
        selected=selected, per_band=_band_counts(candidates, selected), rate=rate
    )


def _band_counts(
    candidates: list[tuple[tuple[str, str], float | None]], selected: set[tuple[str, str]]
) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for key, error_rate in candidates:
        band = band_for(error_rate)
        sampled, total = counts.get(band, (0, 0))
        counts[band] = (sampled + (1 if key in selected else 0), total + 1)
    return {name: counts.get(name, (0, 0)) for name, _, _ in CONFIDENCE_BANDS}
