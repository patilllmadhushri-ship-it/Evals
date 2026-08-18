"""Shared metric value type, kept separate so metric implementations can import
it without a circular dependency on the registry in ``metrics/__init__``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricValue:
    """One metric's outcome for one (ground truth, prediction) pair."""

    key: str
    value: float | None = None
    reasoning: str | None = None
    error: str | None = None
    #: Numerator/denominator for pooled metrics, so the dataset-level figure is
    #: a pooled rate rather than an average of per-clip rates.
    errors: float | None = None
    length: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None
