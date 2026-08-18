"""Cost estimation from the bundled rate table.

The same calculation runs before the run (on total uploaded duration) and after
it (on the audio actually processed), so the two figures are comparable by
construction. Both are estimates — see `config.RATE_TABLE_NOTE`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PROVIDER_RATES_USD_PER_MINUTE


@dataclass(frozen=True)
class CostEstimate:
    provider: str
    minutes: float
    rate_usd_per_minute: float

    @property
    def usd(self) -> float:
        return self.minutes * self.rate_usd_per_minute


def rate_for(provider: str) -> float:
    return PROVIDER_RATES_USD_PER_MINUTE.get(provider, 0.0)


def estimate(provider: str, minutes: float) -> CostEstimate:
    return CostEstimate(provider=provider, minutes=minutes, rate_usd_per_minute=rate_for(provider))


def estimate_run(providers: list[str], total_minutes: float) -> dict[str, CostEstimate]:
    return {provider: estimate(provider, total_minutes) for provider in providers}


def total_usd(estimates: dict[str, CostEstimate]) -> float:
    return sum(item.usd for item in estimates.values())


def percentiles(values: list[float]) -> dict[str, float | None]:
    """p50/p95/p99 of measured latencies, using nearest-rank."""
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
        return ordered[rank - 1]

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}
