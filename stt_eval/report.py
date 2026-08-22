"""Aggregation: per-clip rows and the per-provider leaderboard.

Rate metrics are pooled across the dataset (summed errors over summed reference
length) rather than averaged per clip. Score metrics (intent/entity, semantic
match) are means over the clips that produced a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .costs import percentiles, rate_for
from .metrics import METRICS_BY_KEY
from .store import StoredResult


@dataclass
class ProviderSummary:
    provider: str
    clips_scored: int = 0
    clips_failed: int = 0
    audio_minutes: float = 0.0
    metrics: dict[str, float | None] = field(default_factory=dict)
    metric_errors: dict[str, int] = field(default_factory=dict)
    latency: dict[str, float | None] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    #: Real-time factor: processing time over audio duration, pooled. Below 1.0
    #: means the provider consumes speech faster than it arrives.
    rtf: float | None = None

    @property
    def rate_usd_per_minute(self) -> float:
        return rate_for(self.provider)


def summarize(
    results: list[StoredResult], *, enabled_metrics: list[str]
) -> dict[str, ProviderSummary]:
    summaries: dict[str, ProviderSummary] = {}

    for result in results:
        summary = summaries.setdefault(result.provider, ProviderSummary(provider=result.provider))
        if not result.ok:
            summary.clips_failed += 1
            continue
        summary.clips_scored += 1
        summary.audio_minutes += (result.duration_seconds or 0.0) / 60.0

    pooled: dict[tuple[str, str], list[float]] = {}
    means: dict[tuple[str, str], list[float]] = {}
    latencies: dict[str, list[float]] = {}

    for result in results:
        if not result.ok:
            continue
        if result.latency_seconds is not None:
            latencies.setdefault(result.provider, []).append(result.latency_seconds)

        for key in enabled_metrics:
            spec = METRICS_BY_KEY.get(key)
            if spec is None:
                continue
            entry = result.metrics.get(key) or {}
            if entry.get("error"):
                summaries[result.provider].metric_errors[key] = (
                    summaries[result.provider].metric_errors.get(key, 0) + 1
                )
                continue
            if entry.get("value") is None:
                continue

            if spec.aggregation == "pooled" and entry.get("length") is not None:
                bucket = pooled.setdefault((result.provider, key), [0.0, 0.0])
                bucket[0] += float(entry.get("errors") or 0.0)
                bucket[1] += float(entry.get("length") or 0.0)
            else:
                means.setdefault((result.provider, key), []).append(float(entry["value"]))

    for (provider, key), (errors, length) in pooled.items():
        summaries[provider].metrics[key] = (errors / length) if length > 0 else None
    for (provider, key), values in means.items():
        summaries[provider].metrics[key] = sum(values) / len(values) if values else None

    for provider, summary in summaries.items():
        summary.latency = percentiles(latencies.get(provider, []))
        summary.estimated_cost_usd = summary.audio_minutes * summary.rate_usd_per_minute
        summary.rtf = _pooled_rtf(results, provider)

    return summaries


def _pooled_rtf(results: list[StoredResult], provider: str) -> float | None:
    """Processing time over audio duration, pooled across the provider's clips.

    Pooled rather than averaged per clip for the same reason as WER: a
    half-second clip should not weigh as much as a two-minute one. Below 1.0
    means the provider is faster than real time.
    """
    total_latency = 0.0
    total_audio = 0.0
    for result in results:
        if result.provider != provider or not result.ok:
            continue
        if result.latency_seconds is None or not result.duration_seconds:
            continue
        total_latency += result.latency_seconds
        total_audio += result.duration_seconds
    return (total_latency / total_audio) if total_audio > 0 else None


def rank(summaries: dict[str, ProviderSummary], *, metric_key: str) -> list[str]:
    """Providers ordered best-first on one metric; providers with no value last."""
    spec = METRICS_BY_KEY.get(metric_key)
    lower_is_better = spec.lower_is_better if spec else True

    def sort_key(provider: str):
        value = summaries[provider].metrics.get(metric_key)
        if value is None:
            return (1, 0.0)
        return (0, value if lower_is_better else -value)

    return sorted(summaries, key=sort_key)


def winners(summaries: dict[str, ProviderSummary], *, metric_keys: list[str]) -> dict[str, str | None]:
    """The best provider per metric, for highlighting in the leaderboard."""
    result: dict[str, str | None] = {}
    for key in metric_keys:
        ordered = rank(summaries, metric_key=key)
        best = next(
            (provider for provider in ordered if summaries[provider].metrics.get(key) is not None),
            None,
        )
        result[key] = best
    return result


def per_clip_rows(
    results: list[StoredResult], *, enabled_metrics: list[str]
) -> list[dict]:
    """One row per (clip, provider), ready for a table or an export file."""
    rows: list[dict] = []
    for result in results:
        row: dict = {
            "id": result.clip_id,
            "provider": result.provider,
            "language": result.language,
            "status": result.status,
            "ground_truth": result.ground_truth,
            "prediction": result.prediction,
            "audio_duration_seconds": round(result.duration_seconds or 0.0, 3),
            "latency_seconds": (
                round(result.latency_seconds, 3) if result.latency_seconds is not None else None
            ),
            "rtf": (
                round(result.latency_seconds / result.duration_seconds, 3)
                if result.latency_seconds and result.duration_seconds
                else None
            ),
            "error": result.error,
        }
        for key in enabled_metrics:
            spec = METRICS_BY_KEY.get(key)
            if spec is None:
                continue
            entry = result.metrics.get(key) or {}
            value = entry.get("value")
            row[key] = round(value, 4) if isinstance(value, (int, float)) else None
            if spec.has_reasoning:
                row[f"{key}_reasoning"] = entry.get("reasoning")
            if entry.get("error"):
                row[f"{key}_error"] = entry.get("error")
        rows.append(row)
    return rows


def summary_rows(
    summaries: dict[str, ProviderSummary], *, enabled_metrics: list[str]
) -> list[dict]:
    rows: list[dict] = []
    for provider, summary in summaries.items():
        row: dict = {
            "provider": provider,
            "clips_scored": summary.clips_scored,
            "clips_failed": summary.clips_failed,
            "audio_minutes": round(summary.audio_minutes, 3),
            "estimated_cost_usd": round(summary.estimated_cost_usd, 4),
            "rate_usd_per_minute": summary.rate_usd_per_minute,
        }
        for key in enabled_metrics:
            value = summary.metrics.get(key)
            row[key] = round(value, 4) if isinstance(value, (int, float)) else None
        row["rtf"] = round(summary.rtf, 3) if summary.rtf is not None else None
        for label in ("p50", "p95", "p99"):
            value = summary.latency.get(label)
            row[f"latency_{label}_seconds"] = round(value, 3) if value is not None else None
        rows.append(row)
    return rows
