"""Signals: what the metrics together are telling you.

WER and the meaning-aware metrics fail in different directions, and the
interesting information is in their *disagreement*, not in either number alone.

* **WER high, meaning preserved.** The provider is being punished for wording —
  digits versus words, an expanded abbreviation, a dropped filler. Its raw WER
  understates how usable it actually is.
* **WER low, meaning broken.** The dangerous case. A one-word error that flips a
  negation, corrupts an amount or swaps a name reads as a near-perfect
  transcript by WER while being useless — or worse than useless — downstream.

A production monitor would alert on these. In a batch benchmark the equivalent
is surfacing them per run: threshold breaches, and every clip where the two
families of metric disagree, so the disagreement gets investigated rather than
averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .report import ProviderSummary
from .store import StoredResult

Severity = Literal["critical", "warning", "info"]

#: Meaning-aware metrics where a *high* value is good (a preserved meaning).
_MEANING_SCORE_KEYS = ("intent_entity", "semantic_match")
#: Meaning-aware error rates, comparable to WER, where low is good.
_MEANING_ERROR_KEYS = ("semantic_wer", "llm_wer")


@dataclass(frozen=True)
class Signal:
    severity: Severity
    title: str
    detail: str
    #: (clip_id, provider) pairs this signal points at, for drill-down.
    pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Thresholds:
    """What counts as bad. Defaults are deliberately loose — tighten per use case."""

    wer: float = 0.25
    semantic_error: float = 0.15
    meaning_score: float = 0.80


def meaning_verdict(result: StoredResult) -> tuple[bool | None, str]:
    """Did the meaning survive, according to whichever meaning metric ran?

    Returns (preserved, which metric decided). `None` when nothing meaning-aware
    was recorded for this pair — usually because it was not in the sample.
    """
    for key in _MEANING_SCORE_KEYS:
        value = result.metric_value(key)
        if value is not None:
            return value >= 0.5, key
    for key in _MEANING_ERROR_KEYS:
        value = result.metric_value(key)
        if value is not None:
            return value <= 0.30, key
    return None, ""


def divergences(
    results: list[StoredResult], *, thresholds: Thresholds
) -> tuple[list[StoredResult], list[StoredResult]]:
    """Split the disagreements into (meaning broken despite low WER, the reverse)."""
    meaning_broken: list[StoredResult] = []
    wording_only: list[StoredResult] = []

    for result in results:
        if not result.ok:
            continue
        wer = result.metric_value("wer")
        if wer is None:
            continue
        preserved, _ = meaning_verdict(result)
        if preserved is None:
            continue
        if wer <= thresholds.wer and not preserved:
            meaning_broken.append(result)
        elif wer > thresholds.wer and preserved:
            wording_only.append(result)

    meaning_broken.sort(key=lambda item: item.metric_value("wer") or 0.0)
    wording_only.sort(key=lambda item: item.metric_value("wer") or 0.0, reverse=True)
    return meaning_broken, wording_only


def evaluate(
    results: list[StoredResult],
    summaries: dict[str, ProviderSummary],
    *,
    thresholds: Thresholds | None = None,
) -> list[Signal]:
    """Everything worth flagging about this run, most severe first."""
    thresholds = thresholds or Thresholds()
    signals: list[Signal] = []

    meaning_broken, wording_only = divergences(results, thresholds=thresholds)

    if meaning_broken:
        signals.append(
            Signal(
                severity="critical",
                title=f"{len(meaning_broken)} clip(s) read as accurate but lost the meaning",
                detail=(
                    f"WER at or below {thresholds.wer:.0%} while the meaning-aware "
                    "metrics say the transcript does not mean the same thing. These "
                    "are the failures a WER-only benchmark misses: a flipped "
                    "negation, a corrupted amount, a swapped name. Read the judge's "
                    "reasoning on each before trusting the provider that produced it."
                ),
                pairs=tuple((item.clip_id, item.provider) for item in meaning_broken),
            )
        )

    if wording_only:
        signals.append(
            Signal(
                severity="info",
                title=f"{len(wording_only)} clip(s) scored badly on WER but kept the meaning",
                detail=(
                    f"WER above {thresholds.wer:.0%} yet the meaning survived — the "
                    "provider is being penalised for wording, not for errors. Its raw "
                    "WER understates how usable it is for anything that acts on "
                    "meaning rather than exact text."
                ),
                pairs=tuple((item.clip_id, item.provider) for item in wording_only),
            )
        )

    for provider, summary in sorted(summaries.items()):
        wer = summary.metrics.get("wer")
        if wer is not None and wer > thresholds.wer:
            signals.append(
                Signal(
                    severity="warning",
                    title=f"{provider}: WER {wer:.1%} exceeds the {thresholds.wer:.0%} threshold",
                    detail=(
                        "Pooled across the dataset. Check whether it is concentrated "
                        "in particular clips — acoustic conditions, one language, one "
                        "speaker — before concluding the provider is unsuitable."
                    ),
                )
            )

        for key in _MEANING_ERROR_KEYS:
            value = summary.metrics.get(key)
            if value is not None and value > thresholds.semantic_error:
                signals.append(
                    Signal(
                        severity="warning",
                        title=(
                            f"{provider}: {key} {value:.1%} exceeds the "
                            f"{thresholds.semantic_error:.0%} threshold"
                        ),
                        detail=(
                            "Errors that survived meaning-aware forgiveness — these "
                            "changed what was said, not merely how it was worded."
                        ),
                    )
                )

        for key in _MEANING_SCORE_KEYS:
            value = summary.metrics.get(key)
            if value is not None and value < thresholds.meaning_score:
                signals.append(
                    Signal(
                        severity="warning",
                        title=(
                            f"{provider}: {key} {value:.1%} is below the "
                            f"{thresholds.meaning_score:.0%} threshold"
                        ),
                        detail="Meaning was not preserved often enough to rely on.",
                    )
                )

        if summary.clips_failed:
            signals.append(
                Signal(
                    severity="warning",
                    title=f"{provider}: {summary.clips_failed} clip(s) failed to transcribe",
                    detail=(
                        "Failures are excluded from the accuracy figures, so this "
                        "provider's scores describe only the clips it managed."
                    ),
                )
            )

    order = {"critical": 0, "warning": 1, "info": 2}
    signals.sort(key=lambda signal: order[signal.severity])
    return signals


def coverage(results: list[StoredResult]) -> tuple[int, int]:
    """(judged, total) pairs — how much of the run the LLM metrics actually saw."""
    scored = [result for result in results if result.ok]
    judged = [result for result in scored if result.sampled and _has_meaning_metric(result)]
    return len(judged), len(scored)


def _has_meaning_metric(result: StoredResult) -> bool:
    return any(
        result.metric_value(key) is not None
        for key in _MEANING_SCORE_KEYS + _MEANING_ERROR_KEYS
    )
