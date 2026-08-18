"""The metrics ladder, from strict-literal to meaning-aware.

Each metric is independently toggleable and independently fault-isolated:
turning one off to save cost, or having one fail, never removes or blocks the
others. That contract is enforced in `evaluate_pair` — every LLM metric runs
inside its own try/except and records its own error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from ..judge import Judge, JudgeError
from . import llm_metrics
from .base import MetricValue
from .wer import DeterministicScores, score_deterministic

MetricKind = Literal["deterministic", "llm", "derived", "measured"]
Aggregation = Literal["pooled", "mean", "rate"]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    kind: MetricKind
    question: str
    default_on: bool
    aggregation: Aggregation
    lower_is_better: bool
    #: Metrics keyed here produce a written explanation alongside the number.
    has_reasoning: bool = False


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        key="wer",
        label="Word Error Rate (WER)",
        kind="deterministic",
        question=(
            "What fraction of words were substituted, deleted or inserted, after "
            "case/punctuation normalisation? Pooled across the dataset."
        ),
        default_on=True,
        aggregation="pooled",
        lower_is_better=True,
    ),
    MetricSpec(
        key="cer",
        label="Character Error Rate (CER)",
        kind="deterministic",
        question=(
            "Same as WER at the character level — more forgiving of spelling and "
            "segmentation differences."
        ),
        default_on=True,
        aggregation="pooled",
        lower_is_better=True,
    ),
    MetricSpec(
        key="intent_entity",
        label="Intent & entity preservation",
        kind="llm",
        question=(
            "Ignoring exact wording, did the transcript preserve the speaker's "
            "intent and the key entities (names, numbers, amounts)?"
        ),
        default_on=True,
        aggregation="mean",
        lower_is_better=False,
        has_reasoning=True,
    ),
    MetricSpec(
        key="llm_wer",
        label="LLM-WER",
        kind="llm",
        question=(
            "WER recomputed after an LLM forgives differing segments that are "
            "meaning-equivalent rewordings — directly comparable to plain WER."
        ),
        default_on=True,
        aggregation="pooled",
        lower_is_better=True,
        has_reasoning=True,
    ),
    MetricSpec(
        key="llm_cer",
        label="LLM-CER",
        kind="llm",
        question="CER recomputed on the same meaning-corrected text as LLM-WER.",
        default_on=True,
        aggregation="pooled",
        lower_is_better=True,
        has_reasoning=True,
    ),
    MetricSpec(
        key="semantic_wer",
        label="Semantic WER",
        kind="llm",
        question=(
            "A single LLM call aligns, normalises and semantically forgives "
            "differences end to end, returning a WER-comparable error count."
        ),
        default_on=True,
        aggregation="pooled",
        lower_is_better=True,
        has_reasoning=True,
    ),
    MetricSpec(
        key="semantic_match",
        label="Semantic match (LLM judge)",
        kind="llm",
        question=(
            "Binary pass/fail — does this transcription mean the same thing as the "
            "ground truth? With a written reason."
        ),
        default_on=False,
        aggregation="mean",
        lower_is_better=False,
        has_reasoning=True,
    ),
]

METRICS_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}
DETERMINISTIC_KEYS = [spec.key for spec in METRIC_SPECS if spec.kind == "deterministic"]
LLM_KEYS = [spec.key for spec in METRIC_SPECS if spec.kind == "llm"]


@dataclass
class PairEvaluation:
    clip_id: str
    provider: str
    ground_truth: str
    prediction: str
    metrics: dict[str, MetricValue] = field(default_factory=dict)

    def value(self, key: str) -> float | None:
        metric = self.metrics.get(key)
        return metric.value if metric and metric.ok else None


_LLM_RUNNERS: dict[str, Callable[..., MetricValue]] = {
    "intent_entity": llm_metrics.intent_entity,
    "llm_wer": llm_metrics.llm_wer_cer,
    "semantic_wer": llm_metrics.semantic_wer,
    "semantic_match": llm_metrics.semantic_match,
}


def evaluate_pair(
    *,
    clip_id: str,
    provider: str,
    ground_truth: str,
    prediction: str,
    enabled_metrics: list[str],
    judge: Judge | None,
    language: str,
) -> PairEvaluation:
    """Score one (ground truth, prediction) pair across the enabled metrics."""
    evaluation = PairEvaluation(
        clip_id=clip_id, provider=provider, ground_truth=ground_truth, prediction=prediction
    )

    deterministic: DeterministicScores = score_deterministic(ground_truth, prediction)
    evaluation.metrics["wer"] = MetricValue(
        key="wer",
        value=deterministic.wer,
        errors=deterministic.word_errors,
        length=deterministic.word_length,
    )
    evaluation.metrics["cer"] = MetricValue(
        key="cer",
        value=deterministic.cer,
        errors=deterministic.char_errors,
        length=deterministic.char_length,
    )

    requested_llm = [key for key in enabled_metrics if key in LLM_KEYS]
    if not requested_llm:
        return evaluation

    if judge is None:
        for key in requested_llm:
            evaluation.metrics[key] = MetricValue(key=key, error="No judge configured.")
        return evaluation

    # llm_wer and llm_cer share one judge call — they are two readings of the
    # same meaning-corrected text.
    to_run = [key for key in requested_llm if key != "llm_cer"]
    for key in to_run:
        runner = _LLM_RUNNERS[key]
        try:
            result = runner(
                judge=judge,
                ground_truth=ground_truth,
                prediction=prediction,
                language=language,
                deterministic=deterministic,
            )
        except JudgeError as exc:
            failure = MetricValue(key=key, error=str(exc))
            evaluation.metrics[key] = failure
            if key == "llm_wer" and "llm_cer" in requested_llm:
                evaluation.metrics["llm_cer"] = MetricValue(key="llm_cer", error=str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - one metric must never sink the rest
            message = f"{type(exc).__name__}: {exc}"
            evaluation.metrics[key] = MetricValue(key=key, error=message)
            if key == "llm_wer" and "llm_cer" in requested_llm:
                evaluation.metrics["llm_cer"] = MetricValue(key="llm_cer", error=message)
            continue

        if key == "llm_wer":
            word_value, char_value = result  # type: ignore[misc]
            evaluation.metrics["llm_wer"] = word_value
            if "llm_cer" in requested_llm:
                evaluation.metrics["llm_cer"] = char_value
        else:
            evaluation.metrics[key] = result  # type: ignore[assignment]

    return evaluation


__all__ = [
    "METRIC_SPECS",
    "METRICS_BY_KEY",
    "DETERMINISTIC_KEYS",
    "LLM_KEYS",
    "MetricSpec",
    "MetricValue",
    "PairEvaluation",
    "evaluate_pair",
]
