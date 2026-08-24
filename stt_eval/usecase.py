"""Use-case-aware scoring, comparison and explanation.

Also the public face of the layer: the UI imports this one module rather than
five, while the pieces live where the architecture says they should —
`prompt_analyzer`, `requirement_extractor`, `metric_selector`,
`scenario_generator` and `metrics/use_case_metrics`.

The product principle this exists to serve: the provider with the lowest word
error rate is not necessarily the best provider for the job. A run is scored
against what the agent actually needs, and the winner is explained in those
terms rather than by a number nobody can decompose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metric_selector import (  # noqa: F401 - re-exported as the layer's API
    DERIVED_LABELS,
    TIER_RATIONALE,
    TIER_WEIGHTS,
    MetricPlan,
    Selection,
    apply_default_weights,
    select,
    set_weights,
)
from .metrics.use_case_metrics import category_scores
from .prompt_analyzer import COMMON_USE_CASES, analyze  # noqa: F401
from .requirement_extractor import (  # noqa: F401
    FIELD_TYPES,
    CriticalField,
    Requirements,
)
from .scenario_generator import SCENARIO_TYPES, TestScenario, generate  # noqa: F401
from .store import StoredResult


@dataclass
class UseCaseScore:
    score: float
    parts: dict[str, tuple[float, float]] = field(default_factory=dict)  # metric -> (value, weight)
    missing: list[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        return self.score * 100.0

    @property
    def complete(self) -> bool:
        return not self.missing


def metric_values(result: StoredResult) -> dict[str, float]:
    """Every metric this result can contribute, as an accuracy in 0..1.

    Error rates are inverted so that higher always means better, and the derived
    accuracy metrics are rolled up from the per-field verdicts that
    `critical_fields` already produced — no extra judge calls.
    """
    values: dict[str, float] = {}

    for key in ("wer", "cer", "semantic_wer", "llm_wer", "llm_cer"):
        value = result.metric_value(key)
        if value is not None:
            values[key] = max(0.0, 1.0 - value)

    for key in ("semantic_match", "intent_entity", "critical_fields"):
        value = result.metric_value(key)
        if value is not None:
            values[key] = value

    per_field = result.metric_extra("critical_fields").get("fields", {})
    values.update(category_scores(per_field))
    return values


def score(plan: MetricPlan, result: StoredResult) -> UseCaseScore:
    """Weighted score over the metrics this plan selected.

    Weights are renormalised across the metrics that actually produced a value,
    so a judge failure lowers confidence rather than silently deflating the
    score — and `missing` names what did not run so the UI can say so.
    """
    available = metric_values(result)
    parts: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    total = 0.0
    total_weight = 0.0

    for selection in plan.selections:
        if selection.weight <= 0:
            continue
        value = available.get(selection.metric)
        if value is None:
            missing.append(selection.metric)
            continue
        parts[selection.metric] = (value, selection.weight)
        total += value * selection.weight
        total_weight += selection.weight

    return UseCaseScore(
        score=(total / total_weight) if total_weight else 0.0,
        parts=parts,
        missing=missing,
    )


def comparison_table(
    plan: MetricPlan, results: list[StoredResult]
) -> tuple[list[str], list[dict]]:
    """One row per metric, one column per provider — the PRD's comparison view."""
    providers = [result.provider for result in results]
    scores = {result.provider: score(plan, result) for result in results}
    values = {result.provider: metric_values(result) for result in results}

    rows: list[dict] = [
        {
            "Metric": "Use-case score",
            **{
                provider: f"{scores[provider].percent:.0f}%" for provider in providers
            },
        }
    ]

    for selection in plan.selections:
        row = {"Metric": f"{selection.label} ({selection.tier.lower()})"}
        for provider in providers:
            value = values[provider].get(selection.metric)
            row[provider] = f"{value:.0%}" if value is not None else "—"
        rows.append(row)

    for label, key in (("p95 latency", "latency"), ("Estimated cost", "cost")):
        row = {"Metric": label}
        for result in results:
            if key == "latency":
                row[result.provider] = (
                    f"{result.latency_seconds:.2f}s" if result.latency_seconds else "—"
                )
            else:
                from .costs import rate_for

                row[result.provider] = (
                    f"${(result.duration_seconds / 60.0) * rate_for(result.provider):.5f}"
                )
        rows.append(row)

    return providers, rows


@dataclass
class Verdict:
    winner: str | None
    explanation: str
    runner_up: str | None = None


def explain_winner(
    plan: MetricPlan,
    results: list[StoredResult],
    *,
    use_case: str = "this use case",
    labels: dict[str, str] | None = None,
) -> Verdict:
    """Say which provider to use, and why — in the agent's terms, not WER's.

    Deliberately deterministic rather than another LLM call: the facts it cites
    (which fields were lost, which provider had lower WER) are already known, and
    an explanation that can disagree with the numbers beside it is worse than
    none.
    """
    usable = [result for result in results if result.ok]
    if not usable:
        return Verdict(winner=None, explanation="No provider produced a transcript.")

    def name(provider: str) -> str:
        return (labels or {}).get(provider, provider)

    ranked = sorted(usable, key=lambda item: score(plan, item).score, reverse=True)
    best = ranked[0]
    best_score = score(plan, best)
    lost = _lost_fields(best)

    lines = [
        f"**{name(best.provider)}** is recommended for {use_case}, "
        f"scoring {best_score.percent:.0f}%."
    ]

    if len(ranked) > 1:
        second = ranked[1]
        second_score = score(plan, second)
        best_wer = best.metric_value("wer")
        second_wer = second.metric_value("wer")
        second_lost = _lost_fields(second)

        # The interesting case, and the reason this whole mode exists: the
        # winner on use-case terms is not the winner on WER.
        if (
            best_wer is not None
            and second_wer is not None
            and best_wer > second_wer
            and second_lost
        ):
            lines.append(
                f"It had a *higher* word error rate than {name(second.provider)} "
                f"({best_wer:.1%} against {second_wer:.1%}), but "
                f"{name(second.provider)} lost {_join(second_lost)}, which for this "
                "agent is a failed interaction rather than a wording difference."
            )
        elif second_lost:
            lines.append(
                f"{name(second.provider)} scored {second_score.percent:.0f}% and lost "
                f"{_join(second_lost)}."
            )
        else:
            lines.append(
                f"{name(second.provider)} was close behind at "
                f"{second_score.percent:.0f}% with every critical value intact; "
                "either would serve, so pick on latency or cost."
            )

    if lost:
        lines.append(
            f"Note that even {name(best.provider)} did not capture {_join(lost)} — "
            "check whether that field is tolerable to lose before deploying."
        )

    failed = [name(result.provider) for result in results if not result.ok]
    if failed:
        lines.append(
            f"{_join(failed)} failed to transcribe and could not be compared."
        )

    return Verdict(
        winner=best.provider,
        runner_up=ranked[1].provider if len(ranked) > 1 else None,
        explanation=" ".join(lines),
    )


def _lost_fields(result: StoredResult) -> list[str]:
    per_field = result.metric_extra("critical_fields").get("fields", {})
    return [name for name, verdict in per_field.items() if not verdict.get("preserved")]


def _join(names: list[str]) -> str:
    if not names:
        return "nothing"
    if len(names) == 1:
        return f"**{names[0]}**"
    return ", ".join(f"**{name}**" for name in names[:-1]) + f" and **{names[-1]}**"
