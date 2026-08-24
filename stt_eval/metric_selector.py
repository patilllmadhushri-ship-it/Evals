"""Choose the metrics that matter for one agent, and say why.

The product principle is that the same metric is not equally important for every
use case. A dictation agent lives or dies by exact wording; a logistics agent
does not care whether the caller said "five hundred" or "500", but a wrong digit
in the order number is a failed delivery. So the selection, the tiering and the
weighting all derive from what the prompt actually asks for.

Three tiers, in the PRD's terms:

* **PRIMARY** — accuracy of the specific values this agent collects. These are
  what a failure actually looks like for this use case.
* **SECONDARY** — supporting signals: meaning preserved, entities intact.
* **BASELINE** — WER and CER, which stay comparable across every run and every
  provider regardless of use case.

Every selection carries the fields that drove it, so no recommendation is a
black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .requirement_extractor import Requirements

Tier = Literal["PRIMARY", "SECONDARY", "BASELINE"]

#: Share of the use-case score each tier carries, before being split within it.
TIER_WEIGHTS: dict[Tier, float] = {
    "PRIMARY": 0.60,
    "SECONDARY": 0.25,
    "BASELINE": 0.15,
}

TIER_RATIONALE: dict[Tier, str] = {
    "PRIMARY": (
        "The values this agent exists to collect. Losing one is a failed "
        "interaction however good the rest of the transcript is."
    ),
    "SECONDARY": (
        "Meaning and entity preservation beyond the named fields — a flipped "
        "negation elsewhere in the sentence still breaks the agent."
    ),
    "BASELINE": (
        "General transcription quality, comparable across runs and providers. "
        "Kept at a minority weight so a provider that mangles everything else "
        "cannot score well on fields alone."
    ),
}

#: Metrics that are always evaluated, whatever the prompt says.
BASELINE_SELECTIONS = {
    "wer": "Baseline literal accuracy, comparable across providers and datasets.",
    "cer": "Catches spelling and segmentation errors that WER's word-level view misses.",
}

#: Human labels for the derived accuracy metrics, which are rollups rather than
#: entries in the main metric registry.
DERIVED_LABELS = {
    "number_accuracy": "Number Accuracy",
    "quantity_accuracy": "Quantity Accuracy",
    "identifier_accuracy": "Identifier Accuracy",
    "amount_accuracy": "Amount Accuracy",
    "date_accuracy": "Date Accuracy",
    "time_accuracy": "Time Accuracy",
    "location_accuracy": "Location Accuracy",
    "name_accuracy": "Name Accuracy",
    "action_accuracy": "Action Accuracy",
    "entity_accuracy": "Entity Accuracy",
}


@dataclass
class Selection:
    metric: str
    tier: Tier
    reason: str
    driven_by: list[str] = field(default_factory=list)
    weight: float = 0.0

    @property
    def label(self) -> str:
        from .metrics import METRICS_BY_KEY  # local import avoids a cycle

        if self.metric in DERIVED_LABELS:
            return DERIVED_LABELS[self.metric]
        spec = METRICS_BY_KEY.get(self.metric)
        return spec.label if spec else self.metric.replace("_", " ").title()


@dataclass
class MetricPlan:
    selections: list[Selection] = field(default_factory=list)

    def by_tier(self, tier: Tier) -> list[Selection]:
        return [item for item in self.selections if item.tier == tier]

    @property
    def weights(self) -> dict[str, float]:
        return {item.metric: item.weight for item in self.selections}

    @property
    def runnable_metrics(self) -> list[str]:
        """Metrics the existing engine can actually run.

        The derived accuracy metrics are rollups of `critical_fields`, so they
        are not passed to the runner — computing them costs nothing extra once
        that one metric has run.
        """
        from .metrics import METRICS_BY_KEY

        keys = [item.metric for item in self.selections if item.metric in METRICS_BY_KEY]
        if any(item.metric in DERIVED_LABELS for item in self.selections):
            if "critical_fields" not in keys:
                keys.append("critical_fields")
        return keys


def select(requirements: Requirements) -> MetricPlan:
    """Build the metric plan for one agent's requirements."""
    selections: list[Selection] = []

    # PRIMARY — one per group of critical fields present.
    for group, fields in requirements.groups.items():
        names = [item.name for item in fields]
        consequence = fields[0].consequence
        selections.append(
            Selection(
                metric=group,
                tier="PRIMARY",
                reason=(
                    f"The system prompt requires the agent to capture "
                    f"{_join(names)}, and {consequence}."
                ),
                driven_by=names,
            )
        )

    # SECONDARY — meaning, and the field check that feeds the primaries.
    if requirements.fields:
        selections.append(
            Selection(
                metric="critical_fields",
                tier="SECONDARY",
                reason=(
                    "Checks each required value individually, and is what the "
                    "primary accuracy metrics above are computed from."
                ),
                driven_by=requirements.field_names,
            )
        )
    if requirements.semantic_matters:
        selections.append(
            Selection(
                metric="semantic_match",
                tier="SECONDARY",
                reason=(
                    "The agent acts on what the caller meant, so a transcript can "
                    "be field-perfect and still be wrong if the surrounding "
                    "meaning changed."
                ),
            )
        )
        selections.append(
            Selection(
                metric="intent_entity",
                tier="SECONDARY",
                reason="Scores intent and entity preservation as a single figure.",
            )
        )
    if requirements.exact_transcription_matters:
        selections.append(
            Selection(
                metric="llm_wer",
                tier="PRIMARY",
                reason=(
                    "This agent needs the words themselves, not only the meaning, "
                    "so word-level error matters directly — measured after "
                    "forgiving differences that are genuinely equivalent."
                ),
            )
        )

    # BASELINE — always.
    for metric, reason in BASELINE_SELECTIONS.items():
        selections.append(Selection(metric=metric, tier="BASELINE", reason=reason))

    plan = MetricPlan(selections=selections)
    apply_default_weights(plan)
    return plan


def apply_default_weights(plan: MetricPlan) -> None:
    """Split each tier's share evenly across the metrics in it.

    Only metrics that contribute a score are weighted: `critical_fields` sits in
    the plan because the primaries are derived from it, but weighting it as well
    would count the same evidence twice.
    """
    for tier, share in TIER_WEIGHTS.items():
        members = [
            item
            for item in plan.by_tier(tier)
            if item.metric != "critical_fields"
        ]
        for item in plan.by_tier(tier):
            item.weight = 0.0
        if not members:
            continue
        each = share / len(members)
        for item in members:
            item.weight = each
    _normalise(plan)


def _normalise(plan: MetricPlan) -> None:
    total = sum(item.weight for item in plan.selections)
    if total <= 0:
        return
    for item in plan.selections:
        item.weight = item.weight / total


def set_weights(plan: MetricPlan, weights: dict[str, float]) -> MetricPlan:
    """Apply user-adjusted weights, renormalised so they always sum to 1."""
    for item in plan.selections:
        if item.metric in weights:
            item.weight = max(0.0, float(weights[item.metric]))
    _normalise(plan)
    return plan


def _join(names: list[str]) -> str:
    if not names:
        return "nothing"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"
