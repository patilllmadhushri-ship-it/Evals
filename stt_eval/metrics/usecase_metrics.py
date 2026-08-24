"""The critical-field metric: did the transcript keep what this agent needs?

One judge call checks every critical field at once and returns a verdict per
field. That is deliberate: asking separately for number accuracy, date accuracy
and location accuracy would cost one call each and then need reconciling when
they disagreed about the same sentence. With per-field verdicts in hand, the
category rollups — numbers, dates, locations, names — are arithmetic, free, and
consistent with the detail shown to the user.

The per-field verdicts are also the explanation. `✓ Location ✓ Order number
✗ Date` falls straight out of them, alongside the judge's reason for the one
that failed.
"""

from __future__ import annotations

from ..judge import Judge
from ..prompts import JUDGE_SYSTEM_PROMPT
from .base import MetricValue
from .wer import DeterministicScores

#: Field types that roll up into each category metric.
CATEGORY_TYPES = {
    "number_accuracy": ("number", "identifier", "quantity"),
    "date_accuracy": ("date", "time"),
    "location_accuracy": ("location",),
    "name_accuracy": ("name",),
    "entity_accuracy": ("other",),
}


def critical_fields(
    *,
    judge: Judge,
    ground_truth: str,
    prediction: str,
    language: str,
    deterministic: DeterministicScores,
    context: dict | None = None,
) -> MetricValue:
    """Check each field the system prompt requires, individually.

    `context["fields"]` is a list of `{name, type}` describing what matters, and
    `context["expected"]` optionally maps a field to the value that was actually
    spoken — which makes the check exact rather than inferred.
    """
    fields = (context or {}).get("fields") or []
    if not fields:
        return MetricValue(
            key="critical_fields",
            error="No critical fields were supplied for this run.",
        )

    expected = (context or {}).get("expected") or {}
    described = "\n".join(
        f"- {item['name']} ({item.get('type', 'other')})"
        + (f", spoken value: \"{expected[item['name']]}\"" if item["name"] in expected else "")
        for item in fields
    )

    schema = {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "preserved": {"type": "boolean"},
                        "heard_as": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "preserved", "heard_as", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    }

    prompt = (
        f"Language: {language}\n\n"
        f"REFERENCE (what was actually said):\n{ground_truth}\n\n"
        f"PREDICTION (speech-to-text output):\n{prediction}\n\n"
        "A voice agent must capture these fields from this utterance:\n"
        f"{described}\n\n"
        "For each field, decide whether the PREDICTION preserves its value "
        "usably. Apply the usual rules: a different surface form of the same "
        "value is preserved (\"500\" for \"five hundred\", \"25th August\" for "
        "\"August 25th\"), while a changed digit, a different date, a different "
        "place or a name that now identifies someone else is not. A field that "
        "is missing from the prediction entirely is not preserved.\n"
        "Report `heard_as` with what the prediction actually contains for that "
        "field, or an empty string if it is absent."
    )

    payload = judge.ask_json(system=JUDGE_SYSTEM_PROMPT, prompt=prompt, schema=schema)

    verdicts: dict[str, dict] = {}
    for entry in payload.get("fields", []):
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        verdicts[name] = {
            "preserved": bool(entry.get("preserved")),
            "heard_as": str(entry.get("heard_as", "")).strip(),
            "reason": str(entry.get("reason", "")).strip(),
        }

    # A field the judge failed to report counts as not preserved: silence is not
    # evidence of success.
    per_field = {}
    for item in fields:
        name = item["name"]
        verdict = verdicts.get(
            name,
            {"preserved": False, "heard_as": "", "reason": "not reported by the judge"},
        )
        per_field[name] = {**verdict, "type": item.get("type", "other")}

    preserved = sum(1 for verdict in per_field.values() if verdict["preserved"])
    total = len(per_field)
    failed = [name for name, verdict in per_field.items() if not verdict["preserved"]]

    if failed:
        summary = f"{preserved}/{total} critical fields preserved. Lost: " + "; ".join(
            f"{name} ({per_field[name]['reason']})" for name in failed
        )
    else:
        summary = f"All {total} critical fields preserved."

    return MetricValue(
        key="critical_fields",
        value=(preserved / total) if total else None,
        reasoning=summary,
        errors=total - preserved,
        length=total,
        extra={"fields": per_field},
    )


def category_scores(per_field: dict[str, dict]) -> dict[str, float]:
    """Roll per-field verdicts up into number/date/location/name accuracy.

    Arithmetic on verdicts already paid for, rather than extra judge calls, so
    the categories can never contradict the per-field detail beside them.
    """
    scores: dict[str, float] = {}
    for metric, types in CATEGORY_TYPES.items():
        relevant = [
            verdict for verdict in per_field.values() if verdict.get("type") in types
        ]
        if not relevant:
            continue
        scores[metric] = sum(1 for item in relevant if item["preserved"]) / len(relevant)
    return scores
