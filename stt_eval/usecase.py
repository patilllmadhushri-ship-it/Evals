"""Use-case-aware evaluation, driven by a voice agent's own system prompt.

The dataset workflow asks "how accurate is this provider on my audio". This
layer asks a narrower and more useful question: **does this provider capture the
things my agent actually needs?** A logistics agent that mishears a delivery
date has failed, whatever its word error rate; a support agent that renders
"five hundred" as "500" has not.

The chain is:

    system prompt -> critical fields -> recommended metrics -> test scenario

Every step is derived from the prompt rather than a fixed list of domains, so an
agent for a use case nobody anticipated works the same way. The judge client
does the extraction, which is why this module holds no domain vocabulary of its
own beyond the field *types* used to group results.

Nothing here transcribes or scores audio — that stays with the existing runner,
providers and metric registry. This module only decides what to look for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import Judge, JudgeError

#: Field types the extractor may assign. Kept small and generic so the rollups
#: below stay meaningful; the *fields* themselves are free-form.
FIELD_TYPES = ("number", "identifier", "date", "time", "location", "name", "other")

#: Which recommended metric each field type implies, and why. The rationale is
#: shown to the user verbatim — a recommendation nobody can interrogate is
#: indistinguishable from a guess.
TYPE_TO_METRIC = {
    "number": ("number_accuracy", "the agent must capture numeric values exactly"),
    "identifier": ("number_accuracy", "identifiers are digit strings where one wrong character is a wrong record"),
    "quantity": ("number_accuracy", "quantities decide what is actually ordered or dispensed"),
    "date": ("date_accuracy", "a misheard date sends the work to the wrong day"),
    "time": ("date_accuracy", "a misheard time sends the work to the wrong slot"),
    "location": ("location_accuracy", "a misheard place sends the work to the wrong address"),
    "name": ("name_accuracy", "a misheard name attaches the work to the wrong person"),
    "other": ("entity_accuracy", "this field is required by the prompt and must survive transcription"),
}

#: Metrics that are always recommended, whatever the prompt says.
BASELINE_METRICS = {
    "wer": "the standard literal accuracy baseline, comparable across providers and datasets",
    "cer": "catches spelling and segmentation errors that WER's word-level view misses",
    "critical_fields": "checks the specific values this system prompt says the agent must collect",
    "semantic_match": "confirms the transcript still means what the speaker said, beyond the individual fields",
}


@dataclass
class CriticalField:
    """One thing the agent must capture correctly, per its own system prompt."""

    name: str
    type: str
    why: str = ""

    @property
    def metric(self) -> str:
        return TYPE_TO_METRIC.get(self.type, TYPE_TO_METRIC["other"])[0]


@dataclass
class UseCaseProfile:
    use_case: str
    summary: str
    fields: list[CriticalField] = field(default_factory=list)
    raw_prompt: str = ""

    @property
    def field_names(self) -> list[str]:
        return [item.name for item in self.fields]

    def fields_of_type(self, *types: str) -> list[CriticalField]:
        return [item for item in self.fields if item.type in types]


@dataclass
class Recommendation:
    metric: str
    reason: str
    #: Fields that drove this recommendation, for the explanation.
    driven_by: list[str] = field(default_factory=list)


@dataclass
class TestScenario:
    """A sentence written to exercise this prompt's critical fields."""

    sentence: str
    expected: dict[str, str] = field(default_factory=dict)  # field name -> spoken value
    notes: str = ""


_EXTRACTION_SYSTEM = """\
You analyse voice-agent system prompts to decide what a speech-to-text engine \
must transcribe correctly for that agent to work.

Read the prompt and identify the information the agent is required to collect, \
confirm or act on. These are the values where a transcription error changes the \
outcome — an order number, a delivery date, a dosage, a customer name, an \
address, an amount.

Ignore instructions about tone, persona, formatting, escalation policy or \
conversation flow: those do not depend on transcription accuracy. Do not invent \
fields the prompt does not call for, and do not pad the list to look thorough — \
a prompt that collects two things has two critical fields.

Classify each field by type so downstream metrics can be grouped: number, \
identifier, date, time, location, name, or other. Name the use case in two or \
three words, in the domain's own vocabulary.\
"""

_SCENARIO_SYSTEM = """\
You write short test utterances for evaluating speech-to-text engines.

Given a use case and the fields an agent must capture, write ONE natural \
sentence a real caller would say, containing a concrete value for every field. \
It has to be speakable aloud in a few seconds by someone reading it off a \
screen.

Make the values genuinely testable — the kinds of thing recognisers get wrong. \
Prefer multi-digit numbers over "one" or "two", a real place name over "here", a \
specific date over "tomorrow", a name that is not the most common spelling. \
Never use placeholder text like [NAME] or XXX; write the actual value.

Return the sentence and, separately, the exact spoken value you used for each \
field, so the evaluation can check each one individually.\
"""


def extract_requirements(judge: Judge, system_prompt: str) -> UseCaseProfile:
    """Read a voice agent's system prompt and find what its STT must get right."""
    if not (system_prompt or "").strip():
        raise JudgeError("Paste a system prompt first.")

    schema = {
        "type": "object",
        "properties": {
            "use_case": {"type": "string"},
            "summary": {"type": "string"},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": list(FIELD_TYPES)},
                        "why": {"type": "string"},
                    },
                    "required": ["name", "type", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["use_case", "summary", "fields"],
        "additionalProperties": False,
    }

    payload = judge.ask_json(
        system=_EXTRACTION_SYSTEM,
        prompt=(
            "Analyse this voice-agent system prompt.\n\n"
            "--- SYSTEM PROMPT ---\n"
            f"{system_prompt.strip()}\n"
            "--- END ---\n\n"
            "Return the use case, a one-sentence summary of what the agent does, "
            "and the critical fields its speech-to-text must capture correctly. "
            "For each field give a short `why` explaining the consequence of "
            "getting it wrong."
        ),
        schema=schema,
    )

    fields = []
    for entry in payload.get("fields", []):
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        field_type = str(entry.get("type", "other")).strip().lower()
        fields.append(
            CriticalField(
                name=name,
                type=field_type if field_type in FIELD_TYPES else "other",
                why=str(entry.get("why", "")).strip(),
            )
        )

    return UseCaseProfile(
        use_case=str(payload.get("use_case", "")).strip() or "Unclassified",
        summary=str(payload.get("summary", "")).strip(),
        fields=fields,
        raw_prompt=system_prompt.strip(),
    )


def recommend_metrics(profile: UseCaseProfile) -> list[Recommendation]:
    """Choose the metrics that matter for this prompt, with reasons.

    Baselines are always included — WER and CER stay comparable across runs and
    providers — and field-driven metrics are added on top, each carrying the
    fields that justified it.
    """
    recommendations: list[Recommendation] = [
        Recommendation(metric=key, reason=reason)
        for key, reason in BASELINE_METRICS.items()
    ]

    driven: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    for item in profile.fields:
        metric, why = TYPE_TO_METRIC.get(item.type, TYPE_TO_METRIC["other"])
        driven.setdefault(metric, []).append(item.name)
        reasons.setdefault(metric, why)

    for metric, names in driven.items():
        if metric in BASELINE_METRICS:
            # Fold the field names into the baseline's own explanation.
            for recommendation in recommendations:
                if recommendation.metric == metric:
                    recommendation.driven_by = names
            continue
        recommendations.append(
            Recommendation(
                metric=metric,
                reason=(
                    f"The system prompt requires the agent to capture "
                    f"{_join(names)}, and {reasons[metric]}."
                ),
                driven_by=names,
            )
        )
    return recommendations


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def generate_scenario(judge: Judge, profile: UseCaseProfile, *, language: str = "en-IN") -> TestScenario:
    """Write a sentence that exercises this prompt's critical fields."""
    if not profile.fields:
        raise JudgeError(
            "No critical fields were extracted, so there is nothing specific to test."
        )

    schema = {
        "type": "object",
        "properties": {
            "sentence": {"type": "string"},
            "values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["field", "value"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["sentence", "values", "notes"],
        "additionalProperties": False,
    }

    described = "\n".join(f"- {item.name} ({item.type})" for item in profile.fields)
    payload = judge.ask_json(
        system=_SCENARIO_SYSTEM,
        prompt=(
            f"Use case: {profile.use_case}\n"
            f"What the agent does: {profile.summary}\n"
            f"Language: {language}\n\n"
            f"Fields the speech-to-text must capture:\n{described}\n\n"
            "Write one sentence a caller would plausibly say that contains a "
            "concrete, testable value for every field above. In `notes`, say in "
            "one sentence which parts you expect to be hardest to transcribe."
        ),
        schema=schema,
    )

    expected = {
        str(entry.get("field", "")).strip(): str(entry.get("value", "")).strip()
        for entry in payload.get("values", [])
        if str(entry.get("field", "")).strip()
    }
    return TestScenario(
        sentence=str(payload.get("sentence", "")).strip(),
        expected=expected,
        notes=str(payload.get("notes", "")).strip(),
    )


# -- scoring ---------------------------------------------------------------

#: How the headline use-case score is composed. Exposed so the UI can show it —
#: a single percentage nobody can decompose is not worth reporting.
SCORE_WEIGHTS = {
    "critical_fields": 0.60,
    "semantic_match": 0.25,
    "wer": 0.15,
}

WEIGHT_RATIONALE = {
    "critical_fields": (
        "Weighted highest because these are the values the agent exists to "
        "collect — losing one is a failed call regardless of the rest."
    ),
    "semantic_match": (
        "Catches meaning changes outside the named fields, such as a flipped "
        "negation in the surrounding sentence."
    ),
    "wer": (
        "Kept as a minority weight so a provider that mangles everything else "
        "cannot score well on fields alone. Contributes as 1 - WER."
    ),
}


@dataclass
class UseCaseScore:
    score: float
    parts: dict[str, tuple[float, float]] = field(default_factory=dict)  # metric -> (value, weight)
    missing: list[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        return self.score * 100.0


def score(metric_values: dict[str, float | None]) -> UseCaseScore:
    """Combine the weighted metrics into one number, skipping any that are absent.

    Weights are renormalised over the metrics that actually ran, so a run
    without the judge still produces a score rather than silently deflating it.
    """
    parts: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    total_weight = 0.0
    total = 0.0

    for metric, weight in SCORE_WEIGHTS.items():
        value = metric_values.get(metric)
        if value is None:
            missing.append(metric)
            continue
        # WER is an error rate; everything else is already an accuracy.
        contribution = max(0.0, 1.0 - value) if metric in {"wer", "cer"} else value
        parts[metric] = (contribution, weight)
        total += contribution * weight
        total_weight += weight

    return UseCaseScore(
        score=(total / total_weight) if total_weight else 0.0,
        parts=parts,
        missing=missing,
    )
