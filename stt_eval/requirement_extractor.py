"""What a voice agent needs from its speech-to-text, as data.

The types here are deliberately free-form: a field is whatever the agent's own
system prompt says it must collect, and the only closed vocabulary is the field
*type*, which exists so metrics can be grouped and rolled up. A use case nobody
anticipated therefore needs no code change — it arrives as fields with types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Field types the analyser may assign. Closed so rollups stay meaningful.
FIELD_TYPES = (
    "number",
    "quantity",
    "identifier",
    "amount",
    "date",
    "time",
    "location",
    "name",
    "action",
    "other",
)

#: Types whose accuracy is reported together, and the metric they roll into.
TYPE_GROUPS = {
    "number_accuracy": ("number", "quantity"),
    "identifier_accuracy": ("identifier",),
    "amount_accuracy": ("amount",),
    "date_accuracy": ("date",),
    "time_accuracy": ("time",),
    "location_accuracy": ("location",),
    "name_accuracy": ("name",),
    "action_accuracy": ("action",),
    "entity_accuracy": ("other",),
}

#: Why an error in each type matters. Shown to the user as the metric rationale.
TYPE_CONSEQUENCE = {
    "number": "a wrong number changes what is actually ordered, dispensed or paid",
    "quantity": "a wrong quantity changes what the customer receives",
    "identifier": "identifiers are digit strings where one wrong character is a different record",
    "amount": "a wrong amount is a wrong transaction",
    "date": "a misheard date schedules the work on the wrong day",
    "time": "a misheard time schedules the work in the wrong slot",
    "location": "a misheard place sends the work to the wrong address",
    "name": "a misheard name attaches the work to the wrong person",
    "action": "a misheard command makes the agent do the wrong thing",
    "other": "the prompt requires this value, so it must survive transcription",
}


@dataclass
class CriticalField:
    """One value the agent must capture correctly, per its own system prompt."""

    name: str
    type: str
    why: str = ""

    @property
    def group(self) -> str:
        for metric, types in TYPE_GROUPS.items():
            if self.type in types:
                return metric
        return "entity_accuracy"

    @property
    def consequence(self) -> str:
        return TYPE_CONSEQUENCE.get(self.type, TYPE_CONSEQUENCE["other"])


@dataclass
class Requirements:
    """The full picture of what this prompt demands of its STT."""

    use_case: str
    objective: str = ""
    summary: str = ""
    user_actions: list[str] = field(default_factory=list)
    critical_action: str = ""
    fields: list[CriticalField] = field(default_factory=list)
    #: Does the agent depend on meaning surviving, beyond the named fields?
    semantic_matters: bool = True
    #: Does it need the words themselves, not just the meaning? (dictation,
    #: transcription-of-record, compliance recording)
    exact_transcription_matters: bool = False
    raw_prompt: str = ""

    @property
    def field_names(self) -> list[str]:
        return [item.name for item in self.fields]

    @property
    def groups(self) -> dict[str, list[CriticalField]]:
        """Critical fields bucketed by the metric they roll into."""
        grouped: dict[str, list[CriticalField]] = {}
        for item in self.fields:
            grouped.setdefault(item.group, []).append(item)
        return grouped

    def as_context(self, expected: dict[str, str] | None = None) -> dict:
        """The payload the critical-field metric needs at scoring time."""
        return {
            "fields": [{"name": item.name, "type": item.type} for item in self.fields],
            "expected": expected or {},
        }
