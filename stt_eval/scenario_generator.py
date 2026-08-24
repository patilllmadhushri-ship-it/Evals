"""Write test utterances that probe what this particular agent depends on.

A generic sentence tells you little. The point of generating from the system
prompt is that the utterance contains exactly the values whose corruption would
break *this* agent — and, in the adversarial modes, the kind of corruption that
a word error rate would barely register.

Scenario types map to the PRD's list. They differ only in what the generator is
told to stress, so adding another is a dict entry rather than a code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import Judge, JudgeError
from .requirement_extractor import Requirements

#: type -> (label, what the generator should stress)
SCENARIO_TYPES: dict[str, tuple[str, str]] = {
    "normal": (
        "Normal case",
        "All required information spoken clearly and naturally, as a cooperative "
        "caller would say it in one breath.",
    ),
    "number": (
        "Number stress",
        "Load the sentence with the numeric material this agent collects — "
        "multi-digit identifiers, quantities, amounts, phone numbers. Prefer "
        "digit strings that are easy to mishear, and say some as words and some "
        "as digits.",
    ),
    "datetime": (
        "Date and time stress",
        "Load the sentence with dates and times, including a relative one "
        "('the day after tomorrow') alongside an absolute one, and a time with "
        "minutes rather than an exact hour.",
    ),
    "entity": (
        "Entity stress",
        "Load the sentence with proper nouns — place names, personal names, "
        "organisation or product names — favouring ones a recogniser is likely "
        "to mangle rather than the most common spellings.",
    ),
    "semantic": (
        "Semantic variation",
        "Say the required information in a roundabout, natural way, using "
        "wording a recogniser may render differently while the meaning is "
        "unchanged. This tests whether the evaluation forgives harmless "
        "rewording.",
    ),
    "critical": (
        "Critical / adversarial",
        "Include at least one element where a single-word transcription error "
        "reverses or materially changes the meaning — a negation, a direction "
        "('to' versus 'from' an account), or a near-homophone that would send "
        "the agent somewhere else. Keep it natural: a caller would really say "
        "this.",
    ),
}

_SYSTEM = """\
You write short test utterances for evaluating speech-to-text engines.

Write ONE natural sentence a real caller would say to the agent described, \
containing a concrete value for every field listed. It must be speakable aloud \
in a few seconds by someone reading it off a screen.

Make the values genuinely testable — the kinds of thing recognisers get wrong. \
Prefer multi-digit numbers over "one" or "two", a real place name over "here", a \
specific date over "tomorrow", a name that is not the most common spelling. \
Never write placeholder text such as [NAME] or XXX; write the actual value.

Return the sentence, the exact spoken value you used for each field so each can \
be checked individually, and a note saying which part you expect to be hardest \
to transcribe and why.\
"""


@dataclass
class TestScenario:
    sentence: str
    scenario_type: str = "normal"
    expected: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def label(self) -> str:
        return SCENARIO_TYPES.get(self.scenario_type, ("Custom", ""))[0]


def generate(
    judge: Judge,
    requirements: Requirements,
    *,
    scenario_type: str = "normal",
    language: str = "en-IN",
) -> TestScenario:
    """Write one utterance of the requested type for these requirements."""
    if not requirements.fields:
        raise JudgeError(
            "No critical fields were extracted, so there is nothing specific to test."
        )
    if scenario_type not in SCENARIO_TYPES:
        raise JudgeError(f"Unknown scenario type: {scenario_type}")

    label, emphasis = SCENARIO_TYPES[scenario_type]
    described = "\n".join(
        f"- {item.name} ({item.type})" for item in requirements.fields
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

    payload = judge.ask_json(
        system=_SYSTEM,
        prompt=(
            f"Use case: {requirements.use_case}\n"
            f"What the agent does: {requirements.summary or requirements.objective}\n"
            f"Language: {language}\n\n"
            f"Fields the speech-to-text must capture:\n{described}\n\n"
            f"Scenario type: {label}\n"
            f"What to stress: {emphasis}\n\n"
            "Write the sentence in the language given above. If that language is "
            "commonly code-switched with English, code-switch the way a real "
            "speaker would rather than translating every term."
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
        scenario_type=scenario_type,
        expected=expected,
        notes=str(payload.get("notes", "")).strip(),
    )
