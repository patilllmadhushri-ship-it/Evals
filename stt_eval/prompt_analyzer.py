"""Read a voice agent's system prompt and work out what its STT must get right.

This is the only place that interprets a system prompt, and it does so through
the judge rather than pattern matching, so an agent for an unanticipated domain
is handled the same way as a logistics one. No domain vocabulary is hard-coded
here — the analyser returns whatever the prompt actually calls for.
"""

from __future__ import annotations

from .judge import Judge, JudgeError
from .requirement_extractor import FIELD_TYPES, CriticalField, Requirements

_SYSTEM = """\
You analyse voice-agent system prompts to decide what a speech-to-text engine \
must transcribe correctly for that agent to work.

Identify the information the agent is required to collect, confirm or act on — \
the values where a transcription error changes the outcome. An order number, a \
delivery date, a dosage, an account name, an address, an amount, a spoken \
command.

Ignore instructions about tone, persona, formatting, escalation policy or \
conversation flow: none of those depend on transcription accuracy. Do not \
invent fields the prompt does not ask for, and do not pad the list to look \
thorough — a prompt that collects two things has exactly two critical fields.

Classify each field by type so metrics can be grouped: number, quantity, \
identifier, amount, date, time, location, name, action, or other. Use `action` \
for spoken commands the agent must execute, and `identifier` for reference \
codes and account numbers rather than plain numbers.

Also judge two things about the agent as a whole:
- semantic_matters: does the agent depend on the meaning of what was said \
surviving, beyond the individual fields? True for almost every conversational \
agent.
- exact_transcription_matters: does it need the exact words, not just the \
meaning? True for dictation, medical or legal transcription, and compliance \
recording; false for an agent that merely acts on what it hears.\
"""


def analyze(judge: Judge, system_prompt: str) -> Requirements:
    """Turn a system prompt into the requirements its STT has to satisfy."""
    if not (system_prompt or "").strip():
        raise JudgeError("Paste a system prompt first.")

    schema = {
        "type": "object",
        "properties": {
            "use_case": {"type": "string"},
            "objective": {"type": "string"},
            "summary": {"type": "string"},
            "user_actions": {"type": "array", "items": {"type": "string"}},
            "critical_action": {"type": "string"},
            "semantic_matters": {"type": "boolean"},
            "exact_transcription_matters": {"type": "boolean"},
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
        "required": [
            "use_case", "objective", "summary", "user_actions", "critical_action",
            "semantic_matters", "exact_transcription_matters", "fields",
        ],
        "additionalProperties": False,
    }

    payload = judge.ask_json(
        system=_SYSTEM,
        prompt=(
            "Analyse this voice-agent system prompt.\n\n"
            "--- SYSTEM PROMPT ---\n"
            f"{system_prompt.strip()}\n"
            "--- END ---\n\n"
            "Return: the use case in two or three words using the domain's own "
            "vocabulary; the agent's objective in one sentence; a short summary; "
            "what the user is expected to say or do; the single most critical "
            "action the agent must get right; the two importance judgments; and "
            "the critical fields with a short `why` giving the consequence of "
            "getting each one wrong."
        ),
        schema=schema,
    )

    fields: list[CriticalField] = []
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

    return Requirements(
        use_case=str(payload.get("use_case", "")).strip() or "Unclassified",
        objective=str(payload.get("objective", "")).strip(),
        summary=str(payload.get("summary", "")).strip(),
        user_actions=[
            str(item).strip() for item in payload.get("user_actions", []) if str(item).strip()
        ],
        critical_action=str(payload.get("critical_action", "")).strip(),
        semantic_matters=bool(payload.get("semantic_matters", True)),
        exact_transcription_matters=bool(payload.get("exact_transcription_matters", False)),
        fields=fields,
        raw_prompt=system_prompt.strip(),
    )
