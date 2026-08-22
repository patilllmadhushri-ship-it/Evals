"""Meaning-aware metrics, each one judge call.

Plain WER over-penalises transcriptions that differ in wording but preserve
meaning ("500" vs "five hundred rupees"). These metrics ask an LLM which of the
differences actually mattered, and every one of them returns its written
reasoning so a user can see *why* a clip passed or failed.
"""

from __future__ import annotations

from ..judge import Judge
from ..prompts import JUDGE_SYSTEM_PROMPT
from .align import differing_segments
from .base import MetricValue
from .wer import DeterministicScores, score_deterministic

#: Every metric judges against the same rules — see `stt_eval/prompts.py`.
_SYSTEM = JUDGE_SYSTEM_PROMPT


def _context(ground_truth: str, prediction: str, language: str) -> str:
    return (
        f"Language: {language}\n\n"
        f"REFERENCE (correct transcript):\n{ground_truth}\n\n"
        f"PREDICTION (speech-to-text output):\n{prediction}\n"
    )


def intent_entity(
    *,
    judge: Judge,
    ground_truth: str,
    prediction: str,
    language: str,
    deterministic: DeterministicScores,
) -> MetricValue:
    """Did the transcript preserve the speaker's intent and the key entities?"""
    schema = {
        "type": "object",
        "properties": {
            "intent_preserved": {"type": "boolean"},
            "entities_total": {"type": "integer"},
            "entities_preserved": {"type": "integer"},
            "score": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": [
            "intent_preserved", "entities_total", "entities_preserved", "score", "reasoning",
        ],
        "additionalProperties": False,
    }
    prompt = (
        _context(ground_truth, prediction, language)
        + "\nIgnoring exact wording, decide whether the prediction preserves the "
        "speaker's intent and the key entities (names, numbers, amounts, dates, "
        "places, identifiers).\n"
        "Count the key entities in the REFERENCE, then count how many survive in "
        "the PREDICTION with the same value.\n"
        "Return `score` between 0 and 1: 1.0 when intent and every entity are "
        "preserved, 0.0 when the intent is lost. When intent holds but entities "
        "are corrupted, score the fraction preserved. Explain in `reasoning`."
    )
    payload = judge.ask_json(system=_SYSTEM, prompt=prompt, schema=schema)
    score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
    entities = payload.get("entities_total")
    preserved = payload.get("entities_preserved")
    reasoning = str(payload.get("reasoning", "")).strip()
    if isinstance(entities, int) and isinstance(preserved, int) and entities > 0:
        reasoning = f"{preserved}/{entities} key entities preserved. {reasoning}"
    return MetricValue(key="intent_entity", value=score, reasoning=reasoning)


def llm_wer_cer(
    *,
    judge: Judge,
    ground_truth: str,
    prediction: str,
    language: str,
    deterministic: DeterministicScores,
) -> tuple[MetricValue, MetricValue]:
    """Forgive meaning-equivalent segments, then recompute WER and CER.

    The result is directly comparable to plain WER/CER, so the pair of numbers
    shows how much of the raw error was harmless.
    """
    segments = differing_segments(deterministic.word_alignment)
    if not segments:
        return (
            MetricValue(
                key="llm_wer",
                value=deterministic.wer,
                errors=deterministic.word_errors,
                length=deterministic.word_length,
                reasoning="No differences to judge — the transcript matches exactly.",
            ),
            MetricValue(
                key="llm_cer",
                value=deterministic.cer,
                errors=deterministic.char_errors,
                length=deterministic.char_length,
                reasoning="No differences to judge — the transcript matches exactly.",
            ),
        )

    listing = "\n".join(
        f"{index}. reference: \"{segment.ref_text or '(nothing)'}\" | "
        f"prediction: \"{segment.hyp_text or '(nothing)'}\""
        for index, segment in enumerate(segments)
    )
    schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "equivalent": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["index", "equivalent", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    }
    prompt = (
        _context(ground_truth, prediction, language)
        + "\nThese are the aligned segments where the two transcripts differ:\n"
        + listing
        + "\n\nFor each segment, decide whether the prediction is a "
        "meaning-equivalent rewording of the reference (`equivalent: true`) or a "
        "real error that changes meaning or loses information "
        "(`equivalent: false`). Judge each segment in the context of the full "
        "sentences above. Return one entry per segment index."
    )
    payload = judge.ask_json(system=_SYSTEM, prompt=prompt, schema=schema)

    verdicts = {}
    for entry in payload.get("segments", []):
        try:
            verdicts[int(entry["index"])] = (bool(entry["equivalent"]), str(entry.get("reason", "")))
        except (KeyError, TypeError, ValueError):
            continue

    # Rebuild the prediction with forgiven segments replaced by the reference,
    # then rescore it with the ordinary deterministic metric.
    corrected_tokens: list[str] = []
    segment_index = 0
    notes: list[str] = []
    forgiven = 0
    for operation in deterministic.word_alignment.operations:
        if operation.op == "equal":
            corrected_tokens.extend(operation.hyp)
            continue
        equivalent, reason = verdicts.get(segment_index, (False, "not judged — counted as an error"))
        if equivalent:
            forgiven += 1
            corrected_tokens.extend(operation.ref)
            notes.append(
                f"forgiven: \"{operation.ref_text or '(nothing)'}\" ≈ "
                f"\"{operation.hyp_text or '(nothing)'}\" ({reason})"
            )
        else:
            corrected_tokens.extend(operation.hyp)
            notes.append(
                f"error: \"{operation.ref_text or '(nothing)'}\" vs "
                f"\"{operation.hyp_text or '(nothing)'}\" ({reason})"
            )
        segment_index += 1

    corrected = score_deterministic(ground_truth, " ".join(corrected_tokens))
    summary = (
        f"{forgiven}/{len(segments)} differing segments judged meaning-equivalent. "
        + " ".join(notes)
    )
    return (
        MetricValue(
            key="llm_wer",
            value=corrected.wer,
            errors=corrected.word_errors,
            length=corrected.word_length,
            reasoning=summary,
        ),
        MetricValue(
            key="llm_cer",
            value=corrected.cer,
            errors=corrected.char_errors,
            length=corrected.char_length,
            reasoning=summary,
        ),
    )


def semantic_wer(
    *,
    judge: Judge,
    ground_truth: str,
    prediction: str,
    language: str,
    deterministic: DeterministicScores,
) -> MetricValue:
    """One call that aligns, normalises and forgives end to end."""
    schema = {
        "type": "object",
        "properties": {
            "reference_words": {"type": "integer"},
            "semantic_errors": {"type": "integer"},
            "reasoning": {"type": "string"},
        },
        "required": ["reference_words", "semantic_errors", "reasoning"],
        "additionalProperties": False,
    }
    prompt = (
        _context(ground_truth, prediction, language)
        + "\nAlign the two transcripts yourself, normalising surface forms and "
        "forgiving differences that preserve meaning. Then count `semantic_errors`: "
        "the number of reference words whose meaning is missing, wrong or "
        "contradicted in the prediction, plus any words the prediction invents "
        "that add information not in the reference. Report `reference_words` as "
        "the number of words in the reference. Explain the count in `reasoning`."
    )
    payload = judge.ask_json(system=_SYSTEM, prompt=prompt, schema=schema)

    length = int(payload.get("reference_words") or 0) or deterministic.word_length
    errors = max(0, int(payload.get("semantic_errors") or 0))
    rate = (errors / length) if length > 0 else (0.0 if errors == 0 else 1.0)
    return MetricValue(
        key="semantic_wer",
        value=rate,
        errors=errors,
        length=length,
        reasoning=str(payload.get("reasoning", "")).strip(),
    )


def semantic_match(
    *,
    judge: Judge,
    ground_truth: str,
    prediction: str,
    language: str,
    deterministic: DeterministicScores,
) -> MetricValue:
    """Binary pass/fail: does this transcription mean the same thing?"""
    schema = {
        "type": "object",
        "properties": {
            "match": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["match", "reasoning"],
        "additionalProperties": False,
    }
    prompt = (
        _context(ground_truth, prediction, language)
        + "\nDoes the prediction mean the same thing as the reference? Answer "
        "`match: true` only if someone acting on the prediction would do exactly "
        "what the reference asks, with the same names, numbers and amounts. "
        "Give the deciding reason in `reasoning`."
    )
    payload = judge.ask_json(system=_SYSTEM, prompt=prompt, schema=schema)
    matched = bool(payload.get("match"))
    return MetricValue(
        key="semantic_match",
        value=1.0 if matched else 0.0,
        reasoning=str(payload.get("reasoning", "")).strip(),
    )
