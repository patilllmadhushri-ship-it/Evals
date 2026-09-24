"""Scoring a reading, given its transcript and (optionally) the audio model's output.

Everything after "speech to text" lives here, in plain Python with numpy only,
so the same code runs in two places:

* the local server (server.py), which gets the transcript from a speech
  engine and the frame posteriors from a local torch model;
* the browser (static/offline.js), which runs the model itself with
  onnxruntime-web and calls this module through Pyodide — the free, keyless
  public version.

Nothing here loads a model, reads a key or touches the network.
"""

from __future__ import annotations

from collections import Counter

from .. import g2p, languages
from ..acoustic import Emissions, evidence_for_words, letter_decisions
from ..aser import Level, TaskOutcome, next_task, place
from ..confusion import hand_made_confusions
from ..pipeline import assess_task, is_letter_text
from .content import CONTENT, NEXT_STEPS, TASKS
from .mistakes import wrong_letters

LANGUAGES = ("hi", "mr")


class ScoringError(ValueError):
    """Bad input from the page: shown to the user as it is."""


def base_config(language: str) -> dict:
    return {
        "languages": [{"code": c, "name": languages.get(c).name} for c in LANGUAGES],
        "language": language,
        "content": {level.name: text for level, text in CONTENT[language].items()},
        "tasks": {level.name: copy for level, copy in TASKS.items()},
    }


def is_phoneme_model(vocab: dict[str, int]) -> bool:
    return "ə" in vocab


def evidence_from_emissions(em: Emissions, canonical: str, language: str):
    """Forced alignment + GOP of the audio against the expected text.

    With a letter model (the default), each aligned letter is later labelled
    with the sounds G2P says it stands for (घ -> /ɡʰ ə/); the unwritten /ə/
    shares its consonant's frames. With a phoneme model, G2P's sounds are
    aligned directly.
    """
    words = languages.get(language).normalize(canonical)[0].split()
    if is_phoneme_model(em.vocab):
        return evidence_for_words(em.log_probs, words, em.vocab, frame_seconds=em.frame_seconds,
                                  blank=em.blank, word_delimiter=None,
                                  units=[g2p.word_to_phonemes(w) for w in words])
    return evidence_for_words(em.log_probs, words, em.vocab, frame_seconds=em.frame_seconds,
                              blank=em.blank, word_delimiter=em.word_delimiter)


def unit_rows(word: str, units, phoneme_units: bool) -> list[dict]:
    """Each aligned unit of `word`, labelled with the sounds it stands for.

    For a letter model, G2P's per-letter sounds: घ -> "ɡʰ ə". Letters come
    back in word order (minus any the model's vocabulary lacks), so they are
    matched to their positions in the word by walking it.
    """
    sounds = g2p.letter_sounds(word) if not phoneme_units else []
    rows, pos = [], 0
    for u in units:
        label = u.unit
        if not phoneme_units:
            while pos < len(word) and word[pos] != u.unit:
                pos += 1
            if pos < len(word):
                label = sounds[pos]
                pos += 1
        rows.append({"letter": None if phoneme_units else u.unit, "sounds": label,
                     "start_s": round(u.start_s, 2), "end_s": round(u.end_s, 2), "gop": round(u.llr, 1),
                     "heard": u.heard})
    return rows


def letter_forms(em: Emissions):
    """How a spoken letter may sound, in the model's units."""
    if not is_phoneme_model(em.vocab):
        return None  # letter model: the letter, or letter + ा

    def forms(letter: str):
        sounds = g2p.word_to_phonemes(letter)
        if len(sounds) == 2 and sounds[1] == g2p.SCHWA:  # a consonant: "ka", "kaa", or just /k/
            return [sounds, [sounds[0], "aː"], [sounds[0]]]
        return [sounds]

    return forms


def outcome_dict(outcome: TaskOutcome) -> dict:
    return {
        "level": outcome.level.name, "passed": outcome.passed, "mistakes": outcome.mistakes,
        "correct": outcome.correct, "total": outcome.total, "wpm": outcome.wpm,
        "longest_pause": outcome.longest_pause, "reasons": outcome.reasons,
    }


def parse_level(value) -> Level:
    try:
        return Level[str(value or "PARAGRAPH").upper()]
    except KeyError as error:
        raise ScoringError(f"Unknown level: {value}") from error


def score_reading(
    *,
    language: str,
    level: Level,
    text: str,
    transcript: str,
    engine: str,
    seconds: float = 0.0,
    em: Emissions | None = None,
    notes: list[str] | None = None,
    with_gop: bool = True,
    gop_threshold: float | None = -2.0,
    audio_decides: bool = True,
    model_label: str = "",
    typed: bool = False,
) -> dict:
    """The full result for one reading, as the page renders it."""
    notes = list(notes or [])
    text = text.strip()
    if not text:
        raise ScoringError("The text shown to the child is empty.")

    evidence = None
    if em is not None and with_gop:
        try:
            evidence = evidence_from_emissions(em, text, language)
        except ValueError as error:  # recording too short for the text
            notes.append(f"GOP skipped: {error}")

    # Letters: an open engine cannot hear aspiration in a half-second clip
    # (it wrote का का गा गा for क ख ग घ). When the audio model ran, each letter
    # is decided by a closed-set check instead: the expected letter against
    # its known confusions only. The engine's text is kept to compare.
    profile = languages.get(language)
    letters = profile.normalize(text)[0].split()
    letter_check = None
    engine_transcript = transcript
    if em is not None and is_letter_text(" ".join(letters)):
        try:
            decisions = letter_decisions(em.log_probs, letters, lambda l: hand_made_confusions(l, profile.tables),
                                         em.vocab, em.blank,
                                         None if is_phoneme_model(em.vocab) else em.word_delimiter,
                                         forms=letter_forms(em))
            letter_check = [{"letter": d.expected, "heard": d.best, "margin": round(d.margin, 1),
                             "candidates": sorted(d.scores, key=d.scores.get, reverse=True)} for d in decisions]
            transcript = " ".join(d.best for d in decisions)
            notes.append("Letters were judged by the acoustic letter check (each letter against the letters it "
                         "is usually confused with); the engine's own transcript is shown for comparison.")
        except ValueError as error:
            notes.append(f"Letter check skipped: {error}")

    outcome, items = assess_task(
        level, [(text, transcript)], language=language, acoustic=[evidence] if evidence else None,
        gop_threshold=float(gop_threshold) if evidence and gop_threshold is not None else None,
        audio_decides=audio_decides,
    )
    item = items[0]
    phoneme_units = em is not None and is_phoneme_model(em.vocab)
    ops = []
    for op in item.result.ops:
        entry = {k: v for k, v in op.__dict__.items()}
        entry["subtypes"] = list(op.subtypes)
        if evidence and op.ref_index is not None and op.ref_index < len(evidence.words):
            w = evidence.words[op.ref_index]
            entry.update(start_s=w.start_s, end_s=w.end_s, gop=w.llr,
                         units=unit_rows(op.ref, w.units, phoneme_units))
        ops.append(entry)

    if evidence is not None and phoneme_units:
        notes.append("Phoneme model: this multilingual model separated right from wrong Hindi reading poorly "
                     "in testing. The default Hindi model is the more reliable judge.")
    kinds = Counter(op["kind"] for op in ops)
    gop_scored = [op for op in ops if op.get("gop") is not None]
    if evidence is not None:
        gop_detail = (f"Ran on {len(gop_scored)} words ({sum(len(op.get('units') or []) for op in gop_scored)} "
                      f"sounds) with {model_label}. "
                      f"{sum(1 for op in gop_scored if op['gop'] < float(gop_threshold or 0))} below the threshold.")
    elif typed:
        gop_detail = "Not run: typed transcripts have no audio."
    elif not with_gop:
        gop_detail = "Off (turned off in Setup)."
    else:
        gop_detail = next((n for n in notes if n.startswith("GOP")), "Not run.")
    pipeline = [
        {"stage": "Speech to text", "detail": f"{engine}: {engine_transcript or '(nothing heard)'}"},
        {"stage": "Normalise the text on screen", "rules": item.canonical_trace.steps,
         "result": item.canonical_normalized},
        {"stage": "Normalise the text heard", "rules": item.transcript_trace.steps,
         "result": item.transcript_normalized},
        {"stage": "Align word by word", "detail": f"{kinds['match']} matched, {kinds['substitution']} substituted, "
                                                 f"{kinds['deletion']} not read, {kinds['insertion']} extra"},
        {"stage": "Forgiveness rules", "detail": ", ".join(f"{r} ×{n}" for r, n in item.result.rules_applied.items())
                                                  or "None applied."},
        {"stage": "Pronunciation (GOP)", "detail": gop_detail},
    ]
    if letter_check is not None:
        pipeline.append({"stage": "Letter check", "detail": ", ".join(
            f"{c['letter']}→{c['heard']}" for c in letter_check)})
    pipeline.append({"stage": "Verdict", "detail": "; ".join(outcome.reasons)})

    response = {
        "outcome": outcome_dict(outcome),
        "transcript": transcript,
        "engine_transcript": engine_transcript,
        "letter_check": letter_check,
        "seconds": round(seconds, 2),
        "canonical_normalized": item.canonical_normalized,
        "transcript_normalized": item.transcript_normalized,
        "changes": [{"rule": r, "before": b, "after": a} for r, b, a in item.transcript_trace.steps],
        "acoustic_doubts": item.acoustic_doubts,
        "gop_ran": evidence is not None,
        "notes": notes,
        "pipeline": pipeline,
        "mistake_profile": dict(item.result.mistake_profile),
        "ops": ops,
    }
    threshold = float(gop_threshold) if gop_threshold is not None else -2.0
    response["wrong_letters"] = wrong_letters(response, language, threshold)
    return response


def next_step(outcomes_body: dict) -> dict:
    """The next ASER task, or the final placement."""
    outcomes: dict[Level, TaskOutcome] = {}
    for name, o in (outcomes_body or {}).items():
        level = Level[name]
        outcomes[level] = TaskOutcome(level, bool(o["passed"]), int(o.get("mistakes", 0)),
                                      int(o.get("correct", 0)), int(o.get("total", 0)),
                                      reasons=list(o.get("reasons", [])))
    upcoming = next_task(outcomes)
    if upcoming is not None:
        return {"next": upcoming.name, "placement": None}
    placement = place(outcomes)
    return {"next": None, "placement": {
        "level": placement.level.name, "label": placement.level.label,
        "path": placement.path, "next_step": NEXT_STEPS[placement.level],
    }}
