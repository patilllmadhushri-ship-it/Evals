"""ReadNet test bench: a plain HTTP server and one page. No web framework.

    py -m readnet.web                 # http://127.0.0.1:8600
    py -m readnet.web --port 8502

The page (static/) records the child in the browser as WAV and calls a small
JSON API:

    GET  /api/config?language=hi   engines, texts and task copy
    POST /api/score                transcribe one reading and mark every word
    POST /api/next                 the next ASER task, or the final placement
    POST /api/dispute              save an assessor disagreement as a field case

Audio is used for the one request and never written to disk. Disputes go to
.stt_eval_runs/ (git-ignored) because they contain children's readings.
"""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import mimetypes
import time
from collections import Counter
from datetime import datetime
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from stt_eval import audio as stt_audio
from stt_eval import env, providers
from stt_eval.providers import ProviderError

from .. import languages
from .. import g2p
from ..acoustic import DEFAULT_MODELS, PHONEME_MODEL
from ..aser import Level, TaskOutcome, next_task, place
from ..confusion import hand_made_confusions
from ..pipeline import assess_task, is_letter_text
from .content import CONTENT, NEXT_STEPS, TASKS

STATIC = Path(__file__).with_name("static")
DISPUTE_LOG = Path(".stt_eval_runs") / "readnet_disputes.csv"
LOCALES = {"hi": "hi-IN", "mr": "mr-IN"}
#: Engines that reject long audio in one request: split it, at a quiet moment.
MAX_SECONDS = {"sarvam": 29.0}
TYPED, LOCAL = "typed", "local"
#: Indic-first engines first: Sarvam is built for Hindi and Marathi.
PREFERRED = ["sarvam", "google", "deepgram", "elevenlabs", "openai", "mock"]


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


# -- engines ------------------------------------------------------------------------


@lru_cache(maxsize=1)
def local_model_problem() -> str:
    """Why the local model cannot run here, or "" when it can.

    Installed is not the same as loadable: on Windows, torch also needs the
    Microsoft Visual C++ Redistributable, and fails at import without it.
    """
    missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
    if missing:
        return f"Needs {' and '.join(missing)}: py -m pip install torch transformers, then restart."
    try:
        import torch  # noqa: F401
    except OSError:
        return ("torch is installed but cannot load: install the Microsoft Visual C++ Redistributable "
                "(https://aka.ms/vs/17/release/vc_redist.x64.exe), then restart.")
    except ImportError as error:
        return f"torch cannot load: {error}"
    return ""


def local_model_available() -> bool:
    return not local_model_problem()


def engines(language: str) -> list[dict]:
    available = providers.providers_for_language(LOCALES[language])
    out: list[dict] = []
    for key in sorted(available, key=lambda k: PREFERRED.index(k) if k in PREFERRED else len(PREFERRED)):
        if key == "mock":
            out.append({"key": key, "label": "Mock (simulated reading)",
                        "note": "Drops words at random from the screen text. For trying the flow without a microphone."})
        elif env.provider_key(key):
            out.append({"key": key, "label": f"{providers.provider_label(key)} (cloud)", "note": ""})
    out.append({"key": LOCAL, "label": "Local model (wav2vec2, runs on this machine)",
                "note": local_model_problem()})
    out.append({"key": TYPED, "label": "No speech engine: type what the child said", "note": ""})
    return out


@lru_cache(maxsize=2)
def load_local_model(model_id: str):
    from ..acoustic import Wav2Vec2Emissions

    return Wav2Vec2Emissions(model_id)


def split_wav(wav_bytes: bytes, max_seconds: float) -> list[bytes]:
    """Cut mono 16-bit WAV into pieces under `max_seconds`, each cut at the
    quietest 100 ms in the last third of the window, so no word is sliced."""
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    limit, step = int(max_seconds * rate), int(0.1 * rate)
    pieces, start = [], 0
    while len(samples) - start > limit:
        lo, hi = start + int(limit * 0.66), start + limit - step
        energies = [np.abs(samples[i : i + step].astype(np.int32)).mean() for i in range(lo, hi, step)]
        cut = lo + step * int(np.argmin(energies)) + step // 2
        pieces.append(samples[start:cut])
        start = cut
    pieces.append(samples[start:])
    out = []
    for piece in pieces:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(piece.tobytes())
        out.append(buf.getvalue())
    return out


def is_phoneme_model(vocab: dict[str, int]) -> bool:
    return "ə" in vocab


def gop_evidence(wav_bytes: bytes, canonical: str, language: str, model_id: str):
    """Forced alignment + GOP of the audio against the expected text.

    Default: the Hindi (or Marathi) letter model, which was trained on
    thousands of hours of the language and separates right from wrong
    reading. Each aligned letter is labelled with the sounds G2P says it
    stands for (घ -> /ɡʰ ə/), so the result reads sound by sound; the unwritten
    /ə/ shares its consonant's frames and score, because this model hears
    them as one unit.

    A phoneme model (IPA output, e.g. PHONEME_MODEL) aligns G2P's sounds
    directly, /ə/ included — but the multilingual one tested here separated
    right from wrong Hindi reading poorly, so it is opt-in.
    """
    from ..acoustic import evidence_for_words

    em = load_local_model(model_id or DEFAULT_MODELS[language]).emissions(wav_bytes)
    words = languages.get(language).normalize(canonical)[0].split()
    if is_phoneme_model(em.vocab):
        evidence = evidence_for_words(em.log_probs, words, em.vocab, frame_seconds=em.frame_seconds,
                                      blank=em.blank, word_delimiter=None,
                                      units=[g2p.word_to_phonemes(w) for w in words])
    else:
        evidence = evidence_for_words(em.log_probs, words, em.vocab, frame_seconds=em.frame_seconds,
                                      blank=em.blank, word_delimiter=em.word_delimiter)
    return evidence, em


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


def letter_forms(em):
    """How a spoken letter may sound, in the model's units."""
    if not is_phoneme_model(em.vocab):
        return None  # letter model: the letter, or letter + ा
    def forms(letter: str):
        sounds = g2p.word_to_phonemes(letter)
        if len(sounds) == 2 and sounds[1] == g2p.SCHWA:  # a consonant: "ka", "kaa", or just /k/
            return [sounds, [sounds[0], "aː"], [sounds[0]]]
        return [sounds]
    return forms


def transcribe(engine: str, audio: bytes | None, filename: str, canonical: str, language: str,
               model_id: str, with_gop: bool):
    """Returns (transcript, seconds, acoustic evidence or None, notes, emissions or None).

    GOP runs on every engine that has audio when `with_gop` is set: the chosen
    engine writes the transcript, and the local model scores each expected
    word's pronunciation against the same recording.
    """
    started = time.perf_counter()
    notes: list[str] = []
    if engine == "mock":
        mock = providers.build("mock", "")
        mock._ground_truth_hint = canonical
        text = mock.transcribe(audio or canonical.encode("utf-8"), LOCALES[language]).text
        if with_gop:
            notes.append("GOP needs a real recording; Mock has none.")
        return text, time.perf_counter() - started, None, notes, None
    if not audio:
        raise ApiError("Record or upload the child's reading first.")

    clip = stt_audio.normalize(audio, filename or "recording.wav", sample_rate=16000)
    evidence = em = None
    if with_gop or engine == LOCAL:
        problem = local_model_problem()
        if problem:
            if engine == LOCAL:
                raise ApiError(problem)
            notes.append(f"GOP skipped: {problem}")
        else:
            try:
                evidence, em = gop_evidence(clip.wav_bytes, canonical, language, model_id)
            except ValueError as error:  # recording too short for the text
                notes.append(f"GOP skipped: {error}")
            except OSError as error:
                raise ApiError(model_error(model_id or DEFAULT_MODELS[language], error)) from error

    if engine == LOCAL:
        from ..acoustic import greedy_decode

        # The transcript comes from the letter model; GOP above from the phoneme model.
        letters_em = load_local_model(DEFAULT_MODELS[language]).emissions(clip.wav_bytes)
        return greedy_decode(letters_em), time.perf_counter() - started, evidence, notes, em

    if engine not in providers.PROVIDER_CLASSES or not env.provider_key(engine):
        raise ApiError(f"No key configured for {engine}.")
    provider = providers.build(engine, env.provider_key(engine))
    limit = MAX_SECONDS.get(engine)
    pieces = split_wav(clip.wav_bytes, limit) if limit and clip.duration_seconds > limit else [clip.wav_bytes]
    if len(pieces) > 1:
        notes.append(f"Recording is {clip.duration_seconds:.0f}s; sent to {engine} in {len(pieces)} parts.")
    text = " ".join(provider.transcribe(piece, LOCALES[language]).text for piece in pieces)
    if not text.strip():
        notes.append("The engine returned no words. Check the recording has speech in it.")
    return text, time.perf_counter() - started, evidence, notes, em


def model_error(model_id: str, error: Exception) -> str:
    message = str(error)
    if "gated" in message or "401" in message or "403" in message:
        return (f"The model {model_id} is gated: log in at huggingface.co, open the model page and accept "
                f"its terms, with HUGGINGFACE_TOKEN in .env. Or clear the model field to use the ungated "
                f"default ({DEFAULT_MODELS.get('hi')}).")
    return f"Could not load the local model {model_id}: {message[:300]}"


# -- API handlers -----------------------------------------------------------------------


def _language(value) -> str:
    code = str(value or "hi")
    if code not in LOCALES:
        raise ApiError(f"Unsupported language: {code}")
    return code


def _outcome_dict(outcome: TaskOutcome) -> dict:
    return {
        "level": outcome.level.name, "passed": outcome.passed, "mistakes": outcome.mistakes,
        "correct": outcome.correct, "total": outcome.total, "wpm": outcome.wpm,
        "longest_pause": outcome.longest_pause, "reasons": outcome.reasons,
    }


def api_config(query: dict) -> dict:
    language = _language(query.get("language", ["hi"])[0])
    return {
        "languages": [{"code": c, "name": languages.get(c).name} for c in LOCALES],
        "language": language,
        "engines": engines(language),
        "default_local_model": DEFAULT_MODELS[language],
        "phoneme_model": PHONEME_MODEL,
        "gop_available": local_model_available(),
        "content": {level.name: text for level, text in CONTENT[language].items()},
        "tasks": {level.name: copy for level, copy in TASKS.items()},
    }


def api_score(body: dict) -> dict:
    language = _language(body.get("language"))
    try:
        level = Level[str(body.get("level", "PARAGRAPH")).upper()]
    except KeyError as error:
        raise ApiError(f"Unknown level: {body.get('level')}") from error
    text = str(body.get("text") or "").strip()
    if not text:
        raise ApiError("The text shown to the child is empty.")
    engine = str(body.get("engine") or TYPED)
    gop_threshold = body.get("gop_threshold")
    with_gop = bool(body.get("gop", True))
    notes: list[str] = []

    em = None
    if engine == TYPED:
        transcript, seconds, evidence = str(body.get("typed") or ""), 0.0, None
    else:
        audio = base64.b64decode(body["audio_b64"]) if body.get("audio_b64") else None
        try:
            transcript, seconds, evidence, notes, em = transcribe(
                engine, audio, str(body.get("filename") or ""), text, language, str(body.get("model_id") or ""),
                with_gop,
            )
        except (ProviderError, stt_audio.AudioError, OSError, ImportError, ValueError) as error:
            raise ApiError(f"Could not transcribe: {error}", HTTPStatus.BAD_GATEWAY) from error

    # Letters: an open engine cannot hear aspiration in a half-second clip
    # (it wrote का का गा गा for क ख ग घ). When the audio model is loaded, each
    # letter is decided by a closed-set check instead: the expected letter
    # against its known confusions only. The engine's text is kept to compare.
    profile = languages.get(language)
    letters = profile.normalize(text)[0].split()
    letter_check = None
    engine_transcript = transcript
    if em is not None and is_letter_text(" ".join(letters)):
        from ..acoustic import letter_decisions

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
        audio_decides=bool(body.get("audio_decides", True)),
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
                      f"sounds) with {body.get('model_id') or DEFAULT_MODELS[language]}. "
                      f"{sum(1 for op in gop_scored if op['gop'] < float(gop_threshold or 0))} below the threshold.")
    elif engine == TYPED:
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

    return {
        "outcome": _outcome_dict(outcome),
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


def api_next(body: dict) -> dict:
    outcomes: dict[Level, TaskOutcome] = {}
    for name, o in (body.get("outcomes") or {}).items():
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


def api_dispute(body: dict) -> dict:
    DISPUTE_LOG.parent.mkdir(exist_ok=True)
    new = not DISPUTE_LOG.exists()
    with DISPUTE_LOG.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["saved_at", "language", "level", "canonical", "transcript",
                             "system_mistakes", "human_mistakes", "note"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), _language(body.get("language")),
                         str(body.get("level", "")).lower(), body.get("text", ""), body.get("transcript", ""),
                         body.get("system_mistakes", ""), body.get("human_mistakes", ""), body.get("note", "")])
    return {"saved": str(DISPUTE_LOG)}


# -- HTTP -----------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "ReadNet"

    def log_message(self, fmt, *args):  # quieter: method and path only
        print(f"{self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/config":
            return self._handle(lambda: api_config(parse_qs(url.query)))
        name = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
        file = (STATIC / name).resolve()
        if STATIC.resolve() not in file.parents or not file.is_file():
            return self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
        kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind.endswith("javascript"):
            kind += "; charset=utf-8"
        self._send(HTTPStatus.OK, file.read_bytes(), kind)

    def do_POST(self):
        routes = {"/api/score": api_score, "/api/next": api_next, "/api/dispute": api_dispute}
        handler = routes.get(urlparse(self.path).path)
        if handler is None:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "Body must be JSON"})
        self._handle(lambda: handler(body))

    def _handle(self, fn) -> None:
        try:
            self._json(HTTPStatus.OK, fn())
        except ApiError as error:
            self._json(error.status, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - report, don't drop the connection
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})


def make_server(host: str = "127.0.0.1", port: int = 8600) -> ThreadingHTTPServer:
    env.load()
    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="readnet.web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print(f"ReadNet test bench on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
