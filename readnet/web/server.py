"""Akshar test bench: a plain HTTP server and one page. No web framework.

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
import os
import time
from datetime import datetime
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from stt_eval import audio as stt_audio
from stt_eval import env, providers
from stt_eval.providers import ProviderError

from . import core
from ..acoustic import DEFAULT_MODELS, PHONEME_MODEL

STATIC = Path(__file__).with_name("static")
DISPUTE_LOG = Path(".stt_eval_runs") / "readnet_disputes.csv"
LOCALES = {"hi": "hi-IN", "mr": "mr-IN"}
#: Engines that reject long audio in one request: split it, at a quiet moment.
MAX_SECONDS = {"sarvam": 29.0}
TYPED, LOCAL = "typed", "local"
#: A public demo: shows a notice that everything runs on open models, no key.
PUBLIC = os.environ.get("READNET_PUBLIC", "") not in ("", "0", "false")
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
    local = {"key": LOCAL, "label": "Local model (wav2vec2, runs on this machine)", "note": local_model_problem()}
    if any(e["key"] not in ("mock",) for e in out):
        out.append(local)
    else:
        out.insert(0, local)  # no cloud keys (a public demo): the local model is the engine
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


def gop_evidence(wav_bytes: bytes, canonical: str, language: str, model_id: str):
    """Run the local model on the recording; alignment and GOP happen in core."""
    em = load_local_model(model_id or DEFAULT_MODELS[language]).emissions(wav_bytes)
    return core.evidence_from_emissions(em, canonical, language), em


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

        # The transcript comes from the language's letter model: reuse the GOP
        # pass when it ran on that same model.
        letters_model = DEFAULT_MODELS[language]
        letters_em = em if em is not None and (model_id or letters_model) == letters_model else             load_local_model(letters_model).emissions(clip.wav_bytes)
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


def api_config(query: dict) -> dict:
    language = _language(query.get("language", ["hi"])[0])
    return {
        **core.base_config(language),
        "engines": engines(language),
        "default_local_model": DEFAULT_MODELS[language],
        "phoneme_model": PHONEME_MODEL,
        "gop_available": local_model_available(),
        "public": PUBLIC,
    }


def api_score(body: dict) -> dict:
    language = _language(body.get("language"))
    try:
        level = core.parse_level(body.get("level"))
    except core.ScoringError as error:
        raise ApiError(str(error)) from error
    text = str(body.get("text") or "").strip()
    if not text:
        raise ApiError("The text shown to the child is empty.")
    engine = str(body.get("engine") or TYPED)
    with_gop = bool(body.get("gop", True))
    model_id = str(body.get("model_id") or "")

    notes: list[str] = []
    em = None
    if engine == TYPED:
        transcript, seconds = str(body.get("typed") or ""), 0.0
    else:
        audio = base64.b64decode(body["audio_b64"]) if body.get("audio_b64") else None
        try:
            transcript, seconds, _, notes, em = transcribe(
                engine, audio, str(body.get("filename") or ""), text, language, model_id, with_gop,
            )
        except (ProviderError, stt_audio.AudioError, OSError, ImportError, ValueError) as error:
            raise ApiError(f"Could not transcribe: {error}", HTTPStatus.BAD_GATEWAY) from error

    gop_threshold = body.get("gop_threshold")
    return core.score_reading(
        language=language, level=level, text=text, transcript=transcript, engine=engine, seconds=seconds,
        em=em, notes=notes, with_gop=with_gop,
        gop_threshold=float(gop_threshold) if gop_threshold is not None else -2.0,
        audio_decides=bool(body.get("audio_decides", True)),
        model_label=model_id or DEFAULT_MODELS[language], typed=engine == TYPED,
    )


def api_next(body: dict) -> dict:
    return core.next_step(body.get("outcomes") or {})


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
    server_version = "Akshar"

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
    if not PUBLIC:
        env.load()  # a public demo never reads .env: its keys would be spendable by anyone with the link
    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="readnet.web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--public", action="store_true",
                        help="public demo: ignore .env and any API keys, use only the local model")
    args = parser.parse_args(argv)
    if args.public:
        global PUBLIC
        PUBLIC = True
        for variable in env.PROVIDER_ENV_VARS.values():
            os.environ.pop(variable, None)
    server = make_server(args.host, args.port)
    print(f"Akshar test bench on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
