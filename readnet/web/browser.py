"""The entry points the browser calls through Pyodide (see static/offline.js).

In the free public version there is no server: the page runs the Hindi
speech model itself with onnxruntime-web, then hands the model's per-frame
log-probabilities to these functions, which run the same scoring code as the
local server (core.py). Every function takes and returns JSON strings, which
cross the JavaScript/Python boundary cleanly.
"""

from __future__ import annotations

import json

import numpy as np

from ..acoustic import Emissions, detect_blank, greedy_decode
from . import core

LANGUAGE = "hi"  # the only model licensed for redistribution (MIT); Marathi runs in the local app
MODEL_LABEL = "Vakyansh Hindi wav2vec2, running in this browser"
ENGINE = "Hindi speech model (in this browser)"


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, tuple) else str(o))


def config(language: str = LANGUAGE) -> str:
    base = core.base_config(LANGUAGE)
    base["languages"] = [lang for lang in base["languages"] if lang["code"] == LANGUAGE]
    return _dump({
        **base,
        "engines": [
            {"key": "local", "label": ENGINE, "note": "Runs entirely on this computer: no account, no key."},
            {"key": "typed", "label": "No speech engine: type what the child said", "note": ""},
        ],
        "default_local_model": MODEL_LABEL,
        "phoneme_model": "",
        "gop_available": True,
        "public": True,
        "offline": True,
    })


def score(body_json: str, log_probs=None, frames: int = 0, vocab_size: int = 0,
          seconds: float = 0.0, vocab_json: str = "{}", elapsed: float = 0.0) -> str:
    body = json.loads(body_json)
    level = core.parse_level(body.get("level"))
    text = str(body.get("text") or "")
    threshold = body.get("gop_threshold")
    common = dict(language=LANGUAGE, level=level, text=text,
                  gop_threshold=float(threshold) if threshold is not None else -2.0,
                  audio_decides=bool(body.get("audio_decides", True)))
    if body.get("engine") == "typed" or log_probs is None:
        return _dump(core.score_reading(transcript=str(body.get("typed") or ""), engine="typed", typed=True, **common))

    buffer = log_probs.to_py() if hasattr(log_probs, "to_py") else log_probs
    lp = np.frombuffer(buffer, dtype=np.float32).reshape(int(frames), int(vocab_size)).astype(np.float64)
    vocab = json.loads(vocab_json)
    em = Emissions(lp, vocab, float(seconds) / max(int(frames), 1),
                   detect_blank(lp, vocab, vocab.get("<pad>", 0)), "|")
    return _dump(core.score_reading(
        transcript=greedy_decode(em), engine=ENGINE, seconds=float(elapsed), em=em,
        with_gop=bool(body.get("gop", True)), model_label=MODEL_LABEL, **common))


def next_step(outcomes_json: str) -> str:
    return _dump(core.next_step(json.loads(outcomes_json)))
