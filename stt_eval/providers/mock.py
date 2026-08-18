"""An offline provider for trying the app end to end without any API key.

It degrades the ground truth in ways real systems do — number words become
digits, filler words vanish, occasional words are dropped — so the metrics
ladder has something realistic to disagree about.
"""

from __future__ import annotations

import random
import time

from .base import STTProvider, Transcription

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "twenty": "20", "fifty": "50", "hundred": "100",
    "thousand": "1000",
}
_FILLERS = {"um", "uh", "like", "you", "know", "actually", "just"}


class MockProvider(STTProvider):
    key = "mock"
    label = "Mock (offline demo)"
    credential_hint = "No key required — degrades the ground truth locally."

    def __init__(self, api_key: str = "", *, error_rate: float = 0.12, timeout: float = 0.0):
        super().__init__(api_key, timeout=timeout)
        self.error_rate = error_rate

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        # Deterministic per clip so re-runs are reproducible.
        rng = random.Random(len(audio))
        started = time.perf_counter()
        source = getattr(self, "_ground_truth_hint", "") or ""
        tokens = source.split()
        output: list[str] = []
        for token in tokens:
            bare = token.strip(".,!?").lower()
            roll = rng.random()
            if bare in _NUMBER_WORDS and roll < 0.7:
                output.append(_NUMBER_WORDS[bare])
            elif bare in _FILLERS and roll < 0.5:
                continue
            elif roll < self.error_rate:
                continue  # dropped word
            else:
                output.append(token)
        time.sleep(0.05)
        return Transcription(
            text=" ".join(output),
            latency_seconds=time.perf_counter() - started,
            raw={"mock": True},
        )
