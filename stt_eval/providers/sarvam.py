"""Sarvam AI speech-to-text (Indic languages)."""

from __future__ import annotations

import requests

from .base import ProviderError, STTProvider, Transcription

ENDPOINT = "https://api.sarvam.ai/speech-to-text"

# Indic-only by design — this is what Sarvam is built for.
SUPPORTED = frozenset(
    {
        "en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
        "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
    }
)


class SarvamProvider(STTProvider):
    key = "sarvam"
    label = "Sarvam AI"
    credential_hint = "Sarvam API subscription key (dashboard.sarvam.ai)"
    supported_languages = SUPPORTED

    def __init__(self, api_key: str, *, model: str = "saarika:v2", timeout: float = 180.0):
        super().__init__(api_key, timeout=timeout)
        self.model = model

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        files = {"file": ("clip.wav", audio, "audio/wav")}
        data = {"model": self.model, "language_code": language}
        headers = {"api-subscription-key": self.api_key}

        def call():
            return requests.post(
                ENDPOINT, headers=headers, files=files, data=data, timeout=self.timeout
            )

        try:
            response, latency = self._timed(call)
        except requests.RequestException as exc:
            raise ProviderError(f"Sarvam request failed: {exc}") from exc

        self._raise_for_status(response, "Sarvam AI")
        payload = response.json()
        text = payload.get("transcript")
        if text is None:
            raise ProviderError(
                f"Sarvam response had no 'transcript' field: {str(payload)[:300]}",
                retryable=False,
            )
        return Transcription(text=str(text).strip(), latency_seconds=latency, raw=payload)
