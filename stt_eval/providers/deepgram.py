"""Deepgram (pre-recorded listen endpoint)."""

from __future__ import annotations

import requests

from .base import ProviderError, STTProvider, Transcription

ENDPOINT = "https://api.deepgram.com/v1/listen"


class DeepgramProvider(STTProvider):
    key = "deepgram"
    label = "Deepgram"
    credential_hint = "Deepgram API key (console.deepgram.com)"

    def __init__(self, api_key: str, *, model: str = "nova-3", timeout: float = 180.0):
        super().__init__(api_key, timeout=timeout)
        self.model = model

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        params = {
            "model": self.model,
            "language": language.split("-")[0] if self.model == "nova-3" else language,
            "smart_format": "true",
            "punctuate": "true",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }

        def call():
            return requests.post(
                ENDPOINT, params=params, headers=headers, data=audio, timeout=self.timeout
            )

        try:
            response, latency = self._timed(call)
        except requests.RequestException as exc:
            raise ProviderError(f"Deepgram request failed: {exc}") from exc

        self._raise_for_status(response, "Deepgram")
        payload = response.json()
        try:
            alternatives = payload["results"]["channels"][0]["alternatives"]
            text = alternatives[0]["transcript"] if alternatives else ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Deepgram response had an unexpected shape: {str(payload)[:300]}",
                retryable=False,
            ) from exc

        return Transcription(text=text.strip(), latency_seconds=latency, raw=payload)
