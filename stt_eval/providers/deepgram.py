"""Deepgram (pre-recorded listen endpoint)."""

from __future__ import annotations

import requests

from ..audio import wav_sample_rate
from .base import ProviderError, STTProvider, Transcription

ENDPOINT = "https://api.deepgram.com/v1/listen"


class DeepgramProvider(STTProvider):
    key = "deepgram"
    label = "Deepgram"
    credential_hint = "Deepgram API key (console.deepgram.com)"

    def __init__(self, api_key: str, *, model: str = "", timeout: float = 180.0):
        super().__init__(api_key, timeout=timeout)
        #: Empty means "pick per sample rate" — Deepgram ships a phone-call
        #: variant trained on narrowband audio.
        self.model = model

    def _model_for(self, sample_rate: int) -> str:
        if self.model:
            return self.model
        return "nova-2-phonecall" if sample_rate <= 8_000 else "nova-3"

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        model = self._model_for(wav_sample_rate(audio))
        params = {
            "model": model,
            # nova-3 takes a bare language code; the nova-2 family takes a locale.
            "language": language.split("-")[0] if model.startswith("nova-3") else language,
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
