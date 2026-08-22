"""ElevenLabs Scribe speech-to-text."""

from __future__ import annotations

import requests

from ..config import LANGUAGES
from .base import ProviderError, STTProvider, Transcription

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

# Scribe covers 99 languages — every code in the app's list.
SUPPORTED = frozenset(lang.code for lang in LANGUAGES)


class ElevenLabsProvider(STTProvider):
    key = "elevenlabs"
    label = "ElevenLabs Scribe"
    credential_hint = "ElevenLabs API key (elevenlabs.io → Profile → API Keys)"
    supported_languages = SUPPORTED

    def __init__(self, api_key: str, *, model: str = "scribe_v1", timeout: float = 180.0):
        super().__init__(api_key, timeout=timeout)
        self.model = model

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        files = {"file": ("clip.wav", audio, "audio/wav")}
        data = {
            "model_id": self.model,
            # Scribe takes a bare ISO-639 code, not a locale.
            "language_code": language.split("-")[0],
            # Diarisation and audio-event tagging would add text the ground
            # truth does not contain, which would show up as insertions.
            "diarize": "false",
            "tag_audio_events": "false",
        }
        headers = {"xi-api-key": self.api_key}

        def call():
            return requests.post(
                ENDPOINT, headers=headers, files=files, data=data, timeout=self.timeout
            )

        try:
            response, latency = self._timed(call)
        except requests.RequestException as exc:
            raise ProviderError(f"ElevenLabs request failed: {exc}") from exc

        self._raise_for_status(response, "ElevenLabs Scribe")
        payload = response.json()
        text = payload.get("text")
        if text is None:
            raise ProviderError(
                f"ElevenLabs response had no 'text' field: {str(payload)[:300]}",
                retryable=False,
            )
        return Transcription(text=str(text).strip(), latency_seconds=latency, raw=payload)
