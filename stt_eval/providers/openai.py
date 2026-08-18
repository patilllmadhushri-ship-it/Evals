"""OpenAI audio transcriptions."""

from __future__ import annotations

import requests

from ..config import LANGUAGES
from .base import ProviderError, STTProvider, Transcription

ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"


# Whisper-family models cover ~99 languages; every code in the app's list is
# supported except Odia, which Whisper does not recognise. Declared as a class
# attribute so the registry's language filter sees it without instantiating.
UNSUPPORTED = frozenset({"od-IN"})
SUPPORTED = frozenset(lang.code for lang in LANGUAGES) - UNSUPPORTED


class OpenAIProvider(STTProvider):
    key = "openai"
    label = "OpenAI"
    credential_hint = "OpenAI API key (platform.openai.com)"
    supported_languages = SUPPORTED

    def __init__(self, api_key: str, *, model: str = "gpt-4o-transcribe", timeout: float = 180.0):
        super().__init__(api_key, timeout=timeout)
        self.model = model

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        files = {"file": ("clip.wav", audio, "audio/wav")}
        data = {
            "model": self.model,
            # The API takes a bare ISO-639-1 code, not a locale.
            "language": language.split("-")[0],
            "response_format": "json",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        def call():
            return requests.post(
                ENDPOINT, headers=headers, files=files, data=data, timeout=self.timeout
            )

        try:
            response, latency = self._timed(call)
        except requests.RequestException as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        self._raise_for_status(response, "OpenAI")
        payload = response.json()
        text = payload.get("text")
        if text is None:
            raise ProviderError(
                f"OpenAI response had no 'text' field: {str(payload)[:300]}", retryable=False
            )
        return Transcription(text=str(text).strip(), latency_seconds=latency, raw=payload)
