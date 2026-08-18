"""Google Cloud Speech-to-Text (v1 synchronous recognize, API-key auth)."""

from __future__ import annotations

import base64

import requests

from ..config import TARGET_SAMPLE_RATE
from .base import ProviderError, STTProvider, Transcription

ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"


class GoogleProvider(STTProvider):
    key = "google"
    label = "Google Speech-to-Text"
    credential_hint = "Google Cloud API key with the Speech-to-Text API enabled"

    def __init__(self, api_key: str, *, model: str = "latest_long", timeout: float = 180.0):
        super().__init__(api_key, timeout=timeout)
        self.model = model

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        body = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": TARGET_SAMPLE_RATE,
                "audioChannelCount": 1,
                "languageCode": language,
                "model": self.model,
                "enableAutomaticPunctuation": True,
            },
            # The WAV header is stripped by the service; sending the container is fine.
            "audio": {"content": base64.b64encode(audio).decode("ascii")},
        }

        def call():
            return requests.post(
                ENDPOINT, params={"key": self.api_key}, json=body, timeout=self.timeout
            )

        try:
            response, latency = self._timed(call)
        except requests.RequestException as exc:
            raise ProviderError(f"Google request failed: {exc}") from exc

        self._raise_for_status(response, "Google Speech-to-Text")
        payload = response.json()
        results = payload.get("results") or []
        parts = [
            result["alternatives"][0].get("transcript", "")
            for result in results
            if result.get("alternatives")
        ]
        return Transcription(text=" ".join(parts).strip(), latency_seconds=latency, raw=payload)
