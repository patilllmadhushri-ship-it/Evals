"""Google Cloud Speech-to-Text (v1 synchronous recognize, API-key auth)."""

from __future__ import annotations

import base64

import requests

from ..audio import wav_sample_rate
from .base import ProviderError, STTProvider, Transcription

ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"


class GoogleProvider(STTProvider):
    key = "google"
    label = "Google Speech-to-Text"
    credential_hint = "Google Cloud API key with the Speech-to-Text API enabled"

    def __init__(self, api_key: str, *, model: str = "", timeout: float = 180.0):
        super().__init__(api_key, timeout=timeout)
        #: Empty means "pick per sample rate" — Google ships a dedicated
        #: narrowband model for telephony audio.
        self.model = model

    def _model_for(self, sample_rate: int) -> str:
        if self.model:
            return self.model
        return "phone_call" if sample_rate <= 8_000 else "latest_long"

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        rate = wav_sample_rate(audio)
        body = {
            "config": {
                "encoding": "LINEAR16",
                # Read from the payload rather than assumed, so the declared
                # rate can never disagree with the bytes being sent.
                "sampleRateHertz": rate,
                "audioChannelCount": 1,
                "languageCode": language,
                "model": self._model_for(rate),
                "useEnhanced": True,
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
