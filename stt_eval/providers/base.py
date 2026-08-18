"""The one interface every STT provider implements.

Given audio bytes and a language, return a transcript. Adding a provider means
adding one subclass and one registry entry — it must not require touching the
scoring or UI layers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """A provider call failed. Retried a few times, then recorded as a failure
    for that (clip, provider) pair only — the run continues for everything else."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Transcription:
    text: str
    latency_seconds: float
    raw: dict | None = None


class STTProvider:
    #: Registry key, also used in results tables and the rate table.
    key: str = ""
    #: Human-readable name for the UI.
    label: str = ""
    #: Where the user gets a key, shown next to the credential field.
    credential_hint: str = ""
    #: Language codes this provider supports; empty means "all listed languages".
    supported_languages: frozenset[str] = frozenset()

    def __init__(self, api_key: str, *, timeout: float = 180.0):
        self.api_key = api_key
        self.timeout = timeout

    #: Canonical app language code -> this provider's code, where they differ.
    #: Anything absent is passed through unchanged.
    language_code_overrides: dict[str, str] = {}

    def supports(self, language: str) -> bool:
        return not self.supported_languages or language in self.supported_languages

    def language_code(self, language: str) -> str:
        """Translate the app's canonical code into the provider's own code.

        The app keeps one canonical code per language so results are comparable
        across providers; each provider maps it to whatever its API expects.
        """
        return self.language_code_overrides.get(language, language)

    def transcribe(self, audio: bytes, language: str) -> Transcription:
        """Transcribe mono 16 kHz 16-bit PCM WAV `audio`."""
        raise NotImplementedError

    # -- helpers shared by the HTTP-backed implementations ------------------

    def _timed(self, call):
        started = time.perf_counter()
        result = call()
        return result, time.perf_counter() - started

    @staticmethod
    def _raise_for_status(response, provider_label: str) -> None:
        if response.status_code < 400:
            return
        body = (response.text or "")[:400]
        retryable = response.status_code == 429 or response.status_code >= 500
        raise ProviderError(
            f"{provider_label} returned HTTP {response.status_code}: {body}",
            retryable=retryable,
        )
