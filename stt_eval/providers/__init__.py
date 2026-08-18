"""Provider registry.

Adding a provider is: write a subclass of `STTProvider`, add it here, add a
per-minute rate in `config.PROVIDER_RATES_USD_PER_MINUTE`. Nothing in the
scoring or UI layers changes.
"""

from __future__ import annotations

from .base import ProviderError, STTProvider, Transcription
from .deepgram import DeepgramProvider
from .google import GoogleProvider
from .mock import MockProvider
from .openai import OpenAIProvider
from .sarvam import SarvamProvider

PROVIDER_CLASSES: dict[str, type[STTProvider]] = {
    DeepgramProvider.key: DeepgramProvider,
    OpenAIProvider.key: OpenAIProvider,
    GoogleProvider.key: GoogleProvider,
    SarvamProvider.key: SarvamProvider,
    MockProvider.key: MockProvider,
}


def provider_label(key: str) -> str:
    cls = PROVIDER_CLASSES.get(key)
    return cls.label if cls else key


def credential_hint(key: str) -> str:
    cls = PROVIDER_CLASSES.get(key)
    return cls.credential_hint if cls else ""


def requires_key(key: str) -> bool:
    return key != MockProvider.key


def supports_language(key: str, language: str) -> bool:
    cls = PROVIDER_CLASSES.get(key)
    if cls is None:
        return False
    supported = cls.supported_languages
    return not supported or language in supported


def providers_for_language(language: str) -> list[str]:
    return [key for key in PROVIDER_CLASSES if supports_language(key, language)]


def build(key: str, api_key: str, **kwargs) -> STTProvider:
    cls = PROVIDER_CLASSES.get(key)
    if cls is None:
        raise KeyError(f"Unknown provider: {key}")
    return cls(api_key, **kwargs)


__all__ = [
    "PROVIDER_CLASSES",
    "ProviderError",
    "STTProvider",
    "Transcription",
    "build",
    "credential_hint",
    "provider_label",
    "providers_for_language",
    "requires_key",
    "supports_language",
]
