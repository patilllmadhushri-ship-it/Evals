"""Static configuration: languages, provider rate table, defaults.

Everything in here is data, not behaviour — the rate table in particular is a
bundled snapshot of published list prices and will drift from any real invoice.
"""

from __future__ import annotations

from dataclasses import dataclass

# Target audio format every provider receives, so accuracy differences reflect
# the provider rather than the input encoding.
DEFAULT_SAMPLE_RATE = 16_000
TARGET_SAMPLE_RATE = DEFAULT_SAMPLE_RATE  # back-compat alias
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit PCM

# Telephony audio is narrowband 8 kHz. Evaluating at the rate your production
# audio actually arrives at matters: upsampling a phone call to 16 kHz invents
# no new information but does change how some providers' models behave, so a
# 16 kHz benchmark can mispredict how a provider performs on your phone traffic.
SAMPLE_RATE_OPTIONS = {
    8_000: "8 kHz — narrowband (telephony, IVR, call recordings)",
    16_000: "16 kHz — wideband (mic capture, VoIP, most datasets)",
}

SUPPORTED_UPLOAD_EXTENSIONS = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "aac"]

DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 16
DEFAULT_RETRIES = 2

DEFAULT_JUDGE_MODEL = "claude-opus-5"
JUDGE_MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]

# Judge pricing (USD per million tokens) for the estimated-cost display.
JUDGE_PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass(frozen=True)
class Language:
    code: str
    label: str


LANGUAGES: list[Language] = [
    # English variants — accent matters as much as language for STT accuracy.
    Language("en-IN", "English (India)"),
    Language("en-US", "English (US)"),
    Language("en-GB", "English (UK)"),
    Language("en-AU", "English (Australia)"),
    # Indic
    Language("hi-IN", "Hindi"),
    Language("mr-IN", "Marathi"),
    Language("bn-IN", "Bengali"),
    Language("ta-IN", "Tamil"),
    Language("te-IN", "Telugu"),
    Language("kn-IN", "Kannada"),
    Language("ml-IN", "Malayalam"),
    Language("gu-IN", "Gujarati"),
    Language("pa-IN", "Punjabi"),
    Language("od-IN", "Odia"),
    Language("as-IN", "Assamese"),
    Language("ur-IN", "Urdu"),
    # European
    Language("es-ES", "Spanish (Spain)"),
    Language("es-419", "Spanish (Latin America)"),
    Language("pt-BR", "Portuguese (Brazil)"),
    Language("pt-PT", "Portuguese (Portugal)"),
    Language("fr-FR", "French"),
    Language("de-DE", "German"),
    Language("it-IT", "Italian"),
    Language("nl-NL", "Dutch"),
    Language("pl-PL", "Polish"),
    Language("sv-SE", "Swedish"),
    Language("da-DK", "Danish"),
    Language("uk-UA", "Ukrainian"),
    Language("ru-RU", "Russian"),
    Language("tr-TR", "Turkish"),
    # Middle East & Africa
    Language("ar-SA", "Arabic (Gulf)"),
    Language("ar-EG", "Arabic (Egypt)"),
    Language("he-IL", "Hebrew"),
    Language("sw-KE", "Swahili"),
    # East & Southeast Asia
    Language("zh-CN", "Chinese (Mandarin, Simplified)"),
    Language("ja-JP", "Japanese"),
    Language("ko-KR", "Korean"),
    Language("id-ID", "Indonesian"),
    Language("vi-VN", "Vietnamese"),
    Language("th-TH", "Thai"),
    Language("ms-MY", "Malay"),
    Language("tl-PH", "Filipino"),
]

LANGUAGE_LABELS = {lang.code: lang.label for lang in LANGUAGES}

# Published per-minute list rates in USD, bundled as a snapshot.
# Tiers, minimums and rounding are NOT modelled — figures are estimates.
PROVIDER_RATES_USD_PER_MINUTE = {
    "deepgram": 0.0043,
    "openai": 0.0060,
    "google": 0.0160,
    "sarvam": 0.0050,
    "elevenlabs": 0.0067,
    "mock": 0.0,
}

RATE_TABLE_NOTE = (
    "Cost figures are estimates from a bundled per-minute rate table. They do not "
    "model tiers, volume discounts, minimum billable durations or per-request "
    "rounding, and may not match a provider's actual invoice."
)

# Languages where word boundaries are ambiguous — CER is the more meaningful
# deterministic metric there, and the UI says so.
CHARACTER_ORIENTED_LANGUAGES = {"ja-JP", "zh-CN", "th-TH", "ms-MY"}

LANGUAGE_SUPPORT_NOTE = (
    "Per-provider language support is a bundled snapshot and providers add "
    "languages regularly. If a provider you expect is missing for a language, "
    "check its current docs — the list here is conservative, not authoritative."
)
