"""Read credentials from a local, git-ignored `.env`.

This is a convenience for repeated local runs — it saves retyping keys into
step 3 on every session. The security posture is unchanged: values are read
into process memory, used for the run, and never written to the results
database, the logs, or an exported file.

A tiny parser rather than a `python-dotenv` dependency, because the format we
need is `KEY=value` and nothing more.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_PATH = Path(".env")

#: Provider registry key -> environment variable holding its credential.
PROVIDER_ENV_VARS = {
    "deepgram": "DEEPGRAM_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "sarvam": "SARVAM_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
}

JUDGE_ENV_VAR = "ANTHROPIC_API_KEY"


def parse(text: str) -> dict[str, str]:
    """Parse `KEY=value` lines, ignoring blanks, comments and `export` prefixes."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name:
            values[name] = value
    return values


def load(path: Path | str = DEFAULT_ENV_PATH, *, override: bool = False) -> dict[str, str]:
    """Load `.env` into `os.environ` and return what was found.

    Real environment variables win by default, so a key exported in the shell
    beats a stale one in the file.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values = parse(env_path.read_text(encoding="utf-8"))
    for name, value in values.items():
        if value and (override or not os.environ.get(name)):
            os.environ[name] = value
    return values


def provider_key(provider: str) -> str:
    """The credential for `provider` from the environment, or an empty string."""
    variable = PROVIDER_ENV_VARS.get(provider)
    return os.environ.get(variable, "") if variable else ""


def judge_key() -> str:
    return os.environ.get(JUDGE_ENV_VAR, "")


def configured_providers() -> list[str]:
    return [provider for provider in PROVIDER_ENV_VARS if provider_key(provider)]


def mask(secret: str) -> str:
    """Render a key for display without exposing it — `sk_pk…9B3O`."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "…"
    return f"{secret[:5]}…{secret[-4:]}"
