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

#: Judge backend -> environment variable holding its credential.
JUDGE_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

JUDGE_ENV_VAR = JUDGE_ENV_VARS["anthropic"]  # back-compat alias


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


def _from_streamlit_secrets() -> dict[str, str]:
    """Credentials from Streamlit's secrets store, when running on a host.

    A deployed app has no `.env` — it is git-ignored and never shipped — so
    Streamlit Community Cloud and similar hosts supply keys through their own
    secrets UI instead. Imported lazily and defensively so the engine keeps
    working outside Streamlit entirely, which is how the test scripts run it.
    """
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:  # noqa: BLE001 - not running under Streamlit, or no secrets file
        return {}

    wanted = list(PROVIDER_ENV_VARS.values()) + list(JUDGE_ENV_VARS.values())
    found: dict[str, str] = {}
    for name in wanted:
        try:
            value = secrets.get(name)  # raises if no secrets.toml exists at all
        except Exception:  # noqa: BLE001
            return {}
        if value:
            found[name] = str(value)
    return found


def load(path: Path | str = DEFAULT_ENV_PATH, *, override: bool = False) -> dict[str, str]:
    """Load credentials from `.env` and Streamlit secrets into `os.environ`.

    Precedence, highest first: an already-exported environment variable, then
    `.env`, then Streamlit secrets. So a key exported in the shell beats a stale
    one on disk, and a local `.env` beats a deployed default while developing.
    """
    values: dict[str, str] = _from_streamlit_secrets()

    env_path = Path(path)
    if env_path.exists():
        values.update(parse(env_path.read_text(encoding="utf-8")))

    for name, value in values.items():
        if value and (override or not os.environ.get(name)):
            os.environ[name] = value
    return values


def provider_key(provider: str) -> str:
    """The credential for `provider` from the environment, or an empty string."""
    variable = PROVIDER_ENV_VARS.get(provider)
    return os.environ.get(variable, "") if variable else ""


def judge_key(backend: str = "anthropic") -> str:
    variable = JUDGE_ENV_VARS.get(backend)
    return os.environ.get(variable, "") if variable else ""


def configured_judge_backends() -> list[str]:
    return [backend for backend in JUDGE_ENV_VARS if judge_key(backend)]


def configured_providers() -> list[str]:
    return [provider for provider in PROVIDER_ENV_VARS if provider_key(provider)]


def mask(secret: str) -> str:
    """Render a key for display without exposing it — `sk_pk…9B3O`."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "…"
    return f"{secret[:5]}…{secret[-4:]}"
