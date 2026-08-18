"""A single configurable judge client.

Every LLM-based metric is routed through this one class so the judge model can
be swapped without touching each metric's implementation. The key lives only in
this object, for the lifetime of the run — it is never written to disk, logs or
any exported results file.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from .config import DEFAULT_JUDGE_MODEL, JUDGE_PRICING


class JudgeError(RuntimeError):
    """A judge call failed. Callers isolate this per metric."""


@dataclass
class JudgeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def cost_usd(self, model: str) -> float:
        input_rate, output_rate = JUDGE_PRICING.get(model, (0.0, 0.0))
        return (
            self.input_tokens * input_rate + self.output_tokens * output_rate
        ) / 1_000_000


@dataclass
class Judge:
    api_key: str
    model: str = DEFAULT_JUDGE_MODEL
    effort: str = "medium"
    max_tokens: int = 8000
    usage: JudgeUsage = field(default_factory=JudgeUsage)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _client: object | None = field(default=None, repr=False)

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise JudgeError(
                    "The 'anthropic' package is required for LLM-based metrics. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=3)
        return self._client

    def ask_json(self, *, system: str, prompt: str, schema: dict) -> dict:
        """Run one judged call and return its parsed, schema-constrained JSON."""
        client = self._ensure_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced per metric, never fatal
            raise JudgeError(f"{type(exc).__name__}: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise JudgeError("The judge model declined to score this pair.")

        usage = getattr(response, "usage", None)
        if usage is not None:
            with self._lock:
                self.usage.calls += 1
                self.usage.input_tokens += getattr(usage, "input_tokens", 0) or 0
                self.usage.output_tokens += getattr(usage, "output_tokens", 0) or 0

        text = next(
            (block.text for block in response.content if getattr(block, "type", "") == "text"),
            "",
        )
        if not text.strip():
            raise JudgeError("The judge returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"The judge returned unparseable JSON: {text[:200]}") from exc

    def estimated_cost_usd(self) -> float:
        with self._lock:
            return self.usage.cost_usd(self.model)

    def check(self) -> None:
        """Cheap round-trip so a bad key fails at setup, not mid-run."""
        self.ask_json(
            system="You validate connectivity. Answer exactly.",
            prompt="Reply with {\"ok\": true}.",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
