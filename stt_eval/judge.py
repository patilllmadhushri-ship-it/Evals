"""Judge clients — one interface, swappable backends.

Every LLM-based metric routes through a `Judge`, so the judging model can be
changed without touching a single metric. Two backends ship:

* **Anthropic** — Claude directly, with server-enforced JSON schemas.
* **OpenRouter** — any model on OpenRouter, including reasoning models, with a
  unified `reasoning` effort control and per-call cost reported by the API.

Keys live only in the judge object for the lifetime of the run. They are never
written to the results database, the logs, or an exported file.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field

from .config import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    JUDGE_PRICING,
    OPENROUTER_FALLBACKS,
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
)
from .prompts import JSON_ONLY_INSTRUCTION, JUDGE_SYSTEM_PROMPT

ANTHROPIC_BACKEND = "anthropic"
OPENROUTER_BACKEND = "openrouter"


class JudgeError(RuntimeError):
    """A judge call failed. Callers isolate this per metric, so one failing
    metric never blocks the others."""


@dataclass
class JudgeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    #: Cost reported by the provider itself, when it reports one.
    reported_cost_usd: float = 0.0


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Reasoning models sometimes wrap the object in a fenced block or prepend a
    stray line despite instructions, so fall back to the first balanced object
    rather than failing the whole metric on a cosmetic wrapper.
    """
    candidate = (text or "").strip()
    if not candidate:
        raise JudgeError("The judge returned an empty response.")

    fenced = re.match(r"^```(?:json)?\s*(.+?)\s*```$", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : index + 1])
                    except json.JSONDecodeError:
                        break

    raise JudgeError(f"The judge returned unparseable JSON: {candidate[:200]}")


@dataclass
class Judge:
    """Base judge. Subclasses implement `_call`; everything else is shared."""

    api_key: str
    model: str
    effort: str = "medium"
    max_tokens: int = 8000
    usage: JudgeUsage = field(default_factory=JudgeUsage)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    #: Registry key of the backend, for display and export metadata.
    backend: str = "base"

    @property
    def system_prompt(self) -> str:
        return JUDGE_SYSTEM_PROMPT

    def ask_json(self, *, system: str | None = None, prompt: str, schema: dict) -> dict:
        raise NotImplementedError

    def _record(self, *, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, cost: float = 0.0) -> None:
        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += input_tokens or 0
            self.usage.output_tokens += output_tokens or 0
            self.usage.reasoning_tokens += reasoning_tokens or 0
            self.usage.reported_cost_usd += cost or 0.0

    def estimated_cost_usd(self) -> float:
        """Provider-reported cost when available, else the bundled rate table."""
        with self._lock:
            if self.usage.reported_cost_usd:
                return self.usage.reported_cost_usd
            input_rate, output_rate = JUDGE_PRICING.get(self.model, (0.0, 0.0))
            return (
                self.usage.input_tokens * input_rate
                + self.usage.output_tokens * output_rate
            ) / 1_000_000

    def check(self) -> None:
        """Cheap round-trip so a bad key or model fails at setup, not mid-run.

        The prompt deliberately contains no literal JSON example. Showing one
        alongside a schema makes some models nest the example inside the
        schema's own field — `{"ok": {"ok": true}}` — which is a failure of the
        prompt, not of the model or the key.
        """
        payload = self.ask_json(
            system=(
                "You are validating that structured output works. Set the field "
                "to true. Return only the object described by the schema."
            ),
            prompt="Confirm you can respond in the required format by setting ok to true.",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        if not payload.get("ok"):
            raise JudgeError(f"Unexpected validation response: {payload}")


@dataclass
class AnthropicJudge(Judge):
    """Claude via the Anthropic SDK, with server-enforced JSON schemas."""

    model: str = DEFAULT_JUDGE_MODEL
    backend: str = ANTHROPIC_BACKEND
    _client: object | None = field(default=None, repr=False)

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise JudgeError(
                    "The 'anthropic' package is required for the Anthropic judge. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=3)
        return self._client

    def ask_json(self, *, system: str | None = None, prompt: str, schema: dict) -> dict:
        client = self._ensure_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or self.system_prompt,
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
            self._record(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )

        text = next(
            (block.text for block in response.content if getattr(block, "type", "") == "text"),
            "",
        )
        return _extract_json(text)


@dataclass
class OpenRouterJudge(Judge):
    """Any OpenRouter model, including reasoning models.

    Reasoning effort maps to OpenRouter's unified `reasoning` parameter, so the
    same setting works across model families. Structured output is requested
    via `response_format`; models that reject it fall back to a JSON-only
    instruction in the prompt, which the tolerant parser then handles.
    """

    model: str = DEFAULT_OPENROUTER_MODEL
    backend: str = OPENROUTER_BACKEND
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: float = 240.0
    #: Set once a model has rejected response_format, so we stop retrying it.
    _schema_unsupported: bool = field(default=False, repr=False)
    #: Models to try when the chosen one is overloaded, and which one we
    #: started from if a substitution happened — the UI should say so rather
    #: than report a verdict from a model the user did not pick.
    fallback_models: list[str] = field(default_factory=list)
    substituted_from: str = ""
    _tried: set[str] = field(default_factory=set, repr=False)

    def _next_fallback(self) -> str:
        """The next untried fallback, or empty when they are exhausted."""
        self._tried.add(self.model)
        for candidate in self.fallback_models:
            if candidate not in self._tried:
                return candidate
        return ""

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers OpenRouter uses for its dashboards.
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        }

    def _body(self, *, system: str, prompt: str, schema: dict, with_schema: bool) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            # Unified across model families — reasoning models spend more
            # thinking, non-reasoning models ignore it.
            "reasoning": {"effort": self.effort},
            # Ask OpenRouter to report what the call actually cost.
            "usage": {"include": True},
        }
        if with_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "stt_eval", "strict": True, "schema": schema},
            }
        else:
            body["messages"][1]["content"] = (
                prompt
                + "\n\n"
                + JSON_ONLY_INSTRUCTION.format(schema=json.dumps(schema, indent=2))
            )
        return body

    def ask_json(self, *, system: str | None = None, prompt: str, schema: dict) -> dict:
        import requests

        system_prompt = system or self.system_prompt
        original_model = self.model

        # Two independent retries are in play: dropping the JSON schema for a
        # model that rejects it, and switching model when this one is
        # overloaded. Rebuilding the attempt list per model keeps them from
        # consuming each other's budget.
        attempts = [not self._schema_unsupported]
        if attempts[0]:
            attempts.append(False)  # retry without response_format on rejection

        last_error: str = ""
        while attempts:
            with_schema = attempts.pop(0)
            body = self._body(
                system=system_prompt, prompt=prompt, schema=schema, with_schema=with_schema
            )
            try:
                response = requests.post(
                    self.endpoint, headers=self._headers(), json=body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                raise JudgeError(f"OpenRouter request failed: {exc}") from exc

            if response.status_code >= 400:
                detail = (response.text or "")[:300]
                # A model that cannot do structured output says so with a 4xx;
                # remember that and use the prompt-only path from here on.
                if with_schema and response.status_code < 500 and (
                    "response_format" in detail or "json_schema" in detail
                ):
                    self._schema_unsupported = True
                    last_error = detail
                    continue
                # A quota or overload response is not a bug to debug — it is a
                # condition with a specific fix, so say which one it is.
                if response.status_code == 429:
                    limit = "daily free-tier" if "free-models-per-day" in detail else "rate"
                    raise JudgeError(
                        f"{self.model} hit its {limit} limit on OpenRouter. Either add "
                        "credits at openrouter.ai/credits, switch to a paid model in "
                        "the judge settings, use an Anthropic key instead, or wait "
                        "for the daily quota to reset."
                    )
                if response.status_code in (502, 503):
                    # Overload is per-model, so a sibling usually answers. Try
                    # one, once, and say which model actually produced the
                    # verdict rather than silently substituting it.
                    substitute = self._next_fallback()
                    if substitute:
                        self.model = substitute
                        self.substituted_from = self.substituted_from or original_model
                        self._schema_unsupported = False
                        attempts = [True, False]  # fresh budget for the new model
                        continue
                    raise JudgeError(
                        f"{self.model} is temporarily overloaded upstream — common on "
                        "`:free` endpoints — and its fallbacks were too. Retry, or "
                        "pick a paid model for a run that has to finish."
                    )
                raise JudgeError(f"OpenRouter returned HTTP {response.status_code}: {detail}")

            payload = response.json()
            if "error" in payload and payload["error"]:
                # OpenRouter reports an upstream failure as HTTP 200 with the
                # real status inside the body, so the same conditions have to
                # be recognised here as well as on the response code.
                inner = payload["error"] if isinstance(payload["error"], dict) else {}
                inner_code = inner.get("code")
                inner_detail = str(payload["error"])[:300]

                if inner_code == 429 or "rate limit" in inner_detail.lower():
                    limit = (
                        "daily free-tier"
                        if "free-models-per-day" in inner_detail
                        else "rate"
                    )
                    raise JudgeError(
                        f"{self.model} hit its {limit} limit on OpenRouter. Either add "
                        "credits at openrouter.ai/credits, switch to a paid model in "
                        "the judge settings, use an Anthropic key instead, or wait "
                        "for the daily quota to reset."
                    )

                if inner_code in (500, 502, 503) or "overloaded" in inner_detail.lower():
                    substitute = self._next_fallback()
                    if substitute:
                        self.model = substitute
                        self.substituted_from = self.substituted_from or original_model
                        self._schema_unsupported = False
                        attempts = [True, False]
                        continue
                    raise JudgeError(
                        f"{self.model} is temporarily overloaded upstream — common on "
                        "`:free` endpoints — and its fallbacks were too. Retry, or "
                        "pick a paid model for a run that has to finish."
                    )

                raise JudgeError(f"OpenRouter error: {inner_detail}")

            usage = payload.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            self._record(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=details.get("reasoning_tokens", 0),
                cost=float(usage.get("cost") or 0.0),
            )

            choices = payload.get("choices") or []
            if not choices:
                raise JudgeError(f"OpenRouter returned no choices: {str(payload)[:200]}")
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
            if not text.strip():
                finish = choices[0].get("finish_reason")
                raise JudgeError(
                    "The judge returned no content"
                    + (f" (finish_reason={finish})" if finish else "")
                    + ". Reasoning models can spend the whole budget thinking — "
                    "try a higher max_tokens or a lower effort."
                )
            return _extract_json(text)

        raise JudgeError(f"OpenRouter rejected the request: {last_error}")


def create_judge(
    *, backend: str, api_key: str, model: str, effort: str = "medium", max_tokens: int = 8000
) -> Judge:
    """Build the judge for `backend`. This is the only place a backend is chosen."""
    if backend == ANTHROPIC_BACKEND:
        return AnthropicJudge(api_key=api_key, model=model, effort=effort, max_tokens=max_tokens)
    if backend == OPENROUTER_BACKEND:
        return OpenRouterJudge(
            api_key=api_key,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            # Only free models fall back automatically: substituting a paid
            # model would spend money the user did not choose to spend.
            fallback_models=(
                [item for item in OPENROUTER_FALLBACKS if item != model]
                if model.endswith(":free")
                else []
            ),
        )
    raise JudgeError(f"Unknown judge backend: {backend}")
