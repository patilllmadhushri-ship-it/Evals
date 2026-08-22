"""Run orchestration: transcribe, score, persist — concurrently and resumably.

The run is a two-stage pipeline with **separate worker pools**, because the two
stages have nothing in common. Transcription is I/O against a provider, fast and
rate-limited per provider. Scoring is one or more LLM calls, an order of
magnitude slower and limited by the judge instead. Running them in one pool
means a slow judge call occupies a worker that a fast provider could be using,
and the whole run paces itself to the judge.

    clips × providers ──▶ [transcription pool] ──▶ scoring queue ──▶ [judge pool]
                            per-provider limit                       judge limit

Deterministic metrics are computed in stage one, since they need no API and make
the leaderboard useful the moment transcripts land. LLM metrics run in stage two
on the sampled subset (see `sampling.py`).

Everything else is unchanged: the whole thing runs on a background thread so the
UI never blocks, partial results persist as each pair completes, and a failure
on one (clip, provider) pair is retried a few times and then recorded as a
failure for that pair only.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import sampling
from .dataset import Clip, Dataset
from .judge import Judge
from .metrics import DETERMINISTIC_KEYS, LLM_KEYS, evaluate_pair
from .providers import ProviderError, STTProvider, build
from .store import ResultStore, StoredResult


@dataclass
class RunConfig:
    run_id: str
    language: str
    provider_keys: list[str]
    api_keys: dict[str, str]
    enabled_metrics: list[str]
    per_provider_concurrency: int = 4
    #: Judge calls in flight at once. Separate from the provider limit because
    #: the judge is a different service with its own rate limit.
    judge_concurrency: int = 4
    #: Fraction of pairs that receive the LLM metrics, stratified by
    #: deterministic error rate. 1.0 judges everything.
    llm_sample_rate: float = 1.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5

    @property
    def llm_metrics(self) -> list[str]:
        return [key for key in self.enabled_metrics if key in LLM_KEYS]

    @property
    def deterministic_metrics(self) -> list[str]:
        return [key for key in self.enabled_metrics if key in DETERMINISTIC_KEYS]


@dataclass
class RunProgress:
    total: int = 0
    transcribed: int = 0
    scored: int = 0
    skipped: int = 0
    failed: int = 0
    to_score: int = 0
    started_at: float = 0.0
    finished_at: float | None = None
    stage: str = "transcribing"
    current: list[str] = field(default_factory=list)
    fatal_error: str | None = None
    sample_summary: str = ""

    @property
    def done(self) -> bool:
        return self.finished_at is not None

    @property
    def completed(self) -> int:
        """Pairs that have a transcript — what the leaderboard is built from."""
        return self.transcribed

    @property
    def fraction(self) -> float:
        # Both stages count toward progress, weighted by how much work each has.
        total_units = self.total + self.to_score
        if total_units <= 0:
            return 1.0
        done_units = self.transcribed + self.skipped + self.scored
        return min(1.0, done_units / total_units)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)


class Runner:
    """Owns one run: its two thread pools, its progress, and its store writes."""

    def __init__(
        self,
        *,
        config: RunConfig,
        dataset: Dataset,
        store: ResultStore,
        judge: Judge | None,
    ):
        self.config = config
        self.dataset = dataset
        self.store = store
        self.judge = judge
        self.progress = RunProgress()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._semaphores = {
            key: threading.Semaphore(config.per_provider_concurrency)
            for key in config.provider_keys
        }
        self._providers: dict[str, STTProvider] = {}

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.store.ensure_run(
            self.config.run_id,
            language=self.config.language,
            providers=self.config.provider_keys,
            metrics=self.config.enabled_metrics,
            judge_model=self.judge.model if self.judge else None,
        )
        self._thread = threading.Thread(
            target=self._run, name=f"stt-run-{self.config.run_id}", daemon=True
        )
        self.progress.started_at = time.time()
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> tuple[RunProgress, list[StoredResult]]:
        with self._lock:
            progress = RunProgress(**{**self.progress.__dict__, "current": list(self.progress.current)})
        return progress, self.store.load(self.config.run_id)

    # -- internals ----------------------------------------------------------

    def _build_providers(self) -> None:
        for key in self.config.provider_keys:
            self._providers[key] = build(key, self.config.api_keys.get(key, ""))

    def _track(self, label: str, add: bool) -> None:
        with self._lock:
            if add:
                self.progress.current.append(label)
            elif label in self.progress.current:
                self.progress.current.remove(label)

    def _run(self) -> None:
        try:
            self._build_providers()
            already_done = self.store.completed_keys(
                self.config.run_id,
                required_metrics=self.config.deterministic_metrics,
                llm_metrics=self.config.llm_metrics,
            )
            tasks = [
                (clip, provider_key)
                for clip in self.dataset.clips
                for provider_key in self.config.provider_keys
                if (clip.clip_id, provider_key) not in already_done
            ]

            with self._lock:
                self.progress.total = len(self.dataset.clips) * len(self.config.provider_keys)
                self.progress.skipped = self.progress.total - len(tasks)
                self.progress.stage = "transcribing"

            # ---- stage 1: transcription + deterministic scoring ----------
            workers = max(
                1, self.config.per_provider_concurrency * max(1, len(self.config.provider_keys))
            )
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stt-asr") as pool:
                for future in [pool.submit(self._transcribe, clip, key) for clip, key in tasks]:
                    future.result()  # exceptions are handled inside _transcribe

            if self._cancel.is_set() or not self.config.llm_metrics or self.judge is None:
                return

            # ---- stage 2: LLM scoring on the sampled subset --------------
            self._score_stage()
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never a crash
            with self._lock:
                self.progress.fatal_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self.progress.finished_at = time.time()
                self.progress.stage = "done"
                self.progress.current = []

    def _score_stage(self) -> None:
        """Pick the pairs worth judging, then judge them in their own pool."""
        results = {
            (result.clip_id, result.provider): result
            for result in self.store.load(self.config.run_id)
            if result.ok
        }
        clips = {clip.clip_id: clip for clip in self.dataset.clips}

        candidates: list[tuple[tuple[str, str], float | None]] = []
        for key, result in results.items():
            if all(result.metrics.get(metric, {}).get("value") is not None for metric in self.config.llm_metrics):
                continue  # already judged on an earlier run
            candidates.append((key, result.metric_value("wer")))

        if not candidates:
            return

        plan = sampling.plan(
            candidates,
            rate=self.config.llm_sample_rate,
            seed=abs(hash(self.config.run_id)) % (2**32),
        )

        with self._lock:
            self.progress.stage = "scoring"
            self.progress.to_score = plan.total_selected
            self.progress.sample_summary = plan.describe()

        # Pairs left unjudged are recorded as such, so the UI can distinguish
        # "not sampled" from "judged and failed".
        for key, _ in candidates:
            if key not in plan.selected:
                result = results[key]
                result.metrics.update(
                    {metric: {"value": None, "skipped": True} for metric in self.config.llm_metrics}
                )
                result.sampled = False
                self.store.save(result)

        with ThreadPoolExecutor(
            max_workers=max(1, self.config.judge_concurrency), thread_name_prefix="stt-judge"
        ) as pool:
            futures = [
                pool.submit(self._score, clips[clip_id], results[(clip_id, provider)])
                for clip_id, provider in sorted(plan.selected)
                if clip_id in clips
            ]
            for future in futures:
                future.result()

    def _transcribe_with_retries(self, provider: STTProvider, clip: Clip) -> tuple[str, float]:
        # A clip recorded in a specific language overrides the run's language,
        # so a mixed-language dataset is never sent as one language.
        language = clip.language_for(self.config.language)
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if self._cancel.is_set():
                raise ProviderError("Run cancelled.", retryable=False)
            try:
                result = provider.transcribe(clip.audio.wav_bytes, language)
                return result.text, result.latency_seconds
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.config.max_retries:
                    break
            except Exception as exc:  # noqa: BLE001 - treat unexpected errors as retryable
                last_error = exc
                if attempt == self.config.max_retries:
                    break
            time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise ProviderError(str(last_error) if last_error else "Unknown provider failure.")

    def _transcribe(self, clip: Clip, provider_key: str) -> None:
        """Stage one: get the transcript, score the deterministic metrics."""
        label = f"{clip.clip_id} · {provider_key}"
        self._track(label, True)
        try:
            with self._semaphores[provider_key]:
                if self._cancel.is_set():
                    return
                provider = self._providers[provider_key]
                # The offline mock provider degrades the ground truth rather
                # than decoding audio, so it needs the reference text.
                if provider_key == "mock":
                    provider._ground_truth_hint = clip.ground_truth  # noqa: SLF001

                language = clip.language_for(self.config.language)
                try:
                    prediction, latency = self._transcribe_with_retries(provider, clip)
                except Exception as exc:  # noqa: BLE001 - isolated to this pair
                    self.store.save(
                        StoredResult(
                            run_id=self.config.run_id,
                            clip_id=clip.clip_id,
                            provider=provider_key,
                            status="failed",
                            ground_truth=clip.ground_truth,
                            error=str(exc),
                            duration_seconds=clip.duration_seconds,
                            language=language,
                        )
                    )
                    with self._lock:
                        self.progress.failed += 1
                        self.progress.transcribed += 1
                    return

                # Deterministic metrics only — no judge, no cost, available the
                # moment the transcript lands.
                evaluation = evaluate_pair(
                    clip_id=clip.clip_id,
                    provider=provider_key,
                    ground_truth=clip.ground_truth,
                    prediction=prediction,
                    enabled_metrics=self.config.deterministic_metrics,
                    judge=None,
                    language=language,
                )

                self.store.save(
                    StoredResult(
                        run_id=self.config.run_id,
                        clip_id=clip.clip_id,
                        provider=provider_key,
                        status="ok",
                        ground_truth=clip.ground_truth,
                        prediction=prediction,
                        latency_seconds=latency,
                        duration_seconds=clip.duration_seconds,
                        language=language,
                        metrics=_metrics_payload(evaluation.metrics),
                    )
                )
                with self._lock:
                    self.progress.transcribed += 1
        finally:
            self._track(label, False)

    def _score(self, clip: Clip, result: StoredResult) -> None:
        """Stage two: add the meaning-aware metrics to an existing result."""
        label = f"judging {result.clip_id} · {result.provider}"
        self._track(label, True)
        try:
            if self._cancel.is_set():
                return
            evaluation = evaluate_pair(
                clip_id=result.clip_id,
                provider=result.provider,
                ground_truth=result.ground_truth,
                prediction=result.prediction,
                enabled_metrics=self.config.llm_metrics,
                judge=self.judge,
                # The judge is told the clip's own language, so it applies the
                # right transliteration and code-switching rules.
                language=clip.language_for(self.config.language),
            )
            merged = dict(result.metrics)
            merged.update(_metrics_payload(evaluation.metrics))
            result.metrics = merged
            result.sampled = True
            self.store.save(result)
            with self._lock:
                self.progress.scored += 1
        finally:
            self._track(label, False)


def _metrics_payload(metrics: dict) -> dict:
    return {
        key: {
            "value": metric.value,
            "reasoning": metric.reasoning,
            "error": metric.error,
            "errors": metric.errors,
            "length": metric.length,
        }
        for key, metric in metrics.items()
    }
