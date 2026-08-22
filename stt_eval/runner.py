"""Run orchestration: transcribe, score, persist — concurrently and resumably.

The run happens on a background thread so the Streamlit main thread never
blocks; the UI polls `Runner.snapshot()` for progress and partial results.
Providers run in parallel with each other, clips run in parallel within a
provider, and a per-provider semaphore keeps the app inside each provider's
rate limits. A failure on one (clip, provider) pair is retried a few times and
then recorded as a failure for that pair only — every other pair continues.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .dataset import Clip, Dataset
from .judge import Judge
from .metrics import LLM_KEYS, evaluate_pair
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
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5


@dataclass
class RunProgress:
    total: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    started_at: float = 0.0
    finished_at: float | None = None
    current: list[str] = field(default_factory=list)
    fatal_error: str | None = None

    @property
    def done(self) -> bool:
        return self.finished_at is not None

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 1.0
        return min(1.0, (self.completed + self.skipped) / self.total)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)


class Runner:
    """Owns one run: its thread pool, its progress, and its store writes."""

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
        self._thread = threading.Thread(target=self._run, name=f"stt-run-{self.config.run_id}", daemon=True)
        self.progress.started_at = time.time()
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> tuple[RunProgress, list[StoredResult]]:
        with self._lock:
            progress = RunProgress(
                total=self.progress.total,
                completed=self.progress.completed,
                skipped=self.progress.skipped,
                failed=self.progress.failed,
                started_at=self.progress.started_at,
                finished_at=self.progress.finished_at,
                current=list(self.progress.current),
                fatal_error=self.progress.fatal_error,
            )
        return progress, self.store.load(self.config.run_id)

    # -- internals ----------------------------------------------------------

    def _build_providers(self) -> None:
        for key in self.config.provider_keys:
            self._providers[key] = build(key, self.config.api_keys.get(key, ""))

    def _run(self) -> None:
        try:
            self._build_providers()
            already_done = self.store.completed_keys(
                self.config.run_id, required_metrics=self.config.enabled_metrics
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

            workers = max(1, self.config.per_provider_concurrency * len(self.config.provider_keys))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stt") as pool:
                futures = [pool.submit(self._process, clip, key) for clip, key in tasks]
                for future in futures:
                    future.result()  # exceptions are handled inside _process
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never a crash
            with self._lock:
                self.progress.fatal_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self.progress.finished_at = time.time()
                self.progress.current = []

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

    def _process(self, clip: Clip, provider_key: str) -> None:
        label = f"{clip.clip_id} · {provider_key}"
        with self._lock:
            self.progress.current.append(label)

        semaphore = self._semaphores[provider_key]
        try:
            with semaphore:
                if self._cancel.is_set():
                    return
                provider = self._providers[provider_key]
                # The offline mock provider degrades the ground truth rather than
                # decoding audio, so it needs the reference text.
                if provider_key == "mock":
                    provider._ground_truth_hint = clip.ground_truth  # noqa: SLF001

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
                            language=clip.language_for(self.config.language),
                        )
                    )
                    with self._lock:
                        self.progress.failed += 1
                        self.progress.completed += 1
                    return

                evaluation = evaluate_pair(
                    clip_id=clip.clip_id,
                    provider=provider_key,
                    ground_truth=clip.ground_truth,
                    prediction=prediction,
                    enabled_metrics=self.config.enabled_metrics,
                    judge=self.judge if any(k in LLM_KEYS for k in self.config.enabled_metrics) else None,
                    # The judge is told the clip's own language, so it applies
                    # the right transliteration and code-switching rules.
                    language=clip.language_for(self.config.language),
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
                        language=clip.language_for(self.config.language),
                        metrics={
                            key: {
                                "value": metric.value,
                                "reasoning": metric.reasoning,
                                "error": metric.error,
                                "errors": metric.errors,
                                "length": metric.length,
                            }
                            for key, metric in evaluation.metrics.items()
                        },
                    )
                )
                with self._lock:
                    self.progress.completed += 1
        finally:
            with self._lock:
                if label in self.progress.current:
                    self.progress.current.remove(label)
