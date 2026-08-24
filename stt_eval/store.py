"""SQLite persistence for run results.

Two requirements shape this module. Partial results are written as each
(clip, provider) pair completes, not at the end, so a browser refresh or a
dropped connection never loses finished work. And completed pairs are keyed by
(run_id, clip_id, provider), so re-running a session with an added clip or an
added provider only processes the new work.

No credential ever reaches this file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_DB_PATH = Path(".stt_eval_runs") / "results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    language    TEXT NOT NULL,
    providers   TEXT NOT NULL,
    metrics     TEXT NOT NULL,
    judge_model TEXT
);

CREATE TABLE IF NOT EXISTS results (
    run_id            TEXT NOT NULL,
    clip_id           TEXT NOT NULL,
    provider          TEXT NOT NULL,
    status            TEXT NOT NULL,
    ground_truth      TEXT NOT NULL DEFAULT '',
    prediction        TEXT NOT NULL DEFAULT '',
    error             TEXT,
    latency_seconds   REAL,
    duration_seconds  REAL NOT NULL DEFAULT 0,
    language          TEXT NOT NULL DEFAULT '',
    sampled           INTEGER NOT NULL DEFAULT 1,
    metrics_json      TEXT NOT NULL DEFAULT '{}',
    updated_at        REAL NOT NULL,
    PRIMARY KEY (run_id, clip_id, provider)
);

CREATE INDEX IF NOT EXISTS results_by_run ON results (run_id);
"""


@dataclass
class StoredResult:
    run_id: str
    clip_id: str
    provider: str
    status: str  # "ok" | "failed"
    ground_truth: str = ""
    prediction: str = ""
    error: str | None = None
    latency_seconds: float | None = None
    duration_seconds: float = 0.0
    #: The language this clip was actually transcribed in.
    language: str = ""
    #: False when the pair was deliberately left out of the LLM-metric sample,
    #: which is different from having been judged and failed.
    sampled: bool = True
    metrics: dict = field(default_factory=dict)
    updated_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def metric_value(self, key: str) -> float | None:
        entry = self.metrics.get(key) or {}
        return entry.get("value")

    def metric_reasoning(self, key: str) -> str | None:
        entry = self.metrics.get(key) or {}
        return entry.get("reasoning")

    def metric_error(self, key: str) -> str | None:
        entry = self.metrics.get(key) or {}
        return entry.get("error")

    def metric_extra(self, key: str) -> dict:
        """Structured detail a metric kept beside its score — per-field verdicts,
        for instance. Empty for metrics that record none."""
        entry = self.metrics.get(key) or {}
        return entry.get("extra") or {}


class ResultStore:
    """Thread-safe, process-local SQLite store."""

    def __init__(self, path: Path | str = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(_SCHEMA)
            self._migrate()
            self._connection.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        `CREATE TABLE IF NOT EXISTS` leaves an existing table untouched, so a
        store written by an older version keeps its old shape and every insert
        naming a new column fails. Adding the column back is cheap and safe.
        """
        existing = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(results)")
        }
        for column, definition in (
            ("language", "TEXT NOT NULL DEFAULT ''"),
            ("sampled", "INTEGER NOT NULL DEFAULT 1"),
        ):
            if column not in existing:
                self._connection.execute(
                    f"ALTER TABLE results ADD COLUMN {column} {definition}"
                )

    # -- runs ---------------------------------------------------------------

    def ensure_run(
        self, run_id: str, *, language: str, providers: list[str], metrics: list[str], judge_model: str | None
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO runs (run_id, created_at, language, providers, metrics, judge_model)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    language = excluded.language,
                    providers = excluded.providers,
                    metrics = excluded.metrics,
                    judge_model = excluded.judge_model
                """,
                (
                    run_id,
                    time.time(),
                    language,
                    json.dumps(providers),
                    json.dumps(metrics),
                    judge_model,
                ),
            )
            self._connection.commit()

    def list_runs(self) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "language": row["language"],
                "providers": json.loads(row["providers"]),
                "metrics": json.loads(row["metrics"]),
                "judge_model": row["judge_model"],
            }
            for row in rows
        ]

    # -- results ------------------------------------------------------------

    def save(self, result: StoredResult) -> None:
        payload = asdict(result)
        payload["updated_at"] = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO results (
                    run_id, clip_id, provider, status, ground_truth, prediction,
                    error, latency_seconds, duration_seconds, language, sampled,
                    metrics_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, clip_id, provider) DO UPDATE SET
                    status = excluded.status,
                    ground_truth = excluded.ground_truth,
                    prediction = excluded.prediction,
                    error = excluded.error,
                    latency_seconds = excluded.latency_seconds,
                    duration_seconds = excluded.duration_seconds,
                    language = excluded.language,
                    sampled = excluded.sampled,
                    metrics_json = excluded.metrics_json,
                    updated_at = excluded.updated_at
                """,
                (
                    payload["run_id"],
                    payload["clip_id"],
                    payload["provider"],
                    payload["status"],
                    payload["ground_truth"],
                    payload["prediction"],
                    payload["error"],
                    payload["latency_seconds"],
                    payload["duration_seconds"],
                    payload["language"],
                    1 if payload["sampled"] else 0,
                    json.dumps(payload["metrics"]),
                    payload["updated_at"],
                ),
            )
            self._connection.commit()

    def load(self, run_id: str) -> list[StoredResult]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY clip_id, provider",
                (run_id,),
            ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def completed_keys(
        self,
        run_id: str,
        *,
        required_metrics: list[str],
        llm_metrics: list[str] | None = None,
    ) -> set[tuple[str, str]]:
        """Pairs that need no further transcription — these are skipped.

        A pair counts as complete once it has a transcript and every
        `required_metrics` value, so enabling a new metric re-scores just that
        metric's missing work rather than nothing at all.

        `llm_metrics` are judged in a later stage on a sampled subset, so they
        deliberately do not gate this: a pair left out of the sample is complete
        as far as transcription is concerned, and re-running must not
        re-transcribe it just because it was never judged.
        """
        complete: set[tuple[str, str]] = set()
        for result in self.load(run_id):
            if not result.ok:
                continue
            if all(
                (result.metrics.get(key) or {}).get("value") is not None
                for key in required_metrics
            ):
                complete.add((result.clip_id, result.provider))
        return complete

    def close(self) -> None:
        """Release the SQLite handle. Streamlit keeps one store per process, so
        this exists mainly for tests and short-lived scripts."""
        with self._lock:
            self._connection.close()

    def delete_run(self, run_id: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
            self._connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._connection.commit()

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> StoredResult:
        return StoredResult(
            run_id=row["run_id"],
            clip_id=row["clip_id"],
            provider=row["provider"],
            status=row["status"],
            ground_truth=row["ground_truth"],
            prediction=row["prediction"],
            error=row["error"],
            latency_seconds=row["latency_seconds"],
            duration_seconds=row["duration_seconds"],
            language=(row["language"] if "language" in row.keys() else "") or "",
            sampled=bool(row["sampled"]) if "sampled" in row.keys() else True,
            metrics=json.loads(row["metrics_json"] or "{}"),
            updated_at=row["updated_at"],
        )
