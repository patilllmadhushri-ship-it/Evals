"""STT Evaluation Studio — a Streamlit app for benchmarking speech-to-text
accuracy against your own audio and ground truth.

One guided flow: upload → configure → credentials → run → review → export.
Transcription and scoring run on a background thread, so the UI stays live and
partial results stream in as each clip completes.
"""

from __future__ import annotations

import time
import uuid

import pandas as pd
import streamlit as st

from stt_eval import costs, export, report
from stt_eval.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RETRIES,
    JUDGE_MODELS,
    LANGUAGES,
    MAX_CONCURRENCY,
    RATE_TABLE_NOTE,
    SUPPORTED_UPLOAD_EXTENSIONS,
)
from stt_eval.dataset import build_dataset, clip_id_for, parse_ground_truth_csv
from stt_eval.judge import Judge, JudgeError
from stt_eval.metrics import LLM_KEYS, METRIC_SPECS, METRICS_BY_KEY
from stt_eval.providers import (
    PROVIDER_CLASSES,
    credential_hint,
    provider_label,
    providers_for_language,
    requires_key,
)
from stt_eval.runner import RunConfig, Runner
from stt_eval.store import ResultStore

STEPS = ["1 · Upload", "2 · Configure", "3 · Credentials", "4 · Run", "5 · Review", "6 · Export"]

st.set_page_config(page_title="STT Evaluation Studio", page_icon="🎙️", layout="wide")


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


def init_state() -> None:
    defaults = {
        "step": 0,
        "run_id": uuid.uuid4().hex[:12],
        "language": "en-IN",
        "selected_providers": [],
        "enabled_metrics": [spec.key for spec in METRIC_SPECS if spec.default_on],
        "api_keys": {},
        "judge_key": "",
        "judge_model": DEFAULT_JUDGE_MODEL,
        "judge_effort": "medium",
        "concurrency": DEFAULT_CONCURRENCY,
        "inline_ground_truth": {},
        "dataset": None,
        "runner": None,
        "uploaded_audio": [],
        "ground_truth_map": {},
        "ground_truth_source": "ground_truth.csv",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


@st.cache_resource
def get_store() -> ResultStore:
    return ResultStore()


store = get_store()


def goto(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def enabled_metrics() -> list[str]:
    return [key for key in st.session_state.enabled_metrics if key in METRICS_BY_KEY]


def uses_judge() -> bool:
    return any(key in LLM_KEYS for key in enabled_metrics())


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🎙️ STT Evaluation Studio")
    st.caption(
        "Upload your audio and the correct transcripts, run several providers "
        "over the same data, and see which one wins — on accuracy, meaning, "
        "cost and latency."
    )
    st.divider()
    for index, label in enumerate(STEPS):
        marker = "▶" if index == st.session_state.step else ("✓" if index < st.session_state.step else "○")
        if st.button(f"{marker}  {label}", key=f"nav_{index}", width="stretch"):
            goto(index)
    st.divider()
    st.caption(f"Run id: `{st.session_state.run_id}`")
    if st.button("Start a new run", width="stretch"):
        for key in ("dataset", "runner", "uploaded_audio", "ground_truth_map", "inline_ground_truth"):
            st.session_state[key] = None if key in ("dataset", "runner") else ({} if key.endswith("map") or key.startswith("inline") else [])
        st.session_state.run_id = uuid.uuid4().hex[:12]
        goto(0)
    st.caption(
        "API keys live only in this session's memory. They are never written to "
        "disk, logs or the exported results file."
    )


# --------------------------------------------------------------------------
# Step 1 — Upload
# --------------------------------------------------------------------------


def step_upload() -> None:
    st.header("1 · Upload audio and ground truth")
    st.write(
        "Upload one or more recordings, then supply the correct transcript for "
        "each — as a CSV with `id,text` columns, or typed inline below."
    )

    uploads = st.file_uploader(
        "Audio files",
        type=SUPPORTED_UPLOAD_EXTENSIONS,
        accept_multiple_files=True,
        help="WAV works end to end. MP3/M4A/FLAC/OGG are transcoded to mono 16 kHz PCM first.",
    )
    if uploads:
        st.session_state.uploaded_audio = [(item.name, item.getvalue()) for item in uploads]

    audio_files = st.session_state.uploaded_audio or []
    if audio_files:
        st.success(f"{len(audio_files)} audio file(s) ready.")

    mode = st.radio(
        "Ground truth",
        ["CSV file (id, text)", "Type inline per clip"],
        horizontal=True,
    )

    ground_truth: dict[str, str] = {}
    csv_errors: list[str] = []

    if mode.startswith("CSV"):
        st.session_state.ground_truth_source = "ground_truth.csv"
        csv_upload = st.file_uploader("Ground-truth CSV", type=["csv"], key="gt_csv")
        if csv_upload is not None:
            st.session_state.ground_truth_source = csv_upload.name
            ground_truth, csv_errors = parse_ground_truth_csv(csv_upload.getvalue(), csv_upload.name)
            st.session_state.ground_truth_map = ground_truth
            st.caption(f"Parsed {len(ground_truth)} ground-truth row(s) from {csv_upload.name}.")
        else:
            ground_truth = st.session_state.ground_truth_map or {}
        with st.expander("CSV format"):
            st.code("id,text\nclip1,The invoice is for five hundred rupees.\nclip2,Please call me back tomorrow.", language="csv")
            st.caption("`id` matches the audio filename without its extension — `clip1` ↔ `clip1.wav`.")
    else:
        st.session_state.ground_truth_source = "the inline transcripts"
        st.caption("One box per uploaded file. Leave none blank — empty ground truth is rejected.")
        inline = dict(st.session_state.inline_ground_truth or {})
        for filename, _ in audio_files:
            clip_id = clip_id_for(filename)
            inline[clip_id] = st.text_area(
                f"{filename}  →  id `{clip_id}`",
                value=inline.get(clip_id, ""),
                key=f"inline_{clip_id}",
                height=80,
            )
        st.session_state.inline_ground_truth = inline
        ground_truth = {cid: text.strip() for cid, text in inline.items() if text.strip()}
        st.session_state.ground_truth_map = ground_truth

    for message in csv_errors:
        st.warning(message)

    st.divider()
    if st.button("Validate and continue →", type="primary", disabled=not audio_files):
        with st.spinner("Decoding audio and matching ground truth…"):
            dataset = build_dataset(
                audio_files,
                ground_truth,
                ground_truth_source=st.session_state.ground_truth_source,
            )
        st.session_state.dataset = dataset

        if dataset.errors:
            st.error("Fix these before running:")
            for message in dataset.errors:
                st.markdown(f"- {message}")
        for message in dataset.warnings:
            st.warning(message)

        if dataset.is_runnable:
            st.success(
                f"{len(dataset.clips)} clip(s) matched · "
                f"{dataset.total_duration_minutes:.2f} minutes of audio."
            )
            goto(1)

    dataset = st.session_state.dataset
    if dataset and dataset.clips:
        st.subheader("Matched clips")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": clip.clip_id,
                        "file": clip.filename,
                        "seconds": round(clip.duration_seconds, 2),
                        "ground truth": clip.ground_truth,
                    }
                    for clip in dataset.clips
                ]
            ),
            width="stretch",
            hide_index=True,
        )


# --------------------------------------------------------------------------
# Step 2 — Configure
# --------------------------------------------------------------------------


def step_configure() -> None:
    st.header("2 · Configure the comparison")
    dataset = st.session_state.dataset
    if not dataset or not dataset.is_runnable:
        st.info("Upload and validate a dataset first.")
        return

    labels = {lang.code: f"{lang.label} ({lang.code})" for lang in LANGUAGES}
    codes = list(labels)
    st.session_state.language = st.selectbox(
        "Language",
        codes,
        index=codes.index(st.session_state.language) if st.session_state.language in codes else 0,
        format_func=lambda code: labels[code],
    )
    language = st.session_state.language

    available = providers_for_language(language)
    unavailable = [key for key in PROVIDER_CLASSES if key not in available]

    st.subheader("Providers")
    st.caption("Only providers that support the selected language are listed.")
    selected = st.multiselect(
        "Compare",
        available,
        default=[key for key in st.session_state.selected_providers if key in available]
        or [key for key in available if key != "mock"][:2],
        format_func=provider_label,
    )
    st.session_state.selected_providers = selected
    if unavailable:
        st.caption(
            "Hidden for this language: "
            + ", ".join(provider_label(key) for key in unavailable)
        )

    st.subheader("Metrics")
    st.caption(
        "Deterministic metrics are always on and need no extra API. LLM metrics "
        "cost money and time — each is independently toggleable and independently "
        "fault-isolated, so turning one off never affects the others."
    )
    chosen: list[str] = []
    for spec in METRIC_SPECS:
        columns = st.columns([1, 4])
        with columns[0]:
            if spec.kind == "deterministic":
                st.checkbox(spec.label, value=True, disabled=True, key=f"metric_{spec.key}")
                chosen.append(spec.key)
            else:
                if st.checkbox(
                    spec.label,
                    value=spec.key in st.session_state.enabled_metrics,
                    key=f"metric_{spec.key}",
                ):
                    chosen.append(spec.key)
        with columns[1]:
            badge = "deterministic" if spec.kind == "deterministic" else "LLM judge"
            st.caption(f"**{badge}** — {spec.question}")
    st.session_state.enabled_metrics = chosen

    st.subheader("Concurrency")
    st.session_state.concurrency = st.slider(
        "Simultaneous requests per provider",
        min_value=1,
        max_value=MAX_CONCURRENCY,
        value=st.session_state.concurrency,
        help="Lower this if a provider starts rate-limiting your key.",
    )

    st.divider()
    st.subheader("Estimated cost before running")
    minutes = dataset.total_duration_minutes
    estimates = costs.estimate_run(selected, minutes)
    if estimates:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "provider": provider_label(key),
                        "audio minutes": round(estimate.minutes, 2),
                        "rate USD/min": estimate.rate_usd_per_minute,
                        "estimated USD": round(estimate.usd, 4),
                    }
                    for key, estimate in estimates.items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.metric("Estimated provider total", f"${costs.total_usd(estimates):.4f}")
    if any(key in LLM_KEYS for key in chosen):
        st.caption(
            "LLM judge calls are billed separately by token and are not included "
            "above; the actual judge spend is reported after the run."
        )
    st.info(RATE_TABLE_NOTE)

    if st.button("Continue to credentials →", type="primary", disabled=not selected):
        goto(2)


# --------------------------------------------------------------------------
# Step 3 — Credentials
# --------------------------------------------------------------------------


def step_credentials() -> None:
    st.header("3 · Credentials")
    st.warning(
        "Keys are held in this session's memory for the current run only. They are "
        "never persisted to disk, written to logs, or included in exported results."
    )

    selected = st.session_state.selected_providers
    if not selected:
        st.info("Pick at least one provider first.")
        return

    keys = dict(st.session_state.api_keys)
    for provider_key in selected:
        if not requires_key(provider_key):
            st.success(f"{provider_label(provider_key)} — {credential_hint(provider_key)}")
            keys[provider_key] = ""
            continue
        keys[provider_key] = st.text_input(
            provider_label(provider_key),
            value=keys.get(provider_key, ""),
            type="password",
            help=credential_hint(provider_key),
            key=f"key_{provider_key}",
        )
    st.session_state.api_keys = keys

    judge_needed = uses_judge()
    if judge_needed:
        st.divider()
        st.subheader("LLM judge")
        st.caption(
            "All meaning-aware metrics route through one judge client, so the "
            "judge model can be swapped without changing any metric."
        )
        st.session_state.judge_key = st.text_input(
            "Anthropic API key",
            value=st.session_state.judge_key,
            type="password",
            help="Used only for the enabled LLM metrics.",
        )
        columns = st.columns(2)
        with columns[0]:
            st.session_state.judge_model = st.selectbox(
                "Judge model",
                JUDGE_MODELS,
                index=JUDGE_MODELS.index(st.session_state.judge_model)
                if st.session_state.judge_model in JUDGE_MODELS
                else 0,
            )
        with columns[1]:
            st.session_state.judge_effort = st.selectbox(
                "Judge effort", ["low", "medium", "high"],
                index=["low", "medium", "high"].index(st.session_state.judge_effort),
                help="Lower effort is cheaper and faster; higher is more careful.",
            )
        if st.button("Test judge connection"):
            try:
                Judge(
                    api_key=st.session_state.judge_key,
                    model=st.session_state.judge_model,
                    effort=st.session_state.judge_effort,
                ).check()
                st.success("Judge reachable.")
            except JudgeError as exc:
                st.error(str(exc))

    missing = [p for p in selected if requires_key(p) and not keys.get(p)]
    if judge_needed and not st.session_state.judge_key:
        missing.append("LLM judge")
    if missing:
        st.info("Still needed: " + ", ".join(missing))

    if st.button("Continue to run →", type="primary", disabled=bool(missing)):
        goto(3)


# --------------------------------------------------------------------------
# Step 4 — Run
# --------------------------------------------------------------------------


def step_run() -> None:
    st.header("4 · Run")
    dataset = st.session_state.dataset
    if not dataset or not dataset.is_runnable:
        st.info("Upload and validate a dataset first.")
        return

    metrics = enabled_metrics()
    selected = st.session_state.selected_providers
    total_pairs = len(dataset.clips) * len(selected)
    st.write(
        f"**{len(dataset.clips)}** clips × **{len(selected)}** providers = "
        f"**{total_pairs}** transcriptions, each scored on **{len(metrics)}** metric(s)."
    )

    already_done = store.completed_keys(st.session_state.run_id, required_metrics=metrics)
    if already_done:
        st.info(
            f"{len(already_done)} pair(s) from an earlier run are already complete "
            "and will be reused, not re-transcribed or re-scored."
        )

    runner: Runner | None = st.session_state.runner
    columns = st.columns(3)
    with columns[0]:
        if st.button("▶ Run", type="primary", disabled=bool(runner and runner.is_running)):
            judge = None
            if uses_judge():
                judge = Judge(
                    api_key=st.session_state.judge_key,
                    model=st.session_state.judge_model,
                    effort=st.session_state.judge_effort,
                )
            runner = Runner(
                config=RunConfig(
                    run_id=st.session_state.run_id,
                    language=st.session_state.language,
                    provider_keys=selected,
                    api_keys=st.session_state.api_keys,
                    enabled_metrics=metrics,
                    per_provider_concurrency=st.session_state.concurrency,
                    max_retries=DEFAULT_RETRIES,
                ),
                dataset=dataset,
                store=store,
                judge=judge,
            )
            st.session_state.runner = runner
            runner.start()
            st.rerun()
    with columns[1]:
        if st.button("Stop", disabled=not (runner and runner.is_running)):
            runner.cancel()
    with columns[2]:
        if st.button("Discard this run's results"):
            store.delete_run(st.session_state.run_id)
            st.session_state.runner = None
            st.rerun()

    if runner is None:
        results = store.load(st.session_state.run_id)
        if results:
            st.caption(f"{len(results)} result(s) already stored for this run id.")
        return

    progress, results = runner.snapshot()
    if progress.fatal_error:
        st.error(progress.fatal_error)

    st.progress(progress.fraction, text=f"{progress.completed + progress.skipped}/{progress.total} pairs")
    stats = st.columns(4)
    stats[0].metric("Completed", progress.completed)
    stats[1].metric("Reused", progress.skipped)
    stats[2].metric("Failed", progress.failed)
    stats[3].metric("Elapsed", f"{progress.elapsed_seconds:.0f}s")

    if progress.current:
        st.caption("In flight: " + ", ".join(sorted(progress.current)[:8]))

    if results:
        st.subheader("Partial results")
        st.dataframe(
            pd.DataFrame(report.per_clip_rows(results, enabled_metrics=metrics))[
                ["id", "provider", "status", "prediction"] + [k for k in metrics]
            ],
            width="stretch",
            hide_index=True,
        )

    if runner.is_running:
        time.sleep(1.5)
        st.rerun()
    elif progress.done:
        st.success(f"Run finished in {progress.elapsed_seconds:.0f}s.")
        if st.button("Review results →", type="primary"):
            goto(4)


# --------------------------------------------------------------------------
# Step 5 — Review
# --------------------------------------------------------------------------


def step_review() -> None:
    st.header("5 · Review")
    metrics = enabled_metrics()
    results = store.load(st.session_state.run_id)
    if not results:
        st.info("No results for this run yet.")
        return

    summaries = report.summarize(results, enabled_metrics=metrics)
    best = report.winners(summaries, metric_keys=metrics)

    st.subheader("Leaderboard")
    sort_metric = st.selectbox(
        "Rank by",
        metrics,
        format_func=lambda key: METRICS_BY_KEY[key].label,
    )
    order = report.rank(summaries, metric_key=sort_metric)
    rows = {row["provider"]: row for row in report.summary_rows(summaries, enabled_metrics=metrics)}
    table = pd.DataFrame([rows[provider] for provider in order if provider in rows])
    if not table.empty:
        table.insert(0, "rank", range(1, len(table) + 1))
        table["provider"] = table["provider"].map(provider_label)
        highlight = {METRICS_BY_KEY[key].label: best.get(key) for key in metrics}
        renamed = table.rename(columns={key: METRICS_BY_KEY[key].label for key in metrics})

        def mark_winner(column: pd.Series):
            winner = highlight.get(column.name)
            if winner is None:
                return ["" for _ in column]
            winner_label = provider_label(winner)
            return [
                "background-color: rgba(46, 160, 67, 0.25); font-weight: 600"
                if renamed.loc[index, "provider"] == winner_label
                else ""
                for index in column.index
            ]

        st.dataframe(
            renamed.style.apply(mark_winner, axis=0, subset=[METRICS_BY_KEY[k].label for k in metrics]),
            width="stretch",
            hide_index=True,
        )
    st.caption("The best provider per metric is highlighted. " + RATE_TABLE_NOTE)

    failures = [result for result in results if not result.ok]
    if failures:
        with st.expander(f"⚠️ {len(failures)} failed transcription(s)"):
            for failure in failures:
                st.markdown(
                    f"- `{failure.clip_id}` · **{provider_label(failure.provider)}** — {failure.error}"
                )

    st.divider()
    st.subheader("Per-clip drill-down")
    dataset = st.session_state.dataset
    clip_ids = sorted({result.clip_id for result in results})
    clip_id = st.selectbox("Clip", clip_ids)
    clip_results = [result for result in results if result.clip_id == clip_id]

    clip = dataset.by_id(clip_id) if dataset else None
    if clip is not None:
        st.audio(clip.audio.wav_bytes, format="audio/wav")
        st.caption(f"{clip.filename} · {clip.duration_seconds:.2f}s")

    ground_truth = clip_results[0].ground_truth if clip_results else ""
    st.markdown("**Ground truth**")
    st.info(ground_truth or "—")

    for result in clip_results:
        with st.container(border=True):
            header = st.columns([3, 1, 1])
            header[0].markdown(f"### {provider_label(result.provider)}")
            header[1].metric(
                "Latency",
                f"{result.latency_seconds:.2f}s" if result.latency_seconds is not None else "—",
            )
            header[2].metric("Status", "ok" if result.ok else "failed")

            if not result.ok:
                st.error(result.error or "Transcription failed.")
                continue

            st.markdown("**Transcript**")
            st.write(result.prediction or "_(empty)_")

            metric_columns = st.columns(min(4, max(1, len(metrics))))
            for index, key in enumerate(metrics):
                spec = METRICS_BY_KEY[key]
                value = result.metric_value(key)
                error = result.metric_error(key)
                with metric_columns[index % len(metric_columns)]:
                    if error:
                        st.metric(spec.label, "error")
                        st.caption(error)
                    elif value is None:
                        st.metric(spec.label, "—")
                    else:
                        st.metric(spec.label, f"{value:.3f}")

            reasons = [
                (METRICS_BY_KEY[key].label, result.metric_reasoning(key))
                for key in metrics
                if METRICS_BY_KEY[key].has_reasoning and result.metric_reasoning(key)
            ]
            if reasons:
                with st.expander("Judge reasoning"):
                    for label, reasoning in reasons:
                        st.markdown(f"**{label}** — {reasoning}")

    st.divider()
    if st.button("Go to export →", type="primary"):
        goto(5)


# --------------------------------------------------------------------------
# Step 6 — Export
# --------------------------------------------------------------------------


def step_export() -> None:
    st.header("6 · Export")
    metrics = enabled_metrics()
    results = store.load(st.session_state.run_id)
    if not results:
        st.info("Nothing to export yet.")
        return

    summaries = report.summarize(results, enabled_metrics=metrics)
    per_clip = report.per_clip_rows(results, enabled_metrics=metrics)
    summary = report.summary_rows(summaries, enabled_metrics=metrics)

    actual_provider_cost = sum(item.estimated_cost_usd for item in summaries.values())
    estimated_before = costs.total_usd(
        costs.estimate_run(
            st.session_state.selected_providers,
            st.session_state.dataset.total_duration_minutes if st.session_state.dataset else 0.0,
        )
    )
    columns = st.columns(3)
    columns[0].metric("Estimated before run", f"${estimated_before:.4f}")
    columns[1].metric("Actual (audio processed)", f"${actual_provider_cost:.4f}")
    columns[2].metric("Clips scored", sum(item.clips_scored for item in summaries.values()))
    st.caption(RATE_TABLE_NOTE)

    metadata = {
        "run_id": st.session_state.run_id,
        "language": st.session_state.language,
        "providers": ", ".join(st.session_state.selected_providers),
        "metrics": ", ".join(metrics),
        "judge_model": st.session_state.judge_model if uses_judge() else "not used",
        "estimated_cost_before_usd": round(estimated_before, 4),
        "actual_cost_usd": round(actual_provider_cost, 4),
    }

    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)

    downloads = st.columns(3)
    with downloads[0]:
        st.download_button(
            "⬇ Per-clip CSV",
            data=export.per_clip_csv(per_clip),
            file_name=f"stt_eval_{st.session_state.run_id}_per_clip.csv",
            mime="text/csv",
            width="stretch",
        )
    with downloads[1]:
        st.download_button(
            "⬇ Summary CSV",
            data=export.summary_csv(summary),
            file_name=f"stt_eval_{st.session_state.run_id}_summary.csv",
            mime="text/csv",
            width="stretch",
        )
    with downloads[2]:
        try:
            workbook = export.workbook(per_clip, summary, metadata=metadata)
        except ImportError:
            workbook = None
        if workbook:
            st.download_button(
                "⬇ Excel workbook",
                data=workbook,
                file_name=f"stt_eval_{st.session_state.run_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        else:
            st.caption("Install `pandas` and `XlsxWriter` for the Excel export.")


# --------------------------------------------------------------------------

PAGES = [step_upload, step_configure, step_credentials, step_run, step_review, step_export]
PAGES[st.session_state.step]()
