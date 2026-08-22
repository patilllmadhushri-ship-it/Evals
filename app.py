"""STT Evaluation Studio — a Streamlit app for benchmarking speech-to-text
accuracy against your own audio and ground truth.

One guided flow: upload → configure → credentials → run → review → export.
Transcription and scoring run on a background thread, so the UI stays live and
partial results stream in as each clip completes.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st


def _reload_local_modules() -> None:
    """Pick up edits to `stt_eval/*` without restarting the server.

    Streamlit re-executes this script on every rerun, but Python caches
    imported modules, and Streamlit does not reload them. A server left
    running across a code change therefore keeps serving the old modules,
    and any newly added symbol fails to import from a stale one.

    Reload deepest-first so parents rebind to fresh children; drop a module
    outright if reloading raises, which forces a clean import next run.
    Skipped while a run is in flight, since its worker threads are executing
    code from these very modules.
    """
    if os.environ.get("STT_EVAL_AUTORELOAD", "1") == "0":
        return
    runner = st.session_state.get("runner")
    if runner is not None and getattr(runner, "is_running", False):
        return
    names = [name for name in sys.modules if name == "stt_eval" or name.startswith("stt_eval.")]
    for name in sorted(names, key=lambda name: name.count("."), reverse=True):
        module = sys.modules.get(name)
        if module is None:
            continue
        try:
            importlib.reload(module)
        except Exception:  # noqa: BLE001 - a stale module is worse than a re-import
            sys.modules.pop(name, None)


_reload_local_modules()

from stt_eval import costs, env, export, report
from stt_eval import signals as signals_module
from stt_eval import streaming
from stt_eval.config import (
    CHARACTER_ORIENTED_LANGUAGES,
    DEFAULT_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RETRIES,
    DEFAULT_SAMPLE_RATE,
    FREE_TIER_NOTE,
    FREE_TIER_SUFFIX,
    JUDGE_BACKENDS,
    JUDGE_MODELS,
    LANGUAGE_SUPPORT_NOTE,
    LANGUAGES,
    MAX_CONCURRENCY,
    OPENROUTER_MODELS,
    RATE_TABLE_NOTE,
    SAMPLE_RATE_OPTIONS,
    SUPPORTED_UPLOAD_EXTENSIONS,
)
from stt_eval.audio import AudioError, normalize
from stt_eval.dataset import build_dataset, clip_id_for, parse_ground_truth_csv
from stt_eval.judge import JudgeError, create_judge
from stt_eval.prompts import JUDGE_SYSTEM_PROMPT
from stt_eval.metrics import LLM_KEYS, METRIC_SPECS, METRICS_BY_KEY
from stt_eval.providers import (
    PROVIDER_CLASSES,
    ProviderError,
    build,
    credential_hint,
    provider_label,
    providers_for_language,
    requires_key,
    supports_language,
)
from stt_eval.runner import RunConfig, Runner
from stt_eval.store import ResultStore

STEPS = ["1 · Upload", "2 · Configure", "3 · Credentials", "4 · Run", "5 · Review", "6 · Export"]

st.set_page_config(page_title="STT Evaluation Studio", page_icon="🎙️", layout="wide")

# Credentials from a local, git-ignored .env pre-fill step 3. Values live in
# process memory only — never in the results database, logs or exports.
ENV_VALUES = env.load()


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
        "api_keys": {
            provider: env.provider_key(provider) for provider in env.PROVIDER_ENV_VARS
        },
        "judge_backend": "anthropic",
        "judge_key": env.judge_key("anthropic"),
        "judge_model": DEFAULT_JUDGE_MODEL,
        "judge_effort": "medium",
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "concurrency": DEFAULT_CONCURRENCY,
        "judge_concurrency": DEFAULT_CONCURRENCY,
        "llm_sample_rate": 1.0,
        "wer_threshold": 0.25,
        "semantic_threshold": 0.15,
        "probe_result": None,
        "inline_ground_truth": {},
        "dataset": None,
        "runner": None,
        "uploaded_audio": [],
        "recorded_audio": [],
        "mic_round": 0,
        "mic_preview": {},
        "mic_providers": [],
        "mic_chosen_text": "",
        "input_source": "🎤 Microphone",
        "clip_languages": {},
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
        for key, empty in {
            "dataset": None,
            "runner": None,
            "uploaded_audio": [],
            "recorded_audio": [],
            "ground_truth_map": {},
            "inline_ground_truth": {},
        }.items():
            st.session_state[key] = empty
        st.session_state.mic_round += 1  # reset the recorder widget too
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
    st.header("1 · Audio and ground truth")
    st.write(
        "Record a clip or upload files, give each one its correct transcript, "
        "then continue. Uploading a `ground_truth.csv` instead is under options."
    )

    recording = None
    clean_id = ""
    duplicate = False
    recorded = st.session_state.recorded_audio

    # --- SOURCE card ------------------------------------------------------
    with st.container(border=True):
        st.markdown("##### SOURCE")
        source_columns = st.columns([2, 3])
        with source_columns[0]:
            source = st.segmented_control(
                "Source",
                ["🎤 Microphone", "📁 Upload Audio"],
                default=st.session_state.input_source,
                key="input_source_control",
                label_visibility="collapsed",
            ) or st.session_state.input_source
            st.session_state.input_source = source
        with source_columns[1]:
            st.caption(
                "Record a clip and transcribe it below."
                if source.endswith("Microphone")
                else "WAV works end to end. MP3/M4A/FLAC/OGG are transcoded first."
            )

        # Audio language lives here as well as in step 2 — you need it before
        # transcribing a recording, and both controls write the same state.
        labels = {lang.code: lang.label for lang in LANGUAGES}
        codes = list(labels)
        language_columns = st.columns([2, 3])
        with language_columns[0]:
            st.session_state.language = st.selectbox(
                "Audio language",
                codes,
                index=codes.index(st.session_state.language)
                if st.session_state.language in codes
                else 0,
                format_func=lambda code: labels[code],
                key="lang_step1",
            )

        if source.endswith("Upload Audio"):
            uploads = st.file_uploader(
                "Audio files",
                type=SUPPORTED_UPLOAD_EXTENSIONS,
                accept_multiple_files=True,
            )
            if uploads:
                st.session_state.uploaded_audio = [
                    (item.name, item.getvalue()) for item in uploads
                ]
        else:
            mic_columns = st.columns([2, 3])
            with mic_columns[0]:
                clip_name = st.text_input(
                    "Clip id",
                    value=f"rec{len(recorded) + 1}",
                    key=f"mic_name_{st.session_state.mic_round}",
                    help="Becomes the clip's id, exactly like a filename's stem.",
                )
            with mic_columns[1]:
                recording = st.audio_input(
                    "Record",
                    key=f"mic_input_{st.session_state.mic_round}",
                    help="Your browser asks for microphone permission the first time.",
                )

            clean_id = clip_id_for(clip_name or "")
            taken = {clip_id_for(name) for name, _ in recorded} | {
                clip_id_for(name) for name, _ in (st.session_state.uploaded_audio or [])
            }
            duplicate = clean_id in taken
            if duplicate:
                st.warning(f"`{clean_id}` is already taken — pick a different id.")

    # --- TRANSCRIPT card --------------------------------------------------
    if st.session_state.input_source.endswith("Microphone"):
        with st.container(border=True):
            header = st.columns([4, 1, 1])
            header[0].markdown("##### Transcript")

            with_keys = [
                key
                for key in env.configured_providers()
                if supports_language(key, st.session_state.language)
            ]
            if with_keys:
                chosen_providers = st.multiselect(
                    "Recognise with",
                    with_keys,
                    default=[
                        key
                        for key in (st.session_state.mic_providers or with_keys[:2])
                        if key in with_keys
                    ],
                    format_func=provider_label,
                    key="mic_providers_select",
                    help="Every provider you pick transcribes the same recording, so "
                    "you can see them disagree before committing to a full run.",
                )
                st.session_state.mic_providers = chosen_providers
            else:
                chosen_providers = []
                st.caption(
                    f"No provider key found that supports {labels[st.session_state.language]}. "
                    "Add one to `.env` to recognise from here."
                )

            def _transcribe_all(raw: bytes) -> dict:
                """Run every chosen provider over the same audio, concurrently.

                Everything the worker threads need is read from session state
                *here*, on the main thread, and closed over as plain values.
                `st.session_state` is only bound on the script-run thread, so
                touching it inside a pool worker raises AttributeError.
                """
                sample_rate = st.session_state.sample_rate
                language = st.session_state.language
                clip_audio = normalize(raw, "recording.wav", sample_rate=sample_rate)

                def one(provider_key: str) -> tuple[str, dict]:
                    try:
                        client = build(provider_key, env.provider_key(provider_key))
                        outcome = client.transcribe(clip_audio.wav_bytes, language)
                        return provider_key, {
                            "text": outcome.text,
                            "latency": outcome.latency_seconds,
                        }
                    except (AudioError, ProviderError) as exc:
                        return provider_key, {"error": str(exc)}
                    except Exception as exc:  # noqa: BLE001 - one provider must not sink the rest
                        return provider_key, {"error": f"{type(exc).__name__}: {exc}"}

                with ThreadPoolExecutor(max_workers=max(1, len(chosen_providers))) as pool:
                    outcomes = dict(pool.map(one, chosen_providers))
                return {
                    "by_provider": outcomes,
                    "seconds": clip_audio.duration_seconds,
                    "audio_id": hashlib.sha1(raw).hexdigest(),
                }

            # Transcribe as soon as a new recording lands — no button press, so
            # the transcript is simply there when you stop speaking.
            preview = st.session_state.mic_preview
            if recording is not None and chosen_providers:
                audio_id = hashlib.sha1(recording.getvalue()).hexdigest()
                stale = preview.get("audio_id") != audio_id or set(
                    preview.get("by_provider", {})
                ) != set(chosen_providers)
                if stale:
                    with st.spinner(
                        f"Recognising with {len(chosen_providers)} provider(s)…"
                    ):
                        st.session_state.mic_preview = _transcribe_all(recording.getvalue())
                    st.rerun()

            if header[1].button(
                "🎙️", help="Transcribe again", disabled=recording is None or not chosen_providers
            ):
                st.session_state.mic_preview = _transcribe_all(recording.getvalue())
                st.rerun()

            if header[2].button("✕", help="Clear the transcript"):
                st.session_state.mic_preview = {}
                st.session_state.mic_chosen_text = ""
                st.rerun()

            preview = st.session_state.mic_preview
            by_provider = preview.get("by_provider", {})

            if not by_provider:
                st.text_area(
                    "Transcript",
                    value="",
                    placeholder="Recognition results will appear here",
                    height=120,
                    key=f"mic_empty_{st.session_state.mic_round}",
                    label_visibility="collapsed",
                    disabled=True,
                )
            else:
                st.caption(f"{preview.get('seconds', 0):.1f}s of audio")
                for provider_key in chosen_providers:
                    outcome = by_provider.get(provider_key, {})
                    with st.container(border=True):
                        row = st.columns([3, 1])
                        row[0].markdown(f"**{provider_label(provider_key)}**")
                        if outcome.get("error"):
                            st.error(outcome["error"])
                            continue
                        row[1].caption(f"{outcome.get('latency', 0):.2f}s")
                        st.write(outcome.get("text") or "_(returned nothing)_")
                        if st.button(
                            "Use this as ground truth",
                            key=f"use_{provider_key}_{st.session_state.mic_round}",
                            disabled=not (outcome.get("text") or "").strip(),
                        ):
                            st.session_state.mic_chosen_text = outcome.get("text", "")
                            st.rerun()

                texts = [
                    (outcome.get("text") or "").strip()
                    for outcome in by_provider.values()
                    if not outcome.get("error")
                ]
                if len(set(texts)) > 1:
                    st.info(
                        "The providers disagree on this clip — which is exactly the "
                        "kind of clip worth having in your dataset."
                    )

            st.markdown("**Ground truth for this clip**")
            draft = st.text_area(
                "Ground truth",
                value=st.session_state.mic_chosen_text,
                placeholder="Correct the transcript above, or type what you actually said",
                height=110,
                key=f"mic_draft_{st.session_state.mic_round}",
                label_visibility="collapsed",
            )

            # Save as you type. Requiring a button press here meant a typed
            # transcript silently failed validation later, which looked like
            # the app losing the text rather than never having taken it.
            if clean_id and (draft or "").strip():
                st.session_state.inline_ground_truth = {
                    **(st.session_state.inline_ground_truth or {}),
                    clean_id: draft.strip(),
                }
                st.caption(f"Saved as the ground truth for `{clean_id}`.")

            actions = st.columns([1, 1])
            with actions[0]:
                if st.button(
                    "Add recording to dataset",
                    type="primary",
                    disabled=recording is None or not clean_id or duplicate,
                    width="stretch",
                ):
                    st.session_state.recorded_audio = recorded + [
                        (f"{clean_id}.wav", recording.getvalue())
                    ]
                    # Tag the clip with the language it was recorded in, so a
                    # mixed-language dataset transcribes each clip correctly.
                    st.session_state.clip_languages = {
                        **(st.session_state.clip_languages or {}),
                        clean_id: st.session_state.language,
                    }
                    # Bump the widget keys so the recorder resets for the next clip.
                    st.session_state.mic_round += 1
                    st.session_state.mic_preview = {}
                    st.session_state.mic_chosen_text = ""
                    st.rerun()

            st.caption(
                "⚠️ A transcript used as its own ground truth scores near-zero error "
                "for that provider by construction, which makes the comparison "
                "meaningless. Correct every mistake above first — that correction "
                "*is* the ground truth."
            )

        if recorded:
            st.markdown("**Recorded clips**")
            clip_languages = st.session_state.clip_languages or {}
            for index, (name, raw) in enumerate(list(recorded)):
                recorded_id = clip_id_for(name)
                row = st.columns([2, 4, 1])
                tagged = clip_languages.get(recorded_id)
                row[0].markdown(
                    f"`{recorded_id}`"
                    + (f"<br><small>{labels.get(tagged, tagged)}</small>" if tagged else ""),
                    unsafe_allow_html=True,
                )
                row[1].audio(raw)
                if row[2].button("Remove", key=f"drop_rec_{index}"):
                    st.session_state.recorded_audio = [
                        item for position, item in enumerate(recorded) if position != index
                    ]
                    st.session_state.clip_languages = {
                        key: value
                        for key, value in clip_languages.items()
                        if key != recorded_id
                    }
                    st.rerun()

    # Uploads and recordings feed one dataset; nothing downstream knows the difference.
    audio_files = list(st.session_state.uploaded_audio or []) + list(
        st.session_state.recorded_audio or []
    )
    typed_truth = st.session_state.inline_ground_truth or {}
    recorded_ids = [clip_id_for(name) for name, _ in (st.session_state.recorded_audio or [])]
    uploaded_ids = [clip_id_for(name) for name, _ in (st.session_state.uploaded_audio or [])]

    # The recording panel already collects the transcript, so a clip captured
    # here needs nothing further. Only ask about ground truth for clips that
    # still lack it — usually uploads, which cannot carry their own.
    needs_truth = [
        clip_id
        for clip_id in recorded_ids + uploaded_ids
        if not (typed_truth.get(clip_id) or "").strip()
    ]

    if audio_files and not needs_truth:
        ready = st.columns([3, 1])
        with ready[0]:
            st.success(
                f"**{len(audio_files)} clip(s) ready** — every one has ground truth."
            )
            st.caption(
                ", ".join(f"`{clip_id}`" for clip_id in recorded_ids + uploaded_ids)
            )
        with ready[1]:
            if st.button("Validate and continue →", type="primary", width="stretch", key="quick_go"):
                with st.spinner("Decoding audio and matching ground truth…"):
                    dataset = build_dataset(
                        audio_files,
                        {
                            clip_id: text
                            for clip_id, text in typed_truth.items()
                            if clip_id in recorded_ids + uploaded_ids
                        },
                        ground_truth_source="the transcripts you typed",
                        sample_rate=st.session_state.sample_rate,
                        languages=st.session_state.clip_languages,
                    )
                st.session_state.dataset = dataset
                if dataset.is_runnable:
                    goto(1)
                for message in dataset.errors:
                    st.error(message)

    with st.expander("Audio and ground-truth options", expanded=bool(needs_truth)):
        rates = list(SAMPLE_RATE_OPTIONS)
        st.session_state.sample_rate = st.radio(
            "Normalise every clip to",
            rates,
            index=rates.index(st.session_state.sample_rate)
            if st.session_state.sample_rate in rates
            else rates.index(DEFAULT_SAMPLE_RATE),
            format_func=lambda rate: SAMPLE_RATE_OPTIONS[rate],
            help=(
                "Every provider in the run receives audio at this rate, so the numbers "
                "stay comparable. Benchmark at the rate your production audio actually "
                "arrives at — for phone traffic that is 8 kHz."
            ),
        )
        if st.session_state.sample_rate <= 8_000:
            st.caption(
                "Narrowband run: Deepgram and Google switch to their telephony models "
                "automatically."
            )

        if needs_truth:
            st.warning(
                "Still need ground truth for "
                + ", ".join(f"`{clip_id}`" for clip_id in needs_truth)
            )

        mode = st.radio(
            "Ground truth",
            ["CSV file (id, text)", "Type inline per clip"],
            horizontal=True,
            index=1 if recorded_ids and not uploaded_ids else 0,
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

            # A transcript typed in the recording panel counts even in CSV mode —
            # otherwise text you just corrected by hand would be thrown away for
            # want of a row in a file you never needed for that clip.
            typed = {
                clip_id_for(name): (st.session_state.inline_ground_truth or {}).get(
                    clip_id_for(name), ""
                )
                for name, _ in st.session_state.recorded_audio
            }
            typed = {key: value for key, value in typed.items() if value.strip()}
            if typed:
                ground_truth = {**ground_truth, **typed}
                st.caption(
                    "Using the transcripts you typed for "
                    + ", ".join(f"`{key}`" for key in sorted(typed))
                    + " — no CSV row needed for those."
                )
            missing_rows = [
                clip_id_for(name)
                for name, _ in st.session_state.recorded_audio
                if clip_id_for(name) not in ground_truth
            ]
            if missing_rows:
                st.caption(
                    "Still need ground truth for "
                    + ", ".join(f"`{key}`" for key in missing_rows)
                    + " — type it in the recording panel above, or add a CSV row."
                )
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
                sample_rate=st.session_state.sample_rate,
                languages=st.session_state.clip_languages,
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
                f"{dataset.total_duration_minutes:.2f} minutes of audio · "
                f"normalised to {dataset.sample_rate // 1000} kHz mono."
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
        help="Providers are filtered to those that support the language you pick.",
    )
    language = st.session_state.language

    # Clips recorded with their own language override the run's, so a provider
    # is only usable if it supports every language actually in the dataset.
    languages_in_use = dataset.languages_in_use(language)
    available = [
        key
        for key in PROVIDER_CLASSES
        if all(supports_language(key, code) for code in languages_in_use)
    ]
    unavailable = [key for key in PROVIDER_CLASSES if key not in available]

    if len(languages_in_use) > 1:
        st.info(
            "This dataset is multilingual — "
            + ", ".join(sorted(labels.get(code, code) for code in languages_in_use))
            + ". Each clip is transcribed in its own language, and only providers "
            "that support all of them are listed."
        )

    if language in CHARACTER_ORIENTED_LANGUAGES:
        st.info(
            f"{labels[language]} has ambiguous word boundaries — read **CER** as the "
            "primary deterministic metric here, not WER."
        )

    st.subheader("Providers")
    st.caption(
        f"{len(available)} of {len(PROVIDER_CLASSES)} providers support "
        f"{labels[language]}. " + LANGUAGE_SUPPORT_NOTE
    )
    # Default to providers you actually hold keys for, so step 3 does not open
    # by demanding a credential for a provider you never chose.
    with_keys = [key for key in available if key in env.configured_providers()]
    fallback = with_keys or [key for key in available if key != "mock"][:2]
    selected = st.multiselect(
        "Compare",
        available,
        default=[key for key in st.session_state.selected_providers if key in available]
        or fallback,
        format_func=provider_label,
        help="Providers with a key in your .env are selected by default.",
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

    st.subheader("Throughput")
    st.caption(
        "Transcription and judging run in separate worker pools, so a slow judge "
        "call never occupies a worker that a provider could be using. Each pool "
        "has its own limit because each talks to a different service."
    )
    throughput = st.columns(2)
    with throughput[0]:
        st.session_state.concurrency = st.slider(
            "Provider requests in flight",
            min_value=1,
            max_value=MAX_CONCURRENCY,
            value=st.session_state.concurrency,
            help="Per provider. Lower this if a provider starts rate-limiting your key.",
        )
    with throughput[1]:
        st.session_state.judge_concurrency = st.slider(
            "Judge calls in flight",
            min_value=1,
            max_value=MAX_CONCURRENCY,
            value=st.session_state.judge_concurrency,
            disabled=not any(key in LLM_KEYS for key in chosen),
            help="Lower this on a rate-limited or free-tier judge endpoint.",
        )

    if any(key in LLM_KEYS for key in chosen):
        st.subheader("Sampling")
        st.caption(
            "Deterministic metrics run on every clip — they are free once the "
            "transcript exists. The LLM metrics cost money per clip, and on a "
            "large dataset most of that confirms what WER already showed. Judging "
            "a stratified slice instead keeps every confidence band represented "
            "while cutting the bill."
        )
        st.session_state.llm_sample_rate = (
            st.slider(
                "Judge this share of clips",
                min_value=5,
                max_value=100,
                step=5,
                value=int(st.session_state.llm_sample_rate * 100),
                format="%d%%",
                help=(
                    "Stratified by deterministic error rate, so exact matches, "
                    "near matches, diverging and poor clips are all represented. "
                    "100% judges everything."
                ),
            )
            / 100.0
        )
        pairs = len(dataset.clips) * len(selected)
        judged = max(1, round(pairs * st.session_state.llm_sample_rate))
        if st.session_state.llm_sample_rate < 1.0:
            st.caption(
                f"≈{judged} of {pairs} pairs judged. The leaderboard's LLM columns "
                "then describe the sample, not the whole dataset — WER and CER still "
                "cover everything."
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
        "never written to the results database, the logs, or an exported file."
    )

    from_env = env.configured_providers()
    if from_env or env.judge_key():
        loaded = [provider_label(key) for key in from_env]
        if env.judge_key():
            loaded.append("LLM judge")
        st.info(
            "Pre-filled from your local `.env`: " + ", ".join(loaded)
            + ". Edit a field below to override it for this run only."
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
            value=keys.get(provider_key, "") or env.provider_key(provider_key),
            type="password",
            help=credential_hint(provider_key),
            key=f"key_{provider_key}",
        )
        source = env.provider_key(provider_key)
        if source and keys[provider_key] == source:
            st.caption(
                f"from `.env` · `{env.PROVIDER_ENV_VARS[provider_key]}` = "
                f"`{env.mask(source)}`"
            )
    st.session_state.api_keys = keys

    judge_needed = uses_judge()
    if judge_needed:
        st.divider()
        st.subheader("LLM judge")
        st.caption(
            "Every meaning-aware metric routes through one judge client against "
            "one shared system prompt, so a verdict means the same thing whichever "
            "backend and model you pick."
        )

        backends = list(JUDGE_BACKENDS)
        st.session_state.judge_backend = st.radio(
            "Backend",
            backends,
            index=backends.index(st.session_state.judge_backend)
            if st.session_state.judge_backend in backends
            else 0,
            format_func=lambda key: JUDGE_BACKENDS[key],
            horizontal=True,
        )
        backend = st.session_state.judge_backend
        env_var = env.JUDGE_ENV_VARS[backend]
        env_value = env.judge_key(backend)

        st.session_state.judge_key = st.text_input(
            f"{JUDGE_BACKENDS[backend]} API key",
            value=st.session_state.judge_key or env_value,
            type="password",
            help="Used only for the enabled LLM metrics.",
            key=f"judge_key_{backend}",
        )
        if env_value and st.session_state.judge_key == env_value:
            st.caption(f"from `.env` · `{env_var}` = `{env.mask(env_value)}`")
        elif not env_value:
            st.caption(f"Set `{env_var}` in `.env` to pre-fill this.")

        columns = st.columns(2)
        with columns[0]:
            if backend == "openrouter":
                st.session_state.judge_model = st.selectbox(
                    "Judge model",
                    OPENROUTER_MODELS,
                    index=OPENROUTER_MODELS.index(st.session_state.judge_model)
                    if st.session_state.judge_model in OPENROUTER_MODELS
                    else 0,
                    accept_new_options=True,
                    help=(
                        "Any OpenRouter model slug works — type one in if it is not "
                        "listed. Reasoning models judge borderline rewordings more "
                        "reliably; they also cost more and take longer."
                    ),
                )
            else:
                st.session_state.judge_model = st.selectbox(
                    "Judge model",
                    JUDGE_MODELS,
                    index=JUDGE_MODELS.index(st.session_state.judge_model)
                    if st.session_state.judge_model in JUDGE_MODELS
                    else 0,
                )
        with columns[1]:
            efforts = ["low", "medium", "high"]
            st.session_state.judge_effort = st.selectbox(
                "Reasoning effort",
                efforts,
                index=efforts.index(st.session_state.judge_effort),
                help=(
                    "How much the judge thinks before answering. Lower is cheaper "
                    "and faster; higher is more careful on borderline segments."
                ),
            )

        if backend == "openrouter":
            if st.session_state.judge_model.endswith(FREE_TIER_SUFFIX):
                st.warning(FREE_TIER_NOTE)
            st.caption(
                "OpenRouter reports the real cost of each judge call, so the judge "
                "spend shown after the run is actual, not estimated."
            )

        if st.button("Test judge connection"):
            try:
                create_judge(
                    backend=backend,
                    api_key=st.session_state.judge_key,
                    model=st.session_state.judge_model,
                    effort=st.session_state.judge_effort,
                ).check()
                st.success(f"{st.session_state.judge_model} reachable and returning valid JSON.")
            except JudgeError as exc:
                st.error(str(exc))

        with st.expander("The judge's system prompt"):
            st.caption(
                "Shared by every metric and every backend. Edit it in "
                "`stt_eval/prompts.py`."
            )
            st.code(JUDGE_SYSTEM_PROMPT, language="markdown")

    missing_providers = [p for p in selected if requires_key(p) and not keys.get(p)]
    judge_missing = judge_needed and not st.session_state.judge_key

    if missing_providers:
        st.warning(
            "Still needed: "
            + ", ".join(provider_label(p) for p in missing_providers)
            + ". These are selected in step 2 but have no key — either paste one "
            "above, or drop them from this run."
        )
        if st.button(
            f"Drop {', '.join(provider_label(p) for p in missing_providers)} from this run"
        ):
            st.session_state.selected_providers = [
                p for p in selected if p not in missing_providers
            ]
            st.rerun()
    if judge_missing:
        st.info("Still needed: a key for the LLM judge, or turn the LLM metrics off in step 2.")

    blocked = bool(missing_providers or judge_missing)
    if st.button("Continue to run →", type="primary", disabled=blocked):
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
                judge = create_judge(
                    backend=st.session_state.judge_backend,
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
                    judge_concurrency=st.session_state.judge_concurrency,
                    llm_sample_rate=st.session_state.llm_sample_rate,
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

    stage_label = {
        "transcribing": "Stage 1 · transcribing",
        "scoring": "Stage 2 · judging",
        "done": "Finished",
    }.get(progress.stage, progress.stage)
    st.progress(
        progress.fraction,
        text=f"{stage_label} — {progress.transcribed + progress.skipped}/{progress.total} transcribed"
        + (f", {progress.scored}/{progress.to_score} judged" if progress.to_score else ""),
    )
    stats = st.columns(5)
    stats[0].metric("Transcribed", progress.transcribed)
    stats[1].metric("Judged", progress.scored)
    stats[2].metric("Reused", progress.skipped)
    stats[3].metric("Failed", progress.failed)
    stats[4].metric("Elapsed", f"{progress.elapsed_seconds:.0f}s")
    if progress.sample_summary:
        st.caption(f"Judge sample — {progress.sample_summary}")

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
        if runner.judge is not None and runner.judge.usage.calls:
            usage = runner.judge.usage
            judge_columns = st.columns(3)
            judge_columns[0].metric("Judge calls", usage.calls)
            judge_columns[1].metric(
                "Judge tokens",
                f"{usage.input_tokens + usage.output_tokens:,}",
                help=f"{usage.reasoning_tokens:,} of them reasoning tokens"
                if usage.reasoning_tokens
                else None,
            )
            judge_columns[2].metric(
                "Judge spend", f"${runner.judge.estimated_cost_usd():.4f}"
            )
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

    # --- Signals ----------------------------------------------------------
    st.subheader("Signals")
    threshold_columns = st.columns([1, 1, 2])
    with threshold_columns[0]:
        st.session_state.wer_threshold = st.number_input(
            "WER threshold", min_value=0.0, max_value=1.0, step=0.05,
            value=float(st.session_state.wer_threshold),
        )
    with threshold_columns[1]:
        st.session_state.semantic_threshold = st.number_input(
            "Semantic error threshold", min_value=0.0, max_value=1.0, step=0.05,
            value=float(st.session_state.semantic_threshold),
        )
    with threshold_columns[2]:
        judged, total_scored = signals_module.coverage(results)
        st.caption(
            f"Meaning-aware metrics cover {judged} of {total_scored} transcribed "
            "pairs. WER spikes point at acoustic or model degradation; a semantic "
            "error rate that rises *without* a matching WER rise points at "
            "meaning-critical failures that WER cannot see."
        )

    thresholds = signals_module.Thresholds(
        wer=st.session_state.wer_threshold,
        semantic_error=st.session_state.semantic_threshold,
    )
    found = signals_module.evaluate(results, summaries, thresholds=thresholds)
    if not found:
        st.success("No threshold breaches and no divergence between WER and meaning.")
    for signal in found:
        render = {"critical": st.error, "warning": st.warning, "info": st.info}[signal.severity]
        render(f"**{signal.title}**  \n{signal.detail}")
        if signal.pairs:
            st.caption(
                "Clips: "
                + ", ".join(
                    f"`{clip_id}`/{provider_label(provider)}" for clip_id, provider in signal.pairs[:12]
                )
                + (f" and {len(signal.pairs) - 12} more" if len(signal.pairs) > 12 else "")
            )

    st.divider()
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
    with st.expander("⏱️ Streaming latency probe — partial emission, finals, RTF"):
        st.caption(
            "A batch request only tells you how long the whole call took. The "
            "metrics that decide whether an engine works in a voice agent — how "
            "fast partials appear while someone is still talking, and how quickly "
            "a phrase is locked in — only exist over a streaming connection. This "
            "replays a clip over the provider's WebSocket at real-time pace and "
            "timestamps every result as it arrives, so the numbers are what a live "
            "speaker would have experienced."
        )
        dataset_for_probe = st.session_state.dataset
        streamable = [
            key
            for key in st.session_state.selected_providers
            if key in streaming.STREAMING_PROVIDERS and env.provider_key(key)
        ]
        if not dataset_for_probe or not dataset_for_probe.clips:
            st.info("Load a dataset in step 1 to probe a clip.")
        elif not streamable:
            st.info(
                "No streaming-capable provider selected. Only "
                + ", ".join(sorted(streaming.STREAMING_PROVIDERS))
                + " has a streaming endpoint wired up here; the others expose batch "
                "HTTP only, so partial and finals latency cannot be measured for them."
            )
        else:
            probe_columns = st.columns([2, 2, 1])
            with probe_columns[0]:
                probe_clip = st.selectbox(
                    "Clip", [clip.clip_id for clip in dataset_for_probe.clips], key="probe_clip"
                )
            with probe_columns[1]:
                probe_provider = st.selectbox(
                    "Provider", streamable, format_func=provider_label, key="probe_provider"
                )
            with probe_columns[2]:
                probe_pacing = st.selectbox(
                    "Pacing", ["realtime", "fast"], key="probe_pacing",
                    help="Real-time pacing measures latency as a live speaker would "
                    "see it. Fast sends everything at once and measures RTF instead.",
                )

            if st.button("Measure", type="primary"):
                clip = dataset_for_probe.by_id(probe_clip)
                with st.spinner(
                    f"Streaming {clip.duration_seconds:.1f}s of audio to "
                    f"{provider_label(probe_provider)}…"
                ):
                    st.session_state.probe_result = streaming.measure(
                        provider=probe_provider,
                        api_key=env.provider_key(probe_provider),
                        wav_bytes=clip.audio.wav_bytes,
                        language=clip.language_for(st.session_state.language),
                        pacing=probe_pacing,
                    )
                st.rerun()

            probe = st.session_state.probe_result
            if probe is not None:
                if not probe.ok:
                    st.error(probe.error)
                elif not probe.events:
                    st.warning(
                        "The engine returned no transcript events for this clip, so "
                        "there is nothing to time. Usually that means silence, noise "
                        "or a language mismatch rather than a latency problem."
                    )
                else:
                    figures = st.columns(4)
                    figures[0].metric(
                        "Time to first partial",
                        f"{probe.time_to_first_partial:.2f}s"
                        if probe.time_to_first_partial is not None
                        else "—",
                        help="How long before the engine says anything at all.",
                    )
                    figures[1].metric(
                        "Partial emission",
                        f"{probe.partial_emission_latency:.2f}s"
                        if probe.partial_emission_latency is not None
                        else "—",
                        help="Median delay of incremental updates behind the speech they carry.",
                    )
                    figures[2].metric(
                        "Finals latency",
                        f"{probe.finals_latency:.2f}s"
                        if probe.finals_latency is not None
                        else "—",
                        help="Median delay between a phrase ending and being locked in — "
                        "this is what gates a voice agent's turn-taking.",
                    )
                    figures[3].metric(
                        "RTF",
                        f"{probe.rtf:.2f}" if probe.rtf is not None else "n/a",
                        help="Processing time over audio duration. Only measured under "
                        "fast pacing — under real-time pacing the clock is the pacing.",
                    )
                    st.caption(
                        f"{len(probe.partials)} partial(s), {len(probe.finals)} final(s) "
                        f"over {probe.audio_seconds:.1f}s of audio."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "arrived at": round(event.arrival_offset, 2),
                                    "covers audio": f"{event.audio_start:.1f}–{event.audio_end:.1f}s",
                                    "behind by": round(event.latency, 2),
                                    "final": event.is_final,
                                    "text": event.text,
                                }
                                for event in probe.events
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
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
        "sample_rate_hz": st.session_state.sample_rate,
        "providers": ", ".join(st.session_state.selected_providers),
        "metrics": ", ".join(metrics),
        "judge_backend": st.session_state.judge_backend if uses_judge() else "not used",
        "judge_model": st.session_state.judge_model if uses_judge() else "not used",
        "judge_effort": st.session_state.judge_effort if uses_judge() else "not used",
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
