"""End-to-end feature test against the real provider and judge APIs.

    py feature_test.py

Unlike `smoke_test.py` (offline, deterministic) and `ui_test.py` (renders the
pages), this exercises the whole product against live services: real speech,
real transcription, real judging, real streaming.

Speech is synthesised locally with the Windows speech engine so the ground
truth is known exactly — you cannot measure word error rate without knowing
what was actually said — and so the test costs nothing to run. ElevenLabs TTS
is used instead where the local engine is unavailable and the account has
credit. Everything else uses whichever keys are present in `.env`; missing ones
are skipped, not failed.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests

from stt_eval import env, export, report, signals, streaming
from stt_eval.dataset import build_dataset
from stt_eval.judge import JudgeError, create_judge
from stt_eval.metrics import METRICS_BY_KEY
from stt_eval.providers import provider_label
from stt_eval.runner import RunConfig, Runner
from stt_eval.store import ResultStore

RUN_ID = "featuretest"

# Sentences chosen to exercise the metrics ladder: numbers that recognisers
# render as digits, a name they can mangle, and a negation they can drop.
SCRIPT = {
    "spoken1": "The invoice is for five hundred rupees and it is due on Monday.",
    "spoken2": "Please tell doctor Mehta that the payment was not approved.",
}

TTS_VOICE = "21m00Tcm4TlvDq8ikWAM"  # a stock ElevenLabs voice
PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    (PASSED if condition else FAILED).append(label)
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
    return condition


def skip(label: str, why: str) -> None:
    SKIPPED.append(label)
    print(f"  SKIP  {label} — {why}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def synthesize_locally(text: str) -> bytes | None:
    """Windows' built-in speech engine, written straight to a WAV file.

    Free and offline, and its output is recognisable enough that real providers
    transcribe it accurately — which is all this needs.
    """
    if sys.platform != "win32":
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "speech.wav"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f'$s.SetOutputToWaveFile("{target}"); '
            f'$s.Speak("{text}"); '
            "$s.Dispose()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True, capture_output=True, timeout=120,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return target.read_bytes() if target.exists() else None


def synthesize_remotely(text: str, api_key: str) -> bytes | None:
    """ElevenLabs TTS as a fallback. Requests PCM rather than MP3 so the test
    does not depend on an MP3 decoder being installed."""
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE}",
        params={"output_format": "pcm_16000"},
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=120,
    )
    if response.status_code >= 400:
        print(f"  (ElevenLabs TTS unavailable: HTTP {response.status_code})")
        return None
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(response.content)
    return buffer.getvalue()


def synthesize(text: str, api_key: str) -> bytes | None:
    return synthesize_locally(text) or (
        synthesize_remotely(text, api_key) if api_key else None
    )


def main() -> int:
    env.load()
    providers = env.configured_providers()
    judge_backends = env.configured_judge_backends()
    print(f"providers with keys: {providers or 'none'}")
    print(f"judge backends:      {judge_backends or 'none'}")

    # -- speech ------------------------------------------------------------
    section("1. Test audio (synthesised speech with known ground truth)")
    uploads = []
    for clip_id, text in SCRIPT.items():
        wav = synthesize(text, env.provider_key("elevenlabs"))
        if wav is None:
            print("  No speech synthesiser available — cannot measure accuracy.")
            return 1
        uploads.append((f"{clip_id}.wav", wav))
        check(f"synthesised {clip_id}", len(wav) > 10_000, f"{len(wav):,} bytes")

    # -- dataset -----------------------------------------------------------
    section("2. Dataset validation and audio normalisation")
    dataset = build_dataset(uploads, SCRIPT, languages={"spoken2": "en-US"})
    check("both clips matched to ground truth", len(dataset.clips) == 2 and not dataset.errors)
    check("per-clip language honoured", dataset.by_id("spoken2").language == "en-US")
    check("run language used as fallback", dataset.by_id("spoken1").language_for("en-IN") == "en-IN")
    check("normalised to 16 kHz mono", dataset.clips[0].audio.sample_rate == 16_000)

    narrowband = build_dataset(uploads, SCRIPT, sample_rate=8_000)
    check("8 kHz telephony normalisation", narrowband.clips[0].audio.sample_rate == 8_000)

    # -- the run -----------------------------------------------------------
    section("3. Two-stage run against live providers")
    run_providers = [key for key in providers if key != "google"]  # google's API is disabled
    if "google" in providers:
        skip("google in the run", "Speech-to-Text API not enabled on the project")

    llm_metrics = ["semantic_match", "intent_entity"] if judge_backends else []
    metrics = ["wer", "cer"] + llm_metrics
    judge = None
    if judge_backends:
        judge = create_judge(
            backend="openrouter",
            api_key=env.judge_key("openrouter"),
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            effort="low",
        )
        try:
            judge.check()
            check("judge reachable", True, judge.model)
        except JudgeError as exc:
            check("judge reachable", False, str(exc)[:80])
            judge, llm_metrics = None, []
            metrics = ["wer", "cer"]

    store = ResultStore()
    store.delete_run(RUN_ID)
    config = RunConfig(
        run_id=RUN_ID,
        language="en-IN",
        provider_keys=run_providers,
        api_keys={key: env.provider_key(key) for key in run_providers},
        enabled_metrics=metrics,
        per_provider_concurrency=3,
        judge_concurrency=2,
        llm_sample_rate=0.5 if llm_metrics else 1.0,  # exercises stratified sampling
    )
    runner = Runner(config=config, dataset=dataset, store=store, judge=judge)
    started = time.perf_counter()
    runner.start()
    runner._thread.join(timeout=600)  # noqa: SLF001
    elapsed = time.perf_counter() - started

    progress, results = runner.snapshot()
    check("run completed", progress.done and not progress.fatal_error, f"{elapsed:.1f}s")
    check(
        "every pair attempted",
        len(results) == len(dataset.clips) * len(run_providers),
        f"{len(results)} results",
    )
    check("transcripts produced", all(r.prediction for r in results if r.ok))
    if llm_metrics:
        check("sampling limited the judged set", progress.to_score < len(results), progress.sample_summary)

    print("\n  transcripts:")
    for result in sorted(results, key=lambda r: (r.clip_id, r.provider)):
        if result.ok:
            print(f"    {result.clip_id} · {provider_label(result.provider):18} {result.prediction[:70]!r}")
        else:
            print(f"    {result.clip_id} · {provider_label(result.provider):18} FAILED: {result.error[:60]}")

    # -- metrics -----------------------------------------------------------
    section("4. Metrics and leaderboard")
    summaries = report.summarize(results, enabled_metrics=metrics)
    check("a summary per provider", set(summaries) == set(run_providers))
    check("pooled WER computed", all(s.metrics.get("wer") is not None for s in summaries.values()))
    check("RTF computed", any(s.rtf is not None for s in summaries.values()))
    check("latency percentiles computed", all(s.latency.get("p50") is not None for s in summaries.values()))

    def figure(value: float | None, places: int = 3) -> str:
        return f"{value:.{places}f}" if value is not None else "-"

    print("\n  leaderboard (ranked by WER):")
    print(f"    {'provider':20} {'WER':>7} {'CER':>7} {'RTF':>6} {'p50 s':>7} {'cost $':>9}")
    for provider in report.rank(summaries, metric_key="wer"):
        item = summaries[provider]
        print(
            f"    {provider_label(provider):20} "
            f"{figure(item.metrics.get('wer')):>7} "
            f"{figure(item.metrics.get('cer')):>7} "
            f"{figure(item.rtf, 2):>6} "
            f"{figure(item.latency.get('p50'), 2):>7} "
            f"{item.estimated_cost_usd:>9.5f}"
        )
        for key in llm_metrics:
            value = item.metrics.get(key)
            if value is not None:
                print(f"      {METRICS_BY_KEY[key].label}: {value:.2f}")

    # -- signals -----------------------------------------------------------
    section("5. Signals (WER vs meaning divergence)")
    found = signals.evaluate(results, summaries, thresholds=signals.Thresholds())
    judged, total = signals.coverage(results)
    check("coverage reported", total > 0, f"{judged}/{total} pairs judged")
    check("signal evaluation runs", isinstance(found, list), f"{len(found)} signal(s)")
    for signal in found:
        print(f"    [{signal.severity}] {signal.title}")

    # -- streaming ---------------------------------------------------------
    section("6. Streaming latency (Deepgram WebSocket)")
    if "deepgram" not in providers:
        skip("streaming probe", "no Deepgram key")
    else:
        clip = dataset.clips[0]
        probe = streaming.measure(
            provider="deepgram",
            api_key=env.provider_key("deepgram"),
            wav_bytes=clip.audio.wav_bytes,
            language="en-IN",
            pacing="realtime",
        )
        if not probe.ok:
            check("streaming probe", False, probe.error[:90])
        else:
            check("stream produced events", bool(probe.events), f"{len(probe.partials)} partial(s), {len(probe.finals)} final(s)")
            if probe.events:
                check("partial emission latency measured", probe.partial_emission_latency is not None,
                      f"{probe.partial_emission_latency:.2f}s" if probe.partial_emission_latency else "")
                check("finals latency measured", probe.finals_latency is not None,
                      f"{probe.finals_latency:.2f}s" if probe.finals_latency else "")
                check("RTF withheld under real-time pacing", probe.rtf is None)
                print(f"    streamed transcript: {probe.transcript[:80]!r}")

        fast = streaming.measure(
            provider="deepgram", api_key=env.provider_key("deepgram"),
            wav_bytes=dataset.clips[0].audio.wav_bytes, language="en-IN", pacing="fast",
        )
        if fast.ok:
            check("RTF measured under fast pacing", fast.rtf is not None,
                  f"{fast.rtf:.2f}" if fast.rtf else "")

    # -- resumability and export -------------------------------------------
    section("7. Resumability and export")
    succeeded = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    rerun = Runner(config=config, dataset=dataset, store=store, judge=judge)
    rerun.start()
    rerun._thread.join(timeout=300)  # noqa: SLF001
    second, _ = rerun.snapshot()
    # Successful pairs are reused; failed ones are deliberately retried, since
    # a provider that failed once may well succeed on a second attempt.
    check(
        "successful pairs reused",
        second.skipped == len(succeeded),
        f"{second.skipped} reused of {len(succeeded)} successful",
    )
    check(
        "failed pairs retried rather than cached",
        second.transcribed == len(failed),
        f"{second.transcribed} retried of {len(failed)} failed",
    )

    per_clip = report.per_clip_rows(results, enabled_metrics=metrics)
    summary_rows = report.summary_rows(summaries, enabled_metrics=metrics)
    check("per-clip export has RTF and language", "rtf" in per_clip[0] and "language" in per_clip[0])
    check("per-clip CSV written", len(export.per_clip_csv(per_clip)) > 0)
    check("summary CSV written", len(export.summary_csv(summary_rows)) > 0)
    check("Excel workbook written", len(export.workbook(per_clip, summary_rows, metadata={"run": RUN_ID})) > 0)

    if judge is not None:
        usage = judge.usage
        print(f"\n  judge: {usage.calls} calls, {usage.input_tokens + usage.output_tokens} tokens, "
              f"${judge.estimated_cost_usd():.5f}")

    store.delete_run(RUN_ID)
    store.close()

    section("Result")
    print(f"  {len(PASSED)} passed · {len(FAILED)} failed · {len(SKIPPED)} skipped")
    if FAILED:
        for label in FAILED:
            print(f"    FAILED: {label}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
