"""End-to-end offline check: synthesize audio, run the mock provider through the
runner, score the deterministic metrics, aggregate, and export.

    py smoke_test.py

No API keys and no network access are required — the LLM metrics are skipped.
"""

from __future__ import annotations

import io
import math
import sys
import tempfile
import wave
from pathlib import Path

from stt_eval import export, report
from stt_eval.audio import wav_sample_rate
from stt_eval.dataset import build_dataset, parse_ground_truth_csv
from stt_eval.metrics import evaluate_pair
from stt_eval.metrics.align import align, tokenize_words
from stt_eval.providers.deepgram import DeepgramProvider
from stt_eval.providers.google import GoogleProvider
from stt_eval.runner import RunConfig, Runner
from stt_eval.store import ResultStore

GROUND_TRUTH = {
    "clip1": "The invoice is for five hundred rupees and it is due on Monday",
    "clip2": "Please call doctor Mehta back tomorrow morning at ten",
    "clip3": "Um I actually think we should just cancel the order",
}


def make_wav(seconds: float, frequency: float = 220.0, rate: int = 44_100) -> bytes:
    frames = bytearray()
    for index in range(int(seconds * rate)):
        value = int(12000 * math.sin(2 * math.pi * frequency * index / rate))
        frames += value.to_bytes(2, "little", signed=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise SystemExit(1)


def main() -> int:
    print("alignment + deterministic metrics")
    alignment = align(tokenize_words("the cat sat on the mat"), tokenize_words("the cat sat on a mat"))
    check("one substitution detected", alignment.substitutions == 1)
    check("error rate is 1/6", abs(alignment.error_rate - 1 / 6) < 1e-9)

    evaluation = evaluate_pair(
        clip_id="clip1",
        provider="mock",
        ground_truth="five hundred rupees",
        prediction="500 rupees",
        enabled_metrics=["wer", "cer"],
        judge=None,
        language="en-IN",
    )
    check("WER penalises the reworded number", evaluation.value("wer") > 0)
    check("CER computed", evaluation.value("cer") is not None)

    print("language / provider matrix")
    from stt_eval.config import LANGUAGES
    from stt_eval.providers import PROVIDER_CLASSES, providers_for_language

    codes = [lang.code for lang in LANGUAGES]
    check("no duplicate language codes", len(codes) == len(set(codes)))
    check(
        "every language has at least one provider",
        all(providers_for_language(code) for code in codes),
    )
    check("Sarvam is Indic-only", "sarvam" not in providers_for_language("ja-JP"))
    check("Sarvam covers Hindi", "sarvam" in providers_for_language("hi-IN"))
    check(
        "provider language sets only use canonical codes",
        all(
            cls.supported_languages <= set(codes)
            for cls in PROVIDER_CLASSES.values()
            if cls.supported_languages
        ),
    )
    check(
        "Google maps Odia to its own code",
        GoogleProvider("k").language_code("od-IN") == "or-IN"
        and GoogleProvider("k").language_code("hi-IN") == "hi-IN",
    )

    print("credential loading")
    from stt_eval import env as env_module

    parsed = env_module.parse(
        "# comment\nexport DEEPGRAM_API_KEY=abc123\nSARVAM_API_KEY='quoted'\nBLANK=\n"
    )
    check("env parsing handles export/quotes/comments", parsed["DEEPGRAM_API_KEY"] == "abc123")
    check("quoted values unquoted", parsed["SARVAM_API_KEY"] == "quoted")
    check("masking hides the middle", env_module.mask("sk_abcdefghijklmnop") == "sk_ab…mnop")
    check(
        "every provider needing a key has an env var",
        all(
            key in env_module.PROVIDER_ENV_VARS
            for key in PROVIDER_CLASSES
            if key != "mock"
        ),
    )
    check(
        "no key material is persisted",
        "api_key" not in Path("stt_eval/store.py").read_text(encoding="utf-8"),
    )

    print("judge backends")
    from stt_eval.judge import (
        AnthropicJudge,
        JudgeError,
        OpenRouterJudge,
        _extract_json,
        create_judge,
    )
    from stt_eval.prompts import JUDGE_SYSTEM_PROMPT

    check("anthropic backend builds", isinstance(
        create_judge(backend="anthropic", api_key="k", model="claude-opus-5"), AnthropicJudge))
    check("openrouter backend builds", isinstance(
        create_judge(backend="openrouter", api_key="k", model="deepseek/deepseek-r1"),
        OpenRouterJudge))
    try:
        create_judge(backend="nope", api_key="k", model="m")
        check("unknown backend rejected", False)
    except JudgeError:
        check("unknown backend rejected", True)

    check("plain JSON parses", _extract_json('{"ok": true}')["ok"] is True)
    check(
        "fenced JSON parses",
        _extract_json('```json\n{"ok": true}\n```')["ok"] is True,
    )
    check(
        "JSON after reasoning preamble parses",
        _extract_json('Let me think about this.\n{"match": false, "reasoning": "x"}')
        ["match"] is False,
    )
    check(
        "braces inside strings do not break extraction",
        _extract_json('{"reasoning": "he said {yes}", "match": true}')["match"] is True,
    )
    try:
        _extract_json("not json at all")
        check("unparseable input raises", False)
    except JudgeError:
        check("unparseable input raises", True)

    judge = create_judge(backend="openrouter", api_key="k", model="deepseek/deepseek-r1")
    body = judge._body(system="s", prompt="p", schema={"type": "object"}, with_schema=True)
    check("reasoning effort sent", body["reasoning"]["effort"] == "medium")
    check("cost reporting requested", body["usage"]["include"] is True)
    check("schema requested when supported", "response_format" in body)
    fallback = judge._body(system="s", prompt="p", schema={"type": "object"}, with_schema=False)
    check("schema inlined in prompt on fallback", "response_format" not in fallback)
    check("fallback prompt carries the schema", "type" in fallback["messages"][1]["content"])

    check(
        "both backends share one system prompt",
        judge.system_prompt == JUDGE_SYSTEM_PROMPT
        and create_judge(backend="anthropic", api_key="k", model="m").system_prompt
        == JUDGE_SYSTEM_PROMPT,
    )
    check(
        "prompt covers STT-specific failure modes",
        all(
            term in JUDGE_SYSTEM_PROMPT.lower()
            for term in ("homophone", "negation", "hallucinat", "entity", "translation")
        ),
    )

    print("csv parsing")
    mapping, errors = parse_ground_truth_csv(
        b"id,text\nclip1,hello world\nclip2,\n", "ground_truth.csv"
    )
    check("valid row parsed", mapping == {"clip1": "hello world"})
    check("empty row reported", any("clip2" in message for message in errors))

    print("dataset validation")
    uploads = [(f"{clip_id}.wav", make_wav(1.0 + index * 0.5)) for index, clip_id in enumerate(GROUND_TRUTH)]
    dataset = build_dataset(uploads, GROUND_TRUTH)
    check("all clips matched", len(dataset.clips) == 3 and not dataset.errors)
    check("audio normalised to 16 kHz mono", dataset.clips[0].audio.wav_bytes[:4] == b"RIFF")

    narrowband = build_dataset(uploads, GROUND_TRUTH, sample_rate=8_000)
    check("8 kHz normalisation", narrowband.clips[0].audio.sample_rate == 8_000)
    check(
        "8 kHz WAV header agrees with the payload",
        wav_sample_rate(narrowband.clips[0].audio.wav_bytes) == 8_000,
    )
    check(
        "duration preserved across resampling",
        abs(narrowband.clips[0].duration_seconds - dataset.clips[0].duration_seconds) < 0.01,
    )
    upsample = build_dataset(uploads, GROUND_TRUTH, sample_rate=48_000)
    check(
        "upsampling is warned about",
        any("Upsampled" in message for message in upsample.warnings),
    )
    check(
        "telephony model chosen at 8 kHz",
        DeepgramProvider("k")._model_for(8_000) == "nova-2-phonecall"
        and GoogleProvider("k")._model_for(8_000) == "phone_call",
    )
    check(
        "wideband model chosen at 16 kHz",
        DeepgramProvider("k")._model_for(16_000) == "nova-3"
        and GoogleProvider("k")._model_for(16_000) == "latest_long",
    )

    # Microphone recordings arrive as (id.wav, bytes) exactly like uploads, so
    # they merge into one dataset and nothing downstream distinguishes them.
    mixed_truth = {**GROUND_TRUTH, "rec1": "This one was recorded in the browser"}
    mixed = build_dataset(
        uploads + [("rec1.wav", make_wav(0.8))], mixed_truth
    )
    check("recordings merge with uploads", len(mixed.clips) == 4 and not mixed.errors)
    check(
        "recorded clip keeps its id",
        any(clip.clip_id == "rec1" for clip in mixed.clips),
    )
    collision = build_dataset(
        uploads + [("clip1.wav", make_wav(0.8))], GROUND_TRUTH
    )
    check(
        "id collision between sources is caught",
        any("duplicate clip id" in message for message in collision.errors),
    )

    unmatched = build_dataset([("orphan.wav", make_wav(0.5))], GROUND_TRUTH)
    check(
        "unmatched audio reported specifically",
        any("orphan.wav has no matching row" in message for message in unmatched.errors),
    )

    print("runner (mock provider, offline)")
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResultStore(Path(tmpdir) / "results.db")
        config = RunConfig(
            run_id="smoke",
            language="en-IN",
            provider_keys=["mock"],
            api_keys={"mock": ""},
            enabled_metrics=["wer", "cer"],
            per_provider_concurrency=3,
        )
        runner = Runner(config=config, dataset=dataset, store=store, judge=None)
        runner.start()
        runner._thread.join(timeout=60)  # noqa: SLF001 - test-only join

        progress, results = runner.snapshot()
        check("run finished", progress.done and not progress.fatal_error)
        check("one result per (clip, provider)", len(results) == 3)
        check("no failures", progress.failed == 0)
        check("predictions produced", all(result.ok for result in results))

        print("resumability")
        rerun = Runner(config=config, dataset=dataset, store=store, judge=None)
        rerun.start()
        rerun._thread.join(timeout=60)  # noqa: SLF001
        second_progress, _ = rerun.snapshot()
        check("completed pairs reused, not recomputed", second_progress.skipped == 3)
        check("no new work done", second_progress.completed == 0)

        print("aggregation + export")
        summaries = report.summarize(results, enabled_metrics=["wer", "cer"])
        check("summary per provider", set(summaries) == {"mock"})
        check("pooled WER present", summaries["mock"].metrics.get("wer") is not None)

        per_clip = report.per_clip_rows(results, enabled_metrics=["wer", "cer"])
        summary = report.summary_rows(summaries, enabled_metrics=["wer", "cer"])
        check("per-clip CSV non-empty", len(export.per_clip_csv(per_clip)) > 0)
        check("summary CSV non-empty", len(export.summary_csv(summary)) > 0)
        check("workbook builds", len(export.workbook(per_clip, summary, metadata={"run": "smoke"})) > 0)

        store.close()  # Windows will not remove the temp dir while the db is open

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
