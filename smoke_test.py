"""End-to-end offline check: synthesize audio, run the mock provider through the
runner, score the deterministic metrics, aggregate, and export.

    py smoke_test.py

No API keys and no network access are required — the LLM metrics are skipped.
"""

from __future__ import annotations

import ast
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
    from stt_eval.providers import (
        PROVIDER_CLASSES,
        providers_for_language,
        supports_language,
    )

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

    free_judge = create_judge(
        backend="openrouter", api_key="k", model="nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    check("free models get overload fallbacks", bool(free_judge.fallback_models))
    check(
        "a model never falls back to itself",
        free_judge.model not in free_judge.fallback_models,
    )
    paid_judge = create_judge(backend="openrouter", api_key="k", model="openai/gpt-5")
    check(
        "paid models do not silently switch",
        paid_judge.fallback_models == [],  # substituting would spend unasked-for money
    )
    free_judge._tried.add(free_judge.model)
    first = free_judge._next_fallback()
    check("fallback picks an untried model", first and first != free_judge.model)
    free_judge._tried.update(free_judge.fallback_models)
    check("fallbacks run out rather than looping", free_judge._next_fallback() == "")
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

    print("threading discipline")
    # st.session_state is bound to Streamlit's script-run thread. Reading it
    # from a pool worker raises AttributeError at runtime, which no import or
    # render check catches — so assert statically that worker bodies only use
    # values hoisted on the main thread.
    app_source = Path("app.py").read_text(encoding="utf-8")
    app_tree = ast.parse(app_source)
    offenders: list[str] = []
    for node in ast.walk(app_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in {"one", "_worker", "_task"}:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Attribute)
                and inner.value.attr == "session_state"
            ):
                offenders.append(f"{node.name}() line {inner.lineno}")
    check("no session_state access inside thread workers", not offenders)
    if offenders:
        print("      offenders:", offenders)

    print("streaming latency maths")
    from stt_eval import streaming

    # A result covering audio 0.0-1.0s that arrives at 1.2s was 0.2s behind the
    # speech it carried. That subtraction is the whole measurement.
    events = [
        streaming.StreamEvent(arrival_offset=1.2, audio_start=0.0, audio_end=1.0, text="hello", is_final=False),
        streaming.StreamEvent(arrival_offset=2.1, audio_start=1.0, audio_end=2.0, text="there", is_final=False),
        streaming.StreamEvent(arrival_offset=2.6, audio_start=0.0, audio_end=2.0, text="hello there", is_final=True),
    ]
    measured = streaming.StreamingMetrics(
        provider="deepgram", audio_seconds=2.0, wall_seconds=2.7, events=events, pacing="realtime"
    )
    check("partial latency subtracts the audio it covers", abs(measured.partial_emission_latency - 0.15) < 1e-9)
    check("finals latency measured from phrase end", abs(measured.finals_latency - 0.6) < 1e-9)
    check("time to first partial is an absolute offset", abs(measured.time_to_first_partial - 1.2) < 1e-9)
    check("partials and finals counted separately", len(measured.partials) == 2 and len(measured.finals) == 1)
    check(
        "RTF withheld under real-time pacing",
        measured.rtf is None,  # pacing dominates the clock; the number would be meaningless
    )
    fast = streaming.StreamingMetrics(
        provider="deepgram", audio_seconds=4.0, wall_seconds=1.0, events=events, pacing="fast"
    )
    check("RTF reported when sending as fast as possible", abs(fast.rtf - 0.25) < 1e-9)
    check(
        "a result cannot have negative latency",
        streaming.StreamEvent(arrival_offset=0.5, audio_start=0.0, audio_end=1.0, text="x", is_final=True).latency == 0.0,
    )
    check("batch RTF divides latency by duration", abs(streaming.real_time_factor(3.0, 6.0) - 0.5) < 1e-9)
    check("batch RTF is undefined without both figures", streaming.real_time_factor(None, 6.0) is None)
    try:
        streaming.measure(provider="sarvam", api_key="k", wav_bytes=b"", language="en-IN")
        check("non-streaming providers are rejected clearly", False)
    except streaming.StreamingUnsupported:
        check("non-streaming providers are rejected clearly", True)

    print("use-case layer")
    from stt_eval import usecase
    from stt_eval.metrics import use_case_metrics

    from stt_eval.store import StoredResult as SRes

    profile = usecase.Requirements(
        use_case="Logistics",
        summary="Collects delivery details.",
        fields=[
            usecase.CriticalField("Order number", "identifier"),
            usecase.CriticalField("Quantity", "quantity"),
            usecase.CriticalField("Delivery date", "date"),
            usecase.CriticalField("Location", "location"),
        ],
    )
    plan = usecase.select(profile)
    selected = {item.metric for item in plan.selections}
    check("WER and CER always selected", {"wer", "cer"} <= selected)
    check("critical fields metric selected", "critical_fields" in selected)
    check("identifier accuracy driven by the order number", "identifier_accuracy" in selected)
    check("number accuracy driven by the quantity", "number_accuracy" in selected)
    check("date accuracy driven by the date field", "date_accuracy" in selected)
    check("location accuracy driven by the location field", "location_accuracy" in selected)
    check("no name metric without a name field", "name_accuracy" not in selected)

    tiers = {item.metric: item.tier for item in plan.selections}
    check("field accuracy is PRIMARY", tiers["date_accuracy"] == "PRIMARY")
    check("WER is BASELINE", tiers["wer"] == "BASELINE")
    check("semantic match is SECONDARY", tiers["semantic_match"] == "SECONDARY")
    check("every selection carries a reason", all(item.reason for item in plan.selections))
    check(
        "field-driven reasons name the fields that drove them",
        "Order number" in {item.metric: item.reason for item in plan.selections}["identifier_accuracy"],
    )
    check("weights sum to one", abs(sum(item.weight for item in plan.selections) - 1.0) < 1e-9)
    check(
        "primary tier carries the largest share",
        sum(item.weight for item in plan.by_tier("PRIMARY"))
        > sum(item.weight for item in plan.by_tier("BASELINE")),
    )
    check(
        "derived metrics are not sent to the runner",
        "date_accuracy" not in plan.runnable_metrics
        and "critical_fields" in plan.runnable_metrics,
    )

    usecase.set_weights(plan, {"wer": 0.9})
    check("user weights renormalise to one", abs(sum(i.weight for i in plan.selections) - 1.0) < 1e-9)
    usecase.apply_default_weights(plan)

    # A prompt with no data collection selects baselines and meaning only.
    bare = usecase.Requirements(use_case="Chit-chat", summary="Small talk.", fields=[])
    bare_plan = usecase.select(bare)
    check(
        "a prompt collecting nothing has no PRIMARY field metrics",
        not [i for i in bare_plan.by_tier("PRIMARY") if i.metric.endswith("_accuracy")],
    )

    per_field = {
        "Order number": {"preserved": True, "type": "identifier"},
        "Quantity": {"preserved": True, "type": "quantity"},
        "Delivery date": {"preserved": False, "type": "date"},
        "Location": {"preserved": True, "type": "location"},
    }
    categories = use_case_metrics.category_scores(per_field)
    check("identifier accuracy rolled up", categories["identifier_accuracy"] == 1.0)
    check("date accuracy reflects the lost date", categories["date_accuracy"] == 0.0)
    check("location accuracy computed", categories["location_accuracy"] == 1.0)
    check("absent categories are omitted", "name_accuracy" not in categories)

    good = SRes(
        run_id="x", clip_id="c", provider="alpha", status="ok",
        metrics={
            "wer": {"value": 0.30},
            "cer": {"value": 0.20},
            "semantic_match": {"value": 1.0},
            "intent_entity": {"value": 1.0},
            "critical_fields": {"value": 1.0, "extra": {"fields": {
                name: {**verdict, "preserved": True} for name, verdict in per_field.items()
            }}},
        },
    )
    sloppy = SRes(
        run_id="x", clip_id="c", provider="beta", status="ok",
        metrics={
            "wer": {"value": 0.10},  # better WER...
            "cer": {"value": 0.08},
            "semantic_match": {"value": 0.0},
            "intent_entity": {"value": 0.5},
            "critical_fields": {"value": 0.5, "extra": {"fields": per_field}},  # ...lost a field
        },
    )
    good_score = usecase.score(plan, good)
    sloppy_score = usecase.score(plan, sloppy)
    check("a run preserving every field scores higher", good_score.score > sloppy_score.score)
    check("score is complete when every metric ran", good_score.complete)
    check("score exposes its parts", "wer" in good_score.parts)

    verdict = usecase.explain_winner(plan, [good, sloppy], use_case="logistics")
    check("the field-preserving provider wins despite worse WER", verdict.winner == "alpha")
    check(
        "the explanation says WER was higher",
        "higher" in verdict.explanation and "word error rate" in verdict.explanation,
    )
    check("the explanation names the lost field", "Delivery date" in verdict.explanation)

    providers, rows = usecase.comparison_table(plan, [good, sloppy])
    check("comparison table covers both providers", providers == ["alpha", "beta"])
    check("comparison table leads with the use-case score", rows[0]["Metric"] == "Use-case score")
    check("comparison table includes latency and cost", {r["Metric"] for r in rows} >= {"p95 latency", "Estimated cost"})

    missing_fields = SRes(
        run_id="x", clip_id="c", provider="gamma", status="ok",
        metrics={"wer": {"value": 0.1}, "cer": {"value": 0.1}},
    )
    partial = usecase.score(plan, missing_fields)
    check("weights renormalise over what ran", partial.score > 0)
    check("missing metrics are named", not partial.complete and "semantic_match" in partial.missing)

    check("scenario types cover the PRD list", set(usecase.SCENARIO_TYPES) >= {
        "normal", "number", "datetime", "entity", "semantic", "critical"
    })

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

    print("per-clip language")
    tagged = build_dataset(
        uploads + [("rec1.wav", make_wav(0.8))],
        {**GROUND_TRUTH, "rec1": "recorded in hindi"},
        languages={"rec1": "hi-IN"},
    )
    rec = tagged.by_id("rec1")
    check("recorded clip carries its language", rec.language == "hi-IN")
    check("tagged clip overrides the run language", rec.language_for("en-IN") == "hi-IN")
    check(
        "untagged clips fall back to the run language",
        tagged.by_id("clip1").language_for("en-IN") == "en-IN",
    )
    check(
        "languages in use are reported",
        tagged.languages_in_use("en-IN") == {"en-IN", "hi-IN"},
    )
    check("one tagged language is not multilingual", not tagged.is_multilingual)
    two_languages = build_dataset(
        uploads + [("rec1.wav", make_wav(0.8)), ("rec2.wav", make_wav(0.8))],
        {**GROUND_TRUTH, "rec1": "hindi clip", "rec2": "tamil clip"},
        languages={"rec1": "hi-IN", "rec2": "ta-IN"},
    )
    check("two tagged languages is multilingual", two_languages.is_multilingual)
    check(
        "provider filter respects every language in use",
        # Sarvam covers both en-IN and hi-IN; Google covers both; a Japanese
        # clip in the mix would exclude Sarvam.
        all(supports_language("sarvam", code) for code in tagged.languages_in_use("en-IN"))
        and not all(
            supports_language("sarvam", code)
            for code in build_dataset(
                uploads + [("rec1.wav", make_wav(0.8))],
                {**GROUND_TRUTH, "rec1": "japanese clip"},
                languages={"rec1": "ja-JP"},
            ).languages_in_use("en-IN")
        ),
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

        print("sampling")
        from stt_eval import sampling

        pool = [((f"clip{index}", "mock"), index / 40.0) for index in range(40)]
        full = sampling.plan(pool, rate=1.0)
        check("rate 1.0 judges everything", full.total_selected == 40)

        tenth = sampling.plan(pool, rate=0.10, seed=7)
        check("sampling reduces the judged set", 4 <= tenth.total_selected <= 12)
        check(
            "every populated band keeps representation",
            all(
                sampled >= 1
                for _, (sampled, total) in tenth.per_band.items()
                if total
            ),
        )
        check(
            "sampling is deterministic for a given seed",
            sampling.plan(pool, rate=0.10, seed=7).selected == tenth.selected,
        )
        check("exact matches land in their own band", sampling.band_for(0.0) == "exact match")
        check("bad clips land in the poor band", sampling.band_for(0.9) == "poor")

        print("signals")
        from stt_eval import signals as sig
        from stt_eval.store import StoredResult as SR

        low_wer_broken = SR(
            run_id="x", clip_id="c1", provider="mock", status="ok",
            metrics={"wer": {"value": 0.05}, "semantic_match": {"value": 0.0}},
        )
        high_wer_fine = SR(
            run_id="x", clip_id="c2", provider="mock", status="ok",
            metrics={"wer": {"value": 0.60}, "semantic_match": {"value": 1.0}},
        )
        broken, wording = sig.divergences(
            [low_wer_broken, high_wer_fine], thresholds=sig.Thresholds()
        )
        check("low WER with broken meaning is flagged", [r.clip_id for r in broken] == ["c1"])
        check("high WER with intact meaning is flagged", [r.clip_id for r in wording] == ["c2"])

        found = sig.evaluate(
            [low_wer_broken, high_wer_fine],
            report.summarize([low_wer_broken, high_wer_fine], enabled_metrics=["wer"]),
            thresholds=sig.Thresholds(),
        )
        check("meaning-critical failure is critical severity", found[0].severity == "critical")
        check(
            "a clean run raises nothing",
            sig.evaluate(
                [SR(run_id="x", clip_id="c", provider="mock", status="ok",
                    metrics={"wer": {"value": 0.01}, "semantic_match": {"value": 1.0}})],
                report.summarize(
                    [SR(run_id="x", clip_id="c", provider="mock", status="ok",
                        metrics={"wer": {"value": 0.01}})],
                    enabled_metrics=["wer"],
                ),
                thresholds=sig.Thresholds(),
            ) == [],
        )

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
