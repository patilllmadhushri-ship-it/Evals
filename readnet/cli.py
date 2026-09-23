"""Command line.

    py -m readnet normalize --lang hi "ज़रा रुको।"
    py -m readnet assess --lang hi --level paragraph --text "..." --transcript "..."
    py -m readnet assess --lang hi --level word --text "घर" --audio child.wav --provider sarvam
    py -m readnet evaluate levels.csv      # columns: human_level, system_level
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import languages, metrics
from .pipeline import assess_task


def _transcribe(audio_path: str, provider: str, language: str) -> str:
    """Run one of stt_eval's providers over a local file, keys from .env."""
    from stt_eval import audio, env, providers

    env.load()
    key = env.provider_key(provider)
    if providers.requires_key(provider) and not key:
        raise SystemExit(f"No key for {provider}: set {env.PROVIDER_ENV_VARS[provider]} in .env")
    locale = {"hi": "hi-IN", "mr": "mr-IN"}[languages.get(language).code]
    clip = audio.normalize(Path(audio_path).read_bytes(), Path(audio_path).name, sample_rate=16000)
    return providers.build(provider, key).transcribe(clip.wav_bytes, locale).text


def cmd_normalize(args) -> int:
    text, trace = languages.get(args.lang).normalize(args.text)
    print(text)
    for step, before, after in trace.steps:
        print(f"  {step}: {before!r} -> {after!r}", file=sys.stderr)
    return 0


def cmd_assess(args) -> int:
    transcript = args.transcript
    if args.audio:
        transcript = _transcribe(args.audio, args.provider, args.lang)
    if transcript is None:
        raise SystemExit("Give --transcript or --audio")
    outcome, items = assess_task(args.level, [(args.text, transcript)], language=args.lang)
    if args.json:
        print(json.dumps({"outcome": outcome.__dict__, "items": [i.as_dict() for i in items]},
                         ensure_ascii=False, default=str, indent=2))
        return 0
    item = items[0]
    print(f"heard:      {transcript}")
    print(f"normalised: {item.transcript_normalized}")
    for op in item.result.ops:
        if op.kind == "match":
            continue
        flag = "MISTAKE" if op.counts_as_mistake else "ok     "
        detail = ", ".join(filter(None, [op.category, "/".join(op.subtypes), op.note]))
        print(f"  {flag} {op.kind:<12} {op.ref or '-'} -> {op.hyp or '-'}  ({detail})")
    print(f"{outcome.level.label}: {'PASS' if outcome.passed else 'FAIL'} — {'; '.join(outcome.reasons)}")
    return 0


def cmd_evaluate(args) -> int:
    with open(args.csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = metrics.level_agreement((r["human_level"], r["system_level"]) for r in rows)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="readnet")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("normalize", help="show what normalisation does to a text")
    p.add_argument("text")
    p.add_argument("--lang", default="hi")
    p.set_defaults(fn=cmd_normalize)

    p = sub.add_parser("assess", help="score one reading against its text")
    p.add_argument("--lang", default="hi")
    p.add_argument("--level", default="paragraph", choices=["letter", "word", "paragraph", "story"])
    p.add_argument("--text", required=True, help="what was on the screen")
    p.add_argument("--transcript", help="what the ASR heard")
    p.add_argument("--audio", help="a recording to transcribe instead")
    p.add_argument("--provider", default="sarvam", help="stt_eval provider key for --audio")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_assess)

    p = sub.add_parser("evaluate", help="agreement with human assessors")
    p.add_argument("csv")
    p.set_defaults(fn=cmd_evaluate)

    args = parser.parse_args(argv)
    return args.fn(args)
