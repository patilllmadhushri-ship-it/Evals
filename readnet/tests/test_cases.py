"""ReadNet regression suite.

    py -m readnet.tests.test_cases

Two parts:

* ``field_cases.csv`` — one row per disputed or tricky reading, with the
  mistake count and correct-word count a human assessor gave it. Every time the
  app and a field assessor disagree and the assessor is right, the case is
  added here, so the same disagreement can never come back. Linguists can add
  rows without writing code.
* The ``test_*`` functions below — the machinery (alignment, ASER placement,
  forced alignment, GOP, agreement metrics), checked on constructed inputs.

Works under pytest too; no dependency on it.
"""

from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path

import numpy as np

from readnet import benchmark, confusion, languages, metrics
from readnet.acoustic import (
    AcousticEvidence,
    WordEvidence,
    closed_set_decision,
    ctc_forced_align,
    evidence_for_words,
    gop,
)
from readnet.aser import Level, Rules, TaskOutcome, next_task, place
from readnet.pipeline import assess_item, assess_task
from readnet.score import SHARED_RULES

CASES = Path(__file__).with_name("field_cases.csv")


def load_cases() -> list[dict]:
    with CASES.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_case(case: dict) -> str | None:
    item = assess_item(case["canonical"], case["transcript"], language=case["language"], level=case["level"])
    got = (item.result.mistakes, item.result.correct)
    want = (int(case["expected_mistakes"]), int(case["expected_correct"] or got[1]))
    if got != want:
        ops = "; ".join(
            f"{o.kind}({o.ref}->{o.hyp}{', ' + o.category if o.category else ''})"
            for o in item.result.ops
            if o.kind != "match"
        )
        return f"{case['id']}: got mistakes/correct {got}, want {want} [{ops}] — {case['note']}"
    return None


def test_field_cases():
    failures = [f for f in map(run_case, load_cases()) if f]
    assert not failures, "\n".join(failures)


def test_every_case_names_a_real_rule_and_every_rule_has_a_case():
    cases = load_cases()
    for code in languages.supported():
        profile = languages.get(code)
        known = profile.rule_ids() | SHARED_RULES
        mine = [c for c in cases if c["language"] == code]
        unknown = {c["id"]: c["rule"] for c in mine if c["rule"] not in known}
        assert not unknown, f"{code}: cases cite rules that do not exist: {unknown}"
        untested = profile.rule_ids() - {c["rule"] for c in mine}
        assert not untested, f"{code}: rules with no field case: {sorted(untested)}"


def test_forgiven_differences_cite_their_rule():
    item = assess_item("मेरा गाँव बड़ा है", "मेरा गाव बड़ा है")
    assert item.result.rules_applied == {"HI-20": 1}


# -- normalisation ----------------------------------------------------------------


def test_normalisation_is_idempotent():
    for code in languages.supported():
        profile = languages.get(code)
        for case in load_cases():
            for text in (case["canonical"], case["transcript"]):
                once, _ = profile.normalize(text)
                twice, _ = profile.normalize(once)
                assert once == twice, (code, text, once, twice)


def test_nfd_input_equals_nfc():
    profile = languages.get("hi")
    word = "क़िला"
    assert profile.normalize(unicodedata.normalize("NFD", word))[0] == profile.normalize(word)[0]


def test_zero_width_joiners_are_removed():
    profile = languages.get("hi")
    assert profile.normalize("क्\u200dष")[0] == profile.normalize("क्ष")[0]
    assert profile.normalize("घर\u200c")[0] == "घर"


def test_trace_records_which_rule_changed_what():
    _, trace = languages.get("hi").normalize("ghar जाना।")
    assert [rule for rule, _, _ in trace.steps] == ["HI-02", "HI-04"], trace.steps
    assert trace.romanised_words == ["घर"]


def test_locale_aliases():
    assert languages.get("hi-IN").code == "hi"
    assert languages.get("mr-IN").code == "mr"


# -- scoring details ----------------------------------------------------------------


def test_mistake_profile_names_the_sound():
    item = assess_item("राजा ने रानी को देखा", "राजा ने राणी को देका")
    profile = item.result.mistake_profile
    assert profile["substitution:phonetic"] == 2
    assert profile["sound:aspiration"] == 1 and profile["sound:retroflex_dental"] == 1


def test_count_insertions_option():
    strict = Rules(count_insertions=True)
    item = assess_item("मेरा घर बड़ा है", "मेरा घर बहुत बड़ा है", rules=strict)
    assert item.result.mistakes == 1


# -- ASER placement ---------------------------------------------------------------


def outcome(level: Level, passed: bool) -> TaskOutcome:
    return TaskOutcome(level, passed, 0, 0, 0, reasons=["test"])


def test_placement_paths():
    L = Level
    table = [
        ({L.PARAGRAPH: True, L.STORY: True}, L.STORY),
        ({L.PARAGRAPH: True, L.STORY: False}, L.PARAGRAPH),
        ({L.PARAGRAPH: False, L.WORD: True}, L.WORD),
        ({L.PARAGRAPH: False, L.WORD: False, L.LETTER: True}, L.LETTER),
        ({L.PARAGRAPH: False, L.WORD: False, L.LETTER: False}, L.BEGINNER),
    ]
    for given, expected in table:
        outcomes = {lvl: outcome(lvl, ok) for lvl, ok in given.items()}
        assert next_task(outcomes) is None, given
        assert place(outcomes).level == expected, (given, place(outcomes).level)


def test_next_task_follows_aser_order():
    L = Level
    seen: dict = {}
    assert next_task(seen) == L.PARAGRAPH
    seen[L.PARAGRAPH] = outcome(L.PARAGRAPH, False)
    assert next_task(seen) == L.WORD
    seen[L.WORD] = outcome(L.WORD, False)
    assert next_task(seen) == L.LETTER


def test_placement_refuses_incomplete_sessions():
    try:
        place({Level.PARAGRAPH: outcome(Level.PARAGRAPH, True)})
    except ValueError:
        return
    raise AssertionError("placed a child without the story task")


def test_end_to_end_word_child():
    paragraph = "मेरा नाम सीता है। मैं स्कूल जाती हूँ। मेरे घर में एक गाय है।"
    heard = "मेरा नाम सिता है मैं सकूल जाति हूं मेरे गर में गाय है"  # skipped एक
    para, _ = assess_task("paragraph", [(paragraph, heard)])
    assert not para.passed and para.mistakes == 5, para
    words, _ = assess_task(
        "word",
        [("घर", "घर"), ("पानी", "पानी"), ("आम", "आम"), ("नाक", "नाग"), ("कमल", "कमल")],
    )
    assert words.passed and words.correct == 4
    assert place({Level.PARAGRAPH: para, Level.WORD: words}).level == Level.WORD


def test_fluency_gate_only_when_configured():
    evidence = AcousticEvidence(
        [WordEvidence("घर", 0.0, 0.5, -0.1, 0.0), WordEvidence("है", 4.0, 4.5, -0.1, 0.0)]
    )
    lenient, _ = assess_task("paragraph", [("घर है", "घर है")], acoustic=[evidence])
    assert lenient.passed and lenient.longest_pause == 3.5
    strict, _ = assess_task("paragraph", [("घर है", "घर है")], acoustic=[evidence], rules=Rules(max_pause_seconds=2.0))
    assert not strict.passed


# -- forced alignment and GOP ---------------------------------------------------------


def _emissions(frames: list[int], vocab_size: int, confidence: float = 0.9) -> np.ndarray:
    """Log-posteriors where frame t strongly favours token frames[t]."""
    rest = (1 - confidence) / (vocab_size - 1)
    probs = np.full((len(frames), vocab_size), rest)
    for t, token in enumerate(frames):
        probs[t, token] = confidence
    return np.log(probs)


def test_ctc_forced_alignment_finds_the_boundaries():
    # blank=0, tokens 1 2 3; audio: blank, 1 1, blank, 2 2 2, 3, blank
    log_probs = _emissions([0, 1, 1, 0, 2, 2, 2, 3, 0], 4)
    spans = ctc_forced_align(log_probs, [1, 2, 3])
    assert [(s.start, s.end) for s in spans] == [(1, 3), (4, 7), (7, 8)]


def test_ctc_aligns_long_sequences():
    # 100 tokens = 201 CTC states: past int8, which once overflowed the backtrack.
    targets = [1 + (i % 5) for i in range(100)]
    frames = [f for t in targets for f in (t, t, 0)]
    spans = ctc_forced_align(_emissions(frames, 6), targets)
    assert [(s.start, s.end) for s in spans] == [(3 * i, 3 * i + 2) for i in range(100)]


def test_ctc_handles_repeated_tokens():
    log_probs = _emissions([1, 1, 0, 1, 1], 3)
    spans = ctc_forced_align(log_probs, [1, 1])
    assert [(s.start, s.end) for s in spans] == [(0, 2), (3, 5)]


def test_ctc_rejects_impossible_alignments():
    try:
        ctc_forced_align(_emissions([1], 3), [1, 2])
    except ValueError:
        return
    raise AssertionError("aligned two tokens into one frame")


def test_gop_drops_when_the_audio_says_something_else():
    # Expected token 2, but frames 3-4 sound like token 3.
    good = _emissions([0, 1, 1, 0, 2, 2, 0], 4)
    bad = _emissions([0, 1, 1, 0, 3, 3, 0], 4)
    targets = [1, 2]
    good_gop = gop(good, ctc_forced_align(good, targets), targets)
    bad_gop = gop(bad, ctc_forced_align(bad, targets), targets)
    assert good_gop[1].llr > 0 > bad_gop[1].llr
    assert good_gop[1].mean_log_posterior > bad_gop[1].mean_log_posterior


def test_acoustic_doubt_is_flagged_not_counted_by_default():
    vocab = {"<pad>": 0, "|": 1, "घ": 2, "ग": 3, "र": 4}
    # Canonical घर; the child said गर; the ASR "auto-corrected" to घर.
    log_probs = _emissions([0, 3, 3, 0, 4, 4, 0], len(vocab))
    evidence = evidence_for_words(log_probs, ["घर"], vocab, frame_seconds=0.02)
    item = assess_item("घर", "घर", level="word", acoustic=evidence, gop_threshold=0.0)
    assert item.acoustic_doubts == ["घर"] and item.result.mistakes == 0
    counted = assess_item("घर", "घर", level="word", acoustic=evidence, gop_threshold=0.0, count_acoustic_doubts=True)
    assert counted.result.mistakes == 1


def test_closed_set_letter_decision():
    vocab = {"<pad>": 0, "घ": 1, "ग": 2, "ध": 3}
    candidates = confusion.hand_made_confusions("घ", languages.get("hi").tables)
    assert {"ग", "ध"} <= set(candidates)
    said_gha = _emissions([0, 1, 1, 1, 0], len(vocab))
    said_ga = _emissions([0, 2, 2, 2, 0], len(vocab))
    ok = closed_set_decision(said_gha, "घ", candidates, vocab)
    wrong = closed_set_decision(said_ga, "घ", candidates, vocab)
    assert ok.accepted and ok.margin > 0
    assert not wrong.accepted and wrong.best == "ग" and wrong.margin < 0


# -- benchmark and confusions ------------------------------------------------------------


def test_neutral_benchmark_does_not_borrow_forgiveness():
    rows = [
        {"level": "word", "reference": "गाँव", "hypothesis": "गाव"},  # forgiven for a child; an ASR error here
        {"level": "word", "reference": "ज़रा", "hypothesis": "जरा"},  # not in the speech: neutral too
        {"level": "paragraph", "reference": "मेरा घर बड़ा है", "hypothesis": "मेरा घर बड़ा है"},
    ]
    by_level = {r.level: r for r in benchmark.benchmark(rows, "hi")}
    assert by_level["Word"].wer == 0.5 and by_level["Word"].forgiving_wer == 0.0  # the flattery
    assert by_level["Paragraph"].wer == 0.0
    assert "Letter" in benchmark.format_table(list(by_level.values()))  # flags the missing level


def test_confusions_are_estimated_from_pairs():
    pairs = [("घर", "गर"), ("घड़ा", "गड़ा"), ("घास", "गास"), ("धन", "घन"), ("घर", "घर")]
    matrix = confusion.estimate(pairs, "hi")
    assert matrix.counts[("घ", "ग")] == 3
    assert matrix.confusions_of("घ") == ["ग"]
    assert abs(matrix.rate("घ", "ग") - 3 / 4) < 1e-9  # घ was expected in 4 words
    audit = confusion.audit(matrix, languages.get("hi").tables, min_count=1)
    assert "ग/घ" not in audit["frequent_but_unlisted"]


# -- agreement metrics -----------------------------------------------------------------


def test_metrics():
    perfect = metrics.level_agreement([("story", "story"), ("word", "word"), ("letter", "letter")])
    assert perfect.accuracy == 1.0 and abs(perfect.weighted_kappa - 1.0) < 1e-9
    mixed = metrics.level_agreement([("story", "paragraph"), ("word", "word"), ("letter", "word")])
    assert abs(mixed.false_fail_rate - 1 / 3) < 1e-9 and abs(mixed.false_pass_rate - 1 / 3) < 1e-9
    near = metrics.level_agreement([("story", "paragraph"), ("word", "word"), ("beginner", "beginner")])
    far = metrics.level_agreement([("story", "beginner"), ("word", "word"), ("beginner", "beginner")])
    assert near.weighted_kappa > far.weighted_kappa
    pr = metrics.mistake_precision_recall([({1, 2}, {2, 3}), (set(), set())])
    assert pr["precision"] == 0.5 and pr["recall"] == 0.5


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL  {name}\n      {error}")
    cases = len(load_cases())
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed ({cases} field cases)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
