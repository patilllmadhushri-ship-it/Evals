"""Language-agnostic alignment and mistake counting.

Input is two already-normalised token lists: what was on the screen and what
the ASR heard. Output is one operation per word — match, substitution,
deletion, insertion — each classified and marked as counting, or not, as an
ASER mistake.

Two things make this more than plain WER:

* **Weighted alignment.** A substitution between similar words (घर/गर) is
  cheaper than one between unrelated words, so the aligner pairs the words a
  human would pair, and the substitution can then be classified.
* **ASER's leniency.** A child who repeats a word, starts a word and restarts
  it, or corrects themselves is not making a mistake under ASER, and neither is
  "umm". Those insertions are labelled and never counted. Additions of other
  words are reported but, by default, not counted either: ASER counts words read
  wrongly or skipped, and an extra word is far more often an ASR hallucination
  than a child's error — the false-fail rate is the metric that matters most.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .languages import LanguageProfile
from .languages import devanagari as dv

# -- generic weighted Levenshtein ----------------------------------------------


def align(
    ref: Sequence[str],
    hyp: Sequence[str],
    sub_cost: Callable[[str, str], float] | None = None,
) -> list[tuple[str, int | None, int | None]]:
    """Return ops as (kind, ref_index, hyp_index). Insert/delete cost 1."""
    sub_cost = sub_cost or (lambda a, b: 0.0 if a == b else 1.0)
    n, m = len(ref), len(hyp)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]),
                cost[i - 1][j] + 1.0,
                cost[i][j - 1] + 1.0,
            )
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diagonal = cost[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1])
            if abs(cost[i][j] - diagonal) < 1e-9:
                ops.append(("match" if ref[i - 1] == hyp[j - 1] else "substitution", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(cost[i][j] - (cost[i - 1][j] + 1.0)) < 1e-9:
            ops.append(("deletion", i - 1, None))
            i -= 1
        else:
            ops.append(("insertion", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def char_distance_ratio(a: str, b: str) -> float:
    if a == b:
        return 0.0
    edits = sum(1 for kind, _, _ in align(a, b) if kind != "match")
    return edits / max(len(a), len(b), 1)


def word_sub_cost(a: str, b: str) -> float:
    """0 for equal; 0.5–1.5 by spelling distance — always below delete+insert (2)."""
    return 0.0 if a == b else 0.5 + char_distance_ratio(a, b)


# -- classifying one wrong word -------------------------------------------------


def _char_label(r: str, h: str) -> str:
    pair = frozenset((r, h))
    if not r or not h:
        present = r or h
        if present in (dv.ANUSVARA, dv.CHANDRABINDU):
            return "nasalisation"
        if present == "ा":
            return "vowel_length"
        if present in dv.MATRAS:
            return "matra"
        if present == dv.VIRAMA:
            return "conjunct"
        return "other"
    if pair <= {dv.ANUSVARA, dv.CHANDRABINDU}:
        return "nasalisation"
    for label, table in (
        ("aspiration", dv.ASPIRATION),
        ("voicing", dv.VOICING),
        ("retroflex_dental", dv.RETROFLEX_DENTAL),
        ("sibilant", dv.SIBILANT),
        ("vowel_length", dv.VOWEL_LENGTH),
        ("visual", dv.VISUAL),
    ):
        if pair in table:
            return label
    if r in dv.MATRAS and h in dv.MATRAS:
        return "matra"
    return "other"


PHONETIC = {"aspiration", "voicing", "retroflex_dental", "sibilant", "vowel_length", "nasalisation"}
VISUAL_LIKE = {"visual", "matra", "conjunct"}
#: What Latin-script spelling cannot encode, so cannot be judged from it.
ROMANISATION_BLIND = {"vowel_length", "retroflex_dental", "nasalisation"}


@dataclass(frozen=True)
class Classification:
    category: str  # phonetic | visual | mixed | partial | different_word
    subtypes: tuple[str, ...]
    char_diffs: tuple[tuple[str, str], ...]


def classify_substitution(ref: str, hyp: str) -> Classification:
    diffs = tuple(
        (ref[i] if i is not None else "", hyp[j] if j is not None else "")
        for kind, i, j in align(ref, hyp)
        if kind != "match"
    )
    if hyp and ref.startswith(hyp) and len(hyp) < len(ref):
        return Classification("partial", ("incomplete",), diffs)
    labels = tuple(_char_label(r, h) for r, h in diffs)
    if char_distance_ratio(ref, hyp) > 0.6 or "other" in labels:
        return Classification("different_word", labels, diffs)
    kinds = {("phonetic" if l in PHONETIC else "visual") for l in labels}
    return Classification(kinds.pop() if len(kinds) == 1 else "mixed", labels, diffs)


# -- scoring --------------------------------------------------------------------


@dataclass
class WordOp:
    kind: str  # match | substitution | deletion | insertion
    ref: str | None
    hyp: str | None
    ref_index: int | None
    #: For a substitution: phonetic/visual/mixed/partial/different_word.
    #: For an insertion: repetition/self_correction/filler/extra_word.
    category: str | None = None
    subtypes: tuple[str, ...] = ()
    counts_as_mistake: bool = False
    note: str = ""


@dataclass
class ScoreResult:
    ops: list[WordOp]
    ref_len: int

    @property
    def mistakes(self) -> int:
        return sum(op.counts_as_mistake for op in self.ops)

    @property
    def correct(self) -> int:
        """Reference words read acceptably (matched, forgiven or self-corrected)."""
        return sum(
            1 for op in self.ops if op.ref_index is not None and not op.counts_as_mistake
        )

    @property
    def mistake_ref_indices(self) -> set[int]:
        return {op.ref_index for op in self.ops if op.counts_as_mistake and op.ref_index is not None}

    @property
    def mistake_profile(self) -> Counter:
        """Kinds of error, not just their number — what the teacher acts on."""
        profile: Counter = Counter()
        for op in self.ops:
            if op.counts_as_mistake:
                profile[op.kind if op.kind != "substitution" else f"substitution:{op.category}"] += 1
                for subtype in op.subtypes:
                    profile[f"sound:{subtype}"] += 1
        return profile

    def as_dict(self) -> dict:
        return {
            "mistakes": self.mistakes,
            "correct": self.correct,
            "ref_len": self.ref_len,
            "mistake_profile": dict(self.mistake_profile),
            "ops": [op.__dict__ for op in self.ops],
        }


def _is_restart_of(token: str, word: str) -> bool:
    """A false start: the word itself, its beginning, or a near-miss of either
    ("ग… घर" — the child began with the wrong sound and fixed it)."""
    if not token:
        return False
    if word.startswith(token) or char_distance_ratio(token, word) <= 0.5:
        return True
    head = word[: len(token)]
    return len(token) < len(word) and _char_label(head[0], token[0]) != "other" and (
        char_distance_ratio(token, head) <= 0.5 or len(token) == 1
    )


def score(
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
    profile: LanguageProfile,
    *,
    letter_task: bool = False,
    count_insertions: bool = False,
    romanised: set[str] | frozenset[str] = frozenset(),
) -> ScoreResult:
    """`romanised` are hyp words the ASR wrote in Latin script: their vowel
    length and dental/retroflex choice were guessed by the transliterator, so
    differences of exactly those kinds are not held against the child."""
    hyp_tokens = list(hyp_tokens)
    if letter_task:
        # Map each spoken-letter spelling back onto its letter before aligning.
        variants = {v: r for r in ref_tokens for v in profile.letter_variants(r)}
        hyp_tokens = [variants.get(h, h) for h in hyp_tokens]

    raw = align(ref_tokens, hyp_tokens, word_sub_cost)
    ops: list[WordOp] = []
    for kind, i, j in raw:
        ref = ref_tokens[i] if i is not None else None
        hyp = hyp_tokens[j] if j is not None else None
        op = WordOp(kind, ref, hyp, i)
        if kind == "substitution":
            c = classify_substitution(ref, hyp)
            op.category, op.subtypes = c.category, c.subtypes
            forgiven = c.char_diffs and all(
                frozenset(d) in profile.forgiven_pairs for d in c.char_diffs
            )
            lossy = hyp in romanised and set(c.subtypes) <= ROMANISATION_BLIND
            op.counts_as_mistake = not (forgiven or lossy)
            if forgiven:
                op.note = "forgiven: spelling-only difference"
            elif lossy:
                op.note = "forgiven: romanised ASR output cannot show this difference"
        elif kind == "deletion":
            op.counts_as_mistake = True
            op.note = "word not read"
        ops.append(op)

    # Label insertions: ASER does not penalise repeats, restarts or fillers.
    for k, op in enumerate(ops):
        if op.kind != "insertion":
            continue
        prev_said = next((o.hyp for o in reversed(ops[:k]) if o.hyp is not None), None)
        nxt = next((o for o in ops[k + 1 :] if o.kind != "insertion"), None)
        if op.hyp in profile.fillers:
            op.category = "filler"
        elif op.hyp == prev_said or (nxt is not None and nxt.kind == "match" and op.hyp == nxt.ref):
            op.category = "repetition"
        elif nxt is not None and nxt.ref and _is_restart_of(op.hyp, nxt.ref):
            op.category = "self_correction"
            op.note = f"restarted before reading {nxt.ref}"
        else:
            op.category = "extra_word"
            op.counts_as_mistake = count_insertions
    return ScoreResult(ops, len(ref_tokens))
