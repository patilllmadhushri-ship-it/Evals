"""Forced alignment and Goodness-of-Pronunciation, over any CTC acoustic model.

The ASR transcript alone cannot catch an engine that "hears" घर because घर was
the likely word, when the child actually said गर. So the audio is checked
against the *expected* text directly:

1. **Forced alignment.** Given a CTC model's frame-level log-probabilities and
   the canonical text's token sequence, Viterbi finds the single best path that
   emits exactly those tokens, giving every token (and so every word) a start
   and end frame — hence timestamps, words per minute and pause lengths.
2. **GOP.** Over each token's frames, the blueprint's score
   ``GOP(p) = 1/T Σ log P(p | O_t)``, plus the log-likelihood-ratio form that
   compares p against the best competing token in each frame. A low score
   means the audio does not sound like the expected token, whatever the
   transcript says.

This module is pure numpy and model-agnostic: it takes a ``(frames, vocab)``
log-probability matrix. `EmissionModel` is the one seam where a trained model
plugs in; `Wav2Vec2Emissions` is an adapter for Hugging Face CTC checkpoints
(for example ``ai4bharat/indicwav2vec-hindi``) and needs ``torch`` and
``transformers`` installed.

GOP thresholds must be calibrated on labelled child audio before they are
allowed to fail anyone. Until then the pipeline reports low-GOP words for
review and does not count them (see pipeline.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class TokenSpan:
    token_index: int  # position in the target sequence
    start: int  # first frame, inclusive
    end: int  # last frame, exclusive


def ctc_forced_align(log_probs: np.ndarray, targets: Sequence[int], blank: int = 0) -> list[TokenSpan]:
    """Viterbi alignment of `targets` to a (T, V) log-probability matrix."""
    return _viterbi(log_probs, targets, blank)[0]


def ctc_path_score(log_probs: np.ndarray, targets: Sequence[int], blank: int = 0) -> float:
    """Log-probability of the best path that emits exactly `targets`."""
    return _viterbi(log_probs, targets, blank)[1]


def _viterbi(log_probs: np.ndarray, targets: Sequence[int], blank: int) -> tuple[list[TokenSpan], float]:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T = log_probs.shape[0]
    L = len(targets)
    if L == 0:
        return [], float(np.asarray(log_probs)[:, blank].sum())
    states = [blank]
    for t in targets:
        states += [t, blank]
    S = len(states)
    repeats = sum(1 for a, b in zip(targets, targets[1:]) if a == b)
    if T < L + repeats:
        raise ValueError(f"{T} frames cannot hold {L} tokens ({repeats} repeated)")

    neg = -np.inf
    score = np.full((T, S), neg)
    back = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    emit = log_probs[:, states]
    score[0, 0] = emit[0, 0]
    score[0, 1] = emit[0, 1]
    skip_ok = np.array(
        [s >= 2 and states[s] != blank and states[s] != states[s - 2] for s in range(S)]
    )
    for t in range(1, T):
        prev = score[t - 1]
        stay = prev
        step = np.concatenate(([neg], prev[:-1]))
        skip = np.where(skip_ok, np.concatenate(([neg, neg], prev[:-2])), neg)
        stacked = np.stack([stay, step, skip])
        choice = np.argmax(stacked, axis=0)
        score[t] = stacked[choice, np.arange(S)] + emit[t]
        back[t] = choice

    s = S - 1 if score[T - 1, S - 1] >= score[T - 1, S - 2] else S - 2
    best = float(score[T - 1, s])
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= back[t, s]

    spans: list[TokenSpan] = []
    for token_index in range(L):
        frames = np.nonzero(path == 2 * token_index + 1)[0]
        spans.append(TokenSpan(token_index, int(frames[0]), int(frames[-1]) + 1))
    return spans, best


@dataclass(frozen=True)
class TokenGop:
    token_index: int
    mean_log_posterior: float  # the blueprint's GOP(p)
    llr: float  # mean of log P(p|O_t) - max_{q != p} log P(q|O_t); <= 0 when p is out-scored


def gop(log_probs: np.ndarray, spans: Sequence[TokenSpan], targets: Sequence[int], blank: int = 0) -> list[TokenGop]:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    out: list[TokenGop] = []
    for span in spans:
        target = targets[span.token_index]
        frames = log_probs[span.start : span.end]
        own = frames[:, target]
        rivals = frames.copy()
        rivals[:, [target, blank]] = -np.inf
        out.append(TokenGop(span.token_index, float(own.mean()), float((own - rivals.max(axis=1)).mean())))
    return out


# -- closed-set decisions (letters) -----------------------------------------------------


@dataclass(frozen=True)
class ClosedSetDecision:
    expected: str
    best: str
    #: Best-path log-probability per candidate; higher is a better fit.
    scores: dict[str, float]
    #: expected's score minus the best rival's; negative means a rival fits better.
    margin: float

    @property
    def accepted(self) -> bool:
        return self.best == self.expected


def closed_set_decision(
    log_probs: np.ndarray,
    expected: str,
    candidates: Sequence[str],
    vocab: dict[str, int],
    blank: int = 0,
) -> ClosedSetDecision:
    """Does this audio match `expected`, or one of its known confusions?

    For a 400 ms clip of one letter, open transcription over every sound in
    the language is the wrong question. Force-align the audio to the expected
    letter and to each likely confusion (``confusion.hand_made_confusions`` or
    a learned ``ConfusionMatrix.confusions_of``), and pick the best fit. This
    is a closed decision over a handful of candidates, the standard approach in
    the reading-assessment literature. It needs only frame log-probabilities,
    so it works with any CTC model that exposes them.
    """
    scores: dict[str, float] = {}
    for candidate in dict.fromkeys([expected, *candidates]):
        ids = [vocab[ch] for ch in candidate if ch in vocab]
        if len(ids) != len(candidate):
            continue  # the model cannot emit this candidate
        try:
            scores[candidate] = ctc_path_score(log_probs, ids, blank)
        except ValueError:
            continue  # too long for the clip
    if expected not in scores:
        raise ValueError(f"The model's vocabulary cannot represent {expected!r}")
    best = max(scores, key=scores.get)
    rivals = [v for k, v in scores.items() if k != expected]
    margin = scores[expected] - max(rivals) if rivals else float("inf")
    return ClosedSetDecision(expected, best, scores, margin)


# -- text <-> model tokens --------------------------------------------------------


@dataclass
class TargetSequence:
    ids: list[int]
    #: For each canonical word, the (start, end) range of its tokens in `ids`.
    word_ranges: list[tuple[int, int]]
    #: Characters the model's vocabulary does not contain (skipped, and logged).
    unknown: list[str] = field(default_factory=list)


def targets_for_words(words: Sequence[str], vocab: dict[str, int], word_delimiter: str | None = "|") -> TargetSequence:
    ids: list[int] = []
    ranges: list[tuple[int, int]] = []
    unknown: list[str] = []
    for w, word in enumerate(words):
        if w and word_delimiter is not None and word_delimiter in vocab:
            ids.append(vocab[word_delimiter])
        start = len(ids)
        for ch in word:
            if ch in vocab:
                ids.append(vocab[ch])
            else:
                unknown.append(ch)
        ranges.append((start, len(ids)))
    return TargetSequence(ids, ranges, unknown)


@dataclass
class WordEvidence:
    word: str
    start_s: float | None
    end_s: float | None
    gop: float | None  # weakest token's mean log posterior
    llr: float | None  # weakest token's LLR


@dataclass
class AcousticEvidence:
    words: list[WordEvidence]
    unknown_chars: list[str] = field(default_factory=list)

    def pauses(self) -> list[float]:
        timed = [w for w in self.words if w.start_s is not None]
        return [b.start_s - a.end_s for a, b in zip(timed, timed[1:])]

    def words_per_minute(self) -> float | None:
        timed = [w for w in self.words if w.start_s is not None]
        if len(timed) < 2:
            return None
        seconds = timed[-1].end_s - timed[0].start_s
        return 60.0 * len(timed) / seconds if seconds > 0 else None


def evidence_for_words(
    log_probs: np.ndarray,
    words: Sequence[str],
    vocab: dict[str, int],
    *,
    frame_seconds: float,
    blank: int = 0,
    word_delimiter: str | None = "|",
) -> AcousticEvidence:
    """Align the canonical words to the audio and score each one."""
    target = targets_for_words(words, vocab, word_delimiter)
    spans = ctc_forced_align(log_probs, target.ids, blank)
    scores = gop(log_probs, spans, target.ids, blank)
    evidence: list[WordEvidence] = []
    for word, (a, b) in zip(words, target.word_ranges):
        if a == b:
            evidence.append(WordEvidence(word, None, None, None, None))
            continue
        weakest = min(scores[a:b], key=lambda g: g.llr)
        evidence.append(
            WordEvidence(
                word,
                spans[a].start * frame_seconds,
                spans[b - 1].end * frame_seconds,
                min(g.mean_log_posterior for g in scores[a:b]),
                weakest.llr,
            )
        )
    return AcousticEvidence(evidence, target.unknown)


# -- the model seam ---------------------------------------------------------------


@dataclass
class Emissions:
    log_probs: np.ndarray  # (frames, vocab), natural-log posteriors
    vocab: dict[str, int]
    frame_seconds: float
    blank: int = 0
    word_delimiter: str | None = "|"


def greedy_decode(emissions: Emissions) -> str:
    """Best-per-frame CTC decoding: the model's own open transcript.

    Lets a local CTC model act as the ASR engine as well as the GOP scorer, so
    one recording gives both a transcript and word timings.
    """
    ids = np.argmax(emissions.log_probs, axis=1)
    by_id = {i: tok for tok, i in emissions.vocab.items()}
    out: list[str] = []
    previous = None
    for i in ids:
        if i != previous and i != emissions.blank:
            token = by_id.get(int(i), "")
            if token == emissions.word_delimiter:
                out.append(" ")
            elif not (token.startswith("<") and token.endswith(">")):
                out.append(token)
        previous = i
    return " ".join("".join(out).split())


class EmissionModel(Protocol):
    def emissions(self, wav_bytes: bytes) -> Emissions: ...


class Wav2Vec2Emissions:
    """Hugging Face wav2vec2-CTC adapter. Requires torch + transformers.

    Not exercised by the offline test suite — it needs a model download.
    """

    def __init__(self, model_id: str = "ai4bharat/indicwav2vec-hindi"):
        import torch  # noqa: F401  (fail fast with a clear ImportError)
        from transformers import AutoModelForCTC, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).eval()

    def emissions(self, wav_bytes: bytes) -> Emissions:
        import io

        import soundfile as sf
        import torch

        samples, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        inputs = self.processor(samples, sampling_rate=rate, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1).numpy()
        tokenizer = self.processor.tokenizer
        seconds = len(samples) / rate
        return Emissions(
            log_probs=log_probs,
            vocab=tokenizer.get_vocab(),
            frame_seconds=seconds / log_probs.shape[0],
            blank=tokenizer.pad_token_id or 0,
            word_delimiter=getattr(tokenizer, "word_delimiter_token", "|"),
        )
