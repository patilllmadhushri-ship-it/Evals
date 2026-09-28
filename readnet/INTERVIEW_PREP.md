# Akshar — the whole story, for explaining it live

One consolidated answer to: how did you plan this, how did you build it, how
does error-finding actually work for letters vs words, what framework is GOP
built on, and what technical questions should you be ready for.

---

## 1. The plan — how this was approached

**Starting point:** two source documents — a PadhAI/ReadNet pipeline
blueprint, and a "Turning speech into a reading score" problem write-up.
Both make the same core point: **a speech-to-text engine transcribes audio,
it does not grade reading.** Everything between "here's what was said" and
"here's the child's level" had to be built.

**The build order, in practice:**
1. **The rulebook first, code second.** Every decision about what counts as
   a mistake was written in plain language, with an ID, before it was coded —
   `RULES.md` per language. The idea: the rules are the deliverable a
   non-programmer (a teacher, a linguist) can review and argue with; the code
   is just an executable copy of the same document.
2. **A language-independent scorer.** Weighted Levenshtein alignment,
   written once, shared by every language, so adding Marathi (or later
   Telugu, Bangla) is a new rulebook, not new scoring logic.
3. **The acoustic layer**, added once text-only scoring worked: forced
   alignment + Goodness-of-Pronunciation (GOP), so the audio itself — not
   just the transcript — could catch mistakes an engine auto-corrects away.
4. **The ASER decision logic** — Pratham's actual test order (paragraph →
   story, or words → letters), pass thresholds, adaptive next-task logic.
5. **The web app**, built so the *same* scoring code runs two ways: on a
   local Python server, or entirely inside a browser tab with no server at
   all (for a free, keyless public demo).
6. **Iterate on real audio.** Nearly every real bug (see §5) was only found
   by actually recording and reading the numbers, not by reasoning about the
   code — this was tested continuously against real Hindi speech throughout.

---

## 2. The architecture — how it's built

```
speech engine (ASR) ──► transcript ─────┐
                                          ▼
audio ──► acoustic model ──► GOP/letter check ──► SCORING CORE ──► verdict
                                                   (normalise → align →
                                                    classify → forgive →
                                                    GOP → ASER decision)
```

- **`languages/<lang>/`** — the rulebook: text-normalisation rules (one
  function per rule ID), a confusion table (which sounds/letters get mixed
  up), and *forgiveness rules* that need to know the actual expected word
  (a dropped nasal is fine in गाँव, not in चाँद).
- **`score.py`** — the shared, language-independent aligner: weighted edit
  distance so similar words pair up, then classifies each difference
  (aspiration, voicing, vowel length, visual look-alike, …).
- **`g2p.py`** — grapheme-to-phoneme: converts written text into the sounds
  a reader actually makes, including the unwritten schwa and the rules for
  when it's dropped (कमल → /k ə m ə l/, not /k ə m ə l ə/).
- **`acoustic.py`** — the CTC math: Viterbi forced alignment, GOP scoring,
  and the letter-specific closed-set decision.
- **`pipeline.py` / `web/core.py`** — glues it all together into one scored
  result; this layer is plain Python + numpy with no I/O, so it's identical
  whether it runs on a server or inside a browser via Pyodide.
- **`web/server.py`** — the only place that touches API keys or picks an
  ASR engine.
- **`web/browser.py` + `static/offline.js`** — the free version: runs the
  same `core.py` under Pyodide, and the acoustic model as ONNX via
  onnxruntime-web, entirely on the visitor's machine.

---

## 3. How mistakes are actually found — letters vs words

This is the part worth being precise about, because **the two task types use
genuinely different logic**, not just different UI.

### Words and paragraphs
1. An ASR engine (Sarvam, Deepgram, or the local model) transcribes the
   reading.
2. The transcript is normalised and aligned word-by-word against the
   canonical text.
3. Forced alignment + GOP score every sound in every word from the raw
   audio, independently of what the transcript said.
4. **If a word's transcript says "correct" but GOP disagrees past the
   threshold, the word is rebuilt from what the audio actually holds** — this
   is how चाद (nasal dropped) gets caught even when the engine auto-corrects
   its transcript to the dictionary word चाँद.
5. That rebuilt word is re-run through the same rulebook — so the rulebook,
   not GOP directly, makes the final call (गाँव→गाव is forgiven, चाँद→चाद
   isn't).

### Letters
A single letter is under half a second of audio — no ASR engine can
transcribe that reliably (an engine will happily write का का गा गा for
क ख ग घ). So letters use a **different, closed-set decision** instead of
trusting any transcript:

1. For each expected letter, force-align roughly where it was said.
2. Score that stretch of audio against **only** the expected letter and the
   handful of letters it's commonly confused with (ख vs क, घ, …) — never the
   whole alphabet.
3. Whichever candidate scores highest **is** the verdict.
4. **GOP is calculated for letters too, and shown, but it never overrides
   this verdict** — the audio-rebuild step from the word/paragraph case is
   explicitly switched off for letters in the code
   (`if audio_decides and ... and not letter_task`). GOP for a letter is
   advisory only, because it's a much stricter, absolute test (must beat
   *every* sound including silence) versus the letter check's easier,
   relative one (beat 2–4 named rivals).

**In one sentence:** words are scored by ASR-text-plus-audio-correction
through a rulebook; letters are scored by a closed-set audio comparison, with
GOP riding alongside as a second opinion that can flag but never decide.

---

## 4. GOP — what it actually is, and what it's built on

**GOP is a score, not a model.** Goodness-of-Pronunciation: at a sound's
sharpest moment, how much more likely is the model's output for the expected
sound than for anything else — including silence.

**The model underneath:** Vakyansh wav2vec2 (`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`,
and a Marathi equivalent) — an open, CTC-based ASR model trained on ~4,200
hours of Hindi, MIT-licensed.

**Framework stack:**
- **PyTorch + Hugging Face `transformers`** — loading and running the model
  on the local server.
- **numpy** — the actual forced-alignment (Viterbi) and GOP math, written
  from scratch, model-agnostic — it only needs a `(frames, vocab)`
  log-probability matrix, so it isn't tied to this one model.
- **ONNX + onnxruntime-web** — the same model, exported and quantisation-
  tested, running client-side in the browser for the keyless public demo.
  (8-bit quantisation was tried and rejected — it changed transcripts and
  broke correct/incorrect separation; full precision was kept.)
- **A CTC phoneme model** (`wav2vec2-xlsr-53-espeak-cv-ft`) was also tried
  for sound-level (not letter-level) scoring — it separated correct from
  incorrect Hindi reading poorly in testing, so the letter-level Hindi model
  is the default, and the phoneme model is opt-in with a warning.

---

## 5. Real bugs found along the way (good technical-depth talking points)

- **Viterbi backtrack overflow:** the alignment path index was stored as
  `int8`, silently wrapping past 127 states on long sentences — fixed by
  forcing the arithmetic to Python `int`.
- **Wrong CTC blank ID:** fairseq-converted checkpoints ship a config naming
  `<pad>` as blank, but the model actually trained `<s>` as blank — detected
  at runtime by checking which special token the model emits most, instead
  of trusting the config.
- **GOP's peak search was too narrow:** it only looked inside the exact,
  sometimes single-frame window Viterbi assigned a sound, missing real peaks
  just outside it in fast speech — fixed by widening the search to the
  midpoint with each neighbouring sound (the same technique the letter check
  already used).
- **GOP threshold is uncalibrated on real voices:** tuned on clean
  synthetic test speech; real microphones score lower across the board.
  This is flagged everywhere as an open item, not hidden.

---

## 6. Technical questions to be ready for, with the short answer

| Question | Short answer |
|---|---|
| Why wav2vec2/CTC and not Whisper or a Conformer? | CTC gives per-frame, per-letter probabilities directly, which forced alignment and GOP both need. Whisper's attention-based decoding doesn't expose that as cleanly. |
| How is forced alignment implemented? | Viterbi over the standard CTC extended-state lattice (blank between every token), in numpy, O(T×S) time. |
| Why does GOP treat silence as a rival? | Without it, a skipped sound still "wins" against other letters and looks fine — comparing against blank is what catches a sound that was never said at all. |
| Why can't GOP override the verdict for letters? | Design choice to protect against false fails: GOP is uncalibrated on real voices, so for the narrower letter decision it's shown for review only, never counted alone. |
| How do you avoid false fails generally? | Extra words aren't counted (usually ASR hallucination), forgiveness rules run after alignment with the actual expected word in hand, and low-confidence GOP only flags, doesn't fail, until calibrated. |
| How does this extend to a new language? | New rulebook folder (`RULES.md` + one function per rule), a script/confusion table, a G2P if the script needs one — the scorer, aligner and GOP code don't change. |
| How is this tested? | A rulebook regression suite: every field case names the rule it tests; the suite fails if a rule has no test or a test cites a rule that doesn't exist. |
| How would this run offline on a phone (PadhAI's actual use case)? | The acoustic model already runs as ONNX; on-device would mean ONNX Runtime Mobile plus porting or embedding the pure-Python scoring core. |
| What's not finished? | GOP threshold calibration on real, teacher-labelled child recordings; the Hindi/Marathi rulebooks are drafts pending sign-off from Pratham's assessment team; Marathi has no dropped-nasal forgiveness rule yet (needs a linguist). |
| Why two separate normalisers (neutral vs forgiving)? | Benchmarking the ASR model fairly (neutral: only removes things that aren't really in the speech) versus scoring a child leniently (forgiving: also applies pedagogical rules) — using the forgiving one to benchmark the model would flatter it. |
