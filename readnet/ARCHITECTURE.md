# Akshar: architecture and system design

Akshar turns a child's reading aloud into an ASER reading level and a list
of the exact letters and sounds they got wrong. This document describes how
the system is built. For the user-facing flow, see
[HOW_IT_WORKS.md](HOW_IT_WORKS.md).

---

## 1. Goals and constraints

| Goal / constraint | What it means for the design |
|---|---|
| **Never fail a child who read correctly** | Every "mistake" must survive normalisation and the rulebook. Uncertain evidence (low GOP) is flagged, not counted, unless a rule says so. |
| **Judge reading, not transcription** | A speech engine only writes text. Scoring, rules and pronunciation checks are Akshar's own layers. |
| **Explainable verdicts** | Every forgiven difference cites a rule id; every result shows each pipeline stage. |
| **Many languages, few experts** | One language-independent scorer; each language is a pluggable rulebook plus data. |
| **Offline, low-cost devices (PadhAI)** | The scoring core is plain Python + numpy; the model runs from a portable ONNX file. |
| **Free to demo, no keys** | The whole system can run inside a browser, with no server. |
| **Privacy** | No audio is stored. The in-browser version never uploads the recording. |

---

## 2. High-level architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENT: web page (index.html, app.js, style.css)                         │
│  record WAV · upload · ASER flow · results, GOP table, wrong letters      │
└───────────────┬──────────────────────────────────────────────────────────┘
                │  JSON API: /api/config  /api/score  /api/next
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RUNTIME: one of                                                          │
│   (a) local server   readnet/web/server.py   (Python, http.server)       │
│   (b) in-browser     static/offline.js + Pyodide + onnxruntime-web        │
└───────┬───────────────────────────────┬──────────────────────────────────┘
        │                               │
        ▼                               ▼
┌─────────────────────┐     ┌────────────────────────────────────────────────┐
│  ASR LAYER           │     │  ACOUSTIC MODEL                                 │
│  speech → text       │     │  Vakyansh wav2vec2 (Hindi / Marathi)            │
│  · Sarvam, Deepgram, │     │  audio → log-probability of every letter,       │
│    Google, ElevenLabs│     │  every 20 ms ("emissions")                      │
│  · or the local model│     │  torch (server) or ONNX (browser)               │
└─────────┬───────────┘     └───────────────────┬────────────────────────────┘
          │ transcript                           │ emissions
          ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SCORING CORE  readnet/web/core.py (pure Python + numpy)                 │
│                                                                           │
│  normalise ─► align ─► classify ─► forgive ─► GOP / letter check ─► ASER  │
│  (languages/)  (score.py)          (rules)    (acoustic.py, g2p.py) (aser)│
└──────────────────────────────────────────────────────────────────────────┘
                │
                ▼
      result JSON: verdict, word ops, wrong letters, sounds + GOP, pipeline trace
```

**The key architectural decision:** everything after speech-to-text lives in
one pure module (`core.py`, numpy only, no I/O, no keys). The same code runs
on the server and in the browser, so both give identical results. This was
verified on the same recordings.

---

## 3. Components

### 3.1 Client (`readnet/web/static/`)
- **Recorder:** Web Audio → 16-bit PCM WAV in the browser, so no converter is needed on the server. Upload is the fallback.
- **ASER flow controller:** calls `/api/next` after each accepted task to get the next task (paragraph → story, or words → letters) or the final placement.
- **Renderers:** coloured words, "What the child got wrong", the per-sound GOP table, and the step-by-step pipeline trace.
- **Resilience:** start-up loads settings first and shows a clear message if the server is older than the page, instead of a blank screen.

### 3.2 ASR layer (server mode)
- A provider interface from `stt_eval/providers/`: one class per engine, same `transcribe(audio, language)` call.
- Engine-specific handling, e.g. Sarvam's 30 s limit: long audio is split at the quietest 100 ms and sent in parts.
- **Keys** come from `.env`. `--public` mode ignores `.env` entirely, so a shared link can never spend credits.

### 3.3 Acoustic model (`readnet/acoustic.py`)
- **Model:** Vakyansh wav2vec2 (base, CTC, ~4,200 h of Hindi, MIT licence). It is an ASR model; Akshar uses its *per-frame probabilities*, not just its text.
- **Adapters:** `Wav2Vec2Emissions` (torch, server) and ONNX via onnxruntime-web (browser).
- **Blank detection:** the model's real CTC blank is detected from its output. The fairseq-converted checkpoints ship a config naming the wrong one.

### 3.4 Scoring core
| Module | Responsibility |
|---|---|
| `languages/` | Per-language **rulebook**: text rules (one function per rule id), script tables (phonetic/visual confusion pairs), context rules that need the expected word (`Forgiveness`). |
| `score.py` | Language-independent weighted Levenshtein alignment, error typing, shared leniencies (repeats, restarts, fillers, extra words). |
| `g2p.py` | Text → sounds, including the unwritten schwa and its deletion rules; traces each sound to its letter. |
| `acoustic.py` | CTC forced alignment (Viterbi), GOP per unit with blank as a rival, closed-set letter decisions. |
| `pipeline.py` | Assembles one item: normalise → score → GOP → **audio correction** (rebuild a word from the audio, re-score through the rulebook). |
| `aser.py` | Levels, pass rules (≤ 3 mistakes; 4 of 5 right), the adaptive task order, placement. |
| `web/mistakes.py` | Turns a result into "wrong letters": letter, sound, word, what was heard. |
| `web/core.py` | The whole post-ASR scoring for one reading, as JSON the page renders. |

---

## 4. The scoring request, step by step

```
Client                    Runtime                        Core
  │ POST /api/score          │                              │
  │ {text, level, audio,     │                              │
  │  engine, gop, threshold} │                              │
  │─────────────────────────►│ decode + resample to 16 kHz  │
  │                          │ ASR: transcript              │
  │                          │ model: emissions [T × V]     │
  │                          │─────────────────────────────►│ 1 normalise text + transcript (rule ids)
  │                          │                              │ 2 weighted word alignment
  │                          │                              │ 3 classify each difference
  │                          │                              │ 4 forgiveness rules
  │                          │                              │ 5 forced alignment → GOP per sound
  │                          │                              │ 6 letters: closed-set letter check
  │                          │                              │ 7 audio correction → re-score via rulebook
  │                          │                              │ 8 ASER task verdict
  │                          │◄─────────────────────────────│ result JSON
  │◄─────────────────────────│                              │
  │ render; on Accept:       │                              │
  │ POST /api/next {outcomes}│─────────────────────────────►│ next task or final placement
```

### Main data structures
| Object | Contents |
|---|---|
| `Emissions` | `log_probs[T, V]`, vocabulary, seconds per frame (~20 ms), blank id |
| `WordOp` | kind (match/substitution/deletion/insertion), expected and heard word, type, counts-as-mistake, rule id, note |
| `WordEvidence` / `UnitEvidence` | per word and per sound: start, end, GOP, what the audio held |
| `TaskOutcome` | level, passed, mistakes, correct/total, reasons |
| Result JSON | `outcome`, `ops`, `wrong_letters`, `letter_check`, `pipeline`, transcripts, notes |

---

## 5. Pronunciation subsystem (GOP) in detail

```
expected text ──► G2P ──► sounds /ɡʰ ə ɾ/
                                 │
audio ──► model ──► emissions ───┤
                                 ▼
                     forced alignment (Viterbi over CTC states)
                                 │  frames for each sound → timings
                                 ▼
          GOP(sound) = max over its frames of
                       log P(expected) − log P(strongest alternative, silence included)
                                 │
            ┌────────────────────┴─────────────────────┐
            ▼                                          ▼
   words/paragraphs:                          letters:
   GOP < threshold → rebuild the word         closed-set check: expected letter
   from what the audio holds → rulebook       vs its known confusions → verdict
   decides (e.g. चाँद→चाद counts, HI-21;      (GOP shown for review only)
   गाँव→गाव forgiven, HI-20)
```

Why silence is a rival: forced alignment always places every expected sound
somewhere. Without comparing against "nothing said", a skipped sound would
still look fine.

Why letters use a closed set: on a half-second clip, open ASR cannot hear
aspiration (it wrote का का गा गा for क ख ग घ); a choice among a few likely
letters can.

---

## 6. Deployment topologies

| | (a) Local app | (b) Public link from a laptop | (c) Free online (in-browser) |
|---|---|---|---|
| Command / URL | `py -m readnet.web` | `py -m readnet.web --public` + Cloudflare tunnel | https://madhushripatil032003-readnet.static.hf.space |
| Where scoring runs | Python server | Python server | Visitor's browser (Pyodide) |
| Where the model runs | torch on the server | torch on the server | onnxruntime-web in the browser |
| ASR | cloud engines or local model | local model only | local model only |
| Keys | `.env` | none (ignored) | none |
| Languages | Hindi, Marathi | Hindi, Marathi | Hindi (Marathi model has no redistribution licence) |
| Cost | free | free (laptop must stay on) | free (static Space; model in a free model repo) |

In-browser specifics: the model is 378 MB at full precision, downloaded once
and kept in Cache Storage. 8-bit versions were rejected because they changed
transcripts and marked correct words wrong. The page's `/api/*` calls are
answered inside the browser by `offline.js`, so `app.js` is identical in all
modes.

---

## 7. Extensibility: adding a language

```
languages/<xx>/
  RULES.md            decisions in plain language, with ids and sign-off
  normalize_xx.py     one function per rule + PROFILE (rules, script tables, forgiveness)
tests/field_cases.csv real disagreements, each naming the rule it tests
```

- `score.py`, `aser.py`, `acoustic.py` and `core.py` do not change.
- The test suite fails if a rule has no case, or a case names a missing rule.
- The acoustic side needs a CTC model for the language, or a phoneme model plus that language's G2P.
- Template: `languages/_template/` (archaeology of annotation conventions first, then disagreements, then rules, then code).

---

## 8. Key design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Rules as the deliverable, with ids | Arguable by educators, survives staff changes, makes each new language cheaper | Slower to write than ad-hoc code |
| Context rules after alignment | Some forgiveness depends on the expected word (गाँव vs चाँद) | More complex than string cleaning |
| Two normalisers (neutral vs forgiving) | Benchmark the model fairly; score children leniently | Two paths to keep consistent |
| Extra words not counted (S-04) | Usually ASR hallucination; protects against false fails | A child's real addition is missed |
| Low GOP flagged, not counted, for letters | Threshold not calibrated on real voices | Some real mispronunciations only shown, not scored |
| Audio correction via the rulebook | Catches the engine writing the dictionary word, without word-specific code | Depends on the GOP threshold |
| Hindi letter model for GOP (not a phoneme model) | Separated right from wrong reading in tests; the multilingual phoneme model did not | Unwritten /ə/ shares its consonant's score |
| Full-precision model in the browser | 8-bit changed results | 378 MB first download |
| Pure-Python core | Same code on server, browser, and later on-device | Browser speed depends on the visitor's computer |

---

## 9. Non-functional properties

| Property | Current state |
|---|---|
| **Latency** | local model: ~0.5 s for a 5 s reading (laptop); Sarvam + GOP: ~13 s; in-browser: ~5 s |
| **Privacy** | no audio stored anywhere; in-browser version never uploads it; no student records |
| **Security** | keys only in `.env` (git-ignored); `--public` strips them; static files served only from `static/` |
| **Reliability** | per-stage failure notes (e.g. "GOP skipped: …") instead of a failed request |
| **Testing** | rulebook suite (29 tests, 62 field cases), end-to-end API test over HTTP, browser-path vs server-path equivalence check |

---

## 10. Path to production (PadhAI)

Proposed, not built:

1. **On-device:** run the ONNX model with ONNX Runtime Mobile on Android. Run the scoring core as Python (e.g. Chaquopy) or port it to Kotlin, using `field_cases.csv` as the conformance suite. Fully offline; sync only results.
2. **Calibration loop:** collect real readings where the app and a teacher disagree; add them as field cases; set the GOP threshold per sound from that data.
3. **Child-speech model:** fine-tune the acoustic model on children's recordings; ask the ASR partner for frame probabilities or a biasing hook for closed-set letter decisions.
4. **More languages:** Telugu, Odia, Bangla, Assamese and Spanish through the template, starting with annotation-convention archaeology and a language expert.
