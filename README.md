# Evals

**STT Evaluation Studio** — a Streamlit app for benchmarking speech-to-text accuracy against your own audio and ground truth.

Speech-to-text providers disagree with each other, and no provider is uniformly best across languages, accents and domains. This app answers one question with evidence instead of guesswork: *on our own audio, with our own correct transcripts, which STT provider performs best — and by how much?*

Upload recordings, supply the correct transcripts, run several providers over the same data, and get a ranked, explainable comparison across accuracy, meaning, cost and latency. No code, no CLI, no scoring math to understand up front.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Then walk the six steps in the sidebar. To try it with no API keys at all, pick the **Mock (offline demo)** provider in step 2 — it degrades your ground truth locally the way real systems do (number words become digits, fillers vanish, words drop), so the metrics ladder has something realistic to disagree about.

---

## The flow

| Step | You do | The app does |
|---|---|---|
| 1 · Upload | Upload audio, supply ground truth as `id,text` CSV or inline per clip | Decodes and normalises every clip, matches audio to ground truth by id, and reports specific errors — *"clip3.wav has no matching row in ground_truth.csv"* — before the run can start |
| 2 · Configure | Pick the language, the providers, the metrics | Shows only providers that support the language; estimates the cost before you spend anything |
| 3 · Credentials | Paste the API key(s) | Holds them in session memory for this run only |
| 4 · Run | Click Run | Transcribes every clip with every provider in parallel, scores each pair, streams partial results as they land |
| 5 · Review | Read the leaderboard, drill into any clip | Sortable leaderboard with the winner highlighted per metric; per-clip view with audio playback, every transcript side by side, and the judge's written reasoning |
| 6 · Export | Download | CSV or Excel, per-clip rows plus the aggregate summary |

---

## The metrics ladder

Plain literal comparison over-penalises transcriptions that differ in wording but preserve meaning — "500" vs. "five hundred rupees" is a WER error and a perfectly good transcription. So the app computes a ladder from strict to meaning-aware, and you see both the raw error and how much of it actually mattered.

| Metric | Type | What it answers | Default |
|---|---|---|---|
| **Word Error Rate (WER)** | deterministic | What fraction of words were substituted, deleted or inserted, after case/punctuation normalisation? Pooled across the dataset, not averaged per clip. | Always on |
| **Character Error Rate (CER)** | deterministic | The same at character level — more forgiving of spelling and segmentation differences, and the more meaningful figure where word boundaries are ambiguous. | Always on |
| **Intent & entity preservation** | LLM judge | Ignoring exact wording, did the transcript preserve the speaker's intent and the key entities (names, numbers, amounts)? | On |
| **LLM-WER / LLM-CER** | hybrid | Find the exact spots where the transcripts differ, ask an LLM whether each differing segment is a meaning-equivalent rewording, forgive the equivalent ones, then recompute WER/CER on the corrected text — directly comparable to plain WER/CER, so you can read off how much of the error was harmless. | On |
| **Semantic WER** | LLM judge | One call aligns, normalises and semantically forgives end to end, returning a WER-comparable error count. | On |
| **Semantic match** | LLM judge | Binary pass/fail per clip — does this mean the same thing? — with a written reason. | Optional |
| **Estimated cost** | derived | Audio minutes × each provider's published per-minute rate. | Always shown |
| **Latency** | measured | Response time per clip, reported as p50/p95/p99. | On |

Every LLM metric is **independently toggleable and independently fault-isolated**: turning one off to save cost, or having one fail mid-run, never removes or blocks the others or the deterministic metrics. Every judged verdict carries its reasoning into the per-clip view and the export.

Rate metrics aggregate as **pooled** figures — summed errors over summed reference length — so a three-word clip cannot outweigh a three-minute one.

---

## Design notes

**Provider abstraction.** Every provider implements one interface: given audio bytes and a language, return a transcript ([`stt_eval/providers/base.py`](stt_eval/providers/base.py)). Adding a provider is a subclass, a registry entry and a rate — it touches neither the scoring nor the UI layer.

**Languages.** 42 languages and locales — four English variants (accent matters as much as language for STT), twelve Indic, fourteen European, Arabic/Hebrew/Swahili, and eight East and Southeast Asian. Each provider declares which of them it covers, so step 2 shows only the providers that actually support your language: Sarvam is Indic-only by design, Deepgram and Google cover most of the list, OpenAI's Whisper family covers all but Odia. The app keeps one canonical code per language so results stay comparable, and each provider maps it to whatever its own API expects — Google, for instance, wants `or-IN` for Odia, `fil-PH` for Filipino and `pa-Guru-IN` for Punjabi. For Chinese, Japanese, Thai and Malay the UI points you at CER rather than WER, since word boundaries are ambiguous there. The support lists are a conservative bundled snapshot — providers add languages regularly, so check current docs if one you expect is missing.

**Judge indirection.** All LLM-based scoring routes through a single configurable judge client ([`stt_eval/judge.py`](stt_eval/judge.py)), so the judge model swaps without touching any metric.

**Concurrency.** Providers run in parallel with each other and clips run in parallel within a provider, bounded by a per-provider semaphore you control in step 2. A failure on one (clip, provider) pair is retried a few times and then recorded as a failure *for that pair only* — the run continues for every other pair.

**No blocking.** Transcription and scoring run on a background thread; the UI polls for progress, so the app never freezes on a long run.

**Resumability.** Results are written to SQLite as each pair completes, not at the end ([`stt_eval/store.py`](stt_eval/store.py)). A browser refresh or a dropped connection loses nothing, and re-running the same run id with an added clip or an added provider processes only the new work — completed pairs are neither re-transcribed nor re-scored.

**Audio normalisation, at 8 kHz or 16 kHz.** Every clip is converted to mono 16-bit PCM at one run-level sample rate before it reaches any provider, so accuracy differences reflect the provider rather than the encoding of whatever file was uploaded. Choose the rate in step 1:

- **8 kHz — narrowband** for telephony, IVR and call recordings. Benchmark at the rate your production audio actually arrives at: upsampling a phone call to 16 kHz invents no information but does change how some models behave, so a 16 kHz benchmark can mispredict how a provider performs on your phone traffic. On a narrowband run, Deepgram and Google switch to their telephony models (`nova-2-phonecall`, `phone_call`) automatically.
- **16 kHz — wideband** for mic capture, VoIP and most public datasets.

The app warns when a clip had to be upsampled, and providers that must be told the rate explicitly read it from the WAV header of the payload they are about to send, so the declared rate can never disagree with the bytes. WAV/FLAC/OGG decode via libsndfile; MP3/M4A/WebM fall back to `ffmpeg` if it is on your PATH.

**Secrets.** Provider and judge API keys live only in the current session's memory. They are never written to disk, to logs, or to any exported results file.

---

## Cost figures are estimates

Cost is computed as audio minutes × a bundled per-minute rate table ([`stt_eval/config.py`](stt_eval/config.py)). It does **not** model tiers, volume discounts, minimum billable durations or per-request rounding, and will not match a provider's invoice exactly. The same calculation runs before the run (on total uploaded duration) and after it (on the audio actually processed), so the two figures are comparable by construction. LLM judge spend is billed by token and is reported separately.

---

## Layout

```
app.py                      Streamlit UI — the six-step flow
stt_eval/
  config.py                 Languages, rate table, defaults
  audio.py                  Decode + normalise to mono 16 kHz PCM
  dataset.py                Upload/ground-truth matching and validation
  judge.py                  The single configurable LLM judge client
  runner.py                 Concurrent, retrying, resumable orchestration
  store.py                  SQLite persistence for partial results
  report.py                 Pooled aggregation, leaderboard, ranking
  costs.py                  Cost estimates and latency percentiles
  export.py                 CSV / Excel export
  providers/                One module per STT provider, one shared interface
  metrics/                  align.py · wer.py · llm_metrics.py · the registry
smoke_test.py               Offline end-to-end check (no keys, no network)
ui_test.py                  Headless render check for all six steps
```

## Tests

```bash
py smoke_test.py
```

```bash
py ui_test.py
```

`smoke_test.py` synthesises audio, runs the mock provider through the real runner, and checks alignment, validation, resumability, aggregation and export. `ui_test.py` renders every step of the Streamlit app headlessly. Neither needs an API key or network access.

---

## Not in scope (v1)

Live microphone capture and streaming transcription in the browser; training or fine-tuning any model; TTS or full voice-agent evaluation; multi-user accounts and a hosted dataset library; SSO, billing or usage metering beyond the estimated per-run cost.
