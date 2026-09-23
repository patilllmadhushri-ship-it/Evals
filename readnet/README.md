# ReadNet

ReadNet takes the text on the screen and what a child read aloud, and returns
the child's ASER level: **Beginner → Letter → Word → Paragraph → Story**. It
also gives a profile of the kinds of mistakes the child made. This package
implements the ReadNet / PadhAI pipeline blueprint for Hindi and Marathi.

```bash
py -m readnet.tests.test_cases
```

```bash
py -m readnet assess --lang hi --level paragraph --text "मेरा घर बड़ा है।" --transcript "मेरा गर घर बड़ा है"
```

```bash
py -m readnet assess --lang hi --level word --text "घर" --audio child.wav --provider sarvam
```

```bash
py -m readnet evaluate levels.csv
```

`--audio` sends the recording through any provider in `stt_eval`, using keys
from `.env`. `evaluate` reads a CSV with `human_level` and `system_level`
columns.

## How it maps to the blueprint

| Blueprint stage | Where | Status |
|---|---|---|
| Canonical text + audio in | `pipeline.assess_item` / `assess_task` | Built. Takes any ASR's transcript. |
| ASR | `stt_eval.providers` (Sarvam, Google, Deepgram, …) via the CLI | Built. These are cloud engines; an on-device child-speech model plugs in the same way. |
| Normalisation steps 1–5 | `languages/devanagari.py`, configured by `languages/hi/normalize_hi.py` and `languages/mr/normalize_mr.py` | Built |
| Forced alignment + GOP | `acoustic.py`: CTC Viterbi and both GOP forms, in numpy | Algorithm built and tested. A real model still has to be connected (see below). |
| Weighted Levenshtein + classification | `score.py` | Built |
| ASER aggregation + adaptive order | `aser.py` | Built |
| RULES.md per language | `languages/hi/RULES.md`, `languages/mr/RULES.md` | Built |
| Regression suite | `tests/field_cases.csv` + `tests/test_cases.py`, run in CI | Built with 42 constructed cases. Needs real field cases. |
| Kappa, precision/recall, false fail rate | `metrics.py` | Built |

## Decisions that protect the false fail rate

- **Extra words are reported, not counted.** An inserted word is far more often
  the ASR hallucinating than the child adding one (`Rules.count_insertions`).
- **Repeats, restarts, self-corrections and fillers are never mistakes.** This
  matches ASER practice.
- **Romanised ASR output is judged only on what Latin script can show.**
- **Low GOP is flagged, not failed.** A word whose audio disagrees with an
  "all correct" transcript is listed in `acoustic_doubts`. It becomes a mistake
  only with `count_acoustic_doubts=True`, which should wait until the GOP
  threshold is calibrated on labelled child audio.
- **Fluency (words per minute, pauses) is measured, not enforced.** ASER has no
  official number for it. Set `Rules.min_wpm` / `max_pause_seconds` once a
  programme picks one.

## What still needs outside input

1. **An acoustic model for GOP.** `acoustic.Wav2Vec2Emissions` wraps any
   Hugging Face CTC checkpoint, for example `ai4bharat/indicwav2vec-hindi`, and
   needs `torch` + `transformers`. It has not been run here. An adult-speech
   model will score children's voices low, so a child-speech fine-tune is the
   real requirement.
2. **Real field cases.** The 200-case suite the blueprint asks for should come
   from real disagreements between the app and assessors. Add them as rows to
   `field_cases.csv`.
3. **Human-labelled levels,** to run `evaluate` and report kappa and the false
   fail rate for real.
4. **The open questions** at the end of `languages/hi/RULES.md`, for Pratham's
   assessment team.
