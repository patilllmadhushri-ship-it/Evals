# ReadNet

ReadNet takes the text on the screen and what a child read aloud, and returns
the child's ASER level: **Beginner → Letter → Word → Paragraph → Story**. It
also gives the child's mistake profile. It implements the ReadNet / PadhAI
blueprint and the "Turning speech into a reading score" approach, for Hindi
and Marathi, with a template for the next languages.

## Try it: the test bench

```bash
py -m readnet.web
```

Then open http://127.0.0.1:8600. It is a plain page served by a small
Python server (`readnet/web/`), with no web framework. Pick a language and a
speech engine: Sarvam, Google, Deepgram or ElevenLabs (keys from `.env`), the
local wav2vec2 model, Mock, or "type what the child said". The child reads
each task aloud; the page records WAV in the browser. The app then
transcribes, colours every word (read right / forgiven, with the rule id /
mistake / extra sound), and walks the ASER order to a level.

**Pronunciation check (GOP) runs on every reading**, whichever engine writes
the transcript, and it works on sounds, not letters:

```
text on screen  घर
  → G2P (readnet/g2p.py)        /ɡʰ ə ɾ/   (the /ə/ is never written)
  → phoneme model, per 20 ms frame: probability of every sound
  → forced alignment (Viterbi)  /ɡʰ/ 1.52–1.58s  /ə/ 1.58–1.62s  /ɾ/ 1.62–1.64s
  → GOP per sound               how much better the expected sound fits its
                                frames than the strongest other sound
```

The page shows this per word, sound by sound, with times and GOP. The
phoneme model is `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (ungated, IPA
output, about 1.2 GB, downloaded on first use); a letter-level model can be
entered instead and is detected automatically. For letter tasks the same
model runs a closed-set check per letter: does this stretch of audio sound
like /kʰ ə/ (ख) or one of its confusions, /k ə/, /ɡʰ ə/…? Low-GOP sounds are
shown for review, never counted as mistakes, until the threshold is
calibrated on labelled child audio.

This needs `torch` and `transformers`, and on Windows the Microsoft Visual C++
Redistributable; the page says which is missing. Sarvam accepts at most 30
seconds per request, so longer recordings are split at quiet moments and
sent in parts. Each verdict can be disputed; disputes go to
`.stt_eval_runs/readnet_disputes.csv` as candidate field cases. Audio is never
stored. `py -m readnet.tests.test_web` runs a full session over the API.

## Student records: what each child cannot yet say

Choose a student in Setup (or add one) and every accepted reading is saved to
their record. The **Students** tab lists every student with their latest ASER
level and the letters they miss most; a student's page shows:

- **Letters and sounds to practise**: each letter the child has tried, how
  often, how often wrong, the last five results and the words it came up in,
  most often wrong first. A letter is marked right or wrong by the audio
  (the letter check, or the sound's GOP against the threshold) when the audio
  model ran, otherwise from the transcript. Differences the rulebook forgives
  are not counted against the child.
- **Words to practise**, the **ASER level over time**, every reading, and a
  CSV of every sound.

Records live in `.stt_eval_runs/readnet.db` on this computer (git-ignored):
names, transcripts and scores, never audio. `py -m readnet.web --db other.db`
uses a different file.

## Command line

```bash
py -m readnet.tests.test_cases
```

```bash
py -m readnet assess --lang hi --text "चाँद और गाँव" --transcript "चाद और गाव"
```

```bash
py -m readnet assess --lang hi --level word --text "घर" --audio child.wav --provider sarvam
```

```bash
py -m readnet benchmark --lang hi asr.csv
```

```bash
py -m readnet confusions --lang hi pairs.csv
```

```bash
py -m readnet evaluate levels.csv
```

- `assess` scores one reading and names the rule behind every forgiven
  difference. `--audio` goes through any `stt_eval` provider, with keys from
  `.env`.
- `benchmark` measures the model with the **neutral** normaliser, split by
  level. The CSV has columns `level,reference,hypothesis`.
- `confusions` estimates letter confusions from `expected,heard` pairs and
  shows where the hand-made tables disagree.
- `evaluate` compares against human levels. It reports kappa and the false
  fail rate from a CSV with columns `human_level,system_level`.

## Layout: one folder per language, four artifacts

```
readnet/
  RULES_SHARED.md             S-00..S-04 and the ASER decision — every language
  score.py                    alignment, counting, typing — language-independent
  languages/
    __init__.py               the contract: Rule, Forgiveness, ScriptTables, the two normalisers
    devanagari.py             script helpers + hand-made confusion tables
    hi/RULES.md               the Hindi rulebook (the deliverable)
    hi/normalize_hi.py        one function per rule, named hi_NN_...
    mr/RULES.md, normalize_mr.py
    _template/                start here for Spanish, Telugu, Odia, Bangla, Assamese
  tests/field_cases.csv       cases + human verdict + the rule each one tests
  tests/test_cases.py         runs them; fails if a rule has no case or a case cites no rule
  aser.py                     levels, pass rules, adaptive order
  acoustic.py                 CTC forced alignment, GOP, closed-set letter decisions
  benchmark.py                neutral ASR benchmark, split by level
  confusion.py                confusion matrix estimated from data
  metrics.py                  accuracy, weighted kappa, false fail rate, mistake precision/recall
  pipeline.py                 normalise → align → classify → judge
```

## The pipeline

```
raw transcript
→ text rules, in order (NFC, script, invisible marks, safe-to-ignore marks,
  word exceptions, spelling conventions)            languages/<xx>/normalize_xx.py
→ ALIGN against the canonical text (weighted)       score.py
→ classify: substitution / deletion / insertion
→ type substitutions: phonetic / visual / partial / different word
→ context rules that need the expected word         Forgiveness in normalize_xx.py
  (गाँव→गाव forgiven, चाँद→चाद counted)
→ shared leniencies (repeats, restarts, fillers)    RULES_SHARED.md
→ aggregate → mistakes, mistake words, profile, fluency, ASER level
```

**Two normalisers.** The *neutral* one uses only the text rules marked
Neutral, which remove differences that are not in the speech. It is used to
benchmark the model. The *forgiving* path adds the teaching rules, and is used
to score children. For Hindi and Marathi today, every text rule is neutral;
all the forgiveness lives in the scoring rules. That is why `benchmark`
reports both the neutral WER and the forgiving figure you must not publish.

## Decisions that protect the false fail rate

- Extra words are reported, not counted (S-04).
- Low GOP is flagged, not failed. `acoustic_doubts` lists words where the
  audio disagrees with a "correct" transcript. They count only with
  `count_acoustic_doubts=True`, which should wait until the threshold is
  calibrated on labelled child audio.
- Fluency is measured, and enforced only when a programme sets a number.

## What needs people, not code

1. **Reconcile with production Hindi.** The Hindi rulebook here is a
   reconstruction. Compare it rule by rule with the rules in production, and
   get sign-off.
2. **Field cases.** 100–200 real disagreements per language in
   `field_cases.csv`. The current 62 cases are constructed.
3. **Archaeology per language** before its rules (see `_template/RULES.md`),
   including the Hindi word-final schwa question.
4. **A model for the acoustic path.** `acoustic.Wav2Vec2Emissions` wraps a
   Hugging Face CTC checkpoint and has not been run here. The larger question
   for the ASR partner: can the offline model expose frame log-probabilities, a
   lattice, or a biasing hook? If so, `closed_set_decision` turns letter
   assessment from open transcription into "घ, or one of its known
   confusions?".
