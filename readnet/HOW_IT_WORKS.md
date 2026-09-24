# ReadNet: how it works

ReadNet listens to a child reading Hindi or Marathi aloud and tells the teacher
two things:

1. **The child's reading level** on Pratham's ASER ladder:
   Beginner → Letter → Word → Paragraph → Story.
2. **Exactly what the child got wrong**: which words, which letters, which
   sounds, so the teacher knows what to practise next.

It is built for the PadhAI setting: low-cost phones, rural schools, and a
teacher who must be able to trust the verdict. The rule that guides every
design choice is **never mark a child wrong when they read correctly**. A
false fail is the error that loses a teacher's trust.

---

## 1. What the user does

| Step | On screen | What happens |
|---|---|---|
| 1 | Choose the language and the speech engine in **Setup** | Hindi or Marathi. Engines: Sarvam, Deepgram, Google or ElevenLabs (cloud, needs a key), the local model (free, no key), or "type what the child said". |
| 2 | The **Paragraph** text appears | The ASER test always starts at the paragraph. |
| 3 | Press **Record**, the child reads, press **Stop** (or upload a file) | The browser records WAV audio. |
| 4 | Press **Score this reading** | Every word is coloured: green read right, amber difference forgiven (with the rule that forgave it), red mistake. Below it: *What the child got wrong*, the per-sound GOP table, and *How this was scored*, step by step. |
| 5 | Press **Accept and continue** | The app picks the next task the way ASER does: paragraph passed → story; paragraph failed → words → letters. |
| 6 | The final screen | The child's **level**, what to teach next, how the app got there, and **the letters this child got wrong** across the whole test, most-missed first. |

The **Quick check** tab scores a single reading against any text, which is
useful for trying a tricky word or a rule.

---

## 2. The pipeline

```
 child's voice
      │
      ▼
 1. SPEECH TO TEXT ─────────── any engine: Sarvam / Deepgram / Google / local model
      │   "मेरा गर बड़ा है"
      ▼
 2. NORMALISE BOTH TEXTS ────── the language rulebook (HI-01 … HI-10), each rule has an id
      │   same bytes, same script, no punctuation, one spelling per word
      ▼
 3. ALIGN WORD BY WORD ──────── weighted Levenshtein: similar words pair up
      │   घर ↔ गर
      ▼
 4. CLASSIFY EACH DIFFERENCE ── substitution / deletion / insertion,
      │                          and what kind: aspiration, voicing, vowel length,
      │                          nasal, look-alike letter, partial word …
      ▼
 5. FORGIVENESS RULES ────────── repeats, restarts, "umm", and rulebook exceptions
      │                          are not mistakes (S-01 … S-04, HI-20 …)
      ▼
 6. PRONUNCIATION CHECK (GOP) ── the audio itself, sound by sound (section 3)
      │   can overrule the transcript
      ▼
 7. ASER DECISION ────────────── paragraph/story: ≤ 3 mistakes to pass
      │                          words/letters: 4 of 5 right to pass
      ▼
 level + what the child got wrong
```

### Step 1: Speech to text
Any engine can be plugged in. Cloud engines return text. The local model
(Vakyansh Hindi wav2vec2, trained on about 4,200 hours of Hindi) returns
text *and* a probability for every letter in every 20 ms of audio, which the
pronunciation check needs. Sarvam accepts at most 30 seconds per request, so
longer recordings are cut at quiet moments and sent in parts.

### Step 2: Normalisation (the rulebook)
A speech engine and a textbook often spell the same spoken word differently.
Without normalisation, the child would be blamed for the engine's spelling.
Each language has a **rulebook** (`readnet/languages/hi/RULES.md`), and each
rule has an id, the case, the decision, the reasoning and a sign-off line.
Examples:

| Rule | Case | Decision |
|---|---|---|
| HI-01 | same word stored as different bytes | treat as equal |
| HI-02 | engine answers in Latin script, `ghar` | turn it back into घर |
| HI-04 | invisible joiners, punctuation | remove |
| HI-05 | ज़रा vs जरा (Urdu dot) | same; but ड़/ढ़ keep their dot |
| HI-06 | हूँ vs हूं | one nasal mark |
| HI-10 | गयी vs गई | one spelling |

The code has **one function per rule, named after its id**, and every test
case names the rule it tests. The test suite fails if a rule has no test, or
a test cites a rule that doesn't exist.

### Steps 3–5: Alignment, classification, forgiveness
The two texts are aligned word by word, so a misread word pairs with the
word it replaced. Each difference is typed, for example घर → गर is a
*phonetic substitution, aspiration dropped*.

Some differences are **not reading mistakes** and are forgiven, each citing
its rule:
- **S-01 to S-04 (every language):** repeating a word, a false start or
  self-correction, "umm", and an extra word (usually the engine inventing one).
- **HI-20:** a dropped nasal, गाँव → गाव. The word was still read.
- **HI-21**, the exception to HI-20: चाँद → चाद **is** a mistake, because the
  word is destroyed.

That last pair is the key design point. **Whether a difference is forgivable
can depend on which word was expected**, so these rules run *after*
alignment, when both words are known.

---

## 3. The pronunciation check (GOP)

Speech engines write the *dictionary* word. A child who says **चाद** often
comes back as **चाँद**, so a text-only check can't see the mistake. GOP checks
the audio itself.

```
 text on screen:   घर
       │
       ▼  G2P (readnet/g2p.py): text → the sounds a reader says
 sounds:           /ɡʰ ə ɾ/        (the /ə/ is never written; final ones are dropped)
       │
       ▼  the model scores every 20 ms frame: how likely is each letter?
 frames:           ░░▓▓░░▓░░░░
       │
       ▼  forced alignment (Viterbi): which frames carry which sound
 timing:           /ɡʰ ə/ 1.52–1.54s   /ɾ/ 1.62–1.64s
       │
       ▼  GOP per sound
 score:            /ɡʰ ə/ +9.5          /ɾ/ +12.4        → said right
```

**GOP for a sound** = how much more the model believes the expected sound
than the strongest alternative, at that sound's strongest frame. "Nothing was
said here" (silence) counts as an alternative, so a *left-out* sound scores
negative. Positive means said; negative means something else was said, or
nothing. The default threshold is **−2**, adjustable in Setup.

### The audio can correct the transcript
When a sound scores below the threshold, the word is **rebuilt from what the
audio actually holds**, for example चाँद with the nasal missing becomes चाद.
The rebuilt word then goes through the **same normalisation and rulebook**, so
चाद counts as a mistake (HI-21) while गाव stays forgiven (HI-20). Nothing is
special-cased for a particular word.

### Letters are judged by the audio
On a half-second clip of one letter, an open speech engine can't hear the
puff of breath: it wrote का का गा गा for क ख ग घ. So for letter tasks, each
letter's stretch of audio is compared **only against the expected letter and
the letters it is usually confused with** (ख against क, ग, घ …). This is the
closed-set decision the reading-assessment literature recommends.

### What the teacher sees
- **What the child got wrong:** letter, sound, word, and what was heard
  instead ("nothing" when a sound was left out).
- **Sounds:** for every word, each sound with its time span and GOP. Low
  sounds are red.

---

## 4. Evidence from testing

These tests used Hindi speech generated with Sarvam's text-to-speech, not
children's voices, so the thresholds still need calibrating on real
recordings marked by a teacher.

| Test | Result |
|---|---|
| Correct paragraph, transcript claims perfect | **0 mistakes**: no false fails |
| Paragraph with planted errors, transcript claims perfect | The audio alone found रंग→रख, घास→खास, गाय→डाय and 2 skipped words |
| Sounds in correctly read words | all above 0 (median +6.5) |
| घास read as खास | only /ɡʰ/ fell (−8.1); /aː/ and /s/ stayed positive |
| Letters क ख ग घ, read as क **क** ग **ग** | letter check caught both; Sarvam's text could not |
| Rulebook regression suite | 29 tests, 62 field cases, all passing |

---

## 5. Where it runs

| Mode | How | Keys | Notes |
|---|---|---|---|
| **Local app** | `py -m readnet.web` → http://localhost:8600 | cloud keys from `.env` (optional) | Hindi and Marathi, all engines |
| **Public demo from a laptop** | `py -m readnet.web --public` + a Cloudflare tunnel | none; `--public` ignores `.env` | free; works while the laptop is on |
| **Free online version** | https://madhushripatil032003-readnet.static.hf.space | none | runs entirely in the visitor's browser (below) |

### The free online version
There is no server. The page runs everything in the browser:
- the scoring code is the project's own Python, run by **Pyodide**;
- the Hindi model runs with **onnxruntime-web**, downloaded once (378 MB)
  and cached by the browser for later visits;
- the recording never leaves the visitor's computer.

It is Hindi only, because the Marathi model's licence does not allow
redistribution. The model was kept at full precision: 8-bit compressed
versions changed transcripts and marked correct words wrong.

---

## 6. Code map

```
readnet/
  languages/hi/RULES.md      the Hindi rulebook (the deliverable)
  languages/hi/normalize_hi.py   one function per rule
  languages/mr/…             Marathi; _template/ for new languages
  RULES_SHARED.md            rules for every language (S-00 … S-04) and the ASER decision
  score.py                   alignment, classification, forgiveness (language-independent)
  g2p.py                     text → sounds, with schwa deletion
  acoustic.py                forced alignment, GOP, letter check, model adapter
  pipeline.py                normalise → align → classify → GOP → verdict
  aser.py                    levels, pass rules, the ASER order
  benchmark.py               ASR accuracy with the neutral normaliser, split by level
  confusion.py               letter confusions learned from data
  web/core.py                scoring after speech-to-text (shared by server and browser)
  web/server.py              the local app's server (engines, keys, model loading)
  web/browser.py, static/offline.js   the in-browser version
  web/static/                the page
  deploy/                    Hugging Face Space builds
  tests/                     test_cases.py + field_cases.csv, test_web.py
```

---

## 7. Limits and next steps

- **Thresholds need real data.** GOP and the rulebook were tuned on
  synthetic speech. The next step is 100–200 real readings where a teacher
  and the app disagree, added to `tests/field_cases.csv`.
- **The Hindi rulebook is a reconstruction.** It must be checked against the
  rules already in production and signed off. Marathi needs a Marathi
  linguist.
- **The unwritten /ə/** shares its consonant's score, because the Hindi model
  hears them as one unit. The multilingual phoneme model that separates them
  judged Hindi poorly in testing.
- **ङ** is misjudged by the letter check; the model has rarely heard it.
- **Scaling to Telugu, Odia, Bangla, Assamese and Spanish** follows
  `languages/_template/`: find the annotation conventions first, collect
  disagreements, write the rules with a language expert, then code them.

---

## Suggested Loom walkthrough (about 5 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00 | The page | The problem: a speech engine transcribes; it doesn't judge reading. |
| 0:30 | Read the paragraph correctly, score | Green words, 0 mistakes, level passed. |
| 1:00 | Read घर as गर, score | The red word, its type (aspiration), and "What the child got wrong". |
| 1:40 | Quick check: read चाँद as चाद | The engine writes चाँद anyway; GOP hears the nasal is missing and marks it. Then गाँव → गाव: forgiven by rule HI-20. Same kind of change, different verdict, decided by the rulebook. |
| 2:40 | Open the Sounds table | G2P → 20 ms frames → forced alignment → GOP per sound. |
| 3:20 | Quick check: letters क ख ग घ | The engine hears का का गा गा; the letter check hears ख and घ. |
| 3:50 | Finish a full test | The ASER path, the level, and the letters to practise. |
| 4:30 | `RULES.md` and the test suite | Rules as the deliverable; every rule has a test. |
| 4:50 | The online link | Runs in the browser, no keys, free. |
