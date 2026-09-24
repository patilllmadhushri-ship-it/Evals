# Akshar — Loom script (starts with letters)

About 7–8 minutes. Read the **SAY** lines roughly as written; **SHOW**
tells you what to click. Practice once before recording — you don't need to
memorise it, just know the order.

---

## 0. Cold open (15s)

**SHOW:** the app open, Setup panel visible, nothing scored yet.

**SAY:**
> "This is Akshar. It listens to a child reading Hindi or Marathi aloud and
> tells the teacher two things: their ASER reading level, and exactly which
> letters and sounds they got wrong. I'll start with the hardest case —
> single letters — because that's where speech recognition normally falls
> apart."

---

## 1. Letters: the problem (45s)

**SHOW:** Quick check tab → Task type: Letters → type `क ख ग घ`.

**SAY:**
> "A child reading letters says each one for less than half a second — 'ka',
> 'kha', 'ga', 'gha'. A normal speech engine can't hear the difference. Watch
> what happens if I feed it a recording where the child actually mixed up ख
> and घ."

**SHOW:** pick **Sarvam AI**, play/upload a letters recording with mistakes
(or record live if you're confident with the mic), press **Score**.

**SAY (pointing at "Heard"):**
> "Sarvam wrote का का गा गा — it flattened both real letters into the wrong
> ones. If we trusted this text alone, we'd either mark everything wrong, or
> miss the real mistakes entirely."

---

## 2. Letters: the fix — the letter check (60s)

**SHOW:** point at the **Letter check** section / the wrong-letters row.

**SAY:**
> "So for letters, we don't trust the transcript. We run a second, local
> speech model directly on the audio, and instead of asking it to transcribe
> freely, we ask a narrower question for each letter: *does this sound match
> the expected letter, or one of the letters it's usually confused with?*
> For ख, that's ख, क, घ — not the whole alphabet. That's a much easier,
> more reliable question, and it's what actually decides the verdict here."

**SHOW:** point at the correct/wrong marks — this should now correctly show
ख and घ as wrong, matching what was actually read.

---

## 3. Letters: GOP is shown, but never decides (60s)

**SHOW:** the **Sounds (GOP)** table under the same result, negative numbers
visible even on letters that were marked correct.

**SAY:**
> "You'll also see a GOP score under each letter — Goodness of Pronunciation.
> This is a second, much stricter check on the same recording. The letter
> check only has to beat two or three named rivals. GOP has to beat *every*
> sound the model knows, including silence, at that letter's single sharpest
> moment. When we say a letter aloud we say 'ka', not just 'k', so the vowel
> competes with the consonant and the score often goes negative — even on a
> correct reading.
>
> That's why, for letters, GOP is shown for review only and never changes the
> verdict by itself. It's a hint for a teacher to double-check, not proof of
> a mistake. The letter check is what decides."

---

## 4. Words and paragraphs: GOP earns its keep (90s)

**SHOW:** Quick check → type text `चाँद और गाँव`, type/record `चाद और गाव`
(the child dropped both nasals), Score.

**SAY:**
> "For whole words, it's the opposite: GOP *does* change the verdict, and
> here's why it has to.
>
> Speech engines auto-correct. A child says चाद — without the nasal — and the
> engine still writes the dictionary word, चाँद. A text-only check would call
> that perfect.
>
> GOP listens to the actual audio, sound by sound, and if the nasal simply
> isn't there, it rebuilds the word from what was really said — चाद instead
> of चाँद — and that corrected word goes through the same rulebook as
> everything else."

**SHOW:** point at चाँद → marked wrong, note "audio check heard चाद".
Then point at गाँव → marked *correct* despite the same kind of dropped nasal.

**SAY:**
> "And here's the important part: चाँद → चाद is marked wrong, but गाँव → गाव,
> right next to it, is marked correct — same missing sound. That's not
> inconsistency, it's a deliberate rule. Dropping the nasal in गाँव still
> leaves a real, understandable word — गाव. Dropping it in चाँद destroys the
> word entirely. Every rule like this has an ID and a written justification
> in the rulebook, so nothing here is a hidden guess."

---

## 5. Under the hood: how a sound gets scored (60s)

**SHOW:** the **Sounds** table on any word result, e.g. घर.

**SAY:**
> "One more layer, quickly. Before any of this, the written word घर is
> converted into the sounds a reader actually says — /ɡʰ ə ɾ/ — including a
> vowel that's never written down. The recording is cut into 20-millisecond
> frames, and an algorithm called forced alignment lines up each of those
> sounds against the exact frames where it was said. GOP is then calculated
> per sound, not per word, which is why we can point at exactly which letter
> in a word went wrong, not just that the word was wrong."

---

## 6. Full test → the level, and what to practise (60s)

**SHOW:** Assessment tab, read the paragraph (or Mock it), Accept, continue
through to a final level.

**SAY:**
> "Putting it together: the app follows Pratham's own ASER test order —
> paragraph first, then story if that passes, or words and letters if it
> doesn't. At the end you get the child's level, and — this is the part a
> teacher actually needs — a combined list of every letter this child got
> wrong across the whole test, ranked by how often, so they know exactly
> what to practise next."

**SHOW:** the "Letters this child got wrong" summary table.

---

## 7. Honesty: what's not finished (45s)

**SAY:**
> "Two things I want to be upfront about.
>
> First, GOP's threshold was tuned on clean, computer-generated test speech.
> Real microphones and real children's voices are messier, so right now it
> can flag correct sounds too. I already found and fixed one real bug in the
> alignment that caused this, but full calibration needs real recordings
> marked by an actual teacher — that's the next step, not a solved problem.
>
> Second, the rulebook itself — which differences are forgivable — is a
> first draft built from the ASER documentation, not yet signed off by
> Pratham's assessment team. Every rule has an ID specifically so it can be
> reviewed and challenged line by line."

---

## 8. Where this can run (30s)

**SHOW:** the free public link in a browser tab (or just mention it).

**SAY:**
> "This runs three ways: as a local app with any speech engine, as a link
> shared from a laptop, or — the version I'd send anyone to try — entirely
> inside their own browser, no server, no account, no API key. The speech
> model itself runs on their computer after one download, and the recording
> never leaves it."

---

## 9. Close (15s)

**SAY:**
> "So: real speech engines can't judge reading on their own — they
> transcribe, they don't grade. Akshar adds the layer that does: a rulebook
> that says what actually counts as a mistake, and an audio check that
> catches what the transcript hides. Happy to go deeper into any part of
> this — the rules, the scoring code, or the pronunciation model."

---

## Quick reference: if asked live

| Question | Short answer |
|---|---|
| "What's GOP?" | A score, not a model — Goodness of Pronunciation, calculated from an open-source Hindi ASR model's frame-by-frame probabilities. |
| "Which model?" | Vakyansh wav2vec2, ~4,200 hours of Hindi, MIT licence. |
| "Why do letters and GOP disagree?" | Letter check = relative, beats 2–3 named rivals. GOP = absolute, beats everything including silence. Different strictness, by design. |
| "Why doesn't GOP fail a child on letters?" | Uncalibrated on real voices — a low score is a hint to review, never a verdict on its own. |
| "What's next?" | Real recordings + teacher-marked ground truth to calibrate thresholds; sign-off on the rulebook from Pratham's assessment team. |
