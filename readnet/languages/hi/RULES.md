# Hindi rulebook

Every decision the app makes about Hindi reading, in plain language. This
document is the deliverable. `normalize_hi.py` is its executable copy: one
function per rule, named after the rule's id. Change the two in the same
commit, and add a row to `tests/field_cases.csv` that cites the rule.

**Status: reconstruction, unsigned.** Hindi is already in production, and its
rulebook lives partly in code and partly in the head of the person who wrote
it. These rules were rebuilt from the problem statement and general Hindi
practice. They must be reconciled with the production rules and signed off
before they replace anything. A rule with no sign-off is a proposal.

Rules shared by every language (S-00 to S-04) are in
[`../../RULES_SHARED.md`](../../RULES_SHARED.md).

## How to read a rule

- **Case**: what the strings look like.
- **Decision**: what the app does.
- **Why**: the reasoning, so the decision can be argued with.
- **Neutral?**: "yes" means the difference is not in the speech at all, so the
  model benchmark uses this rule too. "no" means it is a teaching choice and
  applies only when scoring children.
- **Sign-off**: who agreed to it, and when.

## Text rules (applied to each string alone, in this order)

### HI-01: One code for each letter
- **Case:** the same visible word stored as different bytes (NFC vs NFD; क़ as
  one code point or as क + ़).
- **Decision:** convert both strings to Unicode NFC first.
- **Why:** identical-looking words must compare equal (Reason 4).
- **Neutral?** yes · **Sign-off:** —

### HI-02: Latin script back to Devanagari
- **Case:** the engine answers `ghar` for घर.
- **Decision:** transliterate Latin words to Devanagari.
- **Why:** a script difference is the engine's choice, not the child's (Reason 5).
- **Neutral?** yes · **Sign-off:** —

### HI-03: One set of digits
- **Case:** ५ vs 5.
- **Decision:** treat as the same digit.
- **Why:** the same number.
- **Neutral?** yes · **Sign-off:** —

### HI-04: Invisible marks and punctuation
- **Case:** zero-width joiner/non-joiner; । , ! ?
- **Decision:** remove them.
- **Why:** they change only how text looks. A child does not read punctuation
  aloud (Reason 4).
- **Neutral?** yes · **Sign-off:** —

### HI-05: The Urdu nukta is silent
- **Case:** ज़रा vs जरा; also क़ ख़ ग़ फ़.
- **Decision:** remove the dot, **except under ड़ and ढ़**, where it is kept.
- **Why:** most Hindi speakers do not make the Urdu sounds. ड़ and ढ़ are native
  Hindi sounds, so पढ़ ≠ पड़.
- **Neutral?** yes · **Sign-off:** —

### HI-06: Chandrabindu is written as anusvara
- **Case:** हूँ vs हूं; चाँद vs चांद.
- **Decision:** treat as the same nasal marker, except for words on HI-07.
- **Why:** engines and writers use the two interchangeably ("consistent nasal
  markers", Reason 4).
- **Neutral?** yes · **Sign-off:** —

### HI-07: Words that keep their chandrabindu (exception to HI-06)
- **Case:** हँस (laugh) vs हंस (swan); हँसी, हँसना.
- **Decision:** these words are not changed by HI-06.
- **Why:** the mark is the only thing telling two different words apart.
- **Risk:** if engines routinely write हंसना for हँसना, this rule causes false
  fails. Check it against field data before signing.
- **Neutral?** yes · **Sign-off:** —

### HI-08: Half nasal before a letter of its own group
- **Case:** सन्त vs संत; अङ्क vs अंक.
- **Decision:** write the half nasal as an anusvara. **Not** before ह or
  another nasal: तुम्हारा ≠ तुमहारा.
- **Why:** two spelling conventions for one sound.
- **Neutral?** yes · **Sign-off:** —

### HI-09: One code per letter shape
- **Case:** ॲ vs ऍ; ऱ vs र.
- **Decision:** treat as the same letter.
- **Why:** two code points for one letter and one sound.
- **Neutral?** yes · **Sign-off:** —

### HI-10: Interchangeable spellings
- **Case:** गयी/गई, गये/गए, नयी/नई, आये/आए, लिये/लिए, दिये/दिए, किये/किए,
  हुये/हुए, चाहिये/चाहिए, जायेगा/जाएगा, जायेगी/जाएगी.
- **Decision:** fold each pair to one spelling.
- **Why:** spoken identically. Both spellings are standard.
- **Neutral?** yes · **Sign-off:** —

## Scoring rules (need the expected word, so they run after alignment)

The problem statement's key point applies here. Whether a difference is
forgivable can depend on *which word was expected*, so these rules cannot be
applied to each string alone.

### HI-20: A dropped nasal is not assessed
- **Case:** गाँव read as गाव. The word was read; only the nasalisation is missing.
- **Decision:** not a mistake, unless the word is on HI-21.
- **Why:** we test decoding, not elocution, and the meaning survives (Reason 2).
  Only a *dropped* nasal is forgiven. An added one (गाव → गांव) still counts.
- **Neutral?** no. Forgiving only · **Sign-off:** —

### HI-21: Words where a dropped nasal destroys the word (exception to HI-20)
- **Case:** चाँद read as चाद. It is the same surface difference as गाँव → गाव.
- **Decision:** count it as a mistake. The current list is चाँद and हँस.
- **Why:** the word is gone (Reason 3). This list is the hand-maintained
  exception list that the literature describes as the state of the art. Grow
  it from field disagreements.
- **Neutral?** no · **Sign-off:** —

### HI-22: श and ष are one sound
- **Case:** विशेष read or written as विशेश.
- **Decision:** not a mistake.
- **Why:** Hindi speakers do not distinguish them, and engines spell either.
  स for श is **not** forgiven (open question 3).
- **Neutral?** no · **Sign-off:** —

### HI-23: A spoken letter carries its vowel
- **Case:** a letter task shows क. The child says "ka", and the engine writes
  क, का or कअ.
- **Decision:** all three count as reading क.
- **Why:** that is how letters are named aloud.
- **Neutral?** no · **Sign-off:** —

### HI-24: Romanised output is judged only on what Latin can show
- **Case:** the engine writes `sita` for सीता. After transliteration it reads
  सिता.
- **Decision:** for words that arrived in Latin script, forgive vowel length,
  dental/retroflex and nasalisation differences. Nothing else is forgiven:
  `gar` for घर still counts.
- **Why:** Latin spelling does not carry those distinctions, so the difference
  is the transliterator's guess, not the child's reading.
- **Neutral?** no · **Sign-off:** —

## Types of mistake (for the mistake profile)

| Type | Meaning | Example |
|---|---|---|
| aspiration | breathy/plain letter swapped | घर → गर |
| voicing | voiced/unvoiced swapped | दाल → ताल |
| retroflex_dental | ट-group vs त-group | रानी → राणी |
| vowel_length | short/long vowel swapped | दिन → दीन |
| nasalisation | nasal mark added, or dropped where HI-21 applies | चाँद → चाद |
| visual | letters that look alike | धन → घन |
| matra / conjunct | vowel sign or half letter dropped or wrong | तुम्हारा → तुमहारा |
| partial | word left unfinished | बाज़ार → बा |
| different_word | a different word altogether | |

The phonetic/visual split uses two hand-made lists
(`languages/devanagari.py`). `py -m readnet confusions` estimates the real
confusions from field pairs and shows where the lists and the data disagree.

## Open questions

1. **Word-final schwa.** Were annotators consistent about word-final schwa?
   If not, paragraph-level error counts are silently inflated. This is an
   archaeology task: ask the people who briefed the annotators.
2. **Annotation conventions.** Were annotators told to write anything a
   particular way that is not written down anywhere? That was the Colombia "B"
   problem. Any such convention becomes a rule here.
3. **Dialect sounds.** Should स for श count as correct? Today it does not.
4. **Skipped lines.** Should a skipped line count as one mistake or one per
   word? Today it is one per word.
