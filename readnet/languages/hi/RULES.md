# Hindi reading rules

This page explains, without code, how the app decides whether a child read a
Hindi word correctly. It is for teachers, assessors and linguists. The
executable version of every rule is `normalize_hi.py` in this folder. If you
change one, change the other in the same edit.

## Why the app "cleans" text before comparing

The speech recogniser (ASR) writes down what it heard. There are often several
correct ways to write the same spoken word, and the recogniser may pick a
different one from the book. If the app compared letters directly, the child
would be marked wrong for the recogniser's spelling choice. So before comparing,
the app rewrites both the book text and the heard text into one standard
spelling. It does this in five steps, always in this order.

### Step 1: One code for each letter

A computer can store the same visible letter in more than one way. The app
converts every letter to a single standard form first. You will never see this
step, but without it, identical-looking words can fail to match.

### Step 2: Latin script back to Hindi

Some recognisers answer in English letters, like `ghar` for घर. The app
converts these back to Devanagari. Latin spelling cannot show every Hindi
sound: `sita` could be सिता or सीता, and `t` could be त or ट. So when a
converted word differs from the book only in vowel length (ि/ी, ु/ू),
dental versus retroflex (त/ट), or nasalisation, the child is **not** marked
wrong.

Devanagari digits (५) and English digits (5) are treated as the same.

### Step 3: Invisible marks and punctuation

Some characters change only how a word looks on screen and carry no sound.
Examples are the "zero-width joiner" and "non-joiner". These are removed. So is
punctuation (। , ! ?), because a child does not read it aloud.

### Step 4: Marks that do not change the sound

| Written difference | Treated as | Example |
|---|---|---|
| Chandrabindu ँ vs anusvara ं | Same | हूँ = हूं, चाँद = चांद |
| Dot under क़ ख़ ग़ ज़ फ़ (Urdu sounds) | Dot removed | ज़रा = जरा |
| Dot under ड़ ढ़ | **Kept**, a real Hindi sound | पढ़ ≠ पड़ |
| Half nasal before a letter of its own group | Same as anusvara | सन्त = संत, अङ्क = अंक |
| Half म or न before ह | **Kept**, a real cluster | तुम्हारा ≠ तुमहारा |

### Step 5: Word lists

**Protected words** keep all their marks, because the mark is what tells two
words apart:

- हँस (laugh) and हंस (swan)
- हँसी and हंसी

**Interchangeable spellings** are treated as one word, because they are spoken
the same way:

| Written | Treated as |
|---|---|
| गयी, गये, नयी, आये | गई, गए, नई, आए |
| लिये, दिये, किये, हुये | लिए, दिए, किए, हुए |
| चाहिये, जायेगा, जायेगी | चाहिए, जाएगा, जाएगी |

To add a word to either list, change `normalize_hi.py` and add a row to
`tests/field_cases.csv` that proves it.

## Counting mistakes

After the cleaning, the app lines up the book text against the heard text word
by word.

**Counted as a mistake:**
- A word read as a different word. The app records what kind of difference it
  was (see below).
- A word skipped or not read. If the child stops halfway, every unread word
  counts.

**Not counted as a mistake:**
- Repeating a word (घर घर).
- Starting a word and restarting it, or correcting yourself (गर… घर).
- Hesitation sounds (उम्म, हम्म, अं).
- **श read as ष, or ष as श.** Hindi speakers do not distinguish these sounds,
  and recognisers spell them either way.
- An extra word that is not in the text. It is usually the recogniser
  mishearing, not the child. It is shown in the report but not counted.

**Kinds of mistake**, recorded for the child's mistake profile:

| Kind | Meaning | Example |
|---|---|---|
| aspiration | breathy/plain letter swapped | घर → गर |
| voicing | voiced/unvoiced swapped | दाल → ताल |
| retroflex_dental | ट-group vs त-group | रानी → राणी |
| vowel_length | short/long vowel swapped | दिन → दीन |
| nasalisation | nasal mark added or dropped | हँस → हंस |
| visual | letters that look alike | धन → घन |
| matra / conjunct | vowel sign or half-letter dropped or wrong | तुम्हारा → तुमहारा |
| partial | word left unfinished | बाज़ार → बा |
| different_word | a different word altogether | |

## Single letters

When a child reads a letter aloud, they say क as "ka". Recognisers write that
as क, का or कअ, and the app accepts all three as क.

## Questions for the assessment team

These rules are the app's current best reading of ASER practice. They need
confirming by Pratham's assessment team:

1. Should skipping a whole line count as one mistake or one per word? Today it
   counts one per word.
2. Should extra words ever count? Today they never do.
3. Should dialect pronunciations count as correct, such as स for श, or ज for ज़
   (which is already allowed)? Today only श/ष is forgiven.
