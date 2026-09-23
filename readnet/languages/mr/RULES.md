# Marathi rulebook

Every decision the app makes about Marathi reading, in plain language.
`normalize_mr.py` is its executable copy: one function per rule, named after
the rule's id. Change the two in the same commit, and add a row to
`tests/field_cases.csv` that cites the rule. The rule format and the Neutral?
column are explained in [`../hi/RULES.md`](../hi/RULES.md). Shared rules S-00
to S-04 are in [`../../RULES_SHARED.md`](../../RULES_SHARED.md).

**Status: proposal, unsigned.** Nobody who wrote these rules is a Marathi
linguist. Rules marked "same as Hindi" are script-level and low risk. Every
rule still needs sign-off.

## Text rules

| Rule | Case | Decision | Why | Neutral? | Sign-off |
|---|---|---|---|---|---|
| MR-01 | Same word, different bytes | NFC | Reason 4 | yes | — |
| MR-02 | `ghar` for घर | Transliterate Latin to Devanagari | Reason 5 | yes | — |
| MR-03 | ५ vs 5 | Same digit | Same number | yes | — |
| MR-04 | ZWJ/ZWNJ, punctuation | Remove | No sound | yes | — |
| MR-05 | Any nukta (ज़िल्हा) | Always removed | Marathi has no native nukta sounds | yes | — |
| MR-06 | ँ vs ं | Same nasal marker | Consistent nasal markers | yes | — |
| MR-08 | आम्बा vs आंबा | Half nasal before its own group = anusvara | Two conventions, one sound | yes | — |
| MR-09 | ॲ vs ऍ; eyelash ऱ vs र | Same letter. **ळ is not ल** | ळ/ल are different sounds (शाळा ≠ शाला) | yes | — |

## Scoring rules

| Rule | Case | Decision | Why | Neutral? | Sign-off |
|---|---|---|---|---|---|
| MR-22 | श vs ष | Not a mistake | Same sound for speakers and engines | no | — |
| MR-23 | Letter क heard as का / कअ | Counts as क | How letters are named aloud | no | — |
| MR-24 | Romanised word differs only in length, retroflex or nasal | Not a mistake | Latin cannot show these | no | — |

## Deliberately missing

- **No dropped-nasal rule (Hindi HI-20).** In Marathi the anusvara often
  carries grammar, as in plural and oblique forms (घरांत → घरात changes the
  form). So dropping it is counted until a linguist says otherwise.
- **No protected-word list and no spelling-variant list.** Old and new
  orthography (केलें / केले) is a likely source of both. It needs field
  examples, not guesses.

## Open questions

1. Old versus new orthography in children's textbooks against engine output.
2. Annotation conventions: what were Marathi annotators told that is not
   written down?
3. Does a dropped anusvara ever leave the word intact, as in Hindi गाँव → गाव?
