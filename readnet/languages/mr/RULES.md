# Marathi reading rules

This page explains, without code, how the app decides whether a child read a
Marathi word correctly. The executable version is `normalize_mr.py` in this
folder. Marathi follows the same five cleaning steps and the same mistake
counting as Hindi, described in [`../hi/RULES.md`](../hi/RULES.md). This page
lists only where Marathi differs.

## Differences from Hindi

| Rule | Marathi | Why |
|---|---|---|
| Dot (nukta) under any letter | Always removed | Marathi has no native sounds written with a dot |
| ॲ vs ऍ (the English "a" in ॲप, बॅट) | Same | Two computer codes for one letter |
| Eyelash ऱ (दऱ्या) vs र (दर्या) | Same | The same र sound, written two ways |
| ळ vs ल | **Different**, a mistake | They are different sounds (शाळा ≠ शाला) |
| Protected words | None yet | Needs real minimal pairs from a Marathi linguist |
| Interchangeable spellings | None yet | Same |

Chandrabindu, half nasals, punctuation, invisible marks and Latin-script
answers are handled exactly as in Hindi.

## Open work

The Marathi lists are deliberately short. We have not invented spelling
variants or protected words without a linguist's confirmation, because a wrong
entry would mark children wrong. Two things to collect from the field:

- Words that recognisers spell differently from school textbooks, such as old
  versus new orthography (केलें / केले).
- Pairs of words that differ only by a nasal mark or a dot.

Add each one to `normalize_mr.py`, with a row in `tests/field_cases.csv` that
proves it.
