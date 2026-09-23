# <Language> rulebook (template)

Copy this folder to `languages/<code>/`, rename `normalize_xx.py`, and register
the language in `languages/__init__.py`. **Do the work below in this order.
Code comes last.**

## 1. Archaeology (before any code)

- [ ] Find the annotation guidelines the annotators were given.
- [ ] Find who briefed the annotators, and ask what they said out loud that was
      never written down. Colombia: annotators were told to write "B" for the
      "buh" sound; the analyst was not told, and about a quarter of a
      double-annotated set disagreed.
- [ ] List every such convention here as a rule. The model learned it, so the
      post-processing must expect it.

## 2. Collect disagreements

- [ ] 100–200 real cases where the pipeline and a human scorer disagree. They
      go in `tests/field_cases.csv` with the human verdict and, after step 3,
      the rule they test.

## 3. Sit with a language expert

- [ ] For each case, ask: did the child actually read this wrong? Write the
      answer down. You do not need to speak the language to ask this well.

## 4. Write the rules here (then code them)

Most disagreements collapse into a handful of rules. Use this format:

### XX-01: <short name>
- **Case:** what the strings look like.
- **Decision:** what the app does.
- **Why:** the reasoning.
- **Neutral?** yes if the difference is not in the speech at all (bytes,
  script, invisible marks, spelling conventions), otherwise no.
- **Sign-off:** name, date.

Expect rules in these families (see `../hi/RULES.md` for worked examples):
Unicode normalisation · script/transliteration · invisible characters and
punctuation · marks that do not change the sound · word-specific exceptions ·
annotation conventions · context-dependent forgiveness (needs the expected
word) · letter naming.

## 5. Code, test, keep testing

One function per rule in `normalize_xx.py`, named after its id. The test
`test_every_case_names_a_real_rule_and_every_rule_has_a_case` fails until
every rule has at least one field case.
