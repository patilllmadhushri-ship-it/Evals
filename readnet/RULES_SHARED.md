# Shared rules

These rules are the same in every language. They are implemented once, in
`score.py`, and never copied into a language folder.

| Rule | Case | Decision | Why |
|---|---|---|---|
| S-00 | Any difference no language rule forgives; a skipped word | A mistake. Each unread word counts, so stopping halfway counts every word left. | Real errors are the signal (Reason 1). |
| S-01 | घर घर (a repeated word) | Not a mistake | ASER does not penalise repeats. |
| S-02 | गर… घर (a false start or self-correction) | Not a mistake | ASER counts a self-corrected word as read. |
| S-03 | उम्म, हम्म (a hesitation sound) | Not a mistake | It is not a word in the text. |
| S-04 | An extra word not in the text | Reported, **not counted** (switch with `Rules.count_insertions`) | Far more often the engine hallucinating than the child adding a word. A false fail costs more than a missed insertion. |

## The ASER decision

These rules come from Pratham's ASER tool (`aser.py`):

- The child starts at the **paragraph**. With 3 mistakes or fewer, the child
  moves on to the **story**. With 3 or fewer there, the child is at Story
  level; otherwise at Paragraph level.
- If the child fails the paragraph, they read **words**, then **letters**.
  Each needs 4 of 5 correct. Failing both places the child at Beginner.
- **Fluency** (words per minute, long pauses) is measured whenever timings
  exist. It is only enforced once a programme sets a number, because ASER has
  no official one.
