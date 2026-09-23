"""The per-language contract, and the registry of languages.

Each language is one folder with four artifacts:

* ``RULES.md`` — every decision in plain language: the case, the decision, the
  reasoning, and who signed it off. This is the deliverable; the code follows it.
* ``normalize_xx.py`` — one function per rule, named after the rule's id, and a
  ``PROFILE`` that lists them in order.
* ``score.py`` (shared, package level) — alignment, counting and typing. Written
  once, reused unchanged: everything script-specific reaches it through the
  ``ScriptTables`` and ``Forgiveness`` a profile carries.
* ``tests/field_cases.csv`` (shared) — cases with human verdicts, each pointing
  at the rule it tests.

Two normalisers per language, deliberately separate:

* **neutral** — "how good is this ASR model?". Only the rules that remove
  differences which are not in the speech at all: byte encoding, script,
  invisible characters, spelling conventions. Used for benchmarking.
* **forgiving** — "did the child read the word?". Neutral, plus the
  pedagogical rules (what we do not assess). Used for scoring children.
  Benchmarking with this one flatters the model.

Adding a language is a new folder and one entry in ``_LOADERS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


def pairs(*items: str) -> frozenset[frozenset[str]]:
    """Unordered character pairs from two-character strings: pairs("कख", "गघ")."""
    return frozenset(frozenset(p) for p in items)


@dataclass(frozen=True)
class ScriptTables:
    """What score.py needs to know about a script to type a substitution."""

    #: label -> unordered pairs that sound alike (aspiration, voicing, ...).
    phonetic: dict[str, frozenset[frozenset[str]]] = field(default_factory=dict)
    #: Pairs that look alike.
    visual: frozenset[frozenset[str]] = frozenset()
    #: Dependent vowel signs (matras); an added/dropped one is a "matra" error.
    vowel_signs: frozenset[str] = frozenset()
    #: Nasalisation marks; an added/dropped one is a "nasalisation" difference.
    nasal_marks: frozenset[str] = frozenset()
    #: Cluster-forming marks (virama); an added/dropped one is a "conjunct" error.
    joiners: frozenset[str] = frozenset()
    #: Vowel signs whose presence is a length contrast (Devanagari ा).
    length_marks: frozenset[str] = frozenset()


# -- the rule machinery ---------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A text rule: one entry in RULES.md, one function in normalize_xx.py."""

    id: str
    fn: Callable[[str, "Trace"], str]
    #: Neutral rules remove differences that are not in the speech at all.
    #: Everything else is forgiving-only.
    neutral: bool


@dataclass
class Trace:
    """Which rule changed what — how a disputed score shows its working."""

    steps: list[tuple[str, str, str]] = field(default_factory=list)
    #: Words that arrived in Latin script and were transliterated.
    romanised_words: list[str] = field(default_factory=list)

    def record(self, rule_id: str, before: str, after: str) -> None:
        if before != after:
            self.steps.append((rule_id, before, after))


def run_rules(text: str, rules: Iterable[Rule], *, neutral: bool) -> tuple[str, Trace]:
    trace = Trace()
    for rule in rules:
        if neutral and not rule.neutral:
            continue
        before = text
        text = rule.fn(text, trace)
        trace.record(rule.id, before, text)
    return text, trace


@dataclass(frozen=True)
class Forgiveness:
    """A forgiving rule that needs the *expected* word to decide.

    Most rules are context-free and run on each string alone. These are not:
    whether a dropped nasal is forgivable depends on which word was expected
    (गाँव -> गाव forgiven, चाँद -> चाद not). They run inside score.py, after
    alignment, where both words are known.
    """

    id: str
    #: (expected word, heard word, the character differences, their labels)
    #: -> True when the difference is not a reading mistake.
    applies: Callable[[str, str, tuple[tuple[str, str], ...], tuple[str, ...]], bool]


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    rules: tuple[Rule, ...]
    tables: ScriptTables
    forgiveness: tuple[Forgiveness, ...] = ()
    #: Hesitation sounds that are never words in a reading text.
    fillers: frozenset[str] = frozenset()
    #: For a letter task: spellings an engine uses for one spoken letter.
    letter_variants: Callable[[str], set[str]] = field(default=lambda letter: {letter})
    #: Rule ids cited when letter_variants or romanisation forgives something.
    letter_rule: str = ""
    romanised_rule: str = ""
    #: Word-list rules: exceptions to another rule, kept as data in normalize_xx.py.
    exception_rules: tuple[str, ...] = ()

    def normalize(self, text: str) -> tuple[str, Trace]:
        """The forgiving normaliser — for scoring children."""
        return run_rules(text, self.rules, neutral=False)

    def normalize_neutral(self, text: str) -> tuple[str, Trace]:
        """The neutral normaliser — for benchmarking a model."""
        return run_rules(text, self.rules, neutral=True)

    def rule_ids(self) -> set[str]:
        ids = {r.id for r in self.rules} | {f.id for f in self.forgiveness} | set(self.exception_rules)
        return ids | {r for r in (self.letter_rule, self.romanised_rule) if r}


def _load_hi() -> LanguageProfile:
    from .hi import normalize_hi

    return normalize_hi.PROFILE


def _load_mr() -> LanguageProfile:
    from .mr import normalize_mr

    return normalize_mr.PROFILE


_LOADERS = {"hi": _load_hi, "mr": _load_mr}

#: Accept the app's locale codes too (stt_eval uses hi-IN / mr-IN).
_ALIASES = {"hi-in": "hi", "mr-in": "mr", "hindi": "hi", "marathi": "mr"}


def supported() -> list[str]:
    return sorted(_LOADERS)


def get(code: str) -> LanguageProfile:
    key = _ALIASES.get(code.lower(), code.lower())
    if key not in _LOADERS:
        raise KeyError(f"No ReadNet language module for {code!r}; have {supported()}")
    return _LOADERS[key]()
