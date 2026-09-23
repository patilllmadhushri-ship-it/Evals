"""Language registry.

Each language lives in its own folder with the four artifacts the blueprint
asks for: ``RULES.md`` (plain English, for educators and linguists),
``normalize_xx.py`` (the executable rules), and shared ``score.py`` and
``tests/test_cases.py`` at package level. Adding a language is a new folder and
one entry in ``_LOADERS`` — the scorer and the ASER logic do not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .devanagari import Trace


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    normalize: Callable[[str], tuple[str, Trace]]
    #: Hesitation sounds that are never words in a reading text.
    fillers: frozenset[str] = frozenset()
    #: Substitution differences forgiven outright, as unordered character pairs
    #: (Hindi speakers do not distinguish श and ष, and neither do ASR spellings).
    forgiven_pairs: frozenset[frozenset[str]] = frozenset()
    #: For a letter task: spellings an ASR engine uses for one spoken letter.
    letter_variants: Callable[[str], set[str]] = field(default=lambda letter: {letter})


def _load_hi() -> LanguageProfile:
    from .hi import normalize_hi

    return normalize_hi.PROFILE


def _load_mr() -> LanguageProfile:
    from .mr import normalize_mr

    return normalize_mr.PROFILE


_LOADERS = {"hi": _load_hi, "mr": _load_mr}

#: Accept the app's locale codes too (stt_eval uses hi-IN / mr-IN).
_ALIASES = {"hi-IN": "hi", "mr-IN": "mr", "hindi": "hi", "marathi": "mr"}


def supported() -> list[str]:
    return sorted(_LOADERS)


def get(code: str) -> LanguageProfile:
    key = _ALIASES.get(code, _ALIASES.get(code.lower(), code.lower()))
    if key not in _LOADERS:
        raise KeyError(f"No ReadNet language module for {code!r}; have {supported()}")
    return _LOADERS[key]()
