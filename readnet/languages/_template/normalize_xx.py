"""<Language> rulebook, executable copy — TEMPLATE.

One function per rule in RULES.md, named after the rule id. Deterministic; no
model calls. Replace `xx`/`XX` with the language code, fill the script tables,
and register the profile in languages/__init__.py.
"""

from __future__ import annotations

import unicodedata

from .. import Forgiveness, LanguageProfile, Rule, ScriptTables, Trace


def xx_01_unicode_nfc(text: str, trace: Trace) -> str:
    return unicodedata.normalize("NFC", text)


def xx_04_strip_invisible_and_punctuation(text: str, trace: Trace) -> str:
    text = "".join(ch for ch in text if ch not in {"\u200c", "\u200d", "\u200b", "\ufeff"})
    text = "".join(ch if ch.isalnum() or unicodedata.category(ch).startswith("M") else " " for ch in text)
    return " ".join(text.split())


RULES = (
    Rule("XX-01", xx_01_unicode_nfc, neutral=True),
    Rule("XX-04", xx_04_strip_invisible_and_punctuation, neutral=True),
)

#: Fill from the script: phonetic pairs by label, look-alike pairs, vowel signs,
#: nasal marks, cluster joiners. Then check them with `readnet confusions`.
TABLES = ScriptTables()

PROFILE = LanguageProfile(
    code="xx",
    name="<Language>",
    rules=RULES,
    tables=TABLES,
    forgiveness=(),  # Forgiveness("XX-20", fn) for rules that need the expected word
)
