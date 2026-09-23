"""Student records: who was assessed, what level, and which sounds they miss.

A teacher needs more than one reading's verdict: "this child cannot yet say
झ" only shows up across readings. Every accepted reading is broken into one
row per letter/sound attempted, marked said-right or said-wrong, so the
student page can rank the sounds to practise.

Where a row's verdict comes from, strongest evidence first:

* ``letter_check`` — a letter task decided from the audio, letter by letter;
* ``gop`` — each aligned sound's GOP against the threshold in use;
* ``text`` — no audio model: the letters of a word the transcript shows
  misread (from the same alignment the scorer uses) are wrong, the rest right.

Stored in SQLite under .stt_eval_runs/ (git-ignored): names, transcripts,
scores. Never audio — the same choice PadhAI made for children's safety.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from .. import g2p, languages
from ..score import classify_substitution

DB_PATH = Path(".stt_eval_runs") / "readnet.db"
#: Marks with no sound of their own: never tracked as a "letter".
SILENT = {"\u094d", "\u093c"}  # virama, nukta

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, grade TEXT, language TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY, student_id INTEGER NOT NULL REFERENCES students(id), created_at TEXT NOT NULL,
    language TEXT, task TEXT, text TEXT, transcript TEXT, engine TEXT,
    passed INTEGER, mistakes INTEGER, correct INTEGER, total INTEGER
);
CREATE TABLE IF NOT EXISTS sound_events (
    id INTEGER PRIMARY KEY, attempt_id INTEGER NOT NULL REFERENCES attempts(id),
    student_id INTEGER NOT NULL, created_at TEXT NOT NULL,
    word TEXT, letter TEXT, sound TEXT, gop REAL, ok INTEGER NOT NULL, heard TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY, student_id INTEGER NOT NULL REFERENCES students(id), created_at TEXT NOT NULL,
    language TEXT, level TEXT, level_index INTEGER, path TEXT
);
CREATE INDEX IF NOT EXISTS sound_events_student ON sound_events(student_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Records:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _write(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid

    # -- students ------------------------------------------------------------------

    def add_student(self, name: str, grade: str = "", language: str = "") -> dict:
        name = name.strip()
        if not name:
            raise ValueError("A student needs a name or ID.")
        sid = self._write("INSERT INTO students (name, grade, language, created_at) VALUES (?, ?, ?, ?)",
                          (name, grade.strip(), language, _now()))
        return {"id": sid, "name": name, "grade": grade.strip(), "language": language}

    def students(self) -> list[dict]:
        rows = self._rows("""
            SELECT s.*,
              (SELECT level FROM sessions WHERE student_id = s.id ORDER BY id DESC LIMIT 1) AS latest_level,
              (SELECT created_at FROM attempts WHERE student_id = s.id ORDER BY id DESC LIMIT 1) AS last_seen,
              (SELECT COUNT(*) FROM sessions WHERE student_id = s.id) AS sessions,
              (SELECT COUNT(*) FROM attempts WHERE student_id = s.id) AS readings
            FROM students s ORDER BY s.name COLLATE NOCASE""")
        for row in rows:
            row["practise"] = [p["letter"] for p in self.sounds_to_practise(row["id"], limit=4)]
        return rows

    def student(self, sid: int) -> dict:
        found = self._rows("SELECT * FROM students WHERE id = ?", (sid,))
        if not found:
            raise KeyError(sid)
        return found[0]

    # -- saving --------------------------------------------------------------------

    def save_attempt(self, sid: int, *, language: str, task: str, text: str, result: dict,
                     engine: str, threshold: float) -> dict:
        self.student(sid)
        o = result["outcome"]
        now = _now()
        aid = self._write(
            """INSERT INTO attempts (student_id, created_at, language, task, text, transcript, engine,
                                     passed, mistakes, correct, total) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, now, language, task, text, result.get("engine_transcript") or result.get("transcript", ""),
             engine, int(bool(o["passed"])), o["mistakes"], o["correct"], o["total"]))
        events = list(sound_events(result, language, threshold))
        with self._lock:
            self._db.executemany(
                """INSERT INTO sound_events (attempt_id, student_id, created_at, word, letter, sound, gop, ok,
                                             heard, source) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [(aid, sid, now, e["word"], e["letter"], e["sound"], e["gop"], int(e["ok"]), e["heard"],
                  e["source"]) for e in events])
            self._db.commit()
        return {"attempt_id": aid, "sounds_saved": len(events),
                "sounds_wrong": sum(1 for e in events if not e["ok"])}

    def save_session(self, sid: int, *, language: str, level: str, level_index: int, path: list[str]) -> int:
        self.student(sid)
        return self._write(
            "INSERT INTO sessions (student_id, created_at, language, level, level_index, path) VALUES (?,?,?,?,?,?)",
            (sid, _now(), language, level, level_index, json.dumps(path, ensure_ascii=False)))

    # -- reading back ---------------------------------------------------------------

    def sounds_to_practise(self, sid: int, *, min_tries: int = 1, limit: int | None = None) -> list[dict]:
        """Letters ranked by how often this student gets them wrong."""
        rows = self._rows("""
            SELECT letter, GROUP_CONCAT(DISTINCT sound) AS sound, COUNT(*) AS tries, SUM(1 - ok) AS wrong,
                   MAX(created_at) AS last_seen, ROUND(AVG(gop), 1) AS avg_gop
            FROM sound_events WHERE student_id = ? AND letter != ''
            GROUP BY letter HAVING COUNT(*) >= ? AND SUM(1 - ok) > 0
            ORDER BY (1.0 * SUM(1 - ok) / COUNT(*)) DESC, SUM(1 - ok) DESC, letter""", (sid, min_tries))
        for r in rows:
            history = [h["ok"] for h in self._rows(
                "SELECT ok FROM sound_events WHERE student_id = ? AND letter = ? ORDER BY id", (sid, r["letter"]))]
            r["rate"] = round(r["wrong"] / r["tries"], 2)
            r["recent"] = "".join(str(h) for h in history[-5:])  # oldest to newest, 1 = said right
            r["last_ok"] = bool(history[-1])
            r["words"] = [w["word"] for w in self._rows(
                "SELECT DISTINCT word FROM sound_events WHERE student_id = ? AND letter = ? AND ok = 0 LIMIT 6",
                (sid, r["letter"]))]
        return rows[:limit] if limit else rows

    def words_to_practise(self, sid: int, limit: int = 15) -> list[dict]:
        return self._rows("""
            SELECT word, COUNT(DISTINCT attempt_id) AS readings, SUM(1 - ok) AS sounds_wrong,
                   GROUP_CONCAT(DISTINCT CASE WHEN ok = 0 THEN letter END) AS letters
            FROM sound_events WHERE student_id = ? GROUP BY word HAVING SUM(1 - ok) > 0
            ORDER BY SUM(1 - ok) DESC, word LIMIT ?""", (sid, limit))

    def detail(self, sid: int) -> dict:
        sessions = self._rows("SELECT * FROM sessions WHERE student_id = ? ORDER BY id", (sid,))
        for s in sessions:
            s["path"] = json.loads(s["path"] or "[]")
        return {
            "student": self.student(sid),
            "sessions": sessions,
            "attempts": self._rows("SELECT * FROM attempts WHERE student_id = ? ORDER BY id DESC LIMIT 50", (sid,)),
            "sounds": self.sounds_to_practise(sid),
            "words": self.words_to_practise(sid),
            "sounds_tracked": self._rows(
                "SELECT COUNT(*) AS n, SUM(1 - ok) AS wrong FROM sound_events WHERE student_id = ?", (sid,))[0],
        }

    def export_csv(self, sid: int) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["date", "task", "word", "letter", "sound", "said_right", "gop", "audio_heard", "source"])
        for r in self._rows("""
                SELECT e.created_at, a.task, e.word, e.letter, e.sound, e.ok, e.gop, e.heard, e.source
                FROM sound_events e JOIN attempts a ON a.id = e.attempt_id
                WHERE e.student_id = ? ORDER BY e.id""", (sid,)):
            writer.writerow([r["created_at"], r["task"], r["word"], r["letter"], r["sound"],
                             "yes" if r["ok"] else "no", r["gop"], r["heard"], r["source"]])
        return buf.getvalue()


def sound_events(result: dict, language: str, threshold: float):
    """One row per letter/sound the reader attempted in a scored reading."""
    if result.get("letter_check"):
        for c in result["letter_check"]:
            yield {"word": c["letter"], "letter": c["letter"], "sound": " ".join(g2p.word_to_phonemes(c["letter"])),
                   "gop": c["margin"], "ok": c["heard"] == c["letter"], "heard": c["heard"], "source": "letter_check"}
        return
    tables = languages.get(language).tables
    for op in result.get("ops", []):
        word = op.get("ref")
        if not word or op.get("kind") == "deletion":
            continue  # a skipped word says nothing about which sounds the child can make
        if op.get("units"):
            for u in op["units"]:
                if u.get("sounds") and u.get("letter") not in SILENT:
                    yield {"word": word, "letter": u.get("letter") or u["sounds"], "sound": u["sounds"],
                           "gop": u["gop"], "ok": u["gop"] >= threshold, "heard": u.get("heard", ""),
                           "source": "gop"}
            continue
        sounds = g2p.letter_sounds(word)
        wrong: set[int] = set()
        if op.get("kind") == "substitution" and op.get("counts_as_mistake"):
            hyp = op.get("hyp") or ""
            diffs = classify_substitution(word, hyp, tables).char_diffs
            wrong_letters = {r for r, _ in diffs if r}
            wrong = {i for i, ch in enumerate(word) if ch in wrong_letters}
        for i, ch in enumerate(word):
            if ch in SILENT or not sounds[i]:
                continue
            yield {"word": word, "letter": ch, "sound": sounds[i], "gop": None, "ok": i not in wrong,
                   "heard": "", "source": "text"}
