"""The test bench's server, end to end over real HTTP. No browser, no keys.

    py -m readnet.tests.test_web

Starts the server on a free port, loads the page and its assets, then runs a
whole ASER session through the API with the "typed" engine: a child who fails
the paragraph, fails the words and passes the letters ends at Letter level.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request

from readnet.web.server import make_server


def start():
    import tempfile
    from pathlib import Path

    from readnet.web import server as web
    from readnet.web.records import Records

    web.RECORDS = Records(Path(tempfile.mkdtemp()) / "test.db")  # never the teacher's real records
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def get(base: str, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(base + path) as res:
        return res.status, res.read().decode("utf-8")


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as res:
            return json.loads(res.read())
    except urllib.error.HTTPError as error:
        return {"status": error.code, **json.loads(error.read())}


def test_web() -> None:
    server, base = start()
    try:
        for path in ("/", "/app.js", "/style.css"):
            status, _ = get(base, path)
            assert status == 200, path
        try:
            get(base, "/../server.py")
            raise AssertionError("served a file outside static/")
        except urllib.error.HTTPError as error:
            assert error.code == 404

        _, raw = get(base, "/api/config?language=hi")
        config = json.loads(raw)
        assert config["content"]["PARAGRAPH"] and {"typed", "mock", "local"} <= {e["key"] for e in config["engines"]}

        def read(level: str, heard: str) -> dict:
            return post(base, "/api/score", {"language": "hi", "level": level, "engine": "typed",
                                             "text": config["content"][level], "typed": heard})

        outcomes = {}
        para = read("PARAGRAPH", "मेरा नाम राजू है मेरे गर में गाय है गाय का रग है")
        assert not para["outcome"]["passed"] and para["ops"], para
        outcomes["PARAGRAPH"] = para["outcome"]
        assert post(base, "/api/next", {"outcomes": outcomes})["next"] == "WORD"

        outcomes["WORD"] = read("WORD", "घर पानी")["outcome"]
        assert post(base, "/api/next", {"outcomes": outcomes})["next"] == "LETTER"

        letters = read("LETTER", "का म स ल")
        assert letters["outcome"]["passed"] and any(op["rule"] == "HI-23" for op in letters["ops"])
        outcomes["LETTER"] = letters["outcome"]
        final = post(base, "/api/next", {"outcomes": outcomes})
        assert final["next"] is None and final["placement"]["label"] == "Letter", final

        quick = post(base, "/api/score", {"language": "hi", "level": "WORD", "engine": "typed",
                                          "text": "चाँद और गाँव", "typed": "चाद और गाव"})
        rules = {op["ref"]: (op["counts_as_mistake"], op["rule"]) for op in quick["ops"]}
        assert rules["गांव"] == (False, "HI-20") and rules["चांद"][0] is True, rules

        mock = post(base, "/api/score", {"language": "hi", "level": "PARAGRAPH", "engine": "mock",
                                         "text": config["content"]["PARAGRAPH"]})
        assert "outcome" in mock, mock
        missing = post(base, "/api/score", {"language": "hi", "level": "WORD", "engine": "sarvam", "text": "घर"})
        assert missing["status"] in (400, 502) and missing["error"], missing

        # Student records: the sounds a child keeps missing, across readings.
        student = post(base, "/api/students", {"name": "Asha", "grade": "3", "language": "hi"})
        sid = student["id"]
        assert post(base, "/api/students", {"name": " "})["status"] == 400
        text = "मेरा घर बड़ा है"
        for heard in ("मेरा गर बड़ा है", "मेरा गर बड़ा है", "मेरा घर बड़ा है"):
            result = post(base, "/api/score", {"language": "hi", "level": "PARAGRAPH", "engine": "typed",
                                               "text": text, "typed": heard})
            saved = post(base, "/api/attempts", {"student_id": sid, "language": "hi", "task": "PARAGRAPH",
                                                 "text": text, "result": result, "engine": "typed"})
            assert saved["sounds_saved"] > 0, saved
        post(base, "/api/sessions", {"student_id": sid, "language": "hi", "placement": final["placement"]})
        _, raw = get(base, f"/api/students/{sid}")
        detail = json.loads(raw)
        gha = next(s for s in detail["sounds"] if s["letter"] == "घ")
        assert (gha["tries"], gha["wrong"], gha["recent"], gha["last_ok"]) == (3, 2, "001", True), gha
        assert detail["words"][0]["word"] == "घर" and detail["sessions"][0]["level"] == "Letter"
        roster = json.loads(get(base, "/api/students")[1])["students"]
        assert roster[0]["practise"] == ["घ"] and roster[0]["readings"] == 3
        status, csv_text = get(base, f"/api/students/{sid}/export.csv")
        assert status == 200 and "घर" in csv_text and csv_text.count("\n") > 30
        assert post(base, "/api/attempts", {"student_id": 999, "language": "hi", "result": result})["status"] == 400
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_web()
    print("readnet.web: page, assets, full ASER session over the API, quick check, mock, errors")
    sys.exit(0)
