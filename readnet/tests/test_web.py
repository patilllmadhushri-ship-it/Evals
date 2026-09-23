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

        # Each result says which letters the child got wrong, and nothing is stored.
        misread = post(base, "/api/score", {"language": "hi", "level": "PARAGRAPH", "engine": "typed",
                                            "text": "मेरा घर बड़ा है", "typed": "मेरा गर बड़ा है"})
        assert [(w["letter"], w["word"]) for w in misread["wrong_letters"]] == [("घ", "घर")], misread["wrong_letters"]
        forgiven = post(base, "/api/score", {"language": "hi", "level": "WORD", "engine": "typed",
                                             "text": "गाँव", "typed": "गाव"})
        assert forgiven["wrong_letters"] == []  # HI-20: not held against the child
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_web()
    print("readnet.web: page, assets, full ASER session over the API, quick check, mock, errors")
    sys.exit(0)
