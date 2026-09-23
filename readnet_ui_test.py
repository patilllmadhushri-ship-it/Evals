"""Headless walk-through of readnet_app.py: a full ASER session, no browser.

    py readnet_ui_test.py

Drives the real app with Streamlit's AppTest harness, using the "type what
the child said" engine so no key, microphone or network is needed: a child
who fails the paragraph, fails the words and passes the letters must end at
Letter level, with each step rendered on the way.
"""

from __future__ import annotations

import sys

from streamlit.testing.v1 import AppTest

TYPED = "__typed__"


def fresh() -> AppTest:
    app = AppTest.from_file("readnet_app.py", default_timeout=60)
    app.run()
    app.selectbox(key="engine").set_value(TYPED).run()
    return app


def texts(app: AppTest) -> str:
    return " ".join(m.value for m in app.markdown) + " ".join(str(m.value) for m in app.metric)


def read(app: AppTest, heard: str, level_name: str) -> None:
    key = next(k for k in (t.key for t in app.text_area) if k and k.startswith("typed_") and k.endswith(level_name))
    app.text_area(key=key).set_value(heard).run()
    next(b for b in app.button if b.key and b.key.startswith("go_")).click().run()
    assert not app.exception, app.exception
    next(b for b in app.button if b.key and b.key.startswith("ok_")).click().run()
    assert not app.exception, app.exception


def main() -> int:
    app = fresh()
    assert not app.exception, app.exception
    assert "Paragraph" in texts(app)

    # Paragraph: skips and misreads five words -> fails -> Words.
    read(app, "मेरा नाम राजू है मेरे गर में गाय है गाय का रग है", "PARAGRAPH")
    assert "Words" in texts(app), "did not move to Words after failing the paragraph"
    # Words: 2 of 5 -> fails -> Letters.
    read(app, "घर पानी", "WORD")
    assert "Letters" in texts(app), "did not move to Letters"
    # Letters: 4 of 5, with a spoken-letter spelling (का for क) -> passes.
    read(app, "का म स ल", "LETTER")
    page = texts(app)
    assert "Letter" in page and "Reading level" in page, page[:500]
    assert any("Knows letters" in i.value for i in app.info), "no teaching next step"

    # Quick check: the context-dependent nasal rule, visible in the UI.
    quick = fresh()
    quick.button(key="q_go").click().run()
    assert not quick.exception, quick.exception
    page = texts(quick)
    assert "HI-20" in page and "heard चाद" in page, page[:800]

    print("readnet_app: full session (Paragraph ✗ → Words ✗ → Letters ✓ = Letter) and quick check render")
    return 0


if __name__ == "__main__":
    sys.exit(main())
