"""Assemble the free online version: a static site that runs ReadNet in the browser.

    py readnet/deploy/build_static.py <out_dir>                 # for the Hugging Face static Space
    py readnet/deploy/build_static.py <out_dir> --local <onnx>   # to test on this computer

The site is the normal page plus static/offline.js, which answers the page's
/api/ calls in the browser: the scoring code (readnet/, zipped) runs under
Pyodide and the Hindi model runs with onnxruntime-web. Only Python sources the
browser needs are zipped — never server.py, deploy/, tests or anything outside
readnet/, so no key or recording can be published.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "readnet" / "web" / "static"
SKIP = {("readnet", "web", "server.py")}
SKIP_DIRS = {"deploy", "tests", "__pycache__", "_template"}

SPACE_README = """---
title: ReadNet
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Record a child reading Hindi, get the ASER level and mistakes
---

# ReadNet

Record a child reading Hindi aloud and get their ASER reading level, every
word marked right or wrong, and the letters and sounds they got wrong.

Everything runs in your browser: the scoring code runs under Pyodide and the
Hindi speech model (Vakyansh wav2vec2 by Harveen Chadha, MIT licence,
converted to ONNX) runs with onnxruntime-web. No account or key; the
recording never leaves your computer. The first visit downloads the model
once (about 380 MB).

Source: https://github.com/patilllmadhushri-ship-it/Evals (`readnet/`).
"""


def build(out: Path, local_model: Path | None = None) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("app.js", "style.css", "offline.js"):
        shutil.copy2(STATIC / name, out / name)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    tag = '<script src="/app.js"></script>'
    assert tag in html
    html = html.replace('href="/style.css"', 'href="style.css"')
    html = html.replace(tag, '<script src="config.js"></script>\n  <script src="offline.js"></script>\n  <script src="app.js"></script>')
    (out / "index.html").write_text(html, encoding="utf-8")

    with zipfile.ZipFile(out / "readnet.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for src in sorted((ROOT / "readnet").rglob("*.py")):
            rel = src.relative_to(ROOT)
            if rel.parts in SKIP or SKIP_DIRS & set(rel.parts):
                continue
            z.write(src, rel.as_posix())

    config = {}
    if local_model:
        (out / "models").mkdir()
        shutil.copy2(local_model, out / "models" / "hi.onnx")
        shutil.copy2(local_model.with_name("vocab_hi.json"), out / "models" / "vocab_hi.json")
        config = {"modelUrl": "models/hi.onnx", "vocabUrl": "models/vocab_hi.json"}
    (out / "config.js").write_text(f"window.READNET_CONFIG = {json.dumps(config)};\n", encoding="utf-8")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--local", type=Path, help="an ONNX model file to serve from the site, for testing")
    args = parser.parse_args()
    build(Path(args.out), args.local)
    print(f"built {args.out}")
