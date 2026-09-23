"""Assemble the Hugging Face Space folder: only the code the app needs.

    py readnet/deploy/build_space.py <out_dir>

Copies readnet/ and stt_eval/ (Python sources and the page), the Dockerfile
and requirements. Nothing else from the repository is copied, so `.env`,
run results and recordings can never be published.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KEEP = {".py", ".html", ".js", ".css", ".csv", ".md"}


def build(out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copied: list[Path] = []
    for package in ("readnet", "stt_eval"):
        for src in (ROOT / package).rglob("*"):
            rel = src.relative_to(ROOT)
            if not src.is_file() or src.suffix not in KEEP or "__pycache__" in rel.parts or "deploy" in rel.parts:
                continue
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel)
    shutil.copy2(HERE / "Dockerfile", out / "Dockerfile")
    shutil.copy2(HERE / "requirements.txt", out / "requirements.txt")
    shutil.copy2(HERE / "SPACE_README.md", out / "README.md")
    names = {p.name.lower() for p in out.rglob("*")}
    assert not {".env", "secrets.toml"} & names, "a secrets file got into the bundle"
    return copied


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "space_build")
    files = build(target)
    print(f"{len(files)} source files + Dockerfile, requirements.txt, README.md -> {target}")
