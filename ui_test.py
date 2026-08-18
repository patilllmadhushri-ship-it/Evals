"""Headless render check for every step of the Streamlit flow.

    py ui_test.py

Uses Streamlit's AppTest harness to run app.py without a browser, seeding
session state so each step renders against a real dataset and real stored
results (mock provider, deterministic metrics only).
"""

from __future__ import annotations

import sys

from streamlit.testing.v1 import AppTest

from smoke_test import GROUND_TRUTH, make_wav
from stt_eval.dataset import build_dataset
from stt_eval.runner import RunConfig, Runner
from stt_eval.store import ResultStore

RUN_ID = "uitest"


def seed_results():
    """Produce a small completed run in the app's own store."""
    store = ResultStore()
    store.delete_run(RUN_ID)
    uploads = [(f"{clip_id}.wav", make_wav(1.0)) for clip_id in GROUND_TRUTH]
    dataset = build_dataset(uploads, GROUND_TRUTH)
    runner = Runner(
        config=RunConfig(
            run_id=RUN_ID,
            language="en-IN",
            provider_keys=["mock"],
            api_keys={"mock": ""},
            enabled_metrics=["wer", "cer"],
        ),
        dataset=dataset,
        store=store,
        judge=None,
    )
    runner.start()
    runner._thread.join(timeout=60)  # noqa: SLF001 - test-only join
    return dataset


def main() -> int:
    dataset = seed_results()
    failures = 0

    for step, name in enumerate(
        ["upload", "configure", "credentials", "run", "review", "export"]
    ):
        app = AppTest.from_file("app.py", default_timeout=60)
        app.session_state["step"] = step
        app.session_state["run_id"] = RUN_ID
        app.session_state["dataset"] = dataset
        app.session_state["language"] = "en-IN"
        app.session_state["selected_providers"] = ["mock"]
        app.session_state["enabled_metrics"] = ["wer", "cer"]
        app.session_state["api_keys"] = {"mock": ""}
        app.run()

        if app.exception:
            failures += 1
            print(f"  FAIL  step {step} ({name}): {app.exception[0].value}")
        else:
            headers = [item.value for item in app.header] or ["(no header)"]
            print(f"  PASS  step {step} ({name}) — {headers[0]}")

    ResultStore().delete_run(RUN_ID)
    print("\nAll steps rendered." if not failures else f"\n{failures} step(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
