"""Downloadable results: CSV and Excel, per-clip rows plus an aggregate summary."""

from __future__ import annotations

import csv
import io

from .config import RATE_TABLE_NOTE


def per_clip_csv(rows: list[dict]) -> bytes:
    return _rows_to_csv(rows)


def summary_csv(rows: list[dict]) -> bytes:
    return _rows_to_csv(rows)


def _rows_to_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def workbook(
    per_clip: list[dict], summary: list[dict], *, metadata: dict | None = None
) -> bytes:
    """One .xlsx with a summary sheet, a per-clip sheet, and the run's metadata."""
    import pandas as pd

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Leaderboard", index=False)
        pd.DataFrame(per_clip).to_excel(writer, sheet_name="Per clip", index=False)
        info = list((metadata or {}).items()) + [("cost_note", RATE_TABLE_NOTE)]
        pd.DataFrame(info, columns=["field", "value"]).to_excel(
            writer, sheet_name="Run info", index=False
        )
    return buffer.getvalue()
