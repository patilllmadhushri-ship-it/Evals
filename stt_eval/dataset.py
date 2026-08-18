"""Dataset assembly and validation.

The user uploads audio plus ground truth (a CSV of ``id,text`` or inline text
boxes). This module matches the two by id and produces specific, actionable
errors — "clip3.wav has no matching row in ground_truth.csv" — rather than a
generic failure at run time.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from .audio import AudioError, NormalizedAudio, normalize
from .config import DEFAULT_SAMPLE_RATE


@dataclass
class Clip:
    clip_id: str
    filename: str
    ground_truth: str
    audio: NormalizedAudio

    @property
    def duration_seconds(self) -> float:
        return self.audio.duration_seconds


@dataclass
class Dataset:
    clips: list[Clip] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The rate every clip was normalised to for this run.
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def is_runnable(self) -> bool:
        return bool(self.clips) and not self.errors

    @property
    def total_duration_seconds(self) -> float:
        return sum(clip.duration_seconds for clip in self.clips)

    @property
    def total_duration_minutes(self) -> float:
        return self.total_duration_seconds / 60.0

    def by_id(self, clip_id: str) -> Clip | None:
        return next((clip for clip in self.clips if clip.clip_id == clip_id), None)


def clip_id_for(filename: str) -> str:
    """`clip1.wav` -> `clip1`. Ground-truth ids match the stem, not the extension."""
    return Path(filename).stem.strip()


def parse_ground_truth_csv(raw: bytes, source_name: str = "ground_truth.csv") -> tuple[dict[str, str], list[str]]:
    """Parse an ``id,text`` CSV. Returns (mapping, errors)."""
    errors: list[str] = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        errors.append(f"{source_name} is not valid UTF-8; decoded as latin-1.")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return {}, [f"{source_name} is empty."]

    normalized_fields = {(name or "").strip().lower(): name for name in reader.fieldnames}
    id_field = normalized_fields.get("id")
    text_field = normalized_fields.get("text") or normalized_fields.get("transcript")
    if id_field is None or text_field is None:
        return {}, [
            f"{source_name} must have 'id' and 'text' columns "
            f"(found: {', '.join(reader.fieldnames)})."
        ]

    mapping: dict[str, str] = {}
    for line_number, row in enumerate(reader, start=2):
        row_id = (row.get(id_field) or "").strip()
        row_text = (row.get(text_field) or "").strip()
        if not row_id:
            errors.append(f"{source_name} line {line_number}: empty id — row skipped.")
            continue
        row_id = clip_id_for(row_id)
        if row_id in mapping:
            errors.append(f"{source_name} line {line_number}: duplicate id '{row_id}'.")
            continue
        if not row_text:
            errors.append(f"{source_name} line {line_number}: '{row_id}' has empty ground-truth text.")
            continue
        mapping[row_id] = row_text
    return mapping, errors


def build_dataset(
    uploads: list[tuple[str, bytes]],
    ground_truth: dict[str, str],
    *,
    ground_truth_source: str = "ground_truth.csv",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Dataset:
    """Decode every upload, match it to its ground truth, and report both directions.

    `uploads` is a list of (filename, raw bytes). `ground_truth` maps clip id to
    the correct transcript. `sample_rate` is the run-level normalisation target
    (8000 for telephony audio, 16000 for wideband).
    """
    dataset = Dataset(sample_rate=sample_rate)
    seen_ids: set[str] = set()
    upsampled: list[str] = []

    for filename, raw in uploads:
        clip_id = clip_id_for(filename)
        if clip_id in seen_ids:
            dataset.errors.append(
                f"{filename}: duplicate clip id '{clip_id}' — ids must be unique across uploads."
            )
            continue
        seen_ids.add(clip_id)

        expected = ground_truth.get(clip_id)
        if expected is None:
            dataset.errors.append(
                f"{filename} has no matching row in {ground_truth_source} "
                f"(looked for id '{clip_id}')."
            )
            continue

        try:
            audio = normalize(raw, filename, sample_rate=sample_rate)
        except AudioError as exc:
            dataset.errors.append(f"{filename}: {exc}")
            continue

        if audio.upsampled:
            upsampled.append(f"{filename} ({audio.source_sample_rate} Hz)")

        if audio.duration_seconds < 0.2:
            dataset.warnings.append(
                f"{filename} is only {audio.duration_seconds:.2f}s long — "
                "most providers return an empty transcript for clips this short."
            )

        dataset.clips.append(
            Clip(clip_id=clip_id, filename=filename, ground_truth=expected, audio=audio)
        )

    for row_id in ground_truth:
        if row_id not in seen_ids:
            dataset.errors.append(
                f"{ground_truth_source} has a row for '{row_id}' but no audio file "
                f"named '{row_id}.<ext>' was uploaded."
            )

    if upsampled:
        dataset.warnings.append(
            f"Upsampled to {sample_rate} Hz from a lower rate: {', '.join(upsampled[:5])}"
            + (f" and {len(upsampled) - 5} more" if len(upsampled) > 5 else "")
            + ". Upsampling adds no information — if this is telephony audio, "
            "run at 8 kHz to measure what your providers actually see in production."
        )

    if not uploads:
        dataset.errors.append("No audio files uploaded.")
    elif not ground_truth:
        dataset.errors.append("No ground-truth transcripts supplied.")

    dataset.clips.sort(key=lambda clip: clip.clip_id)
    return dataset
