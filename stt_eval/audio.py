"""Audio normalisation.

Every clip is converted to mono 16 kHz 16-bit PCM WAV before it is sent to any
provider, so accuracy differences reflect the provider and not the encoding of
the file the user happened to upload.

Two decoders are tried in order: libsndfile (via ``soundfile``), then ``ffmpeg``
if it is on PATH. WAV/FLAC/OGG are handled by the first; MP3/M4A/WebM usually
need one or the other depending on the installed libsndfile version.
"""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TARGET_CHANNELS, TARGET_SAMPLE_RATE, TARGET_SAMPLE_WIDTH


class AudioError(RuntimeError):
    """Raised when a clip cannot be decoded by any available backend."""


@dataclass(frozen=True)
class NormalizedAudio:
    wav_bytes: bytes
    duration_seconds: float
    source_name: str

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Pack float or int samples into a 16-bit PCM WAV container."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.dtype.kind == "f":
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 1.0:
            samples = samples / peak
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)
    else:
        samples = samples.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(TARGET_CHANNELS)
        handle.setsampwidth(TARGET_SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample. Adequate for STT input; not a mastering-grade filter."""
    if source_rate == target_rate:
        return samples
    duration = samples.shape[0] / float(source_rate)
    target_length = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _decode_with_soundfile(raw: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf  # imported lazily so ffmpeg-only setups still work

    data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def _decode_with_ffmpeg(raw: bytes, suffix: str) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioError("ffmpeg is not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / f"input{suffix or '.bin'}"
        source.write_bytes(raw)
        completed = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-i", str(source),
                "-ac", str(TARGET_CHANNELS),
                "-ar", str(TARGET_SAMPLE_RATE),
                "-f", "s16le", "-acodec", "pcm_s16le", "-",
            ],
            capture_output=True,
        )
        if completed.returncode != 0:
            raise AudioError(completed.stderr.decode("utf-8", "replace").strip()[:400])
        pcm = np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32) / 32768.0
        return pcm, TARGET_SAMPLE_RATE


def normalize(raw: bytes, filename: str) -> NormalizedAudio:
    """Decode `raw` and return mono 16 kHz 16-bit PCM WAV bytes plus duration."""
    suffix = Path(filename).suffix.lower()
    errors: list[str] = []

    for decoder in (
        lambda: _decode_with_soundfile(raw),
        lambda: _decode_with_ffmpeg(raw, suffix),
    ):
        try:
            samples, sample_rate = decoder()
            break
        except AudioError as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - decoder backends raise broadly
            errors.append(f"{type(exc).__name__}: {exc}")
    else:
        raise AudioError(
            f"Could not decode {filename}. Tried libsndfile and ffmpeg: "
            + " | ".join(errors)
        )

    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = _resample(samples, sample_rate, TARGET_SAMPLE_RATE)
    duration = float(samples.shape[0]) / TARGET_SAMPLE_RATE
    if duration <= 0:
        raise AudioError(f"{filename} decoded to zero audio frames.")

    return NormalizedAudio(
        wav_bytes=_to_wav_bytes(samples, TARGET_SAMPLE_RATE),
        duration_seconds=duration,
        source_name=filename,
    )


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Read the duration of a PCM WAV payload without fully decoding it."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def pcm_peak_dbfs(wav_bytes: bytes) -> float:
    """Peak level in dBFS — used to warn about silent or clipped uploads."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    if not frames:
        return -float("inf")
    count = len(frames) // 2
    values = struct.unpack(f"<{count}h", frames[: count * 2])
    peak = max(abs(v) for v in values) if values else 0
    if peak == 0:
        return -float("inf")
    return 20.0 * float(np.log10(peak / 32768.0))
