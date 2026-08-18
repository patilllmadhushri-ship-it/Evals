"""Audio normalisation.

Every clip is converted to mono 16-bit PCM WAV at one chosen sample rate before
it is sent to any provider, so accuracy differences reflect the provider and not
the encoding of the file the user happened to upload. The rate is a run-level
choice — 8 kHz to benchmark telephony audio at the rate it actually arrives at,
16 kHz for wideband capture.

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

from .config import DEFAULT_SAMPLE_RATE, TARGET_CHANNELS, TARGET_SAMPLE_WIDTH


class AudioError(RuntimeError):
    """Raised when a clip cannot be decoded by any available backend."""


@dataclass(frozen=True)
class NormalizedAudio:
    wav_bytes: bytes
    duration_seconds: float
    source_name: str
    sample_rate: int = DEFAULT_SAMPLE_RATE
    #: Rate of the file as uploaded, before normalisation — used to warn when a
    #: run upsamples narrowband audio and invents no new information.
    source_sample_rate: int | None = None

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    @property
    def upsampled(self) -> bool:
        return self.source_sample_rate is not None and self.source_sample_rate < self.sample_rate


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


def _decode_with_ffmpeg(raw: bytes, suffix: str, sample_rate: int) -> tuple[np.ndarray, int]:
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
                "-ar", str(sample_rate),
                "-f", "s16le", "-acodec", "pcm_s16le", "-",
            ],
            capture_output=True,
        )
        if completed.returncode != 0:
            raise AudioError(completed.stderr.decode("utf-8", "replace").strip()[:400])
        pcm = np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32) / 32768.0
        return pcm, sample_rate


def normalize(
    raw: bytes, filename: str, *, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> NormalizedAudio:
    """Decode `raw` and return mono 16-bit PCM WAV bytes at `sample_rate`.

    `sample_rate` is the run-level target — 8000 for telephony audio, 16000 for
    wideband. Every provider in the run receives the same rate, so the numbers
    stay comparable.
    """
    suffix = Path(filename).suffix.lower()
    errors: list[str] = []

    for decoder in (
        lambda: _decode_with_soundfile(raw),
        lambda: _decode_with_ffmpeg(raw, suffix, sample_rate),
    ):
        try:
            samples, decoded_rate = decoder()
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

    source_rate = decoded_rate
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = _resample(samples, decoded_rate, sample_rate)
    duration = float(samples.shape[0]) / sample_rate
    if duration <= 0:
        raise AudioError(f"{filename} decoded to zero audio frames.")

    return NormalizedAudio(
        wav_bytes=_to_wav_bytes(samples, sample_rate),
        duration_seconds=duration,
        source_name=filename,
        sample_rate=sample_rate,
        source_sample_rate=source_rate,
    )


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Read the duration of a PCM WAV payload without fully decoding it."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def wav_sample_rate(wav_bytes: bytes) -> int:
    """Sample rate declared in a WAV header.

    Providers that must be told the rate explicitly (Google) read it from the
    payload they are about to send, so they can never disagree with it.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        return handle.getframerate()


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
