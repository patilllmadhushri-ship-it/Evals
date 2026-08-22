"""Streaming latency measurement.

A batch HTTP transcription can only tell you how long the whole request took.
The metrics that decide whether an STT engine is usable in a voice agent are
about *when information arrives while the person is still talking*, and those
only exist over a streaming connection:

* **Partial emission latency** — how long after speech is spoken does an
  incremental (interim) transcript update carrying it arrive.
* **Finals latency** — how long after a phrase ends does the engine commit it
  as final. This is what gates a voice agent's turn-taking: the agent cannot
  respond until the transcript is final.
* **Real-time factor (RTF)** — processing time over audio duration. Below 1.0
  means the engine consumes speech faster than it arrives, which it must, or
  it falls progressively further behind during a long utterance.

The technique here needs no browser and no live microphone: **replay recorded
audio over the provider's WebSocket at real-time pace, and timestamp every
event as it arrives.** Because each result carries the audio-time span it
covers, the latency of a result is

    arrival_offset - audio_end_time_of_that_result

which is exactly the delay a live speaker would have experienced. Pacing the
send at 1x real time is what makes the number meaningful; blasting the file as
fast as possible measures throughput (RTF), not latency, so the two modes are
separate and explicit.
"""

from __future__ import annotations

import asyncio
import io
import json
import statistics
import time
import wave
from dataclasses import dataclass, field
from typing import Literal

PacingMode = Literal["realtime", "fast"]

#: Audio sent per WebSocket frame. 100 ms is small enough that pacing is smooth
#: and large enough to avoid drowning the connection in tiny writes.
FRAME_MILLISECONDS = 100


class StreamingUnsupported(RuntimeError):
    """The provider has no streaming endpoint wired up here."""


class StreamingError(RuntimeError):
    """The streaming attempt failed. Reported per provider, never fatal."""


@dataclass
class StreamEvent:
    """One result from the engine, with when it arrived and what it covered."""

    arrival_offset: float  # seconds since the first audio frame was sent
    audio_start: float  # start of the audio span this result covers
    audio_end: float  # end of that span
    text: str
    is_final: bool

    @property
    def latency(self) -> float:
        """Delay between the speech being spoken and this result arriving."""
        return max(0.0, self.arrival_offset - self.audio_end)


@dataclass
class StreamingMetrics:
    provider: str
    audio_seconds: float
    wall_seconds: float
    events: list[StreamEvent] = field(default_factory=list)
    transcript: str = ""
    pacing: PacingMode = "realtime"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def partials(self) -> list[StreamEvent]:
        return [event for event in self.events if not event.is_final]

    @property
    def finals(self) -> list[StreamEvent]:
        return [event for event in self.events if event.is_final]

    @property
    def time_to_first_partial(self) -> float | None:
        """How long before the engine says anything at all."""
        return self.partials[0].arrival_offset if self.partials else None

    @property
    def partial_emission_latency(self) -> float | None:
        """Median delay of incremental updates behind the speech they carry."""
        values = [event.latency for event in self.partials]
        return statistics.median(values) if values else None

    @property
    def finals_latency(self) -> float | None:
        """Median delay between a phrase ending and it being locked in."""
        values = [event.latency for event in self.finals]
        return statistics.median(values) if values else None

    @property
    def finals_latency_p95(self) -> float | None:
        values = sorted(event.latency for event in self.finals)
        if not values:
            return None
        index = max(0, min(len(values) - 1, int(round(0.95 * len(values) + 0.5)) - 1))
        return values[index]

    @property
    def rtf(self) -> float | None:
        """Processing time over audio duration. Only meaningful when sending fast.

        Under real-time pacing the wall clock is dominated by the pacing itself,
        so an RTF computed from it would always be ~1.0 and would say nothing
        about the engine.
        """
        if self.audio_seconds <= 0 or self.pacing != "fast":
            return None
        return self.wall_seconds / self.audio_seconds

    def summary(self) -> dict[str, float | None]:
        return {
            "time_to_first_partial_s": self.time_to_first_partial,
            "partial_emission_latency_s": self.partial_emission_latency,
            "finals_latency_s": self.finals_latency,
            "finals_latency_p95_s": self.finals_latency_p95,
            "rtf": self.rtf,
            "partials": len(self.partials),
            "finals": len(self.finals),
        }


def _frames(wav_bytes: bytes, frame_ms: int) -> tuple[list[bytes], int, float]:
    """Split PCM WAV into fixed-duration frames. Returns (frames, rate, seconds)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        pcm = handle.readframes(handle.getnframes())

    bytes_per_second = rate * width * channels
    frame_bytes = max(1, int(bytes_per_second * frame_ms / 1000))
    frames = [pcm[index : index + frame_bytes] for index in range(0, len(pcm), frame_bytes)]
    seconds = len(pcm) / bytes_per_second if bytes_per_second else 0.0
    return frames, rate, seconds


async def _deepgram_stream(
    *,
    api_key: str,
    wav_bytes: bytes,
    language: str,
    pacing: PacingMode,
    model: str,
) -> StreamingMetrics:
    import websockets

    frames, rate, audio_seconds = _frames(wav_bytes, FRAME_MILLISECONDS)
    metrics = StreamingMetrics(
        provider="deepgram", audio_seconds=audio_seconds, pacing=pacing, wall_seconds=0.0
    )

    query = {
        "model": model,
        "language": language.split("-")[0] if model.startswith("nova-3") else language,
        "encoding": "linear16",
        "sample_rate": str(rate),
        "channels": "1",
        "interim_results": "true",  # the whole point — partials as they form
        "punctuate": "true",
    }
    url = "wss://api.deepgram.com/v1/listen?" + "&".join(
        f"{key}={value}" for key, value in query.items()
    )

    started = time.perf_counter()
    pieces: list[str] = []

    try:
        async with websockets.connect(
            url, additional_headers={"Authorization": f"Token {api_key}"}
        ) as socket:

            async def send() -> None:
                for index, frame in enumerate(frames):
                    if pacing == "realtime":
                        # Hold each frame until its real playback moment, so
                        # arrival times reflect what a live speaker would see.
                        target = started + (index * FRAME_MILLISECONDS / 1000)
                        delay = target - time.perf_counter()
                        if delay > 0:
                            await asyncio.sleep(delay)
                    await socket.send(frame)
                await socket.send(json.dumps({"type": "CloseStream"}))

            async def receive() -> None:
                async for message in socket:
                    arrival = time.perf_counter() - started
                    try:
                        payload = json.loads(message)
                    except (TypeError, ValueError):
                        continue
                    if payload.get("type") == "Metadata":
                        break
                    channel = payload.get("channel") or {}
                    alternatives = channel.get("alternatives") or []
                    if not alternatives:
                        continue
                    text = (alternatives[0].get("transcript") or "").strip()
                    if not text:
                        continue
                    audio_start = float(payload.get("start", 0.0))
                    audio_end = audio_start + float(payload.get("duration", 0.0))
                    is_final = bool(payload.get("is_final"))
                    metrics.events.append(
                        StreamEvent(
                            arrival_offset=arrival,
                            audio_start=audio_start,
                            audio_end=audio_end,
                            text=text,
                            is_final=is_final,
                        )
                    )
                    if is_final:
                        pieces.append(text)

            await asyncio.gather(send(), receive())
    except Exception as exc:  # noqa: BLE001 - reported per provider, never fatal
        metrics.error = f"{type(exc).__name__}: {exc}"

    metrics.wall_seconds = time.perf_counter() - started
    metrics.transcript = " ".join(pieces).strip()
    return metrics


#: Providers with a streaming implementation here. Others raise
#: StreamingUnsupported, which the UI reports rather than hiding.
STREAMING_PROVIDERS = {"deepgram"}


def measure(
    *,
    provider: str,
    api_key: str,
    wav_bytes: bytes,
    language: str,
    pacing: PacingMode = "realtime",
    model: str = "",
) -> StreamingMetrics:
    """Stream one clip and measure when its results arrived."""
    if provider not in STREAMING_PROVIDERS:
        raise StreamingUnsupported(
            f"{provider} has no streaming endpoint wired up — batch latency only."
        )
    if not model:
        _, rate, _ = _frames(wav_bytes, FRAME_MILLISECONDS)
        model = "nova-2-phonecall" if rate <= 8_000 else "nova-3"

    return asyncio.run(
        _deepgram_stream(
            api_key=api_key,
            wav_bytes=wav_bytes,
            language=language,
            pacing=pacing,
            model=model,
        )
    )


def real_time_factor(latency_seconds: float | None, audio_seconds: float | None) -> float | None:
    """Batch RTF: how long the provider took over how long the audio was.

    Below 1.0 means it processed faster than real time — mandatory for a
    streaming deployment, and a useful sanity check even for batch use.
    """
    if not latency_seconds or not audio_seconds:
        return None
    return latency_seconds / audio_seconds
