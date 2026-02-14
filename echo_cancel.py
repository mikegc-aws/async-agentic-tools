"""Echo-cancelled audio I/O for BidiAgent using LiveKit WebRTC APM.

Drop-in replacement for BidiAudioIO that adds acoustic echo cancellation,
noise suppression, high-pass filtering, and auto gain control using
LiveKit's WebRTC Audio Processing Module.

The speaker output is registered as the far-end reference signal via
process_reverse_stream(), and mic input is cleaned via process_stream().
Dynamic delay estimation keeps the canceller aligned with actual latency.

Usage:
    from echo_cancel import AecAudioIO

    audio_io = AecAudioIO()           # same API as BidiAudioIO
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()],
    )
"""

import asyncio
import base64
import logging
import queue
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pyaudio
from livekit import rtc

from strands.experimental.bidi.types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
)

if TYPE_CHECKING:
    from strands.experimental.bidi.agent.agent import BidiAgent

logger = logging.getLogger(__name__)

# Audio rates matching Nova Sonic defaults
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1

# 10ms frame sizes (in samples)
INPUT_FRAMES_PER_10MS = int(INPUT_SAMPLE_RATE * 10 / 1000)   # 160
OUTPUT_FRAMES_PER_10MS = int(OUTPUT_SAMPLE_RATE * 10 / 1000)  # 240


# ---------------------------------------------------------------------------
# Shared APM state
# ---------------------------------------------------------------------------


class ApmState:
    """Shared WebRTC Audio Processing Module state between input and output.

    The output side feeds speaker audio as the far-end reference via
    process_reverse_stream(). The input side cleans mic audio via
    process_stream(). Dynamic delay estimation keeps the two aligned.
    """

    def __init__(self) -> None:
        self.apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True,
        )
        self.apm.set_stream_delay_ms(50)

        self.last_input_time: float = 0
        self.last_output_time: float = 0

    def update_stream_delay(self) -> None:
        """Update the APM stream delay based on observed input/output timing."""
        if self.last_output_time > 0 and self.last_input_time > 0:
            delay_ms = int(abs(self.last_input_time - self.last_output_time) * 1000)
            delay_ms = max(10, min(delay_ms, 500))
            try:
                self.apm.set_stream_delay_ms(delay_ms)
            except Exception:
                pass

    def process_mic_frames(self, raw_bytes: bytes) -> tuple[bytes, bytes]:
        """Process mic audio through APM in 10ms frames.

        Args:
            raw_bytes: Raw 16-bit PCM mic audio at INPUT_SAMPLE_RATE.

        Returns:
            (processed_audio, residual_bytes) — processed audio ready to
            send to the model, and any leftover bytes shorter than one frame.
        """
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        processed_chunks: list[bytes] = []

        i = 0
        while i + INPUT_FRAMES_PER_10MS <= len(samples):
            frame_data = samples[i : i + INPUT_FRAMES_PER_10MS]
            audio_frame = rtc.AudioFrame(
                data=frame_data.tobytes(),
                sample_rate=INPUT_SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=INPUT_FRAMES_PER_10MS,
            )
            self.apm.process_stream(audio_frame)
            processed_chunks.append(audio_frame.data.tobytes())
            i += INPUT_FRAMES_PER_10MS

        processed = b"".join(processed_chunks)
        residual = samples[i:].tobytes() if i < len(samples) else b""
        return processed, residual

    def process_speaker_reference(self, speaker_bytes: bytes) -> tuple[bytes, bytes]:
        """Register speaker audio as far-end reference in 10ms frames.

        Args:
            speaker_bytes: Raw 16-bit PCM speaker audio at OUTPUT_SAMPLE_RATE.

        Returns:
            (processed_audio, residual_bytes) — the audio to play (unchanged
            content, just frame-aligned), and any leftover bytes.
        """
        samples = np.frombuffer(speaker_bytes, dtype=np.int16)
        processed_chunks: list[bytes] = []

        i = 0
        while i + OUTPUT_FRAMES_PER_10MS <= len(samples):
            frame_data = samples[i : i + OUTPUT_FRAMES_PER_10MS]
            audio_frame = rtc.AudioFrame(
                data=frame_data.tobytes(),
                sample_rate=OUTPUT_SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=OUTPUT_FRAMES_PER_10MS,
            )
            self.apm.process_reverse_stream(audio_frame)
            processed_chunks.append(audio_frame.data.tobytes())
            i += OUTPUT_FRAMES_PER_10MS

        processed = b"".join(processed_chunks)
        residual = samples[i:].tobytes() if i < len(samples) else b""
        return processed, residual


# ---------------------------------------------------------------------------
# Echo-cancelled BidiInput (mic)
# ---------------------------------------------------------------------------


class _AecAudioInput:
    """Mic input with WebRTC APM echo cancellation applied before sending to the model."""

    _FRAMES_PER_BUFFER = 512

    def __init__(self, apm_state: ApmState, config: dict[str, Any]) -> None:
        self._apm = apm_state
        self._device_index = config.get("input_device_index")
        self._buf: queue.Queue[bytes] = queue.Queue()
        self._residual = b""

    async def start(self, agent: "BidiAgent") -> None:
        cfg = agent.model.config["audio"]
        self._channels = cfg["channels"]
        self._format = cfg["format"]
        self._rate = cfg["input_rate"]

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            channels=self._channels,
            format=pyaudio.paInt16,
            frames_per_buffer=self._FRAMES_PER_BUFFER,
            input=True,
            input_device_index=self._device_index,
            rate=self._rate,
            stream_callback=self._callback,
        )
        logger.debug("AEC audio input started")

    async def stop(self) -> None:
        if hasattr(self, "_stream"):
            self._stream.close()
        if hasattr(self, "_pa"):
            self._pa.terminate()
        logger.debug("AEC audio input stopped")

    async def __call__(self) -> BidiAudioInputEvent:
        raw = await asyncio.to_thread(self._buf.get)

        self._apm.last_input_time = time.time()

        # Prepend any residual from the previous call
        if self._residual:
            raw = self._residual + raw
            self._residual = b""

        processed, residual = self._apm.process_mic_frames(raw)
        self._residual = residual

        # If processing consumed nothing (chunk too small), pass raw through
        if not processed:
            processed = raw

        return BidiAudioInputEvent(
            audio=base64.b64encode(processed).decode("utf-8"),
            channels=self._channels,
            format=self._format,
            sample_rate=self._rate,
        )

    def _callback(self, in_data: bytes, *_: Any) -> tuple[None, int]:
        self._buf.put(in_data)
        return (None, pyaudio.paContinue)


# ---------------------------------------------------------------------------
# Echo-cancelled BidiOutput (speakers)
# ---------------------------------------------------------------------------


class _AecAudioOutput:
    """Speaker output that registers audio as APM far-end reference.

    Architecture matches the proven a2a_client pattern:
    - __call__ is non-blocking: just queues audio data or sets interruption flag
    - A background asyncio task runs the playback loop independently
    - The playback loop processes APM reference + blocking stream.write()
    - asyncio.sleep(0.001) between chunk writes yields control for interruptions

    This keeps process_reverse_stream() tightly coupled with actual speaker
    playback, AND allows the BidiAgent to deliver interruption events without
    waiting for writes to complete.
    """

    _FRAMES_PER_BUFFER = 512
    _WRITE_CHUNK_SIZE = 2048  # bytes per write (1024 samples at 16-bit)

    def __init__(self, apm_state: ApmState, config: dict[str, Any]) -> None:
        self._apm = apm_state
        self._device_index = config.get("output_device_index")
        self._residual = b""
        self._interrupted = False
        self._stopped = False
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._playback_task: asyncio.Task | None = None

    async def start(self, agent: "BidiAgent") -> None:
        cfg = agent.model.config["audio"]
        self._channels = cfg["channels"]
        self._rate = cfg["output_rate"]

        self._pa = pyaudio.PyAudio()
        # No stream_callback — playback loop uses blocking stream.write()
        self._stream = self._pa.open(
            channels=self._channels,
            format=pyaudio.paInt16,
            frames_per_buffer=self._FRAMES_PER_BUFFER,
            output=True,
            output_device_index=self._device_index,
            rate=self._rate,
        )

        # Start background playback task
        self._stopped = False
        self._playback_task = asyncio.create_task(self._playback_loop())
        logger.debug("AEC audio output started (decoupled playback loop)")

    async def stop(self) -> None:
        self._stopped = True
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        if hasattr(self, "_stream"):
            self._stream.close()
        if hasattr(self, "_pa"):
            self._pa.terminate()
        logger.debug("AEC audio output stopped")

    async def __call__(self, event: BidiOutputEvent) -> None:
        """Non-blocking: queue audio data or handle interruption."""
        if isinstance(event, BidiAudioStreamEvent):
            data = base64.b64decode(event["audio"])
            await self._audio_queue.put(data)

        elif isinstance(event, BidiInterruptionEvent):
            self._interrupted = True
            # Drain the queue immediately
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._residual = b""
            logger.debug("interruption — draining queue and stopping playback")

    async def _playback_loop(self) -> None:
        """Background task: read from queue, process APM reference, write to speakers."""
        loop = asyncio.get_running_loop()

        while not self._stopped:
            try:
                # Check for interruption first
                if self._interrupted:
                    # Drain any remaining audio
                    while not self._audio_queue.empty():
                        try:
                            self._audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self._interrupted = False
                    self._residual = b""
                    await asyncio.sleep(0.05)
                    continue

                # Wait for audio data with timeout (allows interruption checks)
                try:
                    data = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if not data or self._interrupted:
                    continue

                self._apm.last_output_time = time.time()
                self._apm.update_stream_delay()

                # Prepend residual from previous chunk
                if self._residual:
                    data = self._residual + data
                    self._residual = b""

                # Register as far-end reference for echo cancellation
                processed, residual = self._apm.process_speaker_reference(data)
                self._residual = residual

                if not processed:
                    continue

                # Write to speakers in chunks, yielding between each
                for i in range(0, len(processed), self._WRITE_CHUNK_SIZE):
                    if self._interrupted or self._stopped:
                        break
                    end = min(i + self._WRITE_CHUNK_SIZE, len(processed))
                    chunk = processed[i:end]
                    await loop.run_in_executor(None, self._stream.write, chunk)
                    await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._stopped:
                    logger.debug("playback loop error: %s", e)
                await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Public factory (drop-in replacement for BidiAudioIO)
# ---------------------------------------------------------------------------


class AecAudioIO:
    """Echo-cancelled audio I/O for BidiAgent.

    Drop-in replacement for ``BidiAudioIO``. Uses LiveKit's WebRTC Audio
    Processing Module for echo cancellation, noise suppression, high-pass
    filtering, and auto gain control.

    Args:
        **config: Additional config passed to PyAudio (input_device_index, etc.).
    """

    def __init__(self, **config: Any) -> None:
        self._apm_state = ApmState()
        self._config = config

    def input(self) -> _AecAudioInput:
        return _AecAudioInput(self._apm_state, self._config)

    def output(self) -> _AecAudioOutput:
        return _AecAudioOutput(self._apm_state, self._config)
