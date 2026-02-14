"""Echo-cancelled audio I/O for BidiAgent.

Drop-in replacement for BidiAudioIO that adds acoustic echo cancellation
using speexdsp (via pyaec). The speaker output is recorded as a reference
signal and subtracted from the mic input so the model doesn't hear itself.

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
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import pyaudio
from pyaec import Aec

from strands.experimental.bidi.types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
)
from strands.experimental.bidi.types.io import BidiInput, BidiOutput

if TYPE_CHECKING:
    from strands.experimental.bidi.agent.agent import BidiAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared AEC state
# ---------------------------------------------------------------------------


class _ReferenceBuffer:
    """Thread-safe byte buffer that stores the speaker reference signal."""

    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._max_bytes = int(max_seconds * sample_rate) * 2  # 16-bit = 2 bytes/sample

    def write(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)
            if len(self._buf) > self._max_bytes:
                del self._buf[: len(self._buf) - self._max_bytes]

    def read(self, n_bytes: int) -> bytes:
        """Read n_bytes from the buffer. Pads with silence if not enough data."""
        with self._lock:
            available = min(n_bytes, len(self._buf))
            data = bytes(self._buf[:available])
            del self._buf[:available]
        if available < n_bytes:
            data += b"\x00" * (n_bytes - available)
        return data

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


class AecState:
    """Shared acoustic echo cancellation state between input and output.

    The output side feeds speaker audio as the far-end reference.
    The input side processes mic audio through the canceller.
    """

    def __init__(
        self,
        mic_rate: int = 16000,
        speaker_rate: int = 24000,
        frame_size: int = 160,
        filter_ms: int = 300,
        duck_gain: float = 0.3,
    ) -> None:
        filter_length = int(mic_rate * filter_ms / 1000)
        self.aec = Aec(frame_size, filter_length, mic_rate, True)
        self.frame_size = frame_size
        self.mic_rate = mic_rate
        self.speaker_rate = speaker_rate
        self.duck_gain = duck_gain

        self._ref = _ReferenceBuffer(max_seconds=2.0, sample_rate=mic_rate)
        self._playing = False
        self._playing_lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        with self._playing_lock:
            return self._playing

    def feed_reference(self, speaker_bytes: bytes) -> None:
        """Called by the output handler with raw speaker audio (at speaker_rate).

        Resamples to mic_rate and stores as the far-end reference for the canceller.
        """
        samples = np.frombuffer(speaker_bytes, dtype=np.int16)
        if len(samples) == 0:
            return

        # Resample speaker_rate → mic_rate
        n_out = int(len(samples) * self.mic_rate / self.speaker_rate)
        if n_out == 0:
            return
        indices = np.linspace(0, len(samples) - 1, n_out)
        resampled = np.interp(indices, np.arange(len(samples)), samples.astype(np.float64))
        self._ref.write(resampled.astype(np.int16).tobytes())

        with self._playing_lock:
            self._playing = True

    def clear_reference(self) -> None:
        """Called on interruption — discard stale reference data."""
        self._ref.clear()
        with self._playing_lock:
            self._playing = False

    def process_mic(self, mic_bytes: bytes) -> bytes:
        """Process a mic chunk through AEC + mild ducking."""
        mic_samples = np.frombuffer(mic_bytes, dtype=np.int16)
        if len(mic_samples) == 0:
            return mic_bytes

        frame_bytes = self.frame_size * 2  # 16-bit
        output_chunks: list[bytes] = []

        for i in range(0, len(mic_samples), self.frame_size):
            frame = mic_samples[i : i + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

            # Read aligned reference frame
            ref_bytes = self._ref.read(frame_bytes)
            ref = np.frombuffer(ref_bytes, dtype=np.int16)
            has_ref = np.any(ref != 0)

            # Run speex echo canceller
            cleaned = np.array(self.aec.cancel_echo(frame, ref), dtype=np.int16)

            # Mild ducking as safety net when speaker is active
            if has_ref:
                cleaned = (cleaned.astype(np.float32) * self.duck_gain).astype(np.int16)
            else:
                with self._playing_lock:
                    self._playing = False

            output_chunks.append(cleaned.tobytes())

        return b"".join(output_chunks)


# ---------------------------------------------------------------------------
# Echo-cancelled BidiInput (mic)
# ---------------------------------------------------------------------------


class _AecAudioInput:
    """Mic input with AEC processing applied before sending to the model."""

    _FRAMES_PER_BUFFER = 512

    def __init__(self, aec_state: AecState, config: dict[str, Any]) -> None:
        self._aec = aec_state
        self._device_index = config.get("input_device_index")
        self._buf: queue.Queue[bytes] = queue.Queue()

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

        # Process through echo canceller
        cleaned = self._aec.process_mic(raw)

        return BidiAudioInputEvent(
            audio=base64.b64encode(cleaned).decode("utf-8"),
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
    """Speaker output that records audio as AEC reference."""

    _FRAMES_PER_BUFFER = 512

    def __init__(self, aec_state: AecState, config: dict[str, Any]) -> None:
        self._aec = aec_state
        self._device_index = config.get("output_device_index")
        self._buf: queue.Queue[bytes] = queue.Queue()

    async def start(self, agent: "BidiAgent") -> None:
        cfg = agent.model.config["audio"]
        self._channels = cfg["channels"]
        self._rate = cfg["output_rate"]

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            channels=self._channels,
            format=pyaudio.paInt16,
            frames_per_buffer=self._FRAMES_PER_BUFFER,
            output=True,
            output_device_index=self._device_index,
            rate=self._rate,
            stream_callback=self._callback,
        )
        logger.debug("AEC audio output started")

    async def stop(self) -> None:
        if hasattr(self, "_stream"):
            self._stream.close()
        if hasattr(self, "_pa"):
            self._pa.terminate()
        logger.debug("AEC audio output stopped")

    async def __call__(self, event: BidiOutputEvent) -> None:
        if isinstance(event, BidiAudioStreamEvent):
            data = base64.b64decode(event["audio"])

            # Record as AEC reference BEFORE queueing for playback
            self._aec.feed_reference(data)

            self._buf.put(data)
            logger.debug("aec_ref_fed=<%d> | audio chunk buffered", len(data))

        elif isinstance(event, BidiInterruptionEvent):
            self._aec.clear_reference()
            # Drain playback buffer
            while not self._buf.empty():
                try:
                    self._buf.get_nowait()
                except queue.Empty:
                    break
            logger.debug("interruption — cleared AEC reference and playback buffer")

    def _callback(self, _in_data: None, frame_count: int, *_: Any) -> tuple[bytes, int]:
        byte_count = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
        try:
            data = self._buf.get_nowait()
            # Pad or trim to requested size
            if len(data) < byte_count:
                data += b"\x00" * (byte_count - len(data))
            elif len(data) > byte_count:
                # Put the remainder back
                self._buf.put(data[byte_count:])
                data = data[:byte_count]
        except queue.Empty:
            data = b"\x00" * byte_count
        return (data, pyaudio.paContinue)


# ---------------------------------------------------------------------------
# Public factory (drop-in replacement for BidiAudioIO)
# ---------------------------------------------------------------------------


class AecAudioIO:
    """Echo-cancelled audio I/O for BidiAgent.

    Drop-in replacement for ``BidiAudioIO``. Uses speexdsp (via pyaec) to
    remove speaker echo from the mic signal so the model doesn't hear itself.

    Args:
        filter_ms: AEC filter length in milliseconds. Longer catches more echo
            but uses more CPU. 300ms is a good default for laptops.
        duck_gain: Residual ducking gain applied to mic when speaker is active.
            0.3 = 70% attenuation as safety net. Set to 1.0 to disable ducking.
        **config: Additional config passed to PyAudio (input_device_index, etc.).
    """

    def __init__(self, filter_ms: int = 300, duck_gain: float = 0.3, **config: Any) -> None:
        self._aec_state = AecState(
            mic_rate=16000,
            speaker_rate=24000,
            frame_size=160,
            filter_ms=filter_ms,
            duck_gain=duck_gain,
        )
        self._config = config

    def input(self) -> _AecAudioInput:
        return _AecAudioInput(self._aec_state, self._config)

    def output(self) -> _AecAudioOutput:
        return _AecAudioOutput(self._aec_state, self._config)
