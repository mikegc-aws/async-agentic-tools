# Echo Cancellation Research Notes

## Problem

When running `voice.py` on a laptop, the built-in mic picks up the speaker
output. Nova Sonic hears its own voice, interprets it as user speech, and
enters a feedback loop.

## Attempt 1: speexdsp via pyaec (failed)

**Library:** [pyaec](https://pypi.org/project/pyaec/) -- Python bindings for
the Speex echo cancellation DSP.

**Dependencies:** `pyaec>=1.0.1`, `numpy>=2.4.2`

### Architecture

Created `echo_cancel.py` as a drop-in replacement for Strands'
`BidiAudioIO`. The design had three main pieces:

1. **`_ReferenceBuffer`** -- Thread-safe `bytearray` storing the far-end
   (speaker) reference signal. The output side writes to it; the input side
   reads from it frame-by-frame.

2. **`AecState`** -- Shared state between input and output:
   - Holds the `pyaec.Aec` instance (`frame_size=160`, filter length derived
     from `filter_ms` parameter, default 300 ms).
   - `feed_reference(speaker_bytes)` -- called by the output handler.
     Resamples 24 kHz speaker audio to 16 kHz (mic rate) using
     `numpy.interp`, then stores in the reference buffer.
   - `process_mic(mic_bytes)` -- called by the input handler. Walks the mic
     signal in 160-sample frames, reads an aligned reference frame, runs
     `aec.cancel_echo(mic_frame, ref_frame)`, and applies mild ducking
     (`duck_gain=0.3`, i.e. 70% attenuation) as a safety net while the
     speaker is active.
   - `clear_reference()` -- called on barge-in / interruption to discard
     stale reference data.

3. **`AecAudioIO`** -- Factory class (same API as `BidiAudioIO`):
   - `.input()` returns `_AecAudioInput` -- PyAudio mic stream with AEC
     processing applied before base64-encoding and sending to Nova Sonic.
   - `.output()` returns `_AecAudioOutput` -- PyAudio speaker stream that
     records all outgoing audio as AEC reference before playing it.

### What went wrong

Two separate issues:

1. **Garbled/choppy playback.** The output buffer used `queue.get_nowait()`
   per PyAudio callback, padding with silence when chunks were too small.
   This created `[audio][silence][audio][silence]...` gaps. Fixing to an
   accumulating bytearray resolved the choppiness but not the feedback.

2. **Feedback persisted.** Speex's linear adaptive filter couldn't handle:
   - Timing misalignment between reference and actual speaker-to-mic path
   - Manual 24 kHz to 16 kHz resampling via `numpy.interp` (lossy)
   - Complex acoustic paths of laptop speakers at close range
   - No non-linear echo processing (speex is linear-only)

The speexdsp code is preserved in git history at commit `2fe5cc2`.

## Attempt 2: LiveKit WebRTC APM (current)

**Library:** [livekit](https://pypi.org/project/livekit/) -- wraps Google's
WebRTC Audio Processing Module, the same AEC used in Chrome/WebRTC calls.

**Dependencies:** `livekit>=1.0.25`, `numpy>=2.4.2`

### Why this approach

Found a working implementation in a previous project
(`2025/ossna/server/strands-a2a-demo/a2a_client.py`) that used LiveKit's
`rtc.AudioProcessingModule` with Nova Sonic v1 and confirmed it worked.

Key advantages over speex:
- **Non-linear echo processing** -- handles the complex acoustic paths
  that speex's linear filter couldn't model
- **Bundled audio pipeline** -- echo cancellation, noise suppression,
  high-pass filter, and auto gain control in one module
- **No manual resampling** -- input and output operate at their native
  sample rates (16 kHz and 24 kHz respectively)
- **Dynamic delay estimation** -- tracks actual input/output timing to
  keep the canceller aligned with real latency

### Architecture

**`ApmState`** -- Shared WebRTC APM state:
- Creates `rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, ...)`
- Initial stream delay: 50 ms
- `process_mic_frames(raw_bytes)` -- segments mic audio into 10 ms frames
  (160 samples at 16 kHz), runs each through `apm.process_stream()`
- `process_speaker_reference(speaker_bytes)` -- segments speaker audio into
  10 ms frames (240 samples at 24 kHz), runs each through
  `apm.process_reverse_stream()` to register as far-end reference
- `update_stream_delay()` -- computes delay from timing difference between
  last input and last output, clamps to 10-500 ms

**`_AecAudioInput`** -- BidiInput (mic):
- PyAudio callback captures raw mic audio into a queue
- `__call__()` gets raw bytes, processes through `apm.process_stream()` in
  10 ms frames, returns cleaned audio as `BidiAudioInputEvent`
- Carries residual samples between calls for frame alignment

**`_AecAudioOutput`** -- BidiOutput (speakers):
- Receives `BidiAudioStreamEvent`, decodes audio
- Processes through `apm.process_reverse_stream()` in 10 ms frames
- Appends to an accumulating `bytearray` playback buffer
- PyAudio callback reads exactly the requested bytes (no gaps)
- Clears buffer on `BidiInterruptionEvent` for barge-in

**`AecAudioIO`** -- Factory (drop-in replacement for `BidiAudioIO`):
- Creates shared `ApmState`
- `.input()` / `.output()` return the AEC-enabled input/output

### Integration

```python
from echo_cancel import AecAudioIO

audio_io = AecAudioIO()
await agent.run(
    inputs=[audio_io.input(), result_input],
    outputs=[audio_io.output(), transcript_output],
)
```
