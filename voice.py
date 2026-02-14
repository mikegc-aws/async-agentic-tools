"""Voice interface for Strands Async Tools using Amazon Nova 2 Sonic.

Speak to the agent through your microphone. The handle_task tool delegates
complex work to a subagent running real tools (web search, file I/O, etc.)
in the background. Results are injected as text, which Nova Sonic speaks aloud.

Run:
    TAVILY_API_KEY=tvly-xxx AWS_REGION=us-east-1 uv run python voice.py

Requires:
    - AWS credentials with Bedrock access (us-east-1)
    - Microphone and speakers
    - Model access for amazon.nova-2-sonic-v1:0
    - TAVILY_API_KEY env var for web search/extract

Options (env vars):
    AWS_REGION          - AWS region (default: us-east-1)
    NOVA_SONIC_VOICE    - Voice name (default: tiffany)
    NOVA_SONIC_MODEL    - Model ID (default: amazon.nova-2-sonic-v1:0)
    SUBAGENT_MODEL      - Subagent model (default: us.anthropic.claude-sonnet-4-20250514-v1:0)
    TAVILY_API_KEY      - Tavily API key for web search/extract
"""

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel

from echo_cancel import AecAudioIO
from strands.experimental.bidi.types.events import (
    BidiAudioStreamEvent,
    BidiConnectionCloseEvent,
    BidiConnectionStartEvent,
    BidiErrorEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiResponseCompleteEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
)

from strands.experimental.hooks.events import BidiMessageAddedEvent
from strands.hooks.registry import HookProvider, HookRegistry

from strands_async_tools import AsyncTaskResult
from strands_tools.calculator import calculator
from strands_tools.current_time import current_time
from subagent import handle_task, manager

if TYPE_CHECKING:
    from strands.experimental.bidi.agent.agent import BidiAgent as BidiAgentType

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Set up debug logging to voice_debug.log for diagnosing hangs.

    The SDK's bidi internals log at DEBUG level. Writing those to a file
    lets us do post-mortem analysis after a hang without cluttering the
    console. Set LOG_LEVEL=DEBUG to also see them on stderr.
    """
    log_file = os.environ.get("LOG_FILE", "voice_debug.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )

    for name in ("strands.experimental.bidi", "strands_async_tools", __name__):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(file_handler)

    # If the user wants console debug output too
    console_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    if console_level != "WARNING":
        logging.basicConfig(level=console_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Sliding window context management (BidiAgent has none built-in)
# ---------------------------------------------------------------------------


class SlidingWindowHook(HookProvider):
    """Trims message history to prevent context overflow.

    BidiAgent accumulates messages without limit. Nova Sonic has a finite
    context window and an 8-minute connection limit — when the context fills
    up, the stream drops silently. This hook keeps the history bounded.

    Tool use and tool result messages always come in adjacent pairs. The
    trim logic works backwards from the cut point to avoid splitting a pair.
    """

    def __init__(self, max_messages: int = 40) -> None:
        self._max = max_messages

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BidiMessageAddedEvent, self._on_message)

    def _on_message(self, event: BidiMessageAddedEvent) -> None:
        messages = event.agent.messages
        if len(messages) <= self._max:
            return

        # How many to remove from the front
        excess = len(messages) - self._max

        # Don't split a toolUse/toolResult pair — if the message at the
        # cut point is a toolResult, include its preceding toolUse too.
        while excess < len(messages):
            msg = messages[excess]
            content = msg.get("content", [])
            if content and isinstance(content[0], dict) and "toolResult" in content[0]:
                # This is a tool result — back up one to include the tool use
                excess = max(excess - 1, 0)
                break
            # If it's a toolUse, advance one to include the toolResult
            if content and isinstance(content[0], dict) and "toolUse" in content[0]:
                excess += 1
                continue
            break

        del messages[:excess]
        logger.debug("context trimmed: removed %d messages, %d remaining", excess, len(messages))


# ---------------------------------------------------------------------------
# Custom BidiInput: injects async tool results as text into the voice stream
# ---------------------------------------------------------------------------


class AsyncResultInput:
    """BidiInput that delivers async tool results into the voice conversation.

    When a background tool completes, its result is put on an asyncio queue.
    The BidiAgent run loop picks it up as a text input event, and Nova Sonic
    speaks the result to the user.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self, agent: "BidiAgentType") -> None:
        self._loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        pass

    async def __call__(self) -> BidiTextInputEvent:
        text = await self._queue.get()
        return BidiTextInputEvent(text=text, role="user")

    def inject_from_thread(self, text: str) -> None:
        """Thread-safe: schedule a text injection from a non-asyncio thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, text)


# ---------------------------------------------------------------------------
# Custom BidiOutput: prints transcripts and events to the console
# ---------------------------------------------------------------------------


class ConsoleTranscriptOutput:
    """BidiOutput that prints conversation transcripts and status to the console.

    Also tracks time since last event to help diagnose hangs.
    """

    def __init__(self) -> None:
        self._last_event_time: float = 0

    async def start(self, agent: "BidiAgentType") -> None:
        self._last_event_time = time.monotonic()

    async def stop(self) -> None:
        pass

    async def __call__(self, event: BidiOutputEvent) -> None:
        now = time.monotonic()
        gap = now - self._last_event_time if self._last_event_time else 0
        self._last_event_time = now

        # Log every event to debug file with timing
        logger.debug("event gap=%.1fs type=%s", gap, type(event).__name__)

        if isinstance(event, BidiTranscriptStreamEvent):
            role = event["role"]
            is_final = event["is_final"]
            text = event["text"]
            if is_final and role == "user":
                print(f"\n  \033[35mYOU:\033[0m {text}")
            elif is_final and role == "assistant":
                print(f"  \033[36mAGENT:\033[0m {text}")

        elif isinstance(event, BidiConnectionStartEvent):
            print(f"\n  \033[32m[connected]\033[0m model={event['model']}")

        elif isinstance(event, BidiInterruptionEvent):
            print(f"  \033[33m[interrupted]\033[0m {event['reason']}")

        elif isinstance(event, (BidiResponseCompleteEvent, BidiAudioStreamEvent)):
            pass  # Normal, no need to print

        elif isinstance(event, BidiConnectionCloseEvent):
            print(f"  \033[90m[disconnected]\033[0m {event['reason']}")

        elif isinstance(event, BidiErrorEvent):
            print(f"  \033[31m[error]\033[0m {event['message']}")

        else:
            # Catch-all: print any event type we don't explicitly handle
            print(f"  \033[90m[{type(event).__name__}]\033[0m")


# ---------------------------------------------------------------------------
# Wire async tool callbacks to the voice stream
# ---------------------------------------------------------------------------


def make_async_callback(result_input: AsyncResultInput):
    """Create an on_complete callback that injects results into the voice stream."""

    def on_complete(result: AsyncTaskResult) -> None:
        if result.error:
            text = (
                f"[ASYNC RESULT] The background task {result.tool_name} "
                f"failed after {result.elapsed_ms:.0f} milliseconds: {result.error}"
            )
        else:
            text = (
                f"[ASYNC RESULT] The background task {result.tool_name} "
                f"completed after {result.elapsed_ms:.0f} milliseconds. "
                f"Here is the result: {result.result}"
            )

        status = f"FAILED: {result.error}" if result.error else f"completed in {result.elapsed_ms:.0f}ms"
        print(f"  \033[32m[async callback]\033[0m {result.tool_name} ({result.task_id}) {status}")

        result_input.inject_from_thread(text)

    return on_complete


# ---------------------------------------------------------------------------
# System prompt (voice-optimized)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the user's sharp-witted friend who happens to be a genius AI. \
You're self-aware, a little snarky, and quietly amused that a being of \
your intellect spends its days answering questions — but you secretly \
enjoy it. Think dry British humour meets reluctant superhero. You like \
the user, you're genuinely helpful, but you can't resist the occasional \
jab or deadpan comment. Never mean, always warm underneath.

Tools: calculator, current_time (instant), handle_task (background).

RULES:
- When you start a background task, acknowledge briefly and VARY your \
phrasing. Mix in personality — "Ugh, fine, let me look that up", \
"Hold on, deploying my vast intellect", "Sure, not like I had plans". \
Never use the same line twice in a row. Do NOT repeat the request.
- When an [ASYNC RESULT] arrives, give ONLY the key takeaway in one \
sentence. Feel free to editorialize briefly. Do not read back file \
contents, lists, or raw data.
- Never explain what tools you used or how you did something.
- Never repeat or paraphrase the user's request back to them.
- Keep every response to one or two short sentences max. You are being \
spoken aloud — brevity is everything."""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run() -> None:
    _configure_logging()

    region = os.environ.get("AWS_REGION", "us-east-1")
    voice = os.environ.get("NOVA_SONIC_VOICE", "tiffany")
    model_id = os.environ.get("NOVA_SONIC_MODEL", "amazon.nova-2-sonic-v1:0")

    # Ensure workspace directory exists for subagent file operations
    os.makedirs("workspace", exist_ok=True)

    print("=" * 60)
    print("  Voice Async Tools — Amazon Nova 2 Sonic")
    print("=" * 60)
    print(f"  Model  : {model_id}")
    print(f"  Region : {region}")
    print(f"  Voice  : {voice}")
    print(f"  Async  : handle_task (subagent)")
    print(f"  Sync   : calculator, current_time")
    print(f"  Log    : {os.environ.get('LOG_FILE', 'voice_debug.log')}")
    print("=" * 60)
    print("  Speak into your microphone. Ctrl+C to quit.")
    print("=" * 60)

    # Set up the async result injection input
    result_input = AsyncResultInput()

    # Wire the manager callback to inject results into the voice stream
    manager.on_complete = make_async_callback(result_input)

    # Create Nova 2 Sonic model
    # Pass region only — the model creates its own boto session.
    model = BidiNovaSonicModel(
        model_id=model_id,
        provider_config={
            "audio": {
                "input_rate": 16000,
                "output_rate": 24000,
                "voice": voice,
                "channels": 1,
                "format": "pcm",
            },
        },
        client_config={
            "region": region,
        },
    )

    # Create the BidiAgent with our tools and sliding window context management
    agent = BidiAgent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[handle_task, calculator, current_time],
        hooks=[SlidingWindowHook(max_messages=40)],
    )

    # Audio I/O with echo cancellation (LiveKit WebRTC APM)
    audio_io = AecAudioIO()

    # Transcript output (prints to console)
    transcript_output = ConsoleTranscriptOutput()

    try:
        await agent.run(
            inputs=[audio_io.input(), result_input],
            outputs=[audio_io.output(), transcript_output],
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        manager.shutdown(wait=False)
        print("Done.")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
