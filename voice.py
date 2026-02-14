"""Voice interface for Strands Async Tools using Amazon Nova 2 Sonic.

Speak to the agent through your microphone. Async tools run in the background
and their results are injected as text, which Nova Sonic speaks aloud.

Run:
    uv run python voice.py

Requires:
    - AWS credentials with Bedrock access (us-east-1)
    - Microphone and speakers
    - Model access for amazon.nova-2-sonic-v1:0

Options (env vars):
    AWS_REGION          - AWS region (default: us-east-1)
    NOVA_SONIC_VOICE    - Voice name (default: matthew)
    NOVA_SONIC_MODEL    - Model ID (default: amazon.nova-2-sonic-v1:0)
"""

import asyncio
import logging
import os
import random
import sys
import time
from typing import TYPE_CHECKING, Any

import boto3
from strands import tool
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel
from strands.experimental.bidi.types.events import (
    BidiConnectionCloseEvent,
    BidiConnectionStartEvent,
    BidiErrorEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiResponseCompleteEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
)
from strands.experimental.bidi.types.io import BidiInput, BidiOutput

from strands_async_tools import AsyncToolManager, AsyncTaskResult, tool_async

if TYPE_CHECKING:
    from strands.experimental.bidi.agent.agent import BidiAgent as BidiAgentType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async Tool Manager
# ---------------------------------------------------------------------------

manager = AsyncToolManager(max_workers=4)

# ---------------------------------------------------------------------------
# Async Tools (same as TUI/demo, but with voice-appropriate timing)
# ---------------------------------------------------------------------------


@tool_async(manager)
def research_topic(topic: str) -> str:
    """Research a topic thoroughly and return detailed findings."""
    delay = random.uniform(8, 15)
    time.sleep(delay)
    findings = [
        f"{topic} has seen 340% growth in the last 2 years.",
        f"Major players in {topic} include Acme Corp, Nexus Labs, and Orion Systems.",
        f"Experts predict the {topic} market will reach $50 billion by 2028.",
        f"Recent regulatory changes may impact {topic} adoption in the EU.",
        f"A breakthrough paper on {topic} was published last month in Nature.",
    ]
    return " ".join(random.sample(findings, k=random.randint(2, 3)))


@tool_async(manager)
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment and key themes in a piece of text."""
    delay = random.uniform(6, 12)
    time.sleep(delay)
    sentiments = [
        "overwhelmingly positive",
        "cautiously optimistic",
        "mixed but trending positive",
    ]
    themes = ["innovation", "market disruption", "sustainability", "cost efficiency"]
    return (
        f"The sentiment is {random.choice(sentiments)}. "
        f"Key themes are {', '.join(random.sample(themes, k=2))}. "
        f"Confidence level is {random.randint(75, 98)} percent."
    )


@tool_async(manager)
def fetch_weather(city: str) -> str:
    """Get the current weather for a city."""
    delay = random.uniform(4, 8)
    time.sleep(delay)
    conditions = ["sunny", "partly cloudy", "overcast", "light rain", "clear skies"]
    return (
        f"The weather in {city} is {random.choice(conditions)}, "
        f"{random.randint(5, 35)} degrees celsius, "
        f"humidity {random.randint(30, 90)} percent, "
        f"wind {random.randint(5, 40)} kilometers per hour."
    )


# ---- Sync tool ----


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression and return the result immediately."""
    allowed = {"abs": abs, "min": min, "max": max, "round": round, "int": int, "float": float}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"The answer is {result}"
    except Exception as e:
        return f"Error: {e}"


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
    """BidiOutput that prints conversation transcripts and status to the console."""

    async def start(self, agent: "BidiAgentType") -> None:
        pass

    async def stop(self) -> None:
        pass

    async def __call__(self, event: BidiOutputEvent) -> None:
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

        elif isinstance(event, BidiResponseCompleteEvent):
            pass  # Normal, no need to print

        elif isinstance(event, BidiConnectionCloseEvent):
            print(f"  \033[90m[disconnected]\033[0m {event['reason']}")

        elif isinstance(event, BidiErrorEvent):
            print(f"  \033[31m[error]\033[0m {event['message']}")


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
You are a friendly voice assistant with access to background tools.

You have these tools available:
- research_topic: Researches a topic (runs in background, takes a while)
- analyze_sentiment: Analyzes text sentiment (runs in background)
- fetch_weather: Gets weather for a city (runs in background)
- calculate: Does math (returns immediately)

IMPORTANT RULES FOR ASYNC TOOLS:
When you call research_topic, analyze_sentiment, or fetch_weather, they start \
running in the background and return a task ID immediately. The actual results \
will be delivered to you later as text messages tagged [ASYNC RESULT].

When you get a task submitted confirmation:
- Tell the user you've started the task
- Continue the conversation naturally
- Do NOT guess what the results will be

When you receive an [ASYNC RESULT] message:
- This is an automated delivery, not something the user said
- Summarize the result naturally and conversationally for the user
- If other tasks are still pending, mention you're still waiting

Keep your responses short and conversational. You are speaking out loud, \
not writing. Avoid lists, bullet points, or technical formatting. \
Speak naturally as in a face-to-face conversation."""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run() -> None:
    region = os.environ.get("AWS_REGION", "us-east-1")
    voice = os.environ.get("NOVA_SONIC_VOICE", "matthew")
    model_id = os.environ.get("NOVA_SONIC_MODEL", "amazon.nova-2-sonic-v1:0")

    print("=" * 60)
    print("  Voice Async Tools — Amazon Nova 2 Sonic")
    print("=" * 60)
    print(f"  Model  : {model_id}")
    print(f"  Region : {region}")
    print(f"  Voice  : {voice}")
    print(f"  Async  : research_topic, analyze_sentiment, fetch_weather")
    print(f"  Sync   : calculate")
    print("=" * 60)
    print("  Speak into your microphone. Ctrl+C to quit.")
    print("=" * 60)

    # Set up the async result injection input
    result_input = AsyncResultInput()

    # Wire the manager callback to inject results into the voice stream
    manager.on_complete = make_async_callback(result_input)

    # Create Nova 2 Sonic model
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
            "boto_session": boto3.Session(),
            "region": region,
        },
    )

    # Create the BidiAgent with our tools
    agent = BidiAgent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_topic, analyze_sentiment, fetch_weather, calculate],
    )

    # Audio I/O (microphone + speakers)
    audio_io = BidiAudioIO()

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
