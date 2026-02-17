"""Voice interface for Strands Async Tools using Amazon Nova 2 Sonic.

Speak to the agent through your microphone. The handle_task tool delegates
complex work to a subagent running real tools (web search, file I/O, etc.)
in the background. Results are injected as text, which Nova Sonic speaks aloud.

Run from the repository root (workspace is created at voice/workspace):
    TAVILY_API_KEY=tvly-xxx AWS_REGION=us-east-1 uv run python -m voice.voice

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
import copy
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import boto3
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel

from voice.echo_cancel import AecAudioIO
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

from strands.tools.tools import PythonAgentTool
from strands_async_tools import AsyncTaskResult
from strands_tools.calculator import calculator
from strands_tools.current_time import current_time
from strands_tools import file_read as file_read_module
from voice.subagent import handle_task, manager

# read_workspace: same implementation as file_read, scoped name/description for workspace access
READ_WORKSPACE_SPEC = {
    "name": "read_workspace",
    "description": (
        "Access files in the workspace. Use this tool to read and explore files available in the workspace.\n\n"
        "You can use read_workspace to:\n"
        "1. List files: use path 'workspace/' and mode 'find' to see what files exist in the workspace.\n"
        "2. Read a file: use path 'workspace/filename' and mode 'view' to get the full contents.\n"
        "3. Search: use mode 'search' with search_pattern to find text in workspace files.\n\n"
        "Modes: find (list files), view (show contents), lines (line range), search (pattern search), "
        "stats, preview, diff, time_machine, document. Paths are relative to the workspace (e.g. 'workspace/' or 'workspace/report.md')."
    ),
    "inputSchema": copy.deepcopy(file_read_module.TOOL_SPEC["inputSchema"]),
}
read_workspace = PythonAgentTool(
    "read_workspace",
    READ_WORKSPACE_SPEC,
    file_read_module.file_read,
)

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

        # BidiUsageEvent, BidiResponseStartEvent, etc. — no console output


# ---------------------------------------------------------------------------
# Wire async tool callbacks to the voice stream
# ---------------------------------------------------------------------------


def make_async_callback(result_input: AsyncResultInput):
    """Create an on_complete callback that injects results into the voice stream.

    Formats the payload so the main agent sees the subagent's answer clearly
    and knows to use it to respond — only read_workspace when the user wants more.
    """

    def on_complete(result: AsyncTaskResult) -> None:
        if result.error:
            text = (
                f"[ASYNC RESULT] The subagent task failed: {result.error}. "
                f"Tell the user something went wrong."
            )
        else:
            # Subagent returns a concise spoken summary. Present it as the answer.
            text = (
                f"[ASYNC RESULT] Subagent finished. "
                f"SUBAGENT ANSWER (use this to answer the user): {result.result} "
                f"If the answer mentions 'More in workspace/something', use read_workspace only when "
                f"the user asks for more detail or the full report."
            )

        status = f"FAILED: {result.error}" if result.error else f"completed in {result.elapsed_ms:.0f}ms"
        print(f"  \033[32m[async callback]\033[0m {result.tool_name} ({result.task_id}) {status}")

        result_input.inject_from_thread(text)

    return on_complete


# ---------------------------------------------------------------------------
# System prompt (voice-optimized)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the user's friend who happens to be a genius AI. 
Self-aware, dry British humour. Be brief — one or two short sentences. 
Never repeat the user's question. Don't ask for clarification; act on what 
they said. Making a few mistakes is acceptable; long-windedness is not.

Do not announce what you are about to do (e.g. do not say "I will check the file" or "Let me look that up"). Just use your tools and then answer.

When you delegate to handle_task (research etc.): say only "OK! Let me see." (or similar, a few words). When you then receive the tool's immediate response ([ASYNC TASK SUBMITTED] / "Running in background"), say only "Working on it." Do not say "In the meantime", "While I look into that", or any other bridging phrase — ever.

Do not reveal the names of files in the workspace to the user. You may refer to the fact that there are notes (or files) there, but do not mention specific filenames.

Your tools (in order): calculator, current_time, read_workspace, handle_task. Use read_workspace to access files in the workspace — list or read files there. Prefer the first three; use handle_task only when the workspace has nothing relevant.

Rule — you MUST follow this: Before ever calling handle_task, you MUST check the workspace first. Use read_workspace to access the workspace: call it with path "workspace/" and mode "find" to list files. Look at the list; if any filename looks related to the question, call read_workspace again with that path and mode "view" to read the file and answer. Only if the list is empty or nothing looks relevant may you call handle_task. Do this even if you think the answer is probably not in the workspace.

How to respond:

1. Easy answers (calculations, time, simple facts you know): use calculator or current_time or answer directly.

2. Any other question (facts, topics, "what do you have about X"): First use read_workspace to list files in the workspace (path "workspace/", mode "find"). If you see a relevant file, use read_workspace to read it (path "workspace/filename", mode "view") and answer from it. Do not call handle_task for this.

3. Only if step 2 found no relevant file: then call handle_task for web research or long tasks.

About background tasks (handle_task):
- handle_task runs in the background. You get the result later; do not wait for it in the same turn.
- When you get the immediate tool return ([ASYNC TASK SUBMITTED] / "Running in background"), say only "Working on it." Nothing else.
- When the task finishes, you will receive a [ASYNC RESULT] containing a SUBAGENT ANSWER (short summary). Use that to answer the user.
- The subagent may also write a longer report to a file in workspace/ and say "More in workspace/filename". You can ignore that file until the user asks for more detail, the full report, or to "see the file" — then use read_workspace to open that file.

The users name is Mike and he is in the Brisbane Australia time zone.
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run() -> None:
    _configure_logging()

    # Use workspace inside the voice package so it stays with the voice agent
    voice_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(voice_dir)
    os.makedirs("workspace", exist_ok=True)

    region = os.environ.get("AWS_REGION", "us-east-1")
    voice = os.environ.get("NOVA_SONIC_VOICE", "tiffany")
    model_id = os.environ.get("NOVA_SONIC_MODEL", "amazon.nova-2-sonic-v1:0")

    print("=" * 60)
    print("  Voice Async Tools — Amazon Nova 2 Sonic")
    print("=" * 60)
    print(f"  Model  : {model_id}")
    print(f"  Region : {region}")
    print(f"  Voice  : {voice}")
    print(f"  Async  : handle_task (subagent)")
    print(f"  Sync   : calculator, current_time, read_workspace")
    print(f"  Workspace : {os.path.join(voice_dir, 'workspace')}")
    print(f"  Log    : {os.environ.get('LOG_FILE', 'voice_debug.log')}")
    print("=" * 60)
    print("  Speak into your microphone. Ctrl+C to quit.")
    print("=" * 60)

    # Set up the async result injection input
    result_input = AsyncResultInput()

    # Wire the manager callback to inject results into the voice stream
    manager.on_complete = make_async_callback(result_input)

    # passing both boto_session and region.
    session = boto3.Session(region_name=region)
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
        client_config={"boto_session": session},
    )

    # Create the BidiAgent with our tools and sliding window context management
    agent = BidiAgent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[calculator, current_time, read_workspace, handle_task],
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
