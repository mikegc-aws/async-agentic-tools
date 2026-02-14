"""Interactive TUI for Strands Async Tools.

Run:
    uv run python tui.py

Requires AWS credentials configured for Bedrock access.
Override the model with STRANDS_MODEL env var.
"""

import os
import random
import threading
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

# ---------------------------------------------------------------------------
# Tools are defined at module level so they can reference the shared manager.
# The manager is created at import time; the agent is wired up in on_mount.
# ---------------------------------------------------------------------------

from strands import Agent, tool  # noqa: E402

from strands_async_tools import AsyncAgent, AsyncToolManager, tool_async  # noqa: E402

manager = AsyncToolManager(max_workers=4)

# ---- Async tools ----


@tool_async(manager)
def research_topic(topic: str) -> str:
    """Research a topic thoroughly and return detailed findings."""
    delay = random.uniform(15, 30)
    time.sleep(delay)
    findings = [
        f"Key finding: {topic} has seen 340% growth in the last 2 years.",
        f"Major players in {topic} include Acme Corp, Nexus Labs, and Orion Systems.",
        f"Experts predict the {topic} market will reach $50B by 2028.",
        f"Recent regulatory changes may impact {topic} adoption in the EU.",
        f"A breakthrough paper on {topic} was published last month in Nature.",
    ]
    return "\n".join(random.sample(findings, k=random.randint(2, 4)))


@tool_async(manager)
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment and key themes in a piece of text."""
    delay = random.uniform(10, 20)
    time.sleep(delay)
    sentiments = [
        "overwhelmingly positive",
        "cautiously optimistic",
        "mixed but trending positive",
        "neutral with some concerns",
    ]
    themes = [
        "innovation",
        "market disruption",
        "sustainability",
        "cost efficiency",
        "regulatory compliance",
    ]
    return (
        f"Sentiment: {random.choice(sentiments)}\n"
        f"Key themes: {', '.join(random.sample(themes, k=3))}\n"
        f"Confidence: {random.randint(75, 98)}%\n"
        f"Sample size: {random.randint(500, 5000)} data points analyzed"
    )


@tool_async(manager)
def fetch_weather(city: str) -> str:
    """Get the current weather for a city."""
    delay = random.uniform(20, 50)
    time.sleep(delay)
    conditions = ["sunny", "partly cloudy", "overcast", "light rain", "clear skies"]
    return (
        f"Weather in {city}: {random.choice(conditions)}\n"
        f"Temperature: {random.randint(5, 35)}C\n"
        f"Humidity: {random.randint(30, 90)}%\n"
        f"Wind: {random.randint(5, 40)} km/h"
    )


# ---- Sync tool ----


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression and return the result immediately."""
    allowed = {"abs": abs, "min": min, "max": max, "round": round, "int": int, "float": float}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant with both synchronous and asynchronous tools.

ASYNC TOOLS (run in background, results arrive later):
  research_topic, analyze_sentiment, fetch_weather

When you call an async tool it returns a task ID immediately.
The actual result will arrive in a future message tagged [ASYNC RESULT].
Rules:
  - Do NOT guess or fabricate async results. Wait for [ASYNC RESULT].
  - Tell the user each task has been started.
  - You CAN dispatch multiple async tools at once.

SYNC TOOLS (return immediately):
  calculate

When you receive an [ASYNC RESULT]:
  - Summarize the result naturally for the user.
  - If tasks are still pending, mention you are waiting.
  - Once all results are in, give a cohesive summary.

Keep responses concise."""

# ---------------------------------------------------------------------------
# TUI App
# ---------------------------------------------------------------------------

EVENT_STYLES = {
    "callback": "bold green",
    "queued": "bold yellow",
    "draining": "bold blue",
    "thinking": "bold dim",
    "done": "dim",
}


class AsyncToolsTUI(App):
    """Interactive TUI for experimenting with async Strands tools."""

    TITLE = "Strands Async Tools"
    SUB_TITLE = "callback-driven async demo"

    CSS = """
    Screen {
        layout: vertical;
    }
    #chat {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    #status {
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 2;
    }
    #user-input {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.async_agent: AsyncAgent | None = None
        self._agent_thread: threading.Thread | None = None

    # ---- Layout ----

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat", highlight=True, markup=True, auto_scroll=True, wrap=True)
        yield Static("Initializing agent...", id="status")
        yield Input(placeholder="Type a message and press Enter...", id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).disabled = True
        threading.Thread(target=self._init_agent, daemon=True).start()

    # ---- Agent init (runs in background thread) ----

    def _init_agent(self) -> None:
        model_id = os.environ.get("STRANDS_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

        agent = Agent(
            model=model_id,
            system_prompt=SYSTEM_PROMPT,
            tools=[research_topic, analyze_sentiment, fetch_weather, calculate],
        )

        self.async_agent = AsyncAgent(
            agent=agent,
            manager=manager,
            on_response=self._on_agent_response,
            on_status=self._on_agent_status,
        )

        self.call_from_thread(self._on_agent_ready, model_id)

    def _on_agent_ready(self, model_id: str) -> None:
        self._log_system(
            f"Agent ready  |  model: {model_id}\n"
            "Async tools: research_topic, analyze_sentiment, fetch_weather\n"
            "Sync tools:  calculate\n"
            'Try: "Research quantum computing and check the weather in Tokyo"'
        )
        self._set_status("Ready")
        inp = self.query_one("#user-input", Input)
        inp.disabled = False
        inp.focus()

    # ---- Callbacks from AsyncAgent (called from background threads) ----

    def _on_agent_response(self, text: str) -> None:
        self.call_from_thread(self._log_agent, text)
        self.call_from_thread(self._refresh_status)

    def _on_agent_status(self, event_type: str, message: str) -> None:
        if event_type == "done":
            self.call_from_thread(self._on_agent_done)
            return
        if event_type == "thinking":
            self.call_from_thread(self._refresh_status)
            return
        self.call_from_thread(self._log_event, event_type, message)
        self.call_from_thread(self._refresh_status)

    def _on_agent_done(self) -> None:
        self._refresh_status()
        inp = self.query_one("#user-input", Input)
        inp.disabled = False
        inp.focus()

    # ---- User input ----

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self.async_agent is None:
            return

        event.input.clear()
        event.input.disabled = True
        self._log_user(text)

        # Special commands
        if text.lower() in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if text.lower() == "/clear":
            self.query_one("#chat", RichLog).clear()
            event.input.disabled = False
            return

        self._set_status("Agent thinking...")
        threading.Thread(target=self._process_message, args=(text,), daemon=True).start()

    def _process_message(self, message: str) -> None:
        try:
            assert self.async_agent is not None
            self.async_agent.send(message)
        except Exception as e:
            self.call_from_thread(self._log_system, f"Error: {e}")
            self.call_from_thread(self._on_agent_done)

    # ---- UI helpers ----

    def _log_user(self, text: str) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.write(Text.from_markup(f"[bold magenta]YOU:[/bold magenta] {self._escape(text)}"))
        chat.write(Text(""))

    def _log_agent(self, text: str) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.write(Text.from_markup(f"[bold cyan]AGENT:[/bold cyan] {self._escape(text)}"))
        chat.write(Text(""))

    def _log_event(self, event_type: str, message: str) -> None:
        style = EVENT_STYLES.get(event_type, "white")
        chat = self.query_one("#chat", RichLog)
        chat.write(Text.from_markup(f"  [{style}]\\[{event_type}][/{style}] {self._escape(message)}"))

    def _log_system(self, text: str) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.write(Text.from_markup(f"[dim]{self._escape(text)}[/dim]"))
        chat.write(Text(""))

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _refresh_status(self) -> None:
        pending = manager.pending_count
        busy = self.async_agent is not None and self.async_agent.is_busy
        if busy and pending > 0:
            self._set_status(f"Agent thinking...  |  {pending} async task(s) pending")
        elif busy:
            self._set_status("Agent thinking...")
        elif pending > 0:
            self._set_status(f"Waiting for {pending} async task(s)...")
        else:
            self._set_status("Ready")

    @staticmethod
    def _escape(text: str) -> str:
        """Escape Rich markup characters in user/agent text."""
        return text.replace("[", "\\[")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = AsyncToolsTUI()
    app.run()
