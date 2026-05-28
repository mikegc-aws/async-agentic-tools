"""Async agent demo — tools + simple CLI (input a line, get responses).

Run:
    uv run python demo.py

Requires AWS credentials for Bedrock. STRANDS_MODEL env var overrides the model.
"""

import os
import random
import time

from strands import Agent

from strands_async_tools import AsyncAgent, AsyncToolManager, tool_async
from strands_tools.calculator import calculator
from strands_tools.current_time import current_time

# ---------------------------------------------------------------------------
# Async delay bounds (seconds) — simulated latency per tool
# ---------------------------------------------------------------------------

DELAY_MIN, DELAY_MAX = 10.0, 20.0

# ---------------------------------------------------------------------------
# Async Tool Manager
# ---------------------------------------------------------------------------

manager = AsyncToolManager(max_workers=4)

# ---------------------------------------------------------------------------
# Async Tools — these run in background threads, results arrive via callback
# ---------------------------------------------------------------------------


@tool_async(manager)
def research_topic(topic: str, report_progress=None) -> str:
    """Research a topic thoroughly and return detailed findings."""
    steps = [
        "Searching academic databases...",
        "Analyzing market reports...",
        "Cross-referencing sources...",
        "Synthesizing findings...",
    ]
    total = len(steps)
    for i, step in enumerate(steps):
        report_progress(i, total, step)
        time.sleep(random.uniform(DELAY_MIN / total, DELAY_MAX / total))
    report_progress(total, total, "Research complete")
    findings = [
        f"Key finding: {topic} has seen 340% growth in the last 2 years.",
        f"Major players in {topic} include Acme Corp, Nexus Labs, and Orion Systems.",
        f"Experts predict the {topic} market will reach $50B by 2028.",
        f"Recent regulatory changes may impact {topic} adoption in the EU.",
        f"A breakthrough paper on {topic} was published last month in Nature.",
    ]
    return "\n".join(random.sample(findings, k=random.randint(2, 4)))


@tool_async(manager)
def analyze_sentiment(text: str, report_progress=None) -> str:
    """Analyze the sentiment and key themes in a piece of text."""
    report_progress(0, 3, "Tokenizing input...")
    time.sleep(random.uniform(DELAY_MIN / 3, DELAY_MAX / 3))
    report_progress(1, 3, "Running sentiment model...")
    time.sleep(random.uniform(DELAY_MIN / 3, DELAY_MAX / 3))
    report_progress(2, 3, "Extracting themes...")
    time.sleep(random.uniform(DELAY_MIN / 3, DELAY_MAX / 3))
    report_progress(3, 3, "Analysis complete")
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
def fetch_weather(city: str, report_progress=None) -> str:
    """Get the current weather for a city."""
    report_progress(0, 2, f"Contacting weather service for {city}...")
    time.sleep(random.uniform(DELAY_MIN / 2, DELAY_MAX / 2))
    report_progress(1, 2, "Parsing forecast data...")
    time.sleep(random.uniform(DELAY_MIN / 2, DELAY_MAX / 2))
    report_progress(2, 2, "Done")
    conditions = ["sunny", "partly cloudy", "overcast", "light rain", "clear skies"]
    return (
        f"Weather in {city}: {random.choice(conditions)}\n"
        f"Temperature: {random.randint(5, 35)}C\n"
        f"Humidity: {random.randint(30, 90)}%\n"
        f"Wind: {random.randint(5, 40)} km/h"
    )


# ---------------------------------------------------------------------------
# System Prompt
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
  - You CAN dispatch multiple async tools at once — they run in parallel.

SYNC TOOLS (return immediately):
  calculator, current_time

When you receive an [ASYNC RESULT]:
  - Summarize the result naturally for the user.
  - If tasks are still pending, mention you are waiting.
  - Once all results are in, give a cohesive summary.

Keep responses concise."""


def main() -> None:
    model_id = os.environ.get("STRANDS_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

    print("Async agent — async tools (research, sentiment, weather) + sync (calculator, current_time).")
    print("Type a message and press Enter. quit / exit / q to exit.\n")

    agent = Agent(
        model=model_id,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_topic, analyze_sentiment, fetch_weather, calculator, current_time],
    )
    async_agent = AsyncAgent(agent=agent, manager=manager)

    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        async_agent.send(line)

    manager.shutdown()
    print("Bye.")


if __name__ == "__main__":
    main()
