"""Demo: Async tool calling with callback-driven delivery in Strands Agents.

Run:
    uv run python demo.py

Requires AWS credentials configured for Bedrock access.
Override the model with STRANDS_MODEL env var.
"""

import os
import random
import time

from strands import Agent, tool

from strands_async_tools import AsyncAgent, AsyncToolManager, tool_async

# ---------------------------------------------------------------------------
# Async Tool Manager
# ---------------------------------------------------------------------------

manager = AsyncToolManager(max_workers=4)

# ---------------------------------------------------------------------------
# Async Tools — these run in background threads, results arrive via callback
# ---------------------------------------------------------------------------


@tool_async(manager)
def research_topic(topic: str) -> str:
    """Research a topic thoroughly and return detailed findings."""
    delay = random.uniform(3, 8)
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
    delay = random.uniform(5, 12)
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
    delay = random.uniform(2, 5)
    time.sleep(delay)
    conditions = ["sunny", "partly cloudy", "overcast", "light rain", "clear skies"]
    return (
        f"Weather in {city}: {random.choice(conditions)}\n"
        f"Temperature: {random.randint(5, 35)}C\n"
        f"Humidity: {random.randint(30, 90)}%\n"
        f"Wind: {random.randint(5, 40)} km/h"
    )


# ---------------------------------------------------------------------------
# Sync Tool — for contrast, returns immediately
# ---------------------------------------------------------------------------


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression and return the result immediately."""
    allowed = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "int": int,
        "float": float,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


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
  calculate

When you receive an [ASYNC RESULT]:
  - Summarize the result naturally for the user.
  - If tasks are still pending, mention you are waiting.
  - Once all results are in, give a cohesive summary.

Keep responses concise."""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    model_id = os.environ.get("STRANDS_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

    print("=" * 70)
    print("  Strands Agents — Async Tool Calling Demo (Callback-Driven)")
    print("=" * 70)
    print(f"  Model : {model_id}")
    print(f"  Async : research_topic, analyze_sentiment, fetch_weather")
    print(f"  Sync  : calculate")
    print("=" * 70)

    agent = Agent(
        model=model_id,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_topic, analyze_sentiment, fetch_weather, calculate],
    )

    async_agent = AsyncAgent(agent=agent, manager=manager)

    user_msg = (
        "Research quantum computing, analyze the sentiment of "
        "'AI is transforming every industry and creating unprecedented "
        "opportunities for growth', check the weather in Tokyo, "
        "and calculate 42 * 17."
    )

    print(f"\n\033[35mUSER:\033[0m {user_msg}\n")

    async_agent.send(user_msg)

    # Keep alive while background tasks complete and callbacks fire
    print("\n\033[90m[waiting for remaining async callbacks...]\033[0m")
    while manager.pending_count > 0:
        time.sleep(0.5)

    # Allow final queue drain
    time.sleep(3)

    print("\n" + "=" * 70)
    print("  All async tasks completed. Demo finished.")
    print("=" * 70)

    manager.shutdown()


if __name__ == "__main__":
    main()
