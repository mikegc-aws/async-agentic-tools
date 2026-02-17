"""Synchronous agent demo — regular tools + simple CLI (input a line, get responses).

Run:
    uv run python demo_sync.py

Requires AWS credentials for Bedrock. STRANDS_MODEL env var overrides the model.
"""

import os
import random
import time

from strands import Agent, tool

from strands_tools.calculator import calculator
from strands_tools.current_time import current_time

# ---------------------------------------------------------------------------
# Delay bounds (seconds) — simulated latency per tool
# ---------------------------------------------------------------------------

DELAY_MIN, DELAY_MAX = 10.0, 20.0

# ---------------------------------------------------------------------------
# Synchronous Tools — these block until completion
# ---------------------------------------------------------------------------


@tool
def research_topic(topic: str) -> str:
    """Research a topic thoroughly and return detailed findings."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(delay)
    findings = [
        f"Key finding: {topic} has seen 340% growth in the last 2 years.",
        f"Major players in {topic} include Acme Corp, Nexus Labs, and Orion Systems.",
        f"Experts predict the {topic} market will reach $50B by 2028.",
        f"Recent regulatory changes may impact {topic} adoption in the EU.",
        f"A breakthrough paper on {topic} was published last month in Nature.",
    ]
    return "\n".join(random.sample(findings, k=random.randint(2, 4)))


@tool
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment and key themes in a piece of text."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
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


@tool
def fetch_weather(city: str) -> str:
    """Get the current weather for a city."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(delay)
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
You are a helpful assistant with synchronous tools.

AVAILABLE TOOLS:
  research_topic, analyze_sentiment, fetch_weather, calculator, current_time

All tools are synchronous and will block until they complete.
When you call a tool, you must wait for its result before proceeding.

Keep responses concise."""


def main() -> None:
    model_id = os.environ.get("STRANDS_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

    print("Synchronous agent — all tools block until completion.")
    print("Type a message and press Enter. quit / exit / q to exit.\n")

    agent = Agent(
        model=model_id,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_topic, analyze_sentiment, fetch_weather, calculator, current_time],
    )

    while True:
        try:
            line = input("\nYou: ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break

        # Send message and get response
        agent(line)

    print("Bye.")


if __name__ == "__main__":
    main()

