"""Simple CLI for the async agent — input a line, get responses. No TUI.

Run:
    uv run python cli.py

Requires AWS credentials for Bedrock. STRANDS_MODEL env var overrides the model.
"""

import os

from strands import Agent

from strands_async_tools import AsyncAgent

from demo import (
    SYSTEM_PROMPT,
    calculate,
    analyze_sentiment,
    fetch_weather,
    research_topic,
    manager,
)


def main() -> None:
    model_id = os.environ.get("STRANDS_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

    print("Async agent CLI — async tools (research, sentiment, weather) + sync (calculate).")
    print("Type a message and press Enter. quit / exit / q to exit.\n")

    agent = Agent(
        model=model_id,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_topic, analyze_sentiment, fetch_weather, calculate],
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
