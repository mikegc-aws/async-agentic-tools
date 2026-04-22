<div align="center">

# 🧰⛓️‍💥 async-agentic-tools

[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

[![GitHub stars](https://img.shields.io/github/stars/mikegc-aws/async-agentic-tools.svg)](https://github.com/mikegc-aws/async-agentic-tools/stargazers)
[![License](https://img.shields.io/github/license/mikegc-aws/async-agentic-tools.svg)](https://github.com/mikegc-aws/async-agentic-tools/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://python.org)

True asynchronous agentic tools — the model dispatches a tool, gets an immediate acknowledgement, and keeps talking. Results are delivered via callback when they complete — no blocked loops, no dead air.

</div>

This is **not** parallel tool calling (which most agent frameworks already support). Parallel tool calling still blocks the agent loop until every tool in the batch returns. This is true async: the model stays responsive while tools run in the background, and results stream in as they finish.

The demo is built on [Strands Agents](https://github.com/strands-agents/sdk-python), but the pattern applies to any agent framework with a tool-calling loop.

Read the [blog post](https://blog.mikegchambers.com/posts/async-agentic-tools/) for the full explanation of the problem and how this works.

**Quick walkthrough video here:** "Do async tool calls work now???"
[![Watch the video](https://img.youtube.com/vi/VYLBCoxbPE8/maxresdefault.jpg)](https://youtu.be/VYLBCoxbPE8)

## How it works

Three small components layer on top of a standard Strands Agent:

- **`@tool_async(manager)`** — Decorator that wraps any tool function. The tool is submitted to a background thread and returns a task ID immediately. Your tool code doesn't change at all.
- **`AsyncToolManager`** — Manages a thread pool, tracks pending tasks, and fires a callback when each one completes.
- **`AsyncAgent`** — Wraps a Strands `Agent` to handle result delivery. If the agent is idle when a result arrives, it's delivered immediately. If the agent is busy, results queue up and drain when it finishes.

```python
from strands import Agent
from strands_async_tools import AsyncAgent, AsyncToolManager, tool_async

manager = AsyncToolManager(max_workers=4)

@tool_async(manager)
def slow_research(topic: str) -> str:
    """Research a topic thoroughly."""
    # This runs in a background thread — takes as long as it needs
    time.sleep(15)
    return f"Findings about {topic}..."

agent = Agent(model=model_id, tools=[slow_research])
async_agent = AsyncAgent(agent=agent, manager=manager)
async_agent.send("Research quantum computing")
```

The framework is about 320 lines of Python across three files in `strands_async_tools/`.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended)
- AWS credentials configured for [Amazon Bedrock](https://aws.amazon.com/bedrock/)

## CLI demo

An interactive chat with three simulated async tools (10-20s delays each) and two synchronous tools from `strands-agents-tools` (calculator, current_time).

```bash
git clone https://github.com/mikegc-aws/async-agentic-tools
cd async-agentic-tools
uv run python demo.py
```

The default model is Claude Sonnet on Bedrock. Override with the `STRANDS_MODEL` env var.

Try something like:

```
You: Research Paris

  [thinking] processing...
I've started researching Paris for you (Task a1b2c3).
I'll let you know as soon as the results come in.

You: What time is it there?

  [thinking] processing...
It's currently 15:32 in Paris (CET, UTC+1).

  [callback] research_topic (a1b2c3) completed in 16482ms — delivering to agent now
  [thinking] processing...
The Paris research just came back! Here are some highlights:
- ...
```

The async tool dispatches to a background thread and the agent keeps talking. The sync tool (current_time) returns instantly — and the agent knows "there" means Paris. When the research finishes, the result is delivered via callback and the agent speaks it.

## Voice mode (experimental)

The `voice/` folder contains a voice interface using **Amazon Nova Sonic** (bidirectional streaming voice model). The agent talks to you through your speakers and listens through your microphone. While you chat, it can delegate complex tasks (web research, file I/O) to a background subagent via `@tool_async`. Results are injected back into the voice stream — the agent speaks them to you when they're ready.

### Voice requirements

- AWS credentials with Bedrock access in `us-east-1`
- Microphone and speakers
- Model access for `amazon.nova-2-sonic-v1:0`
- **Tavily API key** (optional but recommended) — the subagent uses [Tavily](https://tavily.com/) for web search and extraction. You can get a free API key at [app.tavily.com](https://app.tavily.com/). Without it, the voice agent works but can't do web searches.

### Running voice mode

From the repo root:

```bash
# With web search (recommended)
TAVILY_API_KEY=tvly-your-key-here uv run python -m voice

# Without web search (still works, just no web research)
uv run python -m voice
```

Speak into your microphone. The agent responds through your speakers with echo cancellation (LiveKit WebRTC APM). Press `Ctrl+C` to quit.

### Voice env vars

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `NOVA_SONIC_VOICE` | `tiffany` | Nova Sonic voice name |
| `NOVA_SONIC_MODEL` | `amazon.nova-2-sonic-v1:0` | Nova Sonic model ID |
| `SUBAGENT_MODEL` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | Subagent Bedrock model |
| `TAVILY_API_KEY` | _(none)_ | Tavily API key for web search |
| `LOG_FILE` | `voice_debug.log` | Debug log path |
| `LOG_LEVEL` | `WARNING` | Console log level |

See [voice/README.md](voice/README.md) for more detail on the voice architecture.

## Project structure

```
strands_async_tools/          # The framework (3 files, ~320 lines)
  manager.py                  # AsyncToolManager — thread pool + callbacks
  decorator.py                # @tool_async — decorator for async tools
  agent.py                    # AsyncAgent — callback-driven result delivery

demo.py                       # Interactive CLI demo (simulated async tools)
demo_sync.py                  # Synchronous comparison demo

voice/                        # Experimental voice interface
  voice.py                    # BidiAgent + Nova Sonic + async result injection
  subagent.py                 # Background agent with real tools (web search, file I/O)
  echo_cancel.py              # Echo cancellation (LiveKit WebRTC APM)

```
