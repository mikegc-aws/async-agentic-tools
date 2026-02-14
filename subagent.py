"""Subagent for handling complex tasks autonomously.

A Strands Agent equipped with real tools (web search, file I/O, HTTP, calculator,
editor) that runs in the background via @tool_async. The main Nova Sonic agent
delegates tasks here through the handle_task tool.

Requires:
    TAVILY_API_KEY  - Tavily API key for web search/extract
    SUBAGENT_MODEL  - Bedrock model ID (default: us.anthropic.claude-sonnet-4-20250514-v1:0)
    AWS_REGION      - AWS region for Bedrock (default: us-east-1)
"""

import os

from strands import Agent

from strands_async_tools import AsyncToolManager, tool_async

# Tools for the subagent (imported as modules — Strands Agent discovers tools inside)
from strands_tools import calculator, editor, file_read, file_write, http_request, tavily

# Bypass tool consent for autonomous operation (file_write, editor need this
# when running non-interactively — there's no human to click "confirm").
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ---------------------------------------------------------------------------
# Async Tool Manager — shared with voice.py
# ---------------------------------------------------------------------------

manager = AsyncToolManager(max_workers=4)

# ---------------------------------------------------------------------------
# Subagent definition
# ---------------------------------------------------------------------------

SUBAGENT_SYSTEM_PROMPT = """\
You are an autonomous task-execution agent. You receive a task and complete it \
using your available tools. You CANNOT ask questions or request clarification — \
work with what you have.

Available tools:
- calculator: Evaluate math expressions (SymPy-powered, supports calculus, \
equation solving, matrix ops)
- editor: View, create, and edit files (str_replace, insert, undo)
- file_read: Read file contents (view, find, search, stats, diff modes)
- file_write: Write content to files
- http_request: Make HTTP requests to APIs (all methods, auth support)
- tavily_search: Search the web for current information
- tavily_extract: Extract content from web pages

File operations are scoped to the workspace/ directory. Always use paths \
under workspace/ (e.g., workspace/notes.txt, workspace/output.json).

RESPONSE FORMAT — THIS IS CRITICAL:
Your response will be READ ALOUD. It MUST be under 30 words. \
One short sentence. No names, no lists, no examples, no details. \
Just the core answer or "Done, saved to workspace/filename.txt". \
Anything over 30 words is a failure."""


def _create_subagent() -> Agent:
    """Create a fresh subagent instance with real tools."""
    model_id = os.environ.get(
        "SUBAGENT_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    return Agent(
        model=model_id,
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        tools=[calculator, editor, file_read, file_write, http_request, tavily],
    )


@tool_async(manager)
def handle_task(task: str) -> str:
    """Handle a complex task by delegating to a capable subagent.

    The subagent has access to web search, file operations, HTTP requests,
    a calculator, and a code editor. Use this for any task that requires
    research, file manipulation, or multi-step reasoning.
    """
    agent = _create_subagent()
    result = agent(task)
    return str(result)
