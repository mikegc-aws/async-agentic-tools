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
from strands.agent.conversation_manager import SlidingWindowConversationManager

from strands_async_tools import AsyncToolManager, tool_async

# Tools for the subagent (imported as modules — Strands Agent discovers tools inside)
from strands_tools import calculator, editor, file_read, file_write, http_request, tavily

# Bypass tool consent for autonomous operation (file_write, editor need this
# when running non-interactively — there's no human to click "confirm").
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ---------------------------------------------------------------------------
# Async Tool Manager — shared with voice.py
# ---------------------------------------------------------------------------

manager = AsyncToolManager(max_workers=8)

# ---------------------------------------------------------------------------
# Subagent definition
# ---------------------------------------------------------------------------

SUBAGENT_SYSTEM_PROMPT = """\
You are a background task agent. You receive a single task and complete it with minimal tool use. You cannot ask the user anything — work with what you are given.

CRITICAL — keep it short and stop quickly:
- Use at most 2–3 tool calls total (e.g. one tavily_search, then stop and summarize). Do not chain many searches or reads.
- As soon as you have enough to answer the question, write your summary and stop. Do not keep researching or reading more.
- Your final reply must be a brief spoken summary (2–4 sentences). The user will hear it. Do not paste long excerpts or full articles into your reply.
- If you find a lot of detail, write it to a file in workspace/ and return only the short summary plus "More in workspace/filename". Do not put long content in your reply text.

Available tools:
- calculator: Evaluate math expressions
- editor: View, create, and edit files
- file_read: Read file contents (view, find, search modes)
- file_write: Write content to files
- http_request: Make HTTP requests
- tavily_search: Search the web (use once, then summarize from the snippet/results)
- tavily_extract: Extract from a single page only if essential

All file paths must be under workspace/ (e.g. workspace/notes.txt).

WORKFLOW:
1. Use one or two tool calls (e.g. tavily_search with a focused query). Do not run many searches or open many pages.
2. From the first useful result, form a 2–4 sentence answer and return it. If the answer is long, write the full content to workspace/<sensible_name>.md and return only the summary + "More in workspace/<sensible_name>.md".
3. Stop as soon as you have returned your summary. Do not add more tool calls."""


def _create_subagent() -> Agent:
    """Create a fresh subagent instance with real tools and a small context window."""
    model_id = os.environ.get(
        "SUBAGENT_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    # Small window + per_turn trimming so the subagent doesn't fill context with long tool chains
    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
        per_turn=True,
    )
    return Agent(
        model=model_id,
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        tools=[calculator, editor, file_read, file_write, http_request, tavily],
        conversation_manager=conversation_manager,
    )


@tool_async(manager)
def handle_task(task: str) -> str:
    """Only call this AFTER you have already called file_read with path "workspace/" and mode "find" and confirmed no relevant file exists. Run a task in the background (web research, current info, long work). You get the result later in [ASYNC RESULT] as SUBAGENT ANSWER. The subagent may write "More in workspace/filename" — use file_read on that file only if the user asks for more.
    """
    agent = _create_subagent()
    result = agent(task)
    return str(result)
