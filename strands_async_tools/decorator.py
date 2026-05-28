"""The @tool_async decorator for Strands Agents.

Supports MCP-compatible progress reporting: decorated tools receive a
`report_progress(progress, total=None, message=None)` callback they can
call during execution to emit progress notifications.
"""

import functools
import inspect
from typing import Any, Callable

from strands import tool

from .manager import AsyncToolManager


def tool_async(manager: AsyncToolManager) -> Callable:
    """Decorator: wraps a function as an async Strands tool.

    The decorated function is dispatched to a background thread via the manager.
    It returns immediately with a task ID. The actual result is delivered later
    through the manager's on_complete callback.

    If the decorated function accepts a `report_progress` parameter, it will
    receive a callback matching MCP's progress notification interface::

        def report_progress(progress: float, total: float | None = None, message: str | None = None) -> None

    Usage::

        manager = AsyncToolManager()

        @tool_async(manager)
        def slow_research(topic: str, report_progress=None) -> str:
            '''Research a topic thoroughly.'''
            report_progress(0, 3, "Starting research...")
            time.sleep(5)
            report_progress(1, 3, "Found sources")
            time.sleep(5)
            report_progress(2, 3, "Synthesizing")
            time.sleep(5)
            report_progress(3, 3, "Done")
            return f"Findings about {topic}..."
    """

    def decorator(fn: Callable) -> Any:
        original_doc = fn.__doc__ or fn.__name__

        async_notice = (
            "\n\nIMPORTANT: This is an ASYNC tool that runs in the background. "
            "It returns immediately with a task ID. The actual result will be "
            "delivered to you in a future turn as an [ASYNC RESULT] message. "
            "Do NOT guess, fabricate, or assume the result. "
            "Acknowledge the task is running and continue with other work."
        )

        # Check if the original function accepts report_progress
        sig = inspect.signature(fn)
        accepts_progress = "report_progress" in sig.parameters

        # functools.wraps copies __name__, __annotations__, __wrapped__ etc.
        # from the original function. inspect.signature() on the wrapper will
        # follow __wrapped__ and return the original's signature, so Strands'
        # @tool decorator builds the correct parameter schema.
        @functools.wraps(fn)
        def wrapper(**kwargs: Any) -> str:
            # Strip report_progress from kwargs before submitting — the manager
            # injects its own. If the tool doesn't accept it, the manager's
            # injected callback will be consumed internally.
            kwargs.pop("report_progress", None)
            task_id = manager.submit(fn.__name__, fn, **kwargs)
            args_summary = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            return (
                f"[ASYNC TASK SUBMITTED]\n"
                f"Task ID: {task_id}\n"
                f"Tool: {fn.__name__}({args_summary})\n"
                f"Status: Running in background.\n"
                f"The result will be delivered in a future message. "
                f"Do NOT fabricate or guess the result."
            )

        # Override the docstring with the async notice appended.
        wrapper.__doc__ = original_doc + async_notice

        # Apply Strands' @tool decorator to produce a DecoratedFunctionTool.
        return tool(wrapper)

    return decorator
