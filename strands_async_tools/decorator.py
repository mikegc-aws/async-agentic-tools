"""The @tool_async decorator for Strands Agents."""

import functools
from typing import Any, Callable

from strands import tool

from .manager import AsyncToolManager


def tool_async(manager: AsyncToolManager) -> Callable:
    """Decorator: wraps a function as an async Strands tool.

    The decorated function is dispatched to a background thread via the manager.
    It returns immediately with a task ID. The actual result is delivered later
    through the manager's on_complete callback.

    Usage::

        manager = AsyncToolManager()

        @tool_async(manager)
        def slow_research(topic: str) -> str:
            '''Research a topic thoroughly.'''
            time.sleep(5)
            return f"Findings about {topic}..."
    """

    def decorator(fn: Callable) -> Any:
        original_doc = fn.__doc__ or fn.__name__

        # Build the async notice that gets appended to the docstring.
        # The model sees this and knows not to fabricate results.
        async_notice = (
            "\n\nIMPORTANT: This is an ASYNC tool that runs in the background. "
            "It returns immediately with a task ID. The actual result will be "
            "delivered to you in a future turn as an [ASYNC RESULT] message. "
            "Do NOT guess, fabricate, or assume the result. "
            "Acknowledge the task is running and continue with other work."
        )

        # functools.wraps copies __name__, __annotations__, __wrapped__ etc.
        # from the original function. inspect.signature() on the wrapper will
        # follow __wrapped__ and return the original's signature, so Strands'
        # @tool decorator builds the correct parameter schema.
        @functools.wraps(fn)
        def wrapper(**kwargs: Any) -> str:
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
        # This must happen after @functools.wraps (which copies fn's doc)
        # but before @tool (which reads it for the tool description).
        wrapper.__doc__ = original_doc + async_notice

        # Apply Strands' @tool decorator to produce a DecoratedFunctionTool.
        return tool(wrapper)

    return decorator
