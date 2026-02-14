"""AsyncAgent — callback-driven wrapper for Strands Agent."""

import threading
from typing import Any, Callable

from strands import Agent

from .manager import AsyncToolManager, AsyncTaskResult


class AsyncAgent:
    """Wraps a Strands Agent to support callback-driven async tool results.

    When an async tool completes:
    - If the agent is idle -> immediately invoke the agent with the result.
    - If the agent is busy -> queue the result; deliver when the agent finishes.

    Callbacks:
        on_response(text)          — called with each agent response
        on_status(event_type, msg) — called for lifecycle events:
            "callback"  — async task completed, delivering to agent
            "queued"    — async task completed, agent busy, queued
            "draining"  — delivering a previously queued result
            "thinking"  — agent invocation started
            "done"      — agent invocation chain finished
    """

    def __init__(
        self,
        agent: Agent,
        manager: AsyncToolManager,
        on_response: Callable[[str], None] | None = None,
        on_status: Callable[[str, str], None] | None = None,
    ):
        self.agent = agent
        self.manager = manager
        self.on_response = on_response or self._default_on_response
        self.on_status = on_status or self._default_on_status
        self._busy = False
        self._lock = threading.Lock()
        self._queued_results: list[AsyncTaskResult] = []

        # Register ourselves as the completion callback
        manager.on_complete = self._on_task_complete

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    @staticmethod
    def _default_on_response(text: str) -> None:
        print(f"\n\033[36mAGENT:\033[0m {text}\n")

    @staticmethod
    def _default_on_status(event_type: str, message: str) -> None:
        colors = {"callback": "32", "queued": "33", "draining": "34", "thinking": "90", "done": "90"}
        color = colors.get(event_type, "0")
        print(f"\n  \033[{color}m[{event_type}]\033[0m {message}")

    # ---- Callback handling ----

    def _on_task_complete(self, result: AsyncTaskResult) -> None:
        """Called from the thread pool when an async task finishes."""
        status = (
            f"FAILED: {result.error}"
            if result.error
            else f"completed in {result.elapsed_ms:.0f}ms"
        )

        with self._lock:
            if self._busy:
                self._queued_results.append(result)
                self.on_status(
                    "queued",
                    f"{result.tool_name} ({result.task_id}) {status} — agent busy, delivering next turn",
                )
                return

        self.on_status(
            "callback",
            f"{result.tool_name} ({result.task_id}) {status} — delivering to agent now",
        )
        self._invoke(self._format_result(result))

    # ---- Formatting ----

    @staticmethod
    def _format_result(result: AsyncTaskResult) -> str:
        args_summary = ", ".join(f"{k}={v!r}" for k, v in result.kwargs.items())
        if result.error:
            return (
                f"[ASYNC RESULT — FAILED]\n"
                f"Task ID: {result.task_id}\n"
                f"Tool: {result.tool_name}({args_summary})\n"
                f"Error: {result.error}\n"
                f"Elapsed: {result.elapsed_ms:.0f}ms"
            )
        return (
            f"[ASYNC RESULT]\n"
            f"Task ID: {result.task_id}\n"
            f"Tool: {result.tool_name}({args_summary})\n"
            f"Result:\n{result.result}\n"
            f"Elapsed: {result.elapsed_ms:.0f}ms"
        )

    # ---- Agent invocation ----

    def _invoke(self, prompt: str) -> None:
        """Run the agent with a prompt. Iteratively drains queued results."""
        with self._lock:
            self._busy = True

        self.on_status("thinking", "processing...")

        to_process = [prompt]

        while to_process:
            current = to_process.pop(0)
            try:
                response = self.agent(current)
                self.on_response(str(response))
            except Exception as e:
                self.on_response(f"[Agent error: {e}]")

            # Drain any results that arrived while we were busy
            with self._lock:
                queued = list(self._queued_results)
                self._queued_results.clear()

            for r in queued:
                status = (
                    f"FAILED: {r.error}"
                    if r.error
                    else f"completed in {r.elapsed_ms:.0f}ms"
                )
                self.on_status(
                    "draining",
                    f"{r.tool_name} ({r.task_id}) {status} — delivering queued result",
                )
                to_process.append(self._format_result(r))

        with self._lock:
            self._busy = False

        self.on_status("done", "")

    def send(self, message: str) -> None:
        """Send a user message to the agent."""
        # Collect any results that arrived while idle
        with self._lock:
            queued = list(self._queued_results)
            self._queued_results.clear()

        prompt = message
        if queued:
            results_text = "\n\n".join(self._format_result(r) for r in queued)
            prompt = (
                f"The following async results arrived while you were idle:\n\n"
                f"{results_text}\n\nNew user message: {message}"
            )

        self._invoke(prompt)
