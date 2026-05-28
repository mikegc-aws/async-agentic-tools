"""Async tool manager — background dispatch, progress reporting, and completion callbacks.

Implements the MCP Task lifecycle (working → completed/failed/cancelled) and
MCP progress notifications (progress, total, message) for long-running tools.
"""

import inspect
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TaskStatus(Enum):
    """MCP Task lifecycle states (per 2025-11-25 spec)."""

    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """MCP-compatible progress notification data."""

    task_id: str
    tool_name: str
    progress: float
    total: float | None = None
    message: str | None = None


@dataclass
class AsyncTaskResult:
    """Result of a completed async tool invocation."""

    task_id: str
    tool_name: str
    kwargs: dict
    result: Any
    error: str | None
    elapsed_ms: float
    status: TaskStatus = TaskStatus.COMPLETED


@dataclass
class TaskState:
    """Internal tracking for a running task."""

    task_id: str
    tool_name: str
    kwargs: dict
    status: TaskStatus = TaskStatus.WORKING
    future: Future | None = None
    created_at: float = field(default_factory=time.monotonic)
    last_progress: TaskProgress | None = None


class AsyncToolManager:
    """Manages async tool dispatch, progress reporting, and result delivery.

    Submit functions for background execution. During execution, tools can
    report progress via a callback. When they complete, the on_complete
    callback fires with the result.

    Callbacks:
        on_complete(result)   — fired when a task reaches a terminal state
        on_progress(progress) — fired when a tool reports progress mid-execution
    """

    def __init__(
        self,
        max_workers: int = 4,
        on_complete: Callable[[AsyncTaskResult], None] | None = None,
        on_progress: Callable[[TaskProgress], None] | None = None,
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskState] = {}
        self._lock = threading.Lock()
        self.on_complete = on_complete
        self.on_progress = on_progress

    def _report_progress(
        self,
        task_id: str,
        tool_name: str,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """Called by running tools to report progress. Thread-safe."""
        notification = TaskProgress(
            task_id=task_id,
            tool_name=tool_name,
            progress=progress,
            total=total,
            message=message,
        )
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.last_progress = notification
        if self.on_progress:
            self.on_progress(notification)

    def submit(self, tool_name: str, fn: Callable, **kwargs: Any) -> str:
        """Submit a function for background execution. Returns a task ID."""
        task_id = uuid.uuid4().hex[:8]
        start = time.monotonic()

        task_state = TaskState(
            task_id=task_id,
            tool_name=tool_name,
            kwargs=kwargs,
        )

        def progress_reporter(
            progress: float,
            total: float | None = None,
            message: str | None = None,
        ) -> None:
            self._report_progress(task_id, tool_name, progress, total, message)

        # Only inject report_progress if the function accepts it
        sig = inspect.signature(fn)
        accepts_progress = "report_progress" in sig.parameters

        def run() -> AsyncTaskResult:
            try:
                if accepts_progress:
                    result = fn(report_progress=progress_reporter, **kwargs)
                else:
                    result = fn(**kwargs)
                return AsyncTaskResult(
                    task_id=task_id,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    result=result,
                    error=None,
                    elapsed_ms=(time.monotonic() - start) * 1000,
                    status=TaskStatus.COMPLETED,
                )
            except Exception as e:
                return AsyncTaskResult(
                    task_id=task_id,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    result=None,
                    error=str(e),
                    elapsed_ms=(time.monotonic() - start) * 1000,
                    status=TaskStatus.FAILED,
                )

        future = self._executor.submit(run)
        task_state.future = future
        with self._lock:
            self._tasks[task_id] = task_state

        def on_done(f: Any) -> None:
            task_result = f.result()
            with self._lock:
                task = self._tasks.get(task_id)
                if task:
                    task.status = task_result.status
            if self.on_complete:
                self.on_complete(task_result)

        future.add_done_callback(on_done)
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task. Returns True if cancellation was requested."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.WORKING:
                return False
            task.status = TaskStatus.CANCELLED
            if task.future:
                task.future.cancel()
            return True

    def get_task_status(self, task_id: str) -> TaskState | None:
        """Get the current state of a task (for polling)."""
        with self._lock:
            return self._tasks.get(task_id)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == TaskStatus.WORKING)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
