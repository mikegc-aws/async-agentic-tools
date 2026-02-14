"""Async tool manager — background dispatch and completion callbacks."""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AsyncTaskResult:
    """Result of a completed async tool invocation."""

    task_id: str
    tool_name: str
    kwargs: dict
    result: Any
    error: str | None
    elapsed_ms: float


class AsyncToolManager:
    """Manages async tool dispatch and result delivery via callbacks.

    Submit functions for background execution. When they complete,
    the on_complete callback fires with the result.
    """

    def __init__(
        self,
        max_workers: int = 4,
        on_complete: Callable[[AsyncTaskResult], None] | None = None,
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.on_complete = on_complete

    def submit(self, tool_name: str, fn: Callable, **kwargs: Any) -> str:
        """Submit a function for background execution. Returns a task ID."""
        task_id = uuid.uuid4().hex[:8]
        start = time.monotonic()

        def run() -> AsyncTaskResult:
            try:
                result = fn(**kwargs)
                return AsyncTaskResult(
                    task_id=task_id,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    result=result,
                    error=None,
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )
            except Exception as e:
                return AsyncTaskResult(
                    task_id=task_id,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    result=None,
                    error=str(e),
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )

        future = self._executor.submit(run)
        with self._lock:
            self._pending[task_id] = future

        def on_done(f: Any) -> None:
            task_result = f.result()
            with self._lock:
                self._pending.pop(task_id, None)
            if self.on_complete:
                self.on_complete(task_result)

        future.add_done_callback(on_done)
        return task_id

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
