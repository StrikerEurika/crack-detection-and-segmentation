from __future__ import annotations
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    fn: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class TaskManager:
    def __init__(self, max_workers: int = 2):
        self._tasks: dict[str, Task] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        import logging
        logging.getLogger("api.task_manager").info("Task manager started.")

    async def stop(self):
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    async def submit(
        self,
        fn: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        tid = task_id or str(uuid.uuid4())
        task = Task(
            task_id=tid,
            status=TaskStatus.PENDING,
            fn=fn,
            args=args,
            kwargs=kwargs,
        )
        self._tasks[tid] = task
        await self._queue.put(tid)
        return tid

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_status(self, task_id: str) -> Optional[dict]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "progress": task.progress,
            "message": task.message,
            "result": task.result,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

    async def _worker_loop(self):
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                task_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            task = self._tasks.get(task_id)
            if task is None:
                continue

            task.status = TaskStatus.RUNNING
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    self._run_sync,
                    task,
                )
                task.result = result
                task.status = TaskStatus.COMPLETED
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                import traceback
                traceback.print_exc()
            finally:
                task.completed_at = datetime.now(timezone.utc)

    def _run_sync(self, task: Task) -> Any:
        return task.fn(*task.args, task=task, **task.kwargs)

    def update_progress(self, task_id: str, progress: float, message: str = None):
        task = self._tasks.get(task_id)
        if task:
            task.progress = progress
            if message:
                task.message = message


task_manager = TaskManager()
