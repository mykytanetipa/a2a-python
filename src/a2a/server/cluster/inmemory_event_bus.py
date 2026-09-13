"""In-process reference implementation of `TaskEventBus`.

Single-process fan-out over `asyncio` queues. This preserves today's behaviour
when no distributed bus is configured; a real multi-server deployment uses a
shared implementation such as `DatabaseTaskEventBus`.
"""

import asyncio

from collections.abc import AsyncIterator

from a2a.server.cluster.event_bus import TaskEventBus, VersionedEvent
from a2a.server.cluster.version import TaskVersion


class InMemoryTaskEventBus(TaskEventBus):
    """`TaskEventBus` that fans out to in-process subscribers.

    `publish` delivers to every current subscriber of the task; `subscribe`
    returns an async iterator backed by a per-subscriber queue and filters on
    the caller's `after` version.
    """

    def __init__(self) -> None:
        # {task_id: set of subscriber queues}
        self._subscribers: dict[str, set[asyncio.Queue[VersionedEvent]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, task_id: str, event: VersionedEvent) -> None:
        """Delivers `event` to all current subscribers of `task_id`."""
        async with self._lock:
            queues = list(self._subscribers.get(task_id, ()))
        for queue in queues:
            await queue.put(event)

    async def subscribe(  # type: ignore[override]
        self, task_id: str, *, after: TaskVersion
    ) -> AsyncIterator[VersionedEvent]:
        """Yields events for `task_id` published after this call, newer than `after`."""
        queue: asyncio.Queue[VersionedEvent] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(task_id, set()).add(queue)
        try:
            while True:
                versioned = await queue.get()
                if versioned.version.is_after(after):
                    yield versioned
        finally:
            async with self._lock:
                subs = self._subscribers.get(task_id)
                if subs is not None:
                    subs.discard(queue)
                    if not subs:
                        del self._subscribers[task_id]

    async def destroy(self, task_id: str) -> None:
        """Drops all subscribers for `task_id`."""
        async with self._lock:
            self._subscribers.pop(task_id, None)
