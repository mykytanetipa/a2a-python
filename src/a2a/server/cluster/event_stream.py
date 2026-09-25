from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from a2a.server.cluster.version import TaskVersion
from a2a.server.events.event_queue import Event


@dataclass(frozen=True)
class VersionedEvent:
    """An event together with the task version produced by applying it.

    The `version` is what makes replay safe: a subscriber holding a snapshot at
    version V discards any `VersionedEvent` whose version is not
    `is_after(V)`.
    """

    event: Event
    version: TaskVersion


class TaskEventStream(ABC):
    """Delivers task events across replicas.

    One logical channel per task id. Implementations may be push (pub/sub) or
    pull (polling a log); callers do not depend on which.
    """

    @abstractmethod
    async def publish(self, task_id: str, event: VersionedEvent) -> None:
        """Publishes one event for `task_id` to all replicas.

        Implementations whose `subscribe` reads a transactional event log may
        implement this as a no-op.
        """

    @abstractmethod
    def subscribe(
        self, task_id: str, *, after: TaskVersion
    ) -> AsyncGenerator[VersionedEvent, None]:
        """Yields events for `task_id` newer than `after`.

        Implementations are async generators: the returned stream blocks when no
        events are available and ends only when closed by the consumer (via
        ``aclose()``, e.g. through ``contextlib.aclosing``) or when the channel
        is destroyed. Returning an async generator lets the consumer drive that
        cleanup deterministically.
        """

    @abstractmethod
    async def destroy(self, task_id: str) -> None:
        """Releases resources for a task that has reached a terminal state."""
