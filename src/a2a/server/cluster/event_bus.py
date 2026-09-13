"""Cross-replica task event delivery.

`TaskEventBus` is the extension point that lets a task's event stream be observed
from any replica, not just the one running the agent. It carries the protocol
events (`Message`, `Task`, status/artifact updates) that clients subscribe to,
each tagged with the `TaskVersion` produced by applying it.

Delivery contract:

* `subscribe(task_id, after=version)` yields events written after `version`, by
  any replica, and blocks (rather than ending) when none are available.
* Duplicates are permitted; subscribers deduplicate on `version` using
  `TaskVersion.is_after`.
* Per task, ordering should be monotonic in `version`, but subscribers must
  tolerate gaps.

A store that persists events transactionally with the task (see
`VersionedDatabaseTaskStore`) may back the bus by reading that log, in which case
`publish` can be a no-op. A store without a transactional log uses an explicit
`publish` to fan out.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
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


class TaskEventBus(ABC):
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
    ) -> AsyncIterator[VersionedEvent]:
        """Yields events for `task_id` newer than `after`.

        The returned iterator blocks when no events are available and ends only
        when closed by the consumer (e.g. via ``aclose()``) or when the channel
        is destroyed.
        """

    @abstractmethod
    async def destroy(self, task_id: str) -> None:
        """Releases resources for a task that has reached a terminal state."""
