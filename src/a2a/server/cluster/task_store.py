"""Task store with optimistic concurrency control for multi-server deployments.

`VersionedTaskStore` is a parallel interface to `a2a.server.tasks.TaskStore`.
It is intentionally *not* a modification of the existing `TaskStore` ABC so that
every current implementation keeps working untouched. A plain `TaskStore` can be
adapted to this interface with `LegacyTaskStoreAdapter`, which reports
`TaskVersion.MISSING` for every version and therefore performs no
compare-and-swap (last-writer-wins, i.e. today's behaviour).

The added capability is optimistic concurrency: `save` takes the version the
task was read at (`prev_version`) and returns the new version, and raises
`ConcurrentTaskModificationError` when the stored version no longer matches.
This is the mechanism that makes concurrent writes across replicas safe and that
lets a cancel on one replica stop an execution running on another (the executing
replica's next `save` fails the compare-and-swap).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from a2a.server.cluster.version import TaskVersion
from a2a.server.tasks.task_store import TaskStore


if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext
    from a2a.server.events.event_queue import Event
    from a2a.types.a2a_pb2 import (
        ListTasksRequest,
        ListTasksResponse,
        Task,
    )


class ConcurrentTaskModificationError(Exception):
    """Raised by `VersionedTaskStore.save` when `prev_version` is stale.

    This is a server-internal coordination signal, never an A2A wire error. The
    A2A error set is closed and has no concurrency error, so callers must handle
    this internally -- typically by reloading the task and retrying, or (for a
    cancel) by treating it as "someone else won, stop". It must never be
    surfaced to the client as-is.
    """

    def __init__(self, task_id: str) -> None:
        super().__init__(
            f'Task {task_id} was modified concurrently by another writer'
        )
        self.task_id = task_id


class VersionedTaskStore(ABC):
    """A `TaskStore` variant with optimistic concurrency control.

    Distinct from `TaskStore` so that existing implementations are unaffected.
    Wrap an existing `TaskStore` in `LegacyTaskStoreAdapter` to use it where a
    `VersionedTaskStore` is expected.
    """

    @abstractmethod
    async def save(
        self,
        task: Task,
        *,
        event: Event | None,
        prev: Task | None,
        prev_version: TaskVersion,
        context: ServerCallContext,
    ) -> TaskVersion:
        """Persists `task` and returns its new version.

        Args:
            task: The task state to persist.
            event: The event that produced this state, or `None` for a direct
                write. Stores backed by an event log should persist it in the
                same transaction as the task row, so the log is a valid source
                for cross-replica replay.
            prev: The task as previously read, for implementations that diff.
            prev_version: The version `task` was derived from. Implementations
                MUST raise `ConcurrentTaskModificationError` if the currently
                stored version differs, unless `prev_version` is
                `TaskVersion.MISSING` (in which case no check is performed).
            context: The server call context (used to resolve the owner).

        Returns:
            The new `TaskVersion` for the persisted task.

        Raises:
            ConcurrentTaskModificationError: If `prev_version` is stale.
        """

    @abstractmethod
    async def get(
        self, task_id: str, context: ServerCallContext
    ) -> tuple[Task | None, TaskVersion]:
        """Retrieves a task and its current version.

        Returns `(None, TaskVersion.MISSING)` if the task does not exist.
        """

    @abstractmethod
    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        """Retrieves a list of tasks from the store."""

    @abstractmethod
    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Deletes a task from the store by ID."""


class LegacyTaskStoreAdapter(VersionedTaskStore):
    """Runs an unversioned `TaskStore` under the `VersionedTaskStore` interface.

    Every version reported and returned is `TaskVersion.MISSING`, so no
    compare-and-swap is performed and behaviour is identical to the wrapped
    store (last-writer-wins). This is the migration path: existing stores drop
    in unchanged. It does NOT make an unversioned store multi-writer safe --
    concurrent writes still clobber -- it only satisfies the interface.
    """

    def __init__(self, inner: TaskStore) -> None:
        self._inner = inner

    @property
    def inner(self) -> TaskStore:
        """The wrapped task store."""
        return self._inner

    async def save(
        self,
        task: Task,
        *,
        event: Event | None,
        prev: Task | None,
        prev_version: TaskVersion,
        context: ServerCallContext,
    ) -> TaskVersion:
        """Saves via the wrapped store; ignores version args, returns MISSING."""
        del event, prev, prev_version  # unversioned store ignores these
        await self._inner.save(task, context)
        return TaskVersion.MISSING

    async def get(
        self, task_id: str, context: ServerCallContext
    ) -> tuple[Task | None, TaskVersion]:
        """Gets from the wrapped store, pairing the result with MISSING."""
        task = await self._inner.get(task_id, context)
        return task, TaskVersion.MISSING

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        """Lists tasks via the wrapped store."""
        return await self._inner.list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Deletes a task via the wrapped store."""
        await self._inner.delete(task_id, context)


class PlainTaskStoreView(TaskStore):
    """Exposes a `VersionedTaskStore` through the plain `TaskStore` interface.

    Code paths that only need the task (not its version) -- e.g. `on_get_task`,
    `on_list_tasks`, and the request-context builder -- can use this so they
    work uniformly whether the configured store is versioned or not. `get`
    unwraps the ``(task, version)`` tuple; `save` writes with no compare-and-swap
    (MISSING), which is appropriate for these non-CAS read/utility paths.
    """

    def __init__(self, inner: VersionedTaskStore) -> None:
        self._inner = inner

    @property
    def versioned(self) -> VersionedTaskStore:
        """The wrapped versioned store."""
        return self._inner

    async def save(self, task: Task, context: ServerCallContext) -> None:
        """Saves without a version check (last-writer-wins)."""
        await self._inner.save(
            task,
            event=None,
            prev=None,
            prev_version=TaskVersion.MISSING,
            context=context,
        )

    async def get(
        self, task_id: str, context: ServerCallContext
    ) -> Task | None:
        """Gets the task, discarding its version."""
        task, _ = await self._inner.get(task_id, context)
        return task

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        """Lists tasks via the wrapped store."""
        return await self._inner.list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Deletes a task via the wrapped store."""
        await self._inner.delete(task_id, context)
