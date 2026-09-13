"""In-memory reference implementation of `VersionedTaskStore`.

Single-process store with optimistic concurrency control. Intended for tests
and local development; a real multi-server deployment uses a shared, durable
store such as `VersionedDatabaseTaskStore`.

The version is a per-task monotonic integer counter, mirroring a2a-go's
in-memory reference store (`stored.version + 1`).
"""

import threading

from a2a.server.cluster.task_store import (
    ConcurrentTaskModificationError,
    VersionedTaskStore,
)
from a2a.server.cluster.version import TaskVersion
from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import Event
from a2a.server.owner_resolver import OwnerResolver, resolve_user_scope
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types.a2a_pb2 import ListTasksRequest, ListTasksResponse, Task


class VersionedInMemoryTaskStore(VersionedTaskStore):
    """`VersionedTaskStore` backed by in-process dictionaries.

    Holds one integer version per (owner, task_id). `save` performs a
    compare-and-swap against `prev_version` and raises
    `ConcurrentTaskModificationError` on mismatch. Reads and list operations
    delegate to a wrapped `InMemoryTaskStore` for the task data itself.
    """

    def __init__(
        self, owner_resolver: OwnerResolver = resolve_user_scope
    ) -> None:
        self._owner_resolver = owner_resolver
        self._store = InMemoryTaskStore()
        # Maps owner to a mapping of task_id to its current integer version.
        self._versions: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def _current_version_locked(self, owner: str, task_id: str) -> int:
        return self._versions.get(owner, {}).get(task_id, 0)

    async def save(
        self,
        task: Task,
        *,
        event: Event | None,
        prev: Task | None,
        prev_version: TaskVersion,
        context: ServerCallContext,
    ) -> TaskVersion:
        """Persists `task`, bumping its version, with a compare-and-swap.

        Raises `ConcurrentTaskModificationError` if `prev_version` does not
        match the currently stored version (unless `prev_version` is MISSING,
        which skips the check for last-writer-wins compatibility).
        """
        del event, prev  # in-memory store keeps no event log
        owner = self._owner_resolver(context)
        with self._lock:
            stored = self._current_version_locked(owner, task_id=task.id)
            if not prev_version.is_missing:
                if stored == 0:
                    # Caller expected an existing version but the task is gone.
                    raise ConcurrentTaskModificationError(task.id)
                if TaskVersion(stored) != prev_version:
                    raise ConcurrentTaskModificationError(task.id)
            new_version = stored + 1
            await self._store.save(task, context)
            self._versions.setdefault(owner, {})[task.id] = new_version
            return TaskVersion(new_version)

    async def get(
        self, task_id: str, context: ServerCallContext
    ) -> tuple[Task | None, TaskVersion]:
        """Returns the task and its current version, or (None, MISSING)."""
        owner = self._owner_resolver(context)
        with self._lock:
            task = await self._store.get(task_id, context)
            if task is None:
                return None, TaskVersion.MISSING
            stored = self._current_version_locked(owner, task_id)
            return task, TaskVersion(stored)

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        """Lists tasks via the wrapped in-memory store."""
        return await self._store.list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Deletes a task and forgets its version."""
        owner = self._owner_resolver(context)
        with self._lock:
            await self._store.delete(task_id, context)
            owner_versions = self._versions.get(owner)
            if owner_versions is not None:
                owner_versions.pop(task_id, None)
                if not owner_versions:
                    del self._versions[owner]
