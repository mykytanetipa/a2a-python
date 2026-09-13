"""Multi-server (clustered) deployment support for the A2A server.

This package provides the extension points needed to run the A2A server as
multiple replicas behind a load balancer without task affinity:

* `TaskVersion` and `VersionedTaskStore` add optimistic concurrency control so
  concurrent writes across replicas are safe.
* `LegacyTaskStoreAdapter` wraps an existing `TaskStore` so it satisfies the
  versioned interface (with last-writer-wins semantics).
* `VersionedInMemoryTaskStore` and `VersionedDatabaseTaskStore` are concrete
  compare-and-set implementations.

These are additive: a single-process deployment that does not use them keeps its
current behaviour unchanged.
"""

import logging

from a2a.server.cluster.inmemory_task_store import (
    VersionedInMemoryTaskStore,
)
from a2a.server.cluster.task_store import (
    ConcurrentTaskModificationError,
    LegacyTaskStoreAdapter,
    PlainTaskStoreView,
    VersionedTaskStore,
)
from a2a.server.cluster.version import TaskVersion


logger = logging.getLogger(__name__)

try:
    from a2a.server.cluster.database_task_store import (
        VersionedDatabaseTaskStore,
    )
except ImportError as e:
    _original_error = e
    logger.debug(
        'Database-backed cluster stores not loaded. This is expected if '
        'database dependencies are not installed. Error: %s',
        _original_error,
    )

    class VersionedDatabaseTaskStore:  # type: ignore[no-redef]
        """Placeholder when database dependencies are not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                'To use VersionedDatabaseTaskStore, its dependencies must be '
                "installed. Install with 'pip install a2a-sdk[sql]'."
            ) from _original_error


__all__ = [
    'ConcurrentTaskModificationError',
    'LegacyTaskStoreAdapter',
    'PlainTaskStoreView',
    'TaskVersion',
    'VersionedDatabaseTaskStore',
    'VersionedInMemoryTaskStore',
    'VersionedTaskStore',
]
