"""Multi-server (clustered) deployment support for the A2A server.

This package provides the extension points needed to run the A2A server as
multiple replicas behind a load balancer without task affinity:

* `TaskVersion` and `VersionedTaskStore` add optimistic concurrency control so
  concurrent writes across replicas are safe.
* `LegacyTaskStoreAdapter` wraps an existing `TaskStore` so it satisfies the
  versioned interface (with last-writer-wins semantics).

These are additive: a single-process deployment that does not use them keeps its
current behaviour unchanged.
"""

from a2a.server.cluster.task_store import (
    ConcurrentTaskModificationError,
    LegacyTaskStoreAdapter,
    PlainTaskStoreView,
    VersionedTaskStore,
)
from a2a.server.cluster.version import TaskVersion


__all__ = [
    'ConcurrentTaskModificationError',
    'LegacyTaskStoreAdapter',
    'PlainTaskStoreView',
    'TaskVersion',
    'VersionedTaskStore',
]
