"""Optimistic-concurrency version token for tasks.

`TaskVersion` is an opaque marker that a `VersionedTaskStore` associates with
each persisted `Task`. Callers treat it as opaque: they never inspect or do
arithmetic on the underlying value, they only pass it back to `save` and compare
two versions with `is_after`.

The special value `TaskVersion.MISSING` denotes "versioning not tracked". A
store that does not implement optimistic concurrency (for example the default
single-process stores wrapped by `LegacyTaskStoreAdapter`) returns
`TaskVersion.MISSING` everywhere, which makes every comparison behave as
last-writer-wins -- i.e. exactly today's behaviour. This is what lets the
versioned interface be adopted incrementally without breaking existing stores.
"""

from typing import ClassVar


class TaskVersion:
    """Opaque optimistic-concurrency token for a stored `Task`.

    The wrapped value is chosen by the `VersionedTaskStore` implementation: a
    monotonic counter, a commit timestamp, an ETag, a row version, etc. It must
    be treated as opaque by callers; only `is_after` and `is_missing` are part
    of the contract.

    `TaskVersion.MISSING` (wrapping `0`) is the sentinel for "not tracked". It
    compares as older than any real version, and any real version compares as
    newer than it, so an unversioned store degrades cleanly to last-writer-wins.
    """

    __slots__ = ('_value',)

    MISSING: 'ClassVar[TaskVersion]'

    def __init__(self, value: int | str) -> None:
        self._value = value

    def __eq__(self, other: object) -> bool:
        """Two versions are equal when they wrap equal values."""
        return isinstance(other, TaskVersion) and self._value == other._value

    def __hash__(self) -> int:
        """Hash by the wrapped value so versions are usable as keys."""
        return hash(self._value)

    def __repr__(self) -> str:
        """Render `MISSING` specially, otherwise show the wrapped value."""
        if self.is_missing:
            return 'TaskVersion.MISSING'
        return f'TaskVersion({self._value!r})'

    @property
    def is_missing(self) -> bool:
        """Whether this token means "versioning is not tracked"."""
        return self._value == 0

    def is_after(self, other: 'TaskVersion') -> bool:
        """Whether `self` represents a strictly later state than `other`.

        The comparison is deliberately asymmetric around `MISSING`:

        * If `other` is `MISSING`, `self` is considered newer (an untracked
          baseline is older than anything).
        * If `self` is `MISSING` (and `other` is not), `self` is not newer.

        This is what makes an unversioned store behave as last-writer-wins:
        every event looks "new", so no compare-and-swap is ever rejected.

        Both versions must wrap the same value type (a store uses one type
        consistently); comparing across types is a programming error.
        """
        if other.is_missing:
            return True
        if self.is_missing:
            return False
        this = self._value
        that = other._value  # noqa: SLF001
        if type(this) is not type(that):
            raise TypeError(
                'Cannot compare TaskVersions with different value types: '
                f'{type(that).__name__} and {type(this).__name__}'
            )
        # Both values are the same type here (guarded above); the checker
        # cannot narrow the int | str union through the runtime type check.
        return that < this  # ty:ignore[unsupported-operator]


TaskVersion.MISSING = TaskVersion(0)
