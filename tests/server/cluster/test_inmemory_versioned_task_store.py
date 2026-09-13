"""Tests for `VersionedInMemoryTaskStore`."""

import pytest

from a2a.auth.user import User
from a2a.server.cluster import (
    ConcurrentTaskModificationError,
    TaskVersion,
    VersionedInMemoryTaskStore,
    VersionedTaskStore,
)
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import (
    ListTasksRequest,
    Task,
    TaskState,
    TaskStatus,
)


class SampleUser(User):
    """A test implementation of the User interface."""

    def __init__(self, user_name: str):
        self._user_name = user_name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._user_name


TEST_CONTEXT = ServerCallContext(user=SampleUser('test_user'))
OTHER_CONTEXT = ServerCallContext(user=SampleUser('other_user'))


def create_task(
    task_id: str = 'task-abc',
    context_id: str = 'session-xyz',
    state: TaskState = TaskState.TASK_STATE_SUBMITTED,
) -> Task:
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=state),
    )


def store() -> VersionedInMemoryTaskStore:
    return VersionedInMemoryTaskStore()


def test_is_a_versioned_task_store() -> None:
    assert isinstance(store(), VersionedTaskStore)


@pytest.mark.asyncio
async def test_first_save_returns_version_one() -> None:
    s = store()
    version = await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    assert version == TaskVersion(1)


@pytest.mark.asyncio
async def test_get_missing_returns_none_and_missing() -> None:
    s = store()
    task, version = await s.get('nope', TEST_CONTEXT)
    assert task is None
    assert version.is_missing


@pytest.mark.asyncio
async def test_save_then_get_roundtrip_with_version() -> None:
    s = store()
    task = create_task()
    v1 = await s.save(
        task,
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    got, version = await s.get('task-abc', TEST_CONTEXT)
    assert got == task
    assert version == v1


@pytest.mark.asyncio
async def test_version_increments_on_each_save() -> None:
    s = store()
    v1 = await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    v2 = await s.save(
        create_task(state=TaskState.TASK_STATE_WORKING),
        event=None,
        prev=None,
        prev_version=v1,
        context=TEST_CONTEXT,
    )
    assert v2.is_after(v1)
    _, current = await s.get('task-abc', TEST_CONTEXT)
    assert current == v2


@pytest.mark.asyncio
async def test_stale_prev_version_raises() -> None:
    s = store()
    v1 = await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    # A second writer advances the task.
    await s.save(
        create_task(state=TaskState.TASK_STATE_WORKING),
        event=None,
        prev=None,
        prev_version=v1,
        context=TEST_CONTEXT,
    )
    # First writer still holds v1 -> stale -> conflict.
    with pytest.raises(ConcurrentTaskModificationError):
        await s.save(
            create_task(state=TaskState.TASK_STATE_COMPLETED),
            event=None,
            prev=None,
            prev_version=v1,
            context=TEST_CONTEXT,
        )


@pytest.mark.asyncio
async def test_update_with_non_missing_version_on_absent_task_raises() -> None:
    s = store()
    with pytest.raises(ConcurrentTaskModificationError):
        await s.save(
            create_task(),
            event=None,
            prev=None,
            prev_version=TaskVersion(5),
            context=TEST_CONTEXT,
        )


@pytest.mark.asyncio
async def test_missing_prev_version_skips_cas() -> None:
    # Passing MISSING always writes (last-writer-wins), even on an existing task.
    s = store()
    await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    version = await s.save(
        create_task(state=TaskState.TASK_STATE_WORKING),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    assert version == TaskVersion(2)


@pytest.mark.asyncio
async def test_delete_forgets_version() -> None:
    s = store()
    await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    await s.delete('task-abc', TEST_CONTEXT)
    task, version = await s.get('task-abc', TEST_CONTEXT)
    assert task is None
    assert version.is_missing
    # Re-creating starts versioning from 1 again.
    v = await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    assert v == TaskVersion(1)


@pytest.mark.asyncio
async def test_versions_are_scoped_per_owner() -> None:
    s = store()
    await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=TEST_CONTEXT,
    )
    # Different owner, same task_id: independent version namespace.
    v_other = await s.save(
        create_task(),
        event=None,
        prev=None,
        prev_version=TaskVersion.MISSING,
        context=OTHER_CONTEXT,
    )
    assert v_other == TaskVersion(1)
    _, v_mine = await s.get('task-abc', TEST_CONTEXT)
    assert v_mine == TaskVersion(1)


@pytest.mark.asyncio
async def test_list_returns_saved_tasks() -> None:
    s = store()
    for i in range(3):
        await s.save(
            create_task(task_id=f'task-{i}'),
            event=None,
            prev=None,
            prev_version=TaskVersion.MISSING,
            context=TEST_CONTEXT,
        )
    resp = await s.list(ListTasksRequest(), TEST_CONTEXT)
    assert {t.id for t in resp.tasks} == {'task-0', 'task-1', 'task-2'}
