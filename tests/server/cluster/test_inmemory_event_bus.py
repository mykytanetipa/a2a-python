"""Tests for `InMemoryTaskEventBus`."""

import asyncio

import pytest

from a2a.server.cluster import (
    InMemoryTaskEventBus,
    TaskEventBus,
    TaskVersion,
    VersionedEvent,
)
from a2a.types.a2a_pb2 import (
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)


def make_event(task_id: str = 'task-abc') -> Task:
    return Task(
        id=task_id,
        context_id='ctx',
        status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
    )


def make_status_event(task_id: str = 'task-abc') -> TaskStatusUpdateEvent:
    return TaskStatusUpdateEvent(
        task_id=task_id,
        context_id='ctx',
        status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
    )


def test_is_a_task_event_bus() -> None:
    assert isinstance(InMemoryTaskEventBus(), TaskEventBus)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_subscriber_receives_published_event() -> None:
    bus = InMemoryTaskEventBus()
    received: list[VersionedEvent] = []

    async def consume() -> None:
        async for ve in bus.subscribe('task-abc', after=TaskVersion.MISSING):
            received.append(ve)
            return  # stop after first

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let the subscriber register
    await bus.publish('task-abc', VersionedEvent(make_event(), TaskVersion(1)))
    await asyncio.wait_for(task, timeout=2)

    assert len(received) == 1
    assert received[0].version == TaskVersion(1)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_events_before_after_version_are_filtered() -> None:
    bus = InMemoryTaskEventBus()
    received: list[TaskVersion] = []

    async def consume() -> None:
        async for ve in bus.subscribe('task-abc', after=TaskVersion(5)):
            received.append(ve.version)
            if ve.version == TaskVersion(7):
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    # v3 and v5 are not after v5 -> filtered; v6, v7 pass.
    await bus.publish('task-abc', VersionedEvent(make_event(), TaskVersion(3)))
    await bus.publish('task-abc', VersionedEvent(make_event(), TaskVersion(5)))
    await bus.publish('task-abc', VersionedEvent(make_event(), TaskVersion(6)))
    await bus.publish('task-abc', VersionedEvent(make_event(), TaskVersion(7)))
    await asyncio.wait_for(task, timeout=2)

    assert received == [TaskVersion(6), TaskVersion(7)]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_multiple_subscribers_all_receive() -> None:
    bus = InMemoryTaskEventBus()
    got_a: list[TaskVersion] = []
    got_b: list[TaskVersion] = []

    async def consume(sink: list[TaskVersion]) -> None:
        async for ve in bus.subscribe('task-abc', after=TaskVersion.MISSING):
            sink.append(ve.version)
            return

    ta = asyncio.create_task(consume(got_a))
    tb = asyncio.create_task(consume(got_b))
    await asyncio.sleep(0.05)
    await bus.publish(
        'task-abc', VersionedEvent(make_status_event(), TaskVersion(1))
    )
    await asyncio.gather(asyncio.wait_for(ta, 2), asyncio.wait_for(tb, 2))

    assert got_a == [TaskVersion(1)]
    assert got_b == [TaskVersion(1)]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_events_are_isolated_per_task_id() -> None:
    bus = InMemoryTaskEventBus()
    received: list[str] = []

    async def consume() -> None:
        async for ve in bus.subscribe('task-1', after=TaskVersion.MISSING):
            received.append(ve.event.id)
            return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    # Publish to a different task -> should NOT be delivered.
    await bus.publish(
        'task-2', VersionedEvent(make_event('task-2'), TaskVersion(1))
    )
    await asyncio.sleep(0.1)
    assert received == []
    # Now publish to the right task.
    await bus.publish(
        'task-1', VersionedEvent(make_event('task-1'), TaskVersion(1))
    )
    await asyncio.wait_for(task, timeout=2)
    assert received == ['task-1']


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_publish_with_no_subscribers_is_noop() -> None:
    bus = InMemoryTaskEventBus()
    # Should not raise even though nobody is listening.
    await bus.publish('task-x', VersionedEvent(make_event(), TaskVersion(1)))


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_subscriber_cleanup_on_close() -> None:
    bus = InMemoryTaskEventBus()
    agen = bus.subscribe('task-abc', after=TaskVersion.MISSING)

    # Register the subscriber by advancing to the first await.
    consumer = asyncio.ensure_future(agen.__anext__())
    await asyncio.sleep(0.05)
    assert 'task-abc' in bus._subscribers  # noqa: SLF001

    await bus.publish('task-abc', VersionedEvent(make_event(), TaskVersion(1)))
    await asyncio.wait_for(consumer, timeout=2)

    # Explicitly close the generator: its finally block deregisters the sub.
    await agen.aclose()
    assert 'task-abc' not in bus._subscribers  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_destroy_removes_subscribers() -> None:
    bus = InMemoryTaskEventBus()
    await bus.destroy('task-abc')  # no-op when absent
    # Register then destroy.
    agen = bus.subscribe('task-abc', after=TaskVersion.MISSING)
    it = agen.__aiter__()
    _ = asyncio.ensure_future(it.__anext__())
    await asyncio.sleep(0.05)
    await bus.destroy('task-abc')
    assert 'task-abc' not in bus._subscribers  # noqa: SLF001
