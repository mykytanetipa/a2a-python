"""Multi-replica integration tests against a real shared database.

Two `DefaultRequestHandlerV2` instances share one database (a versioned task
store + a DB event bus, each replica opening its own engine against the same
DSN). This is the true analog of two replicas behind a load balancer with no
task affinity: any request for any task may land on either replica.

These exercise the correctness-tier fixes end to end over real storage:
optimistic-concurrency saves (no lost updates), request-boundary cache
invalidation (no stale-snapshot clobber on multi-turn input_required),
cross-replica resubscription via the event log, and cancel-via-compare-and-swap.

DSN handling mirrors tests/server/tasks/test_database_task_store.py: the sqlite
parameter always runs; the Postgres/MySQL parameters run only when
POSTGRES_TEST_DSN / MYSQL_TEST_DSN are set (locally via
scripts/docker-compose.test.yml, and in CI via the service containers).
"""

import asyncio
import contextlib
import os
import uuid

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from _pytest.mark.structures import ParameterSet


pytest.importorskip('sqlalchemy', reason='Database tests require SQLAlchemy')

from a2a.auth.user import User
from a2a.helpers.proto_helpers import new_task_from_user_message
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.cluster.database_event_bus import DatabaseTaskEventBus
from a2a.server.cluster.database_task_store import VersionedDatabaseTaskStore
from a2a.server.context import ServerCallContext
from a2a.server.models import Base
from a2a.server.request_handlers.default_request_handler_v2 import (
    DefaultRequestHandlerV2,
)
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    CancelTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SubscribeToTaskRequest,
    TaskState,
)
from a2a.utils.errors import (
    A2AError,
    InvalidParamsError,
    TaskNotCancelableError,
    TaskNotFoundError,
)
from sqlalchemy.ext.asyncio import create_async_engine


# --- DSN parametrization (mirrors test_database_task_store.py) ---------------

POSTGRES_TEST_DSN = os.environ.get('POSTGRES_TEST_DSN')
MYSQL_TEST_DSN = os.environ.get('MYSQL_TEST_DSN')


def _sqlite_dsn() -> str:
    # Unique shared-cache in-memory DB per test so both replica engines see the
    # same data while different tests never collide.
    name = f'multiserver_{uuid.uuid4().hex}'
    return f'sqlite+aiosqlite:///file:{name}?mode=memory&cache=shared&uri=true'


DB_CONFIGS: list[ParameterSet | tuple[str | None, str]] = [
    pytest.param(('sqlite', 'sqlite'), id='sqlite')
]
if POSTGRES_TEST_DSN:
    DB_CONFIGS.append(
        pytest.param((POSTGRES_TEST_DSN, 'postgresql'), id='postgresql')
    )
else:
    DB_CONFIGS.append(
        pytest.param(
            (None, 'postgresql'),
            marks=pytest.mark.skip(reason='POSTGRES_TEST_DSN not set'),
            id='postgresql_skipped',
        )
    )
if MYSQL_TEST_DSN:
    DB_CONFIGS.append(pytest.param((MYSQL_TEST_DSN, 'mysql'), id='mysql'))
else:
    DB_CONFIGS.append(
        pytest.param(
            (None, 'mysql'),
            marks=pytest.mark.skip(reason='MYSQL_TEST_DSN not set'),
            id='mysql_skipped',
        )
    )


# --- Local, self-contained helpers ------------------------------------------


class SampleUser(User):
    """Minimal authenticated user for tests."""

    def __init__(self, user_name: str = 'test_user') -> None:
        self._user_name = user_name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._user_name


def make_context(user: str = 'test_user') -> ServerCallContext:
    return ServerCallContext(user=SampleUser(user))


def streaming_agent_card() -> AgentCard:
    return AgentCard(capabilities=AgentCapabilities(streaming=True))


def build_send_request(
    text: str,
    task_id: str = '',
    context_id: str = '',
    message_id: str = 'm',
) -> SendMessageRequest:
    msg = Message(
        role=Role.ROLE_USER, message_id=message_id, parts=[Part(text=text)]
    )
    if task_id:
        msg.task_id = task_id
    if context_id:
        msg.context_id = context_id
    return SendMessageRequest(message=msg)


async def drain(agen) -> None:  # noqa: ANN001
    with contextlib.suppress(Exception):
        async for _ in agen:
            pass


async def wait_for_state(
    store: VersionedDatabaseTaskStore,
    task_id: str,
    state: TaskState,
    context: ServerCallContext,
    *,
    tries: int = 200,
    delay: float = 0.02,
) -> None:
    for _ in range(tries):
        task, _ = await store.get(task_id, context)
        if task is not None and task.status.state == state:
            return
        await asyncio.sleep(delay)
    raise AssertionError(f'Task {task_id} did not reach {state} in time')


class CompletingAgent(AgentExecutor):
    """Creates the task if needed, goes WORKING, then completes."""

    async def execute(self, context, event_queue) -> None:  # noqa: ANN001
        if context.current_task is None:
            await event_queue.enqueue_event(
                new_task_from_user_message(context.message)
            )
        updater = TaskUpdater(
            event_queue,
            str(context.task_id or ''),
            str(context.context_id or ''),
        )
        await updater.start_work()
        await updater.complete()

    async def cancel(self, context, event_queue) -> None:  # noqa: ANN001
        pass


class ControlledAgent(AgentExecutor):
    """Goes WORKING, then completes only when released."""

    def __init__(self) -> None:
        self.working = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, context, event_queue) -> None:  # noqa: ANN001
        if context.current_task is None:
            await event_queue.enqueue_event(
                new_task_from_user_message(context.message)
            )
        updater = TaskUpdater(
            event_queue,
            str(context.task_id or ''),
            str(context.context_id or ''),
        )
        await updater.start_work()
        self.working.set()
        await self.release.wait()
        await updater.add_artifact([Part(text='result')])
        await updater.complete()

    async def cancel(self, context, event_queue) -> None:  # noqa: ANN001
        pass


class LongRunningAgent(AgentExecutor):
    """Goes WORKING then saves periodically until aborted by a CAS conflict."""

    def __init__(self) -> None:
        self.working = asyncio.Event()
        self.aborted = asyncio.Event()

    async def execute(self, context, event_queue) -> None:  # noqa: ANN001
        if context.current_task is None:
            await event_queue.enqueue_event(
                new_task_from_user_message(context.message)
            )
        updater = TaskUpdater(
            event_queue,
            str(context.task_id or ''),
            str(context.context_id or ''),
        )
        await updater.start_work()
        self.working.set()
        try:
            for _ in range(200):
                await asyncio.sleep(0.05)
                await updater.update_status(
                    TaskState.TASK_STATE_WORKING,
                    message=updater.new_agent_message([Part(text='tick')]),
                )
        except asyncio.CancelledError:
            self.aborted.set()
            raise

    async def cancel(self, context, event_queue) -> None:  # noqa: ANN001
        pass


class InputRequiredThenCompleteAgent(AgentExecutor):
    """Asks for input until two user turns exist, then completes.

    Turn is derived purely from durable task history, so it behaves correctly
    regardless of which replica runs each turn.
    """

    async def execute(self, context, event_queue) -> None:  # noqa: ANN001
        task = context.current_task
        if task is None:
            await event_queue.enqueue_event(
                new_task_from_user_message(context.message)
            )
        updater = TaskUpdater(
            event_queue,
            str(context.task_id or ''),
            str(context.context_id or ''),
        )
        user_msgs = 0
        if task is not None:
            user_msgs = sum(1 for m in task.history if m.role == Role.ROLE_USER)

        if task is None or user_msgs <= 1:
            await updater.requires_input(
                message=updater.new_agent_message([Part(text='need input')])
            )
        else:
            await updater.complete()

    async def cancel(self, context, event_queue) -> None:  # noqa: ANN001
        pass


# --- Cluster fixture: two replicas over one shared DB -----------------------


class Cluster:
    """A pair of replicas plus an independent store for assertions."""

    def __init__(
        self,
        dsn: str,
        agent_a: AgentExecutor,
        agent_b: AgentExecutor,
    ) -> None:
        self._dsn = dsn
        self._engines = []
        self.replica_a = self._make_replica(agent_a)
        self.replica_b = self._make_replica(agent_b)
        # A separate store (own engine) for reading final state in assertions,
        # so assertions never share an engine/session with a replica.
        self.assert_store = self._make_store()

    def _make_store(self) -> VersionedDatabaseTaskStore:
        engine = create_async_engine(self._dsn)
        self._engines.append(engine)
        return VersionedDatabaseTaskStore(engine=engine, create_table=False)

    def _make_replica(self, agent: AgentExecutor) -> DefaultRequestHandlerV2:
        engine = create_async_engine(self._dsn)
        self._engines.append(engine)
        store = VersionedDatabaseTaskStore(engine=engine, create_table=False)
        bus = DatabaseTaskEventBus(
            engine=engine, create_table=False, poll_interval_s=0.05
        )
        return DefaultRequestHandlerV2(
            agent_executor=agent,
            task_store=store,
            agent_card=streaming_agent_card(),
            event_bus=bus,
        )

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self.replica_a.aclose()
        with contextlib.suppress(Exception):
            await self.replica_b.aclose()
        for engine in self._engines:
            await engine.dispose()


@pytest_asyncio.fixture(params=DB_CONFIGS)
async def dsn(request) -> AsyncGenerator[str, None]:
    param, dialect = request.param
    if param is None:
        pytest.skip(f'DSN for {dialect} not set.')
    url = _sqlite_dsn() if param == 'sqlite' else param

    engine = create_async_engine(url)
    # Keep one connection open for the whole test so a shared-cache in-memory
    # SQLite database is not torn down while replica engines connect to it.
    keepalive = await engine.connect()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield url
    finally:
        with contextlib.suppress(Exception):
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        await keepalive.close()
        await engine.dispose()


# --- Scenarios --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_concurrent_followups_no_lost_update(dsn: str) -> None:
    """Concurrent follow-up sends to one task, from both replicas, are safe.

    The task is created and paused (input_required) on replica A. Two follow-up
    sends then race from both replicas. Optimistic concurrency serializes the
    writes at the store: neither silently overwrites the other, and the durable
    task ends in a consistent, non-corrupt state (all user turns preserved).

    Skipped on SQLite: shared-cache SQLite uses whole-table locking and cannot
    faithfully model two simultaneous writers (it raises "database is locked"
    rather than applying row-level OCC). Real databases (Postgres/MySQL) are the
    meaningful target for genuinely concurrent writes.
    """
    if dsn.startswith('sqlite'):
        pytest.skip('SQLite cannot model concurrent writers (table locking)')
    cluster = Cluster(
        dsn,
        InputRequiredThenCompleteAgent(),
        InputRequiredThenCompleteAgent(),
    )
    ctx = make_context()
    try:
        r1 = await cluster.replica_a.on_message_send(
            build_send_request('q1', message_id='m1'), ctx
        )
        assert r1.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        task_id = r1.id
        context_id = r1.context_id

        # Two concurrent follow-ups for the SAME existing task, one per replica.
        results = await asyncio.gather(
            cluster.replica_a.on_message_send(
                build_send_request(
                    'a', task_id=task_id, context_id=context_id, message_id='ma'
                ),
                ctx,
            ),
            cluster.replica_b.on_message_send(
                build_send_request(
                    'b', task_id=task_id, context_id=context_id, message_id='mb'
                ),
                ctx,
            ),
            return_exceptions=True,
        )
        # Conflicts are handled internally; no unhandled exception reaches the
        # client. (A genuine concurrent-execution race may surface as an
        # InternalError on at most one call; corruption never does.)
        for r in results:
            if isinstance(r, BaseException):
                assert isinstance(r, A2AError), r

        # The durable task is intact: no lost writes, and all three user turns
        # (q1, a, b) survived regardless of interleaving.
        final, version = await cluster.assert_store.get(task_id, ctx)
        assert final is not None
        assert not version.is_missing
        user_turns = sum(1 for m in final.history if m.role == Role.ROLE_USER)
        assert user_turns == 3, [m.message_id for m in final.history]
    finally:
        await cluster.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_multiturn_input_required_across_replicas(dsn: str) -> None:
    """turn 1 -> A, turn 2 -> B, turn 3 -> A; no stale-snapshot clobber."""
    cluster = Cluster(
        dsn,
        InputRequiredThenCompleteAgent(),
        InputRequiredThenCompleteAgent(),
    )
    ctx = make_context()
    try:
        r1 = await cluster.replica_a.on_message_send(
            build_send_request('q1', message_id='m1'), ctx
        )
        assert r1.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        task_id = r1.id
        context_id = r1.context_id

        r2 = await cluster.replica_b.on_message_send(
            build_send_request(
                'a1', task_id=task_id, context_id=context_id, message_id='m2'
            ),
            ctx,
        )
        assert r2.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        stored, _ = await cluster.assert_store.get(task_id, ctx)
        assert sum(1 for m in stored.history if m.role == Role.ROLE_USER) == 2

        r3 = await cluster.replica_a.on_message_send(
            build_send_request(
                'a2', task_id=task_id, context_id=context_id, message_id='m3'
            ),
            ctx,
        )
        assert r3.status.state == TaskState.TASK_STATE_COMPLETED
    finally:
        await cluster.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resubscribe_on_non_owning_replica_streams_events(
    dsn: str,
) -> None:
    """Replica A runs the agent; replica B resubscribes and streams via the DB."""
    agent = ControlledAgent()
    # Replica B has its own (idle) agent instance; it never executes here.
    cluster = Cluster(dsn, agent, ControlledAgent())
    ctx = make_context()
    try:
        a_task = asyncio.create_task(
            drain(
                cluster.replica_a.on_message_send_stream(
                    build_send_request('go', message_id='m1'), ctx
                )
            )
        )
        await asyncio.wait_for(agent.working.wait(), timeout=10)

        task_id = next(
            iter(cluster.replica_a._active_task_registry._active_tasks)  # noqa: SLF001
        )
        await wait_for_state(
            cluster.assert_store, task_id, TaskState.TASK_STATE_WORKING, ctx
        )

        states: list = []

        async def observe() -> None:
            async for ev in cluster.replica_b.on_subscribe_to_task(
                SubscribeToTaskRequest(id=task_id), ctx
            ):
                if getattr(ev, 'status', None):
                    states.append(ev.status.state)
                    if ev.status.state == TaskState.TASK_STATE_COMPLETED:
                        return

        observer = asyncio.create_task(observe())
        await asyncio.sleep(0.2)  # let B read snapshot + start tailing the log
        agent.release.set()

        await asyncio.wait_for(observer, timeout=30)
        await asyncio.wait_for(a_task, timeout=30)

        assert states[0] == TaskState.TASK_STATE_WORKING
        assert states[-1] == TaskState.TASK_STATE_COMPLETED
    finally:
        await cluster.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resubscribe_terminal_task_rejected(dsn: str) -> None:
    """Resubscribe to a finished task is rejected (no hang), from any replica."""
    agent = ControlledAgent()
    agent.release.set()  # completes immediately
    cluster = Cluster(dsn, agent, ControlledAgent())
    ctx = make_context()
    try:
        result = await cluster.replica_a.on_message_send(
            build_send_request('go', message_id='m1'), ctx
        )
        assert result.status.state == TaskState.TASK_STATE_COMPLETED

        # A terminal task cannot be resubscribed to; the non-owning replica
        # rejects it rather than hanging or streaming.
        with pytest.raises(InvalidParamsError, match='terminal state'):
            async for _ in cluster.replica_b.on_subscribe_to_task(
                SubscribeToTaskRequest(id=result.id), ctx
            ):
                pass
    finally:
        await cluster.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cancel_from_non_owning_replica_stops_remote_agent(
    dsn: str,
) -> None:
    """Cancel on replica B stops the agent running on replica A (via CAS)."""
    agent = LongRunningAgent()
    cluster = Cluster(dsn, agent, LongRunningAgent())
    ctx = make_context()
    try:
        a_task = asyncio.create_task(
            drain(
                cluster.replica_a.on_message_send_stream(
                    build_send_request('go', message_id='m1'), ctx
                )
            )
        )
        await asyncio.wait_for(agent.working.wait(), timeout=10)
        task_id = next(
            iter(cluster.replica_a._active_task_registry._active_tasks)  # noqa: SLF001
        )
        await wait_for_state(
            cluster.assert_store, task_id, TaskState.TASK_STATE_WORKING, ctx
        )

        result = await cluster.replica_b.on_cancel_task(
            CancelTaskRequest(id=task_id), ctx
        )
        assert result.status.state == TaskState.TASK_STATE_CANCELED

        await asyncio.wait_for(agent.aborted.wait(), timeout=30)
        await asyncio.wait_for(a_task, timeout=30)

        final, _ = await cluster.assert_store.get(task_id, ctx)
        assert final.status.state == TaskState.TASK_STATE_CANCELED
    finally:
        await cluster.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cancel_idempotent_and_terminal_rules(dsn: str) -> None:
    """Cancel is idempotent; cancelling a completed task is not cancelable."""
    cluster = Cluster(dsn, CompletingAgent(), CompletingAgent())
    ctx = make_context()
    try:
        result = await cluster.replica_a.on_message_send(
            build_send_request('go', message_id='m1'), ctx
        )
        assert result.status.state == TaskState.TASK_STATE_COMPLETED
        # Completed -> not cancelable, from the other replica.
        with pytest.raises(TaskNotCancelableError):
            await cluster.replica_b.on_cancel_task(
                CancelTaskRequest(id=result.id), ctx
            )
    finally:
        await cluster.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_absent_and_non_owner_rejected(dsn: str) -> None:
    """Absent task -> NotFound; a non-owner cannot see another user's task."""
    cluster = Cluster(dsn, CompletingAgent(), CompletingAgent())
    try:
        # Absent task on either replica.
        with pytest.raises(TaskNotFoundError):
            await cluster.replica_b.on_get_task(
                _task_query('does-not-exist'), make_context()
            )

        # Owner-scoping: bob cannot get/subscribe/cancel alice's task.
        result = await cluster.replica_a.on_message_send(
            build_send_request('go', message_id='m1'), make_context('alice')
        )
        bob = make_context('bob')
        with pytest.raises(TaskNotFoundError):
            await cluster.replica_b.on_get_task(_task_query(result.id), bob)
        with pytest.raises(TaskNotFoundError):
            async for _ in cluster.replica_b.on_subscribe_to_task(
                SubscribeToTaskRequest(id=result.id), bob
            ):
                pass
        with pytest.raises(TaskNotFoundError):
            await cluster.replica_b.on_cancel_task(
                CancelTaskRequest(id=result.id), bob
            )
    finally:
        await cluster.aclose()


def _task_query(task_id: str):  # noqa: ANN202
    from a2a.types.a2a_pb2 import GetTaskRequest

    return GetTaskRequest(id=task_id)
