"""Shared fixtures and helpers for a2a.server.cluster tests.

Consolidates the scaffolding that the multi-replica tests need: a test User,
context factory, agent card, a message-request builder, reusable
AgentExecutor implementations, and a replica (handler) factory that wires a
shared store + bus. Two handlers built from the same store and bus simulate two
replicas behind a load balancer.
"""

import asyncio
import contextlib

import pytest

from a2a.auth.user import User
from a2a.helpers.proto_helpers import new_task_from_user_message
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.cluster import (
    InMemoryTaskEventBus,
    TaskEventBus,
    VersionedInMemoryTaskStore,
    VersionedTaskStore,
)
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.default_request_handler_v2 import (
    DefaultRequestHandlerV2,
)
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)


# --- Identity / context helpers ---------------------------------------------


class SampleUser(User):
    """Minimal authenticated User for tests."""

    def __init__(self, user_name: str = 'test_user') -> None:
        self._user_name = user_name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._user_name


def make_context(user: str = 'test_user') -> ServerCallContext:
    """Builds a ServerCallContext for the given user name."""
    return ServerCallContext(user=SampleUser(user))


def streaming_agent_card() -> AgentCard:
    """An AgentCard that advertises streaming support."""
    return AgentCard(capabilities=AgentCapabilities(streaming=True))


def build_send_request(
    text: str,
    task_id: str = '',
    context_id: str = '',
    message_id: str = 'm',
) -> SendMessageRequest:
    """Builds a SendMessageRequest, optionally continuing an existing task."""
    msg = Message(
        role=Role.ROLE_USER,
        message_id=message_id,
        parts=[Part(text=text)],
    )
    if task_id:
        msg.task_id = task_id
    if context_id:
        msg.context_id = context_id
    return SendMessageRequest(message=msg)


async def drain(agen) -> None:  # noqa: ANN001
    """Consumes an async generator to completion, swallowing exceptions."""
    with contextlib.suppress(Exception):
        async for _ in agen:
            pass


async def wait_for_state(
    store: VersionedTaskStore,
    task_id: str,
    state: TaskState,
    context: ServerCallContext,
    *,
    tries: int = 100,
    delay: float = 0.02,
) -> None:
    """Polls the store until the task reaches `state` (for async persistence)."""
    for _ in range(tries):
        task, _ = await store.get(task_id, context)
        if task is not None and task.status.state == state:
            return
        await asyncio.sleep(delay)
    raise AssertionError(f'Task {task_id} did not reach {state} in time')


# --- Reusable agent executors -----------------------------------------------


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
    """Creates a task, goes WORKING, then completes only when released.

    `working` is set once execution reaches WORKING; the agent then waits on
    `release` before emitting an artifact and completing. Useful for observing a
    task mid-flight from another replica.
    """

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
    """Goes WORKING then loops, saving periodically until aborted.

    Periodic saves let a remote cancel be observed as a CAS conflict, which
    aborts the execution (`aborted` is set on CancelledError).
    """

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

    Derives the turn purely from durable task history, so it behaves correctly
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


# --- Replica / infra factories ----------------------------------------------


def make_replica(
    store: VersionedTaskStore,
    bus: TaskEventBus,
    agent: AgentExecutor,
) -> DefaultRequestHandlerV2:
    """Builds a handler ('replica') wired to a shared store and bus."""
    return DefaultRequestHandlerV2(
        agent_executor=agent,
        task_store=store,
        agent_card=streaming_agent_card(),
        event_bus=bus,
    )


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def shared_store() -> VersionedInMemoryTaskStore:
    """A versioned in-memory store shared across replicas in a test."""
    return VersionedInMemoryTaskStore()


@pytest.fixture
def shared_bus() -> InMemoryTaskEventBus:
    """An in-memory event bus shared across replicas in a test."""
    return InMemoryTaskEventBus()


@pytest.fixture
def context() -> ServerCallContext:
    """A default server call context for 'test_user'."""
    return make_context()
