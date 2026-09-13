"""Tests for wiring the optional event_bus into DefaultRequestHandlerV2.

Iteration 4 is pure wiring: the bus is threaded through the handler ->
ActiveTaskRegistry -> ActiveTask, and defaults to an in-process bus when not
supplied. No request-path behaviour changes here.
"""

import pytest

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.cluster import InMemoryTaskEventBus, TaskEventBus
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.default_request_handler_v2 import (
    DefaultRequestHandlerV2,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types.a2a_pb2 import AgentCard

from .conftest import make_context


class NoopExecutor(AgentExecutor):
    async def execute(self, context, event_queue) -> None:  # noqa: ANN001
        pass

    async def cancel(self, context, event_queue) -> None:  # noqa: ANN001
        pass


def make_handler(event_bus: TaskEventBus | None = None):
    return DefaultRequestHandlerV2(
        agent_executor=NoopExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=AgentCard(),
        event_bus=event_bus,
    )


def test_default_handler_creates_in_memory_bus() -> None:
    handler = make_handler()
    assert isinstance(handler._event_bus, InMemoryTaskEventBus)  # noqa: SLF001


def test_explicit_bus_is_stored() -> None:
    bus = InMemoryTaskEventBus()
    handler = make_handler(event_bus=bus)
    assert handler._event_bus is bus  # noqa: SLF001


def test_bus_is_threaded_to_registry() -> None:
    bus = InMemoryTaskEventBus()
    handler = make_handler(event_bus=bus)
    assert handler._active_task_registry._event_bus is bus  # noqa: SLF001


def test_default_alias_accepts_event_bus() -> None:
    # DefaultRequestHandler is an alias for v2; ensure the kwarg flows through.
    bus = InMemoryTaskEventBus()
    handler = DefaultRequestHandler(
        NoopExecutor(),
        InMemoryTaskStore(),
        AgentCard(),
        event_bus=bus,
    )
    assert handler._event_bus is bus  # noqa: SLF001


@pytest.mark.asyncio
async def test_active_task_receives_bus_from_registry() -> None:
    bus = InMemoryTaskEventBus()
    handler = make_handler(event_bus=bus)
    context = make_context()

    active_task = await handler._active_task_registry.get_or_create(  # noqa: SLF001
        'task-1',
        call_context=context,
        create_task_if_missing=True,
        initial_message=None,
    )
    try:
        assert active_task._event_bus is bus  # noqa: SLF001
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_none_event_bus_still_gets_default_on_active_task() -> None:
    # When no bus is passed, the ActiveTask still receives the handler's
    # default in-memory bus (uniform wiring), never None.
    handler = make_handler(event_bus=None)
    context = make_context()
    active_task = await handler._active_task_registry.get_or_create(  # noqa: SLF001
        'task-1',
        call_context=context,
        create_task_if_missing=True,
        initial_message=None,
    )
    try:
        assert isinstance(active_task._event_bus, InMemoryTaskEventBus)  # noqa: SLF001
    finally:
        await handler.aclose()
