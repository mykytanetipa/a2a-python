from __future__ import annotations

import asyncio  # noqa: TC003
import logging
import warnings

from typing import TYPE_CHECKING, Any, cast

from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
    RequestContextBuilder,
    SimpleRequestContextBuilder,
)
from a2a.server.agent_execution.active_task import (
    INTERRUPTED_TASK_STATES,
    TERMINAL_TASK_STATES,
    is_final_event,
)
from a2a.server.agent_execution.active_task_registry import ActiveTaskRegistry
from a2a.server.cluster.event_bus import VersionedEvent
from a2a.server.cluster.inmemory_event_bus import InMemoryTaskEventBus
from a2a.server.cluster.task_store import (
    ConcurrentTaskModificationError,
    LegacyTaskStoreAdapter,
    PlainTaskStoreView,
    VersionedTaskStore,
)
from a2a.server.request_handlers.request_handler import (
    RequestHandler,
    validate,
    validate_request_params,
)
from a2a.types.a2a_pb2 import (
    AgentCard,
    CancelTaskRequest,
    DeleteTaskPushNotificationConfigRequest,
    GetExtendedAgentCardRequest,
    GetTaskPushNotificationConfigRequest,
    GetTaskRequest,
    ListTaskPushNotificationConfigsRequest,
    ListTaskPushNotificationConfigsResponse,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskPushNotificationConfig,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import (
    ExtendedAgentCardNotConfiguredError,
    InternalError,
    InvalidParamsError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
)
from a2a.utils.task import (
    apply_history_length,
    validate_history_length,
    validate_page_size,
)
from a2a.utils.telemetry import SpanKind, trace_class


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from a2a.server.agent_execution.active_task import ActiveTask
    from a2a.server.cluster.event_bus import TaskEventBus
    from a2a.server.cluster.version import TaskVersion
    from a2a.server.context import ServerCallContext
    from a2a.server.events import Event
    from a2a.server.tasks import (
        PushNotificationConfigStore,
        PushNotificationSender,
        TaskStore,
    )


logger = logging.getLogger(__name__)

# Bounded retry for cancel-via-CAS when racing an active execution.
_MAX_CANCEL_ATTEMPTS = 10


@trace_class(kind=SpanKind.SERVER)
class DefaultRequestHandlerV2(RequestHandler):
    """Default request handler for all incoming requests.

    The ``queue_manager`` parameter is accepted for signature compatibility
    with `DefaultRequestHandler` but is not used: v2 delegates event streaming
    to an in-memory `ActiveTaskRegistry`. Passing a non-``None`` value emits a
    `DeprecationWarning` and logs a warning.
    """

    _background_tasks: set[asyncio.Task]

    def __init__(  # noqa: PLR0913
        self,
        agent_executor: AgentExecutor,
        task_store: TaskStore,
        agent_card: AgentCard,
        queue_manager: Any
        | None = None,  # Accepted for signature compat; ignored in v2 (warns)
        push_config_store: PushNotificationConfigStore | None = None,
        push_sender: PushNotificationSender | None = None,
        request_context_builder: RequestContextBuilder | None = None,
        extended_agent_card: AgentCard | None = None,
        extended_card_modifier: Callable[
            [AgentCard, ServerCallContext], Awaitable[AgentCard]
        ]
        | None = None,
        push_url_validator: Callable[[str], Awaitable[bool]] | None = None,
        event_bus: TaskEventBus | None = None,
    ) -> None:
        if queue_manager is not None:
            message = (
                'A queue_manager was passed to DefaultRequestHandlerV2, but it '
                'is not used: v2 delegates event streaming to an in-memory '
                'ActiveTaskRegistry, so custom or distributed QueueManager '
                'implementations are ignored. For multi-replica event '
                'streaming, either use LegacyRequestHandler or route '
                'subscription requests to the replica holding the task.'
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            logger.warning(message)
        self.agent_executor = agent_executor
        # Two views of the same store:
        #  * _versioned_store: reads a snapshot with its version and does CAS
        #    writes (used by subscribe/cancel and the TaskManager).
        #  * task_store: the plain TaskStore contract used by the read/utility
        #    paths (on_get_task, on_list_tasks, the request-context builder).
        # A VersionedTaskStore passed as task_store is exposed plainly via
        # PlainTaskStoreView; a plain TaskStore is adapted to the versioned one.
        if isinstance(task_store, VersionedTaskStore):
            self._versioned_store = task_store
            self.task_store: TaskStore = PlainTaskStoreView(task_store)
        else:
            self._versioned_store = LegacyTaskStoreAdapter(task_store)
            self.task_store = task_store
        self._agent_card = agent_card
        self._push_config_store = push_config_store
        self._push_sender = push_sender
        self._push_url_validator = push_url_validator
        self.extended_agent_card = extended_agent_card
        self.extended_card_modifier = extended_card_modifier
        # Optional cross-replica event bus. When None, the server behaves as a
        # single process: a default in-memory bus is used so the wiring is
        # uniform, but it never crosses a process boundary. Pass a shared
        # implementation (e.g. DatabaseTaskEventBus) together with a shared
        # durable task_store to enable multi-replica streaming.
        self._event_bus = event_bus or InMemoryTaskEventBus()
        self._request_context_builder = (
            request_context_builder
            or SimpleRequestContextBuilder(
                should_populate_referred_tasks=False, task_store=self.task_store
            )
        )
        self._active_task_registry = ActiveTaskRegistry(
            # The registry's TaskManager needs the versioned store so its saves
            # perform CAS; a plain store is transparently adapted by TaskManager.
            agent_executor=self.agent_executor,
            task_store=self._versioned_store,
            push_sender=self._push_sender,
            event_bus=self._event_bus,
        )
        self._background_tasks = set()

    async def _reject_unsafe_push_url(self, url: str) -> None:
        """Apply the configured push-URL policy, if any."""
        if self._push_url_validator is None:
            return
        if not await self._push_url_validator(url):
            raise InvalidParamsError(message='Invalid push notification URL')

    async def aclose(self) -> None:
        """Shuts down the handler, draining all active tasks.

        Drains the ``ActiveTaskRegistry`` so a server shutdown leaves no
        pending ``asyncio.Task``. Intended to be wired into an ASGI
        ``lifespan`` / ``on_shutdown`` hook. Safe to call multiple times.
        """
        await self._active_task_registry.aclose()

    @validate_request_params
    async def on_get_task(  # noqa: D102
        self,
        params: GetTaskRequest,
        context: ServerCallContext,
    ) -> Task | None:
        validate_history_length(params)

        task_id = params.id
        task: Task | None = await self.task_store.get(task_id, context)
        if not task:
            raise TaskNotFoundError

        return apply_history_length(task, params)

    @validate_request_params
    async def on_list_tasks(  # noqa: D102
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        validate_history_length(params)
        if params.HasField('page_size'):
            validate_page_size(params.page_size)

        page = await self.task_store.list(params, context)
        for task in page.tasks:
            if not params.include_artifacts:
                task.ClearField('artifacts')

            updated_task = apply_history_length(task, params)
            if updated_task is not task:
                task.CopyFrom(updated_task)

        return page

    @validate_request_params
    async def on_cancel_task(  # noqa: D102
        self,
        params: CancelTaskRequest,
        context: ServerCallContext,
    ) -> Task | None:
        task_id = params.id

        # Owner-scoped read of the current state (also the idempotency and
        # cancelability check). Masked as not-found so existence is not leaked.
        task, version = await self._versioned_store.get(task_id, context)
        if task is None:
            raise TaskNotFoundError

        # Idempotent: already cancelled is success, not an error.
        if task.status.state == TaskState.TASK_STATE_CANCELED:
            return task
        if task.status.state in TERMINAL_TASK_STATES:
            raise TaskNotCancelableError

        # Fast path: this replica is running the agent -> stop it directly.
        local = await self._active_task_registry.get(task_id)
        if local is not None and local.has_running_execution:
            try:
                result = await local.cancel(context)
            except InvalidParamsError as e:
                raise TaskNotCancelableError from e
            if isinstance(result, Message):
                raise InternalError(
                    message='Cancellation returned a message instead of a task.'
                )
            return result

        # Executing on another replica (or nowhere): record the cancellation in
        # shared state. The owning replica observes the version bump on its next
        # save (CAS failure) and aborts its execution.
        return await self._cancel_remote(task_id, task, version, context)

    async def _cancel_remote(
        self,
        task_id: str,
        task: Task,
        version: TaskVersion,
        context: ServerCallContext,
    ) -> Task:
        """Cancels a task whose agent is running on another replica.

        Writes the CANCELED state with a compare-and-swap: the version bump
        makes the owning replica's next save fail, which aborts its execution.
        No RPC to that replica is needed. Requires a versioned store; with an
        unversioned store this degrades to a best-effort last-writer-wins write.
        """
        current: Task = task
        for _ in range(_MAX_CANCEL_ATTEMPTS):
            cancelled = Task()
            cancelled.CopyFrom(current)
            cancelled.status.state = TaskState.TASK_STATE_CANCELED
            event = TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=current.context_id,
                status=cancelled.status,
            )
            try:
                new_version = await self._versioned_store.save(
                    cancelled,
                    event=event,
                    prev=current,
                    prev_version=version,
                    context=context,
                )
            except ConcurrentTaskModificationError:
                # The owning replica wrote between our read and our write.
                # Reload and decide again.
                reloaded, version = await self._versioned_store.get(
                    task_id, context
                )
                if reloaded is None:
                    raise TaskNotFoundError from None
                if reloaded.status.state == TaskState.TASK_STATE_CANCELED:
                    return reloaded
                if reloaded.status.state in TERMINAL_TASK_STATES:
                    raise TaskNotCancelableError from None
                current = reloaded
                continue

            # Notify any subscribers (on any replica) tailing this task.
            await self._event_bus.publish(
                task_id, VersionedEvent(event=event, version=new_version)
            )
            return cancelled

        raise InternalError(
            message=f'Could not cancel task {task_id} after '
            f'{_MAX_CANCEL_ATTEMPTS} attempts due to concurrent updates'
        )

    def _validate_task_id_match(self, task_id: str, event_task_id: str) -> None:
        if task_id != event_task_id:
            logger.error(
                'Agent generated task_id=%s does not match the RequestContext task_id=%s.',
                event_task_id,
                task_id,
            )
            raise InternalError(message='Task ID mismatch in agent response')

    async def _setup_active_task(
        self,
        params: SendMessageRequest,
        call_context: ServerCallContext,
    ) -> tuple[ActiveTask, RequestContext]:
        validate_history_length(params.configuration)

        original_task_id = params.message.task_id or None
        original_context_id = params.message.context_id or None

        if original_task_id:
            task = await self.task_store.get(original_task_id, call_context)
            if not task:
                raise TaskNotFoundError(f'Task {original_task_id} not found')

        # Build context to resolve or generate missing IDs
        request_context = await self._request_context_builder.build(
            params=params,
            task_id=original_task_id,
            context_id=original_context_id,
            # We will get the task when we have to process the request to avoid concurrent read/write issues.
            task=None,
            context=call_context,
        )

        task_id = cast('str', request_context.task_id)
        context_id = cast('str', request_context.context_id)

        if self._push_config_store and params.configuration.HasField(
            'task_push_notification_config'
        ):
            await self._reject_unsafe_push_url(
                params.configuration.task_push_notification_config.url
            )
            await self._push_config_store.set_info(
                task_id,
                params.configuration.task_push_notification_config,
                call_context,
            )

        active_task = await self._active_task_registry.get_or_create(
            task_id,
            context_id=context_id,
            call_context=call_context,
            create_task_if_missing=True,
            initial_message=request_context.message,
        )

        return active_task, request_context

    @validate_request_params
    async def on_message_send(  # noqa: D102
        self,
        params: SendMessageRequest,
        context: ServerCallContext,
    ) -> Message | Task:
        active_task, request_context = await self._setup_active_task(
            params, context
        )
        task_id = cast('str', request_context.task_id)

        result: Message | Task | None = None

        async for raw_event in active_task.subscribe(
            request=request_context,
            include_initial_task=False,
            replace_status_update_with_task=True,
        ):
            event = raw_event
            logger.debug(
                'Processing[%s] event [%s] %s',
                params.message.task_id,
                type(event).__name__,
                event,
            )
            if isinstance(event, Task) and (
                params.configuration.return_immediately
                or event.status.state
                in (TERMINAL_TASK_STATES | INTERRUPTED_TASK_STATES)
            ):
                self._validate_task_id_match(task_id, event.id)
                result = event
                # A FAILED task may be followed by a producer exception. Keep
                # the task as the fallback result, but let the subscription
                # surface that exception or finish the current request.
                if (
                    params.configuration.return_immediately
                    or event.status.state != TaskState.TASK_STATE_FAILED
                ):
                    # AgentExecutor will continue to run in the background
                    # when return_immediately is set.
                    break

            if isinstance(event, Message):
                result = event
                # Do NOT break here as Message is supposed to be the only
                # event in "Message-only" interaction.
                # ActiveTask consumer (see active_task.py) validates the event
                # stream and raises InvalidAgentResponseError if more events are
                # pushed after a Message.

        if result is None:
            logger.debug('Missing result for task %s', request_context.task_id)
            result = await active_task.get_task()

        if isinstance(result, Task):
            result = apply_history_length(result, params.configuration)

        logger.debug(
            'Returning result for task %s: %s',
            request_context.task_id,
            result,
        )
        return result

    @validate_request_params
    @validate(
        lambda self: self._agent_card.capabilities.streaming,
        'Streaming is not supported by the agent',
    )
    async def on_message_send_stream(  # noqa: D102
        self,
        params: SendMessageRequest,
        context: ServerCallContext,
    ) -> AsyncGenerator[Event, None]:
        active_task, request_context = await self._setup_active_task(
            params, context
        )

        task_id = cast('str', request_context.task_id)

        async for event in active_task.subscribe(
            request=request_context,
            include_initial_task=False,
        ):
            # Do NOT break here as we rely on AgentExecutor to yield control.
            # ActiveTask consumer (see active_task.py) validates the event
            # stream and raises InvalidAgentResponseError on misbehaving agents:
            #   - an event after a Message
            #   - Message after entering task mode
            #   - an event after a terminal state
            if isinstance(event, Task):
                self._validate_task_id_match(task_id, event.id)
                yield apply_history_length(event, params.configuration)
            else:
                yield event

    @validate_request_params
    @validate(
        lambda self: self._agent_card.capabilities.push_notifications,
        error_message='Push notifications are not supported by the agent',
        error_type=PushNotificationNotSupportedError,
    )
    async def on_create_task_push_notification_config(  # noqa: D102
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext,
    ) -> TaskPushNotificationConfig:
        if not self._push_config_store:
            raise PushNotificationNotSupportedError

        task_id = params.task_id
        task: Task | None = await self.task_store.get(task_id, context)
        if not task:
            raise TaskNotFoundError

        await self._reject_unsafe_push_url(params.url)

        await self._push_config_store.set_info(
            task_id,
            params,
            context,
        )

        return params

    @validate_request_params
    @validate(
        lambda self: self._agent_card.capabilities.push_notifications,
        error_message='Push notifications are not supported by the agent',
        error_type=PushNotificationNotSupportedError,
    )
    async def on_get_task_push_notification_config(  # noqa: D102
        self,
        params: GetTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> TaskPushNotificationConfig:
        if not self._push_config_store:
            raise PushNotificationNotSupportedError

        task_id = params.task_id
        config_id = params.id
        task: Task | None = await self.task_store.get(task_id, context)
        if not task:
            raise TaskNotFoundError

        push_notification_configs: list[TaskPushNotificationConfig] = (
            await self._push_config_store.get_info(task_id, context) or []
        )

        for config in push_notification_configs:
            if config.id == config_id:
                return config

        raise TaskNotFoundError

    @validate_request_params
    @validate(
        lambda self: self._agent_card.capabilities.streaming,
        'Streaming is not supported by the agent',
    )
    async def on_subscribe_to_task(  # noqa: D102
        self,
        params: SubscribeToTaskRequest,
        context: ServerCallContext,
    ) -> AsyncGenerator[Event, None]:
        task_id = params.id

        # Owner-scoped existence check (issue #1159, CWE-639): resolve the task
        # under the caller's context so a non-owner cannot subscribe to another
        # user's task. Masked as not-found so existence is not leaked. This
        # guards both the local and remote paths below.
        task, snapshot_version = await self._versioned_store.get(
            task_id, context
        )
        if task is None:
            raise TaskNotFoundError

        # A task that has already reached a terminal state cannot be resubscribed
        # to (there is nothing further to stream). This matches the behaviour of
        # ActiveTask.start(), which rejects terminal tasks, and is validated by
        # the resubscription scenario tests.
        if task.status.state in TERMINAL_TASK_STATES:
            raise InvalidParamsError(
                message=f'Task {task_id} is in terminal state: '
                f'{task.status.state}'
            )

        # Fast path: this replica is running the agent, so tap its live queue.
        local = await self._active_task_registry.get(task_id)
        if local is not None:
            async for event in local.subscribe(include_initial_task=True):
                yield event
            return

        # Otherwise serve from shared state. Crucially do NOT get_or_create: that
        # would spawn a producer/consumer pair with nothing to feed them, so the
        # stream would emit only the snapshot and then hang forever. Instead emit
        # the snapshot and tail the shared event bus.
        async for event in self._subscribe_remote(
            task_id, context, task, snapshot_version
        ):
            yield event

    async def _subscribe_remote(
        self,
        task_id: str,
        context: ServerCallContext,
        task: Task,
        snapshot_version: TaskVersion,
    ) -> AsyncGenerator[Event, None]:
        """Serves a resubscription for a task executing on another replica.

        Emits the given task snapshot first (per spec), then tails the shared
        event bus for subsequent events, discarding any already reflected in the
        snapshot. Ends when a final event is seen. Requires a shared durable
        store and a shared bus to be useful; with the in-process defaults it
        simply serves the local snapshot. The caller has already rejected
        terminal tasks. An interrupted (input_required/auth_required) task is
        still tailed, because a subsequent request may resume it and the
        subscriber should observe those events.
        """
        del context  # snapshot already read by the caller under this context

        yield task

        subscription = self._event_bus.subscribe(
            task_id, after=snapshot_version
        )
        try:
            async for versioned in subscription:
                if not versioned.version.is_after(snapshot_version):
                    continue
                yield versioned.event
                if is_final_event(versioned.event):
                    return
        finally:
            # Close the underlying async generator so its cleanup (e.g.
            # deregistering the subscriber) runs promptly.
            aclose = getattr(subscription, 'aclose', None)
            if aclose is not None:
                await aclose()

    @validate_request_params
    @validate(
        lambda self: self._agent_card.capabilities.push_notifications,
        error_message='Push notifications are not supported by the agent',
        error_type=PushNotificationNotSupportedError,
    )
    async def on_list_task_push_notification_configs(  # noqa: D102
        self,
        params: ListTaskPushNotificationConfigsRequest,
        context: ServerCallContext,
    ) -> ListTaskPushNotificationConfigsResponse:
        if not self._push_config_store:
            raise PushNotificationNotSupportedError

        task_id = params.task_id
        task: Task | None = await self.task_store.get(task_id, context)
        if not task:
            raise TaskNotFoundError

        push_notification_config_list = await self._push_config_store.get_info(
            task_id, context
        )

        return ListTaskPushNotificationConfigsResponse(
            configs=push_notification_config_list
        )

    @validate_request_params
    @validate(
        lambda self: self._agent_card.capabilities.push_notifications,
        error_message='Push notifications are not supported by the agent',
        error_type=PushNotificationNotSupportedError,
    )
    async def on_delete_task_push_notification_config(  # noqa: D102
        self,
        params: DeleteTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> None:
        if not self._push_config_store:
            raise PushNotificationNotSupportedError

        task_id = params.task_id
        config_id = params.id
        task: Task | None = await self.task_store.get(task_id, context)
        if not task:
            raise TaskNotFoundError

        await self._push_config_store.delete_info(task_id, context, config_id)

    @validate_request_params
    @validate(
        lambda self: self._agent_card.capabilities.extended_agent_card,
        error_message='The agent does not support authenticated extended cards',
    )
    async def on_get_extended_agent_card(
        self,
        params: GetExtendedAgentCardRequest,
        context: ServerCallContext,
    ) -> AgentCard:
        """Default handler for 'GetExtendedAgentCard'.

        Requires `capabilities.extended_agent_card` to be true.
        """
        extended_card = self.extended_agent_card
        if not extended_card:
            raise ExtendedAgentCardNotConfiguredError

        if self.extended_card_modifier:
            extended_card = await self.extended_card_modifier(
                extended_card, context
            )

        return extended_card
