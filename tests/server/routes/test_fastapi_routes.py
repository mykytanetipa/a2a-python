from unittest.mock import AsyncMock

import pytest

from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.routes.fastapi_routes import (
    _AGENT_CARD_TAG,
    _JSONRPC_TAG,
    _REST_TAG,
)
from a2a.types.a2a_pb2 import AgentCard, Task
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    PROTOCOL_VERSION_1_0,
    VERSION_HEADER,
)


fastapi = pytest.importorskip('fastapi')
from fastapi import (  # noqa: E402
    Depends,
    FastAPI,
    HTTPException,
    Header,
    Security,
)
from fastapi.security import APIKeyHeader  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.routing import Route, WebSocketRoute  # noqa: E402


pytestmark = pytest.mark.httpx2

_API_KEY = 'secret'
_VERSION = {VERSION_HEADER: PROTOCOL_VERSION_1_0}


def _require_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=403)


def _require_key_security(
    key: str | None = Security(
        APIKeyHeader(name='x-api-key', auto_error=False)
    ),
) -> None:
    if key != _API_KEY:
        raise HTTPException(status_code=403)


@pytest.fixture
def agent_card() -> AgentCard:
    return AgentCard(name='Test Agent', version='1.0.0')


@pytest.fixture
def mock_handler() -> AsyncMock:
    return AsyncMock(spec=RequestHandler)


def _build_app(
    agent_card: AgentCard,
    mock_handler: AsyncMock,
    app: FastAPI | None = None,
) -> FastAPI:
    app = app or FastAPI()
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        jsonrpc_routes=create_jsonrpc_routes(mock_handler, rpc_url='/'),
        rest_routes=create_rest_routes(mock_handler),
    )
    return app


def test_routes_appear_in_openapi_with_tags(
    agent_card: AgentCard, mock_handler: AsyncMock
) -> None:
    """Each group is documented and tagged for the Swagger UI."""
    app = _build_app(agent_card, mock_handler)
    paths = app.openapi()['paths']

    assert paths[AGENT_CARD_WELL_KNOWN_PATH]['get']['tags'] == [_AGENT_CARD_TAG]
    assert paths['/']['post']['tags'] == [_JSONRPC_TAG]
    assert paths['/message:send']['post']['tags'] == [_REST_TAG]
    assert paths['/tasks']['get']['tags'] == [_REST_TAG]


def test_routes_dispatch_under_fastapi(
    agent_card: AgentCard, mock_handler: AsyncMock
) -> None:
    """Re-registered routes still dispatch correctly under a FastAPI app."""
    mock_handler.on_message_send.return_value = Task(id='task-123')

    app = _build_app(agent_card, mock_handler)
    client = TestClient(app)

    assert client.get(AGENT_CARD_WELL_KNOWN_PATH).json()['name'] == 'Test Agent'
    rpc_response = client.post(
        '/', json={'jsonrpc': '2.0', 'id': '1', 'method': 'NoSuchMethod'}
    ).json()
    assert rpc_response['error']['code'] == -32601

    rest_response = client.post(
        '/message:send',
        json={},
        headers={VERSION_HEADER: PROTOCOL_VERSION_1_0},
    )
    assert rest_response.status_code == 200
    assert rest_response.json()['task']['id'] == 'task-123'


def test_tenant_mount_still_dispatches(mock_handler: AsyncMock) -> None:
    """`Mount` entries (tenant routing) keep dispatching after registration."""
    mock_handler.on_message_send.return_value = Task(id='tenant-task')

    app = FastAPI()
    add_a2a_routes_to_fastapi(app, rest_routes=create_rest_routes(mock_handler))
    client = TestClient(app)

    response = client.post(
        '/my-tenant/message:send',
        json={},
        headers={VERSION_HEADER: PROTOCOL_VERSION_1_0},
    )
    assert response.status_code == 200
    context = mock_handler.on_message_send.call_args[0][1]
    assert context.tenant == 'my-tenant'


def test_partial_groups(agent_card: AgentCard, mock_handler: AsyncMock) -> None:
    """Calling with only a subset of groups works and tags only those."""
    app = FastAPI()
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
    )
    paths = app.openapi()['paths']
    assert list(paths.keys()) == [AGENT_CARD_WELL_KNOWN_PATH]


def test_request_body_schemas_are_attached(
    agent_card: AgentCard, mock_handler: AsyncMock
) -> None:
    """JSON-RPC and REST POST bodies expose schemas derived from proto types."""
    app = _build_app(agent_card, mock_handler)
    schema = app.openapi()
    components = schema['components']['schemas']

    assert 'A2ARequest' in components
    assert components['A2ARequest']['properties']['method']['enum']
    rpc_body = schema['paths']['/']['post']['requestBody']
    assert rpc_body['content']['application/json']['schema'] == {
        '$ref': '#/components/schemas/A2ARequest'
    }

    send_body = schema['paths']['/message:send']['post']['requestBody']
    assert send_body['required'] is True
    assert send_body['content']['application/json']['schema'] == {
        '$ref': '#/components/schemas/SendMessageRequest'
    }

    cancel_body = schema['paths']['/tasks/{id}:cancel']['post']['requestBody']
    assert cancel_body['required'] is False
    assert cancel_body['content']['application/json']['schema'] == {
        '$ref': '#/components/schemas/CancelTaskRequest'
    }

    assert 'Message' in components
    assert 'Part' in components
    assert components['Message']['properties']['role']['enum'] == [
        'ROLE_UNSPECIFIED',
        'ROLE_USER',
        'ROLE_AGENT',
    ]


def test_routes_without_body_have_no_request_body(
    agent_card: AgentCard, mock_handler: AsyncMock
) -> None:
    """GET/DELETE/parameterless POST routes don't get a fabricated body."""
    app = _build_app(agent_card, mock_handler)
    paths = app.openapi()['paths']

    assert 'requestBody' not in paths[AGENT_CARD_WELL_KNOWN_PATH]['get']
    assert 'requestBody' not in paths['/tasks']['get']
    assert 'requestBody' not in paths['/tasks/{id}:subscribe']['post']
    assert (
        'requestBody'
        not in paths['/tasks/{id}/pushNotificationConfigs/{push_id}']['delete']
    )


def test_a2a_version_header_on_dispatcher_routes(
    agent_card: AgentCard, mock_handler: AsyncMock
) -> None:
    """JSON-RPC and REST routes declare the version header so Swagger pre-fills it."""
    app = _build_app(agent_card, mock_handler)
    paths = app.openapi()['paths']

    def _has_version_header(op: dict) -> bool:
        return any(
            p.get('name') == VERSION_HEADER for p in op.get('parameters', [])
        )

    assert _has_version_header(paths['/']['post'])
    assert _has_version_header(paths['/message:send']['post'])
    assert _has_version_header(paths['/tasks']['get'])
    assert _has_version_header(paths['/tasks/{id}:cancel']['post'])
    assert not _has_version_header(paths[AGENT_CARD_WELL_KNOWN_PATH]['get'])


_PROTECTED_REQUESTS = [
    pytest.param('GET', AGENT_CARD_WELL_KNOWN_PATH, None, id='agent-card'),
    pytest.param(
        'POST',
        '/',
        {
            'jsonrpc': '2.0',
            'id': '1',
            'method': 'GetTask',
            'params': {'id': 't'},
        },
        id='jsonrpc',
    ),
    pytest.param('POST', '/message:send', {}, id='rest'),
    pytest.param('GET', '/tasks/t', None, id='rest-get-task'),
    pytest.param('POST', '/my-tenant/message:send', {}, id='rest-tenant'),
    pytest.param('GET', '/my-tenant/tasks/t', None, id='rest-tenant-get-task'),
]


@pytest.mark.parametrize(('method', 'path', 'json'), _PROTECTED_REQUESTS)
@pytest.mark.parametrize(
    'dependency',
    [Depends(_require_key), Security(_require_key_security)],
    ids=['depends', 'security'],
)
def test_app_dependencies_guard_a2a_routes(  # noqa: PLR0913
    agent_card: AgentCard,
    mock_handler: AsyncMock,
    dependency: object,
    method: str,
    path: str,
    json: dict | None,
) -> None:
    """App-level dependencies run for every A2A route, as for any other."""
    mock_handler.on_message_send.return_value = Task(id='t')
    mock_handler.on_get_task.return_value = Task(id='t')
    app = _build_app(
        agent_card, mock_handler, FastAPI(dependencies=[dependency])
    )
    client = TestClient(app)

    denied = client.request(method, path, json=json, headers=_VERSION)
    assert denied.status_code == 403
    assert not mock_handler.method_calls

    allowed = client.request(
        method, path, json=json, headers={**_VERSION, 'x-api-key': _API_KEY}
    )
    assert allowed.status_code == 200


def test_app_dependencies_guard_streaming(
    agent_card: AgentCard, mock_handler: AsyncMock
) -> None:
    """A rejected stream request never reaches the handler."""
    app = _build_app(
        agent_card, mock_handler, FastAPI(dependencies=[Depends(_require_key)])
    )

    response = TestClient(app).post(
        '/message:stream', json={}, headers=_VERSION
    )

    assert response.status_code == 403
    mock_handler.on_message_send_stream.assert_not_called()


def test_tenant_routes_are_documented(mock_handler: AsyncMock) -> None:
    """Tenant routes are registered as API routes with the REST extras."""
    app = FastAPI()
    add_a2a_routes_to_fastapi(app, rest_routes=create_rest_routes(mock_handler))
    operation = app.openapi()['paths']['/{tenant}/message:send']['post']

    assert operation['tags'] == [_REST_TAG]
    assert operation['requestBody']['content']['application/json'][
        'schema'
    ] == {'$ref': '#/components/schemas/SendMessageRequest'}
    assert any(p['name'] == VERSION_HEADER for p in operation['parameters'])


def test_unsupported_route_type_is_rejected() -> None:
    """Routes the helper cannot register are rejected, not left unguarded."""

    async def endpoint(websocket: object) -> None: ...

    with pytest.raises(TypeError, match='Unsupported A2A route'):
        add_a2a_routes_to_fastapi(
            FastAPI(), rest_routes=[WebSocketRoute('/ws', endpoint)]
        )


def test_route_without_methods_is_rejected() -> None:
    """A method-less route would match every method, so it is rejected."""

    async def endpoint(request: object) -> None: ...

    route = Route('/raw', endpoint)
    route.methods = None

    with pytest.raises(TypeError, match='Unsupported A2A route'):
        add_a2a_routes_to_fastapi(FastAPI(), rest_routes=[route])
