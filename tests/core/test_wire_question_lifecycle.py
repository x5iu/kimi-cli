from __future__ import annotations

from pathlib import Path

import pytest
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.wire.jsonrpc import (
    JSONRPCErrorObject,
    JSONRPCErrorResponse,
    JSONRPCRequestMessage,
    JSONRPCSuccessResponse,
)
from kimi_cli.wire.server import WireServer
from kimi_cli.wire.types import QuestionItem, QuestionNotSupported, QuestionOption, QuestionRequest


def _make_server(runtime: Runtime, tmp_path: Path) -> WireServer:
    soul = KimiSoul(
        Agent(
            name="Wire Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )
    return WireServer(soul)


def _make_request() -> QuestionRequest:
    return QuestionRequest(
        id="question-1",
        tool_call_id="tool-1",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[QuestionOption(label="JSON"), QuestionOption(label="YAML")],
            )
        ],
    )


@pytest.mark.asyncio
async def test_request_question_sets_exception_when_client_unsupported(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    server = _make_server(runtime, tmp_path)
    server._client_supports_question = False
    request = _make_request()

    await server._request_question(request)

    with pytest.raises(QuestionNotSupported):
        await request.wait()
    assert server._pending_requests == {}


@pytest.mark.asyncio
async def test_request_question_registers_pending_request_and_sends_jsonrpc_request(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_server(runtime, tmp_path)
    server._client_supports_question = True
    request = _make_request()
    sent: list[JSONRPCRequestMessage] = []

    async def fake_send_msg(msg):
        assert isinstance(msg, JSONRPCRequestMessage)
        sent.append(msg)

    monkeypatch.setattr(server, "_send_msg", fake_send_msg)

    await server._request_question(request)

    assert server._pending_requests == {request.id: request}
    assert sent == [JSONRPCRequestMessage(id=request.id, params=request)]
    assert request.resolved is False


@pytest.mark.asyncio
async def test_handle_response_resolves_pending_question_request(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    server = _make_server(runtime, tmp_path)
    request = _make_request()
    server._pending_requests[request.id] = request

    await server._handle_response(
        JSONRPCSuccessResponse(
            id=request.id,
            result={
                "request_id": request.id,
                "answers": {"Which format should I use?": "JSON"},
            },
        )
    )

    assert await request.wait() == {"Which format should I use?": "JSON"}
    assert server._pending_requests == {}


@pytest.mark.asyncio
async def test_handle_response_error_resolves_question_request_to_empty_answers(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    server = _make_server(runtime, tmp_path)
    request = _make_request()
    server._pending_requests[request.id] = request

    await server._handle_response(
        JSONRPCErrorResponse(
            id=request.id,
            error=JSONRPCErrorObject(code=-32000, message="client cancelled"),
        )
    )

    assert await request.wait() == {}
    assert server._pending_requests == {}


@pytest.mark.asyncio
async def test_shutdown_resolves_pending_question_requests_to_empty_answers(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    server = _make_server(runtime, tmp_path)
    request = _make_request()
    server._pending_requests[request.id] = request

    await server._shutdown()

    assert await request.wait() == {}
    assert server._pending_requests == {}
