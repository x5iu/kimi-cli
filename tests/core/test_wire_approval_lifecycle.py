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
from kimi_cli.wire.types import ApprovalRequest


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


@pytest.mark.asyncio
async def test_request_approval_registers_pending_request_and_sends_jsonrpc_request(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_server(runtime, tmp_path)
    request = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="Shell",
        action="run command",
        description="Run command.",
    )
    sent: list[JSONRPCRequestMessage] = []

    async def fake_send_msg(msg):
        assert isinstance(msg, JSONRPCRequestMessage)
        sent.append(msg)

    monkeypatch.setattr(server, "_send_msg", fake_send_msg)

    await server._request_approval(request)

    assert server._pending_requests == {request.id: request}
    assert sent == [JSONRPCRequestMessage(id=request.id, params=request)]
    assert request.resolved is False


@pytest.mark.asyncio
async def test_handle_response_resolves_pending_approval_request(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    server = _make_server(runtime, tmp_path)
    request = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="Shell",
        action="run command",
        description="Run command.",
    )
    server._pending_requests[request.id] = request

    await server._handle_response(
        JSONRPCSuccessResponse(
            id=request.id,
            result={"request_id": request.id, "response": "approve_for_session"},
        )
    )

    assert await request.wait() == "approve_for_session"
    assert server._pending_requests == {}


@pytest.mark.asyncio
async def test_handle_response_error_rejects_pending_approval_request(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    server = _make_server(runtime, tmp_path)
    request = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="Shell",
        action="run command",
        description="Run command.",
    )
    server._pending_requests[request.id] = request

    await server._handle_response(
        JSONRPCErrorResponse(
            id=request.id,
            error=JSONRPCErrorObject(code=-32000, message="client rejected"),
        )
    )

    assert await request.wait() == "reject"
    assert server._pending_requests == {}


@pytest.mark.asyncio
async def test_shutdown_rejects_pending_approval_requests(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    server = _make_server(runtime, tmp_path)
    request = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="Shell",
        action="run command",
        description="Run command.",
    )
    server._pending_requests[request.id] = request

    await server._shutdown()

    assert await request.wait() == "reject"
    assert server._pending_requests == {}
