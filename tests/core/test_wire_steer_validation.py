from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.wire.jsonrpc import (
    ErrorCodes,
    JSONRPCErrorResponse,
    JSONRPCSteerMessage,
    JSONRPCSuccessResponse,
    Statuses,
)
from kimi_cli.wire.server import WireServer
from kimi_cli.wire.types import ImageURLPart


def _make_server(runtime: Runtime, tmp_path: Path) -> tuple[WireServer, KimiSoul]:
    soul = KimiSoul(
        Agent(
            name="Wire Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )
    server = WireServer(soul)
    server._cancel_event = asyncio.Event()
    return server, soul


@pytest.mark.asyncio
async def test_handle_steer_returns_llm_not_set_when_active_turn_has_no_llm(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    runtime.llm = None
    server, _ = _make_server(runtime, tmp_path)

    response = await server._handle_steer(
        JSONRPCSteerMessage(
            id="steer-1",
            params=JSONRPCSteerMessage.Params(user_input="later"),
        )
    )

    assert isinstance(response, JSONRPCErrorResponse)
    assert response.error.code == ErrorCodes.LLM_NOT_SET
    assert response.error.message == "LLM is not set"


@pytest.mark.asyncio
async def test_handle_steer_returns_llm_not_supported_for_unsupported_content_parts(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    assert runtime.llm is not None
    runtime.llm.capabilities = set()
    server, _ = _make_server(runtime, tmp_path)

    response = await server._handle_steer(
        JSONRPCSteerMessage(
            id="steer-1",
            params=JSONRPCSteerMessage.Params(
                user_input=[
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")
                    )
                ]
            ),
        )
    )

    assert isinstance(response, JSONRPCErrorResponse)
    assert response.error.code == ErrorCodes.LLM_NOT_SUPPORTED
    assert "image_in" in response.error.message


@pytest.mark.asyncio
async def test_handle_steer_accepts_valid_text_input(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, soul = _make_server(runtime, tmp_path)
    recorded: list[object] = []
    monkeypatch.setattr(soul, "steer", lambda content: recorded.append(content))

    response = await server._handle_steer(
        JSONRPCSteerMessage(
            id="steer-1",
            params=JSONRPCSteerMessage.Params(user_input="later"),
        )
    )

    assert isinstance(response, JSONRPCSuccessResponse)
    assert response.result == {"status": Statuses.STEERED}
    assert recorded == ["later"]
