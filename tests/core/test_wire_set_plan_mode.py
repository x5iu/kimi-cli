from __future__ import annotations

from pathlib import Path

import pytest
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.wire.file import WireMessageRecord
from kimi_cli.wire.jsonrpc import (
    JSONRPCEventMessage,
    JSONRPCSetPlanModeMessage,
    JSONRPCSuccessResponse,
)
from kimi_cli.wire.server import WireServer
from kimi_cli.wire.types import StatusUpdate


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
    return WireServer(soul), soul


@pytest.mark.asyncio
async def test_handle_set_plan_mode_updates_state_and_emits_status(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, soul = _make_server(runtime, tmp_path)
    sent: list[JSONRPCEventMessage] = []

    async def fake_send_msg(msg):
        assert isinstance(msg, JSONRPCEventMessage)
        sent.append(msg)

    monkeypatch.setattr(server, "_send_msg", fake_send_msg)
    if soul.wire_file.path.exists():
        soul.wire_file.path.unlink()

    enable_msg = JSONRPCSetPlanModeMessage.model_validate(
        {
            "jsonrpc": "2.0",
            "method": "set_plan_mode",
            "id": "pm-on",
            "params": {"enabled": True},
        }
    )
    enable_resp = await server._handle_set_plan_mode(enable_msg)

    assert isinstance(enable_resp, JSONRPCSuccessResponse)
    assert enable_resp.result == {"status": "ok", "plan_mode": True}
    assert soul.plan_mode is True
    assert runtime.session.state.plan_mode is True
    assert sent == [JSONRPCEventMessage(params=StatusUpdate(plan_mode=True))]
    records = [record async for record in soul.wire_file.iter_records()]
    assert records[-1] == WireMessageRecord.from_wire_message(
        StatusUpdate(plan_mode=True), timestamp=records[-1].timestamp
    )

    sent.clear()

    disable_msg = JSONRPCSetPlanModeMessage.model_validate(
        {
            "jsonrpc": "2.0",
            "method": "set_plan_mode",
            "id": "pm-off",
            "params": {"enabled": False},
        }
    )
    disable_resp = await server._handle_set_plan_mode(disable_msg)

    assert isinstance(disable_resp, JSONRPCSuccessResponse)
    assert disable_resp.result == {"status": "ok", "plan_mode": False}
    assert soul.plan_mode is False
    assert runtime.session.state.plan_mode is False
    assert sent == [JSONRPCEventMessage(params=StatusUpdate(plan_mode=False))]
    records = [record async for record in soul.wire_file.iter_records()]
    assert records[-1] == WireMessageRecord.from_wire_message(
        StatusUpdate(plan_mode=False), timestamp=records[-1].timestamp
    )
