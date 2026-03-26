from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.ask_user import AskUserQuestion
from kimi_cli.tools.plan import ExitPlanMode
from kimi_cli.tools.plan.enter import EnterPlanMode
from kimi_cli.wire.jsonrpc import JSONRPCInitializeMessage, JSONRPCSuccessResponse
from kimi_cli.wire.server import WireServer


def _make_capability_toolset() -> KimiToolset:
    toolset = KimiToolset()
    toolset.add(AskUserQuestion())
    toolset.add(EnterPlanMode())
    toolset.add(ExitPlanMode())
    return toolset


def _visible_tool_names(toolset: KimiToolset) -> set[str]:
    return {tool.name for tool in toolset.tools}


def _make_server(runtime: Runtime, tmp_path: Path) -> tuple[WireServer, KimiToolset]:
    root_toolset = _make_capability_toolset()

    root_agent = Agent(
        name="Wire Capability Root",
        system_prompt="Test system prompt.",
        toolset=root_toolset,
        runtime=runtime,
    )

    soul = KimiSoul(root_agent, context=Context(file_backend=tmp_path / "history.jsonl"))
    return WireServer(soul), root_toolset


def _init_msg(
    *, supports_question: bool | None, supports_plan_mode: bool | None
) -> JSONRPCInitializeMessage:
    capabilities: dict[str, bool] | None = None
    if supports_question is not None or supports_plan_mode is not None:
        capabilities = {}
        if supports_question is not None:
            capabilities["supports_question"] = supports_question
        if supports_plan_mode is not None:
            capabilities["supports_plan_mode"] = supports_plan_mode

    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": "init",
        "params": {"protocol_version": "1.4"},
    }
    if capabilities is not None:
        payload["params"] = {"protocol_version": "1.4", "capabilities": capabilities}
    return JSONRPCInitializeMessage.model_validate(payload)


@pytest.mark.asyncio
async def test_initialize_without_capabilities_hides_question_and_plan_tools(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, root_toolset = _make_server(runtime, tmp_path)
    monkeypatch.setattr(server, "_ensure_notification_watcher", lambda: None)

    response = await server._handle_initialize(
        _init_msg(supports_question=None, supports_plan_mode=None)
    )

    assert isinstance(response, JSONRPCSuccessResponse)
    result = cast(dict[str, Any], response.result)
    assert result["capabilities"] == {"supports_question": True}
    assert server._client_supports_question is False
    assert server._client_supports_plan_mode is False
    assert _visible_tool_names(root_toolset) == set()


@pytest.mark.asyncio
async def test_initialize_with_question_support_only_unhides_ask_user_tool(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, root_toolset = _make_server(runtime, tmp_path)
    monkeypatch.setattr(server, "_ensure_notification_watcher", lambda: None)

    response = await server._handle_initialize(
        _init_msg(supports_question=True, supports_plan_mode=False)
    )

    assert isinstance(response, JSONRPCSuccessResponse)
    assert server._client_supports_question is True
    assert server._client_supports_plan_mode is False
    assert _visible_tool_names(root_toolset) == {"AskUserQuestion"}


@pytest.mark.asyncio
async def test_initialize_with_full_capabilities_unhides_all_capability_gated_tools(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, root_toolset = _make_server(runtime, tmp_path)
    monkeypatch.setattr(server, "_ensure_notification_watcher", lambda: None)

    response = await server._handle_initialize(
        _init_msg(supports_question=True, supports_plan_mode=True)
    )

    assert isinstance(response, JSONRPCSuccessResponse)
    assert server._client_supports_question is True
    assert server._client_supports_plan_mode is True
    expected = {"AskUserQuestion", "EnterPlanMode", "ExitPlanMode"}
    assert _visible_tool_names(root_toolset) == expected
