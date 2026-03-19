from __future__ import annotations

from pathlib import Path

import pytest
from kosong.tooling.empty import EmptyToolset

import kimi_cli.wire.server as wire_server_module
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.wire.jsonrpc import ErrorCodes, JSONRPCErrorResponseNullableID
from kimi_cli.wire.server import WireServer


class _FakeReader:
    def __init__(self) -> None:
        self._calls = 0

    async def readline(self) -> bytes:
        self._calls += 1
        if self._calls == 1:
            raise ValueError("Input line exceeds maximum size")
        return b""

    def close(self) -> None:
        return


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
async def test_read_loop_reports_oversized_line_once(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_server(runtime, tmp_path)
    server._reader = _FakeReader()
    sent: list[JSONRPCErrorResponseNullableID] = []

    async def fake_send_msg(msg):
        assert isinstance(msg, JSONRPCErrorResponseNullableID)
        sent.append(msg)

    monkeypatch.setattr(server, "_send_msg", fake_send_msg)

    await server._read_loop()

    assert len(sent) == 1
    assert sent[0].error.code == ErrorCodes.PARSE_ERROR
    assert sent[0].error.message == "Input line exceeds maximum size"


def test_stdin_readline_discards_oversized_remainder(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeBuffer:
        def __init__(self) -> None:
            self._chunks = [b"123456", b"789\n", b'{"jsonrpc":"2.0"}\n']

        def readline(self, size: int) -> bytes:
            assert size == 6
            return self._chunks.pop(0)

    class _FakeStdin:
        def __init__(self) -> None:
            self.buffer = _FakeBuffer()

    monkeypatch.setattr(wire_server_module.sys, "stdin", _FakeStdin())

    with pytest.raises(ValueError, match="Input line exceeds maximum size"):
        wire_server_module._stdin_readline(5)

    assert wire_server_module._stdin_readline(5) == b'{"jsonrpc":"2.0"}\n'
