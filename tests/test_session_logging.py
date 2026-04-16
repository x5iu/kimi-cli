from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import SecretStr

from kimi_cli.cli.export import _collect_recent_log_files, _session_time_range
from kimi_cli.eventbus.log import BusMessageRecord, EventLogMetadata
from kimi_cli.eventbus.types import TurnBegin
from llmkit.message import TextPart

_TWO_DAYS = 2 * 24 * 60 * 60


def _write_wire_records(wire_path: Path, timestamps: list[float]) -> None:
    with wire_path.open("w") as f:
        meta = EventLogMetadata(protocol_version="1.4")
        f.write(json.dumps(meta.model_dump(mode="json"), ensure_ascii=False) + "\n")
        for ts in timestamps:
            rec = BusMessageRecord.from_bus_message(
                TurnBegin(user_input=[TextPart(text="hi")]),
                timestamp=ts,
            )
            f.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")


class TestSessionTimeRange:
    def test_returns_first_and_last(self, tmp_path: Path):
        wire = tmp_path / "wire.jsonl"
        _write_wire_records(wire, [1000.0, 2000.0, 3000.0])
        first, last = _session_time_range(tmp_path)
        assert first == 1000.0
        assert last == 3000.0

    def test_single_record(self, tmp_path: Path):
        wire = tmp_path / "wire.jsonl"
        _write_wire_records(wire, [5000.0])
        first, last = _session_time_range(tmp_path)
        assert first == 5000.0
        assert last == 5000.0

    def test_no_wire_file(self, tmp_path: Path):
        first, last = _session_time_range(tmp_path)
        assert first is None
        assert last is None

    def test_empty_wire_file(self, tmp_path: Path):
        (tmp_path / "wire.jsonl").write_text("")
        first, last = _session_time_range(tmp_path)
        assert first is None
        assert last is None


class TestCollectRecentLogFiles:
    def _setup_log_dir(self, share_dir: Path) -> Path:
        log_dir = share_dir / "logs"
        log_dir.mkdir(parents=True)
        return log_dir

    def _make_log(self, log_dir: Path, name: str, mtime: float) -> Path:
        f = log_dir / name
        f.write_text(f"log content for {name}")
        os.utime(f, (mtime, mtime))
        return f

    def test_collects_logs_near_export_time(self, tmp_path: Path):
        log_dir = self._setup_log_dir(tmp_path)
        now = time.time()
        self._make_log(log_dir, "kimi.log", now - 3600)
        self._make_log(log_dir, "kimi.old.log", now - 10 * 86400)

        session_dir = tmp_path / "session"
        session_dir.mkdir()

        with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
            files = _collect_recent_log_files(session_dir)

        names = {f.name for f in files}
        assert "kimi.log" in names
        assert "kimi.old.log" not in names

    def test_collects_logs_near_session_time(self, tmp_path: Path):
        log_dir = self._setup_log_dir(tmp_path)
        now = time.time()
        session_time = now - 7 * 86400

        self._make_log(log_dir, "kimi.session-era.log", session_time + 3600)
        self._make_log(log_dir, "kimi.ancient.log", session_time - 30 * 86400)
        self._make_log(log_dir, "kimi.log", now - 60)

        session_dir = tmp_path / "session"
        session_dir.mkdir()
        _write_wire_records(session_dir / "wire.jsonl", [session_time, session_time + 7200])

        with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
            files = _collect_recent_log_files(session_dir)

        names = {f.name for f in files}
        assert "kimi.session-era.log" in names
        assert "kimi.log" in names
        assert "kimi.ancient.log" not in names

    def test_no_wire_file_falls_back_to_export_time(self, tmp_path: Path):
        log_dir = self._setup_log_dir(tmp_path)
        now = time.time()
        self._make_log(log_dir, "kimi.log", now - 3600)
        self._make_log(log_dir, "kimi.old.log", now - 5 * 86400)

        session_dir = tmp_path / "session"
        session_dir.mkdir()

        with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
            files = _collect_recent_log_files(session_dir)

        names = {f.name for f in files}
        assert "kimi.log" in names
        assert "kimi.old.log" not in names

    def test_ignores_non_log_files(self, tmp_path: Path):
        log_dir = self._setup_log_dir(tmp_path)
        (log_dir / "kimi.log").write_text("log")
        (log_dir / ".DS_Store").write_bytes(b"\x00")
        (log_dir / "notes.txt").write_text("not a log")

        session_dir = tmp_path / "session"
        session_dir.mkdir()

        with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
            files = _collect_recent_log_files(session_dir)

        names = {f.name for f in files}
        assert names == {"kimi.log"}

    def test_empty_log_dir(self, tmp_path: Path):
        self._setup_log_dir(tmp_path)
        session_dir = tmp_path / "session"
        session_dir.mkdir()

        with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
            assert _collect_recent_log_files(session_dir) == []

    def test_no_log_dir(self, tmp_path: Path):
        session_dir = tmp_path / "session"
        session_dir.mkdir()

        with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
            assert _collect_recent_log_files(session_dir) == []


class TestToolExecutionLogging:
    async def test_toolset_tool_execution_error_logged(self):
        from kimi_cli.eventbus.types import ToolCall
        from kimi_cli.loop.toolset import KimiToolset

        toolset = KimiToolset()

        class FailingTool:
            name = "FailingTool"
            base = None

            async def call(self, arguments):
                raise RuntimeError("Tool exploded")

        toolset._tool_dict["FailingTool"] = FailingTool()  # type: ignore[assignment]

        tool_call = ToolCall(
            id="tc_1",
            function=ToolCall.FunctionBody(name="FailingTool", arguments="{}"),
        )

        with patch("kimi_cli.loop.toolset.logger") as mock_logger:
            result = toolset.handle(tool_call)
            if isinstance(result, asyncio.Task):
                await result
            mock_logger.exception.assert_called()
            assert "FailingTool" in str(mock_logger.exception.call_args)

    async def test_toolset_json_parse_error_logged(self):
        from kimi_cli.eventbus.types import ToolCall
        from kimi_cli.loop.toolset import KimiToolset

        toolset = KimiToolset()

        class DummyTool:
            name = "DummyTool"
            base = None

        toolset._tool_dict["DummyTool"] = DummyTool()  # type: ignore[assignment]

        tool_call = ToolCall(
            id="tc_2",
            function=ToolCall.FunctionBody(name="DummyTool", arguments="{invalid json}"),
        )

        with patch("kimi_cli.loop.toolset.logger") as mock_logger:
            toolset.handle(tool_call)
            mock_logger.warning.assert_called()
            assert "DummyTool" in str(mock_logger.warning.call_args)


class TestFileToolLogging:
    async def test_read_file_exception_logged(self, read_file_tool):
        from kimi_cli.tools.file.read import Params

        with (
            patch("kimi_cli.tools.file.read.logger") as mock_logger,
            patch("kimi_cli.tools.file.read.KaosPath") as mock_path,
        ):
            mock_path.return_value.expanduser.side_effect = RuntimeError("Unexpected")
            result = await read_file_tool(Params(path="/some/file"))
            assert result.is_error
            mock_logger.warning.assert_called_once()

    async def test_write_file_exception_logged(self, write_file_tool):
        from kimi_cli.tools.file.write import Params

        with (
            patch("kimi_cli.tools.file.write.logger") as mock_logger,
            patch("kimi_cli.tools.file.write.KaosPath") as mock_path,
        ):
            mock_path.return_value.expanduser.side_effect = RuntimeError("Unexpected")
            result = await write_file_tool(Params(path="/some/file", content="test"))
            assert result.is_error
            mock_logger.warning.assert_called_once()

    async def test_replace_file_exception_logged(self, edit_tool):
        from kimi_cli.tools.file.replace import EditParams, ReplaceOp

        with (
            patch("kimi_cli.tools.file.replace.logger") as mock_logger,
            patch("kimi_cli.tools.file.replace.KaosPath") as mock_path,
        ):
            mock_path.return_value.expanduser.side_effect = RuntimeError("Unexpected")
            result = await edit_tool(
                EditParams(path="/some/file", edit=[ReplaceOp(old="a", new="b")])
            )
            assert result.is_error
            mock_logger.warning.assert_called_once()


class TestSearchWebLogging:
    async def test_search_timeout_logged(self, search_web_tool):
        from kimi_cli.tools.web.search import Params
        from tests.conftest import tool_call_context

        with (
            tool_call_context("SearchWeb"),
            patch("kimi_cli.tools.web.search.logger") as mock_logger,
            patch("kimi_cli.tools.web.search.new_client_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(side_effect=TimeoutError())
            result = await search_web_tool(Params(query="test query"))
            assert result.is_error
            mock_logger.warning.assert_called()


class TestLLMLogging:
    def test_create_llm_missing_config_logged(self):
        from kimi_cli.config import LLMModel, LLMProvider

        with patch("kimi_cli.llm.logger") as mock_logger:
            from kimi_cli.llm import create_llm

            result = create_llm(
                LLMProvider(type="kimi", base_url="", api_key=SecretStr("")),
                LLMModel(provider="kimi", model="", max_context_size=100_000),
            )
            assert result is None
            mock_logger.warning.assert_called_once()
