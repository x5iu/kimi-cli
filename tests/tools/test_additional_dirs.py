"""Tests for additional directories support in file tools."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from kaos.path import KaosPath
from kimi_cli.loop.agent import Runtime
from kimi_cli.loop.approval import Approval
from kimi_cli.tools.file.read import Params as ReadParams
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.write import Params as WriteParams
from kimi_cli.tools.file.write import WriteFile
from tests.conftest import tool_call_context


@pytest.fixture
def additional_dir(temp_work_dir: KaosPath) -> Generator[KaosPath]:
    """Create a temporary additional directory outside the work directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir).resolve()
        yield KaosPath.unsafe_from_local_path(p)


@pytest.fixture
def runtime_with_additional_dir(runtime: Runtime, additional_dir: KaosPath) -> Runtime:
    """Runtime with an additional directory configured."""
    runtime.additional_dirs.append(additional_dir)
    return runtime


# ── ReadFile tests ──────────────────────────────────────────────────────────


async def test_read_file_in_additional_dir(
    runtime_with_additional_dir: Runtime, additional_dir: KaosPath
):
    """ReadFile should read files in additional directories."""
    read_tool = ReadFile(runtime_with_additional_dir)
    test_file = additional_dir / "readme.txt"
    await test_file.write_text("Hello from additional dir\n")

    result = await read_tool(ReadParams(path=str(test_file)))
    assert not result.is_error
    assert "Hello from additional dir" in result.output


async def test_read_file_relative_path_in_additional_dir(
    runtime_with_additional_dir: Runtime, additional_dir: KaosPath
):
    """Relative paths that resolve outside work_dir but inside additional dir should work."""
    read_tool = ReadFile(runtime_with_additional_dir)
    test_file = additional_dir / "data.txt"
    await test_file.write_text("data content\n")

    # Absolute path to the file in additional dir should be allowed
    result = await read_tool(ReadParams(path=str(test_file)))
    assert not result.is_error


# ── WriteFile tests ─────────────────────────────────────────────────────────


async def test_write_file_in_additional_dir(
    runtime_with_additional_dir: Runtime, approval: Approval, additional_dir: KaosPath
):
    """WriteFile should write to files in additional directories."""
    with tool_call_context("WriteFile"):
        write_tool = WriteFile(runtime_with_additional_dir, approval)
        target = additional_dir / "output.txt"

        result = await write_tool(WriteParams(path=str(target), content="new content"))
        assert not result.is_error
        assert await target.read_text() == "new content"


async def test_write_file_in_additional_dir_uses_edit_action(
    runtime_with_additional_dir: Runtime, approval: Approval, additional_dir: KaosPath
):
    """Writing in additional dir should use EDIT action (not EDIT_OUTSIDE)."""
    with tool_call_context("WriteFile"):
        write_tool = WriteFile(runtime_with_additional_dir, approval)
        target = additional_dir / "in_workspace.txt"

        result = await write_tool(WriteParams(path=str(target), content="content"))
        assert not result.is_error


# ── Dynamic mutation tests ──────────────────────────────────────────────────
