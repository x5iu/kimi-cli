"""Tests for list_directory tree format and width caps."""

from __future__ import annotations

import os
import platform

import pytest

from kaos.path import KaosPath
from kimi_cli.utils.path import _LIST_DIR_CHILD_WIDTH, _LIST_DIR_ROOT_WIDTH, list_directory


@pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific.")
async def test_list_directory_tree_unix(temp_work_dir: KaosPath) -> None:
    await (temp_work_dir / "regular.txt").write_text("hello")
    await (temp_work_dir / "adir").mkdir()
    await (temp_work_dir / "adir" / "inside.txt").write_text("world")
    await (temp_work_dir / "emptydir").mkdir()
    os.symlink(
        (temp_work_dir / "regular.txt").unsafe_to_local_path(),
        (temp_work_dir / "link_to_regular").unsafe_to_local_path(),
    )
    os.symlink(
        (temp_work_dir / "missing.txt").unsafe_to_local_path(),
        (temp_work_dir / "broken_link").unsafe_to_local_path(),
    )

    out = await list_directory(temp_work_dir)
    lines = out.splitlines()
    assert any("adir/" in line for line in lines)
    assert any("emptydir/" in line for line in lines)
    assert any("inside.txt" in line for line in lines)
    assert any("regular.txt" in line for line in lines)


async def test_list_directory_truncates_root_width(temp_work_dir: KaosPath) -> None:
    overflow = 50
    file_count = _LIST_DIR_ROOT_WIDTH + overflow
    for i in range(file_count):
        (temp_work_dir / f"file_{i:04d}.txt").unsafe_to_local_path().touch()

    out = await list_directory(temp_work_dir)
    lines = out.splitlines()
    assert len(lines) == _LIST_DIR_ROOT_WIDTH + 1
    assert f"... and {overflow} more entries" in lines[-1]


async def test_list_directory_truncates_child_width(temp_work_dir: KaosPath) -> None:
    await (temp_work_dir / "subdir").mkdir()
    overflow = 5
    child_count = _LIST_DIR_CHILD_WIDTH + overflow
    for i in range(child_count):
        (temp_work_dir / "subdir" / f"child_{i:03d}.txt").unsafe_to_local_path().touch()

    out = await list_directory(temp_work_dir)
    lines = out.splitlines()
    assert "subdir/" in lines[0]
    assert f"... and {overflow} more" in lines[-1]


async def test_list_directory_dirs_before_files(temp_work_dir: KaosPath) -> None:
    await (temp_work_dir / "zebra.txt").write_text("z")
    await (temp_work_dir / "alpha").mkdir()
    await (temp_work_dir / "beta.txt").write_text("b")
    await (temp_work_dir / "omega").mkdir()

    out = await list_directory(temp_work_dir)
    lines = out.splitlines()
    dir_indices = [i for i, line in enumerate(lines) if "/" in line]
    file_indices = [i for i, line in enumerate(lines) if "/" not in line and "..." not in line]
    if dir_indices and file_indices:
        assert max(dir_indices) < min(file_indices)


async def test_list_directory_empty(temp_work_dir: KaosPath) -> None:
    out = await list_directory(temp_work_dir)
    assert out == "(empty directory)"
