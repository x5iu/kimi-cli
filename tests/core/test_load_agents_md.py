from __future__ import annotations

from pathlib import Path

import pytest

from kaos.path import KaosPath
from kimi_cli.loop.agent import load_agents_md, load_project_agents_md


@pytest.fixture
def isolated_share_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    share_dir = tmp_path / "share"
    monkeypatch.setenv("KIMI_SHARE_DIR", str(share_dir))
    share_dir.mkdir()
    return share_dir


async def test_load_agents_md_found(temp_work_dir: KaosPath, isolated_share_dir: Path):
    """Test loading project AGENTS.md when it exists."""
    agents_md = temp_work_dir / "AGENTS.md"
    await agents_md.write_text("Test agents content")

    content = await load_agents_md(temp_work_dir)

    assert content == "Test agents content"


async def test_load_agents_md_not_found(temp_work_dir: KaosPath, isolated_share_dir: Path):
    """Test returning None when neither global nor project AGENTS.md exists."""
    content = await load_agents_md(temp_work_dir)

    assert content is None


async def test_load_agents_md_lowercase(temp_work_dir: KaosPath, isolated_share_dir: Path):
    """Test loading lowercase project agents.md."""
    agents_md = temp_work_dir / "agents.md"
    await agents_md.write_text("Lowercase agents content")

    content = await load_agents_md(temp_work_dir)

    assert content == "Lowercase agents content"


async def test_load_agents_md_global_only(temp_work_dir: KaosPath, isolated_share_dir: Path):
    """Test loading global AGENTS.md from the share directory."""
    (isolated_share_dir / "AGENTS.md").write_text("Global agents content", encoding="utf-8")

    content = await load_agents_md(temp_work_dir)

    assert content == "Global agents content"


async def test_load_agents_md_combines_global_and_project(
    temp_work_dir: KaosPath, isolated_share_dir: Path
):
    """Test combining global and project AGENTS.md content."""
    (isolated_share_dir / "AGENTS.md").write_text("Global agents content", encoding="utf-8")
    await (temp_work_dir / "AGENTS.md").write_text("Project agents content")

    content = await load_agents_md(temp_work_dir)

    assert content == (
        "[Global AGENTS.md]\nGlobal agents content\n\n[Project AGENTS.md]\nProject agents content"
    )


async def test_load_project_agents_md_ignores_global(
    temp_work_dir: KaosPath, isolated_share_dir: Path
):
    """Test project loader returns only the working-directory AGENTS.md content."""
    (isolated_share_dir / "AGENTS.md").write_text("Global agents content", encoding="utf-8")
    await (temp_work_dir / "AGENTS.md").write_text("Project agents content")

    content = await load_project_agents_md(temp_work_dir)

    assert content == "Project agents content"
