"""Tests for skill discovery and formatting behavior."""

import sys
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from kaos.path import KaosPath
from kimi_cli.skill import (
    Skill,
    discover_skills,
    discover_skills_from_roots,
    get_builtin_skills_dir,
    resolve_skills_roots,
)


def _write_skill(skill_dir: Path, content: str) -> None:
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


@pytest.mark.asyncio
async def test_discover_skills_parses_frontmatter_and_defaults(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()

    _write_skill(
        root / "alpha",
        """---
name: alpha-skill
description: Alpha description
---
""",
    )
    _write_skill(root / "beta", "# No frontmatter")

    root_path = KaosPath.unsafe_from_local_path(root)
    skills = await discover_skills(root_path)
    base_dir = KaosPath.unsafe_from_local_path(Path("/path/to"))
    for skill in skills:
        relative_dir = skill.dir.relative_to(root_path)
        skill.dir = base_dir / relative_dir

    assert skills == snapshot(
        [
            Skill(
                name="alpha-skill",
                description="Alpha description",
                type="standard",
                dir=KaosPath.unsafe_from_local_path(Path("/path/to/alpha")),
            ),
            Skill(
                name="beta",
                description="No description provided.",
                type="standard",
                dir=KaosPath.unsafe_from_local_path(Path("/path/to/beta")),
            ),
        ]
    )


@pytest.mark.asyncio
async def test_discover_skills_from_roots_prefers_later_dirs(tmp_path):
    root = tmp_path / "root"
    system_dir = root / "system"
    user_dir = root / "user"
    system_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)

    _write_skill(
        system_dir / "shared",
        """---
name: shared
description: System version
---
""",
    )
    _write_skill(
        user_dir / "shared",
        """---
name: shared
description: User version
---
""",
    )

    root_path = KaosPath.unsafe_from_local_path(root)
    skills = await discover_skills_from_roots(
        [
            KaosPath.unsafe_from_local_path(system_dir),
            KaosPath.unsafe_from_local_path(user_dir),
        ]
    )
    base_dir = KaosPath.unsafe_from_local_path(Path("/path/to"))
    for skill in skills:
        relative_dir = skill.dir.relative_to(root_path)
        skill.dir = base_dir / relative_dir

    assert skills == snapshot(
        [
            Skill(
                name="shared",
                description="User version",
                type="standard",
                dir=KaosPath.unsafe_from_local_path(Path("/path/to/user/shared")),
            )
        ]
    )


@pytest.mark.asyncio
async def test_resolve_skills_roots_uses_layers(monkeypatch, tmp_path):
    home_dir = tmp_path / "home"
    user_dir = home_dir / ".config" / "agents" / "skills"
    user_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    work_dir = tmp_path / "project"
    project_dir = work_dir / ".agents" / "skills"
    project_dir.mkdir(parents=True)

    roots = await resolve_skills_roots(KaosPath.unsafe_from_local_path(work_dir))

    assert roots == [
        KaosPath.unsafe_from_local_path(get_builtin_skills_dir()),
        KaosPath.unsafe_from_local_path(user_dir),
        KaosPath.unsafe_from_local_path(project_dir),
    ]


@pytest.mark.asyncio
async def test_resolve_skills_roots_respects_override(tmp_path):
    work_dir = tmp_path / "project"
    override_dir = tmp_path / "override"
    override_dir.mkdir()

    roots = await resolve_skills_roots(
        KaosPath.unsafe_from_local_path(work_dir),
        skills_dir_override=KaosPath.unsafe_from_local_path(override_dir),
    )

    assert roots == [
        KaosPath.unsafe_from_local_path(get_builtin_skills_dir()),
        KaosPath.unsafe_from_local_path(override_dir),
    ]


def test_get_builtin_skills_dir_frozen_env(monkeypatch, tmp_path):
    """In a PyInstaller frozen env, get_builtin_skills_dir uses sys._MEIPASS."""
    fake_meipass = tmp_path / "_meipass"
    fake_meipass.mkdir()

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_meipass), raising=False)

    result = get_builtin_skills_dir()
    assert result == fake_meipass / "kimi_cli" / "skills"


def test_get_builtin_skills_dir_normal_env():
    """In a normal (non-frozen) env, get_builtin_skills_dir uses __file__."""
    result = get_builtin_skills_dir()
    # Should resolve relative to the skill package
    assert result.name == "skills"
    assert result.parent.name == "kimi_cli"
