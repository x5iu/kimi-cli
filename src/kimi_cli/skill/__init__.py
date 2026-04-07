"""Skill specification discovery and loading utilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from kaos.local import local_kaos
from kaos.path import KaosPath
from loguru import logger
from pydantic import BaseModel, ConfigDict

from kaos import get_current_kaos
from kimi_cli.utils.frontmatter import parse_frontmatter

SkillType = Literal["standard"]


def get_builtin_skills_dir() -> Path:
    """
    Get the built-in skills directory path.
    """
    return Path(__file__).parent.parent / "skills"


def _get_user_generic_skills_dir_candidates() -> tuple[KaosPath, ...]:
    """
    Get user-level generic skills directory candidates in priority order.

    Generic group: ``~/.config/agents/skills`` > ``~/.agents/skills``
    """
    return (
        KaosPath.home() / ".config" / "agents" / "skills",
        KaosPath.home() / ".agents" / "skills",
    )


def _get_user_brand_skills_dir_candidates() -> tuple[KaosPath, ...]:
    """
    Get user-level brand skills directory candidates in priority order.

    Brand group: ``~/.kimi/skills`` > ``~/.claude/skills`` > ``~/.codex/skills``
    """
    return (
        KaosPath.home() / ".kimi" / "skills",
        KaosPath.home() / ".claude" / "skills",
        KaosPath.home() / ".codex" / "skills",
    )


def _get_project_generic_skills_dir_candidates(work_dir: KaosPath) -> tuple[KaosPath, ...]:
    """
    Get project-level generic skills directory candidates.

    Generic group: ``.agents/skills``
    """
    return (work_dir / ".agents" / "skills",)


def _get_project_brand_skills_dir_candidates(work_dir: KaosPath) -> tuple[KaosPath, ...]:
    """
    Get project-level brand skills directory candidates in priority order.

    Brand group: ``.kimi/skills`` > ``.claude/skills`` > ``.codex/skills``
    """
    return (
        work_dir / ".kimi" / "skills",
        work_dir / ".claude" / "skills",
        work_dir / ".codex" / "skills",
    )


def _supports_builtin_skills() -> bool:
    """Return True when the active KAOS backend can read bundled skills."""
    current_name = get_current_kaos().name
    return current_name == local_kaos.name


async def find_first_existing_dir(candidates: Iterable[KaosPath]) -> KaosPath | None:
    """
    Return the first existing directory from candidates.
    """
    for candidate in candidates:
        if await candidate.is_dir():
            return candidate
    return None


async def find_user_skills_dirs(
    *,
    merge_brands: bool = False,
) -> list[KaosPath]:
    """
    Return user-level skills directories from both brand and generic groups.

    The brand group comes first because brand-specific directories have
    higher specificity.  When *merge_brands* is ``False`` (default), only the
    first existing brand directory is used.  When ``True``, all existing brand
    directories are included (priority order: kimi > claude > codex).
    """
    dirs: list[KaosPath] = []
    if merge_brands:
        for candidate in _get_user_brand_skills_dir_candidates():
            if await candidate.is_dir():
                dirs.append(candidate)
    else:
        if brand := await find_first_existing_dir(
            _get_user_brand_skills_dir_candidates(),
        ):
            dirs.append(brand)
    if generic := await find_first_existing_dir(
        _get_user_generic_skills_dir_candidates(),
    ):
        dirs.append(generic)
    return dirs


async def find_project_skills_dirs(
    work_dir: KaosPath,
    *,
    merge_brands: bool = False,
) -> list[KaosPath]:
    """
    Return project-level skills directories from both brand and generic groups.

    The brand group comes first because brand-specific directories have
    higher specificity.  When *merge_brands* is ``False`` (default), only the
    first existing brand directory is used.  When ``True``, all existing brand
    directories are included (priority order: kimi > claude > codex).
    """
    dirs: list[KaosPath] = []
    brand_candidates = _get_project_brand_skills_dir_candidates(work_dir)
    if merge_brands:
        for candidate in brand_candidates:
            if await candidate.is_dir():
                dirs.append(candidate)
    else:
        if brand := await find_first_existing_dir(brand_candidates):
            dirs.append(brand)
    generic_candidates = _get_project_generic_skills_dir_candidates(work_dir)
    if generic := await find_first_existing_dir(generic_candidates):
        dirs.append(generic)
    return dirs


async def resolve_skills_roots(
    work_dir: KaosPath,
    *,
    skills_dir_override: KaosPath | None = None,
    merge_brands: bool = False,
) -> list[KaosPath]:
    """
    Resolve layered skill roots in priority order.

    Built-in skills load first when supported by the active KAOS backend.
    When an override is provided, user/project discovery is skipped.
    Otherwise, user-level and project-level directories are discovered from
    two independent groups (brand and generic) whose results are merged —
    brand dirs come first so their skills take priority.
    When *merge_brands* is ``True``, all existing brand directories are loaded
    instead of only the first one found.
    """
    roots: list[KaosPath] = []
    if _supports_builtin_skills():
        roots.append(KaosPath.unsafe_from_local_path(get_builtin_skills_dir()))
    if skills_dir_override is not None:
        roots.append(skills_dir_override)
        return roots
    roots.extend(await find_user_skills_dirs(merge_brands=merge_brands))
    roots.extend(
        await find_project_skills_dirs(work_dir, merge_brands=merge_brands),
    )
    return roots


def normalize_skill_name(name: str) -> str:
    """Normalize a skill name for lookup."""
    return name.casefold()


def index_skills(skills: Iterable[Skill]) -> dict[str, Skill]:
    """Build a lookup table for skills by normalized name."""
    return {normalize_skill_name(skill.name): skill for skill in skills}


async def discover_skills_from_roots(skills_dirs: Iterable[KaosPath]) -> list[Skill]:
    """
    Discover skills from multiple directory roots.
    """
    skills_by_name: dict[str, Skill] = {}
    for skills_dir in skills_dirs:
        for skill in await discover_skills(skills_dir):
            skills_by_name[normalize_skill_name(skill.name)] = skill
    return sorted(skills_by_name.values(), key=lambda s: s.name)


async def read_skill_text(skill: Skill) -> str | None:
    """Read the SKILL.md contents for a skill."""
    try:
        return (await skill.skill_md_file.read_text(encoding="utf-8")).strip()
    except OSError as exc:
        logger.warning(
            "Failed to read skill file {path}: {error}",
            path=skill.skill_md_file,
            error=exc,
        )
        return None


class Skill(BaseModel):
    """Information about a single skill."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    name: str
    description: str
    type: SkillType = "standard"
    dir: KaosPath

    @property
    def skill_md_file(self) -> KaosPath:
        """Path to the SKILL.md file."""
        return self.dir / "SKILL.md"


async def discover_skills(skills_dir: KaosPath) -> list[Skill]:
    """
    Discover all skills in the given directory.

    Args:
        skills_dir: Kaos path to the directory containing skills.

    Returns:
        List of Skill objects, one for each valid skill found.
    """
    if not await skills_dir.is_dir():
        return []

    skills: list[Skill] = []

    async for skill_dir in skills_dir.iterdir():
        if not await skill_dir.is_dir():
            continue

        skill_md = skill_dir / "SKILL.md"
        if not await skill_md.is_file():
            continue

        try:
            content = await skill_md.read_text(encoding="utf-8")
            skills.append(parse_skill_text(content, dir_path=skill_dir))
        except Exception as exc:
            logger.info("Skipping invalid skill at {}: {}", skill_md, exc)
            continue

    return sorted(skills, key=lambda s: s.name)


def parse_skill_text(content: str, *, dir_path: KaosPath) -> Skill:
    """
    Parse SKILL.md contents to extract name and description.
    """
    frontmatter = parse_frontmatter(content) or {}

    name = frontmatter.get("name") or dir_path.name
    description = frontmatter.get("description") or "No description provided."
    skill_type = frontmatter.get("type") or "standard"
    if skill_type != "standard":
        raise ValueError(f'Invalid skill type "{skill_type}"')

    return Skill(
        name=name,
        description=description,
        type=skill_type,
        dir=dir_path,
    )


def _iter_fenced_codeblocks(content: str) -> Iterator[tuple[str, str]]:
    fence = ""
    fence_char = ""
    lang = ""
    buf: list[str] = []
    in_block = False

    for line in content.splitlines():
        stripped = line.lstrip()
        if not in_block:
            if match := _parse_fence_open(stripped):
                fence, fence_char, info = match
                lang = _normalize_code_lang(info)
                in_block = True
                buf = []
            continue

        if _is_fence_close(stripped, fence_char, len(fence)):
            yield lang, "\n".join(buf).strip("\n")
            in_block = False
            fence = ""
            fence_char = ""
            lang = ""
            buf = []
            continue

        buf.append(line)


def _normalize_code_lang(info: str) -> str:
    if not info:
        return ""
    lang = info.split()[0].strip().lower()
    if lang.startswith("{") and lang.endswith("}"):
        lang = lang[1:-1].strip()
    return lang


def _parse_fence_open(line: str) -> tuple[str, str, str] | None:
    if not line or line[0] not in ("`", "~"):
        return None
    fence_char = line[0]
    count = 0
    for ch in line:
        if ch == fence_char:
            count += 1
        else:
            break
    if count < 3:
        return None
    fence = fence_char * count
    info = line[count:].strip()
    return fence, fence_char, info


def _is_fence_close(line: str, fence_char: str, fence_len: int) -> bool:
    if not fence_char or not line or line[0] != fence_char:
        return False
    count = 0
    for ch in line:
        if ch == fence_char:
            count += 1
        else:
            break
    if count < fence_len:
        return False
    return not line[count:].strip()
