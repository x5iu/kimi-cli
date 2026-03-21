"""Plan file naming helpers."""

from __future__ import annotations

import re
import secrets
from datetime import datetime
from pathlib import Path

PLANS_DIR = Path.home() / ".kimi" / "plans"

_DEFAULT_WORK_DIR_BASENAME = "workspace"
_MAX_BASENAME_LEN = 48
_MAX_ATTEMPTS = 20
_INVALID_BASENAME_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_REPEAT_DASH_RE = re.compile(r"-{2,}")

_slug_cache: dict[str, str] = {}


def _sanitize_work_dir_basename(name: str) -> str:
    """Convert a directory basename into a filesystem-friendly slug prefix."""
    sanitized = _INVALID_BASENAME_CHARS_RE.sub("-", name.strip().lower())
    sanitized = _REPEAT_DASH_RE.sub("-", sanitized).strip("-.")
    sanitized = sanitized[:_MAX_BASENAME_LEN].rstrip("-.")
    return sanitized or _DEFAULT_WORK_DIR_BASENAME


def _current_timestamp() -> str:
    """Return a local timestamp suitable for plan filenames."""
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _random_suffix(nbytes: int = 3) -> str:
    """Return a short random lowercase hex suffix."""
    return secrets.token_hex(nbytes)


def _default_work_dir_basename() -> str:
    """Best-effort fallback basename when no work directory is provided."""
    return Path.cwd().name or _DEFAULT_WORK_DIR_BASENAME


def _build_slug(
    work_dir_basename: str,
    *,
    timestamp: str | None = None,
    random_suffix: str | None = None,
) -> str:
    """Build a plan filename slug from basename + timestamp + random chars."""
    return "-".join(
        [
            _sanitize_work_dir_basename(work_dir_basename),
            timestamp or _current_timestamp(),
            random_suffix or _random_suffix(),
        ]
    )


def get_or_create_slug(session_id: str, work_dir_basename: str | None = None) -> str:
    """Get or create a stable plan file slug for the given session."""
    if session_id in _slug_cache:
        return _slug_cache[session_id]

    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    basename = work_dir_basename or _default_work_dir_basename()
    timestamp = _current_timestamp()

    slug = ""
    for _ in range(_MAX_ATTEMPTS):
        slug = _build_slug(basename, timestamp=timestamp)
        if not (PLANS_DIR / f"{slug}.md").exists():
            break
    else:
        slug = _build_slug(basename, timestamp=timestamp, random_suffix=_random_suffix(6))

    _slug_cache[session_id] = slug
    return slug


def get_plan_file_path(session_id: str, work_dir_basename: str | None = None) -> Path:
    """Get the plan file path for the given session."""
    return PLANS_DIR / f"{get_or_create_slug(session_id, work_dir_basename)}.md"


def get_plan_file_path_by_slug(slug: str) -> Path:
    """Get the plan file path for a persisted plan file slug."""
    return PLANS_DIR / f"{slug}.md"


def read_plan_file(session_id: str, work_dir_basename: str | None = None) -> str | None:
    """Read the plan file content for the given session, or None if not found."""
    path = get_plan_file_path(session_id, work_dir_basename)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None
