from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.utils.turns import is_real_user_turn_start_message
from llmkit.message import Message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop

_EDIT_VERIFY_TYPE = "edit_verification_reminder"

# Extensions considered "source code"
_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
    }
)

# Extensions to always exclude even if they happen to
# match (e.g. .json is not source code)
_EXCLUDED_EXTENSIONS = frozenset(
    {
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
    }
)

# Patterns in Shell commands that count as verification
_VERIFY_PATTERNS = re.compile(
    r"(?:pytest|ruff|make\s+test|make\s+check"
    r"|make\s+format|uv\s+run\s+pytest)"
)


class EditVerificationReminderProvider(AttachmentProvider):
    """Remind agent to verify after multi-file source edits.

    When the agent has edited >=2 source files in the current
    turn without running tests or linting, injects a
    ``<system-hint>`` suggesting verification.

    Stateless: re-derives from history each call, so
    compaction is not a concern.  Fires at most once per turn.
    """

    def __init__(self) -> None:
        self._fired_this_turn: bool = False
        self._last_turn_start_index: int = -1

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        if not history:
            return []

        # Find the current turn boundary
        turn_start: int = 0
        for i in range(len(history) - 1, -1, -1):
            if is_real_user_turn_start_message(history[i]):
                turn_start = i
                break

        # Reset on turn change
        if turn_start != self._last_turn_start_index:
            self._last_turn_start_index = turn_start
            self._fired_this_turn = False

        if self._fired_this_turn:
            return []

        # Scan the current turn for edits and verification
        edited_source_files: set[str] = set()
        has_verification = False

        for msg in history[turn_start:]:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                name = tc.function.name
                if name in ("Edit", "WriteFile"):
                    path = _extract_file_path(tc.function.arguments)
                    if path and _is_source_file(path):
                        edited_source_files.add(path)
                elif name == "Shell":
                    cmd = _extract_command(tc.function.arguments)
                    if cmd and _VERIFY_PATTERNS.search(cmd):
                        has_verification = True

        if len(edited_source_files) < 2:
            return []

        if has_verification:
            return []

        n = len(edited_source_files)
        self._fired_this_turn = True
        return [
            Attachment(
                type=_EDIT_VERIFY_TYPE,
                content=(
                    f"You've edited {n} source files in "
                    "this turn without running tests or "
                    "linting. Consider running the project's "
                    "test suite or linter to verify your "
                    "changes before concluding."
                ),
                is_hint=True,
            )
        ]


def _extract_file_path(
    arguments: str | None,
) -> str | None:
    """Extract file_path from Edit/WriteFile arguments."""
    if not arguments:
        return None
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(args, dict):
        d = cast(dict[str, Any], args)
        fp: str | None = d.get("path") or d.get("file_path")
        if isinstance(fp, str):
            return fp
    return None


def _extract_command(
    arguments: str | None,
) -> str | None:
    """Extract command from Shell arguments."""
    if not arguments:
        return None
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(args, dict):
        d = cast(dict[str, Any], args)
        cmd: str | None = d.get("command")
        if isinstance(cmd, str):
            return cmd
    return None


def _is_source_file(path: str) -> bool:
    """Check if a file path is a source code file."""
    # Exclude /tmp/* paths
    if path.startswith("/tmp/") or path.startswith("/tmp\\"):
        return False
    # Check extension
    dot_idx = path.rfind(".")
    if dot_idx < 0:
        return False
    ext = path[dot_idx:].lower()
    if ext in _EXCLUDED_EXTENSIONS:
        return False
    return ext in _SOURCE_EXTENSIONS
