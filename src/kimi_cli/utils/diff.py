from __future__ import annotations

import asyncio
import difflib
import re
from difflib import SequenceMatcher

from kimi_cli.tools.display import DiffDisplayBlock
from llmkit.tooling import DisplayBlock

N_CONTEXT_LINES = 3

_HUGE_FILE_THRESHOLD = 10000
"""Line count above which diff computation is skipped entirely."""

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def format_unified_diff(
    old_text: str,
    new_text: str,
    path: str = "",
    *,
    include_file_header: bool = True,
    old_start_line: int = 1,
    new_start_line: int = 1,
) -> str:
    """
    Format a unified diff between old_text and new_text.

    Args:
        old_text: The original text.
        new_text: The new text.
        path: Optional file path for the diff header.
        include_file_header: Whether to include the ---/+++ file header lines.
        old_start_line: The 1-based starting line number in the old file for this diff snippet.
        new_start_line: The 1-based starting line number in the new file for this diff snippet.

    Returns:
        A unified diff string.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    # Ensure lines end with newline for proper diff formatting
    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    fromfile = f"a/{path}" if path else "a/file"
    tofile = f"b/{path}" if path else "b/file"

    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=fromfile,
            tofile=tofile,
            lineterm="\n",
        )
    )

    old_base_start = 1 if old_lines else 0
    new_base_start = 1 if new_lines else 0
    old_line_offset = old_start_line - old_base_start
    new_line_offset = new_start_line - new_base_start
    if old_line_offset or new_line_offset:
        diff = [
            _offset_hunk_header(line, old_line_offset, new_line_offset)
            if line.startswith("@@ ")
            else line
            for line in diff
        ]

    if (
        not include_file_header
        and len(diff) >= 2
        and diff[0].startswith("--- ")
        and diff[1].startswith("+++ ")
    ):
        diff = diff[2:]

    return "".join(diff)


def _offset_hunk_header(line: str, old_line_offset: int, new_line_offset: int) -> str:
    match = _HUNK_HEADER_RE.match(line)
    if match is None:
        return line

    old_start = int(match.group("old_start")) + old_line_offset
    new_start = int(match.group("new_start")) + new_line_offset
    old_count = match.group("old_count")
    new_count = match.group("new_count")

    old_range = f"{old_start},{old_count}" if old_count is not None else str(old_start)
    new_range = f"{new_start},{new_count}" if new_count is not None else str(new_start)
    return _HUNK_HEADER_RE.sub(f"@@ -{old_range} +{new_range} @@", line, count=1)


def _build_diff_blocks_sync(
    path: str,
    old_text: str,
    new_text: str,
) -> list[DisplayBlock]:
    """Synchronous diff block builder — CPU-bound, meant to run in a thread."""
    if old_text == new_text:
        return []

    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    max_lines = max(len(old_lines), len(new_lines))

    # Huge files: skip diff entirely, return a summary block
    if max_lines > _HUGE_FILE_THRESHOLD:
        old_desc = f"({len(old_lines)} lines)"
        if len(old_lines) == len(new_lines):
            new_desc = f"({len(new_lines)} lines, modified)"
        else:
            new_desc = f"({len(new_lines)} lines)"
        return [
            DiffDisplayBlock(
                path=path,
                old_text=old_desc,
                new_text=new_desc,
                old_start_line=1,
                new_start_line=1,
            )
        ]

    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    blocks: list[DiffDisplayBlock] = []
    for group in matcher.get_grouped_opcodes(n=N_CONTEXT_LINES):
        if not group:
            continue
        i1 = group[0][1]
        i2 = group[-1][2]
        j1 = group[0][3]
        j2 = group[-1][4]
        old_chunk = old_lines[i1:i2]
        new_chunk = new_lines[j1:j2]
        blocks.append(
            DiffDisplayBlock(
                path=path,
                old_text="\n".join(old_chunk),
                new_text="\n".join(new_chunk),
                old_start_line=i1 + 1 if old_chunk else 0,
                new_start_line=j1 + 1 if new_chunk else 0,
            )
        )
    return blocks


async def build_diff_blocks(
    path: str,
    old_text: str,
    new_text: str,
) -> list[DisplayBlock]:
    """Build diff display blocks, offloaded to a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(_build_diff_blocks_sync, path, old_text, new_text)
