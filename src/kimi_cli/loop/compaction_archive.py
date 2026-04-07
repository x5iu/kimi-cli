from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from kimi_cli.eventbus.types import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    VideoURLPart,
)
from kimi_cli.utils.logging import logger
from kimi_cli.utils.turns import is_checkpoint_user_text, is_internal_user_message
from llmkit.message import Message, ToolCall

_MANIFEST_SUFFIX = ".compaction-archives.jsonl"
_SUMMARY_WIDTH = 600


def _truncate_summary(summary: str) -> str:
    """Truncate summary to at most _SUMMARY_WIDTH characters, adding '...' if truncated."""
    if not summary:
        return ""
    stripped = summary.strip()
    if len(stripped) <= _SUMMARY_WIDTH:
        return stripped
    return stripped[:_SUMMARY_WIDTH] + "..."


class CompactionArchiveRecord(BaseModel):
    schema_version: int = 1
    id: str
    archive_file: str
    created_at: str
    message_count: int
    summary: str = ""
    keywords: list[str] = []


class ArchiveRegistrationResult(BaseModel):
    record: CompactionArchiveRecord
    total_archives: int


def manifest_path_for_context(context_file: Path) -> Path:
    return context_file.with_name(f"{context_file.stem}{_MANIFEST_SUFFIX}")


def load_compaction_archives(context_file: Path) -> list[CompactionArchiveRecord]:
    manifest_path = manifest_path_for_context(context_file)
    if not manifest_path.exists():
        return []

    records: list[CompactionArchiveRecord] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(CompactionArchiveRecord.model_validate_json(line))
            except Exception:
                logger.warning(
                    "Skipping malformed compaction archive record in {path}:{line_no}",
                    path=manifest_path,
                    line_no=line_no,
                )

    # Deduplicate by id: preserve the chronological position of the first
    # occurrence but use the content from the last occurrence (supports
    # append-only keyword backfills).
    first_seen: dict[str, int] = {}
    last_content: dict[str, CompactionArchiveRecord] = {}
    for idx, rec in enumerate(records):
        if rec.id not in first_seen:
            first_seen[rec.id] = idx
        last_content[rec.id] = rec
    if len(first_seen) < len(records):
        records = [last_content[rid] for rid, _ in sorted(first_seen.items(), key=lambda x: x[1])]

    return records


def register_compaction_archive(
    context_file: Path,
    archive_file: Path,
    *,
    messages: Sequence[Message] = (),
    message_count: int,
    summary: str,
) -> ArchiveRegistrationResult:
    records = load_compaction_archives(context_file)
    max_id = max((int(r.id[1:]) for r in records), default=0)
    keywords = extract_archive_keywords(messages)
    record = CompactionArchiveRecord(
        id=f"c{max_id + 1:03d}",
        archive_file=archive_file.name,
        created_at=datetime.now().astimezone().isoformat(),
        message_count=message_count,
        summary=_truncate_summary(summary),
        keywords=keywords,
    )

    # NOTE: Assumes single-writer — concurrent compactions on the same trajectory
    # are not expected.  If that changes, add file-level locking here.
    manifest_path = manifest_path_for_context(context_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")

    return ArchiveRegistrationResult(record=record, total_archives=len(records) + 1)


def resolve_compaction_archive_path(context_file: Path, record: CompactionArchiveRecord) -> Path:
    return context_file.parent / record.archive_file


def sanitize_archive_text(text: str) -> str:
    sanitized = text.replace("<system-reminder>", "[system-reminder]\n")
    sanitized = sanitized.replace("</system-reminder>", "")
    sanitized = sanitized.replace("<system-hint>", "[system-hint]\n")
    sanitized = sanitized.replace("</system-hint>", "")
    sanitized = sanitized.replace("<system>", "[system] ")
    sanitized = sanitized.replace("</system>", "")
    return sanitized.strip()


def stringify_content_for_archive(
    parts: Sequence[ContentPart], *, include_thinking: bool = False
) -> str:
    segments: list[str] = []
    for part in parts:
        match part:
            case TextPart(text=text):
                sanitized = sanitize_archive_text(text)
                if sanitized:
                    segments.append(sanitized)
            case ThinkPart(think=think):
                if include_thinking and think.strip():
                    segments.append(f"[thinking]\n{think.strip()}")
            case ImageURLPart():
                segments.append("[image]")
            case AudioURLPart():
                segments.append("[audio]")
            case VideoURLPart():
                segments.append("[video]")
            case _:
                segments.append(f"[{part.type}]")
    return "\n".join(segments)


def stringify_tool_calls(tool_calls: Sequence[ToolCall]) -> str:
    lines: list[str] = []
    for tc in tool_calls:
        args_raw = tc.function.arguments or "{}"
        try:
            args = json.loads(args_raw)
            args_text = json.dumps(args, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            args_text = args_raw
        lines.append(f"Tool Call: {tc.function.name}({args_text})")
    return "\n".join(lines)


def stringify_message_for_archive(message: Message, *, include_thinking: bool = False) -> str:
    segments: list[str] = []
    content_text = stringify_content_for_archive(message.content, include_thinking=include_thinking)
    if content_text:
        segments.append(content_text)
    if message.tool_calls:
        segments.append(stringify_tool_calls(message.tool_calls))
    return "\n".join(segments).strip()


def build_compaction_summary(messages: Sequence[Message]) -> str:
    if not messages:
        return ""
    prefix = "[system] Previous context has been compacted. Here is the compaction output:"
    for message in messages:
        text = stringify_message_for_archive(message)
        if not text:
            continue
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
        if text:
            return text
    return ""


_KEYWORD_RE = re.compile(
    r"(?:[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)"  # dotted identifiers (paths, modules)
    r"|(?:[\w./:-]{2,}/[\w./:-]+)"  # file paths with slashes
    r"|(?:[A-Z][a-zA-Z0-9]+(?:Error|Exception|Warning))"  # error class names
    r"|(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+)",  # CamelCase identifiers
)

_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b", re.ASCII)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

_STOP_KEYWORDS = frozenset(
    {
        "true",
        "false",
        "none",
        "null",
        "self",
        "return",
        "import",
        "from",
        "class",
        "async",
        "await",
        "with",
        "that",
        "this",
        "have",
        "been",
        "will",
        "would",
        "could",
        "should",
        "about",
        "there",
        "their",
        "which",
        "when",
        "what",
        "were",
        "them",
        "then",
        "than",
        "each",
        "make",
        "like",
        "just",
        "over",
        "such",
        "into",
        "only",
        "also",
        "some",
        "very",
        "here",
        "more",
        "after",
        "before",
        "other",
        "these",
        "first",
        "using",
        "file",
        "line",
        "code",
        "output",
        "error",
        "tool",
        "call",
        "type",
        "text",
        "content",
        "message",
        "role",
        "user",
        "assistant",
        "system",
        "function",
        "the",
    }
)

_MAX_KEYWORDS = 10


def extract_archive_keywords(messages: Sequence[Message]) -> list[str]:
    """Extract top distinctive keywords from archived messages."""
    counts: Counter[str] = Counter()
    for message in messages:
        text = stringify_message_for_archive(message)

        # Code identifiers – higher weight (x2)
        for candidate in _KEYWORD_RE.findall(text):
            lowered = candidate.lower()
            if len(lowered) >= 3 and lowered not in _STOP_KEYWORDS:
                counts[lowered] += 2

        # Plain English words – weight x1
        for candidate in _WORD_RE.findall(text):
            lowered = candidate.lower()
            if lowered not in _STOP_KEYWORDS:
                counts[lowered] += 1

        # CJK terms – weight x1
        for candidate in _CJK_RE.findall(text):
            if candidate not in _STOP_KEYWORDS:
                counts[candidate] += 1

    return [kw for kw, _ in counts.most_common(_MAX_KEYWORDS)]


def is_checkpoint_message(message: Message) -> bool:
    if message.role != "user":
        return False
    for part in message.content:
        if isinstance(part, TextPart):
            return is_checkpoint_user_text(part.text)
    return False


_archive_cache: dict[Path, tuple[float, list[Message]]] = {}


def load_archive_messages(archive_file: Path) -> list[Message]:
    try:
        mtime = archive_file.stat().st_mtime
    except OSError:
        mtime = 0.0
    cached = _archive_cache.get(archive_file)
    if cached is not None and cached[0] == mtime:
        return list(cached[1])

    messages: list[Message] = []
    with archive_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed archive line in {path}:{line_no}",
                    path=archive_file,
                    line_no=line_no,
                )
                continue
            role = raw.get("role")
            if role in {"_usage", "_checkpoint"}:
                continue
            try:
                message = Message.model_validate(raw)
            except Exception:
                logger.warning(
                    "Skipping invalid archive message in {path}:{line_no}",
                    path=archive_file,
                    line_no=line_no,
                )
                continue
            if is_checkpoint_message(message):
                continue
            messages.append(message)

    _archive_cache[archive_file] = (mtime, messages)
    return messages


def archive_role_label(message: Message) -> str:
    role = message.role
    if message.role == "user" and is_internal_user_message(message):
        role += " [internal]"
    if message.role == "tool" and message.tool_call_id:
        role += f" [{message.tool_call_id}]"
    return role


def backfill_archive_keywords(context_file: Path) -> int:
    """Re-extract keywords for archives with empty keyword lists.

    Returns the number of records updated.  Uses append-only writes so
    that concurrent ``register_compaction_archive`` calls are never lost.
    ``load_compaction_archives`` deduplicates by id (last occurrence wins).
    """
    records = load_compaction_archives(context_file)
    if not records:
        return 0

    # Collect updated records for archives with empty keyword lists.
    updated_records: list[CompactionArchiveRecord] = []
    for record in records:
        if record.keywords:
            continue
        archive_path = resolve_compaction_archive_path(context_file, record)
        if not archive_path.exists():
            continue
        messages = load_archive_messages(archive_path)
        new_keywords = extract_archive_keywords(messages)
        if new_keywords:
            updated_records.append(
                record.model_copy(
                    update={"keywords": new_keywords},
                )
            )

    if not updated_records:
        return 0

    # Append updated records to the manifest.  load_compaction_archives
    # deduplicates by id (last occurrence wins), so the appended lines
    # will supersede the originals without risking data loss from a
    # concurrent register_compaction_archive append.
    manifest = manifest_path_for_context(context_file)
    with manifest.open("a", encoding="utf-8") as f:
        for rec in updated_records:
            f.write(rec.model_dump_json() + "\n")

    return len(updated_records)
