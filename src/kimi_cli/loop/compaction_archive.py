from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

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
from kimi_cli.utils.metrics import emit_metric
from kimi_cli.utils.turns import is_checkpoint_user_text, is_internal_user_message
from llmkit.message import Message, ToolCall

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

fcntl: Any = _fcntl

_MANIFEST_SUFFIX = ".compaction-archives.jsonl"
_SUMMARY_WIDTH = 600

_KNOWN_NOISE = frozenset(
    {
        "subtype",
        "thinking",
        "delta",
        "ninternal",
        "treturn",
        "tgithub",
        "tt.fatal",
        "treturning",
        "tgithubusercontent",
        "content",
        "text",
        "message",
        "role",
        "function",
        "arguments",
        "tool_calls",
        "tool_call_id",
        "index",
        "id",
        "finish_reason",
        "object",
        "choices",
    }
)

_STREAM_KEY_PATTERNS = (
    (re.compile(r'["\']subtype["\']\s*:\s*["\'][^"\']{0,96}["\']', re.I), " "),
    (re.compile(r'["\']thinking["\']\s*:\s*', re.I), " "),
    (re.compile(r'["\']delta["\']\s*:\s*', re.I), " "),
    (re.compile(r"\{[^{}]{0,120}?\"subtype\"\s*:\s*\"[^\"]{0,32}\"\s*\}", re.I), " "),
)

_TYPE_KV_RE = re.compile(r'["\']type["\']\s*:\s*["\']([^"\']{1,96})["\']', re.I)

_STREAM_ENVELOPE_TYPE_VALUES = frozenset(
    {
        "message",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_start",
        "message_delta",
        "message_stop",
        "thinking",
        "ping",
        "error",
        "assistant",
        "user",
        "system",
        "tool_calls",
        "function_call",
        "function_call_arguments",
        "response",
        "response.created",
        "response.completed",
        "response.failed",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "text",
        "json",
        "input_json_delta",
    }
)


def _redact_stream_type_kv(match: re.Match[str]) -> str:
    raw = match.group(0)
    val = match.group(1)
    if any("A" <= c <= "Z" for c in val):
        return raw
    if val.lower() in _STREAM_ENVELOPE_TYPE_VALUES:
        return " "
    return raw


def _truncate_summary(summary: str) -> str:
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
    pending: bool = False


class ArchiveRegistrationResult(BaseModel):
    record: CompactionArchiveRecord
    total_archives: int


def manifest_path_for_context(context_file: Path) -> Path:
    return context_file.with_name(f"{context_file.stem}{_MANIFEST_SUFFIX}")


def _lock_manifest_file(f: TextIO) -> None:
    if fcntl is None:
        return
    with suppress(OSError):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_manifest_file(f: TextIO) -> None:
    if fcntl is None:
        return
    with suppress(OSError):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _parse_manifest_lines(manifest_path: Path) -> list[CompactionArchiveRecord]:
    records: list[CompactionArchiveRecord] = []
    if not manifest_path.exists():
        return records
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
    return records


def _parse_manifest_from_open_file(f: TextIO) -> list[CompactionArchiveRecord]:
    f.seek(0)
    body = f.read()
    records: list[CompactionArchiveRecord] = []
    for line_no, line in enumerate(body.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(CompactionArchiveRecord.model_validate_json(line))
        except Exception:
            logger.warning(
                "Skipping malformed compaction archive record at line {line_no}",
                line_no=line_no,
            )
    return records


def _dedupe_last_wins(records: Sequence[CompactionArchiveRecord]) -> list[CompactionArchiveRecord]:
    first_seen: dict[str, int] = {}
    last_content: dict[str, CompactionArchiveRecord] = {}
    for idx, rec in enumerate(records):
        if rec.id not in first_seen:
            first_seen[rec.id] = idx
        last_content[rec.id] = rec
    if len(first_seen) < len(records):
        return [last_content[rid] for rid, _ in sorted(first_seen.items(), key=lambda x: x[1])]
    return list(records)


def _committed_visible_records(
    deduped: Sequence[CompactionArchiveRecord],
) -> list[CompactionArchiveRecord]:
    return [r for r in deduped if not r.pending]


def _max_numeric_archive_id(all_records: Sequence[CompactionArchiveRecord]) -> int:
    m = 0
    for r in all_records:
        if len(r.id) >= 2 and r.id[0] == "c":
            with suppress(ValueError):
                m = max(m, int(r.id[1:]))
    return m


def load_compaction_archives(context_file: Path) -> list[CompactionArchiveRecord]:
    manifest_path = manifest_path_for_context(context_file)
    records = _parse_manifest_lines(manifest_path)
    deduped = _dedupe_last_wins(records)
    return _committed_visible_records(deduped)


def scrub_text_for_archive_keywords(text: str) -> str:
    s = text
    s = re.sub(r"\\[ntr]", " ", s)
    for pat, repl in _STREAM_KEY_PATTERNS:
        s = pat.sub(repl, s)
    s = _TYPE_KV_RE.sub(_redact_stream_type_kv, s)
    s = re.sub(r"[\n\r\t]+", " ", s)
    return s


def _emit_keyword_metric(keywords: list[str], *, noisy_removed: int, total_candidates: int) -> None:
    emit_metric(
        "archive.keywords",
        noisy_removed=noisy_removed,
        total_candidates=total_candidates,
        keyword_count=len(keywords),
    )


def register_compaction_archive(
    context_file: Path,
    archive_file: Path,
    *,
    messages: Sequence[Message] = (),
    message_count: int,
    summary: str,
) -> ArchiveRegistrationResult:
    manifest_path = manifest_path_for_context(context_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    keywords, noisy_removed, total_candidates = _extract_archive_keywords_with_stats(messages)
    f = manifest_path.open("a+", encoding="utf-8")
    try:
        _lock_manifest_file(f)
        raw = _parse_manifest_from_open_file(f)
        max_id = _max_numeric_archive_id(raw)
        record = CompactionArchiveRecord(
            id=f"c{max_id + 1:03d}",
            archive_file=archive_file.name,
            created_at=datetime.now().astimezone().isoformat(),
            message_count=message_count,
            summary=_truncate_summary(summary),
            keywords=keywords,
            pending=False,
        )
        f.seek(0, os.SEEK_END)
        f.write(record.model_dump_json() + "\n")
        f.flush()
        committed = _committed_visible_records(_dedupe_last_wins(raw + [record]))
    finally:
        _unlock_manifest_file(f)
        f.close()
    _emit_keyword_metric(keywords, noisy_removed=noisy_removed, total_candidates=total_candidates)
    return ArchiveRegistrationResult(record=record, total_archives=len(committed))


def begin_compaction_archive_registration(
    context_file: Path,
    archive_file: Path,
    *,
    messages: Sequence[Message] = (),
    message_count: int,
    summary: str,
) -> ArchiveRegistrationResult:
    manifest_path = manifest_path_for_context(context_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    keywords, noisy_removed, total_candidates = _extract_archive_keywords_with_stats(messages)
    pending_line: CompactionArchiveRecord | None = None
    total_archives = 0
    f = manifest_path.open("a+", encoding="utf-8")
    try:
        _lock_manifest_file(f)
        raw = _parse_manifest_from_open_file(f)
        max_id = _max_numeric_archive_id(raw)
        rid = f"c{max_id + 1:03d}"
        pending_line = CompactionArchiveRecord(
            id=rid,
            archive_file=archive_file.name,
            created_at=datetime.now().astimezone().isoformat(),
            message_count=message_count,
            summary=_truncate_summary(summary),
            keywords=keywords,
            pending=True,
        )
        committed_before = _committed_visible_records(_dedupe_last_wins(raw))
        total_archives = len(committed_before) + 1
        f.seek(0, os.SEEK_END)
        f.write(pending_line.model_dump_json() + "\n")
        f.flush()
    finally:
        _unlock_manifest_file(f)
        f.close()
    assert pending_line is not None
    committed_record = pending_line.model_copy(update={"pending": False})
    _emit_keyword_metric(keywords, noisy_removed=noisy_removed, total_candidates=total_candidates)
    return ArchiveRegistrationResult(record=committed_record, total_archives=total_archives)


def finalize_compaction_archive_registration(
    context_file: Path, record: CompactionArchiveRecord
) -> None:
    manifest_path = manifest_path_for_context(context_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    final = record.model_copy(update={"pending": False})
    f = manifest_path.open("a+", encoding="utf-8")
    try:
        _lock_manifest_file(f)
        f.seek(0, os.SEEK_END)
        f.write(final.model_dump_json() + "\n")
        f.flush()
    finally:
        _unlock_manifest_file(f)
        f.close()


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
    r"(?:[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)"
    r"|(?:[\w./:-]{2,}/[\w./:-]+)"
    r"|(?:[A-Z][a-zA-Z0-9]+(?:Error|Exception|Warning))"
    r"|(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+)",
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


def _extract_archive_keywords_with_stats(
    messages: Sequence[Message],
) -> tuple[list[str], int, int]:
    counts: Counter[str] = Counter()
    for message in messages:
        raw_text = stringify_message_for_archive(message)
        text = scrub_text_for_archive_keywords(raw_text)

        for candidate in _KEYWORD_RE.findall(text):
            lowered = candidate.lower()
            if len(lowered) >= 3 and lowered not in _STOP_KEYWORDS:
                counts[lowered] += 2

        for candidate in _WORD_RE.findall(text):
            lowered = candidate.lower()
            if lowered not in _STOP_KEYWORDS:
                counts[lowered] += 1

        for candidate in _CJK_RE.findall(text):
            if candidate not in _STOP_KEYWORDS:
                counts[candidate] += 1

    total_candidates = sum(counts.values())
    noisy_removed = 0
    for noise in list(counts.keys()):
        nk = noise.lower() if noise.isascii() else noise
        if nk in _KNOWN_NOISE:
            noisy_removed += counts.pop(noise, 0)

    ranked = [kw for kw, _ in counts.most_common(_MAX_KEYWORDS)]
    return ranked, noisy_removed, total_candidates


def extract_archive_keywords(messages: Sequence[Message]) -> list[str]:
    kws, _, _ = _extract_archive_keywords_with_stats(messages)
    return kws


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
    records = load_compaction_archives(context_file)
    if not records:
        return 0

    updated_records: list[CompactionArchiveRecord] = []
    for record in records:
        need = not record.keywords or bool(set(record.keywords) & _KNOWN_NOISE)
        if not need:
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

    manifest = manifest_path_for_context(context_file)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    f = manifest.open("a+", encoding="utf-8")
    try:
        _lock_manifest_file(f)
        for rec in updated_records:
            f.seek(0, os.SEEK_END)
            f.write(rec.model_dump_json() + "\n")
        f.flush()
    finally:
        _unlock_manifest_file(f)
        f.close()

    return len(updated_records)
