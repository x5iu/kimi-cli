"""Deterministic projection layer for TaskOutput.

Pure functions that turn raw background-task output (Claude stream-json,
Codex JSONL, Kimi/agent NDJSON, or plain text) into a concise, bounded
model-facing summary. Raw logs always remain available via the
``output_path`` field on the tool response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast


class OutputFormat(str, Enum):
    CLAUDE_STREAM_JSON = "claude_stream_json"
    CODEX_JSONL = "codex_jsonl"
    KIMI_AGENT_NDJSON = "kimi_agent_ndjson"
    PLAIN_TEXT = "plain_text"


@dataclass
class ProjectedEvent:
    kind: str
    text: str


@dataclass
class TaskOutputProjection:
    format: OutputFormat
    events: list[ProjectedEvent] = field(default_factory=list[ProjectedEvent])
    truncated: bool = False
    raw_lines_seen: int = 0


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


_CLAUDE_TOP_TYPES = {"system", "assistant", "user", "result"}
_CODEX_TYPES = {
    "task_started",
    "task_complete",
    "agent_message",
    "agent_message_delta",
    "turn.completed",
    "turn_completed",
    "response.output_text.delta",
    "response.completed",
    "tool_call",
    "tool_call_output",
}


def detect_format(sample: str, *, kind: str | None = None) -> OutputFormat:
    """Best-effort format detection from a small sample of output text."""
    found_json = False
    for raw_line in sample.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not (line.startswith("{") and line.endswith("}")):
            return OutputFormat.PLAIN_TEXT
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return OutputFormat.PLAIN_TEXT
        if not isinstance(obj, dict):
            return OutputFormat.PLAIN_TEXT
        obj_dict = cast(dict[str, Any], obj)
        found_json = True
        t = obj_dict.get("type")
        if t in _CLAUDE_TOP_TYPES and ("message" in obj_dict or t == "result" or t == "system"):
            return OutputFormat.CLAUDE_STREAM_JSON
        if t in _CODEX_TYPES:
            return OutputFormat.CODEX_JSONL
        if "role" in obj_dict:
            return OutputFormat.KIMI_AGENT_NDJSON
        # Otherwise keep scanning
    if found_json:
        return OutputFormat.KIMI_AGENT_NDJSON
    return OutputFormat.PLAIN_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in cast(list[Any], content):
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                blk = cast(dict[str, Any], block)
                btype = blk.get("type")
                btext = blk.get("text")
                if btype == "text" and isinstance(btext, str):
                    parts.append(btext)
                elif btype == "tool_use":
                    name = blk.get("name", "?")
                    parts.append(f"[tool_use:{name}]")
                elif btype == "tool_result":
                    parts.append("[tool_result]")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        cdict = cast(dict[str, Any], content)
        text = cdict.get("text") or cdict.get("message") or cdict.get("content")
        if isinstance(text, str):
            return text
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)  # type: ignore[reportUnknownArgumentType]


def _project_assistant_message(msg: Any) -> list[ProjectedEvent]:
    if not isinstance(msg, dict):
        return []
    msg_dict = cast(dict[str, Any], msg)
    events: list[ProjectedEvent] = []
    content = msg_dict.get("content")
    if isinstance(content, list):
        for block in cast(list[Any], content):
            if not isinstance(block, dict):
                continue
            blk = cast(dict[str, Any], block)
            btype = blk.get("type")
            btext = blk.get("text")
            if btype == "text" and isinstance(btext, str):
                text = btext.strip()
                if text:
                    events.append(ProjectedEvent("assistant", _truncate(text, 2000)))
            elif btype == "tool_use":
                name = blk.get("name", "?")
                inp = blk.get("input")
                arg_text = ""
                if inp is not None:
                    try:
                        arg_text = json.dumps(inp, ensure_ascii=False)
                    except (TypeError, ValueError):
                        arg_text = str(inp)
                events.append(ProjectedEvent("tool_call", f"{name}({_truncate(arg_text, 200)})"))
    elif isinstance(content, str):
        text = content.strip()
        if text:
            events.append(ProjectedEvent("assistant", _truncate(text, 2000)))
    return events


def _project_claude_line(obj: dict[str, Any]) -> list[ProjectedEvent]:
    t = obj.get("type")
    if t == "system":
        return []
    if t == "assistant":
        return _project_assistant_message(obj.get("message"))
    if t == "user":
        return []
    if t == "result":
        result_obj: Any = obj.get("result")
        if isinstance(result_obj, dict):
            inner = cast(dict[str, Any], result_obj).get("result")
            if isinstance(inner, str):
                result_obj = inner
        if isinstance(result_obj, str) and result_obj.strip():
            return [ProjectedEvent("result", _truncate(result_obj.strip(), 4000))]
        if obj.get("is_error") or obj.get("subtype") == "error":
            return [ProjectedEvent("error", _truncate(json.dumps(obj)[:500], 500))]
        return [ProjectedEvent("result", "(no text result)")]
    return []


def _project_codex_line(obj: dict[str, Any]) -> list[ProjectedEvent]:
    t = obj.get("type")
    if t in {"agent_message_delta", "response.output_text.delta", "delta"}:
        return []
    if t == "agent_message":
        text = obj.get("message") or obj.get("text") or _stringify_content(obj.get("content"))
        if isinstance(text, str) and text.strip():
            return [ProjectedEvent("assistant", _truncate(text.strip(), 2000))]
        return []
    if t == "task_started":
        return [ProjectedEvent("status", "task started")]
    if t in {"task_complete", "turn.completed", "turn_completed", "response.completed"}:
        return [ProjectedEvent("status", "task completed")]
    if t in {"tool_call", "function_call"}:
        name = obj.get("name") or obj.get("tool") or "?"
        args: Any = obj.get("arguments")
        if args is None:
            args = obj.get("input") or ""
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        return [ProjectedEvent("tool_call", f"{name}({_truncate(args, 200)})")]
    if t in {"tool_call_output", "function_call_output"}:
        return []
    if t == "error":
        msg = obj.get("message") or obj.get("error") or json.dumps(obj)[:500]
        return [ProjectedEvent("error", _truncate(str(msg), 500))]
    return []


def _project_kimi_line(obj: dict[str, Any]) -> list[ProjectedEvent]:
    role = obj.get("role")
    if role == "assistant":
        events: list[ProjectedEvent] = []
        text = _stringify_content(obj.get("content")).strip()
        if text:
            events.append(ProjectedEvent("assistant", _truncate(text, 2000)))
        tool_calls = obj.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in cast(list[Any], tool_calls):
                if not isinstance(tc, dict):
                    continue
                tc_dict = cast(dict[str, Any], tc)
                fn_obj: Any = tc_dict.get("function") or {}
                if isinstance(fn_obj, dict):
                    fn_dict = cast(dict[str, Any], fn_obj)
                    name = fn_dict.get("name", "?")
                    args: Any = fn_dict.get("arguments", "")
                else:
                    name = "?"
                    args = ""
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args = str(args)
                events.append(ProjectedEvent("tool_call", f"{name}({_truncate(args, 200)})"))
        return events
    if role == "tool":
        snippet = _stringify_content(obj.get("content"))
        return [ProjectedEvent("tool_result", _truncate(snippet, 200))]
    if role == "user":
        snippet = _stringify_content(obj.get("content"))
        if snippet.strip():
            return [ProjectedEvent("user", _truncate(snippet.strip(), 500))]
    if role == "system":
        return []
    return []


# ---------------------------------------------------------------------------
# Top-level projection
# ---------------------------------------------------------------------------


def project(
    text: str,
    *,
    format: OutputFormat | None = None,
    kind: str | None = None,
    max_events: int = 80,
    max_bytes: int = 4096,
) -> TaskOutputProjection:
    """Project a raw output chunk into a concise event stream.

    Always returns a ``TaskOutputProjection``. ``max_events`` bounds the
    number of retained events; ``max_bytes`` bounds the total rendered
    payload size after :func:`render_projection` is invoked.
    """
    if format is None:
        format = detect_format(text, kind=kind)

    proj = TaskOutputProjection(format=format)
    if not text:
        return proj

    if format is OutputFormat.PLAIN_TEXT:
        # Bounded tail digest. Lines are kept whole; oldest are dropped.
        lines = text.splitlines()
        proj.raw_lines_seen = len(lines)
        kept: list[str] = []
        budget = max(0, max_bytes)
        for line in reversed(lines):
            chunk_size = len(line) + 1
            if chunk_size > budget:
                proj.truncated = True
                break
            kept.append(line)
            budget -= chunk_size
            if len(kept) >= max_events:
                proj.truncated = True
                break
        kept.reverse()
        if len(kept) < len(lines):
            proj.truncated = True
        for line in kept:
            proj.events.append(ProjectedEvent("text", line))
        return proj

    # Structured formats: parse line-by-line.
    raw_lines = text.splitlines()
    proj.raw_lines_seen = len(raw_lines)
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            proj.events.append(ProjectedEvent("text", _truncate(line, 500)))
            continue
        if not isinstance(obj, dict):
            continue
        obj_dict = cast(dict[str, Any], obj)
        if format is OutputFormat.CLAUDE_STREAM_JSON:
            new = _project_claude_line(obj_dict)
        elif format is OutputFormat.CODEX_JSONL:
            new = _project_codex_line(obj_dict)
        else:
            new = _project_kimi_line(obj_dict)
        proj.events.extend(new)
        if len(proj.events) > max_events:
            # Keep the most recent ``max_events``.
            proj.events = proj.events[-max_events:]
            proj.truncated = True
    return proj


def render_projection(projection: TaskOutputProjection, *, max_bytes: int = 4096) -> str:
    """Render a projection to a bounded text payload.

    Lines are kept whole; oldest events are dropped if the rendered
    payload would exceed ``max_bytes``.
    """
    if not projection.events:
        return ""

    rendered_lines: list[str] = []
    for event in projection.events:
        prefix = f"[{event.kind}] "
        for i, body_line in enumerate(event.text.splitlines() or [""]):
            rendered_lines.append((prefix if i == 0 else "    ") + body_line)

    truncation_marker = "[…older summarized events dropped…]"
    truncated = False

    def _size(lines: list[str], with_marker: bool) -> int:
        body = "\n".join(lines)
        if with_marker:
            body = truncation_marker + "\n" + body
        return len(body.encode("utf-8"))

    while rendered_lines and _size(rendered_lines, truncated) > max_bytes:
        rendered_lines.pop(0)
        truncated = True

    if not rendered_lines:
        return ""

    rendered = "\n".join(rendered_lines)
    if truncated:
        rendered = truncation_marker + "\n" + rendered
    return rendered
