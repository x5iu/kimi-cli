from __future__ import annotations

import json
import threading
from pathlib import Path

from kimi_cli.eventbus.types import (
    AudioURLPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    VideoURLPart,
)
from kimi_cli.loop.compaction_archive import (
    _SUMMARY_WIDTH,
    CompactionArchiveRecord,
    archive_role_label,
    backfill_archive_keywords,
    begin_compaction_archive_registration,
    build_compaction_summary,
    extract_archive_keywords,
    finalize_compaction_archive_registration,
    is_checkpoint_message,
    load_archive_messages,
    load_compaction_archives,
    manifest_path_for_context,
    register_compaction_archive,
    resolve_compaction_archive_path,
    sanitize_archive_text,
    stringify_content_for_archive,
    stringify_message_for_archive,
    stringify_tool_calls,
)
from kimi_cli.loop.message import INTERNAL_USER_NAME
from llmkit.message import Message, ToolCall

# ---------------------------------------------------------------------------
# sanitize_archive_text
# ---------------------------------------------------------------------------


def test_sanitize_archive_text_strips_system_tags():
    text = "<system>hello world</system>"
    assert sanitize_archive_text(text) == "[system] hello world"


def test_sanitize_archive_text_strips_system_reminder_tags():
    text = "<system-reminder>reminder text</system-reminder>"
    assert sanitize_archive_text(text) == "[system-reminder]\nreminder text"


def test_sanitize_archive_text_handles_empty_string():
    assert sanitize_archive_text("") == ""


def test_sanitize_archive_text_strips_surrounding_whitespace():
    assert sanitize_archive_text("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# stringify_content_for_archive
# ---------------------------------------------------------------------------


def test_stringify_content_text_only():
    parts = [TextPart(text="Hello world")]
    assert stringify_content_for_archive(parts) == "Hello world"


def test_stringify_content_multiple_text_parts():
    parts = [TextPart(text="first"), TextPart(text="second")]
    assert stringify_content_for_archive(parts) == "first\nsecond"


def test_stringify_content_image_part():
    parts = [
        ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/img.png")),
    ]
    assert stringify_content_for_archive(parts) == "[image]"


def test_stringify_content_audio_part():
    parts = [
        AudioURLPart(audio_url=AudioURLPart.AudioURL(url="https://example.com/a.mp3")),
    ]
    assert stringify_content_for_archive(parts) == "[audio]"


def test_stringify_content_video_part():
    parts = [
        VideoURLPart(video_url=VideoURLPart.VideoURL(url="https://example.com/v.mp4")),
    ]
    assert stringify_content_for_archive(parts) == "[video]"


def test_stringify_content_think_part_included():
    parts = [ThinkPart(think="deep thought")]
    result = stringify_content_for_archive(parts, include_thinking=True)
    assert result == "[thinking]\ndeep thought"


def test_stringify_content_think_part_excluded_by_default():
    parts = [ThinkPart(think="deep thought")]
    assert stringify_content_for_archive(parts) == ""


def test_stringify_content_empty_think_stripped_even_when_included():
    parts = [ThinkPart(think="   ")]
    result = stringify_content_for_archive(parts, include_thinking=True)
    assert result == ""


def test_stringify_content_mixed_parts():
    parts = [
        TextPart(text="hello"),
        ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/i.png")),
        ThinkPart(think="thinking"),
    ]
    result = stringify_content_for_archive(parts, include_thinking=True)
    assert result == "hello\n[image]\n[thinking]\nthinking"


# ---------------------------------------------------------------------------
# stringify_tool_calls
# ---------------------------------------------------------------------------


def _make_tool_call(name: str, arguments: str | None, tc_id: str = "tc1") -> ToolCall:
    return ToolCall(
        id=tc_id,
        function=ToolCall.FunctionBody(name=name, arguments=arguments),
    )


def test_stringify_tool_calls_valid_json():
    tc = _make_tool_call("read_file", '{"path": "/tmp/a.txt"}')
    result = stringify_tool_calls([tc])
    assert result == 'Tool Call: read_file({"path": "/tmp/a.txt"})'


def test_stringify_tool_calls_pretty_prints_unicode():
    tc = _make_tool_call("search", '{"q": "你好"}')
    result = stringify_tool_calls([tc])
    assert "你好" in result
    assert result == 'Tool Call: search({"q": "你好"})'


def test_stringify_tool_calls_invalid_json_fallback():
    tc = _make_tool_call("broken", "not-json{{{")
    result = stringify_tool_calls([tc])
    assert result == "Tool Call: broken(not-json{{{)"


def test_stringify_tool_calls_empty_arguments():
    tc = _make_tool_call("noop", None)
    result = stringify_tool_calls([tc])
    assert result == "Tool Call: noop({})"


def test_stringify_tool_calls_multiple():
    tcs = [
        _make_tool_call("a", '{"x":1}', tc_id="t1"),
        _make_tool_call("b", '{"y":2}', tc_id="t2"),
    ]
    result = stringify_tool_calls(tcs)
    lines = result.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("Tool Call: a(")
    assert lines[1].startswith("Tool Call: b(")


# ---------------------------------------------------------------------------
# stringify_message_for_archive
# ---------------------------------------------------------------------------


def test_stringify_message_with_tool_calls():
    msg = Message(
        role="assistant",
        content=[TextPart(text="Let me check")],
        tool_calls=[_make_tool_call("grep", '{"pattern": "foo"}')],
    )
    result = stringify_message_for_archive(msg)
    assert "Let me check" in result
    assert "Tool Call: grep(" in result


def test_stringify_message_empty_content():
    msg = Message(role="assistant", content=[])
    assert stringify_message_for_archive(msg) == ""


def test_stringify_message_tool_calls_only():
    msg = Message(
        role="assistant",
        content=[],
        tool_calls=[_make_tool_call("ls", "{}")],
    )
    result = stringify_message_for_archive(msg)
    assert result == "Tool Call: ls({})"


# ---------------------------------------------------------------------------
# build_compaction_summary
# ---------------------------------------------------------------------------


def test_build_compaction_summary_strips_prefix():
    prefix = "[system] Previous context has been compacted. Here is the compaction output:"
    msg = Message(
        role="assistant",
        content=[TextPart(text=f"{prefix}\nActual summary content")],
    )
    result = build_compaction_summary([msg])
    assert result == "Actual summary content"


def test_build_compaction_summary_empty_messages():
    assert build_compaction_summary([]) == ""


def test_build_compaction_summary_skips_empty_text_messages():
    msgs = [
        Message(role="assistant", content=[]),
        Message(role="user", content=[TextPart(text="Hello")]),
    ]
    result = build_compaction_summary(msgs)
    assert result == "Hello"


def test_build_compaction_summary_returns_first_non_empty():
    msgs = [
        Message(role="user", content=[TextPart(text="First")]),
        Message(role="user", content=[TextPart(text="Second")]),
    ]
    assert build_compaction_summary(msgs) == "First"


# ---------------------------------------------------------------------------
# is_checkpoint_message
# ---------------------------------------------------------------------------


def test_is_checkpoint_message_true():
    msg = Message(
        role="user",
        content=[TextPart(text="<system>CHECKPOINT 42</system>")],
    )
    assert is_checkpoint_message(msg) is True


def test_is_checkpoint_message_false_for_non_user_role():
    msg = Message(
        role="assistant",
        content=[TextPart(text="<system>CHECKPOINT 1</system>")],
    )
    assert is_checkpoint_message(msg) is False


def test_is_checkpoint_message_false_for_regular_user():
    msg = Message(role="user", content=[TextPart(text="Hello world")])
    assert is_checkpoint_message(msg) is False


def test_is_checkpoint_message_false_no_text_parts():
    msg = Message(
        role="user",
        content=[ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://x.com/i.png"))],
    )
    assert is_checkpoint_message(msg) is False


# ---------------------------------------------------------------------------
# load_archive_messages (file-based)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_archive_messages_skips_usage_and_checkpoint_roles(
    tmp_path: Path,
):
    archive = tmp_path / "archive.jsonl"
    user_msg = Message(role="user", content=[TextPart(text="hello")])
    lines = [
        json.dumps({"role": "_usage", "content": []}),
        json.dumps({"role": "_checkpoint", "content": []}),
        user_msg.model_dump_json(exclude_none=True),
    ]
    _write_jsonl(archive, lines)

    result = load_archive_messages(archive)
    assert len(result) == 1
    assert result[0].role == "user"


def test_load_archive_messages_skips_malformed_json(tmp_path: Path):
    archive = tmp_path / "archive.jsonl"
    good = Message(role="assistant", content=[TextPart(text="ok")])
    lines = [
        "NOT VALID JSON{{{",
        good.model_dump_json(exclude_none=True),
    ]
    _write_jsonl(archive, lines)

    result = load_archive_messages(archive)
    assert len(result) == 1
    assert result[0].role == "assistant"


def test_load_archive_messages_skips_checkpoint_user_messages(
    tmp_path: Path,
):
    archive = tmp_path / "archive.jsonl"
    checkpoint = Message(
        role="user",
        content=[TextPart(text="<system>CHECKPOINT 1</system>")],
    )
    normal = Message(role="user", content=[TextPart(text="normal")])
    lines = [
        checkpoint.model_dump_json(exclude_none=True),
        normal.model_dump_json(exclude_none=True),
    ]
    _write_jsonl(archive, lines)

    result = load_archive_messages(archive)
    assert len(result) == 1
    assert result[0].content[0].text == "normal"  # type: ignore[union-attr]


def test_load_archive_messages_skips_blank_lines(tmp_path: Path):
    archive = tmp_path / "archive.jsonl"
    msg = Message(role="user", content=[TextPart(text="hi")])
    lines = [
        "",
        msg.model_dump_json(exclude_none=True),
        "   ",
    ]
    _write_jsonl(archive, lines)

    result = load_archive_messages(archive)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# load_compaction_archives (file-based)
# ---------------------------------------------------------------------------


def test_load_compaction_archives_empty_file(tmp_path: Path):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    manifest = manifest_path_for_context(ctx)
    manifest.write_text("", encoding="utf-8")
    assert load_compaction_archives(ctx) == []


def test_load_compaction_archives_missing_manifest(tmp_path: Path):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    assert load_compaction_archives(ctx) == []


def test_load_compaction_archives_skips_malformed_lines(tmp_path: Path):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    manifest = manifest_path_for_context(ctx)
    good = CompactionArchiveRecord(
        id="c001",
        archive_file="a1.jsonl",
        created_at="2026-01-01T00:00:00+00:00",
        message_count=5,
        summary="first archive",
    )
    manifest.write_text(
        "BROKEN LINE\n" + good.model_dump_json() + "\n",
        encoding="utf-8",
    )
    records = load_compaction_archives(ctx)
    assert len(records) == 1
    assert records[0].id == "c001"


def test_load_compaction_archives_normal_records(tmp_path: Path):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    manifest = manifest_path_for_context(ctx)
    r1 = CompactionArchiveRecord(
        id="c001",
        archive_file="a1.jsonl",
        created_at="2026-01-01T00:00:00+00:00",
        message_count=3,
        summary="summary one",
    )
    r2 = CompactionArchiveRecord(
        id="c002",
        archive_file="a2.jsonl",
        created_at="2026-01-02T00:00:00+00:00",
        message_count=7,
        summary="summary two",
    )
    manifest.write_text(
        r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n",
        encoding="utf-8",
    )
    records = load_compaction_archives(ctx)
    assert len(records) == 2
    assert records[0].id == "c001"
    assert records[1].id == "c002"


# ---------------------------------------------------------------------------
# register_compaction_archive
# ---------------------------------------------------------------------------


def test_register_compaction_archive_increments_ids(tmp_path: Path):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    archive1 = tmp_path / "archive_1.jsonl"
    archive1.touch()
    archive2 = tmp_path / "archive_2.jsonl"
    archive2.touch()

    res1 = register_compaction_archive(ctx, archive1, message_count=5, summary="first")
    assert res1.record.id == "c001"
    assert res1.total_archives == 1

    res2 = register_compaction_archive(ctx, archive2, message_count=3, summary="second")
    assert res2.record.id == "c002"
    assert res2.total_archives == 2


def test_register_compaction_archive_truncates_long_summaries(
    tmp_path: Path,
):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    archive = tmp_path / "archive.jsonl"
    archive.touch()
    long_summary = "A" * 800

    result = register_compaction_archive(ctx, archive, message_count=1, summary=long_summary)
    assert len(result.record.summary) <= _SUMMARY_WIDTH + len("...")


def test_register_compaction_archive_id_uses_max_existing(
    tmp_path: Path,
):
    """ID should be max(existing) + 1, not len(records) + 1."""
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    manifest = manifest_path_for_context(ctx)

    # Manually write a record with id c005 (gap in IDs)
    r = CompactionArchiveRecord(
        id="c005",
        archive_file="old.jsonl",
        created_at="2026-01-01T00:00:00+00:00",
        message_count=10,
        summary="old",
    )
    manifest.write_text(r.model_dump_json() + "\n", encoding="utf-8")

    archive = tmp_path / "new.jsonl"
    archive.touch()
    result = register_compaction_archive(ctx, archive, message_count=2, summary="new")
    # Should be c006, not c002
    assert result.record.id == "c006"
    assert result.total_archives == 2


def test_register_compaction_archive_empty_summary(tmp_path: Path):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    archive = tmp_path / "archive.jsonl"
    archive.touch()

    result = register_compaction_archive(ctx, archive, message_count=1, summary="")
    assert result.record.summary == ""


# ---------------------------------------------------------------------------
# archive_role_label
# ---------------------------------------------------------------------------


def test_archive_role_label_internal_user():
    msg = Message(
        role="user",
        name=INTERNAL_USER_NAME,
        content=[TextPart(text="internal stuff")],
    )
    assert archive_role_label(msg) == "user [internal]"


def test_archive_role_label_tool_with_call_id():
    msg = Message(
        role="tool",
        content=[TextPart(text="result")],
        tool_call_id="call_123",
    )
    assert archive_role_label(msg) == "tool [call_123]"


def test_archive_role_label_plain_assistant():
    msg = Message(
        role="assistant",
        content=[TextPart(text="hello")],
    )
    assert archive_role_label(msg) == "assistant"


def test_archive_role_label_regular_user():
    msg = Message(
        role="user",
        content=[TextPart(text="regular user message")],
    )
    assert archive_role_label(msg) == "user"


def test_archive_role_label_system():
    msg = Message(role="system", content=[TextPart(text="sys")])
    assert archive_role_label(msg) == "system"


# ---------------------------------------------------------------------------
# manifest_path_for_context
# ---------------------------------------------------------------------------


def test_manifest_path_for_context():
    ctx = Path("/data/sessions/context.jsonl")
    result = manifest_path_for_context(ctx)
    assert result == Path("/data/sessions/context.compaction-archives.jsonl")


def test_manifest_path_for_context_preserves_parent():
    ctx = Path("/a/b/c/my_ctx.jsonl")
    result = manifest_path_for_context(ctx)
    assert result.parent == ctx.parent
    assert result.name == "my_ctx.compaction-archives.jsonl"


# ---------------------------------------------------------------------------
# resolve_compaction_archive_path
# ---------------------------------------------------------------------------


def test_resolve_compaction_archive_path():
    ctx = Path("/data/sessions/context.jsonl")
    record = CompactionArchiveRecord(
        id="c001",
        archive_file="context_1.jsonl",
        created_at="2026-01-01T00:00:00+00:00",
        message_count=5,
        summary="test",
    )
    result = resolve_compaction_archive_path(ctx, record)
    assert result == Path("/data/sessions/context_1.jsonl")


def test_extract_archive_keywords_english_and_code():
    msgs = [
        Message(
            role="assistant",
            content=[
                TextPart(
                    text=(
                        "The CompactionArchive module lives in kimi_cli.loop.compaction_archive. "
                        "It raised a ValueError when the summary was empty."
                    )
                )
            ],
        ),
    ]
    keywords = extract_archive_keywords(msgs)
    assert isinstance(keywords, list)
    assert len(keywords) <= 10
    assert "kimi_cli.loop.compaction_archive" in keywords


def test_extract_archive_keywords_cjk():
    msgs = [
        Message(
            role="user",
            content=[TextPart(text="请实现压缩功能，处理压缩归档的逻辑")],
        ),
    ]
    keywords = extract_archive_keywords(msgs)
    assert any(all("\u4e00" <= c <= "\u9fff" for c in kw) for kw in keywords), (
        f"Expected CJK keywords, got: {keywords}"
    )


def test_extract_archive_keywords_empty_messages():
    assert extract_archive_keywords([]) == []


def test_word_re_matches_at_cjk_boundary():
    from kimi_cli.loop.compaction_archive import _WORD_RE

    assert _WORD_RE.findall("功能test模块") == ["test"]
    assert _WORD_RE.findall("test功能") == ["test"]
    assert _WORD_RE.findall("功能test") == ["test"]


def test_backfill_archive_keywords(tmp_path: Path):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    archive_file = tmp_path / "context_1.jsonl"
    msgs = [
        Message(
            role="user",
            content=[TextPart(text="Implement CompactionArchive for kimi_cli.loop")],
        ),
        Message(
            role="assistant",
            content=[TextPart(text="Done implementing the compaction archive module.")],
        ),
    ]
    archive_file.write_text(
        "\n".join(m.model_dump_json(exclude_none=True) for m in msgs) + "\n",
        encoding="utf-8",
    )
    registration = register_compaction_archive(
        ctx, archive_file, message_count=2, summary="compaction archive impl"
    )
    assert registration.record.keywords == []

    updated = backfill_archive_keywords(ctx)
    assert updated == 1

    records = load_compaction_archives(ctx)
    assert len(records) == 1
    assert len(records[0].keywords) > 0

    assert backfill_archive_keywords(ctx) == 0


def test_truncate_summary_edge_cases():
    from kimi_cli.loop.compaction_archive import _truncate_summary

    assert _truncate_summary("") == ""
    assert _truncate_summary("   ") == ""
    text_at_limit = "A" * _SUMMARY_WIDTH
    assert _truncate_summary(text_at_limit) == text_at_limit
    text_over = "A" * (_SUMMARY_WIDTH + 1)
    result = _truncate_summary(text_over)
    assert result == "A" * _SUMMARY_WIDTH + "..."
    assert len(result) == _SUMMARY_WIDTH + 3


def test_backfill_does_not_clobber_concurrent_registration(
    tmp_path: Path,
):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()

    archive1 = tmp_path / "context_1.jsonl"
    msgs1 = [
        Message(
            role="user",
            content=[TextPart(text="Implement CompactionArchive")],
        ),
    ]
    archive1.write_text(
        "\n".join(m.model_dump_json(exclude_none=True) for m in msgs1) + "\n",
        encoding="utf-8",
    )
    register_compaction_archive(ctx, archive1, message_count=1, summary="first")

    archive2 = tmp_path / "context_2.jsonl"
    archive2.touch()
    register_compaction_archive(ctx, archive2, message_count=1, summary="second")

    updated = backfill_archive_keywords(ctx)
    assert updated == 1

    records = load_compaction_archives(ctx)
    ids = [r.id for r in records]
    assert "c001" in ids
    assert "c002" in ids
    assert len(records) == 2

    c001 = next(r for r in records if r.id == "c001")
    assert len(c001.keywords) > 0


def test_load_compaction_archives_deduplicates_by_id(
    tmp_path: Path,
):
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    manifest = manifest_path_for_context(ctx)

    old = CompactionArchiveRecord(
        id="c001",
        archive_file="a1.jsonl",
        created_at="2026-01-01T00:00:00+00:00",
        message_count=5,
        summary="first archive",
        keywords=[],
    )
    new = CompactionArchiveRecord(
        id="c001",
        archive_file="a1.jsonl",
        created_at="2026-01-01T00:00:00+00:00",
        message_count=5,
        summary="first archive",
        keywords=["compaction", "archive"],
    )
    other = CompactionArchiveRecord(
        id="c002",
        archive_file="a2.jsonl",
        created_at="2026-01-02T00:00:00+00:00",
        message_count=3,
        summary="second archive",
    )
    manifest.write_text(
        old.model_dump_json()
        + "\n"
        + other.model_dump_json()
        + "\n"
        + new.model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    records = load_compaction_archives(ctx)
    assert len(records) == 2
    assert records[0].id == "c001"
    assert records[1].id == "c002"
    assert records[0].keywords == ["compaction", "archive"]


def test_begin_finalize_pending_manifest(tmp_path: Path) -> None:
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    arch = tmp_path / "context_1.jsonl"
    arch.touch()
    begun = begin_compaction_archive_registration(ctx, arch, message_count=1, summary="s1")
    assert load_compaction_archives(ctx) == []
    finalize_compaction_archive_registration(ctx, begun.record)
    loaded = load_compaction_archives(ctx)
    assert len(loaded) == 1
    assert loaded[0].id == begun.record.id
    assert loaded[0].pending is False


def test_register_concurrent_unique_ids(tmp_path: Path) -> None:
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            arch = tmp_path / f"context_{i}.jsonl"
            arch.touch()
            register_compaction_archive(ctx, arch, message_count=1, summary=str(i))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    recs = load_compaction_archives(ctx)
    assert len(recs) == 16
    assert len({r.id for r in recs}) == 16


def test_load_manifest_skips_malformed_lines(tmp_path: Path) -> None:
    ctx = tmp_path / "context.jsonl"
    ctx.touch()
    manifest = manifest_path_for_context(ctx)
    good = CompactionArchiveRecord(
        id="c001",
        archive_file="a.jsonl",
        created_at="2026-01-01T00:00:00+00:00",
        message_count=1,
        summary="ok",
    )
    manifest.write_text(
        good.model_dump_json() + "\n" + '{"broken": true' + "\n" + "not-json-at-all\n",
        encoding="utf-8",
    )
    recs = load_compaction_archives(ctx)
    assert len(recs) == 1
    assert recs[0].id == "c001"


def test_scrub_preserves_legitimate_json_type_pascalcase() -> None:
    from kimi_cli.loop.compaction_archive import scrub_text_for_archive_keywords

    a = scrub_text_for_archive_keywords('{"type":"PaymentIntent","status":"requires_action"}')
    assert "paymentintent" in a.lower()
    b = scrub_text_for_archive_keywords('{"type":"CustomerCreated","id":"evt_1"}')
    assert "customercreated" in b.lower()


def test_scrub_still_redacts_stream_envelope_type_values() -> None:
    from kimi_cli.loop.compaction_archive import scrub_text_for_archive_keywords

    s = scrub_text_for_archive_keywords('log {"type":"message"} after')
    assert '{"type":"message"}' not in s
    assert "log" in s and "after" in s
    t = scrub_text_for_archive_keywords('{"subtype":"thinking","delta":"noise"}')
    assert "subtype" not in t.lower()
    assert "delta" not in t.lower()


def test_extract_keywords_suppresses_stream_noise_preserves_path_and_cjk() -> None:
    text = (
        '{"subtype":"thinking","delta":"x"} \\ninternal '
        "/Users/proj/src/kimi_cli/loop/server.py 讨论压缩归档"
    )
    msgs = [Message(role="assistant", content=[TextPart(text=text)])]
    kws = extract_archive_keywords(msgs)
    lowered = [k.lower() if k.isascii() else k for k in kws]
    assert "subtype" not in lowered
    assert "ninternal" not in "".join(lowered)
    assert any("server.py" in k for k in kws) or any("kimi_cli" in k for k in kws)
    assert any("\u4e00" <= c <= "\u9fff" for k in kws for c in k)
