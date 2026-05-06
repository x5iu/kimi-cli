from __future__ import annotations

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.compaction import SimpleCompaction
from llmkit.message import Message, ToolCall

SAMPLE_TASK_OUTPUT_TEXT = """retrieval_status: success
task_id: bg-123
kind: shell
status: completed
description: list files
command: ls -la
interrupted: false
timed_out: false
terminal_reason: completed
exit_code: 0
elapsed_s: 0.5

output_path: /tmp/bg-123.log
output_preview_start_line: 1
output_preview_end_line: 10
output_has_before: false
output_has_after: false
output_truncated: false
output_next_offset: 11

full_output_available: true
full_output_tool: ReadFile
full_output_hint: Use ReadFile(...) ...
render_mode: raw
output_format: plain_text

[output]
file_a
file_b
secret_payload_should_be_redacted
"""


def _task_output_call(call_id: str = "tc_task1", name: str = "TaskOutput") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(name=name, arguments='{"task_id":"bg-123"}'),
    )


def _shell_call(call_id: str = "tc_shell1") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(name="Shell", arguments='{"command":"ls"}'),
    )


def _compact_blob(msg: Message) -> str:
    return "".join(p.text for p in msg.content if isinstance(p, TextPart))


def test_redacts_task_output_in_compact_message():
    messages = [
        Message(role="user", content=[TextPart(text="run task")]),
        Message(
            role="assistant",
            content=[TextPart(text="checking output")],
            tool_calls=[_task_output_call()],
        ),
        Message(
            role="tool",
            content=[TextPart(text=SAMPLE_TASK_OUTPUT_TEXT)],
            tool_call_id="tc_task1",
        ),
        Message(role="user", content=[TextPart(text="middle question")]),
        Message(role="assistant", content=[TextPart(text="middle reply")]),
        Message(role="user", content=[TextPart(text="latest")]),
        Message(role="assistant", content=[TextPart(text="latest reply")]),
    ]

    prep = SimpleCompaction(max_preserved_messages=2).prepare(messages)
    assert prep.compact_message is not None
    blob = _compact_blob(prep.compact_message)

    assert "task_id: bg-123" in blob
    assert "retrieval_status: success" in blob
    assert "output_path: /tmp/bg-123.log" in blob
    assert "render_mode: raw" in blob
    assert "output_format: plain_text" in blob
    assert "secret_payload_should_be_redacted" not in blob
    assert "[output]" not in blob
    assert "TaskOutput payload redacted during compaction" in blob
    assert 'TaskOutput(task_id="bg-123")' in blob
    assert 'ReadFile(path="/tmp/bg-123.log")' in blob


def test_does_not_redact_non_task_output_tool_results():
    messages = [
        Message(role="user", content=[TextPart(text="old")]),
        Message(
            role="assistant",
            content=[TextPart(text="running shell")],
            tool_calls=[_shell_call()],
        ),
        Message(
            role="tool",
            content=[TextPart(text="some shell output\nline2")],
            tool_call_id="tc_shell1",
        ),
        Message(role="user", content=[TextPart(text="middle")]),
        Message(role="assistant", content=[TextPart(text="reply")]),
        Message(role="user", content=[TextPart(text="latest")]),
        Message(role="assistant", content=[TextPart(text="latest reply")]),
    ]

    prep = SimpleCompaction(max_preserved_messages=2).prepare(messages)
    assert prep.compact_message is not None
    blob = _compact_blob(prep.compact_message)
    assert "some shell output" in blob
    assert "TaskOutput payload redacted" not in blob


def test_does_not_redact_preserved_task_output_tail():
    messages = [
        Message(role="user", content=[TextPart(text="first")]),
        Message(role="assistant", content=[TextPart(text="reply 1")]),
        Message(role="user", content=[TextPart(text="latest")]),
        Message(
            role="assistant",
            content=[TextPart(text="checking")],
            tool_calls=[_task_output_call("tc_keep")],
        ),
        Message(
            role="tool",
            content=[TextPart(text=SAMPLE_TASK_OUTPUT_TEXT)],
            tool_call_id="tc_keep",
        ),
        Message(role="assistant", content=[TextPart(text="done")]),
    ]

    prep = SimpleCompaction(max_preserved_messages=1).prepare(messages)
    assert prep.compact_message is not None

    preserved_blob = "".join(
        p.text for m in prep.to_preserve for p in m.content if isinstance(p, TextPart)
    )
    assert "secret_payload_should_be_redacted" in preserved_blob
    assert "[output]" in preserved_blob


def test_redact_disabled_preserves_raw_payload_in_compact_message():
    messages = [
        Message(role="user", content=[TextPart(text="run task")]),
        Message(
            role="assistant",
            content=[TextPart(text="checking output")],
            tool_calls=[_task_output_call()],
        ),
        Message(
            role="tool",
            content=[TextPart(text=SAMPLE_TASK_OUTPUT_TEXT)],
            tool_call_id="tc_task1",
        ),
        Message(role="user", content=[TextPart(text="middle")]),
        Message(role="assistant", content=[TextPart(text="middle reply")]),
        Message(role="user", content=[TextPart(text="latest")]),
        Message(role="assistant", content=[TextPart(text="latest reply")]),
    ]

    prep = SimpleCompaction(max_preserved_messages=2, redact_task_output_results=False).prepare(
        messages
    )
    assert prep.compact_message is not None
    blob = _compact_blob(prep.compact_message)
    assert "secret_payload_should_be_redacted" in blob
    assert "[output]" in blob
    assert "TaskOutput payload redacted" not in blob


def test_redacts_unknown_or_empty_task_output_result():
    messages = [
        Message(role="user", content=[TextPart(text="run task")]),
        Message(
            role="assistant",
            content=[TextPart(text="checking")],
            tool_calls=[_task_output_call("tc_empty")],
        ),
        Message(role="tool", content=[TextPart(text="")], tool_call_id="tc_empty"),
        Message(role="user", content=[TextPart(text="middle")]),
        Message(role="assistant", content=[TextPart(text="middle reply")]),
        Message(role="user", content=[TextPart(text="latest")]),
        Message(role="assistant", content=[TextPart(text="latest reply")]),
    ]

    prep = SimpleCompaction(max_preserved_messages=2).prepare(messages)
    assert prep.compact_message is not None
    blob = _compact_blob(prep.compact_message)
    assert "TaskOutput payload redacted during compaction" in blob
    assert "call_id=tc_empty" in blob
