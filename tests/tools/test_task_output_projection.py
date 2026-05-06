from __future__ import annotations

import json

from kimi_cli.tools.background.projection import (
    OutputFormat,
    detect_format,
    project,
    render_projection,
)


def test_detect_format_plain_text():
    assert detect_format("hello\nworld\n") is OutputFormat.PLAIN_TEXT


def test_detect_format_claude_stream_json():
    sample = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}
            ),
        ]
    )
    assert detect_format(sample) is OutputFormat.CLAUDE_STREAM_JSON


def test_detect_format_codex_jsonl():
    sample = json.dumps({"type": "agent_message", "message": "done"})
    assert detect_format(sample) is OutputFormat.CODEX_JSONL


def test_detect_format_kimi_ndjson():
    sample = json.dumps({"role": "assistant", "content": "hi"})
    assert detect_format(sample) is OutputFormat.KIMI_AGENT_NDJSON


def test_project_claude_keeps_assistant_text_drops_noise():
    big = "x" * 10000
    sample_lines = [
        json.dumps({"type": "system", "session_id": "abc", "tools": [big]}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hello world"}]},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": big}]},
            }
        ),
        json.dumps({"type": "result", "result": "Final answer."}),
    ]
    text = "\n".join(sample_lines)
    proj = project(text, format=OutputFormat.CLAUDE_STREAM_JSON, max_bytes=4096)
    rendered = render_projection(proj, max_bytes=4096)

    assert big not in rendered
    assert "Hello world" in rendered
    assert "Bash" in rendered
    assert "Final answer." in rendered
    assert len(rendered.encode("utf-8")) <= 4096


def test_project_codex_drops_deltas():
    text = "\n".join(
        [
            json.dumps({"type": "agent_message_delta", "delta": "He"}),
            json.dumps({"type": "agent_message_delta", "delta": "llo"}),
            json.dumps({"type": "agent_message", "message": "Hello"}),
            json.dumps({"type": "task_complete"}),
        ]
    )
    proj = project(text, format=OutputFormat.CODEX_JSONL)
    rendered = render_projection(proj)
    assert "Hello" in rendered
    assert "completed" in rendered
    assert "delta" not in rendered.lower()


def test_project_kimi_summarizes_tool_calls_and_drops_tool_results():
    big = "y" * 8000
    text = "\n".join(
        [
            json.dumps(
                {
                    "role": "assistant",
                    "content": "Working on it.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "Shell",
                                "arguments": json.dumps({"command": "ls"}),
                            }
                        }
                    ],
                }
            ),
            json.dumps({"role": "tool", "content": big}),
            json.dumps({"role": "assistant", "content": "Done."}),
        ]
    )
    proj = project(text, format=OutputFormat.KIMI_AGENT_NDJSON, max_bytes=4096)
    rendered = render_projection(proj, max_bytes=4096)
    assert "Working on it." in rendered
    assert "Shell" in rendered
    assert "Done." in rendered
    assert big not in rendered


def test_project_plain_text_tail_digest_bounded():
    text = "".join(f"line {i}\n" for i in range(1000))
    proj = project(text, format=OutputFormat.PLAIN_TEXT, max_bytes=200)
    rendered = render_projection(proj, max_bytes=200)
    assert "line 999" in rendered
    assert "line 0" not in rendered
    assert len(rendered.encode("utf-8")) <= 200


def test_render_projection_truncates_oldest_when_over_budget():
    text = "\n".join(json.dumps({"role": "assistant", "content": f"msg {i}"}) for i in range(50))
    proj = project(text, format=OutputFormat.KIMI_AGENT_NDJSON, max_events=50)
    rendered = render_projection(proj, max_bytes=200)
    assert len(rendered.encode("utf-8")) <= 200 + len("[…older summarized events dropped…]\n")
    assert "msg 49" in rendered
