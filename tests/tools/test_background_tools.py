from __future__ import annotations

import json
import time

import pytest

from kimi_cli.background import TaskRuntime, TaskSpec, TaskStatus
from kimi_cli.tools.background import (
    TASK_OUTPUT_PREVIEW_BYTES,
    TaskOutputMode,
    TaskWriteParams,
)
from kimi_cli.tools.shell import Params


def _write_task(runtime, task_id: str, *, status: TaskStatus, output: str = ""):
    store = runtime.background_tasks.store
    spec = TaskSpec(
        id=task_id,
        kind="bash",
        session_id=runtime.session.id,
        description="background build",
        tool_call_id="tool-6",
        command="make build",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
    )
    store.create_task(spec)
    store.output_path(task_id).write_text(output, encoding="utf-8")
    runtime_state = TaskRuntime(status=status, updated_at=time.time())
    if status in {"completed", "failed", "killed", "lost"}:
        runtime_state.finished_at = time.time()
        runtime_state.exit_code = 0 if status == "completed" else 1
    store.write_runtime(task_id, runtime_state)
    return spec


@pytest.mark.asyncio
async def test_shell_background_starts_task_without_live_notifications(
    shell_tool, runtime, monkeypatch
):
    monkeypatch.setattr(runtime.background_tasks, "_launch_worker", lambda task_dir: 9898)

    result = await shell_tool(
        Params(
            command="sleep 1",
            timeout=10,
            run_in_background=True,
            description="sleep task",
        )
    )

    assert not result.is_error
    assert "task_id:" in result.output
    assert "status: starting" in result.output
    assert "automatic_notification: false" in result.output
    assert "timeout_s:" in result.output
    assert "next_steps:" in result.output
    assert "TaskOutput" in result.output


@pytest.mark.asyncio
async def test_shell_background_starts_task_with_live_notifications(
    shell_tool, runtime, monkeypatch
):
    runtime.background_notification_targets = ("llm", "shell")
    monkeypatch.setattr(runtime.background_tasks, "_launch_worker", lambda task_dir: 9898)

    result = await shell_tool(
        Params(
            command="sleep 1",
            timeout=10,
            run_in_background=True,
            description="sleep task",
        )
    )

    assert not result.is_error
    assert "automatic_notification: true" in result.output
    assert "automatically notified in this session" in result.output


@pytest.mark.asyncio
async def test_shell_background_requires_description(shell_tool):
    with pytest.raises(ValueError, match="description"):
        Params(command="sleep 1", timeout=10, run_in_background=True)


@pytest.mark.asyncio
async def test_task_output_returns_completed_output(runtime, task_output_tool):
    spec = _write_task(
        runtime,
        "b5555555",
        status="completed",
        output="build line 1\nbuild line 2\n",
    )

    result = await task_output_tool(task_output_tool.params(task_id=spec.id, block=True, timeout=1))

    output_path = runtime.background_tasks.store.output_path(spec.id).resolve()
    assert not result.is_error
    assert "retrieval_status: success" in result.output
    assert "status: completed" in result.output
    assert f"output_path: {output_path}" in result.output
    assert "output_truncated: false" in result.output
    assert "output_has_before: false" in result.output
    assert "output_has_after: false" in result.output
    assert "output_preview_start_line: 0" in result.output
    assert "output_preview_end_line: 2" in result.output
    assert "full_output_tool: ReadFile" in result.output
    assert "build line 1" in result.output


@pytest.mark.asyncio
async def test_task_output_returns_completed_output_with_line_offset(runtime, task_output_tool):
    lines = [f"line {i}\n" for i in range(10)]
    spec = _write_task(
        runtime,
        "b5555556",
        status="completed",
        output="".join(lines),
    )

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, offset=5)
    )

    assert not result.is_error
    assert "output_has_before: true" in result.output
    assert "output_has_after: false" in result.output
    assert "output_preview_start_line: 5" in result.output
    assert "output_preview_end_line: 10" in result.output
    assert "line 5" in result.output
    assert "line 9" in result.output
    assert "line 4" not in result.output


@pytest.mark.asyncio
async def test_task_output_truncates_many_lines(runtime, task_output_tool):
    # Create output that exceeds 32 KiB
    big_line = "x" * 200 + "\n"
    line_count = (TASK_OUTPUT_PREVIEW_BYTES // len(big_line.encode("utf-8"))) + 50
    spec = _write_task(
        runtime,
        "b5555557",
        status="completed",
        output=big_line * line_count,
    )

    result = await task_output_tool(task_output_tool.params(task_id=spec.id, block=True, timeout=1))

    assert not result.is_error
    assert "output_truncated: true" in result.output
    assert "output_has_before: true" in result.output
    assert "output_has_after: false" in result.output
    assert "[Truncated" in result.output
    # Wording should use inclusive range and line count, not half-open end
    import re

    m = re.search(r"\[Truncated — showing (\d+) lines \((\d+)–(\d+)\)", result.output)
    assert m, f"truncated message wording not found in output: {result.output[:500]}"
    n_lines = int(m.group(1))
    first = int(m.group(2))
    last = int(m.group(3))
    assert n_lines == last - first + 1, "line count should equal inclusive range size"


@pytest.mark.asyncio
async def test_task_output_line_too_large(runtime, task_output_tool):
    # Single line exceeding 32 KiB
    huge_line = "x" * (TASK_OUTPUT_PREVIEW_BYTES + 100) + "\n"
    spec = _write_task(
        runtime,
        "b5555558",
        status="completed",
        output=huge_line,
    )

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, offset=0)
    )

    output_path = runtime.background_tasks.store.output_path(spec.id).resolve()
    assert not result.is_error
    assert "Line too large" in result.output
    assert "ReadFile" in result.output
    assert str(output_path) in result.output


@pytest.mark.asyncio
async def test_task_output_line_too_large_tail_mode(runtime, task_output_tool):
    """Overlong single-line triggers line_too_large in tail mode (offset=None)."""
    huge_line = "y" * (TASK_OUTPUT_PREVIEW_BYTES + 100) + "\n"
    spec = _write_task(
        runtime,
        "b555555a",
        status="completed",
        output=huge_line,
    )

    # Default offset=None → tail mode
    result = await task_output_tool(task_output_tool.params(task_id=spec.id, block=True, timeout=1))

    output_path = runtime.background_tasks.store.output_path(spec.id).resolve()
    assert not result.is_error
    assert "Line too large" in result.output
    assert "ReadFile" in result.output
    assert str(output_path) in result.output


@pytest.mark.asyncio
async def test_task_output_empty_output(runtime, task_output_tool):
    """Empty output file returns no lines and no output placeholder."""
    spec = _write_task(
        runtime,
        "b555555b",
        status="completed",
        output="",
    )

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, offset=0)
    )

    assert not result.is_error
    assert "output_has_before: false" in result.output
    assert "output_has_after: false" in result.output
    assert "[no output available]" in result.output


@pytest.mark.asyncio
async def test_task_output_offset_past_end(runtime, task_output_tool):
    """Offset beyond the last line returns an empty preview without error."""
    spec = _write_task(
        runtime,
        "b555555c",
        status="completed",
        output="only line\n",
    )

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, offset=999)
    )

    assert not result.is_error
    assert "output_has_before: true" in result.output
    assert "output_has_after: false" in result.output
    assert "output_preview_start_line: 1" in result.output
    assert "output_preview_end_line: 1" in result.output
    assert "[no output available]" in result.output


@pytest.mark.asyncio
async def test_task_list_returns_active_tasks(runtime, task_list_tool):
    active_spec = _write_task(runtime, "b4444444", status="running", output="still going\n")
    _write_task(runtime, "b4444445", status="completed", output="done\n")

    result = await task_list_tool(task_list_tool.params(active_only=True, limit=20))

    assert not result.is_error
    assert "active_background_tasks: 1" in result.output
    assert active_spec.id in result.output
    assert "b4444445" not in result.output


@pytest.mark.asyncio
async def test_task_output_returns_not_ready_for_running_task(runtime, task_output_tool):
    spec = _write_task(runtime, "b6666666", status="running", output="still working\n")

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=False, timeout=0)
    )

    assert not result.is_error
    assert "retrieval_status: not_ready" in result.output
    assert "status: running" in result.output
    assert "still working" in result.output


@pytest.mark.asyncio
async def test_task_output_suppresses_llm_notification_for_terminal_task(runtime, task_output_tool):
    """TaskOutput on a terminal task should ack the pending LLM notification."""
    spec = _write_task(runtime, "b7777777", status="completed", output="done\n")

    # Publish the notification first (simulates reconcile running before TaskOutput).
    published = runtime.background_tasks.publish_terminal_notifications(limit=4)
    assert len(published) == 1

    await task_output_tool(task_output_tool.params(task_id=spec.id, block=True, timeout=1))

    nview = runtime.notifications.store.merged_view(published[0])
    assert nview.delivery.sinks["llm"].status == "acked"


@pytest.mark.asyncio
async def test_task_output_does_not_suppress_for_running_task(runtime, task_output_tool):
    """TaskOutput on a still-running task must NOT mark it as observed."""
    spec = _write_task(runtime, "b8888888", status="running", output="wip\n")

    await task_output_tool(task_output_tool.params(task_id=spec.id, block=False, timeout=0))

    assert spec.id not in runtime.background_tasks._observed_terminal_ids


# ---------------------------------------------------------------------------
# TaskWrite tests
# ---------------------------------------------------------------------------


def _write_interactive_task(
    runtime,
    task_id: str,
    *,
    status: TaskStatus = "running",
    stdin_ready: bool = True,
    interactive: bool = True,
):
    store = runtime.background_tasks.store
    spec = TaskSpec(
        id=task_id,
        kind="bash",
        session_id=runtime.session.id,
        description="interactive task",
        tool_call_id="tool-99",
        command="cat",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
        interactive=interactive,
    )
    store.create_task(spec)
    rt = TaskRuntime(status=status, stdin_ready=stdin_ready, updated_at=time.time())
    if status in {"completed", "failed", "killed", "lost"}:
        rt.finished_at = time.time()
        rt.exit_code = 0 if status == "completed" else 1
    store.write_runtime(task_id, rt)
    queue_dir = store.task_dir(task_id) / "stdin_queue"
    queue_dir.mkdir(exist_ok=True)
    return spec


@pytest.mark.asyncio
async def test_task_write_queues_message(runtime, task_write_tool):
    spec = _write_interactive_task(runtime, "bw000001")
    result = await task_write_tool(TaskWriteParams(task_id=spec.id, input="hello"))

    assert not result.is_error
    assert "bytes_queued:" in result.output
    queue_dir = runtime.background_tasks.store.task_dir(spec.id) / "stdin_queue"
    msgs = list(queue_dir.glob("*.msg"))
    assert len(msgs) == 1
    assert msgs[0].read_text() == "hello\n"


@pytest.mark.asyncio
async def test_task_write_no_newline(runtime, task_write_tool):
    spec = _write_interactive_task(runtime, "bw000002")
    result = await task_write_tool(
        TaskWriteParams(task_id=spec.id, input="hello", append_newline=False)
    )

    assert not result.is_error
    queue_dir = runtime.background_tasks.store.task_dir(spec.id) / "stdin_queue"
    msgs = list(queue_dir.glob("*.msg"))
    assert len(msgs) == 1
    assert msgs[0].read_text() == "hello"


@pytest.mark.asyncio
async def test_task_write_rejects_non_interactive(runtime, task_write_tool):
    spec = _write_interactive_task(runtime, "bw000003", interactive=False)
    result = await task_write_tool(TaskWriteParams(task_id=spec.id, input="hello"))

    assert result.is_error
    assert "not interactive" in result.message.lower()


@pytest.mark.asyncio
async def test_task_write_rejects_terminal_task(runtime, task_write_tool):
    spec = _write_interactive_task(runtime, "bw000004", status="completed")
    result = await task_write_tool(TaskWriteParams(task_id=spec.id, input="hello"))

    assert result.is_error
    assert "finished" in result.message.lower()


@pytest.mark.asyncio
async def test_task_write_rejects_stdin_not_ready(runtime, task_write_tool):
    spec = _write_interactive_task(runtime, "bw000005", stdin_ready=False)
    result = await task_write_tool(TaskWriteParams(task_id=spec.id, input="hello"))

    assert result.is_error
    assert "not ready" in result.message.lower()


@pytest.mark.asyncio
async def test_task_write_not_found(runtime, task_write_tool):
    result = await task_write_tool(TaskWriteParams(task_id="bw-nonexist", input="hello"))

    assert result.is_error
    assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_shell_interactive_starts_task(shell_tool, runtime, monkeypatch):
    monkeypatch.setattr(runtime.background_tasks, "_launch_worker", lambda task_dir: 9898)
    result = await shell_tool(
        Params(
            command="cat",
            timeout=3600,
            run_in_background=True,
            interactive=True,
            description="interactive cat",
        )
    )
    assert not result.is_error
    assert "TaskWrite" in result.output


# ---------------------------------------------------------------------------
# TaskOutput rendering modes
# ---------------------------------------------------------------------------


def _structured_log(n_iters: int = 3, big_blob_size: int = 4000) -> str:
    big = "x" * big_blob_size
    lines: list[str] = [
        json.dumps({"type": "system", "subtype": "init", "tools": [big]}),
    ]
    for i in range(n_iters):
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": f"step {i}"}]},
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": [{"type": "tool_result", "content": big}]},
                }
            )
        )
    lines.append(json.dumps({"type": "result", "result": "Final answer."}))
    return "\n".join(lines) + "\n"


@pytest.mark.asyncio
async def test_task_output_default_summarizes_structured_log(runtime, task_output_tool):
    output = _structured_log()
    spec = _write_task(runtime, "b9000001", status="completed", output=output)

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, offset=0)
    )

    assert not result.is_error
    # Big tool_result blobs must not appear.
    assert "x" * 1000 not in result.output
    # Final answer should be preserved.
    assert "Final answer." in result.output
    # At least one step text is preserved (oldest may be dropped by chunk window).
    assert any(f"step {i}" in result.output for i in range(5))
    # Detected format and render mode metadata is exposed.
    assert "render_mode: summary" in result.output
    assert "output_format: claude_stream_json" in result.output
    # Metadata still references raw line numbers.
    assert "output_path:" in result.output


@pytest.mark.asyncio
async def test_task_output_mode_raw_preserves_legacy_behavior(runtime, task_output_tool):
    spec = _write_task(
        runtime,
        "b9000002",
        status="completed",
        output="build line 1\nbuild line 2\n",
    )

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, mode=TaskOutputMode.RAW)
    )

    assert not result.is_error
    assert "render_mode: raw" in result.output
    assert "build line 1" in result.output
    assert "build line 2" in result.output


@pytest.mark.asyncio
async def test_task_output_mode_none_omits_payload(runtime, task_output_tool):
    spec = _write_task(runtime, "b9000003", status="completed", output="secret payload\n")

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, mode=TaskOutputMode.NONE)
    )

    assert not result.is_error
    assert "render_mode: none" in result.output
    assert "[output]" not in result.output
    assert "secret payload" not in result.output
    # Path/metadata is still present.
    assert "output_path:" in result.output


@pytest.mark.asyncio
async def test_task_output_max_bytes_caps_summary_payload(runtime, task_output_tool):
    output = _structured_log(n_iters=20)
    spec = _write_task(runtime, "b9000004", status="completed", output=output)

    result = await task_output_tool(
        task_output_tool.params(
            task_id=spec.id, block=True, timeout=1, mode=TaskOutputMode.SUMMARY, max_bytes=512
        )
    )

    # Extract the [output] block and ensure the *full* payload (including
    # the summary header) fits within max_bytes — no allowance.
    payload = result.output.split("[output]\n", 1)[1]
    assert len(payload.encode("utf-8")) <= 512


@pytest.mark.asyncio
async def test_task_output_max_bytes_caps_raw_payload(runtime, task_output_tool):
    # Generate plenty of raw output so the cap actually triggers.
    big = "abcdefghij" * 200  # 2000 bytes per line
    lines = [f"{big}\n" for _ in range(20)]
    spec = _write_task(runtime, "b9000040", status="completed", output="".join(lines))

    result = await task_output_tool(
        task_output_tool.params(
            task_id=spec.id, block=True, timeout=1, mode=TaskOutputMode.RAW, max_bytes=1024
        )
    )

    payload = result.output.split("[output]\n", 1)[1]
    # Full [output] payload (including the truncation header) must fit
    # within max_bytes.
    assert len(payload.encode("utf-8")) <= 1024
    assert "Truncated — showing" in payload


@pytest.mark.asyncio
async def test_task_output_default_auto_plain_text_bounded(runtime, task_output_tool):
    # Build a large plain-text log well over 4 KiB (and over the default
    # 32 KiB chunk limit isn't required — the cap is on the rendered
    # [output] payload, not on the raw chunk).
    chunk_line = "plain log line with some payload " * 4  # ~128 bytes per line
    plain = "".join(f"{i:05d} {chunk_line}\n" for i in range(400))  # ~50 KiB
    spec = _write_task(runtime, "b9000041", status="completed", output=plain)

    # Default mode=auto, default max_bytes=4096.
    result = await task_output_tool(task_output_tool.params(task_id=spec.id, block=True, timeout=1))

    payload = result.output.split("[output]\n", 1)[1]
    assert len(payload.encode("utf-8")) <= 4096
    # And much smaller than the 32 KiB raw preview that earlier code paths
    # could otherwise dump.
    assert len(payload.encode("utf-8")) < 32 * 1024 // 2


@pytest.mark.asyncio
async def test_task_output_tail_lines_header_reports_subwindow(runtime, task_output_tool):
    # 20 raw lines, tail to last 3 — header must say "3 lines (17–19)",
    # NOT the full 20-line range.
    raw = "".join(f"line {i}\n" for i in range(20))
    spec = _write_task(runtime, "b9000042", status="completed", output=raw)

    result = await task_output_tool(
        task_output_tool.params(
            task_id=spec.id, block=True, timeout=1, mode=TaskOutputMode.RAW, tail_lines=3
        )
    )

    payload = result.output.split("[output]\n", 1)[1]
    assert "Truncated — showing 3 lines (17–19)" in payload
    # Raw pagination metadata should still reference full chunk range.
    assert "output_preview_start_line: 0" in result.output
    assert "output_preview_end_line: 20" in result.output


@pytest.mark.asyncio
async def test_task_output_structured_no_projectable_emits_marker(runtime, task_output_tool):
    # Codex-detected NDJSON with only ignored event types (deltas / outputs).
    raw = (
        "\n".join(
            [
                json.dumps({"type": "agent_message_delta", "delta": "x" * 5000}),
                json.dumps({"type": "tool_call_output", "output": "y" * 5000}),
                json.dumps({"type": "response.output_text.delta", "delta": "z" * 5000}),
            ]
        )
        + "\n"
    )
    spec = _write_task(runtime, "b9000043", status="completed", output=raw)

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, mode=TaskOutputMode.SUMMARY)
    )

    payload = result.output.split("[output]\n", 1)[1]
    # No raw JSON / huge tool output payloads leak through.
    assert "x" * 200 not in payload
    assert "y" * 200 not in payload
    assert "z" * 200 not in payload
    # A structured "no projectable events" marker is emitted instead.
    assert "no projectable events" in payload


@pytest.mark.asyncio
async def test_task_output_tail_lines_in_raw_mode(runtime, task_output_tool):
    lines = [f"line {i}\n" for i in range(20)]
    spec = _write_task(runtime, "b9000005", status="completed", output="".join(lines))

    result = await task_output_tool(
        task_output_tool.params(
            task_id=spec.id, block=True, timeout=1, mode=TaskOutputMode.RAW, tail_lines=3
        )
    )

    assert not result.is_error
    assert "line 19" in result.output
    assert "line 18" in result.output
    assert "line 17" in result.output
    assert "line 0" not in result.output
    assert "line 16" not in result.output


@pytest.mark.asyncio
async def test_task_output_summary_preserves_raw_offset_pagination(runtime, task_output_tool):
    # Generate enough Kimi NDJSON lines to exceed a small offset.
    raw_lines = []
    for i in range(20):
        raw_lines.append(json.dumps({"role": "assistant", "content": f"reply {i}"}))
    output = "\n".join(raw_lines) + "\n"

    spec = _write_task(runtime, "b9000006", status="completed", output=output)

    result = await task_output_tool(
        task_output_tool.params(task_id=spec.id, block=True, timeout=1, offset=0)
    )

    assert not result.is_error
    # output_next_offset should still index raw line numbers (not summary events).
    assert "output_preview_start_line: 0" in result.output
    # Either we read every line, or there's a forward offset.
    if "output_has_after: true" in result.output:
        assert "output_next_offset:" in result.output


@pytest.mark.asyncio
async def test_task_output_line_too_large_in_summary_mode(runtime, task_output_tool):
    huge_line = "z" * (TASK_OUTPUT_PREVIEW_BYTES + 100) + "\n"
    spec = _write_task(runtime, "b9000007", status="completed", output=huge_line)

    result = await task_output_tool(
        task_output_tool.params(
            task_id=spec.id, block=True, timeout=1, offset=0, mode=TaskOutputMode.SUMMARY
        )
    )

    assert not result.is_error
    assert "Line too large" in result.output
    assert "ReadFile" in result.output


def test_shell_interactive_requires_background_marker():
    with pytest.raises(ValueError, match="interactive.*requires.*run_in_background"):
        Params(command="cat", timeout=10, run_in_background=False, interactive=True)
