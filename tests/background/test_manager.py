from __future__ import annotations

import time

import pytest

from kimi_cli.background import TaskRuntime, TaskSpec


def test_create_bash_task_persists_starting_state(runtime, monkeypatch):
    manager = runtime.background_tasks

    monkeypatch.setattr(manager, "_launch_worker", lambda task_dir: 4242)

    view = manager.create_bash_task(
        command="sleep 1",
        description="short sleep",
        timeout_s=10,
        tool_call_id="tool-1",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
    )

    assert view.spec.id.startswith("b")
    assert view.runtime.status == "starting"
    assert view.runtime.worker_pid == 4242


def test_create_bash_task_respects_max_running_tasks(runtime, monkeypatch):
    runtime.config.background.max_running_tasks = 1
    manager = runtime.background_tasks
    store = manager.store
    spec = TaskSpec(
        id="b1111999",
        kind="bash",
        session_id=runtime.session.id,
        description="already running",
        tool_call_id="tool-limit",
        command="sleep 10",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
    )
    store.create_task(spec)
    store.write_runtime(spec.id, TaskRuntime(status="running", updated_at=time.time()))

    monkeypatch.setattr(manager, "_launch_worker", lambda task_dir: 4242)

    with pytest.raises(RuntimeError, match="Too many background tasks"):
        manager.create_bash_task(
            command="sleep 1",
            description="short sleep",
            timeout_s=10,
            tool_call_id="tool-1b",
            shell_name="bash",
            shell_path="/bin/bash",
            cwd=str(runtime.session.work_dir),
        )


def test_recover_marks_stale_running_task_as_lost(runtime):
    manager = runtime.background_tasks
    store = manager.store
    spec = TaskSpec(
        id="b1111111",
        kind="bash",
        session_id=runtime.session.id,
        description="stale task",
        tool_call_id="tool-2",
        command="sleep 10",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
    )
    store.create_task(spec)
    store.write_runtime(
        spec.id,
        TaskRuntime(
            status="running",
            worker_pid=111,
            heartbeat_at=time.time() - 60,
            updated_at=time.time() - 60,
        ),
    )

    manager.recover()

    recovered = store.merged_view(spec.id)
    assert recovered.runtime.status == "lost"
    assert recovered.runtime.failure_reason == "Background worker heartbeat expired"


def test_publish_terminal_notifications_creates_notification(runtime):
    manager = runtime.background_tasks
    store = manager.store
    spec = TaskSpec(
        id="b2222222",
        kind="bash",
        session_id=runtime.session.id,
        description="completed task",
        tool_call_id="tool-3",
        command="echo done",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
    )
    store.create_task(spec)
    store.write_runtime(
        spec.id,
        TaskRuntime(
            status="completed",
            exit_code=0,
            finished_at=time.time(),
            updated_at=time.time(),
        ),
    )

    published = manager.publish_terminal_notifications(limit=4)
    assert len(published) == 1
    notification = runtime.notifications.store.merged_view(published[0])
    assert notification.event.source_id == spec.id
    assert notification.event.type == "task.completed"
    assert notification.event.payload["task_id"] == spec.id


def _write_completed_task(runtime, task_id: str, *, output: str) -> TaskSpec:
    store = runtime.background_tasks.store
    spec = TaskSpec(
        id=task_id,
        kind="bash",
        session_id=runtime.session.id,
        description="tail test",
        tool_call_id="tool-tail",
        command="echo hi",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
    )
    store.create_task(spec)
    store.output_path(task_id).write_text(output, encoding="utf-8")
    store.write_runtime(
        task_id,
        TaskRuntime(
            status="completed",
            exit_code=0,
            finished_at=time.time(),
            updated_at=time.time(),
        ),
    )
    return spec


def test_tail_output_returns_whole_lines(runtime):
    _write_completed_task(runtime, "btail0001", output="hello\nworld\n")
    result = runtime.background_tasks.tail_output("btail0001")
    assert "hello" in result
    assert "world" in result


def test_tail_output_overlong_line_returns_hint(runtime):
    huge = "z" * (runtime.config.background.read_max_bytes + 100) + "\n"
    _write_completed_task(runtime, "btail0002", output=huge)
    result = runtime.background_tasks.tail_output("btail0002")
    assert "Line too large" in result
    assert "TaskOutput" in result
    assert huge.strip() not in result


def test_tail_output_empty_returns_empty_string(runtime):
    _write_completed_task(runtime, "btail0003", output="")
    result = runtime.background_tasks.tail_output("btail0003")
    assert result == ""


def test_tail_output_no_trailing_newline(runtime):
    """File that does NOT end with a newline still returns all lines."""
    _write_completed_task(runtime, "btail0004", output="alpha\nbeta")
    result = runtime.background_tasks.tail_output("btail0004")
    assert "alpha" in result
    assert "beta" in result


def test_tail_output_respects_byte_budget(runtime):
    """Only the last lines fitting within the byte budget are returned."""
    line = "x" * 100 + "\n"  # 101 bytes per line
    line_count = 500  # ~50 KB total, larger than default read_max_bytes (30 000)
    _write_completed_task(runtime, "btail0005", output=line * line_count)
    result = runtime.background_tasks.tail_output("btail0005")
    # The result should contain some lines but NOT all 500
    returned_lines = result.strip().split("\n")
    assert 0 < len(returned_lines) < line_count
    # Each returned line should be the original content
    assert all(l == "x" * 100 for l in returned_lines)


def test_tail_output_single_newline_file(runtime):
    """A file with only a newline returns empty text."""
    _write_completed_task(runtime, "btail0006", output="\n")
    result = runtime.background_tasks.tail_output("btail0006")
    assert result == ""


def test_tail_read_output_lines_pagination_metadata(runtime):
    """Verify pagination metadata is correct for the tail path."""
    store = runtime.background_tasks.store
    _write_completed_task(runtime, "btail0007", output="a\nb\nc\nd\ne\n")
    chunk = store.read_output_lines("btail0007", None, 999_999, status="completed")
    assert chunk.start_line == 0
    assert chunk.end_line == 5
    assert chunk.has_before is False
    assert chunk.has_after is False
    assert chunk.next_offset is None
    assert "a\nb\nc\nd\ne" == chunk.text
