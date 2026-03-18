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
