from __future__ import annotations

import time

import pytest

from kimi_cli.background import TaskRuntime, TaskSpec, TaskStatus
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
    assert "full_output_tool: ReadFile" in result.output
    assert "build line 1" in result.output


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
async def test_task_stop_blocks_in_plan_mode(runtime, task_stop_tool):
    runtime.session.state.plan_mode = True
    result = await task_stop_tool(task_stop_tool.params(task_id="b-noop"))
    assert result.is_error
    assert result.brief == "Blocked in plan mode"
