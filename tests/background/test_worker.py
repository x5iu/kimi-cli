from __future__ import annotations

from kimi_cli.background.models import TaskControl, TaskRuntime
from kimi_cli.background.worker import finalize_task_runtime


def test_finalize_task_runtime_marks_completed_when_kill_arrives_after_exit() -> None:
    runtime = TaskRuntime(status="running")
    control = TaskControl(kill_requested_at=11.0, kill_reason="Stopped by user")

    result = finalize_task_runtime(
        runtime,
        control=control,
        returncode=0,
        finished_at=12.0,
        process_exited_at=10.0,
        timed_out=False,
        timeout_reason=None,
    )

    assert result.status == "completed"
    assert result.exit_code == 0
    assert result.failure_reason is None


def test_finalize_task_runtime_marks_killed_when_kill_precedes_exit() -> None:
    runtime = TaskRuntime(status="running")
    control = TaskControl(kill_requested_at=9.0, kill_reason="Stopped by user")

    result = finalize_task_runtime(
        runtime,
        control=control,
        returncode=0,
        finished_at=12.0,
        process_exited_at=10.0,
        timed_out=False,
        timeout_reason=None,
    )

    assert result.status == "killed"
    assert result.interrupted is True
    assert result.failure_reason == "Stopped by user"
