import time
from pathlib import Path
from typing import override

from kosong.tooling import CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.background import TaskOutputLineChunk, TaskStatus, TaskView, format_task, format_task_list, list_task_views
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.tools.display import BackgroundTaskDisplayBlock
from kimi_cli.tools.utils import ToolRejectedError, load_desc

TASK_OUTPUT_PREVIEW_BYTES = 32 << 10
TASK_OUTPUT_READ_HINT_LINES = 300




def _task_display(runtime: Runtime, task_id: str) -> BackgroundTaskDisplayBlock:
    view = runtime.background_tasks.store.merged_view(task_id)
    return BackgroundTaskDisplayBlock(
        task_id=view.spec.id,
        kind=view.spec.kind,
        status=view.runtime.status,
        description=view.spec.description,
    )


def _format_task_output(
    view: TaskView,
    *,
    retrieval_status: str,
    chunk: TaskOutputLineChunk,
    full_output_available: bool,
) -> str:
    terminal_reason = "timed_out" if view.runtime.timed_out else view.runtime.status
    lines = [
        f"retrieval_status: {retrieval_status}",
        f"task_id: {view.spec.id}",
        f"kind: {view.spec.kind}",
        f"status: {view.runtime.status}",
        f"description: {view.spec.description}",
    ]
    if view.spec.command:
        lines.append(f"command: {view.spec.command}")
    lines.extend(
        [
            f"interrupted: {str(view.runtime.interrupted).lower()}",
            f"timed_out: {str(view.runtime.timed_out).lower()}",
            f"terminal_reason: {terminal_reason}",
        ]
    )
    if view.runtime.exit_code is not None:
        lines.append(f"exit_code: {view.runtime.exit_code}")
    if view.runtime.failure_reason:
        lines.append(f"reason: {view.runtime.failure_reason}")
    if view.runtime.started_at:
        end = view.runtime.finished_at or time.time()
        elapsed = end - view.runtime.started_at
        lines.append(f"elapsed_s: {elapsed:.1f}")
    full_output_hint = (
        (
            "full_output_hint: "
            f'Use ReadFile(path="{chunk.output_path}", line_offset=1, '
            f"n_lines={TASK_OUTPUT_READ_HINT_LINES}) to inspect the full log. "
            "Increase line_offset to continue paging through the file."
        )
        if full_output_available
        else "full_output_hint: No output file is currently available for this task."
    )
    output_truncated = chunk.has_before or chunk.has_after
    lines.extend(
        [
            "",
            f"output_path: {chunk.output_path}",
            f"output_preview_start_line: {chunk.start_line}",
            f"output_preview_end_line: {chunk.end_line}",
            f"output_has_before: {str(chunk.has_before).lower()}",
            f"output_has_after: {str(chunk.has_after).lower()}",
            f"output_truncated: {str(output_truncated).lower()}",
        ]
    )
    if chunk.next_offset is not None:
        lines.append(f"output_next_offset: {chunk.next_offset}")
    lines.extend(
        [
            "",
            f"full_output_available: {str(full_output_available).lower()}",
            "full_output_tool: ReadFile",
            full_output_hint,
        ]
    )
    if chunk.line_too_large:
        rendered_output = (
            f"[Line too large — line {chunk.start_line} exceeds the "
            f"{TASK_OUTPUT_PREVIEW_BYTES // 1024} KiB preview limit. "
            f'Use ReadFile(path="{chunk.output_path}") to inspect the output file directly.]'
        )
    elif not chunk.text:
        rendered_output = "[no output available]"
    elif output_truncated:
        n_lines = chunk.end_line - chunk.start_line
        last_line = chunk.end_line - 1
        rendered_output = (
            f"[Truncated — showing {n_lines} lines ({chunk.start_line}–{last_line})"
            f". Full output: {chunk.output_path}]\n\n{chunk.text}"
        )
    else:
        rendered_output = chunk.text
    return "\n".join(
        lines
        + [
            "",
            "[output]",
            rendered_output,
        ]
    )


class TaskOutputParams(BaseModel):
    task_id: str = Field(description="The background task ID to inspect.")
    block: bool = Field(
        default=True,
        description="Whether to wait for the task to finish before returning.",
    )
    timeout: int = Field(
        default=30,
        ge=0,
        le=3600,
        description="Maximum number of seconds to wait when block=true.",
    )
    offset: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Line offset (0-based) to start reading output from. "
            "If not set, reads the last lines that fit within ~32 KiB (tail). "
            "Set to 0 to read from the beginning."
        ),
    )


class TaskStopParams(BaseModel):
    task_id: str = Field(description="The background task ID to stop.")
    reason: str = Field(
        default="Stopped by TaskStop",
        description="Short reason recorded when the task is stopped.",
    )


class TaskListParams(BaseModel):
    active_only: bool = Field(
        default=True,
        description="Whether to list only non-terminal background tasks.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of tasks to return.",
    )


class TaskList(CallableTool2[TaskListParams]):
    name: str = "TaskList"
    description: str = load_desc(Path(__file__).parent / "list.md")
    params: type[TaskListParams] = TaskListParams

    def __init__(self, runtime: Runtime):
        super().__init__()
        self._runtime = runtime

    @override
    async def __call__(self, params: TaskListParams) -> ToolReturnValue:
        views = list_task_views(
            self._runtime.background_tasks,
            active_only=params.active_only,
            limit=params.limit,
        )
        display = [
            BackgroundTaskDisplayBlock(
                task_id=view.spec.id,
                kind=view.spec.kind,
                status=view.runtime.status,
                description=view.spec.description,
            )
            for view in views
        ]
        return ToolReturnValue(
            is_error=False,
            output=format_task_list(views, active_only=params.active_only),
            message="Task list retrieved.",
            display=list(display),
        )


class TaskOutput(CallableTool2[TaskOutputParams]):
    name: str = "TaskOutput"
    description: str = load_desc(Path(__file__).parent / "output.md")
    params: type[TaskOutputParams] = TaskOutputParams

    def __init__(self, runtime: Runtime):
        super().__init__()
        self._runtime = runtime

    def _render_output_preview(
        self, task_id: str, *, status: TaskStatus, offset: int | None = None
    ) -> tuple[TaskOutputLineChunk, bool]:
        output_path = self._runtime.background_tasks.store.output_path(task_id)
        output_available = output_path.exists()
        chunk = self._runtime.background_tasks.store.read_output_lines(
            task_id,
            offset,
            TASK_OUTPUT_PREVIEW_BYTES,
            status=status,
        )
        return chunk, output_available

    @override
    async def __call__(self, params: TaskOutputParams) -> ToolReturnValue:
        view = self._runtime.background_tasks.get_task(params.task_id)
        if view is None:
            return ToolError(message=f"Task not found: {params.task_id}", brief="Task not found")

        if params.block:
            view = await self._runtime.background_tasks.wait(
                params.task_id,
                timeout_s=params.timeout,
            )
            retrieval_status = (
                "success"
                if view.runtime.status in {"completed", "failed", "killed", "lost"}
                else "timeout"
            )
        else:
            retrieval_status = (
                "success"
                if view.runtime.status in {"completed", "failed", "killed", "lost"}
                else "not_ready"
            )

        chunk, full_output_available = self._render_output_preview(
            params.task_id,
            status=view.runtime.status,
            offset=params.offset,
        )

        return ToolReturnValue(
            is_error=False,
            output=_format_task_output(
                view,
                retrieval_status=retrieval_status,
                chunk=chunk,
                full_output_available=full_output_available,
            ),
            message="Task output retrieved.",
            display=[_task_display(self._runtime, params.task_id)],
        )


class TaskStop(CallableTool2[TaskStopParams]):
    name: str = "TaskStop"
    description: str = load_desc(Path(__file__).parent / "stop.md")
    params: type[TaskStopParams] = TaskStopParams

    def __init__(self, runtime: Runtime, approval: Approval):
        super().__init__()
        self._runtime = runtime
        self._approval = approval

    @override
    async def __call__(self, params: TaskStopParams) -> ToolReturnValue:
        if self._runtime.session.state.plan_mode:
            return ToolError(
                message="TaskStop is not available in plan mode.",
                brief="Blocked in plan mode",
            )
        view = self._runtime.background_tasks.get_task(params.task_id)
        if view is None:
            return ToolError(message=f"Task not found: {params.task_id}", brief="Task not found")

        if not await self._approval.request(
            self.name,
            "stop background task",
            f"Stop background task `{params.task_id}`",
            display=[_task_display(self._runtime, params.task_id)],
        ):
            return ToolRejectedError()

        view = self._runtime.background_tasks.kill(
            params.task_id,
            reason=params.reason.strip() or "Stopped by TaskStop",
        )
        return ToolReturnValue(
            is_error=False,
            output=format_task(view, include_command=True),
            message="Task stop requested.",
            display=[_task_display(self._runtime, params.task_id)],
        )
