from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type TaskKind = Literal["bash", "agent"]
type TaskStatus = Literal["created", "starting", "running", "completed", "failed", "killed", "lost"]
type TaskOwnerRole = Literal["root"]

TERMINAL_TASK_STATUSES: tuple[TaskStatus, ...] = ("completed", "failed", "killed", "lost")


def is_terminal_status(status: TaskStatus) -> bool:
    return status in TERMINAL_TASK_STATUSES


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    id: str
    kind: TaskKind
    session_id: str
    description: str
    tool_call_id: str
    owner_role: TaskOwnerRole = "root"
    created_at: float = Field(default_factory=time.time)

    command: str | None = None
    shell_name: str | None = None
    shell_path: str | None = None
    cwd: str | None = None
    timeout_s: int | None = None
    kind_payload: dict[str, str] | None = None


class TaskRuntime(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: TaskStatus = "created"
    worker_pid: int | None = None
    child_pid: int | None = None
    child_pgid: int | None = None
    started_at: float | None = None
    heartbeat_at: float | None = None
    updated_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    interrupted: bool = False
    timed_out: bool = False
    failure_reason: str | None = None


class TaskControl(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kill_requested_at: float | None = None
    kill_reason: str | None = None
    force: bool = False


class TaskView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    spec: TaskSpec
    runtime: TaskRuntime
    control: TaskControl


class TaskOutputLineChunk(BaseModel):
    """A chunk of output lines from a background task.

    Pagination fields:
    - ``start_line`` / ``end_line``: 0-based half-open range ``[start, end)``.
    - ``has_before``: ``True`` when lines exist before ``start_line``.
    - ``has_after``: ``True`` when lines exist after ``end_line``.
    - ``next_offset``: next line number to pass as *offset* for forward
      pagination; ``None`` when there is nothing more to read.
    """

    model_config = ConfigDict(extra="ignore")

    task_id: str
    start_line: int
    end_line: int
    has_before: bool
    has_after: bool
    next_offset: int | None
    text: str
    status: TaskStatus
    output_path: str
    line_too_large: bool
