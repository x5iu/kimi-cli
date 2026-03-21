import asyncio
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self, override

from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field, model_validator

import kaos
from kaos import AsyncReadable
from kimi_cli.background import TaskView, format_task
from kimi_cli.soul import get_wire_or_none, wire_send
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.soul.toolset import get_current_tool_call_or_none
from kimi_cli.tools.display import BackgroundTaskDisplayBlock, ShellDisplayBlock
from kimi_cli.tools.file.rg_path import find_existing_rg, format_rg_command
from kimi_cli.tools.utils import ToolRejectedError, ToolResultBuilder, load_desc
from kimi_cli.utils.environment import Environment
from kimi_cli.utils.subprocess_env import get_clean_env
from kimi_cli.wire.types import ToolCallOutput

MAX_FOREGROUND_TIMEOUT = 5 * 60
MAX_BACKGROUND_TIMEOUT = 24 * 60 * 60
MAX_TIMEOUT = MAX_BACKGROUND_TIMEOUT


def _build_rg_preference_guidance(*, is_powershell: bool) -> str:
    rg_path = find_existing_rg()
    if rg_path is None:
        return ""

    rg_path_str = str(rg_path)
    command = format_rg_command(rg_path, is_powershell=is_powershell)
    path_lookup = "where" if platform.system() == "Windows" else "which"
    return "\n".join(
        [
            "**Preferred Content Search:**",
            f"- `rg` is available at `{rg_path_str}`.",
            (
                "- When you need to search file contents, prefer Shell with "
                f"`{command}` instead of the Grep tool."
            ),
            (
                "- If you need to verify or rediscover the binary, run "
                f"`{path_lookup} rg`."
            ),
        ]
    )


class Params(BaseModel):
    command: str = Field(description="The bash command to execute.")
    timeout: int = Field(
        description=(
            "The timeout in seconds for the command to execute. "
            "If the command takes longer than this, it will be killed."
        ),
        default=60,
        ge=1,
        le=MAX_BACKGROUND_TIMEOUT,
    )
    run_in_background: bool = Field(
        default=False,
        description="Whether to run the command as a background task.",
    )
    description: str = Field(
        default="",
        description=(
            "A short description for the background task. Required when run_in_background=true."
        ),
    )

    @model_validator(mode="after")
    def _validate_background_fields(self) -> Self:
        if self.run_in_background and not self.description.strip():
            raise ValueError("description is required when run_in_background is true")
        if not self.run_in_background and self.timeout > MAX_FOREGROUND_TIMEOUT:
            raise ValueError(
                f"timeout must be <= {MAX_FOREGROUND_TIMEOUT}s for foreground commands; "
                f"use run_in_background=true for longer timeouts (up to {MAX_BACKGROUND_TIMEOUT}s)"
            )
        return self


class Shell(CallableTool2[Params]):
    name: str = "Shell"
    params: type[Params] = Params

    def __init__(self, approval: Approval, environment: Environment, runtime: Runtime):
        is_powershell = environment.shell_name == "Windows PowerShell"
        super().__init__(
            description=load_desc(
                Path(__file__).parent / ("powershell.md" if is_powershell else "bash.md"),
                {
                    "SHELL": f"{environment.shell_name} (`{environment.shell_path}`)",
                    "RG_PREFERENCE_GUIDANCE": _build_rg_preference_guidance(
                        is_powershell=is_powershell
                    ),
                },
            )
        )
        self._approval = approval
        self._is_powershell = is_powershell
        self._shell_path = environment.shell_path
        self._runtime = runtime

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        builder = ToolResultBuilder()

        if not params.command:
            return builder.error("Command cannot be empty.", brief="Empty command")

        if params.run_in_background:
            return await self._run_in_background(params)

        if not await self._approval.request(
            self.name,
            "run command",
            f"Run command `{params.command}`",
            display=[
                ShellDisplayBlock(
                    language="powershell" if self._is_powershell else "bash",
                    command=params.command,
                )
            ],
        ):
            return ToolRejectedError()

        tool_call = get_current_tool_call_or_none()
        has_live_output = tool_call is not None and get_wire_or_none() is not None

        def output_cb(text: str, stream: Literal["stdout", "stderr"]) -> None:
            if has_live_output and tool_call is not None and text:
                wire_send(ToolCallOutput(tool_call_id=tool_call.id, text=text, stream=stream))

        def stdout_cb(line: bytes):
            line_str = line.decode(encoding="utf-8", errors="replace")
            builder.write(line_str)
            output_cb(line_str, "stdout")

        def stderr_cb(line: bytes):
            line_str = line.decode(encoding="utf-8", errors="replace")
            builder.write(line_str)
            output_cb(line_str, "stderr")

        try:
            exitcode = await self._run_shell_command(
                params.command, stdout_cb, stderr_cb, params.timeout
            )

            if exitcode == 0:
                return builder.ok("Command executed successfully.")
            else:
                return builder.error(
                    f"Command failed with exit code: {exitcode}.",
                    brief=f"Failed with exit code: {exitcode}",
                )
        except TimeoutError:
            return builder.error(
                f"Command killed by timeout ({params.timeout}s)",
                brief=f"Killed by timeout ({params.timeout}s)",
            )

    async def _run_in_background(self, params: Params) -> ToolReturnValue:
        tool_call = get_current_tool_call_or_none()
        if tool_call is None:
            return ToolResultBuilder().error(
                "Background shell requires a tool call context.",
                brief="No tool call context",
            )

        if not await self._approval.request(
            self.name,
            "run background command",
            f"Run background command `{params.command}`",
            display=[
                ShellDisplayBlock(
                    language="powershell" if self._is_powershell else "bash",
                    command=params.command,
                )
            ],
        ):
            return ToolRejectedError()

        try:
            view = self._runtime.background_tasks.create_bash_task(
                command=params.command,
                description=params.description.strip(),
                timeout_s=params.timeout,
                tool_call_id=tool_call.id,
                shell_name="Windows PowerShell" if self._is_powershell else "bash",
                shell_path=str(self._shell_path),
                cwd=str(self._runtime.session.work_dir),
            )
        except Exception as exc:
            builder = ToolResultBuilder()
            return builder.error(f"Failed to start background task: {exc}", brief="Start failed")

        return self._background_ok(view)

    def _background_ok(self, view: TaskView) -> ToolReturnValue:
        builder = ToolResultBuilder()
        live_notification = self._runtime.has_live_background_notifications
        notification_line = (
            "next_step: You will be automatically notified in this session when it completes."
            if live_notification
            else (
                "next_step: No live notification channel is active for this run; "
                "use TaskOutput or reopen the session later to inspect the task."
            )
        )
        builder.write(
            "\n".join(
                [
                    format_task(view, include_command=True),
                    f"automatic_notification: {str(live_notification).lower()}",
                    notification_line,
                    (
                        "next_step: Use TaskOutput with this task_id "
                        "if you need progress or want to wait."
                    ),
                    "next_step: Use TaskStop only if the task must be cancelled.",
                    (
                        "human_shell_hint: For users in the interactive shell, "
                        "the only task-management slash command is /task."
                    ),
                ]
            )
        )
        builder.display(
            BackgroundTaskDisplayBlock(
                task_id=view.spec.id,
                kind=view.spec.kind,
                status=view.runtime.status,
                description=view.spec.description,
            )
        )
        return builder.ok("Background task started", brief=f"Started {view.spec.id}")

    async def _run_shell_command(
        self,
        command: str,
        stdout_cb: Callable[[bytes], None],
        stderr_cb: Callable[[bytes], None],
        timeout: int,
    ) -> int:
        async def _read_stream(stream: AsyncReadable, cb: Callable[[bytes], None]):
            while True:
                line = await stream.readline()
                if line:
                    cb(line)
                else:
                    break

        process = await kaos.exec(*self._shell_args(command), env=get_clean_env())

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _read_stream(process.stdout, stdout_cb),
                    _read_stream(process.stderr, stderr_cb),
                ),
                timeout,
            )
            return await process.wait()
        except asyncio.CancelledError:
            await process.kill()
            raise
        except TimeoutError:
            await process.kill()
            raise

    def _shell_args(self, command: str) -> tuple[str, ...]:
        if self._is_powershell:
            return (str(self._shell_path), "-command", command)
        return (str(self._shell_path), "-c", command)
