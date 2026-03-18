from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from kosong.message import Message, TextPart
from kosong.tooling.empty import EmptyToolset

from kimi_cli.background import TaskRuntime, TaskSpec
from kimi_cli.llm import LLM
from kimi_cli.notifications import NotificationEvent
from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.wire import Wire


class _StaticProvider:
    name = "notification-static"

    @property
    def model_name(self) -> str:
        return "notification-static"

    @property
    def thinking_effort(self):
        return None

    async def generate(self, system_prompt: str, tools, history):
        del system_prompt, tools, history
        return _StaticStream()

    def with_thinking(self, effort):
        del effort
        return self


class _StaticStream:
    def __aiter__(self):
        self._done = False
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return TextPart(text="done")

    @property
    def id(self):
        return "notification-stream"

    @property
    def usage(self):
        return None


def _runtime_with_llm(runtime: Runtime, llm: LLM) -> Runtime:
    return Runtime(
        config=runtime.config,
        llm=llm,
        session=runtime.session,
        builtin_args=runtime.builtin_args,
        denwa_renji=runtime.denwa_renji,
        approval=runtime.approval,
        labor_market=runtime.labor_market,
        environment=runtime.environment,
        notifications=runtime.notifications,
        background_tasks=runtime.background_tasks,
        skills=runtime.skills,
        oauth=runtime.oauth,
        additional_dirs=runtime.additional_dirs,
        role=runtime.role,
    )


def _make_soul(runtime: Runtime, tmp_path: Path) -> tuple[KimiSoul, Context]:
    llm = LLM(
        chat_provider=_StaticProvider(),
        max_context_size=100_000,
        capabilities=set(),
    )
    agent = Agent(
        name="Notification Agent",
        system_prompt="System prompt.",
        toolset=EmptyToolset(),
        runtime=_runtime_with_llm(runtime, llm),
    )
    context = Context(file_backend=tmp_path / "history.jsonl")
    return KimiSoul(agent, context=context), context


def _write_completed_task(runtime: Runtime, task_id: str) -> None:
    spec = TaskSpec(
        id=task_id,
        kind="bash",
        session_id=runtime.session.id,
        description="background completion",
        tool_call_id="tool-8",
        command="echo done",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
    )
    runtime.background_tasks.store.create_task(spec)
    runtime.background_tasks.store.output_path(spec.id).write_text(
        "line 1\nline 2\n", encoding="utf-8"
    )
    runtime.background_tasks.store.write_runtime(
        spec.id,
        TaskRuntime(
            status="completed",
            exit_code=0,
            finished_at=time.time(),
            updated_at=time.time(),
        ),
    )


def test_notification_publish_merges_new_targets(runtime: Runtime) -> None:
    event = NotificationEvent(
        id=runtime.notifications.new_id(),
        category="task",
        type="task.completed",
        source_kind="background_task",
        source_id="b9999999",
        title="Background task completed",
        body="done",
        dedupe_key="task:b9999999:completed",
        targets=["llm"],
    )
    runtime.notifications.publish(event)

    merged = runtime.notifications.publish(
        event.model_copy(update={"id": runtime.notifications.new_id(), "targets": ["llm", "shell"]})
    )

    assert merged.event.targets == ["llm", "shell"]
    assert set(merged.delivery.sinks) == {"llm", "shell"}


@pytest.mark.asyncio
async def test_kimisoul_appends_notification_message(runtime: Runtime, tmp_path: Path) -> None:
    _write_completed_task(runtime, "b3333333")
    runtime.background_tasks.publish_terminal_notifications()

    soul, context = _make_soul(runtime, tmp_path)

    async def _drain_ui(wire: Wire) -> None:
        wire_ui = wire.ui_side(merge=True)
        while True:
            try:
                await wire_ui.receive()
            except QueueShutDown:
                return

    await run_soul(soul, "check status", _drain_ui, asyncio.Event())

    notification_texts = [
        message.extract_text("\n")
        for message in context.history
        if isinstance(message, Message) and "<notification " in message.extract_text("\n")
    ]
    assert len(notification_texts) == 1
    assert "Task ID: b3333333" in notification_texts[0]
    assert "line 2" in notification_texts[0]
