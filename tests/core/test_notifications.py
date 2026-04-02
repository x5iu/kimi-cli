from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from kosong.message import Message, TextPart
from kosong.tooling.empty import EmptyToolset

from kimi_cli.background import TaskRuntime, TaskSpec
from kimi_cli.config import NotificationConfig
from kimi_cli.llm import LLM
from kimi_cli.notifications import NotificationEvent, NotificationManager
from kimi_cli.notifications.llm import build_notification_message, render_notification_text
from kimi_cli.notifications.models import NotificationDelivery, NotificationView
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
        approval=runtime.approval,
        environment=runtime.environment,
        notifications=runtime.notifications,
        background_tasks=runtime.background_tasks,
        skills=runtime.skills,
        additional_dirs=runtime.additional_dirs,
        skills_dirs=runtime.skills_dirs,
        agents_md=runtime.agents_md,
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


# ---------------------------------------------------------------------------
# has_pending_for_sink tests
# ---------------------------------------------------------------------------


def _make_notification_manager(tmp_path: Path) -> NotificationManager:
    return NotificationManager(
        tmp_path / "notifications",
        NotificationConfig(),
    )


def _make_event(
    mgr: NotificationManager,
    *,
    targets: list[str] | None = None,
) -> NotificationEvent:
    return NotificationEvent(
        id=mgr.new_id(),
        category="task",
        type="task.completed",
        source_kind="test",
        source_id="src1",
        title="Test notification",
        body="body",
        targets=targets or ["llm", "shell"],
    )


def test_has_pending_no_notifications(tmp_path: Path) -> None:
    """No notifications at all -> False for any sink."""
    mgr = _make_notification_manager(tmp_path)
    assert mgr.has_pending_for_sink("llm") is False
    assert mgr.has_pending_for_sink("shell") is False


def test_has_pending_matching_sink(tmp_path: Path) -> None:
    """A pending notification targeting 'llm' -> True for 'llm'."""
    mgr = _make_notification_manager(tmp_path)
    event = _make_event(mgr, targets=["llm"])
    mgr.publish(event)

    assert mgr.has_pending_for_sink("llm") is True


def test_has_pending_non_matching_sink(tmp_path: Path) -> None:
    """A pending notification targeting only 'llm' -> False for 'shell'."""
    mgr = _make_notification_manager(tmp_path)
    event = _make_event(mgr, targets=["llm"])
    mgr.publish(event)

    assert mgr.has_pending_for_sink("shell") is False


def test_has_pending_after_ack(tmp_path: Path) -> None:
    """After acknowledging the only notification for a sink -> False."""
    mgr = _make_notification_manager(tmp_path)
    event = _make_event(mgr, targets=["llm"])
    mgr.publish(event)

    # Claim then ack
    claimed = mgr.claim_for_sink("llm")
    assert len(claimed) == 1
    mgr.ack("llm", event.id)

    assert mgr.has_pending_for_sink("llm") is False


# ---------------------------------------------------------------------------
# Notification text rendering – line-bounded output tests
# ---------------------------------------------------------------------------


def _make_task_notification_view(
    runtime: Runtime, task_id: str, *, output: str
) -> NotificationView:
    """Write a completed task and return a NotificationView for it."""
    spec = TaskSpec(
        id=task_id,
        kind="bash",
        session_id=runtime.session.id,
        description="background build",
        tool_call_id="tool-notif",
        command="make build",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd=str(runtime.session.work_dir),
        timeout_s=60,
    )
    runtime.background_tasks.store.create_task(spec)
    runtime.background_tasks.store.output_path(spec.id).write_text(output, encoding="utf-8")
    runtime.background_tasks.store.write_runtime(
        spec.id,
        TaskRuntime(
            status="completed",
            exit_code=0,
            finished_at=time.time(),
            updated_at=time.time(),
        ),
    )
    event = NotificationEvent(
        id=runtime.notifications.new_id(),
        category="task",
        type="task.completed",
        source_kind="background_task",
        source_id=task_id,
        title="Background task completed",
        body=f"Task {task_id} completed.",
        severity="success",
    )
    return NotificationView(event=event, delivery=NotificationDelivery())


def test_notification_text_includes_whole_line_tail(runtime: Runtime) -> None:
    """Normal multi-line output: tail lines are included as whole lines."""
    nv = _make_task_notification_view(
        runtime, "bnotif001", output="line one\nline two\nline three\n"
    )
    msg = build_notification_message(nv, runtime)
    text = msg.extract_text("\n")
    assert "line one" in text
    assert "line three" in text
    assert "<output>" in text
    assert "</output>" in text
    assert "Line too large" not in text


def test_notification_text_overlong_line_emits_hint(runtime: Runtime) -> None:
    """A single overlong line emits a hint instead of raw content."""
    huge = "x" * (runtime.config.background.notification_tail_bytes + 100) + "\n"
    nv = _make_task_notification_view(runtime, "bnotif002", output=huge)
    msg = build_notification_message(nv, runtime)
    text = msg.extract_text("\n")
    assert "Line too large" in text
    assert "TaskOutput" in text
    assert "ReadFile" in text
    # The huge payload must NOT appear in the notification text
    assert huge.strip() not in text


def test_notification_text_empty_output(runtime: Runtime) -> None:
    """Empty output produces no <output> block and no 'Line too large' hint."""
    nv = _make_task_notification_view(runtime, "bnotif003", output="")
    msg = build_notification_message(nv, runtime)
    text = msg.extract_text("\n")
    assert "<output>" not in text
    assert "Line too large" not in text
    assert "Full output:" in text


def test_render_notification_text_includes_tail(runtime: Runtime) -> None:
    """render_notification_text includes bounded whole-line tail."""
    nv = _make_task_notification_view(
        runtime, "bnotif004", output="alpha\nbeta\n"
    )
    text = render_notification_text(nv, runtime)
    assert "alpha" in text
    assert "beta" in text
    assert "Full output:" in text


def test_render_notification_text_overlong_line(runtime: Runtime) -> None:
    """render_notification_text emits hint for overlong lines."""
    huge = "y" * (runtime.config.background.notification_tail_bytes + 100) + "\n"
    nv = _make_task_notification_view(runtime, "bnotif005", output=huge)
    text = render_notification_text(nv, runtime)
    assert "Line too large" in text
    assert huge.strip() not in text


def test_notification_tail_respects_byte_budget(runtime: Runtime) -> None:
    """Output exceeding the byte budget is truncated to whole lines."""
    # Each line is ~50 bytes; with default budget of 3000 chars we fit ~60 lines
    line = "a" * 49 + "\n"
    line_count = 200  # 200 * 50 = 10,000 bytes > 3,000
    nv = _make_task_notification_view(
        runtime, "bnotif006", output=line * line_count
    )
    msg = build_notification_message(nv, runtime)
    text = msg.extract_text("\n")
    # The full 200-line payload must not be in the text
    assert text.count("a" * 49) < line_count
    # But some lines should be present
    assert "a" * 49 in text
