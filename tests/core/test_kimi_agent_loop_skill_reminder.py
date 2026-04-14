from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import kimi_cli.loop.kimi_agent_loop as kimisoul_module
from kaos.path import KaosPath
from kimi_cli.eventbus.types import SkillReminderNotice
from kimi_cli.loop.agent import Agent, Runtime
from kimi_cli.loop.context import Context
from kimi_cli.loop.kimi_agent_loop import (
    KimiAgentLoop,
    SkillRecommendation,
    SkillRecommendationItem,
    StepOutcome,
)
from kimi_cli.loop.message import INTERNAL_USER_NAME
from kimi_cli.skill import Skill
from llmkit.message import Message
from llmkit.tooling.empty import EmptyToolset


def _make_skill(tmp_path: Path, *, name: str, description: str) -> Skill:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    return Skill(
        name=name,
        description=description,
        dir=KaosPath.unsafe_from_local_path(skill_dir),
    )


@pytest.mark.asyncio
async def test_request_skill_recommendation_includes_available_skills(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _make_skill(tmp_path, name="gen-docs", description="Update user documentation.")
    runtime.skills = {"gen-docs": skill}

    soul = KimiAgentLoop(
        Agent(
            name="Skill Reminder Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    captured: dict[str, object] = {}

    async def fake_generate(*, chat_provider, system_prompt, tools, history):
        captured["chat_provider"] = chat_provider
        captured["system_prompt"] = system_prompt
        captured["tools"] = tools
        captured["history"] = history
        return SimpleNamespace(
            message=Message(
                role="assistant",
                content='{"skills":[{"name":"gen-docs","reason":"The user asked for docs help."}]}',
            )
        )

    monkeypatch.setattr(kimisoul_module.llmkit, "generate", fake_generate)

    history = [Message(role="user", content="Please update the docs.")]
    recommendation = await soul._request_skill_recommendation(history)

    assert recommendation == SkillRecommendation(
        skills=(
            SkillRecommendationItem(
                name="gen-docs",
                reason="The user asked for docs help.",
            ),
        )
    )
    assert captured["history"] == history
    assert captured["tools"] == []
    assert "gen-docs" in str(captured["system_prompt"])
    assert str(skill.skill_md_file) in str(captured["system_prompt"])


@pytest.mark.asyncio
async def test_turn_injects_skill_reminder_on_next_step(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _make_skill(tmp_path, name="gen-docs", description="Update user documentation.")
    runtime.skills = {"gen-docs": skill}

    soul = KimiAgentLoop(
        Agent(
            name="Skill Reminder Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    reminder_ready = asyncio.Event()
    step_calls = 0

    async def fake_request(self: KimiAgentLoop, history):
        await reminder_ready.wait()
        return SkillRecommendation(
            skills=(
                SkillRecommendationItem(
                    name="gen-docs",
                    reason="The request is about editing documentation.",
                ),
            )
        )

    async def fake_step(self: KimiAgentLoop) -> StepOutcome:
        nonlocal step_calls
        step_calls += 1
        if step_calls == 1:
            reminder_ready.set()
            await asyncio.sleep(0)
        return StepOutcome(
            stop_reason="no_tool_calls",
            assistant_message=Message(role="assistant", content=f"step {step_calls}"),
        )

    sent_messages: list[object] = []

    monkeypatch.setattr(KimiAgentLoop, "_request_skill_recommendation", fake_request)
    monkeypatch.setattr(KimiAgentLoop, "_step", fake_step)
    monkeypatch.setattr(kimisoul_module, "bus_send", sent_messages.append)

    result = await soul._turn(
        Message(role="user", content="Please update the docs."),
        enable_skill_reminder=True,
    )

    assert step_calls == 2
    assert result.step_count == 2
    assert result.final_message is not None
    assert result.final_message.extract_text(" ") == "step 2"

    reminder_messages = [
        message.extract_text(" ")
        for message in soul.context.history
        if message.role == "user" and "workflow patterns" in message.extract_text(" ")
    ]
    assert [
        message.name
        for message in soul.context.history
        if message.role == "user" and "workflow patterns" in message.extract_text(" ")
    ] == [INTERNAL_USER_NAME]
    assert reminder_messages == [
        (
            "<system>Reminder: the following skills may offer useful workflow patterns for the "
            "current task. They are suggestions only \u2014 do not change your current plan "
            "or goal based on this reminder alone.\n"
            "- gen-docs (standard skill): Update user documentation.\n"
            "  Reason: The request is about editing documentation.\n"
            f"  Path: {skill.skill_md_file}</system>"
        )
    ]
    assert SkillReminderNotice(skills=["/skill:gen-docs"]) in sent_messages
