"""Tests that non-text content parts (e.g. images) attached to a slash command
are forwarded into the skill runner's ``_turn`` call."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kaos.path import KaosPath
from kimi_cli.eventbus.types import ImageURLPart, TextPart
from kimi_cli.loop.agent import Agent, Runtime
from kimi_cli.loop.context import Context
from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop
from kimi_cli.skill import Skill
from llmkit.message import Message
from llmkit.tooling.empty import EmptyToolset


def _make_soul(runtime: Runtime, tmp_path: Path) -> KimiAgentLoop:
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill = Skill(
        name="test-skill",
        description="A test skill",
        type="standard",
        dir=KaosPath.unsafe_from_local_path(skill_dir),
    )
    runtime.skills = {"test-skill": skill}

    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=EmptyToolset(),
        runtime=runtime,
    )
    return KimiAgentLoop(agent, context=Context(file_backend=tmp_path / "history.jsonl"))


@pytest.mark.asyncio
async def test_skill_runner_forwards_image_content_parts(runtime: Runtime, tmp_path: Path) -> None:
    soul = _make_soul(runtime, tmp_path)

    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="data:image/png;base64,AAAA"))

    # Simulate stashed content parts as if the slash-command dispatch had set them.
    soul._slash_command_content_parts = [image_part]

    # Find the skill runner slash command.
    cmd = soul._find_slash_command("skill:test-skill")
    assert cmd is not None

    captured_messages: list[Message] = []

    async def fake_turn(message: Message, **kwargs: object) -> None:
        captured_messages.append(message)

    with (
        patch.object(soul, "_turn", side_effect=fake_turn),
        patch("kimi_cli.loop.kimi_agent_loop.read_skill_text", new_callable=AsyncMock) as mock_read,
    ):
        mock_read.return_value = "Skill instructions here."
        await cmd.func(soul, "describe this image")

    assert len(captured_messages) == 1
    content = captured_messages[0].content
    assert isinstance(content, list)

    text_parts = [p for p in content if isinstance(p, TextPart)]
    image_parts = [p for p in content if isinstance(p, ImageURLPart)]

    assert len(text_parts) == 1
    assert "Skill instructions here." in text_parts[0].text
    assert "describe this image" in text_parts[0].text
    assert len(image_parts) == 1
    assert image_parts[0].image_url.url == "data:image/png;base64,AAAA"


@pytest.mark.asyncio
async def test_skill_runner_without_image_sends_plain_string(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul = _make_soul(runtime, tmp_path)

    # No non-text content parts stashed.
    soul._slash_command_content_parts = []

    cmd = soul._find_slash_command("skill:test-skill")
    assert cmd is not None

    captured_messages: list[Message] = []

    async def fake_turn(message: Message, **kwargs: object) -> None:
        captured_messages.append(message)

    with (
        patch.object(soul, "_turn", side_effect=fake_turn),
        patch("kimi_cli.loop.kimi_agent_loop.read_skill_text", new_callable=AsyncMock) as mock_read,
    ):
        mock_read.return_value = "Skill instructions here."
        await cmd.func(soul, "")

    assert len(captured_messages) == 1
    # When there are no extra parts, content should contain only the skill text.
    content = captured_messages[0].content
    assert isinstance(content, list)
    text_parts = [p for p in content if isinstance(p, TextPart)]
    image_parts = [p for p in content if isinstance(p, ImageURLPart)]
    assert len(text_parts) == 1
    assert "Skill instructions here." in text_parts[0].text
    assert len(image_parts) == 0
