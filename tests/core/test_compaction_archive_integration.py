from __future__ import annotations

from pathlib import Path

import pytest

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.agent import Agent, Runtime
from kimi_cli.loop.compaction import CompactionResult
from kimi_cli.loop.compaction_archive import load_compaction_archives
from kimi_cli.loop.context import Context
from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop
from kimi_cli.loop.message import internal_user_message, system
from kimi_cli.loop.toolset import KimiToolset
from kimi_cli.tools.context import RecallCompactedContext
from kimi_cli.tools.context.recall_compacted import Params
from llmkit.chat_provider import TokenUsage
from llmkit.message import Message


class FakeCompaction:
    async def compact(self, messages, llm, *, custom_instruction: str = "") -> CompactionResult:
        return CompactionResult(
            messages=[
                internal_user_message(
                    [
                        system(
                            "Previous context has been compacted. Here is the compaction output:"
                        ),
                        TextPart(
                            text=(
                                "<current_focus>\nInvestigating alpha issue\n</current_focus>\n"
                                "<important_context>\n- alpha traceback kept\n</important_context>"
                            )
                        ),
                    ]
                ),
                Message(role="user", content=[TextPart(text="Latest question")]),
                Message(role="assistant", content=[TextPart(text="Latest answer")]),
            ],
            usage=TokenUsage(input_other=120, output=40, input_cache_read=0),
        )


async def test_compaction_registers_archive_and_injects_recall_notice(
    runtime: Runtime, tmp_path: Path, monkeypatch
) -> None:
    recall_tool = RecallCompactedContext(runtime)
    toolset = KimiToolset()
    toolset.add(recall_tool)
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=runtime,
    )
    context = Context(file_backend=tmp_path / "history.jsonl")
    await context.append_message(
        [
            Message(role="user", content=[TextPart(text="Old alpha traceback")]),
            Message(role="assistant", content=[TextPart(text="Old alpha analysis")]),
            Message(role="user", content=[TextPart(text="Latest question")]),
            Message(role="assistant", content=[TextPart(text="Latest answer")]),
        ]
    )

    soul = KimiAgentLoop(agent, context=context)
    soul._compaction = FakeCompaction()
    monkeypatch.setattr("kimi_cli.loop.kimi_agent_loop.bus_send", lambda _msg: None)

    assert [tool.name for tool in toolset.tools] == []

    await soul.compact_context()

    records = load_compaction_archives(context.file_backend)
    assert len(records) == 1
    assert records[0].id == "c001"
    assert "Investigating alpha issue" in records[0].summary
    assert any(
        any(
            isinstance(part, TextPart) and "RecallCompactedContext tool" in part.text
            for part in message.content
        )
        for message in context.history
    )
    assert [tool.name for tool in toolset.tools] == ["RecallCompactedContext"]
    assert context.token_count > 0

    recall_result = await recall_tool(Params(query="alpha"))
    assert not recall_result.is_error
    assert "Old alpha traceback" in recall_result.output


async def test_finalize_failure_restores_context_and_does_not_increment_generation(
    runtime: Runtime, tmp_path: Path, monkeypatch
) -> None:
    recall_tool = RecallCompactedContext(runtime)
    toolset = KimiToolset()
    toolset.add(recall_tool)
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=runtime,
    )
    context = Context(file_backend=tmp_path / "history.jsonl")
    original = [
        Message(role="user", content=[TextPart(text="Old alpha traceback")]),
        Message(role="assistant", content=[TextPart(text="Old alpha analysis")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]
    await context.append_message(original)

    soul = KimiAgentLoop(agent, context=context)
    soul._compaction = FakeCompaction()
    monkeypatch.setattr("kimi_cli.loop.kimi_agent_loop.bus_send", lambda _msg: None)

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("finalize failed")

    monkeypatch.setattr(
        "kimi_cli.loop.kimi_agent_loop.finalize_compaction_archive_registration",
        _boom,
    )

    gen_before = soul._compaction_generation
    with pytest.raises(OSError, match="finalize failed"):
        await soul.compact_context()

    assert soul._compaction_generation == gen_before
    assert len(context.history) == len(original)
    assert context.history[0].content[0].text == "Old alpha traceback"  # type: ignore[union-attr]
    assert load_compaction_archives(context.file_backend) == []


async def test_begin_archive_registration_failure_restores_context_and_does_not_increment_generation(
    runtime: Runtime, tmp_path: Path, monkeypatch
) -> None:
    recall_tool = RecallCompactedContext(runtime)
    toolset = KimiToolset()
    toolset.add(recall_tool)
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=runtime,
    )
    context = Context(file_backend=tmp_path / "history.jsonl")
    original = [
        Message(role="user", content=[TextPart(text="Old alpha traceback")]),
        Message(role="assistant", content=[TextPart(text="Old alpha analysis")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]
    await context.append_message(original)

    soul = KimiAgentLoop(agent, context=context)
    soul._compaction = FakeCompaction()
    monkeypatch.setattr("kimi_cli.loop.kimi_agent_loop.bus_send", lambda _msg: None)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("begin registration failed")

    monkeypatch.setattr(
        "kimi_cli.loop.kimi_agent_loop.begin_compaction_archive_registration",
        _boom,
    )

    gen_before = soul._compaction_generation
    with pytest.raises(OSError, match="begin registration failed"):
        await soul.compact_context()

    assert soul._compaction_generation == gen_before
    assert len(context.history) == len(original)
    assert context.history[0].content[0].text == "Old alpha traceback"  # type: ignore[union-attr]
    assert load_compaction_archives(context.file_backend) == []


async def test_recall_tool_is_visible_immediately_when_archives_already_exist(
    runtime: Runtime, tmp_path: Path
) -> None:
    context = Context(file_backend=tmp_path / "history.jsonl")
    context.file_backend.touch()
    archive_file = tmp_path / "history_1.jsonl"
    archive_file.write_text(
        Message(role="user", content=[TextPart(text="Earlier compacted note")]).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    from kimi_cli.loop.compaction_archive import register_compaction_archive

    register_compaction_archive(
        context.file_backend,
        archive_file,
        message_count=1,
        summary="Earlier compacted note",
    )

    recall_tool = RecallCompactedContext(runtime)
    toolset = KimiToolset()
    toolset.add(recall_tool)
    soul = KimiAgentLoop(
        Agent(
            name="Test Agent",
            system_prompt="Test system prompt.",
            toolset=toolset,
            runtime=runtime,
        ),
        context=context,
    )

    assert soul is not None
    assert [tool.name for tool in toolset.tools] == ["RecallCompactedContext"]
