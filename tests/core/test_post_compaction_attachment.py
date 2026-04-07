from __future__ import annotations

from unittest.mock import MagicMock

from kimi_cli.loop.attachments.post_compaction import (
    PostCompactionContinuityAttachmentProvider,
)
from kimi_cli.loop.message import internal_user_message, system
from llmkit.message import Message, TextPart


def _make_agent_loop_mock(*, compaction_generation: int = 0) -> MagicMock:
    mock = MagicMock()
    mock._compaction_generation = compaction_generation
    return mock


def _compaction_summary_msg() -> Message:
    return internal_user_message(
        [
            system("Previous context has been compacted. Here is the compaction output:"),
            TextPart(text="Summary of prior context..."),
        ]
    )


def _user_msg(text: str = "hello") -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _assistant_msg(text: str = "ok") -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


class TestPostCompactionContinuityAttachmentProvider:
    async def test_injects_after_compaction(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()
        history = [
            _compaction_summary_msg(),
            _user_msg("preserved user message"),
        ]

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=1)
        )

        assert len(result) == 1
        assert result[0].type == "post_compaction_continuity"
        assert "compacted" in result[0].content
        assert "RecallCompactedContext" in result[0].content
        assert result[0].is_hint is False

    async def test_fires_even_with_preserved_assistant_messages(self) -> None:
        """Compaction preserves tail turns that include assistant messages.

        The provider must still fire in this common case.
        """
        provider = PostCompactionContinuityAttachmentProvider()
        history = [
            _compaction_summary_msg(),
            _assistant_msg("preserved from previous turn"),
            _user_msg("preserved user message"),
            _assistant_msg("also preserved"),
        ]

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=1)
        )

        assert len(result) == 1

    async def test_fires_only_once_per_compaction(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()
        summary = _compaction_summary_msg()
        history = [summary, _user_msg("preserved")]
        mock = _make_agent_loop_mock(compaction_generation=1)

        result1 = await provider.get_attachments(history, mock)
        assert len(result1) == 1

        # Same generation — should not fire again
        history.append(_assistant_msg("new response"))
        result2 = await provider.get_attachments(history, mock)
        assert result2 == []

    async def test_fires_again_after_second_compaction(self) -> None:
        """A second compaction increments generation; provider must fire again."""
        provider = PostCompactionContinuityAttachmentProvider()

        history1 = [_compaction_summary_msg(), _user_msg("first")]
        result1 = await provider.get_attachments(
            history1, _make_agent_loop_mock(compaction_generation=1)
        )
        assert len(result1) == 1

        # Simulate second compaction — generation incremented
        history2 = [_compaction_summary_msg(), _user_msg("second")]
        result2 = await provider.get_attachments(
            history2, _make_agent_loop_mock(compaction_generation=2)
        )
        assert len(result2) == 1

    async def test_fires_again_after_shorter_second_compaction(self) -> None:
        """Regression: a second compaction with fewer messages must still fire."""
        provider = PostCompactionContinuityAttachmentProvider()

        long_history = [_compaction_summary_msg()] + [_user_msg() for _ in range(10)]
        result1 = await provider.get_attachments(
            long_history, _make_agent_loop_mock(compaction_generation=1)
        )
        assert len(result1) == 1

        # Shorter history after second compaction, generation incremented
        short_history = [_compaction_summary_msg(), _user_msg("only one")]
        result2 = await provider.get_attachments(
            short_history, _make_agent_loop_mock(compaction_generation=2)
        )
        assert len(result2) == 1

    async def test_returns_empty_for_non_compacted_history(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()
        history = [_user_msg("normal message")]

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=0)
        )
        assert result == []

    async def test_returns_empty_for_empty_history(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()

        result = await provider.get_attachments([], _make_agent_loop_mock(compaction_generation=0))
        assert result == []
