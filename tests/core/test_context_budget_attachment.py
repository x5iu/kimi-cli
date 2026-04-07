from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from kimi_cli.loop.attachments.context_budget import ContextBudgetAttachmentProvider
from llmkit.message import Message, TextPart


def _make_agent_loop_mock(*, context_usage: float, compaction_generation: int = 0) -> MagicMock:
    mock = MagicMock()
    type(mock)._context_usage = PropertyMock(return_value=context_usage)
    mock._compaction_generation = compaction_generation
    return mock


def _assistant_msg(text: str = "ok") -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _user_msg(text: str = "hello") -> Message:
    return Message(role="user", content=[TextPart(text=text)])


class TestContextBudgetAttachmentProvider:
    async def test_returns_empty_below_hint_threshold(self) -> None:
        provider = ContextBudgetAttachmentProvider()
        history = [_user_msg(), _assistant_msg()]

        result = await provider.get_attachments(history, _make_agent_loop_mock(context_usage=0.50))

        assert result == []

    async def test_returns_hint_at_hint_threshold(self) -> None:
        provider = ContextBudgetAttachmentProvider()
        history = [_user_msg(), _assistant_msg()]

        result = await provider.get_attachments(history, _make_agent_loop_mock(context_usage=0.65))

        assert len(result) == 1
        assert result[0].is_hint is True
        assert "65%" in result[0].content

    async def test_returns_reminder_at_reminder_threshold(self) -> None:
        provider = ContextBudgetAttachmentProvider()
        history = [_user_msg(), _assistant_msg()]

        result = await provider.get_attachments(history, _make_agent_loop_mock(context_usage=0.85))

        assert len(result) == 1
        assert result[0].is_hint is False
        assert "85%" in result[0].content
        assert "Minimize" in result[0].content

    async def test_respects_cooldown(self) -> None:
        provider = ContextBudgetAttachmentProvider(cooldown=3)
        history = [_user_msg(), _assistant_msg()]

        # First call should inject
        result1 = await provider.get_attachments(history, _make_agent_loop_mock(context_usage=0.70))
        assert len(result1) == 1

        # Add fewer than cooldown assistant messages
        history.append(_assistant_msg())
        history.append(_assistant_msg())

        result2 = await provider.get_attachments(history, _make_agent_loop_mock(context_usage=0.70))
        assert result2 == []

        # Add enough to clear cooldown
        history.append(_assistant_msg())

        result3 = await provider.get_attachments(history, _make_agent_loop_mock(context_usage=0.70))
        assert len(result3) == 1

    async def test_custom_thresholds(self) -> None:
        provider = ContextBudgetAttachmentProvider(hint_threshold=0.40, reminder_threshold=0.70)
        history = [_user_msg()]

        result = await provider.get_attachments(history, _make_agent_loop_mock(context_usage=0.45))
        assert len(result) == 1
        assert result[0].is_hint is True

    async def test_cooldown_resets_after_compaction(self) -> None:
        """After compaction (generation change), cooldown must not be permanently muted."""
        provider = ContextBudgetAttachmentProvider(cooldown=3)
        history = [_user_msg(), _assistant_msg()]

        # First injection at generation 0
        result1 = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.70, compaction_generation=0)
        )
        assert len(result1) == 1

        # Simulate compaction: generation incremented, history shrunk
        short_history = [_user_msg("compacted")]

        result2 = await provider.get_attachments(
            short_history, _make_agent_loop_mock(context_usage=0.70, compaction_generation=1)
        )
        # Should inject because cooldown was reset by generation change
        assert len(result2) == 1
