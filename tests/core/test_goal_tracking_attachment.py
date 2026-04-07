from __future__ import annotations

from unittest.mock import MagicMock

from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from llmkit.message import Message, TextPart


def _make_agent_loop_mock() -> MagicMock:
    return MagicMock()


def _user_msg(text: str, *, name: str | None = None) -> Message:
    msg = Message(role="user", content=[TextPart(text=text)])
    if name is not None:
        msg = Message(role="user", content=[TextPart(text=text)], name=name)
    return msg


def _assistant_msg(text: str = "ok") -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _internal_user_msg(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)], name="_kimi_internal")


class TestGoalTrackingAttachmentProvider:
    async def test_returns_empty_below_activation_threshold(self) -> None:
        provider = GoalTrackingAttachmentProvider(activate_after=10, inject_every=10)
        history = [_user_msg("Fix the bug")] + [_assistant_msg() for _ in range(9)]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert result == []

    async def test_injects_at_activation_threshold(self) -> None:
        provider = GoalTrackingAttachmentProvider(activate_after=10, inject_every=10)
        history = [_user_msg("Fix the bug")] + [_assistant_msg() for _ in range(10)]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert result[0].type == "goal_tracking"
        assert "Fix the bug" in result[0].content
        assert result[0].is_hint is False

    async def test_injects_at_multiples_of_inject_every(self) -> None:
        provider = GoalTrackingAttachmentProvider(activate_after=5, inject_every=5)
        # 15 assistant messages: should inject at 5, 10, 15
        history = [_user_msg("Deploy the app")] + [_assistant_msg() for _ in range(15)]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert "Deploy the app" in result[0].content

    async def test_does_not_inject_between_intervals(self) -> None:
        provider = GoalTrackingAttachmentProvider(activate_after=5, inject_every=5)
        # 7 assistant messages: 7-5=2, 2%5 != 0
        history = [_user_msg("Deploy the app")] + [_assistant_msg() for _ in range(7)]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert result == []

    async def test_returns_empty_when_no_user_message(self) -> None:
        provider = GoalTrackingAttachmentProvider(activate_after=1, inject_every=1)
        history = [_assistant_msg() for _ in range(5)]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert result == []

    async def test_truncates_long_goals(self) -> None:
        provider = GoalTrackingAttachmentProvider(activate_after=1, inject_every=1)
        long_goal = "A" * 500
        history = [_user_msg(long_goal)] + [_assistant_msg()]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert result[0].content.endswith("...")
        assert len(result[0].content) < 500

    async def test_ignores_internal_user_messages(self) -> None:
        provider = GoalTrackingAttachmentProvider(activate_after=2, inject_every=2)
        history = [
            _user_msg("Original goal"),
            _assistant_msg(),
            _internal_user_msg("<system-reminder>\nsome reminder\n</system-reminder>"),
            _assistant_msg(),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert "Original goal" in result[0].content
