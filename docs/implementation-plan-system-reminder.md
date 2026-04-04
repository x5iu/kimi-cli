Now I have a complete understanding of the codebase. Let me produce the detailed implementation plan.

---

# Implementation Plan: Mid-Turn `<system-reminder>` Optimization (P1–P5)

## Table of Contents
1. [P1: GoalTrackingAttachmentProvider](#p1) — New file + registration
2. [P2: ContextBudgetAttachmentProvider](#p2) — New file + registration
3. [P3: Steer Instruction Deduplication](#p3) — Modify existing file
4. [P4: Tool Result Tag Sanitization](#p4) — Modify existing file
5. [P5: PostCompactionContinuityAttachmentProvider](#p5) — New file + registration

---

<a id="p1"></a>
## P1: GoalTrackingAttachmentProvider (HIGH IMPACT)

### Goal
Prevent goal drift in long turns by periodically reminding the agent of the original user request.

### 1.1 New File: `src/kimi_cli/loop/attachments/goal_tracking.py`

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmkit.message import Message

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.utils.turns import is_real_user_turn_start_message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


# Default: start injecting after this many assistant messages in the current turn
_DEFAULT_ACTIVATE_AFTER = 10
# Default: inject every N assistant messages after activation
_DEFAULT_INJECT_EVERY = 10
# Max characters to include from the original user request
_MAX_GOAL_LENGTH = 300


class GoalTrackingAttachmentProvider(AttachmentProvider):
    """Periodically reminds the agent of the original user request during long turns.

    Activates after ``activate_after`` assistant messages in the current turn,
    then injects a goal-tracking reminder every ``inject_every`` assistant messages.
    """

    def __init__(
        self,
        *,
        activate_after: int = _DEFAULT_ACTIVATE_AFTER,
        inject_every: int = _DEFAULT_INJECT_EVERY,
    ) -> None:
        self._activate_after = activate_after
        self._inject_every = inject_every

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        # Find the last real user turn-start message (the current turn's request)
        goal_text: str | None = None
        goal_index: int = -1
        for i in range(len(history) - 1, -1, -1):
            if is_real_user_turn_start_message(history[i]):
                goal_text = _extract_text(history[i])
                goal_index = i
                break

        if goal_text is None or goal_index < 0:
            return []

        # Count assistant messages after the turn-start message
        assistant_count = sum(
            1 for msg in history[goal_index + 1 :] if msg.role == "assistant"
        )

        if assistant_count < self._activate_after:
            return []

        # Only inject at multiples of inject_every (after activate_after)
        steps_since_activation = assistant_count - self._activate_after
        if steps_since_activation % self._inject_every != 0:
            return []

        # Truncate long goals
        truncated = goal_text
        if len(truncated) > _MAX_GOAL_LENGTH:
            truncated = truncated[:_MAX_GOAL_LENGTH] + "..."

        return [
            Attachment(
                type="goal_tracking",
                content=(
                    f"You are now {assistant_count} steps into this turn. "
                    f"Stay focused on the original request:\n\n{truncated}"
                ),
                is_hint=False,  # system-reminder (authoritative)
            )
        ]


def _extract_text(message: Message) -> str | None:
    """Extract the concatenated text content from a message."""
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextPart):
            parts.append(part.text)
    return " ".join(parts).strip() if parts else None
```

### 1.2 Modify: `src/kimi_cli/loop/kimi_agent_loop.py` — Register provider

**BEFORE** (line 64):
```python
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
```

**AFTER**:
```python
from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
```

**BEFORE** (line 223):
```python
        self._attachment_providers: list[AttachmentProvider] = [PreferShellRgAttachmentProvider()]
```

**AFTER**:
```python
        self._attachment_providers: list[AttachmentProvider] = [
            PreferShellRgAttachmentProvider(),
            GoalTrackingAttachmentProvider(),
        ]
```

### 1.3 New Test: `tests/core/test_goal_tracking_attachment.py`

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from llmkit.message import Message, TextPart

from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider


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
```

---

<a id="p2"></a>
## P2: ContextBudgetAttachmentProvider (MEDIUM IMPACT)

### Goal
Inform the agent when context is getting full, escalating from hint to reminder.

### 2.1 New File: `src/kimi_cli/loop/attachments/context_budget.py`

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmkit.message import Message

from kimi_cli.loop.attachment import Attachment, AttachmentProvider

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


# Thresholds for injection
_HINT_THRESHOLD = 0.60  # inject a <system-hint> at 60%
_REMINDER_THRESHOLD = 0.80  # escalate to <system-reminder> at 80%

# Don't inject more than once every N assistant messages
_COOLDOWN_ASSISTANT_MESSAGES = 5

_CONTEXT_BUDGET_TYPE = "context_budget"


class ContextBudgetAttachmentProvider(AttachmentProvider):
    """Warns the agent when context budget usage is high.

    - At 60% usage: injects a ``<system-hint>`` suggesting conciseness.
    - At 80% usage: escalates to ``<system-reminder>`` with stronger guidance.
    Self-throttles by checking history for a recent context_budget attachment.
    """

    def __init__(
        self,
        *,
        hint_threshold: float = _HINT_THRESHOLD,
        reminder_threshold: float = _REMINDER_THRESHOLD,
        cooldown: int = _COOLDOWN_ASSISTANT_MESSAGES,
    ) -> None:
        self._hint_threshold = hint_threshold
        self._reminder_threshold = reminder_threshold
        self._cooldown = cooldown
        self._last_inject_index: int = -1  # history length at last injection

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        usage = agent_loop._context_usage  # noqa: SLF001
        if usage < self._hint_threshold:
            return []

        # Cooldown: count assistant messages since last injection
        if self._last_inject_index >= 0:
            assistant_since = sum(
                1
                for msg in history[self._last_inject_index :]
                if msg.role == "assistant"
            )
            if assistant_since < self._cooldown:
                return []

        pct = int(usage * 100)
        is_critical = usage >= self._reminder_threshold

        if is_critical:
            content = (
                f"Context budget is at {pct}% — approaching the limit. "
                "Be concise in tool calls and responses. "
                "Avoid reading large files in full; use targeted line ranges. "
                "Consider completing the current task soon or summarizing progress."
            )
        else:
            content = (
                f"Context budget is at {pct}%. "
                "Prefer concise tool outputs and avoid unnecessary file reads."
            )

        self._last_inject_index = len(history)
        return [
            Attachment(
                type=_CONTEXT_BUDGET_TYPE,
                content=content,
                is_hint=not is_critical,  # hint at 60%, reminder at 80%
            )
        ]
```

### 2.2 Modify: `src/kimi_cli/loop/kimi_agent_loop.py` — Register provider

**BEFORE** (the import block, after P1 changes):
```python
from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
```

**AFTER**:
```python
from kimi_cli.loop.attachments.context_budget import ContextBudgetAttachmentProvider
from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
```

**BEFORE** (provider list, after P1 changes):
```python
        self._attachment_providers: list[AttachmentProvider] = [
            PreferShellRgAttachmentProvider(),
            GoalTrackingAttachmentProvider(),
        ]
```

**AFTER**:
```python
        self._attachment_providers: list[AttachmentProvider] = [
            PreferShellRgAttachmentProvider(),
            GoalTrackingAttachmentProvider(),
            ContextBudgetAttachmentProvider(),
        ]
```

### 2.3 New Test: `tests/core/test_context_budget_attachment.py`

```python
from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest
from llmkit.message import Message, TextPart

from kimi_cli.loop.attachments.context_budget import ContextBudgetAttachmentProvider


def _make_agent_loop_mock(*, context_usage: float) -> MagicMock:
    mock = MagicMock()
    type(mock)._context_usage = PropertyMock(return_value=context_usage)
    return mock


def _assistant_msg(text: str = "ok") -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _user_msg(text: str = "hello") -> Message:
    return Message(role="user", content=[TextPart(text=text)])


class TestContextBudgetAttachmentProvider:
    async def test_returns_empty_below_hint_threshold(self) -> None:
        provider = ContextBudgetAttachmentProvider()
        history = [_user_msg(), _assistant_msg()]

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.50)
        )

        assert result == []

    async def test_returns_hint_at_hint_threshold(self) -> None:
        provider = ContextBudgetAttachmentProvider()
        history = [_user_msg(), _assistant_msg()]

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.65)
        )

        assert len(result) == 1
        assert result[0].is_hint is True
        assert "65%" in result[0].content

    async def test_returns_reminder_at_reminder_threshold(self) -> None:
        provider = ContextBudgetAttachmentProvider()
        history = [_user_msg(), _assistant_msg()]

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.85)
        )

        assert len(result) == 1
        assert result[0].is_hint is False
        assert "85%" in result[0].content
        assert "Be concise" in result[0].content

    async def test_respects_cooldown(self) -> None:
        provider = ContextBudgetAttachmentProvider(cooldown=3)
        history = [_user_msg(), _assistant_msg()]

        # First call should inject
        result1 = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.70)
        )
        assert len(result1) == 1

        # Add fewer than cooldown assistant messages
        history.append(_assistant_msg())
        history.append(_assistant_msg())

        result2 = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.70)
        )
        assert result2 == []

        # Add enough to clear cooldown
        history.append(_assistant_msg())

        result3 = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.70)
        )
        assert len(result3) == 1

    async def test_custom_thresholds(self) -> None:
        provider = ContextBudgetAttachmentProvider(
            hint_threshold=0.40, reminder_threshold=0.70
        )
        history = [_user_msg()]

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(context_usage=0.45)
        )
        assert len(result) == 1
        assert result[0].is_hint is True
```

---

<a id="p3"></a>
## P3: Steer Instruction Deduplication (MEDIUM IMPACT)

### Goal
Avoid repeating the full ~100-token steer instruction text when multiple steers are sent in one turn.

### 3.1 Modify: `src/kimi_cli/loop/kimi_agent_loop.py`

**Change 1: Add `_turn_steer_count` tracking field to `__init__`.**

**BEFORE** (line 220-221):
```python
        self._steer_queue: asyncio.Queue[_QueuedSteer] = asyncio.Queue()
        self._active_turn_id: int | None = None
```

**AFTER**:
```python
        self._steer_queue: asyncio.Queue[_QueuedSteer] = asyncio.Queue()
        self._active_turn_id: int | None = None
        self._turn_steer_count: int = 0
```

**Change 2: Reset `_turn_steer_count` in `_begin_turn()`.**

**BEFORE** (lines 366-369):
```python
    def _begin_turn(self) -> int:
        self._next_turn_id += 1
        self._active_turn_id = self._next_turn_id
        return self._next_turn_id
```

**AFTER**:
```python
    def _begin_turn(self) -> int:
        self._next_turn_id += 1
        self._active_turn_id = self._next_turn_id
        self._turn_steer_count = 0
        return self._next_turn_id
```

**Change 3: Add abbreviated steer instruction method.**

**BEFORE** (lines 406-418, the `_steer_instruction_text` method):
```python
    @staticmethod
    def _steer_instruction_text() -> str:
        return (
            "The user sent a new reminder during the current turn. "
            "Treat it as an additional user instruction for this task. "
            "Incorporate it into the ongoing turn, but do not stop, summarize, or conclude "
            "the turn only because of this reminder. "
            "Use the reminder as hidden steering: do not explicitly acknowledge, answer, or "
            "quote the reminder by itself in the final response unless the original turn prompt "
            "directly asks for that. Do not use meta phrasing such as 'based on your reminder', "
            "'you just added', or 'you mentioned later'. Keep the final response centered on the "
            "user's original turn-opening request."
        )
```

**AFTER**:
```python
    @staticmethod
    def _steer_instruction_text() -> str:
        return (
            "The user sent a new reminder during the current turn. "
            "Treat it as an additional user instruction for this task. "
            "Incorporate it into the ongoing turn, but do not stop, summarize, or conclude "
            "the turn only because of this reminder. "
            "Use the reminder as hidden steering: do not explicitly acknowledge, answer, or "
            "quote the reminder by itself in the final response unless the original turn prompt "
            "directly asks for that. Do not use meta phrasing such as 'based on your reminder', "
            "'you just added', or 'you mentioned later'. Keep the final response centered on the "
            "user's original turn-opening request."
        )

    @staticmethod
    def _steer_instruction_text_brief() -> str:
        return "Additional user reminder (same handling rules as the previous reminder):"
```

**Change 4: Increment count and use abbreviated text in `_inject_steer`.**

**BEFORE** (lines 481-514, the `_inject_steer` method):
```python
    async def _inject_steer(
        self, content: str | list[ContentPart], *, is_skill: bool = False
    ) -> None:
        """Inject a single steer as a real-time reminder appended to user messages."""
        if is_skill:
            reminder_message = self._build_skill_steer_message(content)
        else:
            reminder_message = self._build_steer_message(content)
```

**AFTER**:
```python
    async def _inject_steer(
        self, content: str | list[ContentPart], *, is_skill: bool = False
    ) -> None:
        """Inject a single steer as a real-time reminder appended to user messages."""
        if is_skill:
            reminder_message = self._build_skill_steer_message(content)
        else:
            if self._turn_steer_count > 0:
                reminder_message = self._build_steer_message_impl(
                    content,
                    instruction=self._steer_instruction_text_brief(),
                    label="Reminder content",
                )
            else:
                reminder_message = self._build_steer_message(content)
            self._turn_steer_count += 1
```

The rest of the method (`if self._runtime.llm is None:` onwards) stays exactly the same. The only change is in the steer-message-building logic before the capability check.

### 3.2 Modify Existing Test: `tests/core/test_kimi_agent_loop_steer.py`

Add a new test function at the end of the file:

```python
@pytest.mark.asyncio
async def test_second_steer_uses_abbreviated_instruction(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    """The second steer in a turn should use abbreviated instruction text."""
    soul = KimiAgentLoop(
        Agent(
            name="Steer Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    turn_id = soul._begin_turn()
    try:
        soul.steer("first reminder")
        await soul._consume_pending_steers()

        soul.steer("second reminder")
        await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    # First steer gets full instruction
    first_text = soul.context.history[0].extract_text(" ")
    assert "additional user instruction" in first_text

    # Second steer gets abbreviated instruction
    second_text = soul.context.history[1].extract_text(" ")
    assert "same handling rules" in second_text
    assert "additional user instruction" not in second_text
    assert "second reminder" in second_text


@pytest.mark.asyncio
async def test_steer_count_resets_on_new_turn(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    """Steer count resets when a new turn begins, so the first steer gets full instruction."""
    soul = KimiAgentLoop(
        Agent(
            name="Steer Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    # First turn
    turn_id = soul._begin_turn()
    try:
        soul.steer("turn1 steer")
        await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    # Second turn
    turn_id2 = soul._begin_turn()
    try:
        soul.steer("turn2 first steer")
        await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id2)

    # Both first steers of each turn should have the full instruction
    first_text = soul.context.history[0].extract_text(" ")
    assert "additional user instruction" in first_text

    second_text = soul.context.history[1].extract_text(" ")
    assert "additional user instruction" in second_text
```

---

<a id="p4"></a>
## P4: Tool Result Tag Sanitization (LOW-MEDIUM IMPACT)

### Goal
Prevent the LLM from interpreting `<system-reminder>` or `<system-hint>` tags inside tool results (e.g., when the agent reads its own source code) as actual directives.

### 4.1 Modify: `src/kimi_cli/loop/message.py`

**BEFORE** (lines 63-75):
```python
def _output_to_content_parts(
    output: str | ContentPart | Sequence[ContentPart],
) -> list[ContentPart]:
    content: list[ContentPart] = []
    match output:
        case str(text):
            if text:
                content.append(TextPart(text=text))
        case ContentPart():
            content.append(output)
        case _:
            content.extend(output)
    return content
```

**AFTER**:
```python
def _sanitize_tool_output_tags(text: str) -> str:
    """Escape system directive tags in tool output to prevent false interpretation."""
    text = text.replace("<system-reminder>", "‹system-reminder›")
    text = text.replace("</system-reminder>", "‹/system-reminder›")
    text = text.replace("<system-hint>", "‹system-hint›")
    text = text.replace("</system-hint>", "‹/system-hint›")
    return text


def _output_to_content_parts(
    output: str | ContentPart | Sequence[ContentPart],
) -> list[ContentPart]:
    content: list[ContentPart] = []
    match output:
        case str(text):
            if text:
                content.append(TextPart(text=_sanitize_tool_output_tags(text)))
        case ContentPart():
            if isinstance(output, TextPart):
                content.append(
                    TextPart(text=_sanitize_tool_output_tags(output.text))
                )
            else:
                content.append(output)
        case _:
            for part in output:
                if isinstance(part, TextPart):
                    content.append(
                        TextPart(text=_sanitize_tool_output_tags(part.text))
                    )
                else:
                    content.append(part)
    return content
```

**Design Note**: We use Unicode angle brackets `‹` (U+2039) and `›` (U+203A) as replacements. These are visually similar to `<`/`>` so the agent can still understand the code it's reading, but they will NOT be parsed as XML-style directive tags by the model. This is preferable to `[system-reminder]` because it preserves the visual structure of source code more faithfully.

### 4.2 Modify Existing Test: `tests/core/test_loop_message.py`

Add these tests at the end of the file:

```python
def test_tool_ok_sanitizes_system_reminder_tags_in_string_output():
    """system-reminder tags in tool string output should be escaped."""
    code = 'text = "<system-reminder>\\nDo something\\n</system-reminder>"'
    tool_ok = ToolOk(output=code)
    tool_result = ToolResult(tool_call_id="call_san", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    result_text = message.content[0].text
    assert "<system-reminder>" not in result_text
    assert "‹system-reminder›" in result_text
    assert "‹/system-reminder›" in result_text


def test_tool_ok_sanitizes_system_hint_tags_in_string_output():
    """system-hint tags in tool string output should be escaped."""
    code = '<system-hint>Prefer rg</system-hint>'
    tool_ok = ToolOk(output=code)
    tool_result = ToolResult(tool_call_id="call_san2", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    result_text = message.content[0].text
    assert "<system-hint>" not in result_text
    assert "‹system-hint›" in result_text


def test_tool_ok_sanitizes_tags_in_text_part_output():
    """system-reminder tags in TextPart tool output should be escaped."""
    text_part = TextPart(text='content with <system-reminder>directive</system-reminder>')
    tool_ok = ToolOk(output=text_part)
    tool_result = ToolResult(tool_call_id="call_san3", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    result_text = message.content[0].text
    assert "<system-reminder>" not in result_text
    assert "‹system-reminder›" in result_text


def test_tool_ok_sanitizes_tags_in_sequence_output():
    """system-reminder tags in sequence of TextPart tool output should be escaped."""
    parts = [
        TextPart(text='first <system-reminder>x</system-reminder>'),
        TextPart(text='second <system-hint>y</system-hint>'),
    ]
    tool_ok = ToolOk(output=parts)
    tool_result = ToolResult(tool_call_id="call_san4", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    for part in message.content:
        if isinstance(part, TextPart):
            assert "<system-reminder>" not in part.text
            assert "<system-hint>" not in part.text


def test_tool_ok_does_not_sanitize_non_text_parts():
    """Non-text parts should pass through unchanged."""
    image_part = ImageURLPart(
        image_url=ImageURLPart.ImageURL(url="https://example.com/image.jpg")
    )
    tool_ok = ToolOk(output=[TextPart(text="<system-reminder>x</system-reminder>"), image_part])
    tool_result = ToolResult(tool_call_id="call_san5", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    assert message.content[-1] == image_part
```

---

<a id="p5"></a>
## P5: PostCompactionContinuityAttachmentProvider (MEDIUM IMPACT)

### Goal
After compaction, inject an authoritative `<system-reminder>` directing the agent to review the compaction summary and continue working without prompting the user.

### 5.1 New File: `src/kimi_cli/loop/attachments/post_compaction.py`

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmkit.message import Message

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.attachment import Attachment, AttachmentProvider

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


_COMPACTION_MARKER = "Previous context has been compacted"
_CONTINUITY_TYPE = "post_compaction_continuity"
_CONTINUITY_CONTENT = (
    "Context was just compacted. "
    "Review the compaction summary and your todo list above carefully. "
    "Continue working on the current task without asking the user to repeat instructions. "
    "If you need specific details from before compaction, use the RecallCompactedContext tool "
    "with targeted keywords."
)


class PostCompactionContinuityAttachmentProvider(AttachmentProvider):
    """Injects a continuity reminder once after each compaction event.

    Detects compaction by looking for the compaction marker in the first message.
    Fires exactly once per compaction by tracking whether it has already injected
    since the last compaction was detected.
    """

    def __init__(self) -> None:
        self._fired_for_history_len: int = -1

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        if not history:
            return []

        # Check if history[0] is a compaction summary
        first_msg = history[0]
        if not _is_compaction_summary(first_msg):
            return []

        # Have we already fired for this compaction?
        # After compaction, history is short (compacted messages + preserved).
        # We fire once — when the first assistant message hasn't been produced yet.
        # Track by the history length at the time of injection.
        history_len = len(history)
        if self._fired_for_history_len >= history_len:
            return []

        # Only fire if no assistant messages exist yet after compaction
        has_post_compaction_assistant = any(
            msg.role == "assistant" for msg in history
        )
        if has_post_compaction_assistant:
            return []

        self._fired_for_history_len = history_len
        return [
            Attachment(
                type=_CONTINUITY_TYPE,
                content=_CONTINUITY_CONTENT,
                is_hint=False,  # system-reminder (authoritative)
            )
        ]


def _is_compaction_summary(message: Message) -> bool:
    """Check if a message is a compaction summary."""
    for part in message.content:
        if isinstance(part, TextPart) and _COMPACTION_MARKER in part.text:
            return True
    return False
```

### 5.2 Modify: `src/kimi_cli/loop/kimi_agent_loop.py` — Register provider

**BEFORE** (imports, after P1+P2 changes):
```python
from kimi_cli.loop.attachments.context_budget import ContextBudgetAttachmentProvider
from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
```

**AFTER**:
```python
from kimi_cli.loop.attachments.context_budget import ContextBudgetAttachmentProvider
from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from kimi_cli.loop.attachments.post_compaction import PostCompactionContinuityAttachmentProvider
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
```

**BEFORE** (provider list, after P1+P2 changes):
```python
        self._attachment_providers: list[AttachmentProvider] = [
            PreferShellRgAttachmentProvider(),
            GoalTrackingAttachmentProvider(),
            ContextBudgetAttachmentProvider(),
        ]
```

**AFTER**:
```python
        self._attachment_providers: list[AttachmentProvider] = [
            PreferShellRgAttachmentProvider(),
            GoalTrackingAttachmentProvider(),
            ContextBudgetAttachmentProvider(),
            PostCompactionContinuityAttachmentProvider(),
        ]
```

### 5.3 New Test: `tests/core/test_post_compaction_attachment.py`

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from llmkit.message import Message, TextPart

from kimi_cli.loop.attachments.post_compaction import (
    PostCompactionContinuityAttachmentProvider,
)
from kimi_cli.loop.message import internal_user_message, system


def _make_agent_loop_mock() -> MagicMock:
    return MagicMock()


def _compaction_summary_msg() -> Message:
    return internal_user_message(
        [system("Previous context has been compacted. Here is the compaction output:"),
         TextPart(text="Summary of prior context...")]
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

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert result[0].type == "post_compaction_continuity"
        assert "compacted" in result[0].content
        assert "RecallCompactedContext" in result[0].content
        assert result[0].is_hint is False

    async def test_fires_only_once_per_compaction(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()
        history = [
            _compaction_summary_msg(),
            _user_msg("preserved"),
        ]

        result1 = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result1) == 1

        # Same history length — should not fire again
        result2 = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result2 == []

    async def test_does_not_fire_if_assistant_already_responded(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()
        history = [
            _compaction_summary_msg(),
            _assistant_msg("I will continue..."),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_returns_empty_for_non_compacted_history(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()
        history = [_user_msg("normal message")]

        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_returns_empty_for_empty_history(self) -> None:
        provider = PostCompactionContinuityAttachmentProvider()

        result = await provider.get_attachments([], _make_agent_loop_mock())
        assert result == []
```

---

## Summary of ALL Changes (Final State)

### New Files (3)

| File | Description |
|---|---|
| `src/kimi_cli/loop/attachments/goal_tracking.py` | P1: Goal drift prevention |
| `src/kimi_cli/loop/attachments/context_budget.py` | P2: Context budget warnings |
| `src/kimi_cli/loop/attachments/post_compaction.py` | P5: Post-compaction continuity |

### Modified Files (2)

| File | Changes |
|---|---|
| `src/kimi_cli/loop/kimi_agent_loop.py` | P1/P2/P5: 3 new imports + expanded provider list; P3: `_turn_steer_count` field, reset in `_begin_turn`, `_steer_instruction_text_brief` method, conditional logic in `_inject_steer` |
| `src/kimi_cli/loop/message.py` | P4: `_sanitize_tool_output_tags` function + updated `_output_to_content_parts` |

### New Test Files (3)

| File | Description |
|---|---|
| `tests/core/test_goal_tracking_attachment.py` | 7 test cases for P1 |
| `tests/core/test_context_budget_attachment.py` | 5 test cases for P2 |
| `tests/core/test_post_compaction_attachment.py` | 5 test cases for P5 |

### Modified Test Files (2)

| File | Changes |
|---|---|
| `tests/core/test_kimi_agent_loop_steer.py` | P3: 2 new test cases for steer dedup + reset |
| `tests/core/test_loop_message.py` | P4: 6 new test cases for tag sanitization |

### Final State of `kimi_agent_loop.py` Imports (lines 64-65 area)

```python
from kimi_cli.loop.attachments.context_budget import ContextBudgetAttachmentProvider
from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from kimi_cli.loop.attachments.post_compaction import PostCompactionContinuityAttachmentProvider
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
```

### Final State of Provider Registration (line 223 area)

```python
        self._attachment_providers: list[AttachmentProvider] = [
            PreferShellRgAttachmentProvider(),
            GoalTrackingAttachmentProvider(),
            ContextBudgetAttachmentProvider(),
            PostCompactionContinuityAttachmentProvider(),
        ]
```

### Execution Order for the Implementor

1. **P4 first** (simplest, no new files, isolated change to `message.py`)
2. **P3 second** (modify only `kimi_agent_loop.py`, no new files)
3. **P1 third** (new file + import/registration)
4. **P2 fourth** (new file + import/registration)
5. **P5 fifth** (new file + import/registration)

This order minimizes merge conflicts since P4 and P3 touch different files/methods, and P1/P2/P5 all touch the same two locations in `kimi_agent_loop.py` (imports + provider list) which should be done together.