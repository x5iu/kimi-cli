from __future__ import annotations

import pytest
from rich.spinner import Spinner

from kimi_cli.eventbus.types import (
    CompactionBegin,
    StatusUpdate,
    StepBegin,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolResult,
    TurnBegin,
    TurnEnd,
)
from kimi_cli.ui.shell.visualize import LiveView
from llmkit.tooling import ToolOk


def _spinners(blocks: list[object]) -> list[Spinner]:
    return [b for b in blocks if isinstance(b, Spinner)]


def test_empty_think_part_creates_thinking_indicator() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(ThinkPart(think=""))
    assert view._current_content_block is not None
    assert view._current_content_block.is_think is True
    assert "Thinking" in view.render_ansi(100)


def test_empty_text_part_still_skipped() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(TextPart(text=""))
    assert view._current_content_block is None


def test_empty_think_then_real_think_no_console_print(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.ui.shell.visualize import _live_view as lv_mod

    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    printed: list[object] = []
    monkeypatch.setattr(lv_mod.console, "print", lambda *a, **k: printed.append(a))
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(ThinkPart(think=""))
    view.dispatch_wire_message(ThinkPart(think="Let me analyze this..."))
    assert view._current_content_block is not None
    assert view._current_content_block.is_think is True
    assert view._current_content_block.raw_text == "Let me analyze this..."
    assert len(printed) == 0


def test_spinner_after_turn_begin() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) >= 1


def test_no_extra_spinner_when_text_visible() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(TextPart(text="Hello"))
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) == 1


def test_spinner_after_tool_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.ui.shell.visualize import _live_view as lv_mod

    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    monkeypatch.setattr(lv_mod.console, "print", lambda *a, **k: None)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(TextPart(text="Let me check."))
    view.dispatch_wire_message(
        ToolCall(id="call_1", function=ToolCall.FunctionBody(name="Shell", arguments="{}"))
    )
    view.dispatch_wire_message(ToolResult(tool_call_id="call_1", return_value=ToolOk(output="ok")))
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) == 1


def test_parallel_tool_hides_moon_until_last_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.ui.shell.visualize import _live_view as lv_mod

    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    monkeypatch.setattr(lv_mod.console, "print", lambda *a, **k: None)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(TextPart(text="Running two tools."))
    view.dispatch_wire_message(
        ToolCall(id="c1", function=ToolCall.FunctionBody(name="Shell", arguments="{}"))
    )
    view.dispatch_wire_message(
        ToolCall(id="c2", function=ToolCall.FunctionBody(name="Shell", arguments="{}"))
    )
    view.dispatch_wire_message(ToolResult(tool_call_id="c1", return_value=ToolOk(output="ok")))
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) == 0


def test_status_update_does_not_clear_spinner_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.ui.shell.visualize import _live_view as lv_mod

    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    monkeypatch.setattr(lv_mod.console, "print", lambda *a, **k: None)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(TextPart(text="Checking."))
    view.dispatch_wire_message(
        ToolCall(id="c1", function=ToolCall.FunctionBody(name="Shell", arguments="{}"))
    )
    view.dispatch_wire_message(ToolResult(tool_call_id="c1", return_value=ToolOk(output="ok")))
    view.dispatch_wire_message(StatusUpdate(context_usage=0.1, context_tokens=1, max_context_tokens=100))
    assert view._active_turn_depth > 0
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) == 1


def test_turn_end_clears_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.ui.shell.visualize import _live_view as lv_mod

    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    monkeypatch.setattr(lv_mod.console, "print", lambda *a, **k: None)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(TextPart(text="Done."))
    view.dispatch_wire_message(TurnEnd())
    assert view._active_turn_depth == 0


def test_compaction_over_moon_fallback() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    view.dispatch_wire_message(CompactionBegin())
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) == 1
    assert view._compacting_spinner is not None


def test_interrupt_resets_active_depth() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    assert view._active_turn_depth > 0
    view.cleanup(is_interrupt=True)
    assert view._active_turn_depth == 0


def test_step_cleanup_keeps_depth() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="test"))
    assert view._active_turn_depth > 0
    view.cleanup(is_interrupt=False)
    assert view._active_turn_depth > 0


def test_nested_turn_end() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnBegin(user_input="outer"))
    assert view._active_turn_depth == 1
    view.dispatch_wire_message(TurnBegin(user_input="inner"))
    assert view._active_turn_depth == 2
    view.dispatch_wire_message(StepBegin(n=1))
    view.dispatch_wire_message(TurnEnd())
    assert view._active_turn_depth == 1
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) >= 1
    view.dispatch_wire_message(TurnEnd())
    assert view._active_turn_depth == 0


def test_turn_end_clamps_depth() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.dispatch_wire_message(TurnEnd())
    view.dispatch_wire_message(TurnEnd())
    assert view._active_turn_depth == 0


def test_step_begin_without_turn_begin_sets_depth() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    assert view._active_turn_depth == 0
    view.dispatch_wire_message(StepBegin(n=1))
    assert view._active_turn_depth == 1
    blocks, _ = view._active_blocks(include_running_indicators=True)
    assert len(_spinners(blocks)) >= 1
