from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from io import StringIO
from typing import Any, cast

from kosong.tooling import ToolError, ToolOk
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from kimi_cli.ui.shell.blocks import _ContentBlock, _StatusBlock, _ToolCallBlock
from kimi_cli.ui.shell.console import _RIGHT_PADDING, console
from kimi_cli.ui.shell.keyboard import KeyEvent
from kimi_cli.ui.shell.panels import (
    _ApprovalRequestPanel,
    _QuestionRequestPanel,
    _show_approval_in_pager,
    _show_question_body_in_pager,
)
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.utils.logging import logger
from kimi_cli.wire import WireUISide
from kimi_cli.wire.types import (
    ApprovalRequest,
    ApprovalResponse,
    CompactionBegin,
    CompactionEnd,
    ContentPart,
    FollowUpInput,
    MCPLoadingBegin,
    MCPLoadingEnd,
    QuestionRequest,
    SkillReminderNotice,
    StatusUpdate,
    StepBegin,
    StepInterrupted,
    SubagentEvent,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
    ToolCallRequest,
    ToolResult,
    TurnBegin,
    TurnEnd,
    WireMessage,
)

MAX_SUBAGENT_TOOL_CALLS_TO_SHOW = 4
MAX_TOOL_ERROR_OUTPUT_LINES = 12
MAX_TOOL_ERROR_OUTPUT_CHARS = 4000
MAX_ACTIVE_TURN_FLUSHED_BLOCKS = 12
MAX_ACTIVE_TURN_PENDING_INPUT_BLOCKS = 3
MAX_ACTIVE_TURN_CONTENT_CHARS = 12_000
LIVE_VIEW_REFRESH_INTERVAL = 0.2


def is_significant_for_render(msg: object) -> bool:
    """Whether a wire message should trigger an immediate repaint."""
    if isinstance(msg, (ContentPart, ToolCallPart, StatusUpdate, ApprovalResponse)):
        return False
    if isinstance(msg, SubagentEvent):
        return is_significant_for_render(msg.event)
    return True


async def visualize(
    wire: WireUISide,
    *,
    initial_status: StatusUpdate,
    cancel_event: asyncio.Event | None = None,
    live_view: LiveView | None = None,
):
    """
    A loop to consume agent events and visualize the agent behavior.

    Args:
        wire: Communication channel with the agent
        initial_status: Initial status snapshot
        cancel_event: Event that can be set (e.g., by ESC key) to cancel the run
    """
    view = live_view or LiveView(initial_status, cancel_event)
    await view.visualize_loop(wire)


def _render_prompt_block(
    text: str,
    *,
    title: str,
    border_style: str,
) -> RenderableType:
    return Panel(
        Text(text, overflow="fold"),
        box=box.ROUNDED,
        border_style=border_style,
        title=Text(title, style=f"bold {border_style}"),
        title_align="left",
        padding=(0, 1),
        expand=False,
    )


def render_user_prompt_block(text: str) -> RenderableType:
    return _render_prompt_block(text, title="User", border_style="blue")


def _render_reminder_block(text: str) -> RenderableType:
    return _render_prompt_block(text, title="Reminder", border_style="cyan")


def _render_skill_reminder_block(skills: Sequence[str]) -> RenderableType:
    return _render_prompt_block(
        f"Recommended {', '.join(skills)} to the main flow",
        title="Reminder",
        border_style="cyan",
    )


def _recent_output_notice_text() -> str:
    return (
        "… showing recent output only during live turn; "
        "press Ctrl-Y to show full history"
    )


def _render_recent_output_notice() -> RenderableType:
    return Text(_recent_output_notice_text(), style="grey50 italic")


class LiveView:
    def __init__(
        self,
        initial_status: StatusUpdate,
        cancel_event: asyncio.Event | None = None,
        *,
        flush_to_console: bool = True,
        allow_expand: bool = True,
    ):
        self._cancel_event = cancel_event

        self._turn_spinner: Spinner | None = None
        self._mooning_spinner: Spinner | None = None
        self._compacting_spinner: Spinner | None = None
        self._mcp_loading_spinner: Spinner | None = None

        self._current_content_block: _ContentBlock | None = None
        self._tool_call_blocks: dict[str, _ToolCallBlock] = {}
        self._last_tool_call_block: _ToolCallBlock | None = None
        self._approval_request_queue = deque[ApprovalRequest]()
        """
        It is possible that multiple subagents request approvals at the same time,
        in which case we will have to queue them up and show them one by one.
        """
        self._current_approval_request_panel: _ApprovalRequestPanel | None = None
        self._reject_all_following = False
        self._question_request_queue = deque[QuestionRequest]()
        self._current_question_panel: _QuestionRequestPanel | None = None
        self._question_waiting_for_other_text = False
        self._status_block = _StatusBlock(initial_status)
        self._live: Live | None = None
        self._flush_to_console = flush_to_console
        self._allow_expand = allow_expand
        self._flushed_blocks: list[RenderableType] = []
        self._reminder_blocks: list[RenderableType] = []

        self._need_recompose = False
        self._render_revision = 0
        self._history_revision = 0
        self._active_revision = 0

    def _reset_live_shape(self, live: Live) -> None:
        # Rich doesn't expose a public API to clear Live's cached render height.
        # After leaving the pager, stale height causes cursor restores to jump,
        # so we reset the private _shape to re-anchor the next refresh.
        cast(Any, live)._live_render._shape = None

    async def visualize_loop(self, wire: WireUISide):
        with Live(
            self.compose(),
            console=console,
            auto_refresh=False,
            transient=True,
            vertical_overflow="visible",
        ) as live:
            self._live = live

            async def _animate() -> None:
                while True:
                    await asyncio.sleep(LIVE_VIEW_REFRESH_INTERVAL)
                    if self.needs_periodic_refresh or self._need_recompose:
                        live.update(self.compose(), refresh=True)
                        self._need_recompose = False

            animate_task = asyncio.create_task(_animate())
            try:
                while True:
                    try:
                        msg = await wire.receive()
                    except QueueShutDown:
                        self.cleanup(is_interrupt=False)
                        self.finish_turn()
                        live.update(self.compose(), refresh=True)
                        break

                    if isinstance(msg, StepInterrupted):
                        self.cleanup(is_interrupt=True)
                        self.finish_turn()
                        live.update(self.compose(), refresh=True)
                        break

                    self.dispatch_wire_message(msg)
                    if self._need_recompose and (
                        is_significant_for_render(msg) or not self.needs_periodic_refresh
                    ):
                        live.update(self.compose(), refresh=True)
                        self._need_recompose = False
            finally:
                animate_task.cancel()
                with suppress(asyncio.CancelledError):
                    await animate_task
                self._live = None

    @property
    def render_revision(self) -> int:
        return self._render_revision

    def refresh_soon(self) -> None:
        self.refresh_all()

    @property
    def history_revision(self) -> int:
        return self._history_revision

    @property
    def active_revision(self) -> int:
        return self._active_revision

    def refresh_history(self) -> None:
        self._need_recompose = True
        self._history_revision += 1
        self._render_revision += 1

    def refresh_active(self) -> None:
        self._need_recompose = True
        self._active_revision += 1
        self._render_revision += 1

    def refresh_all(self) -> None:
        self._need_recompose = True
        self._history_revision += 1
        self._active_revision += 1
        self._render_revision += 1

    def echo_reminder(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        self.flush_content()
        reminder = _render_reminder_block(stripped)
        self._reminder_blocks.append(reminder)
        if self._flush_to_console:
            console.print(reminder)
        else:
            self._flushed_blocks.append(reminder)
        self.refresh_history()

    def echo_user_choice(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        self.flush_content()
        block = render_user_prompt_block(stripped)
        if self._flush_to_console:
            console.print(block)
        else:
            self._flushed_blocks.append(block)
        self.refresh_history()

    def finish_turn(self) -> None:
        if self._turn_spinner is None:
            return
        self._turn_spinner = None
        self.refresh_active()

    def append_skill_reminder(self, skills: Sequence[str]) -> None:
        if not skills:
            return
        self._flushed_blocks.append(_render_skill_reminder_block(skills))
        self.refresh_history()

    @property
    def needs_periodic_refresh(self) -> bool:
        if self.has_pending_input_request:
            return False
        if self._turn_spinner is not None:
            return True
        if self._mcp_loading_spinner is not None:
            return True
        if self._mooning_spinner is not None:
            return True
        if self._compacting_spinner is not None:
            return True
        if self._current_content_block is not None:
            return True
        return any(not block.finished for block in self._tool_call_blocks.values())

    def _renderable_to_ansi(self, renderable: RenderableType, width: int) -> str:
        width = max(20, width - _RIGHT_PADDING)
        sio = StringIO()
        render_console = Console(
            file=sio,
            force_terminal=True,
            width=width,
            color_system="truecolor",
            highlight=False,
        )
        render_console.print(renderable, end="")
        return sio.getvalue()

    def render_ansi(
        self,
        width: int,
        *,
        include_status: bool = False,
        include_running_indicators: bool = True,
    ) -> str:
        return self._renderable_to_ansi(
            self.compose(
                include_status=include_status,
                include_running_indicators=include_running_indicators,
            ),
            width,
        )

    @property
    def has_reminders(self) -> bool:
        return bool(self._reminder_blocks)

    def compose_reminders(self) -> RenderableType:
        return Group(*self._reminder_blocks)

    def render_reminders_ansi(self, width: int) -> str:
        if not self._reminder_blocks:
            return ""
        return self._renderable_to_ansi(self.compose_reminders(), width)

    @property
    def has_pending_input_request(self) -> bool:
        return (
            self._current_approval_request_panel is not None
            or self._current_question_panel is not None
        )

    @property
    def input_mode(self) -> str:
        if self._current_approval_request_panel is not None:
            return "approval"
        if self._current_question_panel is not None:
            return "question_other" if self._question_waiting_for_other_text else "question"
        return "reminder"

    @property
    def can_expand_current_panel(self) -> bool:
        if not self._allow_expand:
            return False
        if self._current_approval_request_panel is not None:
            return self._current_approval_request_panel.has_expandable_content
        if self._current_question_panel is not None:
            return self._current_question_panel.has_expandable_content
        return False

    @property
    def input_hint(self) -> str:
        expand_hint = (
            " Press Ctrl-E or type /more to expand." if self.can_expand_current_panel else ""
        )
        match self.input_mode:
            case "approval":
                return f"Type 1/2/3 and press Enter.{expand_hint}"
            case "question_other":
                return "Enter the custom answer, then press Enter."
            case "question":
                panel = self._current_question_panel
                if panel is not None and panel.is_multi_select:
                    return (
                        "Use ↑/↓ to focus, Space or Enter to select, and Enter to submit. "
                        f"Select Other to type custom text.{expand_hint}"
                    )
                return (
                    "Use ↑/↓ to focus and Enter to choose. "
                    f"Select Other to type custom text.{expand_hint}"
                )
            case _:
                return "Turn is running. Type a message and press Enter to send a reminder."

    @property
    def activity_indicator(self) -> tuple[str, str] | None:
        if self._current_approval_request_panel is not None:
            return ("approval", "Awaiting approval...")
        if self._current_question_panel is not None:
            return ("question", "Awaiting answer...")
        if self._mcp_loading_spinner is not None:
            return ("mcp", "Connecting to MCP servers...")
        if self._mooning_spinner is not None:
            return ("moon", "Running...")
        if self._compacting_spinner is not None:
            return ("compacting", "Compacting...")
        if self._current_content_block is not None:
            return (
                "thinking" if self._current_content_block.is_think else "composing",
                self._current_content_block.status_text,
            )
        for tool_call in reversed(tuple(self._tool_call_blocks.values())):
            if not tool_call.finished:
                return ("tool", tool_call.status_text)
        if self._turn_spinner is not None:
            return ("running", "Running...")
        return None

    @staticmethod
    def is_expand_command(text: str) -> bool:
        return text.strip().casefold() in {"/more", "more", "/expand", "expand"}

    def show_more(self) -> bool:
        if not self.can_expand_current_panel:
            return False
        live = self._live
        if (
            self._current_approval_request_panel is not None
            and self._current_approval_request_panel.has_expandable_content
        ):
            if live is not None:
                live.stop()
            try:
                _show_approval_in_pager(self._current_approval_request_panel)
            finally:
                if live is not None:
                    self._reset_live_shape(live)
                    live.start()
                    live.update(self.compose(), refresh=True)
            return True
        if (
            self._current_question_panel is not None
            and self._current_question_panel.has_expandable_content
        ):
            if live is not None:
                live.stop()
            try:
                _show_question_body_in_pager(self._current_question_panel)
            finally:
                if live is not None:
                    self._reset_live_shape(live)
                    live.start()
                    live.update(self.compose(), refresh=True)
            return True
        return False

    def try_submit_line(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if self.is_expand_command(stripped):
            return self.show_more()
        if self._current_approval_request_panel is not None:
            return self._submit_approval_line(stripped)
        if self._current_question_panel is not None:
            return self._submit_question_line(stripped)
        return False

    @staticmethod
    def _parse_index_token(text: str) -> int | None:
        if not text.isdigit():
            return None
        return int(text) - 1

    @staticmethod
    def _find_question_option_index(panel: _QuestionRequestPanel, text: str) -> int | None:
        normalized = text.casefold()
        for i, (label, _) in enumerate(panel.options):
            if label.casefold() == normalized:
                return i
        return None

    def _submit_approval_line(self, text: str) -> bool:
        panel = self._current_approval_request_panel
        if panel is None:
            return False

        normalized = text.casefold()
        selected_index = {
            "1": 0,
            "approve": 0,
            "yes": 0,
            "2": 1,
            "session": 1,
            "always": 1,
            "approve_for_session": 1,
            "3": 2,
            "reject": 2,
            "no": 2,
        }.get(normalized)
        if selected_index is None:
            return False

        panel.selected_index = selected_index
        self._submit_approval()
        self.refresh_active()
        return True

    def _resolve_question_submission(
        self,
        panel: _QuestionRequestPanel,
        *,
        all_done: bool,
    ) -> None:
        self._question_waiting_for_other_text = False
        if all_done:
            panel.request.resolve(panel.get_answers())
            self.show_next_question_request()
        self.refresh_active()

    def _submit_question_line(self, text: str) -> bool:
        panel = self._current_question_panel
        if panel is None:
            return False

        if self._question_waiting_for_other_text:
            all_done = panel.submit_other(text)
            self._resolve_question_submission(panel, all_done=all_done)
            return True

        if panel.is_multi_select:
            return self._submit_multi_select_question_line(panel, text)
        return self._submit_single_select_question_line(panel, text)

    def _submit_single_select_question_line(self, panel: _QuestionRequestPanel, text: str) -> bool:
        idx = self._parse_index_token(text)
        if idx is None:
            idx = self._find_question_option_index(panel, text)

        if idx is None:
            all_done = panel.submit_other(text)
            self._resolve_question_submission(panel, all_done=all_done)
            return True
        if not panel.select_index(idx):
            return False
        if panel.is_other_selected:
            self._question_waiting_for_other_text = True
            self.refresh_active()
            return True

        all_done = panel.submit()
        self._resolve_question_submission(panel, all_done=all_done)
        return True

    def _submit_multi_select_question_line(self, panel: _QuestionRequestPanel, text: str) -> bool:
        tokens = [token.strip() for token in text.replace("\n", ",").split(",") if token.strip()]
        if not tokens:
            return False

        other_idx = panel.option_count - 1
        selected_indices: set[int] = set()
        custom_tokens: list[str] = []
        wants_other = False

        for token in tokens:
            idx = self._parse_index_token(token)
            if idx is None:
                idx = self._find_question_option_index(panel, token)
            if idx is None:
                custom_tokens.append(token)
            elif not (0 <= idx < panel.option_count):
                return False
            elif idx == other_idx:
                wants_other = True
            else:
                selected_indices.add(idx)

        if custom_tokens:
            panel.set_multi_selected(set(selected_indices))
            all_done = panel.submit_other(", ".join(custom_tokens))
            self._resolve_question_submission(panel, all_done=all_done)
            return True

        if wants_other:
            panel.selected_index = other_idx
            panel.set_multi_selected(set(selected_indices) | {other_idx})
            self._question_waiting_for_other_text = True
            self.refresh_active()
            return True

        if not selected_indices:
            return False

        panel.set_multi_selected(set(selected_indices))
        all_done = panel.submit()
        self._resolve_question_submission(panel, all_done=all_done)
        return True

    def _history_blocks(
        self,
        *,
        tail_block_limit: int | None = None,
    ) -> tuple[list[RenderableType], bool]:
        if tail_block_limit is not None:
            if tail_block_limit <= 0 and self._flushed_blocks:
                return [], True
            if len(self._flushed_blocks) > tail_block_limit:
                return list(self._flushed_blocks[-tail_block_limit:]), True
        return list(self._flushed_blocks), False

    def _active_blocks(
        self,
        *,
        include_running_indicators: bool = True,
        content_char_limit: int | None = None,
        focus_pending_input_panel: bool | None = None,
    ) -> tuple[list[RenderableType], bool]:
        blocks: list[RenderableType] = []
        truncated = False
        if focus_pending_input_panel is None:
            focus_pending_input_panel = self.has_pending_input_request
        has_specific_running_indicator = False

        if focus_pending_input_panel:
            truncated = any(
                block is not None
                for block in (
                    self._mcp_loading_spinner,
                    self._mooning_spinner,
                    self._compacting_spinner,
                    self._current_content_block,
                    self._turn_spinner,
                )
            ) or bool(self._tool_call_blocks)
        elif self._mcp_loading_spinner is not None:
            if include_running_indicators:
                blocks.append(self._mcp_loading_spinner)
                has_specific_running_indicator = True
        elif self._mooning_spinner is not None:
            if include_running_indicators:
                blocks.append(self._mooning_spinner)
                has_specific_running_indicator = True
        elif self._compacting_spinner is not None:
            if include_running_indicators:
                blocks.append(self._compacting_spinner)
                has_specific_running_indicator = True
        else:
            if self._current_content_block is not None:
                if content_char_limit is None:
                    blocks.append(
                        self._current_content_block.compose(
                            show_indicator=include_running_indicators,
                        )
                    )
                    has_specific_running_indicator = include_running_indicators
                else:
                    rendered, content_truncated = self._current_content_block.compose_tail(
                        max_chars=content_char_limit,
                        show_indicator=include_running_indicators,
                    )
                    blocks.append(rendered)
                    truncated = truncated or content_truncated
                    has_specific_running_indicator = include_running_indicators
            for tool_call in self._tool_call_blocks.values():
                blocks.append(tool_call.compose(show_indicator=include_running_indicators))
                if not tool_call.finished and include_running_indicators:
                    has_specific_running_indicator = True

        if (
            not focus_pending_input_panel
            and include_running_indicators
            and self._turn_spinner is not None
            and not has_specific_running_indicator
        ):
            blocks.append(self._turn_spinner)
        if focus_pending_input_panel and self._current_approval_request_panel:
            blocks.append(
                self._current_approval_request_panel.render(allow_expand=self._allow_expand)
            )
        if focus_pending_input_panel and self._current_question_panel:
            blocks.append(self._current_question_panel.render(allow_expand=self._allow_expand))
        return blocks, truncated

    def should_show_recent_output_notice(
        self,
        *,
        tail_block_limit: int | None = None,
        content_char_limit: int | None = None,
    ) -> bool:
        history_blocks, history_truncated = self._history_blocks(tail_block_limit=tail_block_limit)
        active_blocks, active_truncated = self._active_blocks(
            include_running_indicators=False,
            content_char_limit=content_char_limit,
        )
        if history_truncated or active_truncated:
            return True
        tail_mode_active = tail_block_limit is not None or content_char_limit is not None
        if not tail_mode_active:
            return False
        return bool(history_blocks or active_blocks)

    def compose_history_body(
        self,
        *,
        tail_block_limit: int | None = None,
    ) -> RenderableType | None:
        blocks, _ = self._history_blocks(tail_block_limit=tail_block_limit)
        if not blocks:
            return None
        return Group(*blocks)

    def compose_active_body(
        self,
        *,
        include_running_indicators: bool = True,
        content_char_limit: int | None = None,
        focus_pending_input_panel: bool | None = None,
    ) -> RenderableType | None:
        blocks, _ = self._active_blocks(
            include_running_indicators=include_running_indicators,
            content_char_limit=content_char_limit,
            focus_pending_input_panel=focus_pending_input_panel,
        )
        if not blocks:
            return None
        return Group(*blocks)

    def compose_body(
        self,
        *,
        include_running_indicators: bool = True,
        tail_block_limit: int | None = None,
        content_char_limit: int | None = None,
    ) -> RenderableType:
        blocks: list[RenderableType] = []
        if self.should_show_recent_output_notice(
            tail_block_limit=tail_block_limit,
            content_char_limit=content_char_limit,
        ):
            blocks.append(_render_recent_output_notice())
        history = self.compose_history_body(tail_block_limit=tail_block_limit)
        if history is not None:
            blocks.append(history)
        active = self.compose_active_body(
            include_running_indicators=include_running_indicators,
            content_char_limit=content_char_limit,
        )
        if active is not None:
            blocks.append(active)
        return Group(*blocks)

    def compose(
        self,
        *,
        include_status: bool = True,
        include_running_indicators: bool = True,
    ) -> RenderableType:
        """Compose the live view display content."""
        blocks: list[RenderableType] = [
            self.compose_body(
                include_running_indicators=include_running_indicators,
            )
        ]
        if include_status:
            blocks.append(self._status_block.render())
        return Group(*blocks)

    def dispatch_wire_message(self, msg: WireMessage) -> None:
        """Dispatch the Wire message to UI components."""
        assert not isinstance(msg, StepInterrupted)  # handled in visualize_loop

        if isinstance(msg, StepBegin):
            self.cleanup(is_interrupt=False)
            self._mcp_loading_spinner = None
            self._mooning_spinner = Spinner("moon", "")
            self.refresh_active()
            return

        if self._mooning_spinner is not None:
            # any message other than StepBegin should end the mooning state
            self._mooning_spinner = None
            self.refresh_active()

        match msg:
            case TurnBegin():
                self.flush_content()
                self._turn_spinner = Spinner("dots", "Running...")
                self.refresh_active()
            case TurnEnd():
                self.finish_turn()
            case FollowUpInput(text=text):
                self.echo_user_choice(text)
            case CompactionBegin():
                self._compacting_spinner = Spinner("balloon", "Compacting...")
                self.refresh_active()
            case CompactionEnd():
                self._compacting_spinner = None
                self.refresh_active()
            case MCPLoadingBegin():
                self._mcp_loading_spinner = Spinner("dots", "Connecting to MCP servers...")
                self.refresh_active()
            case MCPLoadingEnd():
                self._mcp_loading_spinner = None
                self.refresh_active()
            case SkillReminderNotice(skills=skills):
                self.append_skill_reminder(skills)
            case StatusUpdate():
                if self._status_block.update(msg):
                    self._need_recompose = True
            case ContentPart():
                self.append_content(msg)
            case ToolCall():
                self.append_tool_call(msg)
            case ToolCallPart():
                self.append_tool_call_part(msg)
            case ToolResult():
                self.append_tool_result(msg)
            case ApprovalResponse():
                # we don't need to handle this because the request is resolved on UI
                pass
            case SubagentEvent():
                self.handle_subagent_event(msg)
            case ApprovalRequest():
                self.request_approval(msg)
            case QuestionRequest():
                self.request_question(msg)
            case ToolCallRequest():
                logger.warning("Unexpected ToolCallRequest in shell UI: {msg}", msg=msg)

    def _try_submit_question(self) -> None:
        """Submit the current question answer; if all done, resolve and advance."""
        panel = self._current_question_panel
        if panel is None:
            return
        if panel.is_multi_select and panel.is_other_selected:
            panel.multi_selected.add(panel.option_count - 1)
        if panel.should_prompt_other_input():
            self._question_waiting_for_other_text = True
            self.refresh_active()
            return
        all_done = panel.submit()
        if all_done:
            panel.request.resolve(panel.get_answers())
            self.show_next_question_request()
        self.refresh_active()

    def dispatch_keyboard_event(self, event: KeyEvent) -> None:
        if event == KeyEvent.CTRL_E and self.can_expand_current_panel:
            if self.show_more():
                self.refresh_active()
            return

        # Handle question panel keyboard events
        if self._current_question_panel is not None:
            match event:
                case KeyEvent.UP:
                    self._current_question_panel.move_up()
                case KeyEvent.DOWN:
                    self._current_question_panel.move_down()
                case KeyEvent.LEFT:
                    self._current_question_panel.prev_tab()
                case KeyEvent.RIGHT | KeyEvent.TAB:
                    self._current_question_panel.next_tab()
                case KeyEvent.SPACE:
                    if self._current_question_panel.is_multi_select:
                        self._current_question_panel.toggle_select()
                    else:
                        self._try_submit_question()
                case KeyEvent.ENTER:
                    panel = self._current_question_panel
                    if panel.is_multi_select:
                        other_idx = panel.option_count - 1
                        if panel.selected_index == other_idx:
                            panel.multi_selected.add(other_idx)
                            self._try_submit_question()
                        elif panel.selected_index in panel.multi_selected:
                            self._try_submit_question()
                        else:
                            panel.toggle_select()
                    else:
                        # "Other" is handled in keyboard_handler (async context)
                        self._try_submit_question()
                case KeyEvent.ESCAPE:
                    self._current_question_panel.request.resolve({})
                    self.show_next_question_request()
                case (
                    KeyEvent.NUM_1
                    | KeyEvent.NUM_2
                    | KeyEvent.NUM_3
                    | KeyEvent.NUM_4
                    | KeyEvent.NUM_5
                ):
                    # Number keys select option in question panel
                    num_map = {
                        KeyEvent.NUM_1: 0,
                        KeyEvent.NUM_2: 1,
                        KeyEvent.NUM_3: 2,
                        KeyEvent.NUM_4: 3,
                        KeyEvent.NUM_5: 4,
                    }
                    idx = num_map[event]
                    panel = self._current_question_panel
                    if panel.select_index(idx):
                        if panel.is_multi_select:
                            panel.toggle_select()
                        elif not panel.is_other_selected:
                            # Auto-submit for single-select (unless "Other")
                            self._try_submit_question()
                case _:
                    pass
            self.refresh_active()
            return

        # handle ESC key to cancel the run
        if event == KeyEvent.ESCAPE and self._cancel_event is not None:
            self._cancel_event.set()
            return

        # Handle approval panel keyboard events
        if self._current_approval_request_panel is not None:
            match event:
                case KeyEvent.UP:
                    self._current_approval_request_panel.move_up()
                    self.refresh_active()
                case KeyEvent.DOWN:
                    self._current_approval_request_panel.move_down()
                    self.refresh_active()
                case KeyEvent.ENTER:
                    self._submit_approval()
                case KeyEvent.NUM_1 | KeyEvent.NUM_2 | KeyEvent.NUM_3:
                    # Number keys directly select and submit approval option
                    num_map = {
                        KeyEvent.NUM_1: 0,
                        KeyEvent.NUM_2: 1,
                        KeyEvent.NUM_3: 2,
                    }
                    idx = num_map[event]
                    if idx < len(self._current_approval_request_panel.options):
                        self._current_approval_request_panel.selected_index = idx
                        self._submit_approval()
                case _:
                    pass
            return

    def _submit_approval(self) -> None:
        """Submit the currently selected approval response."""
        assert self._current_approval_request_panel is not None
        resp = self._current_approval_request_panel.get_selected_response()
        self._current_approval_request_panel.request.resolve(resp)
        if resp == "approve_for_session":
            to_remove_from_queue: list[ApprovalRequest] = []
            for request in self._approval_request_queue:
                # approve all queued requests with the same action
                if request.action == self._current_approval_request_panel.request.action:
                    request.resolve("approve_for_session")
                    to_remove_from_queue.append(request)
            for request in to_remove_from_queue:
                self._approval_request_queue.remove(request)
        elif resp == "reject":
            # one rejection should stop the step immediately
            while self._approval_request_queue:
                self._approval_request_queue.popleft().resolve("reject")
            self._reject_all_following = True
        self.show_next_approval_request()

    def cleanup(self, is_interrupt: bool) -> None:
        """Cleanup the live view on step end or interruption."""
        self.flush_content()

        for block in self._tool_call_blocks.values():
            if not block.finished:
                # this should not happen, but just in case
                block.finish(
                    ToolError(message="", brief="Interrupted")
                    if is_interrupt
                    else ToolOk(output="")
                )
        self._last_tool_call_block = None
        self.flush_finished_tool_calls()

        while self._approval_request_queue:
            # should not happen, but just in case
            self._approval_request_queue.popleft().resolve("reject")
        if self._current_approval_request_panel is not None:
            self._current_approval_request_panel.request.resolve("reject")
            self._current_approval_request_panel = None
        self._reject_all_following = False

        while self._question_request_queue:
            self._question_request_queue.popleft().resolve({})
        if self._current_question_panel is not None:
            self._current_question_panel.request.resolve({})
            self._current_question_panel = None
        self._question_waiting_for_other_text = False

    def flush_content(self) -> None:
        """Flush the current content block."""
        if self._current_content_block is not None:
            rendered = self._current_content_block.compose_final()
            if self._flush_to_console:
                console.print(rendered)
            else:
                self._flushed_blocks.append(rendered)
            self._current_content_block = None
            self.refresh_all()

    def flush_finished_tool_calls(self) -> None:
        """Flush all leading finished tool call blocks."""
        tool_call_ids = list(self._tool_call_blocks.keys())
        for tool_call_id in tool_call_ids:
            block = self._tool_call_blocks[tool_call_id]
            if not block.finished:
                break

            self._tool_call_blocks.pop(tool_call_id)
            rendered = block.compose()
            if self._flush_to_console:
                console.print(rendered)
            else:
                self._flushed_blocks.append(rendered)
            if self._last_tool_call_block == block:
                self._last_tool_call_block = None
            self.refresh_all()

    def append_content(self, part: ContentPart) -> None:
        match part:
            case ThinkPart(think=text) | TextPart(text=text):
                if not text:
                    return
                is_think = isinstance(part, ThinkPart)
                if self._current_content_block is None:
                    self._current_content_block = _ContentBlock(is_think)
                elif self._current_content_block.is_think != is_think:
                    self.flush_content()
                    self._current_content_block = _ContentBlock(is_think)
                self._current_content_block.append(text)
                self.refresh_active()
            case _:
                # TODO: support more content part types
                pass

    def append_tool_call(self, tool_call: ToolCall) -> None:
        self.flush_content()
        self._tool_call_blocks[tool_call.id] = _ToolCallBlock(tool_call)
        self._last_tool_call_block = self._tool_call_blocks[tool_call.id]
        self.refresh_active()

    def append_tool_call_part(self, part: ToolCallPart) -> None:
        if not part.arguments_part:
            return
        if self._last_tool_call_block is None:
            return
        if self._last_tool_call_block.append_args_part(part.arguments_part):
            self.refresh_active()

    def append_tool_result(self, result: ToolResult) -> None:
        if block := self._tool_call_blocks.get(result.tool_call_id):
            block.finish(result.return_value)
            self.flush_finished_tool_calls()
            if result.tool_call_id in self._tool_call_blocks:
                self.refresh_active()

    def request_approval(self, request: ApprovalRequest) -> None:
        # If we're rejecting all following requests, reject immediately
        if self._reject_all_following:
            request.resolve("reject")
            return

        self._approval_request_queue.append(request)

        if self._current_approval_request_panel is None:
            console.bell()
            self.show_next_approval_request()

    def show_next_approval_request(self) -> None:
        """
        Show the next approval request from the queue.
        If there are no pending requests, clear the current approval panel.
        """
        if not self._approval_request_queue:
            if self._current_approval_request_panel is not None:
                self._current_approval_request_panel = None
                self.refresh_active()
            return

        while self._approval_request_queue:
            request = self._approval_request_queue.popleft()
            if request.resolved:
                # skip resolved requests
                continue
            self._current_approval_request_panel = _ApprovalRequestPanel(request)
            self.refresh_active()
            break
        else:
            # All queued requests were already resolved
            if self._current_approval_request_panel is not None:
                self._current_approval_request_panel = None
                self.refresh_active()

    def request_question(self, request: QuestionRequest) -> None:
        self._question_request_queue.append(request)
        if self._current_question_panel is None:
            console.bell()
            self.show_next_question_request()

    def show_next_question_request(self) -> None:
        """Show the next question request from the queue."""
        if not self._question_request_queue:
            if self._current_question_panel is not None:
                self._current_question_panel = None
                self._question_waiting_for_other_text = False
                self.refresh_active()
            return

        while self._question_request_queue:
            request = self._question_request_queue.popleft()
            if request.resolved:
                continue
            self._current_question_panel = _QuestionRequestPanel(request)
            self._question_waiting_for_other_text = False
            self.refresh_active()
            break
        else:
            # All queued requests were already resolved
            if self._current_question_panel is not None:
                self._current_question_panel = None
                self._question_waiting_for_other_text = False
                self.refresh_active()

    def handle_subagent_event(self, event: SubagentEvent) -> None:
        block = self._tool_call_blocks.get(event.task_tool_call_id)
        if block is None:
            return

        match event.event:
            case ToolCall() as tool_call:
                block.append_sub_tool_call(tool_call)
            case ToolCallPart() as tool_call_part:
                block.append_sub_tool_call_part(tool_call_part)
            case ToolResult() as tool_result:
                block.finish_sub_tool_call(tool_result)
                self.refresh_active()
            case _:
                # ignore other events for now
                # TODO: may need to handle multi-level nested subagents
                pass
