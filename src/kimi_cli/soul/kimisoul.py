from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

import kosong
import tenacity
from kosong import StepResult
from kosong.chat_provider import (
    APIConnectionError,
    APIEmptyResponseError,
    APIStatusError,
    APITimeoutError,
    RetryableChatProvider,
)
from kosong.message import Message
from tenacity import RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from kimi_cli.background import build_active_task_snapshot
from kimi_cli.llm import ModelCapability
from kimi_cli.notifications import (
    build_notification_message,
    extract_notification_ids,
)
from kimi_cli.skill import Skill, normalize_skill_name, read_skill_text
from kimi_cli.skill.flow import Flow, FlowEdge, FlowNode, parse_choice
from kimi_cli.soul import (
    LLMNotSet,
    LLMNotSupported,
    MaxStepsReached,
    Soul,
    StatusSnapshot,
    get_wire_or_none,
    wire_send,
)
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.attachment import Attachment, AttachmentProvider, normalize_history
from kimi_cli.soul.attachments.plan_mode import PlanModeAttachmentProvider
from kimi_cli.soul.compaction import (
    CompactionResult,
    SimpleCompaction,
    estimate_text_tokens,
    should_auto_compact,
)
from kimi_cli.soul.compaction_archive import (
    build_compaction_summary,
    load_compaction_archives,
    register_compaction_archive,
)
from kimi_cli.soul.context import Context
from kimi_cli.soul.message import (
    check_message,
    internal_user_message,
    system,
    tool_result_to_message,
)
from kimi_cli.soul.slash import registry as soul_slash_registry
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.dmail import NAME as SendDMail_NAME
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.utils.logging import logger
from kimi_cli.utils.message import content_parts_stringify
from kimi_cli.utils.slashcmd import SlashCommand, parse_slash_command_call
from kimi_cli.wire.file import WireFile
from kimi_cli.wire.types import (
    ApprovalRequest,
    ApprovalResponse,
    CompactionBegin,
    CompactionEnd,
    ContentPart,
    FollowUpInput,
    MCPLoadingBegin,
    MCPLoadingEnd,
    QuestionItem,
    QuestionNotSupported,
    QuestionOption,
    QuestionRequest,
    SkillReminderNotice,
    StatusUpdate,
    StepBegin,
    StepInterrupted,
    TextPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
)

if TYPE_CHECKING:

    def type_check(soul: KimiSoul):
        _: Soul = soul


SKILL_COMMAND_PREFIX = "skill:"
FLOW_COMMAND_PREFIX = "flow:"
DEFAULT_MAX_FLOW_MOVES = 1000
MAX_SKILL_RECOMMENDATIONS = 3

TURN_END_QUESTION_DETECTOR_PROMPT = (
    "You are a background analyzer that inspects an AI assistant's message.\n"
    "The user will provide the assistant's latest message. Your job is to determine\n"
    "whether that message ends with a question or decision prompt\n"
    "asking the user to choose between specific options, make a decision,\n"
    "or pick from multiple concrete suggestions.\n"
    "\n"
    "Examples of choice questions:\n"
    '- "Do you want me to proceed with option A or option B?"\n'
    '- "Should I use approach 1, approach 2, or approach 3?"\n'
    '- "Would you like to continue, start over, or stop?"\n'
    '- "Which framework do you prefer: React, Vue, or Angular?"\n'
    '- "请选择 A 还是 B？"\n'
    '- "Should I proceed?" (yes/no — options: Yes, No)\n'
    '- "Do you want me to continue?" (yes/no — options: Yes, No)\n'
    '- "是否继续？" (yes/no — options: Yes, No)\n'
    '- "是否要继续？" (yes/no — options: Yes, No)\n'
    '- "是否要按照这个方案继续？" (yes/no — options: Yes, No)\n'
    '- "是否需要我按上面的步骤直接开始修改？" (yes/no — options: Yes, No)\n'
    '- "如果你愿意，我可以继续直接做下一轮。" (yes/no — options: Continue, Stop)\n'
    '- "如果你愿意，我就按这个方案开始处理。" (yes/no — options: Proceed, Don\'t proceed)\n'
    '- "如果你想，我可以直接继续改下去。" (yes/no — options: Continue, Stop)\n'
    '- "如果你想，我现在就可以按这个方案开始修改。" (yes/no — options: Proceed, Don\'t proceed)\n'
    '- "如果你要，我可以继续直接做下去。" (yes/no — options: Continue, Stop)\n'
    '- "如果你要，我现在就按这个方案开始改。" (yes/no — options: Proceed, Don\'t proceed)\n'
    '- "如果继续，我可以先处理 A。" (yes/no — options: Continue, Stop)\n'
    '- "如果要继续，我现在就开始处理。" (yes/no — options: Proceed, Don\'t proceed)\n'
    '- "下一步我建议做 A、B、C，你想先做哪个？"\n'
    '- "我有 3 个建议：修交互、提性能、收样式。请选择一个。"\n'
    '- "接下来有三个建议：A、B、C。请告诉我先做哪个。"\n'
    '- "下一步可选：修交互 / 提性能 / 收样式，请选一个继续。"\n'
    '- "我建议下一轮做：1. 修交互 2. 提性能 3. 收样式。选一个，我继续。"\n'
    '- "Next steps: 1. Fix interactions 2. Improve performance 3. Tidy styling. '
    'Choose one for me to do first."\n'
    "\n"
    "Do NOT consider these as choice questions:\n"
    "- General clarifying questions without specific options or actionable suggestions\n"
    '- Rhetorical questions like "Does that make sense?"\n'
    "- Questions embedded in the middle of the response that were already addressed\n"
    "- Mere recommendation lists or next-step suggestions "
    "when the assistant is not asking the user to pick one\n"
    "- Numbered plans or recommendation lists without a closing choice/decision prompt\n"
    '- Conditional analysis statements like "如果继续这样做，风险会更高。" '
    "when the assistant is describing consequences, not asking for permission "
    "or a decision\n"
    "\n"
    "Return strict JSON with this exact shape:\n"
    '{"has_question": true/false, "questions": '
    '[{"question": "...", "options": [{"label": "...", "description": "..."}]}]}\n'
    "- If has_question is false, questions should be an empty array.\n"
    "- Treat multiple concrete suggestions or recommended next steps as options "
    "when the user is implicitly or explicitly expected to pick one.\n"
    "- Do not infer has_question=true from a numbered list alone; "
    "the ending still needs a pick-one / choose-next / decision prompt.\n"
    "- This can still count even without a literal question mark "
    'if the ending is a decision prompt like "please choose one", '
    '"tell me which to do first", or a soft permission prompt like '
    'Chinese "是否 + action clause" / "如果你愿意，我可以..." / '
    '"如果你想，我可以..." / "如果你要，我可以..." / '
    '"如果继续，我可以...".\n'
    "- For clear binary permission prompts without explicit options, synthesize "
    "two concise options that preserve the intent, such as 继续/先别 or "
    "开始/先不要.\n"
    "- Each question should have 2-4 options, extracted from the message "
    "when explicit, or synthesized for clear binary permission prompts "
    "when implicit.\n"
    "- Option labels should be concise (1-5 words).\n"
    "- Option descriptions should briefly explain the trade-offs if mentioned.\n"
    "- Do not include markdown or any extra text.\n"
)
SKILL_RECOMMENDER_PROMPT = (
    "You are a background skill recommender for Kimi Code CLI.\n"
    "Given the ongoing conversation and the available skills below, decide whether "
    "the main agent should be reminded about any skill right now.\n"
    "Only recommend skills that are clearly relevant to the current task. "
    "Prefer precision over recall.\n"
    "Return strict JSON with this exact shape:\n"
    '{"skills":[{"name":"exact skill name","reason":"short reason"}]}\n'
    "- Use exact skill names from the catalog.\n"
    "- Return at most 3 skills.\n"
    '- If none are useful, return {"skills":[]}.\n'
    "- Do not include markdown or any extra text.\n\n"
    "Available skills:\n"
)


type StepStopReason = Literal["no_tool_calls", "tool_rejected"]


@dataclass(frozen=True, slots=True)
class StepOutcome:
    stop_reason: StepStopReason
    assistant_message: Message


type TurnStopReason = StepStopReason


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    stop_reason: TurnStopReason
    final_message: Message | None
    step_count: int


@dataclass(frozen=True, slots=True)
class SkillRecommendationItem:
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class SkillRecommendation:
    skills: tuple[SkillRecommendationItem, ...]


@dataclass(frozen=True, slots=True)
class TurnEndQuestionOption:
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class TurnEndQuestionItem:
    question: str
    options: tuple[TurnEndQuestionOption, ...]


@dataclass(frozen=True, slots=True)
class TurnEndQuestionDetection:
    has_question: bool
    questions: tuple[TurnEndQuestionItem, ...]


@dataclass(slots=True)
class SkillReminderState:
    task: asyncio.Task[SkillRecommendation | None]
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _QueuedSteer:
    turn_id: int
    content: str | list[ContentPart]


class KimiSoul:
    """The soul of Kimi Code CLI."""

    def __init__(
        self,
        agent: Agent,
        *,
        context: Context,
    ):
        """
        Initialize the soul.

        Args:
            agent (Agent): The agent to run.
            context (Context): The context of the agent.
        """
        self._agent = agent
        self._runtime = agent.runtime
        self._denwa_renji = agent.runtime.denwa_renji
        self._approval = agent.runtime.approval
        self._context = context
        self._loop_control = agent.runtime.config.loop_control
        self._compaction = SimpleCompaction()  # TODO: maybe configurable and composable

        for tool in agent.toolset.tools:
            if tool.name == SendDMail_NAME:
                self._checkpoint_with_user_message = True
                break
        else:
            self._checkpoint_with_user_message = False

        self._steer_queue: asyncio.Queue[_QueuedSteer] = asyncio.Queue()
        self._active_turn_id: int | None = None
        self._next_turn_id = 0
        self._plan_mode: bool = self._runtime.session.state.plan_mode
        self._plan_session_id: str | None = None
        self._pending_plan_activation_attachment: bool = False
        if self._plan_mode:
            self._ensure_plan_session_id()
        self._attachment_providers: list[AttachmentProvider] = [
            PlanModeAttachmentProvider(),
        ]

        if self._runtime.role == "root":
            self._runtime.notifications.ack_ids("llm", extract_notification_ids(context.history))

        # Bind tool state that depends on the live soul/context
        self._bind_plan_mode_tools()
        self._bind_context_recall_tools()

        self._slash_commands = self._build_slash_commands()
        self._slash_command_map = self._index_slash_commands(self._slash_commands)

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def model_name(self) -> str:
        return self._runtime.llm.chat_provider.model_name if self._runtime.llm else ""

    @property
    def model_capabilities(self) -> set[ModelCapability] | None:
        if self._runtime.llm is None:
            return None
        return self._runtime.llm.capabilities

    @property
    def plan_mode(self) -> bool:
        """Whether plan mode (read-only research and planning) is active."""
        return self._plan_mode

    def add_attachment_provider(self, provider: AttachmentProvider) -> None:
        """Register an additional attachment provider."""
        self._attachment_providers.append(provider)

    async def _collect_attachments(self) -> list[Attachment]:
        """Collect attachments from all registered providers."""
        attachments: list[Attachment] = []
        for provider in self._attachment_providers:
            result = await provider.get_attachments(self._context.history, self)
            attachments.extend(result)
        return attachments

    def _bind_plan_mode_tools(self) -> None:
        """Bind plan mode state to tools that support it."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return

        def checker() -> bool:
            return self._plan_mode

        def path_getter() -> Path | None:
            return self.get_plan_file_path()

        # WriteFile gets both checker and path_getter (for plan file auto-approve)
        from kimi_cli.tools.file.write import WriteFile

        write_tool = self._agent.toolset.find("WriteFile")
        if isinstance(write_tool, WriteFile):
            write_tool.bind_plan_mode(checker, path_getter)

        # ExitPlanMode has a special bind() method
        from kimi_cli.tools.plan import ExitPlanMode

        exit_tool = self._agent.toolset.find("ExitPlanMode")
        if isinstance(exit_tool, ExitPlanMode):
            exit_tool.bind(self.toggle_plan_mode, path_getter, checker)

        # EnterPlanMode has a special bind() with yolo_checker
        from kimi_cli.tools.plan.enter import EnterPlanMode

        enter_tool = self._agent.toolset.find("EnterPlanMode")
        if isinstance(enter_tool, EnterPlanMode):

            def yolo_checker() -> bool:
                return self._approval.is_yolo()

            enter_tool.bind(self.toggle_plan_mode, path_getter, checker, yolo_checker)

        # AskUserQuestion gets plan mode checker for dynamic description
        from kimi_cli.tools.ask_user import AskUserQuestion

        ask_tool = self._agent.toolset.find("AskUserQuestion")
        if isinstance(ask_tool, AskUserQuestion):
            ask_tool.bind_plan_mode(checker)

    def _bind_context_recall_tools(self) -> None:
        """Bind current context file accessors to tools that need trajectory-local state."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return

        from kimi_cli.tools.context import RecallCompactedContext

        recall_tool = self._agent.toolset.find("RecallCompactedContext")
        if isinstance(recall_tool, RecallCompactedContext):
            recall_tool.bind_context_file(lambda: self._context.file_backend)
            self._sync_context_recall_tool_visibility()

    def _sync_context_recall_tool_visibility(self) -> None:
        """Show RecallCompactedContext only when trajectory archives actually exist."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return

        recall_tool = self._agent.toolset.find("RecallCompactedContext")
        if recall_tool is None:
            return

        if load_compaction_archives(self._context.file_backend):
            self._agent.toolset.unhide("RecallCompactedContext")
        else:
            self._agent.toolset.hide("RecallCompactedContext")

    def _ensure_plan_session_id(self) -> None:
        """Allocate a stable plan session ID on first activation."""
        if self._plan_session_id is None:
            import uuid

            self._plan_session_id = uuid.uuid4().hex

    def _set_plan_mode(self, enabled: bool, *, source: Literal["manual", "tool"]) -> bool:
        """Update plan mode state for either manual or tool-driven toggles."""
        if enabled == self._plan_mode:
            return self._plan_mode
        self._plan_mode = enabled
        if enabled:
            self._ensure_plan_session_id()
            self._pending_plan_activation_attachment = source == "manual"
        else:
            self._pending_plan_activation_attachment = False
        # Persist plan mode to session state so it survives process restarts
        self._runtime.session.state.plan_mode = self._plan_mode
        self._runtime.session.save_state()
        return self._plan_mode

    def get_plan_file_path(self) -> Path | None:
        """Get the plan file path for the current session."""
        if self._plan_session_id is None:
            return None
        from kimi_cli.tools.plan.heroes import get_plan_file_path

        return get_plan_file_path(self._plan_session_id)

    def read_current_plan(self) -> str | None:
        """Read the current plan file content."""
        if self._plan_session_id is None:
            return None
        from kimi_cli.tools.plan.heroes import read_plan_file

        return read_plan_file(self._plan_session_id)

    def clear_current_plan(self) -> None:
        """Delete the current plan file."""
        path = self.get_plan_file_path()
        if path and path.exists():
            path.unlink()

    async def toggle_plan_mode(self) -> bool:
        """Toggle plan mode on/off. Returns the new state.

        Tools are not hidden/unhidden — instead, each tool checks plan mode
        state at call time and rejects if blocked.
        Periodic reminders are handled by the attachment system.
        """
        return self._set_plan_mode(not self._plan_mode, source="tool")

    async def toggle_plan_mode_from_manual(self) -> bool:
        """Toggle plan mode from UI/manual entry points (slash command, keybinding)."""
        return self._set_plan_mode(not self._plan_mode, source="manual")

    async def set_plan_mode_from_manual(self, enabled: bool) -> bool:
        """Set plan mode to a specific state from UI/manual entry points.

        Unlike toggle, this accepts the desired state directly, avoiding
        race conditions when the caller already knows the target value.
        """
        return self._set_plan_mode(enabled, source="manual")

    def consume_pending_plan_activation_attachment(self) -> bool:
        """Consume the next-step activation reminder scheduled by a manual toggle."""
        if not self._plan_mode or not self._pending_plan_activation_attachment:
            return False
        self._pending_plan_activation_attachment = False
        return True

    @property
    def thinking(self) -> bool | None:
        """Whether thinking mode is enabled."""
        if self._runtime.llm is None:
            return None
        if thinking_effort := self._runtime.llm.chat_provider.thinking_effort:
            return thinking_effort != "off"
        return None

    @property
    def status(self) -> StatusSnapshot:
        token_count = self._context.token_count
        max_size = self._runtime.llm.max_context_size if self._runtime.llm is not None else 0
        return StatusSnapshot(
            context_usage=self._context_usage,
            yolo_enabled=self._approval.is_yolo(),
            plan_mode=self._plan_mode,
            context_tokens=token_count,
            max_context_tokens=max_size,
        )

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def context(self) -> Context:
        return self._context

    @property
    def _context_usage(self) -> float:
        if self._runtime.llm is not None:
            return self._context.token_count / self._runtime.llm.max_context_size
        return 0.0

    @property
    def wire_file(self) -> WireFile:
        return self._runtime.session.wire_file

    async def _checkpoint(self):
        await self._context.checkpoint(self._checkpoint_with_user_message)

    def _begin_turn(self) -> int:
        self._next_turn_id += 1
        self._active_turn_id = self._next_turn_id
        return self._next_turn_id

    def _end_turn(self, turn_id: int) -> None:
        if self._active_turn_id == turn_id:
            self._active_turn_id = None

    def steer(self, content: str | list[ContentPart]) -> None:
        """Queue a steer message for injection into the current turn."""
        turn_id = self._active_turn_id
        if turn_id is None:
            logger.debug("Ignoring steer because there is no active turn")
            return
        self._steer_queue.put_nowait(_QueuedSteer(turn_id=turn_id, content=content))

    async def _consume_pending_steers(self) -> bool:
        """Drain the steer queue and inject as synthetic tool results.

        Returns True if any steers were consumed.
        """
        consumed = False
        active_turn_id = self._active_turn_id
        while not self._steer_queue.empty():
            steer = self._steer_queue.get_nowait()
            if steer.turn_id != active_turn_id:
                logger.debug(
                    "Dropping stale steer from turn {steer_turn_id}; "
                    "active turn is {active_turn_id}",
                    steer_turn_id=steer.turn_id,
                    active_turn_id=active_turn_id,
                )
                continue
            await self._inject_steer(steer.content)
            consumed = True
        return consumed

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

    @classmethod
    def _build_steer_message(cls, content: str | list[ContentPart]) -> Message:
        content_parts: list[ContentPart] = [
            TextPart(
                text=(
                    "<system-reminder>\n"
                    f"{cls._steer_instruction_text()}\n\n"
                    "Reminder content follows in the rest of this message.\n"
                    "</system-reminder>"
                )
            )
        ]
        if isinstance(content, str):
            stripped = content.strip()
            if stripped:
                content_parts.append(TextPart(text=stripped))
        else:
            content_parts.extend(content)
        return internal_user_message(content_parts)

    @staticmethod
    def _stringify_steer_content(content: str | list[ContentPart]) -> str:
        if isinstance(content, str):
            return content.strip()
        return content_parts_stringify(content).strip()

    async def _inject_steer(self, content: str | list[ContentPart]) -> None:
        """Inject a single steer as a real-time reminder appended to user messages."""
        reminder_message = self._build_steer_message(content)
        if self._runtime.llm is None:
            raise LLMNotSet()
        if missing_caps := check_message(reminder_message, self._runtime.llm.capabilities):
            fallback_text = self._stringify_steer_content(content)
            if not fallback_text:
                fallback_text = (
                    "The reminder included non-text content that is not supported by the current "
                    "model."
                )
            reminder_message = internal_user_message(
                TextPart(
                    text=(
                        "<system-reminder>\n"
                        f"{self._steer_instruction_text()}\n\n"
                        f"Reminder:\n{fallback_text}\n"
                        "</system-reminder>"
                    )
                )
            )
            if missing_caps := check_message(reminder_message, self._runtime.llm.capabilities):
                raise LLMNotSupported(self._runtime.llm, list(missing_caps))
        await self._context.append_message(reminder_message)

    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        return self._slash_commands

    async def run(self, user_input: str | list[ContentPart]):
        turn_id = self._begin_turn()
        try:
            # Refresh OAuth tokens on each turn to avoid idle-time expirations.
            await self._runtime.oauth.ensure_fresh(self._runtime)

            wire_send(TurnBegin(user_input=user_input))
            user_message = Message(role="user", content=user_input)
            text_input = user_message.extract_text(" ").strip()

            outcome: TurnOutcome | None = None
            if command_call := parse_slash_command_call(text_input):
                command = self._find_slash_command(command_call.name)
                if command is None:
                    # this should not happen actually, the shell should have filtered it out
                    wire_send(TextPart(text=f'Unknown slash command "/{command_call.name}".'))
                else:
                    ret = command.func(self, command_call.args)
                    if isinstance(ret, Awaitable):
                        await ret
            elif self._loop_control.max_ralph_iterations != 0:
                runner = FlowRunner.ralph_loop(
                    user_message,
                    self._loop_control.max_ralph_iterations,
                )
                await runner.run(self, "", enable_skill_reminder=True)
            else:
                outcome = await self._turn(user_message, enable_skill_reminder=True)

            # Turn-end question detection: if enabled and the turn produced a final
            # message, check whether it asks the user to pick between options.
            if outcome is not None and self._loop_control.turn_end_question_detection:
                answer = await self._maybe_ask_turn_end_question(outcome)
                if answer:
                    # The user chose an option — echo their selection in the TUI
                    # and run a follow-up turn with their answer as the prompt.
                    wire_send(FollowUpInput(text=answer))
                    await self._turn(
                        Message(role="user", content=answer),
                    )

            wire_send(TurnEnd())
        finally:
            self._end_turn(turn_id)

    async def _turn(
        self,
        user_message: Message,
        *,
        enable_skill_reminder: bool = False,
    ) -> TurnOutcome:
        if self._runtime.llm is None:
            raise LLMNotSet()

        if missing_caps := check_message(user_message, self._runtime.llm.capabilities):
            raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        await self._checkpoint()  # this creates the checkpoint 0 on first run
        await self._context.append_message(user_message)
        logger.debug("Appended user message to context")

        skill_reminder = self._start_skill_reminder_task() if enable_skill_reminder else None
        try:
            return await self._agent_loop(skill_reminder)
        finally:
            await self._finalize_skill_reminder_task(skill_reminder)

    def _build_slash_commands(self) -> list[SlashCommand[Any]]:
        commands: list[SlashCommand[Any]] = list(soul_slash_registry.list_commands())
        seen_names = {cmd.name for cmd in commands}

        for skill in self._runtime.skills.values():
            if skill.type not in ("standard", "flow"):
                continue
            name = f"{SKILL_COMMAND_PREFIX}{skill.name}"
            if name in seen_names:
                logger.warning(
                    "Skipping skill slash command /{name}: name already registered",
                    name=name,
                )
                continue
            commands.append(
                SlashCommand(
                    name=name,
                    func=self._make_skill_runner(skill),
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(name)

        for skill in self._runtime.skills.values():
            if skill.type != "flow":
                continue
            if skill.flow is None:
                logger.warning("Flow skill {name} has no flow; skipping", name=skill.name)
                continue
            command_name = f"{FLOW_COMMAND_PREFIX}{skill.name}"
            if command_name in seen_names:
                logger.warning(
                    "Skipping prompt flow slash command /{name}: name already registered",
                    name=command_name,
                )
                continue
            runner = FlowRunner(skill.flow, name=skill.name)
            commands.append(
                SlashCommand(
                    name=command_name,
                    func=runner.run,
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(command_name)

        return commands

    @staticmethod
    def _index_slash_commands(
        commands: list[SlashCommand[Any]],
    ) -> dict[str, SlashCommand[Any]]:
        indexed: dict[str, SlashCommand[Any]] = {}
        for command in commands:
            indexed[command.name] = command
            for alias in command.aliases:
                indexed[alias] = command
        return indexed

    def _find_slash_command(self, name: str) -> SlashCommand[Any] | None:
        return self._slash_command_map.get(name)

    def _make_skill_runner(self, skill: Skill) -> Callable[[KimiSoul, str], None | Awaitable[None]]:
        async def _run_skill(soul: KimiSoul, args: str, *, _skill: Skill = skill) -> None:
            skill_text = await read_skill_text(_skill)
            if skill_text is None:
                wire_send(
                    TextPart(text=f'Failed to load skill "/{SKILL_COMMAND_PREFIX}{_skill.name}".')
                )
                return
            extra = args.strip()
            if extra:
                skill_text = f"{skill_text}\n\nUser request:\n{extra}"
            await soul._turn(Message(role="user", content=skill_text))

        _run_skill.__doc__ = skill.description
        return _run_skill

    def _start_skill_reminder_task(self) -> SkillReminderState | None:
        if self._runtime.llm is None or not self._runtime.skills:
            return None
        history = list(self._context.history)
        task = asyncio.create_task(self._request_skill_recommendation(history))
        return SkillReminderState(task=task)

    async def _finalize_skill_reminder_task(self, state: SkillReminderState | None) -> None:
        if state is None:
            return
        if not state.task.done():
            state.task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await state.task

    async def _request_skill_recommendation(
        self,
        history: Sequence[Message],
    ) -> SkillRecommendation | None:
        assert self._runtime.llm is not None
        chat_provider = self._runtime.llm.chat_provider.with_thinking("off")

        async def _run_once():
            return await kosong.generate(
                chat_provider=chat_provider,
                system_prompt=(
                    f"{SKILL_RECOMMENDER_PROMPT}{self._format_available_skills_for_recommender()}\n"
                ),
                tools=[],
                history=history,
            )

        try:
            result = await self._run_with_connection_recovery(
                "skill recommendation",
                _run_once,
                chat_provider=chat_provider,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Skill recommendation request failed: {error}", error=exc)
            return None

        recommendation = self._parse_skill_recommendation_payload(result.message.extract_text(" "))
        if recommendation is None:
            logger.warning(
                "Failed to parse skill recommendation response: {text}", text=result.message
            )
        return recommendation

    def _format_available_skills_for_recommender(self) -> str:
        return "\n".join(
            (
                f"- {skill.name} (type: {skill.type})\n"
                f"  - Path: {skill.skill_md_file}\n"
                f"  - Description: {skill.description}"
            )
            for skill in sorted(
                self._runtime.skills.values(), key=lambda item: item.name.casefold()
            )
        )

    def _parse_skill_recommendation_payload(self, text: str) -> SkillRecommendation | None:
        payload_text = self._extract_json_payload(text)
        try:
            payload_obj: object = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload_obj, dict):
            return None
        payload = cast(dict[str, object], payload_obj)
        raw_skills_obj = payload.get("skills", [])
        if not isinstance(raw_skills_obj, list):
            return None
        raw_skills = cast(list[object], raw_skills_obj)

        recommendations: list[SkillRecommendationItem] = []
        seen: set[str] = set()
        for raw_item in raw_skills[:MAX_SKILL_RECOMMENDATIONS]:
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, object], raw_item)
            raw_name = item.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            skill_key = normalize_skill_name(raw_name.strip())
            skill = self._runtime.skills.get(skill_key)
            if skill is None or skill_key in seen:
                continue
            raw_reason = item.get("reason", "")
            reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
            recommendations.append(SkillRecommendationItem(name=skill.name, reason=reason))
            seen.add(skill_key)
        return SkillRecommendation(skills=tuple(recommendations))

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        payload = text.strip()
        if payload.startswith("```"):
            lines = payload.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                payload = "\n".join(lines[1:-1]).strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            return payload[start : end + 1]
        return payload

    async def _consume_ready_skill_reminder(self, state: SkillReminderState | None) -> bool:
        if state is None or state.consumed or not state.task.done():
            return False

        state.consumed = True
        try:
            recommendation = await state.task
        except asyncio.CancelledError:
            return False
        except Exception as exc:
            logger.warning("Skill reminder task failed: {error}", error=exc)
            return False
        if recommendation is None or not recommendation.skills:
            return False

        await self._context.append_message(self._build_skill_reminder_message(recommendation))
        wire_send(
            SkillReminderNotice(
                skills=[f"/skill:{item.name}" for item in recommendation.skills],
            )
        )
        logger.debug("Injected skill reminder into context")
        return True

    def _build_skill_reminder_message(self, recommendation: SkillRecommendation) -> Message:
        lines = ["Reminder: the following skill suggestions may be helpful in the current context."]
        for item in recommendation.skills:
            skill = self._runtime.skills[normalize_skill_name(item.name)]
            lines.append(f"- {skill.name} ({skill.type} skill): {skill.description}")
            if item.reason:
                lines.append(f"  Reason: {item.reason}")
            lines.append(f"  Path: {skill.skill_md_file}")
            lines.append(
                "  Consider reading this skill's SKILL.md before continuing if it seems useful."
            )
        return internal_user_message([system("\n".join(lines))])

    # -- Turn-end question detection --------------------------------------------------

    _TURN_END_DETECT_MAX_ATTEMPTS = 2
    _TURN_END_DETECT_TIMEOUT = 15.0  # seconds

    async def _detect_turn_end_question(
        self,
        assistant_message: Message,
    ) -> TurnEndQuestionDetection | None:
        """Use a side-channel LLM call to check if *assistant_message* asks the user
        to choose between options.  Returns the parsed detection or ``None`` on
        failure.  Retries up to ``_TURN_END_DETECT_MAX_ATTEMPTS`` when the LLM
        returns unparseable output.  The entire detection is capped at
        ``_TURN_END_DETECT_TIMEOUT`` seconds."""
        try:
            return await asyncio.wait_for(
                self._detect_turn_end_question_inner(assistant_message),
                timeout=self._TURN_END_DETECT_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Turn-end question detection timed out after {timeout}s",
                timeout=self._TURN_END_DETECT_TIMEOUT,
            )
            return self._heuristic_turn_end_question(assistant_message.extract_text(" "))

    def _turn_end_question_excerpt(self, text: str) -> str:
        units = [
            unit.strip()
            for unit in re.split(r"(?:\r?\n)+|(?<=[。！？!?])\s*", text)
            if unit.strip()
        ]
        if not units:
            return text.strip()
        return "\n".join(units[-3:])

    def _heuristic_turn_end_question(
        self,
        assistant_text: str,
    ) -> TurnEndQuestionDetection | None:
        excerpt = self._turn_end_question_excerpt(assistant_text)
        units = [
            unit.strip().strip("\"“”'`")
            for unit in re.split(r"(?:\r?\n)+|(?<=[。！？!?])\s*", excerpt)
            if unit.strip()
        ]
        if not units:
            return None

        soft_prefixes = (
            "如果你要",
            "如果你想",
            "如果你愿意",
            "如果你希望",
            "如果继续",
            "如果要继续",
        )
        leading_wrappers = ">》」』】）)]-•·*\"“”'`("
        conditional_offer_tokens = ("我可以", "我现在就", "我现在可以", "我现在就可以")
        direct_offer_tokens = conditional_offer_tokens + ("我就",)
        action_tokens = (
            "继续",
            "开始",
            "按这个方案",
            "修改",
            "处理",
            "推进",
            "做下去",
            "改下去",
            "做下一轮",
            "做下一步",
        )

        for unit in reversed(units):
            normalized = re.sub(r"\s+", "", unit)
            if not normalized:
                continue

            candidate = normalized.lstrip(leading_wrappers)
            prefix = next((token for token in soft_prefixes if candidate.startswith(token)), None)
            if prefix is None:
                continue
            if candidate.startswith(
                ("如果继续这样做", "如果继续这么做", "如果要继续这样做", "如果要继续这么做")
            ):
                continue

            offer_tokens = (
                direct_offer_tokens
                if prefix in {"如果你要", "如果你想", "如果你愿意", "如果你希望"}
                else conditional_offer_tokens
            )
            if not any(token in candidate for token in offer_tokens):
                continue
            if not any(token in candidate for token in action_tokens):
                continue

            continue_like = any(
                token in candidate for token in ("继续", "做下去", "改下去", "做下一轮", "做下一步")
            )
            if continue_like:
                question = "要我继续吗？"
                options = (
                    TurnEndQuestionOption(label="继续", description="继续按当前方案往下做"),
                    TurnEndQuestionOption(label="先别", description="先不要继续"),
                )
            else:
                question = "要我现在开始吗？"
                options = (
                    TurnEndQuestionOption(label="开始", description="现在开始处理"),
                    TurnEndQuestionOption(label="先别", description="先不要开始"),
                )
            return TurnEndQuestionDetection(
                has_question=True,
                questions=(TurnEndQuestionItem(question=question, options=options),),
            )
        return None

    async def _detect_turn_end_question_inner(
        self,
        assistant_message: Message,
    ) -> TurnEndQuestionDetection | None:
        """Inner implementation without timeout wrapper."""
        assert self._runtime.llm is not None
        chat_provider = self._runtime.llm.chat_provider.with_thinking("off")

        # Strip thinking/reasoning content – only send text parts to the
        # side-channel so the detector sees the actual reply, not chain-of-thought.
        # Focus the detector on the tail of the reply because the turn-end prompt
        # is often only present in the last 1-3 sentences.
        # Wrap in a user message so the detector model clearly sees it as content
        # to analyze, not as its own prior output.
        text_only = assistant_message.extract_text(" ")
        heuristic_detection = self._heuristic_turn_end_question(text_only)
        excerpt = self._turn_end_question_excerpt(text_only)
        history: list[Message] = [
            Message(
                role="user",
                content=(
                    "Analyze the following assistant message. Focus on the ending, "
                    "but use the full message if earlier lines contain the options.\n\n"
                    f"Ending excerpt:\n{excerpt}\n\n"
                    f"Full message:\n{text_only}"
                ),
            )
        ]

        for attempt in range(1, self._TURN_END_DETECT_MAX_ATTEMPTS + 1):

            async def _run_once():
                return await kosong.generate(
                    chat_provider=chat_provider,
                    system_prompt=TURN_END_QUESTION_DETECTOR_PROMPT,
                    tools=[],
                    history=history,
                )

            try:
                result = await self._run_with_connection_recovery(
                    "turn-end question detection",
                    _run_once,
                    chat_provider=chat_provider,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Turn-end question detection failed: {error}", error=exc)
                return heuristic_detection

            raw_text = result.message.extract_text(" ")
            detection = self._parse_turn_end_question_payload(raw_text)
            if detection is not None:
                return detection

            if attempt < self._TURN_END_DETECT_MAX_ATTEMPTS:
                logger.debug(
                    "Turn-end question detection returned unparseable output "
                    "(attempt {attempt}), retrying",
                    attempt=attempt,
                )
            else:
                logger.warning(
                    "Turn-end question detection returned unparseable output "
                    "after {attempts} attempts; giving up",
                    attempts=self._TURN_END_DETECT_MAX_ATTEMPTS,
                )

        return heuristic_detection

    def _parse_turn_end_question_payload(self, text: str) -> TurnEndQuestionDetection | None:
        payload_text = self._extract_json_payload(text)
        try:
            payload_obj: object = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload_obj, dict):
            return None
        payload = cast(dict[str, object], payload_obj)

        has_question = payload.get("has_question", False)
        if not isinstance(has_question, bool):
            return None
        if not has_question:
            return TurnEndQuestionDetection(has_question=False, questions=())

        raw_questions = payload.get("questions", [])
        if not isinstance(raw_questions, list):
            return None

        questions: list[TurnEndQuestionItem] = []
        for raw_q in cast(list[object], raw_questions)[:4]:
            if not isinstance(raw_q, dict):
                continue
            q = cast(dict[str, object], raw_q)
            question_text = q.get("question")
            if not isinstance(question_text, str) or not question_text.strip():
                continue
            raw_options = q.get("options", [])
            if not isinstance(raw_options, list):
                continue
            options: list[TurnEndQuestionOption] = []
            for raw_opt in cast(list[object], raw_options)[:4]:
                if not isinstance(raw_opt, dict):
                    continue
                opt = cast(dict[str, object], raw_opt)
                label = opt.get("label")
                if not isinstance(label, str) or not label.strip():
                    continue
                desc = opt.get("description", "")
                desc = desc.strip() if isinstance(desc, str) else ""
                options.append(TurnEndQuestionOption(label=label.strip(), description=desc))
            if len(options) >= 2:
                questions.append(
                    TurnEndQuestionItem(
                        question=question_text.strip(),
                        options=tuple(options),
                    )
                )
        if not questions:
            return TurnEndQuestionDetection(has_question=False, questions=())
        return TurnEndQuestionDetection(has_question=True, questions=tuple(questions))

    async def _maybe_ask_turn_end_question(
        self,
        outcome: TurnOutcome,
    ) -> str | None:
        """Detect choice questions in the turn's final message and present them
        to the user via a structured ``QuestionRequest``.

        Returns the user's answer text to be used as the next turn prompt,
        or ``None`` if no question was detected / the user dismissed it.
        """
        if outcome.stop_reason != "no_tool_calls" or outcome.final_message is None:
            return None

        detection = await self._detect_turn_end_question(outcome.final_message)
        if detection is None or not detection.has_question:
            return None

        wire = get_wire_or_none()
        if wire is None:
            return None

        assistant_reply_body = outcome.final_message.extract_text(sep="\n").strip()

        questions = [
            QuestionItem(
                question=q.question,
                options=[
                    QuestionOption(label=o.label, description=o.description) for o in q.options
                ],
                body=assistant_reply_body,
            )
            for q in detection.questions
        ]

        request = QuestionRequest(
            id=str(uuid4()),
            tool_call_id=f"turn-end-{uuid4().hex[:8]}",
            questions=questions,
        )

        wire_send(request)

        try:
            answers = await request.wait()
        except QuestionNotSupported:
            logger.debug("Client does not support interactive questions; skipping")
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to get user response for turn-end question")
            return None

        if not answers:
            return None

        # Build a concise textual summary of the answers for the next turn.
        parts: list[str] = []
        for _question_text, answer_text in answers.items():
            parts.append(f"{answer_text}")
        return "\n".join(parts) if parts else None

    async def _agent_loop(self, skill_reminder: SkillReminderState | None = None) -> TurnOutcome:
        """The main agent loop for one run."""
        assert self._runtime.llm is not None

        if isinstance(self._agent.toolset, KimiToolset):
            loading = self._agent.toolset.has_pending_mcp_tools()
            if loading:
                wire_send(MCPLoadingBegin())
            try:
                await self._agent.toolset.wait_for_mcp_tools()
            finally:
                if loading:
                    wire_send(MCPLoadingEnd())

        async def _pipe_approval_to_wire():
            while True:
                request = await self._approval.fetch_request()
                # Here we decouple the wire approval request and the soul approval request.
                wire_request = ApprovalRequest(
                    id=request.id,
                    action=request.action,
                    description=request.description,
                    sender=request.sender,
                    tool_call_id=request.tool_call_id,
                    display=request.display,
                )
                wire_send(wire_request)
                # We wait for the request to be resolved over the wire, which means that,
                # for each soul, we will have only one approval request waiting on the wire
                # at a time. However, be aware that subagents (which have their own souls) may
                # also send approval requests to the root wire.
                resp = await wire_request.wait()
                self._approval.resolve_request(request.id, resp)
                wire_send(ApprovalResponse(request_id=request.id, response=resp))

        step_no = 0
        while True:
            step_no += 1
            if step_no > self._loop_control.max_steps_per_turn:
                raise MaxStepsReached(self._loop_control.max_steps_per_turn)

            wire_send(StepBegin(n=step_no))
            approval_task = asyncio.create_task(_pipe_approval_to_wire())
            back_to_the_future: BackToTheFuture | None = None
            step_outcome: StepOutcome | None = None
            try:
                if await self._consume_ready_skill_reminder(skill_reminder):
                    logger.debug("Skill reminder was ready before step {step_no}", step_no=step_no)
                # compact the context if needed
                if should_auto_compact(
                    self._context.token_count,
                    self._runtime.llm.max_context_size,
                    trigger_ratio=self._loop_control.compaction_trigger_ratio,
                    reserved_context_size=self._loop_control.reserved_context_size,
                ):
                    logger.info("Context too long, compacting...")
                    await self.compact_context()

                logger.debug("Beginning step {step_no}", step_no=step_no)
                await self._checkpoint()
                self._denwa_renji.set_n_checkpoints(self._context.n_checkpoints)
                step_outcome = await self._step()
            except BackToTheFuture as e:
                back_to_the_future = e
            except Exception:
                # any other exception should interrupt the step
                wire_send(StepInterrupted())
                # break the agent loop
                raise
            finally:
                approval_task.cancel()  # stop piping approval requests to the wire
                with suppress(asyncio.CancelledError):
                    try:
                        await approval_task
                    except Exception:
                        logger.exception("Approval piping task failed")

            if step_outcome is not None:
                has_steers = await self._consume_pending_steers()
                has_skill_reminder = await self._consume_ready_skill_reminder(skill_reminder)
                if step_outcome.stop_reason == "no_tool_calls" and (
                    has_steers or has_skill_reminder
                ):
                    continue  # extra context injected, force another LLM step
                final_message = (
                    step_outcome.assistant_message
                    if step_outcome.stop_reason == "no_tool_calls"
                    else None
                )
                return TurnOutcome(
                    stop_reason=step_outcome.stop_reason,
                    final_message=final_message,
                    step_count=step_no,
                )

            if back_to_the_future is not None:
                await self._context.revert_to(back_to_the_future.checkpoint_id)
                await self._checkpoint()
                await self._context.append_message(back_to_the_future.messages)

            # Consume any pending steers between steps
            await self._consume_pending_steers()

    async def _step(self) -> StepOutcome | None:
        """Run a single step and return a stop outcome, or None to continue."""
        # already checked in `run`
        assert self._runtime.llm is not None
        chat_provider = self._runtime.llm.chat_provider

        if self._runtime.role == "root":

            async def _append_notification(view):
                await self._context.append_message(build_notification_message(view, self._runtime))

            await self._runtime.notifications.deliver_pending(
                "llm",
                limit=4,
                before_claim=self._runtime.background_tasks.reconcile,
                on_notification=_append_notification,
            )

        # Attachment injection
        attachments = await self._collect_attachments()
        if attachments:
            combined = "\n".join(
                f"<system-reminder>\n{att.content}\n</system-reminder>" for att in attachments
            )
            await self._context.append_message(internal_user_message([TextPart(text=combined)]))

        # Normalize: merge adjacent user messages for clean API input
        effective_history = normalize_history(self._context.history)

        async def _run_step_once() -> StepResult:
            # run an LLM step (may be interrupted)
            return await kosong.step(
                chat_provider,
                self._agent.system_prompt,
                self._agent.toolset,
                effective_history,
                on_message_part=wire_send,
                on_tool_result=wire_send,
            )

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=partial(self._retry_log, "step"),
            wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),
            reraise=True,
        )
        async def _kosong_step_with_retry() -> StepResult:
            return await self._run_with_connection_recovery(
                "step",
                _run_step_once,
                chat_provider=chat_provider,
            )

        result = await _kosong_step_with_retry()
        logger.debug("Got step result: {result}", result=result)
        status_update = StatusUpdate(
            token_usage=result.usage, message_id=result.id, plan_mode=self._plan_mode
        )
        if result.usage is not None:
            # mark the token count for the context before the step
            await self._context.update_token_count(result.usage.input)
            snap = self.status
            status_update.context_usage = snap.context_usage
            status_update.context_tokens = snap.context_tokens
            status_update.max_context_tokens = snap.max_context_tokens
        wire_send(status_update)

        # wait for all tool results (may be interrupted)
        plan_mode_before_tools = self._plan_mode
        results = await result.tool_results()
        logger.debug("Got tool results: {results}", results=results)

        # If a tool (EnterPlanMode/ExitPlanMode) changed plan mode during execution,
        # send a corrected StatusUpdate so the client sees the up-to-date state.
        if self._plan_mode != plan_mode_before_tools:
            wire_send(StatusUpdate(plan_mode=self._plan_mode))

        # shield the context manipulation from interruption
        await asyncio.shield(self._grow_context(result, results))

        rejected = any(isinstance(result.return_value, ToolRejectedError) for result in results)
        if rejected:
            _ = self._denwa_renji.fetch_pending_dmail()
            return StepOutcome(stop_reason="tool_rejected", assistant_message=result.message)

        # handle pending D-Mail
        if dmail := self._denwa_renji.fetch_pending_dmail():
            assert dmail.checkpoint_id >= 0, "DenwaRenji guarantees checkpoint_id >= 0"
            assert dmail.checkpoint_id < self._context.n_checkpoints, (
                "DenwaRenji guarantees checkpoint_id < n_checkpoints"
            )
            # raise to let the main loop take us back to the future
            raise BackToTheFuture(
                dmail.checkpoint_id,
                [
                    internal_user_message(
                        [
                            system(
                                "You just got a D-Mail from your future self. "
                                "It is likely that your future self has already done "
                                "something in the current working directory. Please read "
                                "the D-Mail and decide what to do next. You MUST NEVER "
                                "mention to the user about this information. "
                                f"D-Mail content:\n\n{dmail.message.strip()}"
                            )
                        ]
                    )
                ],
            )

        if result.tool_calls:
            return None
        return StepOutcome(stop_reason="no_tool_calls", assistant_message=result.message)

    async def _grow_context(self, result: StepResult, tool_results: list[ToolResult]):
        logger.debug("Growing context with result: {result}", result=result)

        assert self._runtime.llm is not None
        tool_messages = [tool_result_to_message(tr) for tr in tool_results]
        for tm in tool_messages:
            if missing_caps := check_message(tm, self._runtime.llm.capabilities):
                logger.warning(
                    "Tool result message requires unsupported capabilities: {caps}",
                    caps=missing_caps,
                )
                raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        await self._context.append_message(result.message)
        if result.usage is not None:
            await self._context.update_token_count(result.usage.total)

        logger.debug(
            "Appending tool messages to context: {tool_messages}", tool_messages=tool_messages
        )
        await self._context.append_message(tool_messages)
        # token count of tool results are not available yet

    async def compact_context(self, custom_instruction: str = "") -> None:
        """
        Compact the context.

        Raises:
            LLMNotSet: When the LLM is not set.
            ChatProviderError: When the chat provider returns an error.
        """

        chat_provider = self._runtime.llm.chat_provider if self._runtime.llm is not None else None

        async def _run_compaction_once() -> CompactionResult:
            if self._runtime.llm is None:
                raise LLMNotSet()
            return await self._compaction.compact(
                self._context.history, self._runtime.llm, custom_instruction=custom_instruction
            )

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=partial(self._retry_log, "compaction"),
            wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),
            reraise=True,
        )
        async def _compact_with_retry() -> CompactionResult:
            return await self._run_with_connection_recovery(
                "compaction",
                _run_compaction_once,
                chat_provider=chat_provider,
            )

        wire_send(CompactionBegin())
        original_message_count = len(self._context.history)
        compaction_result = await _compact_with_retry()
        rotated_path = await self._context.clear()

        final_messages = list(compaction_result.messages)
        if compaction_result.usage is not None:
            registration = register_compaction_archive(
                self._context.file_backend,
                rotated_path,
                message_count=original_message_count,
                summary=build_compaction_summary(compaction_result.messages),
            )
            final_messages.append(
                internal_user_message(
                    [
                        system(
                            "Compacted context archives are available via the "
                            "RecallCompactedContext tool for this conversation trajectory. "
                            f"Archive `{registration.record.id}` contains the "
                            "pre-compaction history. "
                            f"There are now {registration.total_archives} compacted "
                            "archive(s) available. Use targeted keywords if the "
                            "compaction summary is not sufficient, and prefer this "
                            "tool over reading raw archive files directly."
                        )
                    ]
                )
            )

        self._sync_context_recall_tool_visibility()

        await self._checkpoint()
        await self._context.append_message(final_messages)
        estimated_token_count = CompactionResult(
            messages=final_messages, usage=compaction_result.usage
        ).estimated_token_count

        if self._runtime.role == "root":
            active_task_snapshot = build_active_task_snapshot(self._runtime.background_tasks)
            if active_task_snapshot is not None:
                active_task_message = Message(
                    role="user",
                    content=[
                        system(
                            "The following background tasks are still active after compaction. "
                            "Use TaskList if you need to re-enumerate them later."
                        ),
                        TextPart(text=active_task_snapshot),
                    ],
                )
                await self._context.append_message(active_task_message)
                estimated_token_count += estimate_text_tokens([active_task_message])

        # Estimate token count so context_usage is not reported as 0%
        await self._context.update_token_count(estimated_token_count)

        wire_send(CompactionEnd())

    @staticmethod
    def _is_retryable_error(exception: BaseException) -> bool:
        if isinstance(exception, (APIConnectionError, APITimeoutError)):
            return not bool(getattr(exception, "_kimi_recovery_exhausted", False))
        if isinstance(exception, APIEmptyResponseError):
            return True
        return isinstance(exception, APIStatusError) and exception.status_code in (
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
        )

    async def _run_with_connection_recovery(
        self,
        name: str,
        operation: Callable[[], Awaitable[Any]],
        *,
        chat_provider: object | None = None,
    ) -> Any:
        try:
            return await operation()
        except (APIConnectionError, APITimeoutError) as error:
            if not isinstance(chat_provider, RetryableChatProvider):
                raise
            try:
                recovered = chat_provider.on_retryable_error(error)
            except Exception:
                logger.exception(
                    "Failed to recover chat provider during {name} after {error_type}.",
                    name=name,
                    error_type=type(error).__name__,
                )
                raise
            if not recovered:
                raise
            logger.info(
                "Recovered chat provider during {name} after {error_type}; retrying once.",
                name=name,
                error_type=type(error).__name__,
            )
            try:
                return await operation()
            except (APIConnectionError, APITimeoutError) as second_error:
                second_error._kimi_recovery_exhausted = True  # type: ignore[attr-defined]
                raise

    @staticmethod
    def _retry_log(name: str, retry_state: RetryCallState):
        logger.info(
            "Retrying {name} for the {n} time. Waiting {sleep} seconds.",
            name=name,
            n=retry_state.attempt_number,
            sleep=retry_state.next_action.sleep
            if retry_state.next_action is not None
            else "unknown",
        )


class BackToTheFuture(Exception):
    """
    Raise when we need to revert the context to a previous checkpoint.
    The main agent loop should catch this exception and handle it.
    """

    def __init__(self, checkpoint_id: int, messages: Sequence[Message]):
        self.checkpoint_id = checkpoint_id
        self.messages = messages


class FlowRunner:
    def __init__(
        self,
        flow: Flow,
        *,
        name: str | None = None,
        max_moves: int = DEFAULT_MAX_FLOW_MOVES,
    ) -> None:
        self._flow = flow
        self._name = name
        self._max_moves = max_moves

    @staticmethod
    def ralph_loop(
        user_message: Message,
        max_ralph_iterations: int,
    ) -> FlowRunner:
        prompt_content = list(user_message.content)
        prompt_text = Message(role="user", content=prompt_content).extract_text(" ").strip()
        total_runs = max_ralph_iterations + 1
        if max_ralph_iterations < 0:
            total_runs = 1000000000000000  # effectively infinite

        nodes: dict[str, FlowNode] = {
            "BEGIN": FlowNode(id="BEGIN", label="BEGIN", kind="begin"),
            "END": FlowNode(id="END", label="END", kind="end"),
        }
        outgoing: dict[str, list[FlowEdge]] = {"BEGIN": [], "END": []}

        nodes["R1"] = FlowNode(id="R1", label=prompt_content, kind="task")
        nodes["R2"] = FlowNode(
            id="R2",
            label=(
                f"{prompt_text}. (You are running in an automated loop where the same "
                "prompt is fed repeatedly. Only choose STOP when the task is fully complete. "
                "Including it will stop further iterations. If you are not 100% sure, "
                "choose CONTINUE.)"
            ).strip(),
            kind="decision",
        )
        outgoing["R1"] = []
        outgoing["R2"] = []

        outgoing["BEGIN"].append(FlowEdge(src="BEGIN", dst="R1", label=None))
        outgoing["R1"].append(FlowEdge(src="R1", dst="R2", label=None))
        outgoing["R2"].append(FlowEdge(src="R2", dst="R2", label="CONTINUE"))
        outgoing["R2"].append(FlowEdge(src="R2", dst="END", label="STOP"))

        flow = Flow(nodes=nodes, outgoing=outgoing, begin_id="BEGIN", end_id="END")
        max_moves = total_runs
        return FlowRunner(flow, max_moves=max_moves)

    async def run(
        self,
        soul: KimiSoul,
        args: str,
        *,
        enable_skill_reminder: bool = False,
    ) -> None:
        if args.strip():
            command = f"/{FLOW_COMMAND_PREFIX}{self._name}" if self._name else "/flow"
            logger.warning("Agent flow {command} ignores args: {args}", command=command, args=args)
            return

        current_id = self._flow.begin_id
        moves = 0
        total_steps = 0
        reminder_enabled = enable_skill_reminder
        while True:
            node = self._flow.nodes[current_id]
            edges = self._flow.outgoing.get(current_id, [])

            if node.kind == "end":
                logger.info("Agent flow reached END node {node_id}", node_id=current_id)
                return

            if node.kind == "begin":
                if not edges:
                    logger.error(
                        'Agent flow BEGIN node "{node_id}" has no outgoing edges; stopping.',
                        node_id=node.id,
                    )
                    return
                current_id = edges[0].dst
                continue

            if moves >= self._max_moves:
                raise MaxStepsReached(total_steps)
            next_id, steps_used = await self._execute_flow_node(
                soul,
                node,
                edges,
                enable_skill_reminder=reminder_enabled,
            )
            reminder_enabled = False
            total_steps += steps_used
            if next_id is None:
                return
            moves += 1
            current_id = next_id

    async def _execute_flow_node(
        self,
        soul: KimiSoul,
        node: FlowNode,
        edges: list[FlowEdge],
        *,
        enable_skill_reminder: bool = False,
    ) -> tuple[str | None, int]:
        if not edges:
            logger.error(
                'Agent flow node "{node_id}" has no outgoing edges; stopping.',
                node_id=node.id,
            )
            return None, 0

        base_prompt = self._build_flow_prompt(node, edges)
        prompt = base_prompt
        steps_used = 0
        while True:
            result = await self._flow_turn(
                soul,
                prompt,
                enable_skill_reminder=enable_skill_reminder,
            )
            enable_skill_reminder = False
            steps_used += result.step_count
            if result.stop_reason == "tool_rejected":
                logger.error("Agent flow stopped after tool rejection.")
                return None, steps_used

            if node.kind != "decision":
                return edges[0].dst, steps_used

            choice = (
                parse_choice(result.final_message.extract_text(" "))
                if result.final_message
                else None
            )
            next_id = self._match_flow_edge(edges, choice)
            if next_id is not None:
                return next_id, steps_used

            options = ", ".join(edge.label or "" for edge in edges)
            logger.warning(
                "Agent flow invalid choice. Got: {choice}. Available: {options}.",
                choice=choice or "<missing>",
                options=options,
            )
            prompt = (
                f"{base_prompt}\n\n"
                "Your last response did not include a valid choice. "
                "Reply with one of the choices using <choice>...</choice>."
            )

    @staticmethod
    def _build_flow_prompt(node: FlowNode, edges: list[FlowEdge]) -> str | list[ContentPart]:
        if node.kind != "decision":
            return node.label

        if not isinstance(node.label, str):
            label_text = Message(role="user", content=node.label).extract_text(" ")
        else:
            label_text = node.label
        choices = [edge.label for edge in edges if edge.label]
        lines = [
            label_text,
            "",
            "Available branches:",
            *(f"- {choice}" for choice in choices),
            "",
            "Reply with a choice using <choice>...</choice>.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _match_flow_edge(edges: list[FlowEdge], choice: str | None) -> str | None:
        if not choice:
            return None
        for edge in edges:
            if edge.label == choice:
                return edge.dst
        return None

    @staticmethod
    async def _flow_turn(
        soul: KimiSoul,
        prompt: str | list[ContentPart],
        *,
        enable_skill_reminder: bool = False,
    ) -> TurnOutcome:
        wire_send(TurnBegin(user_input=prompt))
        res = await soul._turn(  # pyright: ignore[reportPrivateUsage]
            Message(role="user", content=prompt),
            enable_skill_reminder=enable_skill_reminder,
        )
        wire_send(TurnEnd())
        return res
