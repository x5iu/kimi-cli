from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

import tenacity
from tenacity import RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential_jitter

import llmkit
from kimi_cli.background import build_active_task_snapshot
from kimi_cli.eventbus.log import EventLog
from kimi_cli.eventbus.types import (
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
from kimi_cli.llm import ModelCapability
from kimi_cli.loop import (
    AgentLoop,
    LLMNotSet,
    LLMNotSupported,
    MaxStepsReached,
    StatusSnapshot,
    bus_send,
    get_event_bus_or_none,
)
from kimi_cli.loop.agent import Agent, Runtime
from kimi_cli.loop.attachment import Attachment, AttachmentProvider, IncrementalHistoryNormalizer
from kimi_cli.loop.attachments.context_budget import ContextBudgetAttachmentProvider
from kimi_cli.loop.attachments.edit_verify import EditVerificationReminderProvider
from kimi_cli.loop.attachments.goal_tracking import GoalTrackingAttachmentProvider
from kimi_cli.loop.attachments.grep_read_nudge import GrepThenTargetedReadNudgeProvider
from kimi_cli.loop.attachments.post_compaction import PostCompactionContinuityAttachmentProvider
from kimi_cli.loop.attachments.prefer_shell_rg import PreferShellRgAttachmentProvider
from kimi_cli.loop.attachments.recall_nudge import RecallNudgeAfterCompactionProvider
from kimi_cli.loop.attachments.task_poll import TaskPollEscalationAttachmentProvider
from kimi_cli.loop.attachments.tool_storm import ToolStormBreakerAttachmentProvider
from kimi_cli.loop.compaction import (
    Compaction,
    CompactionResult,
    SimpleCompaction,
    estimate_text_tokens,
    should_auto_compact,
)
from kimi_cli.loop.compaction_archive import (
    backfill_archive_keywords,
    build_compaction_summary,
    load_compaction_archives,
    register_compaction_archive,
)
from kimi_cli.loop.context import Context
from kimi_cli.loop.message import (
    check_message,
    internal_user_message,
    system,
    tool_result_to_message,
)
from kimi_cli.loop.slash import registry as agent_loop_slash_registry
from kimi_cli.loop.toolset import KimiToolset, set_session_id
from kimi_cli.notifications import (
    NotificationView,
    build_notification_message,
    extract_notification_ids,
)
from kimi_cli.skill import Skill, normalize_skill_name, read_skill_text
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.utils.logging import logger
from kimi_cli.utils.message import content_parts_stringify
from kimi_cli.utils.slashcmd import SlashCommand, parse_slash_command_call
from kimi_cli.utils.turns import is_real_user_turn_start_message
from llmkit import StepResult
from llmkit.chat_provider import (
    APIConnectionError,
    APIEmptyResponseError,
    APIStatusError,
    APITimeoutError,
    RetryableChatProvider,
)
from llmkit.message import Message

if TYPE_CHECKING:

    def type_check(agent_loop: KimiAgentLoop):
        _: AgentLoop = agent_loop


SKILL_COMMAND_PREFIX = "skill:"
MAX_SKILL_RECOMMENDATIONS = 3

TURN_END_QUESTION_DETECTOR_PROMPT = (
    (Path(__file__).parent.parent / "prompts" / "turn_end_question_detector.md")
    .read_text(encoding="utf-8")
    .strip()
)

SKILL_RECOMMENDER_PROMPT = (
    (Path(__file__).parent.parent / "prompts" / "skill_recommender.md")
    .read_text(encoding="utf-8")
    .strip()
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
    is_skill: bool = False


class KimiAgentLoop:
    """The agent loop of Kimi Code CLI."""

    def __init__(
        self,
        agent: Agent,
        *,
        context: Context,
    ):
        """
        Initialize the agent loop.

        Args:
            agent (Agent): The agent to run.
            context (Context): The context of the agent.
        """
        self._agent = agent
        self._runtime = agent.runtime
        self._approval = agent.runtime.approval
        self._context = context
        self._loop_control = agent.runtime.config.loop_control
        self._compaction: Compaction = SimpleCompaction(
            max_preserved_messages=getattr(self._loop_control, "max_preserved_messages", 2)
        )
        # TODO: maybe configurable and composable
        self._last_compaction_turn: int | None = None
        self._suppress_auto_compaction: bool = False

        if self._runtime.llm is not None:
            mcs = self._runtime.llm.max_context_size
            rcs = self._loop_control.reserved_context_size
            if rcs >= mcs * 0.5:
                logger.warning(
                    "reserved_context_size ({rcs}) is >= 50% of "
                    "max_context_size ({mcs}); compaction may "
                    "trigger too aggressively",
                    rcs=rcs,
                    mcs=mcs,
                )

        self._checkpoint_with_user_message = False

        self._steer_queue: asyncio.Queue[_QueuedSteer] = asyncio.Queue()
        self._active_turn_id: int | None = None
        self._turn_steer_count: int = 0
        self._compaction_generation: int = 0
        self._next_turn_id = 0
        self._attachment_providers: list[AttachmentProvider] = [
            PreferShellRgAttachmentProvider(),
            GoalTrackingAttachmentProvider(),
            ContextBudgetAttachmentProvider(),
            PostCompactionContinuityAttachmentProvider(),
            TaskPollEscalationAttachmentProvider(),
            ToolStormBreakerAttachmentProvider(),
            GrepThenTargetedReadNudgeProvider(),
            RecallNudgeAfterCompactionProvider(),
            EditVerificationReminderProvider(),
        ]
        self._history_normalizer = IncrementalHistoryNormalizer()

        self._runtime.notifications.ack_ids("llm", extract_notification_ids(context.history))

        # Bind tool state that depends on the live agent loop/context
        self._bind_approval_aware_tools()
        self._bind_context_recall_tools()

        self._slash_command_content_parts: list[ContentPart] = []
        """Non-text content parts (e.g. images) attached to the current slash command."""

        self._slash_commands = self._build_slash_commands()
        self._slash_command_map = self._index_slash_commands(self._slash_commands)

        set_session_id(self._runtime.session.id)

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

    def _bind_approval_aware_tools(self) -> None:
        """Bind late-initialized runtime state to tools that need it."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return

        def yolo_checker() -> bool:
            return self._approval.is_yolo()

        from kimi_cli.tools.ask_user import AskUserQuestion

        ask_tool = self._agent.toolset.find("AskUserQuestion")
        if isinstance(ask_tool, AskUserQuestion):
            ask_tool.bind_approval(yolo_checker)

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
    def event_log(self) -> EventLog:
        return self._runtime.session.event_log

    async def _checkpoint(self, *, turn_start: bool = False):
        await self._context.checkpoint(self._checkpoint_with_user_message, turn_start=turn_start)

    async def undo_last_turn(self) -> bool:
        """Undo the last turn by reverting to its turn-level checkpoint.

        Returns True if a turn was undone, False if there was nothing to undo.
        """
        last_turn_cp = self._context.last_turn_checkpoint_id
        if last_turn_cp is None:
            return False

        logger.info("Undoing last turn, reverting to checkpoint {cp}", cp=last_turn_cp)
        await self._context.revert_to(last_turn_cp)

        # Sync dependent state
        self._sync_context_recall_tool_visibility()

        return True

    def _begin_turn(self) -> int:
        self._next_turn_id += 1
        self._active_turn_id = self._next_turn_id
        self._turn_steer_count = 0
        return self._next_turn_id

    def _end_turn(self, turn_id: int) -> None:
        if self._active_turn_id == turn_id:
            self._active_turn_id = None

    def steer(self, content: str | list[ContentPart], *, is_skill: bool = False) -> None:
        """Queue a steer message for injection into the current turn."""
        turn_id = self._active_turn_id
        if turn_id is None:
            logger.debug("Ignoring steer because there is no active turn")
            return
        self._steer_queue.put_nowait(
            _QueuedSteer(turn_id=turn_id, content=content, is_skill=is_skill)
        )

    async def _consume_pending_steers(self) -> bool:
        """Drain the steer queue and inject as synthetic tool results.

        Returns True if any steers were consumed.

        Note: /btw is handled in the shell UI and does not use the steer queue.
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
            await self._inject_steer(steer.content, is_skill=steer.is_skill)
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

    @staticmethod
    def _steer_instruction_text_brief() -> str:
        return "Additional user reminder (same handling rules as the previous reminder):"

    @staticmethod
    def _skill_steer_instruction_text() -> str:
        return (
            "The user activated a skill during the current turn. "
            "The skill content below contains instructions, reference material, and/or "
            "workflow patterns that you MUST read carefully and follow as primary directives "
            "for the remainder of this turn. "
            "Treat this skill content as the new primary instruction for this turn, "
            "superseding the original turn request. "
            "Proceed directly to follow the skill's instructions."
        )

    @classmethod
    def _build_steer_message_impl(
        cls,
        content: str | list[ContentPart],
        instruction: str,
        label: str,
        tag: str = "system-reminder",
    ) -> Message:
        content_parts: list[ContentPart] = [
            TextPart(
                text=(
                    f"<{tag}>\n"
                    f"{instruction}\n\n"
                    f"{label} follows in the rest of this message.\n"
                    f"</{tag}>"
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

    @classmethod
    def _build_steer_message(cls, content: str | list[ContentPart]) -> Message:
        return cls._build_steer_message_impl(
            content,
            instruction=cls._steer_instruction_text(),
            label="Reminder content",
        )

    @classmethod
    def _build_skill_steer_message(cls, content: str | list[ContentPart]) -> Message:
        return cls._build_steer_message_impl(
            content,
            instruction=cls._skill_steer_instruction_text(),
            label="Skill content",
            tag="skill-activation",
        )

    @staticmethod
    def _stringify_steer_content(content: str | list[ContentPart]) -> str:
        if isinstance(content, str):
            return content.strip()
        return content_parts_stringify(content).strip()

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
        if self._runtime.llm is None:
            raise LLMNotSet()
        if missing_caps := check_message(reminder_message, self._runtime.llm.capabilities):
            fallback_text = self._stringify_steer_content(content)
            if not fallback_text:
                if is_skill:
                    fallback_text = (
                        "The skill content included non-text content that is not supported "
                        "by the current model."
                    )
                else:
                    fallback_text = (
                        "The reminder included non-text content that is not supported "
                        "by the current model."
                    )
            instruction = (
                self._skill_steer_instruction_text() if is_skill else self._steer_instruction_text()
            )
            label = "Skill content" if is_skill else "Reminder"
            tag = "skill-activation" if is_skill else "system-reminder"
            reminder_message = internal_user_message(
                TextPart(text=(f"<{tag}>\n{instruction}\n\n{label}:\n{fallback_text}\n</{tag}>"))
            )
            if missing_caps := check_message(reminder_message, self._runtime.llm.capabilities):
                raise LLMNotSupported(self._runtime.llm, list(missing_caps))
        await self._context.append_message(reminder_message)

    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        return self._slash_commands

    async def run(self, user_input: str | list[ContentPart]):
        turn_id = self._begin_turn()
        turn_started = False
        turn_finished = False
        try:
            bus_send(TurnBegin(user_input=user_input))
            turn_started = True
            user_message = Message(role="user", content=user_input)
            text_input = user_message.extract_text(" ").strip()

            outcome: TurnOutcome | None = None
            if command_call := parse_slash_command_call(text_input):
                command = self._find_slash_command(command_call.name)
                if command is None:
                    # this should not happen actually, the shell should have filtered it out
                    bus_send(TextPart(text=f'Unknown slash command "/{command_call.name}".'))
                else:
                    # Stash non-text content parts (e.g. images) so slash-command
                    # handlers like skill runners can include them in their turn.
                    if isinstance(user_message.content, list):  # pyright: ignore[reportUnnecessaryIsInstance]
                        self._slash_command_content_parts = [
                            p for p in user_message.content if not isinstance(p, TextPart)
                        ]
                    else:
                        # `run()` was called with a plain str (e.g. from tests or
                        # non-shell callers), so there are no rich content parts.
                        self._slash_command_content_parts = []
                    try:
                        ret = command.func(self, command_call.args)
                        if isinstance(ret, Awaitable):
                            await ret
                    finally:
                        self._slash_command_content_parts = []
            else:
                outcome = await self._turn(user_message, enable_skill_reminder=True)

            # Turn-end question detection: if enabled and the turn produced a final
            # message, check whether it asks the user to pick between options.
            if outcome is not None and self._loop_control.turn_end_question_detection:
                answer = await self._maybe_ask_turn_end_question(outcome)
                if answer:
                    # The user chose an option — echo their selection in the TUI
                    # and run a follow-up turn with their answer as the prompt.
                    bus_send(FollowUpInput(text=answer))
                    await self._turn(
                        Message(role="user", content=answer),
                    )

            bus_send(TurnEnd())
            turn_finished = True
        finally:
            if turn_started and not turn_finished:
                bus_send(TurnEnd())
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

        await self._checkpoint(turn_start=True)
        await self._context.append_message(user_message)
        logger.debug("Appended user message to context")

        skill_reminder = self._start_skill_reminder_task() if enable_skill_reminder else None
        try:
            return await self._agent_loop(skill_reminder)
        finally:
            await self._finalize_skill_reminder_task(skill_reminder)

    def _build_slash_commands(self) -> list[SlashCommand[Any]]:
        commands: list[SlashCommand[Any]] = list(agent_loop_slash_registry.list_commands())
        seen_names = {cmd.name for cmd in commands}

        for skill in self._runtime.skills.values():
            if skill.type != "standard":
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

    def _make_skill_runner(
        self, skill: Skill
    ) -> Callable[[KimiAgentLoop, str], None | Awaitable[None]]:
        async def _run_skill(
            agent_loop: KimiAgentLoop, args: str, *, _skill: Skill = skill
        ) -> None:
            skill_text = await read_skill_text(_skill)
            if skill_text is None:
                bus_send(
                    TextPart(text=f'Failed to load skill "/{SKILL_COMMAND_PREFIX}{_skill.name}".')
                )
                return
            extra = args.strip()
            if extra:
                skill_text = f"{skill_text}\n\nUser request:\n{extra}"
            # Include non-text content parts (e.g. images) that were attached
            # to the slash command invocation.
            extra_parts = list(agent_loop._slash_command_content_parts)
            content: str | list[ContentPart] = (
                [TextPart(text=skill_text), *extra_parts] if extra_parts else skill_text
            )
            await agent_loop._turn(Message(role="user", content=content))

        _run_skill.__doc__ = skill.description
        return _run_skill

    def _start_skill_reminder_task(self) -> SkillReminderState | None:
        if self._runtime.llm is None or not self._runtime.skills:
            return None
        full_history = list(self._context.history)
        if len(full_history) > 11:
            # Find first user message
            first_user_idx = next((i for i, m in enumerate(full_history) if m.role == "user"), 0)
            tail_start = len(full_history) - 10
            if first_user_idx >= tail_start:
                # First user message is already in the tail — no need to prepend
                history = full_history[-10:]
            else:
                history = [full_history[first_user_idx]] + full_history[-10:]
        else:
            history = full_history
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

        effective_history: list[Message] = []
        if self._runtime.agents_md:
            effective_history.append(
                internal_user_message(
                    system(
                        "The following AGENTS.md instructions are system-level directives "
                        "that provide project context, conventions, and user preferences. "
                        "Consider them when recommending skills.\n\n"
                        f"{self._runtime.agents_md}"
                    )
                )
            )
        effective_history.extend(history)

        async def _run_once():
            return await llmkit.generate(
                chat_provider=chat_provider,
                system_prompt=(
                    f"{SKILL_RECOMMENDER_PROMPT}{self._format_available_skills_for_recommender()}\n"
                ),
                tools=[],
                history=effective_history,
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
        bus_send(
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
    _TURN_END_DETECT_CONTEXT_TURNS = 3
    _TURN_END_DETECT_CONTEXT_CHARS = 600
    _TURN_END_DETECT_EXCERPT_CHARS = 600
    _TURN_END_DETECT_LATEST_MESSAGE_CHARS = 2000

    async def _detect_turn_end_question(
        self,
        assistant_message: Message,
    ) -> TurnEndQuestionDetection | None:
        """Use a side-channel LLM call to check if *assistant_message* asks the user
        to choose between options.  Returns the parsed detection or ``None`` on
        failure.  Retries up to ``_TURN_END_DETECT_MAX_ATTEMPTS`` when the LLM
        returns unparsable output.  The entire detection is capped at
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

    def _clip_turn_end_detector_text(self, text: str, *, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        separator = "\n...\n"
        keep = max(1, (max_chars - len(separator)) // 2)
        return f"{text[:keep].rstrip()}{separator}{text[-keep:].lstrip()}"

    def _recent_turn_end_detection_context(
        self,
        assistant_message: Message,
        *,
        max_turns: int,
    ) -> list[tuple[str, str]]:
        history = self._context.history
        messages = reversed(history)
        if not history or history[-1] != assistant_message:
            messages = chain((assistant_message,), messages)

        turns_reversed: list[tuple[str, str]] = []
        current_assistant: str | None = None

        for msg in messages:
            if msg.role == "assistant" and current_assistant is None:
                current_assistant = msg.extract_text(sep="\n").strip()

            if not is_real_user_turn_start_message(msg):
                continue

            turns_reversed.append((msg.extract_text(sep="\n").strip(), current_assistant or ""))
            current_assistant = None
            if len(turns_reversed) >= max_turns:
                break

        turns_reversed.reverse()
        return turns_reversed

    def _build_turn_end_detector_prompt_input(self, assistant_message: Message) -> str:
        text_only = assistant_message.extract_text(" ").strip()
        excerpt = self._clip_turn_end_detector_text(
            self._turn_end_question_excerpt(text_only),
            max_chars=self._TURN_END_DETECT_EXCERPT_CHARS,
        )
        latest_message = self._clip_turn_end_detector_text(
            text_only,
            max_chars=self._TURN_END_DETECT_LATEST_MESSAGE_CHARS,
        )
        recent_turns = self._recent_turn_end_detection_context(
            assistant_message,
            max_turns=self._TURN_END_DETECT_CONTEXT_TURNS,
        )
        if recent_turns and recent_turns[-1][1] == text_only and text_only:
            recent_turns = [
                *recent_turns[:-1],
                (recent_turns[-1][0], "(latest message shown below)"),
            ]

        lines = [
            (
                "Analyze whether the latest assistant message asks the user to choose "
                "between options or make a decision."
            ),
            (
                "Use recent turns only as supporting context. Base has_question on "
                "the latest assistant message, not on older turns."
            ),
        ]
        if recent_turns:
            lines.extend(
                [
                    "",
                    f"Recent turns (last {len(recent_turns)}, oldest to newest):",
                ]
            )
            for idx, (user_text, assistant_text) in enumerate(recent_turns, start=1):
                clipped_user = self._clip_turn_end_detector_text(
                    user_text,
                    max_chars=self._TURN_END_DETECT_CONTEXT_CHARS,
                )
                clipped_assistant = self._clip_turn_end_detector_text(
                    assistant_text,
                    max_chars=self._TURN_END_DETECT_CONTEXT_CHARS,
                )
                lines.extend(
                    [
                        "",
                        f"[Turn {idx}]",
                        f"User:\n{clipped_user or '(empty)'}",
                        f"Assistant:\n{clipped_assistant or '(no textual reply)'}",
                    ]
                )

        lines.extend(
            [
                "",
                (
                    "Focus on the ending of the latest assistant message, but use "
                    "the full latest message if earlier lines contain the options."
                ),
                "",
                "Latest message ending excerpt:",
                excerpt,
                "",
                "Latest full assistant message (trimmed if needed):",
                latest_message,
            ]
        )
        return "\n".join(lines)

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
        text_only = assistant_message.extract_text(" ").strip()
        if not text_only:
            return None
        heuristic_detection = self._heuristic_turn_end_question(text_only)
        history: list[Message] = [
            Message(
                role="user",
                content=self._build_turn_end_detector_prompt_input(assistant_message),
            )
        ]

        for attempt in range(1, self._TURN_END_DETECT_MAX_ATTEMPTS + 1):

            async def _run_once():
                return await llmkit.generate(
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
                    "Turn-end question detection returned unparsable output "
                    "(attempt {attempt}), retrying",
                    attempt=attempt,
                )
            else:
                logger.warning(
                    "Turn-end question detection returned unparsable output "
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

        bus = get_event_bus_or_none()
        if bus is None:
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

        bus_send(request)

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
                bus_send(MCPLoadingBegin())
            try:
                await self._agent.toolset.wait_for_mcp_tools()
            finally:
                if loading:
                    bus_send(MCPLoadingEnd())

        async def _pipe_approval_to_bus():
            while True:
                request = await self._approval.fetch_request()
                # Here we decouple the bus approval request and the loop approval request.
                bus_request = ApprovalRequest(
                    id=request.id,
                    action=request.action,
                    description=request.description,
                    sender=request.sender,
                    tool_call_id=request.tool_call_id,
                    display=request.display,
                )
                bus_send(bus_request)
                # We wait for the request to be resolved over the bus, which means that,
                # for each agent loop, we will have only one approval request waiting on the
                # bus at a time. However, be aware that subagents (which have their own loops)
                # may also send approval requests to the root bus.
                resp = await bus_request.wait()
                self._approval.resolve_request(request.id, resp)
                bus_send(ApprovalResponse(request_id=request.id, response=resp))

        step_no = 0
        while True:
            step_no += 1
            if step_no > self._loop_control.max_steps_per_turn:
                raise MaxStepsReached(self._loop_control.max_steps_per_turn)

            bus_send(StepBegin(n=step_no))
            approval_task = asyncio.create_task(_pipe_approval_to_bus())
            step_outcome: StepOutcome | None = None
            try:
                if await self._consume_ready_skill_reminder(skill_reminder):
                    logger.debug("Skill reminder was ready before step {step_no}", step_no=step_no)
                # compact the context if needed
                if (
                    not self._suppress_auto_compaction
                    and getattr(
                        self._loop_control,
                        "auto_compact_enabled",
                        True,
                    )
                    and should_auto_compact(
                        self._context.token_count_with_pending,
                        self._runtime.llm.max_context_size,
                        trigger_ratio=self._loop_control.compaction_trigger_ratio,
                        reserved_context_size=self._loop_control.reserved_context_size,
                    )
                ):
                    if (
                        self._last_compaction_turn is not None
                        and self._active_turn_id is not None
                        and self._active_turn_id - self._last_compaction_turn < 2
                    ):
                        logger.debug("Skipping auto-compaction (cooldown)")
                    else:
                        logger.info("Context too long, compacting...")
                        try:
                            await self.compact_context()
                            self._last_compaction_turn = self._active_turn_id
                        except Exception as compact_err:
                            logger.error(
                                "Context compaction failed at step {step_no}: "
                                    "{error_type}: {error}",
                                step_no=step_no,
                                error_type=type(compact_err).__name__,
                                error=compact_err,
                            )
                            logger.opt(exception=True).warning(
                                "Auto-compaction failed, continuing without compaction"
                            )

                logger.debug("Beginning step {step_no}", step_no=step_no)
                await self._checkpoint()
                step_outcome = await self._step()
            except Exception as e:
                req_id = getattr(e, "request_id", None)
                logger.error(
                    "Agent step {step_no} failed: {error_type}: {error}"
                    + (" (request_id={request_id})" if req_id else ""),
                    step_no=step_no,
                    error_type=type(e).__name__,
                    error=e,
                    request_id=req_id,
                )
                bus_send(StepInterrupted())
                raise
            finally:
                approval_task.cancel()  # stop piping approval requests to the bus
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

            # Consume any pending steers between steps
            await self._consume_pending_steers()

    async def _step(self) -> StepOutcome | None:
        """Run a single step and return a stop outcome, or None to continue."""
        # already checked in `run`
        assert self._runtime.llm is not None
        chat_provider = self._runtime.llm.chat_provider

        async def _append_notification(view: NotificationView) -> None:
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
                (
                    f"<{'system-hint' if att.is_hint else 'system-reminder'}>\n"
                    f"{att.content}\n"
                    f"</{'system-hint' if att.is_hint else 'system-reminder'}>"
                )
                for att in attachments
            )
            await self._context.append_message(internal_user_message([TextPart(text=combined)]))

        # Normalize: merge adjacent user messages for clean API input
        effective_history = self._history_normalizer.normalize(self._context.history)

        async def _run_step_once() -> StepResult:
            # run an LLM step (may be interrupted)
            return await llmkit.step(
                chat_provider,
                self._agent.system_prompt,
                self._agent.toolset,
                effective_history,
                on_message_part=bus_send,
                on_tool_result=bus_send,
            )

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=partial(self._retry_log, "step"),
            wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),
            reraise=True,
        )
        async def _llmkit_step_with_retry() -> StepResult:
            return await self._run_with_connection_recovery(
                "step",
                _run_step_once,
                chat_provider=chat_provider,
            )

        t0 = time.monotonic()
        result = await _llmkit_step_with_retry()
        llm_elapsed = time.monotonic() - t0
        usage = result.usage
        logger.info(
            "LLM step completed in {elapsed:.1f}s (input={input_tokens}, output={output_tokens})",
            elapsed=llm_elapsed,
            input_tokens=usage.input if usage else "?",
            output_tokens=usage.output if usage else "?",
        )
        status_update = StatusUpdate(token_usage=usage, message_id=result.id)
        if usage is not None:
            await self._context.update_token_count(usage.input)
            snap = self.status
            status_update.context_usage = snap.context_usage
            status_update.context_tokens = snap.context_tokens
            status_update.max_context_tokens = snap.max_context_tokens
        bus_send(status_update)

        # wait for all tool results (may be interrupted)
        results = await result.tool_results()
        logger.debug("Got tool results: {results}", results=results)

        # shield the context manipulation from interruption
        await asyncio.shield(self._grow_context(result, results))

        rejected = any(isinstance(result.return_value, ToolRejectedError) for result in results)
        if rejected:
            return StepOutcome(stop_reason="tool_rejected", assistant_message=result.message)

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
        # Persist is_error flag alongside tool result messages in context JSONL
        tool_metadata: list[dict[str, object]] = [
            {"is_error": tr.return_value.is_error} for tr in tool_results
        ]
        await self._context.append_message(tool_messages, message_metadata=tool_metadata)
        # Estimate tool result tokens so the compaction trigger
        # has a more accurate count
        if tool_messages:
            tool_token_estimate = estimate_text_tokens(tool_messages)
            current = self._context.token_count
            if current > 0:
                await self._context.update_token_count(current + tool_token_estimate)

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

        bus_send(CompactionBegin())
        try:
            original_message_count = len(self._context.history)
            compaction_result = await _compact_with_retry()
            pre_compaction_messages = list(self._context.history)
            rotated_path = await self._context.clear()

            final_messages = list(compaction_result.messages)
            if compaction_result.usage is not None:
                registration = register_compaction_archive(
                    self._context.file_backend,
                    rotated_path,
                    messages=pre_compaction_messages,
                    message_count=original_message_count,
                    summary=build_compaction_summary(compaction_result.messages),
                )
                keywords_info = ""
                if registration.record.keywords:
                    keywords_info = f" Key topics: {', '.join(registration.record.keywords[:8])}."

                # Build an overview of ALL archives (not just the latest).
                all_archives = load_compaction_archives(self._context.file_backend)
                archive_overview_lines: list[str] = []
                newest_id = registration.record.id
                recent_archives = all_archives[-5:]
                older_count = len(all_archives) - len(recent_archives)
                if older_count > 0:
                    archive_overview_lines.append(
                        f"  ({older_count} older archive(s) not shown"
                        " — use RecallCompactedContext to search them)"
                    )
                for ar in recent_archives:
                    summary_preview = (
                        ar.summary[:80] + "..." if len(ar.summary) > 80 else ar.summary
                    )
                    tag = " [NEW]" if ar.id == newest_id else ""
                    archive_overview_lines.append(
                        f"- {ar.id} ({ar.message_count} msgs): {summary_preview}{tag}"
                    )
                archive_overview = ""
                if archive_overview_lines:
                    archive_overview = "\n\nArchive overview:\n" + "\n".join(archive_overview_lines)

                final_messages.append(
                    internal_user_message(
                        [
                            system(
                                "Compacted context archives are available via "
                                "the RecallCompactedContext tool for this "
                                "conversation trajectory. "
                                f"Archive `{registration.record.id}` contains "
                                "the pre-compaction history. "
                                f"There are now "
                                f"{registration.total_archives} compacted "
                                "archive(s) available."
                                f"{keywords_info} "
                                "Use RecallCompactedContext with targeted "
                                "keywords when you need specific details "
                                "from earlier context — exact error messages, "
                                "file paths, code snippets, function names, "
                                "or design decisions. Prefer this tool over "
                                "guessing or asking the user to repeat "
                                "themselves."
                                f"{archive_overview}"
                            )
                        ]
                    )
                )

            # Inject current todo state so the agent retains awareness after compaction.
            todos = self._runtime.session.state.todos
            if todos:
                todo_lines = [f"- [{t.status}] {t.title}" for t in todos]
                final_messages.append(
                    internal_user_message(
                        [
                            system(
                                "Your current todo list survived compaction. "
                                "Review it before creating a new one.\n" + "\n".join(todo_lines)
                            )
                        ]
                    )
                )

            self._sync_context_recall_tool_visibility()
            self._compaction_generation += 1
            self._turn_steer_count = 0

            # Backfill keywords for any older archives that were created before
            # keyword extraction was implemented.
            try:
                await asyncio.to_thread(backfill_archive_keywords, self._context.file_backend)
            except Exception:
                logger.opt(exception=True).debug("Failed to backfill archive keywords")

            try:
                await self._checkpoint()
                await self._context.append_message(final_messages)
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to append compacted messages; restoring pre-compaction context"
                )
                try:
                    rotated_path.replace(self._context.file_backend)
                except Exception:
                    logger.opt(exception=True).warning("Failed to restore rotated context file")
                raise

            estimated_token_count = CompactionResult(
                messages=final_messages, usage=compaction_result.usage
            ).estimated_token_count

            active_task_snapshot = build_active_task_snapshot(self._runtime.background_tasks)
            if active_task_snapshot is not None:
                active_task_message = Message(
                    role="user",
                    content=[
                        system(
                            "The following background tasks are still "
                            "active after compaction. Use TaskList if "
                            "you need to re-enumerate them later."
                        ),
                        TextPart(text=active_task_snapshot),
                    ],
                )
                await self._context.append_message(active_task_message)
                estimated_token_count += estimate_text_tokens([active_task_message])

            # Estimate token count so context_usage is not reported as 0%
            await self._context.update_token_count(estimated_token_count)

            # Send StatusUpdate so the UI reflects the reduced context
            snap = self.status
            bus_send(
                StatusUpdate(
                    context_usage=snap.context_usage,
                    context_tokens=snap.context_tokens,
                    max_context_tokens=snap.max_context_tokens,
                )
            )
        finally:
            bus_send(CompactionEnd())

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
            504,  # Gateway Timeout
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
                logger.warning(
                    "Chat provider recovery not available for {name} after {error_type}.",
                    name=name,
                    error_type=type(error).__name__,
                )
                raise
            logger.info(
                "Recovered chat provider during {name} after {error_type}; retrying once.",
                name=name,
                error_type=type(error).__name__,
            )
            try:
                return await operation()
            except (APIConnectionError, APITimeoutError) as second_error:
                logger.warning(
                    "Chat provider recovery exhausted for {name}: {error_type}: {error}",
                    name=name,
                    error_type=type(second_error).__name__,
                    error=second_error,
                )
                second_error._kimi_recovery_exhausted = True  # type: ignore[attr-defined]
                raise

    @staticmethod
    def _retry_log(name: str, retry_state: RetryCallState):
        error = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "Retrying {name} for the {n} time (last error: {error_type}: {error}). "
            "Waiting {sleep} seconds.",
            name=name,
            n=retry_state.attempt_number,
            error_type=type(error).__name__ if error else "unknown",
            error=error or "unknown",
            sleep=retry_state.next_action.sleep
            if retry_state.next_action is not None
            else "unknown",
        )
