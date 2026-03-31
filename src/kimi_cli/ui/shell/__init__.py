from __future__ import annotations

import asyncio
import shlex
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kosong.chat_provider import APIStatusError, ChatProviderError
from kosong.message import Message
from loguru import logger
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kimi_cli.notifications import NotificationWatcher
from kimi_cli.skill import normalize_skill_name
from kimi_cli.soul import LLMNotSet, LLMNotSupported, MaxStepsReached, RunCancelled, Soul, run_soul
from kimi_cli.soul.input_validation import validate_live_user_input
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.ui.shell.console import console
from kimi_cli.ui.shell.prompt import (
    CustomPromptSession,
    PromptMode,
    TurnSubmitResult,
    UserInput,
)
from kimi_cli.ui.shell.replay import replay_recent_history
from kimi_cli.ui.shell.slash import SKILL_PREFIX as _SKILL_PREFIX
from kimi_cli.ui.shell.slash import TURN_ALLOWED_COMMANDS as _TURN_ALLOWED_COMMANDS
from kimi_cli.ui.shell.slash import registry as shell_slash_registry
from kimi_cli.ui.shell.slash import shell_mode_registry
from kimi_cli.ui.shell.toast import toast
from kimi_cli.ui.shell.visualize import LiveView, render_user_prompt_block, visualize
from kimi_cli.utils.logging import open_original_stderr
from kimi_cli.utils.message import message_stringify
from kimi_cli.utils.signals import install_sigint_handler
from kimi_cli.utils.slashcmd import SlashCommand, SlashCommandCall, parse_slash_command_call
from kimi_cli.utils.subprocess_env import get_clean_env
from kimi_cli.utils.term import ensure_new_line, ensure_tty_sane
from kimi_cli.wire.types import ContentPart, StatusUpdate


class Shell:
    def __init__(self, soul: Soul, welcome_info: list[WelcomeInfoItem] | None = None):
        self.soul = soul
        self._welcome_info = list(welcome_info or [])
        self._background_tasks: set[asyncio.Task[Any]] = set()
        commands = [*soul.available_slash_commands, *shell_slash_registry.list_commands()]
        self._available_slash_commands: dict[str, SlashCommand[Any]] = {
            cmd.name: cmd for cmd in commands
        }
        self._slash_command_lookup: dict[str, SlashCommand[Any]] = self._index_slash_commands(
            commands
        )
        """Shell-level slash commands + soul-level slash commands. Name to command mapping."""

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

    @property
    def available_slash_commands(self) -> dict[str, SlashCommand[Any]]:
        """Get all available slash commands, including shell-level and soul-level commands."""
        return self._available_slash_commands

    async def run(self, command: str | None = None) -> bool:
        if command is not None:
            # run single command and exit
            logger.info("Running agent with command: {command}", command=command)
            return await self.run_soul_command(command)

        _print_welcome_info(self.soul.name or "Kimi Code CLI", self._welcome_info)

        if isinstance(self.soul, KimiSoul):
            watcher = NotificationWatcher(
                self.soul.runtime.notifications,
                sink="shell",
                before_poll=self.soul.runtime.background_tasks.reconcile,
                on_notification=lambda notification: toast(
                    f"[{notification.event.type}] {notification.event.title}",
                    topic="notification",
                    duration=10.0,
                ),
            )
            self._start_background_task(watcher.run_forever())

        if isinstance(self.soul, KimiSoul):
            await replay_recent_history(
                self.soul.context.history,
                wire_file=self.soul.wire_file,
            )

        async def _plan_mode_toggle() -> bool:
            if isinstance(self.soul, KimiSoul):
                return await self.soul.toggle_plan_mode_from_manual()
            return False

        async def _redraw() -> None:
            if isinstance(self.soul, KimiSoul):
                await replay_recent_history(
                    self.soul.context.history,
                    wire_file=self.soul.wire_file,
                )

        with CustomPromptSession(
            status_provider=lambda: self.soul.status,
            model_capabilities=self.soul.model_capabilities or set(),
            model_name=self.soul.model_name,
            thinking=self.soul.thinking or False,
            agent_mode_slash_commands=list(self._available_slash_commands.values()),
            shell_mode_slash_commands=shell_mode_registry.list_commands(),
            editor_command_provider=lambda: (
                self.soul.runtime.config.default_editor if isinstance(self.soul, KimiSoul) else ""
            ),
            plan_mode_toggle_callback=_plan_mode_toggle,
            working_dir_provider=self._working_dir_text,
            redraw_callback=_redraw,
        ) as prompt_session:
            _MAX_BG_AUTO_TRIGGER_FAILURES = 3
            bg_auto_failures = 0
            try:
                while True:
                    ensure_tty_sane()

                    # Auto-trigger: check for pending LLM notifications from
                    # completed background tasks before showing the prompt.
                    if (
                        isinstance(self.soul, KimiSoul)
                        and bg_auto_failures < _MAX_BG_AUTO_TRIGGER_FAILURES
                    ):
                        self.soul.runtime.background_tasks.reconcile()
                        if self.soul.runtime.notifications.has_pending_for_sink("llm"):
                            logger.info(
                                "Background task completed while idle, auto-triggering agent"
                            )
                            ok = await self.run_soul_command(
                                "<system-reminder>"
                                "Background tasks completed while you were idle."
                                "</system-reminder>"
                            )
                            console.print()
                            if not ok:
                                bg_auto_failures += 1
                                logger.warning(
                                    "Background auto-trigger failed ({n}/{max})",
                                    n=bg_auto_failures,
                                    max=_MAX_BG_AUTO_TRIGGER_FAILURES,
                                )
                                await asyncio.sleep(2)
                            else:
                                bg_auto_failures = 0
                            continue

                    try:
                        ensure_new_line()
                        user_input = await prompt_session.prompt()
                    except KeyboardInterrupt:
                        logger.debug("Exiting by KeyboardInterrupt")
                        prompt_session.execute_deferred_erase()
                        console.print("[grey50]Tip: press Ctrl-D or send 'exit' to quit[/grey50]")
                        continue
                    except EOFError:
                        logger.debug("Exiting by EOF")
                        prompt_session.execute_deferred_erase()
                        console.print("Bye!")
                        break

                    if not user_input:
                        logger.debug("Got empty input, skipping")
                        prompt_session.execute_deferred_erase()
                        continue
                    logger.debug("Got user input: {user_input}", user_input=user_input)
                    bg_auto_failures = 0

                    if not await self._handle_agent_input(prompt_session, user_input):
                        break
            finally:
                self._cancel_background_tasks()
                ensure_tty_sane()

        return True

    @staticmethod
    def _display_user_input(user_input: UserInput) -> str:
        return message_stringify(Message(role="user", content=user_input.content))

    @staticmethod
    def _display_turn_prompt(user_input: str | list[ContentPart]) -> str:
        if isinstance(user_input, str):
            return user_input
        return message_stringify(Message(role="user", content=user_input))

    @staticmethod
    def _pre_render_user_echo(text: str) -> str:
        """Pre-render the user prompt bubble to an ANSI string.

        Separating the Rich render from the actual stdout write lets us
        do the heavy work *before* the idle prompt erases, then flush the
        pre-built bytes in a single fast write right after the erase —
        minimizing the visible gap where no input box is shown.
        """
        from io import StringIO

        from rich.console import Console as _C

        sio = StringIO()
        _C(
            file=sio,
            force_terminal=True,
            color_system=console.color_system,
            width=console.width,
            highlight=False,
        ).print(render_user_prompt_block(text))
        return sio.getvalue()

    def _echo_agent_input(self, user_input: UserInput) -> None:
        if user_input.mode != PromptMode.AGENT:
            return
        console.print(render_user_prompt_block(self._display_user_input(user_input)))

    async def _handle_agent_input(
        self,
        prompt_session: CustomPromptSession,
        user_input: UserInput,
        *,
        echo: bool = True,
    ) -> bool:
        if user_input.command in ["exit", "quit", "/exit", "/quit"]:
            logger.debug("Exiting by slash command")
            prompt_session.execute_deferred_erase()
            console.print("Bye!")
            return False

        if user_input.mode == PromptMode.SHELL:
            prompt_session.execute_deferred_erase()
            await self._run_shell_command(user_input.command)
            return True

        slash_cmd_call = parse_slash_command_call(user_input.command)
        if (
            slash_cmd_call is not None
            and shell_slash_registry.find_command(slash_cmd_call.name) is not None
        ):
            prompt_session.execute_deferred_erase()
            await self._run_slash_command(slash_cmd_call)
            return True

        # Pre-render the user echo *before* the turn starts.  The prompt
        # Application is still on-screen at this point so the Rich render
        # cost is hidden.  We pass the pre-rendered ANSI string into
        # run_turn_ui which writes it via a fast sys.stdout.write right
        # after the idle prompt erases — keeping the gap minimal.
        pre_rendered_echo = ""
        if echo and user_input.mode == PromptMode.AGENT:
            pre_rendered_echo = self._pre_render_user_echo(self._display_user_input(user_input))

        if slash_cmd_call is not None and slash_cmd_call.name in self._slash_command_lookup:
            soul_input: str | list[ContentPart] = slash_cmd_call.raw_input
        else:
            soul_input: str | list[ContentPart] = user_input.content
        keep_running = await self._run_interactive_turn(
            prompt_session,
            soul_input,
            pre_rendered_echo=pre_rendered_echo,
        )
        console.print()
        return keep_running

    def _working_dir_text(self) -> str:
        if isinstance(self.soul, KimiSoul):
            return str(self.soul.runtime.session.work_dir)
        return "."

    def _initial_status_update(self) -> StatusUpdate:
        snap = self.soul.status
        return StatusUpdate(
            context_usage=snap.context_usage,
            context_tokens=snap.context_tokens,
            max_context_tokens=snap.max_context_tokens,
        )

    async def _run_interactive_turn(
        self,
        prompt_session: CustomPromptSession,
        user_input: str | list[ContentPart],
        *,
        pre_rendered_echo: str = "",
    ) -> bool:
        logger.info(
            "Running interactive soul turn with user input: {user_input}",
            user_input=user_input,
        )
        cancel_event = asyncio.Event()
        live_view = LiveView(
            self._initial_status_update(),
            cancel_event=cancel_event,
            flush_to_console=False,
            allow_expand=True,
        )
        queued_input: UserInput | None = None

        def _submit_handler(turn_input: UserInput) -> TurnSubmitResult:
            nonlocal queued_input
            text = turn_input.command.strip()
            if not text:
                return TurnSubmitResult.reject()
            if live_view.has_pending_input_request:
                accepted = live_view.try_submit_line(text)
                return TurnSubmitResult.accept() if accepted else TurnSubmitResult.reject()
            if text in {"exit", "quit"}:
                queued_input = turn_input
                cancel_event.set()
                return TurnSubmitResult.accept()
            slash_call = parse_slash_command_call(text)
            if slash_call is not None and slash_call.name in _TURN_ALLOWED_COMMANDS:
                cmd = shell_mode_registry.find_command(slash_call.name)
                if cmd is not None:
                    try:
                        with console.capture() as capture:
                            ret = cmd.func(self, slash_call.args)
                        if isinstance(ret, Awaitable):
                            ret.close()  # prevent 'coroutine never awaited' warning
                            logger.error(
                                "Async command /%s cannot run during a turn",
                                slash_call.name,
                            )
                            live_view.echo_info("Error: this command cannot run during a turn")
                            return TurnSubmitResult.accept()
                        output = capture.get().strip()
                        if output:
                            live_view.echo_info(output)
                    except Exception as e:
                        logger.exception("Slash command error during turn:")
                        live_view.echo_info(f"Error: {e}")
                    return TurnSubmitResult.accept()
                logger.warning(
                    "Turn-allowed command /%s not found in shell registry",
                    slash_call.name,
                )
                return TurnSubmitResult.accept()
            if slash_call is not None and slash_call.name.startswith(_SKILL_PREFIX):
                # During a turn, inject skill content via steer so the
                # model can perceive the skill within the current turn.
                if isinstance(self.soul, KimiSoul):
                    skill_name = normalize_skill_name(slash_call.name[len(_SKILL_PREFIX) :])
                    skill = self.soul.runtime.skills.get(skill_name)
                    if skill is not None:
                        from pathlib import Path

                        try:
                            skill_text = (
                                Path(str(skill.skill_md_file)).read_text(encoding="utf-8").strip()
                            )
                            extra = slash_call.args.strip()
                            if extra:
                                skill_text = f"{skill_text}\n\nUser request:\n{extra}"
                            self.soul.steer(skill_text, is_skill=True)
                            live_view.echo_reminder(self._display_user_input(turn_input))
                            return TurnSubmitResult.accept(persist_history=True)
                        except OSError:
                            live_view.echo_info(f"Error: failed to load skill {skill_name}")
                            return TurnSubmitResult.accept()
                # Fallback: cancel turn and queue for post-turn execution
                queued_input = turn_input
                cancel_event.set()
                return TurnSubmitResult.accept()
            if not isinstance(self.soul, KimiSoul):
                return TurnSubmitResult.reject()
            try:
                validate_live_user_input(self.soul.runtime.llm, turn_input.content)
            except LLMNotSet:
                return TurnSubmitResult.reject('LLM not set, send "/login" to login')
            except LLMNotSupported as e:
                return TurnSubmitResult.reject(str(e))
            self.soul.steer(turn_input.content)
            live_view.echo_reminder(self._display_user_input(turn_input))
            return TurnSubmitResult.accept(persist_history=True)

        def _cancel_handler() -> None:
            cancel_event.set()

        keep_running = True
        try:
            await run_soul(
                self.soul,
                user_input,
                lambda wire: prompt_session.run_turn_ui(
                    wire=wire.ui_side(merge=False),
                    live_view=live_view,
                    submit_handler=_submit_handler,
                    cancel_handler=_cancel_handler,
                    turn_prompt=self._display_turn_prompt(user_input),
                    pre_rendered_echo=pre_rendered_echo,
                ),
                cancel_event,
                self.soul.wire_file if isinstance(self.soul, KimiSoul) else None,
            )
        except LLMNotSet:
            logger.exception("LLM not set:")
            console.print('[red]LLM not set, send "/login" to login[/red]')
            keep_running = False
        except LLMNotSupported as e:
            logger.exception("LLM not supported:")
            console.print(f"[red]{e}[/red]")
            keep_running = False
        except ChatProviderError as e:
            logger.exception("LLM provider error:")
            if isinstance(e, APIStatusError) and e.status_code == 401:
                console.print(
                    f"[red]Authorization failed, please check your login status: {e}[/red]"
                )
            elif isinstance(e, APIStatusError) and e.status_code == 402:
                console.print("[red]Membership expired, please renew your plan[/red]")
            elif isinstance(e, APIStatusError) and e.status_code == 403:
                console.print(
                    f"[red]Quota exceeded, please upgrade your plan or retry later: {e}[/red]"
                )
            else:
                console.print(f"[red]LLM provider error: {e}[/red]")
            keep_running = True
        except MaxStepsReached as e:
            logger.warning("Max steps reached: {n_steps}", n_steps=e.n_steps)
            console.print(f"[yellow]{e}[/yellow]")
            keep_running = False
        except RunCancelled:
            logger.info("Cancelled by user")
            if queued_input is None:
                console.print("[red]Interrupted by user[/red]")
        except Exception as e:
            logger.exception("Unexpected error:")
            console.print(f"[red]Unexpected error: {e}[/red]")
            raise

        if queued_input is not None:
            return await self._handle_agent_input(prompt_session, queued_input)
        return keep_running

    def _build_live_view(self, cancel_event: asyncio.Event) -> LiveView:
        return LiveView(self._initial_status_update(), cancel_event=cancel_event)

    async def _run_soul_command_with_ui(
        self,
        user_input: str | list[ContentPart],
        *,
        cancel_event: asyncio.Event,
        live_view: LiveView,
    ) -> bool:
        logger.info("Running soul with user input: {user_input}", user_input=user_input)

        try:
            await run_soul(
                self.soul,
                user_input,
                lambda wire: visualize(
                    wire.ui_side(merge=False),
                    initial_status=self._initial_status_update(),
                    cancel_event=cancel_event,
                    live_view=live_view,
                ),
                cancel_event,
                self.soul.wire_file if isinstance(self.soul, KimiSoul) else None,
            )
            return True
        except LLMNotSet:
            logger.exception("LLM not set:")
            console.print('[red]LLM not set, send "/login" to login[/red]')
        except LLMNotSupported as e:
            logger.exception("LLM not supported:")
            console.print(f"[red]{e}[/red]")
        except ChatProviderError as e:
            logger.exception("LLM provider error:")
            if isinstance(e, APIStatusError) and e.status_code == 401:
                console.print(
                    f"[red]Authorization failed, please check your login status: {e}[/red]"
                )
            elif isinstance(e, APIStatusError) and e.status_code == 402:
                console.print("[red]Membership expired, please renew your plan[/red]")
            elif isinstance(e, APIStatusError) and e.status_code == 403:
                console.print(
                    f"[red]Quota exceeded, please upgrade your plan or retry later: {e}[/red]"
                )
            else:
                console.print(f"[red]LLM provider error: {e}[/red]")
        except MaxStepsReached as e:
            logger.warning("Max steps reached: {n_steps}", n_steps=e.n_steps)
            console.print(f"[yellow]{e}[/yellow]")
        except RunCancelled:
            logger.info("Cancelled by user")
            console.print("[red]Interrupted by user[/red]")
        except Exception as e:
            logger.exception("Unexpected error:")
            console.print(f"[red]Unexpected error: {e}[/red]")
            raise
        return False

    async def _run_shell_command(self, command: str) -> None:
        """Run a shell command in foreground."""
        if not command.strip():
            return

        # Check if it's an allowed slash command in shell mode
        if slash_cmd_call := parse_slash_command_call(command):
            if shell_mode_registry.find_command(slash_cmd_call.name):
                await self._run_slash_command(slash_cmd_call)
                return
            else:
                console.print(
                    f'[yellow]"/{slash_cmd_call.name}" is not available in shell mode. '
                    "Press Ctrl-X to switch to agent mode.[/yellow]"
                )
                return

        # Check if user is trying to use 'cd' command
        stripped_cmd = command.strip()
        split_cmd: list[str] | None = None
        try:
            split_cmd = shlex.split(stripped_cmd)
        except ValueError as exc:
            logger.debug("Failed to parse shell command for cd check: {error}", error=exc)
        if split_cmd and len(split_cmd) == 2 and split_cmd[0] == "cd":
            console.print(
                "[yellow]Warning: Directory changes are not preserved across command executions."
                "[/yellow]"
            )
            return

        logger.info("Running shell command: {cmd}", cmd=command)

        proc: asyncio.subprocess.Process | None = None

        def _handler():
            logger.debug("SIGINT received.")
            if proc:
                proc.terminate()

        loop = asyncio.get_running_loop()
        remove_sigint = install_sigint_handler(loop, _handler)
        try:
            # TODO: For the sake of simplicity, we now use `create_subprocess_shell`.
            # Later we should consider making this behave like a real shell.
            with open_original_stderr() as stderr:
                kwargs: dict[str, Any] = {}
                if stderr is not None:
                    kwargs["stderr"] = stderr
                proc = await asyncio.create_subprocess_shell(command, env=get_clean_env(), **kwargs)
                await proc.wait()
        except Exception as e:
            logger.exception("Failed to run shell command:")
            console.print(f"[red]Failed to run shell command: {e}[/red]")
        finally:
            remove_sigint()

    def _start_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _cleanup(done: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Background task failed:")

        task.add_done_callback(_cleanup)
        return task

    def _cancel_background_tasks(self) -> None:
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()

    async def _run_slash_command(self, command_call: SlashCommandCall) -> None:
        from kimi_cli.cli import Reload

        if command_call.name not in self._slash_command_lookup:
            logger.info("Unknown slash command /{command}", command=command_call.name)
            console.print(
                f'[red]Unknown slash command "/{command_call.name}", '
                'type "/" for all available commands[/red]'
            )
            return

        command = shell_slash_registry.find_command(command_call.name)
        if command is None:
            # the input is a soul-level slash command call
            await self.run_soul_command(command_call.raw_input)
            return

        logger.debug(
            "Running shell-level slash command: /{command} with args: {args}",
            command=command_call.name,
            args=command_call.args,
        )

        try:
            ret = command.func(self, command_call.args)
            if isinstance(ret, Awaitable):
                await ret
        except Reload:
            # just propagate
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Handle Ctrl-C during slash command execution, return to shell prompt
            logger.debug("Slash command interrupted by KeyboardInterrupt")
            console.print("[red]Interrupted by user[/red]")
        except Exception as e:
            logger.exception("Unknown error:")
            console.print(f"[red]Unknown error: {e}[/red]")
            raise  # re-raise unknown error

    async def run_soul_command(self, user_input: str | list[ContentPart]) -> bool:
        """
        Run the soul and handle any known exceptions.

        Returns:
            bool: Whether the run is successful.
        """
        cancel_event = asyncio.Event()
        live_view = self._build_live_view(cancel_event)

        def _handler():
            logger.debug("SIGINT received.")
            cancel_event.set()

        loop = asyncio.get_running_loop()
        remove_sigint = install_sigint_handler(loop, _handler)
        try:
            return await self._run_soul_command_with_ui(
                user_input,
                cancel_event=cancel_event,
                live_view=live_view,
            )
        finally:
            remove_sigint()


_KIMI_BLUE = "dodger_blue1"
_LOGO = f"""\
[{_KIMI_BLUE}]\
▐█▛█▛█▌
▐█████▌\
[{_KIMI_BLUE}]\
"""


@dataclass(slots=True)
class WelcomeInfoItem:
    class Level(Enum):
        INFO = "grey50"
        WARN = "yellow"
        ERROR = "red"

    name: str
    value: str
    level: Level = Level.INFO


def _print_welcome_info(name: str, info_items: list[WelcomeInfoItem]) -> None:
    head = Text.from_markup("Welcome to Kimi Code CLI!")
    help_text = Text.from_markup("[grey50]Send /help for help information.[/grey50]")

    # Use Table for precise width control
    logo = Text.from_markup(_LOGO)
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1), expand=False)
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_row(logo, Group(head, help_text))

    rows: list[RenderableType] = [table]

    if info_items:
        rows.append(Text(""))  # empty line
    for item in info_items:
        rows.append(Text(f"{item.name}: {item.value}", style=item.level.value))

    console.print(
        Panel(
            Group(*rows),
            border_style=_KIMI_BLUE,
            expand=False,
            padding=(1, 2),
        )
    )
