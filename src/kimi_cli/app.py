from __future__ import annotations

import asyncio
import contextlib
import os
import warnings
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

import kaos
from kaos.path import KaosPath
from kimi_cli.agentspec import DEFAULT_AGENT_FILE
from kimi_cli.cli import InputFormat, OutputFormat
from kimi_cli.config import Config, LLMModel, LLMProvider, load_config
from kimi_cli.eventbus import EventBus, EventBusConsumer
from kimi_cli.eventbus.types import BusMessage, ContentPart
from kimi_cli.exception import ConfigError
from kimi_cli.llm import augment_provider_with_env_vars, create_llm, model_display_name
from kimi_cli.loop import run_agent_loop
from kimi_cli.loop.agent import Runtime, load_agent
from kimi_cli.loop.context import Context
from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop
from kimi_cli.notifications import NotificationSink
from kimi_cli.session import Session
from kimi_cli.share import get_share_dir
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.utils.logging import logger, redirect_stderr_to_logger
from kimi_cli.utils.path import shorten_home

if TYPE_CHECKING:
    from fastmcp.mcp_config import MCPConfig


def enable_logging(debug: bool = False, *, redirect_stderr: bool = True) -> None:
    # NOTE: stderr redirection is implemented by swapping the process-level fd=2 (dup2).
    # That can hide Click/Typer error output during CLI startup, so some entrypoints delay
    # installing it until after critical initialization succeeds.
    logger.remove()  # Remove default stderr handler
    logger.enable("kimi_cli")
    if debug:
        logger.enable("llmkit")
    logger.add(
        get_share_dir() / "logs" / "kimi.log",
        # FIXME: configure level for different modules
        level="TRACE" if debug else "INFO",
        rotation="06:00",
        retention="10 days",
    )
    if redirect_stderr:
        redirect_stderr_to_logger()


class KimiCLI:
    @staticmethod
    async def create(
        session: Session,
        *,
        # Basic configuration
        config: Config | Path | None = None,
        model_name: str | None = None,
        thinking: bool | None = None,
        # Run mode
        yolo: bool = False,
        # Extensions
        agent_file: Path | None = None,
        mcp_configs: list[MCPConfig] | list[dict[str, Any]] | None = None,
        skills_dir: KaosPath | None = None,
        # Loop control
        max_steps_per_turn: int | None = None,
    ) -> KimiCLI:
        """
        Create a KimiCLI instance.

        Args:
            session (Session): A session created by `Session.create` or `Session.continue_`.
            config (Config | Path | None, optional): Configuration to use, or path to config file.
                Defaults to None.
            model_name (str | None, optional): Name of the model to use. Defaults to None.
            thinking (bool | None, optional): Whether to enable thinking mode. Defaults to None.
            yolo (bool, optional): Approve all actions without confirmation. Defaults to False.
            agent_file (Path | None, optional): Path to the agent file. Defaults to None.
            mcp_configs (list[MCPConfig | dict[str, Any]] | None, optional): MCP configs to load
                MCP tools from. Defaults to None.
            skills_dir (KaosPath | None, optional): Override skills directory discovery. Defaults
                to None.
            max_steps_per_turn (int | None, optional): Maximum number of steps in one turn.
                Defaults to None.
        Raises:
            FileNotFoundError: When the agent file is not found.
            ConfigError(KimiCLIException, ValueError): When the configuration is invalid.
            AgentSpecError(KimiCLIException, ValueError): When the agent specification is invalid.
            SystemPromptTemplateError(KimiCLIException, ValueError): When the system prompt
                template is invalid.
            InvalidToolError(KimiCLIException, ValueError): When any tool cannot be loaded.
            MCPConfigError(KimiCLIException, ValueError): When any MCP configuration is invalid.
            MCPRuntimeError(KimiCLIException, RuntimeError): When any MCP server cannot be
                connected.
        """
        config = config if isinstance(config, Config) else load_config(config)
        if max_steps_per_turn is not None:
            config.loop_control.max_steps_per_turn = max_steps_per_turn
        logger.info("Loaded config: {config}", config=config)
        # Inject user-defined environment variables so that Shell commands,
        # background tasks and subagents all inherit them.
        for key, value in config.env.items():
            existing = os.environ.get(key)
            if existing:
                logger.warning("env var {} already set, skipping config override", key)
            else:
                os.environ[key] = value

        model: LLMModel | None = None
        provider: LLMProvider | None = None

        # try to use config file
        if not model_name and config.default_model:
            # no --model specified && default model is set in config
            model = config.models[config.default_model]
            provider = config.providers[model.provider]
        if model_name and model_name in config.models:
            # --model specified && model is set in config
            model = config.models[model_name]
            provider = config.providers[model.provider]

        if not model:
            model = LLMModel(provider="", model="", max_context_size=100_000)
            provider = LLMProvider(type="kimi", base_url="", api_key=SecretStr(""))

        # try overwrite with environment variables
        if provider is None:
            raise ConfigError("No LLM provider configured; check config or environment variables")
        if model is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ConfigError("No LLM model configured; check config or environment variables")
        env_overrides = augment_provider_with_env_vars(provider, model)

        # determine thinking mode
        thinking = config.default_thinking if thinking is None else thinking

        # determine yolo mode
        yolo = yolo if yolo else config.default_yolo

        llm = create_llm(
            provider,
            model,
            thinking=thinking,
            session_id=session.id,
        )
        if llm is not None:
            logger.info("Using LLM provider: {provider}", provider=provider)
            logger.info("Using LLM model: {model}", model=model)
            logger.info("Thinking mode: {thinking}", thinking=thinking)

        runtime = await Runtime.create(config, llm, session, yolo, skills_dir)
        runtime.notifications.recover()
        runtime.background_tasks.reconcile()

        if agent_file is None:
            agent_file = DEFAULT_AGENT_FILE
        agent = await load_agent(agent_file, runtime, mcp_configs=mcp_configs or [])

        context = Context(session.context_file)
        await context.restore()

        agent_loop = KimiAgentLoop(agent, context=context)
        return KimiCLI(agent_loop, runtime, env_overrides)

    def __init__(
        self,
        _agent_loop: KimiAgentLoop,
        _runtime: Runtime,
        _env_overrides: dict[str, str],
    ) -> None:
        self._agent_loop = _agent_loop
        self._runtime = _runtime
        self._env_overrides = _env_overrides

    @property
    def agent_loop(self) -> KimiAgentLoop:
        """Get the KimiAgentLoop instance."""
        return self._agent_loop

    @property
    def session(self) -> Session:
        """Get the Session instance."""
        return self._runtime.session

    def shutdown_background_tasks(self) -> None:
        """Kill active background tasks on exit, unless keep_alive_on_exit is configured."""
        if self._runtime.config.background.keep_alive_on_exit:
            return
        killed = self._runtime.background_tasks.kill_all_active(reason="CLI session ended")
        if killed:
            logger.info("Stopped {n} background task(s) on exit: {ids}", n=len(killed), ids=killed)

    @contextlib.contextmanager
    def _background_notification_targets(self, *targets: NotificationSink) -> Iterator[None]:
        previous = self._runtime.background_notification_targets
        normalized = tuple(dict.fromkeys(targets)) or ("llm",)
        self._runtime.background_notification_targets = normalized
        try:
            yield
        finally:
            self._runtime.background_notification_targets = previous

    @contextlib.asynccontextmanager
    async def _env(self) -> AsyncGenerator[None]:
        original_cwd = KaosPath.cwd()
        await kaos.chdir(self._runtime.session.work_dir)
        try:
            # to ignore possible warnings from dateparser
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            yield
        finally:
            await kaos.chdir(original_cwd)

    async def run(
        self,
        user_input: str | list[ContentPart],
        cancel_event: asyncio.Event,
        merge_bus_messages: bool = False,
    ) -> AsyncGenerator[BusMessage]:
        """
        Run the Kimi Code CLI instance without any UI and yield EventBus messages directly.

        Args:
            user_input (str | list[ContentPart]): The user input to the agent.
            cancel_event (asyncio.Event): An event to cancel the run.
            merge_bus_messages (bool): Whether to merge EventBus messages as much as possible.

        Yields:
            BusMessage: The EventBus messages from the `KimiAgentLoop`.

        Raises:
            LLMNotSet: When the LLM is not set.
            LLMNotSupported: When the LLM does not have required capabilities.
            ChatProviderError: When the LLM provider returns an error.
            MaxStepsReached: When the maximum number of steps is reached.
            RunCancelled: When the run is cancelled by the cancel event.
        """
        async with self._env():
            bus_future = asyncio.Future[EventBusConsumer]()
            stop_ui_loop = asyncio.Event()

            async def _ui_loop_fn(event_bus: EventBus) -> None:
                bus_future.set_result(event_bus.ui_side(merge=merge_bus_messages))
                await stop_ui_loop.wait()

            loop_task = asyncio.create_task(
                run_agent_loop(self.agent_loop, user_input, _ui_loop_fn, cancel_event)
            )

            try:
                bus_consumer = await bus_future
                while True:
                    msg = await bus_consumer.receive()
                    yield msg
            except QueueShutDown:
                pass
            finally:
                # stop consuming EventBus messages
                stop_ui_loop.set()
                # wait for the agent loop task to finish, or raise
                await loop_task

    async def run_shell(self, command: str | None = None) -> bool:
        """Run the Kimi Code CLI instance with shell UI."""
        from kimi_cli.constant import VERSION
        from kimi_cli.ui.shell import Shell, WelcomeInfoItem

        welcome_info = [
            WelcomeInfoItem(name="Version", value=f"v{VERSION}"),
            WelcomeInfoItem(
                name="Directory", value=str(shorten_home(self._runtime.session.work_dir))
            ),
            WelcomeInfoItem(name="Session", value=self._runtime.session.id),
        ]
        if base_url := self._env_overrides.get("KIMI_BASE_URL"):
            welcome_info.append(
                WelcomeInfoItem(
                    name="API URL",
                    value=f"{base_url} (from KIMI_BASE_URL)",
                    level=WelcomeInfoItem.Level.WARN,
                )
            )
        if self._env_overrides.get("KIMI_API_KEY"):
            welcome_info.append(
                WelcomeInfoItem(
                    name="API Key",
                    value="****** (from KIMI_API_KEY)",
                    level=WelcomeInfoItem.Level.WARN,
                )
            )
        if not self._runtime.llm:
            welcome_info.append(
                WelcomeInfoItem(
                    name="Model",
                    value="not set",
                    level=WelcomeInfoItem.Level.WARN,
                )
            )
        elif "KIMI_MODEL_NAME" in self._env_overrides:
            welcome_info.append(
                WelcomeInfoItem(
                    name="Model",
                    value=f"{self._agent_loop.model_name} (from KIMI_MODEL_NAME)",
                    level=WelcomeInfoItem.Level.WARN,
                )
            )
        else:
            welcome_info.append(
                WelcomeInfoItem(
                    name="Model",
                    value=model_display_name(self._agent_loop.model_name),
                    level=WelcomeInfoItem.Level.INFO,
                )
            )
        notification_targets: tuple[NotificationSink, ...] = (
            ("llm", "shell") if command is None else ("llm",)
        )
        async with self._env():
            with self._background_notification_targets(*notification_targets):
                shell = Shell(self._agent_loop, welcome_info=welcome_info)
                return await shell.run(command)

    async def run_print(
        self,
        input_format: InputFormat,
        output_format: OutputFormat,
        command: str | None = None,
        *,
        final_only: bool = False,
    ) -> int:
        """Run the Kimi Code CLI instance with print UI."""
        from kimi_cli.ui.print import Print

        async with self._env():
            print_ = Print(
                self._agent_loop,
                input_format,
                output_format,
                self._runtime.session.context_file,
                final_only=final_only,
            )
            return await print_.run(command)
