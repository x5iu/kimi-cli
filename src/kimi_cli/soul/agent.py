from __future__ import annotations

import asyncio
import typing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic
from jinja2 import BaseLoader, Environment as JinjaEnvironment
from jinja2 import FileSystemLoader, StrictUndefined, TemplateError, TemplateNotFound, UndefinedError
from kaos.path import KaosPath
from kosong.tooling import Toolset

from kimi_cli.agentspec import load_agent_spec
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.background import BackgroundTaskManager
from kimi_cli.config import Config
from kimi_cli.exception import MCPConfigError, SystemPromptTemplateError
from kimi_cli.llm import LLM
from kimi_cli.notifications import NotificationManager, NotificationSink
from kimi_cli.session import Session
from kimi_cli.share import get_share_dir
from kimi_cli.skill import Skill, discover_skills_from_roots, index_skills, resolve_skills_roots
from kimi_cli.soul.approval import Approval, ApprovalState
from kimi_cli.soul.denwarenji import DenwaRenji
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.utils.environment import Environment
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import is_within_directory, list_directory

if TYPE_CHECKING:
    from fastmcp.mcp_config import MCPConfig


AGENTS_MD_FILENAMES = ("AGENTS.md", "agents.md")


@dataclass(frozen=True, slots=True, kw_only=True)
class BuiltinSystemPromptArgs:
    """Builtin system prompt arguments."""

    KIMI_NOW: str
    """The current datetime."""
    KIMI_WORK_DIR: KaosPath
    """The absolute path of current working directory."""
    KIMI_WORK_DIR_LS: str
    """The directory listing of current working directory."""
    KIMI_AGENTS_MD: str  # TODO: move to first message from system prompt
    """Combined content of the discovered global/project AGENTS.md files."""
    KIMI_SKILLS: str
    """Formatted information about available skills."""
    KIMI_ADDITIONAL_DIRS_INFO: str
    """Formatted information about additional directories in the workspace."""


async def _load_agents_md_from_dir(directory: KaosPath, *, source: str) -> str | None:
    for filename in AGENTS_MD_FILENAMES:
        path = directory / filename
        if await path.is_file():
            logger.info("Loaded {source} AGENTS.md: {path}", source=source, path=path)
            return (await path.read_text()).strip()
    return None


async def load_global_agents_md() -> str | None:
    share_dir = KaosPath.unsafe_from_local_path(get_share_dir()).canonical()
    return await _load_agents_md_from_dir(share_dir, source="global")


async def load_project_agents_md(work_dir: KaosPath) -> str | None:
    return await _load_agents_md_from_dir(work_dir.canonical(), source="project")


def combine_agents_md(global_agents_md: str | None, project_agents_md: str | None) -> str | None:
    sections: list[str] = []
    if global_agents_md:
        sections.append(f"[Global AGENTS.md]\n{global_agents_md}")
    if project_agents_md:
        sections.append(f"[Project AGENTS.md]\n{project_agents_md}")
    if not sections:
        return None
    if len(sections) == 1:
        return global_agents_md or project_agents_md
    return "\n\n".join(sections)


async def load_agents_md(work_dir: KaosPath) -> str | None:
    work_dir = work_dir.canonical()
    share_dir = KaosPath.unsafe_from_local_path(get_share_dir()).canonical()
    if share_dir == work_dir:
        return await load_project_agents_md(work_dir)

    global_agents_md, project_agents_md = await asyncio.gather(
        load_global_agents_md(),
        load_project_agents_md(work_dir),
    )
    agents_md = combine_agents_md(global_agents_md, project_agents_md)
    if agents_md is None:
        logger.info(
            "No AGENTS.md found in share dir {share_dir} or work dir {work_dir}",
            share_dir=share_dir,
            work_dir=work_dir,
        )
    return agents_md


@dataclass(slots=True, kw_only=True)
class Runtime:
    """Agent runtime."""

    config: Config
    oauth: OAuthManager
    llm: LLM | None  # we do not freeze the `Runtime` dataclass because LLM can be changed
    session: Session
    builtin_args: BuiltinSystemPromptArgs
    denwa_renji: DenwaRenji
    approval: Approval
    environment: Environment
    notifications: NotificationManager
    background_tasks: BackgroundTaskManager
    skills: dict[str, Skill]
    additional_dirs: list[KaosPath]
    skills_dirs: list[KaosPath]
    agents_md: str
    background_notification_targets: tuple[NotificationSink, ...] = ("llm",)

    def __post_init__(self) -> None:
        self.background_tasks.bind_notification_targets(
            lambda: self.background_notification_targets
        )

    @property
    def has_live_background_notifications(self) -> bool:
        return any(target != "llm" for target in self.background_notification_targets)

    @staticmethod
    async def create(
        config: Config,
        oauth: OAuthManager,
        llm: LLM | None,
        session: Session,
        yolo: bool,
        skills_dir: KaosPath | None = None,
    ) -> Runtime:
        ls_output, agents_md, environment = await asyncio.gather(
            list_directory(session.work_dir),
            load_agents_md(session.work_dir),
            Environment.detect(),
        )

        # Discover and format skills
        skills_roots = await resolve_skills_roots(session.work_dir, skills_dir_override=skills_dir)
        # Canonicalize so symlinked skill directories match resolved paths
        skills_roots_canonical = [r.canonical() for r in skills_roots]
        skills = await discover_skills_from_roots(skills_roots)
        skills_by_name = index_skills(skills)
        logger.info("Discovered {count} skill(s)", count=len(skills))
        skills_formatted = "\n".join(
            (
                f"- {skill.name}\n"
                f"  - Path: {skill.skill_md_file}\n"
                f"  - Description: {skill.description}"
            )
            for skill in skills
        )

        # Restore additional directories from session state, pruning stale entries
        additional_dirs: list[KaosPath] = []
        pruned = False
        valid_dir_strs: list[str] = []
        for dir_str in session.state.additional_dirs:
            d = KaosPath(dir_str).canonical()
            if await d.is_dir():
                additional_dirs.append(d)
                valid_dir_strs.append(dir_str)
            else:
                logger.warning(
                    "Additional directory no longer exists, removing from state: {dir}",
                    dir=dir_str,
                )
                pruned = True
        if pruned:
            session.state.additional_dirs = valid_dir_strs
            session.save_state()

        # Format additional dirs info for system prompt
        additional_dirs_info = ""
        if additional_dirs:
            parts: list[str] = []
            for d in additional_dirs:
                try:
                    dir_ls = await list_directory(d)
                except OSError:
                    logger.warning(
                        "Cannot list additional directory, skipping listing: {dir}", dir=d
                    )
                    dir_ls = "[directory not readable]"
                parts.append(f"### `{d}`\n\n```\n{dir_ls}\n```")
            additional_dirs_info = "\n\n".join(parts)

        # Merge CLI flag with persisted session state
        effective_yolo = yolo or session.state.approval.yolo
        saved_actions = set(session.state.approval.auto_approve_actions)

        def _on_approval_change() -> None:
            session.state.approval.yolo = approval_state.yolo
            session.state.approval.auto_approve_actions = set(approval_state.auto_approve_actions)
            session.save_state()

        approval_state = ApprovalState(
            yolo=effective_yolo,
            auto_approve_actions=saved_actions,
            on_change=_on_approval_change,
        )

        notifications = NotificationManager(
            session.context_file.parent / "notifications",
            config.notifications,
        )

        # For openai_responses provider, pass KIMI_AGENTS_MD via the `instructions`
        # parameter instead of embedding it in the system prompt.
        agents_md_in_prompt = agents_md or ""
        if (
            llm
            and llm.provider_config
            and llm.provider_config.type == "openai_responses"
            and agents_md
        ):
            from kosong.contrib.chat_provider.openai_responses import OpenAIResponses

            if isinstance(llm.chat_provider, OpenAIResponses):
                llm.chat_provider = llm.chat_provider.with_generation_kwargs(
                    instructions=agents_md,
                )
                agents_md_in_prompt = ""

        return Runtime(
            config=config,
            oauth=oauth,
            llm=llm,
            session=session,
            builtin_args=BuiltinSystemPromptArgs(
                KIMI_NOW=datetime.now().astimezone().isoformat(),
                KIMI_WORK_DIR=session.work_dir,
                KIMI_WORK_DIR_LS=ls_output,
                KIMI_AGENTS_MD=agents_md_in_prompt,
                KIMI_SKILLS=skills_formatted or "No skills found.",
                KIMI_ADDITIONAL_DIRS_INFO=additional_dirs_info,
            ),
            denwa_renji=DenwaRenji(),
            approval=Approval(state=approval_state),
            environment=environment,
            notifications=notifications,
            background_tasks=BackgroundTaskManager(
                session,
                config.background,
                notifications=notifications,
            ),
            skills=skills_by_name,
            additional_dirs=additional_dirs,
            # Only expose skills roots outside the workspace for Glob access;
            # project-level roots are already within work_dir.
            skills_dirs=[
                r for r in skills_roots_canonical
                if not is_within_directory(r, session.work_dir)
            ],
            agents_md=agents_md or "",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Agent:
    """The loaded agent."""

    name: str
    system_prompt: str
    toolset: Toolset
    runtime: Runtime
    """Each agent has its own runtime, which should be derived from its main agent."""


async def load_agent(
    agent_file: Path,
    runtime: Runtime,
    *,
    mcp_configs: list[MCPConfig] | list[dict[str, Any]],
) -> Agent:
    """
    Load agent from specification file.

    Raises:
        FileNotFoundError: When the agent file is not found.
        AgentSpecError(KimiCLIException, ValueError): When the agent specification is invalid.
        SystemPromptTemplateError(KimiCLIException, ValueError): When the system prompt template
            is invalid.
        InvalidToolError(KimiCLIException, ValueError): When any tool cannot be loaded.
        MCPConfigError(KimiCLIException, ValueError): When any MCP configuration is invalid.
        MCPRuntimeError(KimiCLIException, RuntimeError): When any MCP server cannot be connected.
    """
    logger.info("Loading agent: {agent_file}", agent_file=agent_file)
    agent_spec = load_agent_spec(agent_file)

    system_prompt = _load_system_prompt(
        agent_spec.system_prompt_path,
        agent_spec.system_prompt_args,
        runtime.builtin_args,
    )

    toolset = KimiToolset()
    tool_deps = {
        KimiToolset: toolset,
        Runtime: runtime,
        # TODO: remove all the following dependencies and use Runtime instead
        Config: runtime.config,
        BuiltinSystemPromptArgs: runtime.builtin_args,
        Session: runtime.session,
        DenwaRenji: runtime.denwa_renji,
        Approval: runtime.approval,
        Environment: runtime.environment,
    }
    tools = agent_spec.tools
    if agent_spec.exclude_tools:
        logger.debug("Excluding tools: {tools}", tools=agent_spec.exclude_tools)
        tools = [tool for tool in tools if tool not in agent_spec.exclude_tools]
    toolset.load_tools(tools, tool_deps)

    if mcp_configs:
        validated_mcp_configs: list[MCPConfig] = []
        if mcp_configs:
            from fastmcp.mcp_config import MCPConfig

            for mcp_config in mcp_configs:
                try:
                    validated_mcp_configs.append(
                        mcp_config
                        if isinstance(mcp_config, MCPConfig)
                        else MCPConfig.model_validate(mcp_config)
                    )
                except pydantic.ValidationError as e:
                    raise MCPConfigError(f"Invalid MCP config: {e}") from e
        await toolset.load_mcp_tools(validated_mcp_configs, runtime)

    return Agent(
        name=agent_spec.name,
        system_prompt=system_prompt,
        toolset=toolset,
        runtime=runtime,
    )


class _SandboxedLoader(BaseLoader):
    """A Jinja2 loader that wraps FileSystemLoader and blocks path traversal.

    Ensures that all resolved template paths stay within the allowed root directory,
    preventing ``{% include "../../etc/passwd" %}`` style attacks.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._inner = FileSystemLoader(self._root)

    def get_source(
        self, environment: JinjaEnvironment, template: str
    ) -> tuple[str, str | None, typing.Callable[[], bool] | None]:
        resolved = (self._root / template).resolve()
        if not resolved.is_relative_to(self._root):
            raise TemplateNotFound(template)
        return self._inner.get_source(environment, template)

    def list_templates(self) -> list[str]:
        return self._inner.list_templates()


def _load_system_prompt(
    path: Path, args: dict[str, str], builtin_args: BuiltinSystemPromptArgs
) -> str:
    logger.info("Loading system prompt: {path}", path=path)
    system_prompt = path.read_text(encoding="utf-8").strip()
    logger.debug(
        "Substituting system prompt with builtin args: {builtin_args}, spec args: {spec_args}",
        builtin_args=builtin_args,
        spec_args=args,
    )
    env = JinjaEnvironment(
        loader=_SandboxedLoader(path.parent),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
        variable_start_string="${",
        variable_end_string="}",
        undefined=StrictUndefined,
    )
    try:
        template = env.from_string(system_prompt)
        return template.render(asdict(builtin_args), **args)
    except UndefinedError as exc:
        raise SystemPromptTemplateError(f"Missing system prompt arg in {path}: {exc}") from exc
    except TemplateError as exc:
        raise SystemPromptTemplateError(f"Invalid system prompt template: {path}: {exc}") from exc
