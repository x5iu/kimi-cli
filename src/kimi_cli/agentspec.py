from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from pydantic import BaseModel, Field

from kimi_cli.exception import AgentSpecError

DEFAULT_AGENT_SPEC_VERSION = "1"
SUPPORTED_AGENT_SPEC_VERSIONS = (DEFAULT_AGENT_SPEC_VERSION,)


def get_agents_dir() -> Path:
    return Path(__file__).parent / "agents"


DEFAULT_AGENT_FILE = get_agents_dir() / "default" / "agent.yaml"


class Inherit(NamedTuple):
    """Marker class for inheritance in agent spec."""


inherit = Inherit()


class AgentSpec(BaseModel):
    """Agent specification."""

    extend: str | None = Field(default=None, description="Agent file to extend")
    name: str | Inherit = Field(default=inherit, description="Agent name")  # required
    system_prompt_path: Path | Inherit = Field(
        default=inherit, description="System prompt path"
    )  # required
    system_prompt_args: dict[str, str] = Field(
        default_factory=dict, description="System prompt arguments"
    )
    tools: list[str] | None | Inherit = Field(default=inherit, description="Tools")  # required
    exclude_tools: list[str] | None | Inherit = Field(
        default=inherit, description="Tools to exclude"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedAgentSpec:
    """Resolved agent specification."""

    name: str
    system_prompt_path: Path
    system_prompt_args: dict[str, str]
    tools: list[str]
    exclude_tools: list[str]


def load_agent_spec(agent_file: Path) -> ResolvedAgentSpec:
    """
    Load agent specification from file.

    Raises:
        FileNotFoundError: If the agent spec file is not found.
        AgentSpecError: If the agent spec is not valid.
    """
    agent_spec = _load_agent_spec(agent_file)
    assert agent_spec.extend is None, "agent extension should be recursively resolved"
    if isinstance(agent_spec.name, Inherit):
        raise AgentSpecError("Agent name is required")
    if isinstance(agent_spec.system_prompt_path, Inherit):
        raise AgentSpecError("System prompt path is required")
    if isinstance(agent_spec.tools, Inherit):
        raise AgentSpecError("Tools are required")
    if isinstance(agent_spec.exclude_tools, Inherit):
        agent_spec.exclude_tools = []
    return ResolvedAgentSpec(
        name=agent_spec.name,
        system_prompt_path=agent_spec.system_prompt_path,
        system_prompt_args=agent_spec.system_prompt_args,
        tools=agent_spec.tools or [],
        exclude_tools=agent_spec.exclude_tools or [],
    )


def _load_agent_spec(agent_file: Path) -> AgentSpec:
    if not agent_file.exists():
        raise AgentSpecError(f"Agent spec file not found: {agent_file}")
    if not agent_file.is_file():
        raise AgentSpecError(f"Agent spec path is not a file: {agent_file}")
    try:
        with open(agent_file, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise AgentSpecError(f"Invalid YAML in agent spec file: {e}") from e

    version = str(data.get("version", DEFAULT_AGENT_SPEC_VERSION))
    if version not in SUPPORTED_AGENT_SPEC_VERSIONS:
        raise AgentSpecError(f"Unsupported agent spec version: {version}")

    agent_spec = AgentSpec(**data.get("agent", {}))
    if isinstance(agent_spec.system_prompt_path, Path):
        agent_spec.system_prompt_path = (
            agent_file.parent / agent_spec.system_prompt_path
        ).absolute()
    if agent_spec.extend:
        if agent_spec.extend == "default":
            base_agent_file = DEFAULT_AGENT_FILE
        else:
            base_agent_file = (agent_file.parent / agent_spec.extend).absolute()
        base_agent_spec = _load_agent_spec(base_agent_file)

        # Build merged spec without mutating base
        merged_args = dict(base_agent_spec.system_prompt_args)
        merged_args.update(agent_spec.system_prompt_args)

        merged_name = (
            agent_spec.name
            if not isinstance(agent_spec.name, Inherit)
            else base_agent_spec.name
        )
        merged_prompt_path = (
            agent_spec.system_prompt_path
            if not isinstance(agent_spec.system_prompt_path, Inherit)
            else base_agent_spec.system_prompt_path
        )
        merged_tools = (
            agent_spec.tools
            if not isinstance(agent_spec.tools, Inherit)
            else base_agent_spec.tools
        )
        merged_exclude = (
            agent_spec.exclude_tools
            if not isinstance(agent_spec.exclude_tools, Inherit)
            else base_agent_spec.exclude_tools
        )

        agent_spec = AgentSpec(
            name=merged_name,
            system_prompt_path=merged_prompt_path,
            system_prompt_args=merged_args,
            tools=merged_tools,
            exclude_tools=merged_exclude,
        )
    return agent_spec
