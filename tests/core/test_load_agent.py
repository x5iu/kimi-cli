"""Tests for agent loading functionality."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from kimi_cli.config import Config
from kimi_cli.exception import InvalidToolError, SystemPromptTemplateError
from kimi_cli.loop.agent import BuiltinSystemPromptArgs, Runtime, _load_system_prompt, load_agent
from kimi_cli.loop.approval import Approval
from kimi_cli.loop.toolset import KimiToolset
from kimi_cli.session import Session
from kimi_cli.utils.environment import Environment


def test_load_system_prompt(system_prompt_file: Path, builtin_args: BuiltinSystemPromptArgs):
    """Test loading system prompt with template substitution."""
    prompt = _load_system_prompt(system_prompt_file, {"CUSTOM_ARG": "test_value"}, builtin_args)

    assert "Test system prompt with " in prompt
    assert "1970-01-01" in prompt  # Should contain the actual timestamp
    assert builtin_args.KIMI_NOW in prompt
    assert "test_value" in prompt


def test_load_system_prompt_allows_literal_dollar(builtin_args: BuiltinSystemPromptArgs):
    """System prompt should allow literal $ without template errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        system_md = tmpdir / "system.md"
        system_md.write_text("Price is $100, path $PATH, time ${KIMI_NOW}.")
        prompt = _load_system_prompt(system_md, {}, builtin_args)

    assert "$100" in prompt
    assert "$PATH" in prompt
    assert builtin_args.KIMI_NOW in prompt


def test_load_system_prompt_include(builtin_args: BuiltinSystemPromptArgs):
    """System prompt should support {% include "file.md" %} directives."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        included = tmpdir / "extra.md"
        included.write_text("Included content here")
        system_md = tmpdir / "system.md"
        system_md.write_text('Main prompt. {% include "extra.md" %} End.')
        prompt = _load_system_prompt(system_md, {}, builtin_args)

    assert "Main prompt." in prompt
    assert "Included content here" in prompt
    assert "End." in prompt


def test_load_system_prompt_include_path_traversal_blocked(
    builtin_args: BuiltinSystemPromptArgs,
):
    """Path traversal via {% include %} must be blocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a file outside the template directory
        parent = tmpdir / "parent"
        parent.mkdir()
        outside = tmpdir / "outside.txt"
        outside.write_text("OUTSIDE SECRET")

        system_md = parent / "system.md"
        system_md.write_text('Main. {% include "../outside.txt" %} End.')
        with pytest.raises(SystemPromptTemplateError):
            _load_system_prompt(system_md, {}, builtin_args)


def test_load_system_prompt_include_nonexistent_file(
    builtin_args: BuiltinSystemPromptArgs,
):
    """Including a non-existent file should raise an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        system_md = tmpdir / "system.md"
        system_md.write_text('Main. {% include "no_such_file.md" %} End.')
        with pytest.raises(SystemPromptTemplateError):
            _load_system_prompt(system_md, {}, builtin_args)


def test_load_system_prompt_nested_include(builtin_args: BuiltinSystemPromptArgs):
    """Nested includes (A includes B includes C) should work normally."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "c.md").write_text("LEAF")
        (tmpdir / "b.md").write_text('MID {% include "c.md" %} MID')
        system_md = tmpdir / "system.md"
        system_md.write_text('ROOT {% include "b.md" %} ROOT')
        prompt = _load_system_prompt(system_md, {}, builtin_args)

    assert "ROOT" in prompt
    assert "MID" in prompt
    assert "LEAF" in prompt


def test_load_system_prompt_missing_arg_raises(builtin_args: BuiltinSystemPromptArgs):
    """Missing template args should raise a dedicated error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        system_md = tmpdir / "system.md"
        system_md.write_text("Missing ${UNKNOWN_ARG}.")
        with pytest.raises(SystemPromptTemplateError):
            _load_system_prompt(system_md, {}, builtin_args)


def test_load_tools_valid(runtime: Runtime):
    """Test loading valid tools."""
    tool_paths = ["kimi_cli.tools.think:Think", "kimi_cli.tools.shell:Shell"]
    toolset = KimiToolset()
    toolset.load_tools(
        tool_paths,
        {
            Runtime: runtime,
            Config: runtime.config,
            BuiltinSystemPromptArgs: runtime.builtin_args,
            Session: runtime.session,
            Approval: runtime.approval,
            Environment: runtime.environment,
        },
    )
    assert len(toolset.tools) == snapshot(2)


def test_load_tools_invalid(runtime: Runtime):
    """Test loading with invalid tool paths."""
    tool_paths = ["kimi_cli.tools.nonexistent:Tool", "kimi_cli.tools.think:Think"]
    toolset = KimiToolset()
    try:
        toolset.load_tools(
            tool_paths,
            {
                Runtime: runtime,
                Config: runtime.config,
                BuiltinSystemPromptArgs: runtime.builtin_args,
                Session: runtime.session,
                Approval: runtime.approval,
            },
        )
        raise AssertionError("should fail to load non-existing tool")
    except InvalidToolError as e:
        assert "kimi_cli.tools.nonexistent:Tool" in str(e)


async def test_load_agent_invalid_tools(agent_file_invalid_tools: Path, runtime: Runtime):
    """Test loading agent with invalid tools raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tools"):
        await load_agent(agent_file_invalid_tools, runtime, mcp_configs=[])


@pytest.fixture
def agent_file_invalid_tools() -> Generator[Path, Any, Any]:
    """Create an agent configuration file with invalid tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create system.md
        system_md = tmpdir / "system.md"
        system_md.write_text("You are a test agent")

        # Create agent.yaml with invalid tools
        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.nonexistent:Tool"]
""")

        yield agent_yaml


@pytest.fixture
def system_prompt_file() -> Generator[Path, Any, Any]:
    """Create a system prompt file with template variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        system_md = tmpdir / "system.md"
        system_md.write_text("Test system prompt with ${KIMI_NOW} and ${CUSTOM_ARG}")

        yield system_md
