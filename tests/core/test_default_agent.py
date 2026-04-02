from __future__ import annotations

import platform

import pytest

from kimi_cli.agentspec import DEFAULT_AGENT_FILE
from kimi_cli.soul.agent import Runtime, load_agent
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.file.replace import EditTool


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_default_agent(runtime: Runtime):
    agent = await load_agent(DEFAULT_AGENT_FILE, runtime, mcp_configs=[])

    tool_names = [tool.name for tool in agent.toolset.tools]
    assert tool_names == [
        "AskUserQuestion",
        "SetTodoList",
        "Shell",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "TaskWrite",
        "ReadFile",
        "ReadMediaFile",
        "Glob",
        "Grep",
        "RecallCompactedContext",
        "WriteFile",
        "Edit",
        "SearchWeb",
        "FetchURL",
    ]
    assert isinstance(agent.toolset, KimiToolset)
    assert isinstance(agent.toolset.find("Edit"), EditTool)

    prompt = agent.system_prompt.replace(
        f"{runtime.builtin_args.KIMI_WORK_DIR}", "/path/to/work/dir"
    )
    assert "Background Bash" in prompt
    assert "run_in_background=true" in prompt
    assert "TaskList" in prompt
    assert "TaskOutput" in prompt
    assert "TaskStop" in prompt
    assert "/task" in prompt
