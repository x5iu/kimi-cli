"""Test configuration and fixtures."""

from __future__ import annotations

import os
import platform
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import SecretStr

from kaos import get_current_kaos, reset_current_kaos, set_current_kaos
from kaos.local import LocalKaos
from kaos.path import KaosPath
from kimi_cli.background import BackgroundTaskManager
from kimi_cli.config import Config, MoonshotSearchConfig, get_default_config
from kimi_cli.eventbus.log import EventLog
from kimi_cli.llm import ALL_MODEL_CAPABILITIES, LLM
from kimi_cli.loop.agent import BuiltinSystemPromptArgs, Runtime
from kimi_cli.loop.approval import Approval
from kimi_cli.loop.toolset import KimiToolset
from kimi_cli.metadata import WorkDirMeta
from kimi_cli.notifications import NotificationManager
from kimi_cli.session import Session
from kimi_cli.session_state import SessionState
from kimi_cli.tools.background import TaskList, TaskOutput, TaskStop, TaskWrite
from kimi_cli.tools.context import RecallCompactedContext
from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import EditTool
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.shell import Shell
from kimi_cli.tools.todo import SetTodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb
from kimi_cli.utils.environment import Environment
from llmkit.chat_provider.mock import MockChatProvider


@pytest.fixture
def config() -> Config:
    """Create a Config instance."""
    conf = get_default_config()
    conf.services.moonshot_search = MoonshotSearchConfig(
        base_url="https://api.kimi.com/coding/v1/search",
        api_key=SecretStr("test-api-key"),
    )
    return conf


@pytest.fixture
def llm() -> LLM:
    """Create a LLM instance."""
    return LLM(
        chat_provider=MockChatProvider([]),
        max_context_size=100_000,
        capabilities=ALL_MODEL_CAPABILITIES,
    )


@pytest.fixture
def temp_work_dir() -> Generator[KaosPath]:
    """Create a temporary working directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        p = Path(tmpdir).resolve()
        os.chdir(p)
        token = set_current_kaos(LocalKaos())
        try:
            yield KaosPath.unsafe_from_local_path(p)
        finally:
            reset_current_kaos(token)
            os.chdir(original_cwd)


@pytest.fixture
def temp_share_dir() -> Generator[Path]:
    """Create a temporary shared directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def builtin_args(temp_work_dir: KaosPath) -> BuiltinSystemPromptArgs:
    """Create builtin arguments with temporary work directory."""
    return BuiltinSystemPromptArgs(
        KIMI_NOW="1970-01-01T00:00:00+00:00",
        KIMI_WORK_DIR=temp_work_dir,
        KIMI_WORK_DIR_LS="Test ls content",
        KIMI_AGENTS_MD="Test agents content",
        KIMI_SKILLS="No skills found.",
        KIMI_ADDITIONAL_DIRS_INFO="",
        KIMI_OS="macOS",
        KIMI_SHELL="bash (`/bin/bash`)",
    )


@pytest.fixture
def session(temp_work_dir: KaosPath, temp_share_dir: Path) -> Session:
    """Create a Session instance."""
    return Session(
        id="test",
        work_dir=temp_work_dir,
        work_dir_meta=WorkDirMeta(path=str(temp_work_dir), kaos=get_current_kaos().name),
        context_file=temp_share_dir / "context.jsonl",
        event_log=EventLog(path=temp_share_dir / "events.jsonl"),
        state=SessionState(),
        title="Test Session",
        updated_at=0.0,
    )


@pytest.fixture
def approval() -> Approval:
    """Create a Approval instance."""
    return Approval(yolo=True)


@pytest.fixture
def environment() -> Environment:
    """Create an Environment instance."""
    if platform.system() == "Windows":
        return Environment(
            os_kind="Windows",
            os_arch="x86_64",
            os_version="1.0",
            shell_name="Windows PowerShell",
            shell_path=KaosPath("powershell.exe"),
        )
    else:
        return Environment(
            os_kind="Unix",
            os_arch="aarch64",
            os_version="1.0",
            shell_name="bash",
            shell_path=KaosPath("/bin/bash"),
        )


@pytest.fixture
def runtime(
    config: Config,
    llm: LLM,
    builtin_args: BuiltinSystemPromptArgs,
    session: Session,
    approval: Approval,
    environment: Environment,
) -> Runtime:
    """Create a Runtime instance."""
    notifications = NotificationManager(
        session.context_file.parent / "notifications",
        config.notifications,
    )
    rt = Runtime(
        config=config,
        llm=llm,
        builtin_args=builtin_args,
        session=session,
        approval=approval,
        environment=environment,
        notifications=notifications,
        background_tasks=BackgroundTaskManager(
            session,
            config.background,
            notifications=notifications,
        ),
        skills={},
        additional_dirs=[],
        skills_dirs=[],
        agents_md="",
    )
    return rt


@pytest.fixture
def toolset() -> KimiToolset:
    return KimiToolset()


@contextmanager
def tool_call_context(tool_name: str) -> Generator[None]:
    """Create a tool call context."""
    from kimi_cli.eventbus.types import ToolCall
    from kimi_cli.loop.toolset import current_tool_call

    token = current_tool_call.set(
        ToolCall(id="test", function=ToolCall.FunctionBody(name=tool_name, arguments=None))
    )
    try:
        yield
    finally:
        current_tool_call.reset(token)


@pytest.fixture
def set_todo_list_tool(runtime: Runtime) -> SetTodoList:
    """Create a SetTodoList tool instance."""
    return SetTodoList(runtime)


@pytest.fixture
def shell_tool(approval: Approval, environment: Environment, runtime: Runtime) -> Generator[Shell]:
    """Create a Shell tool instance."""
    with tool_call_context("Shell"):
        yield Shell(approval, environment, runtime)


@pytest.fixture
def task_list_tool(runtime: Runtime) -> TaskList:
    return TaskList(runtime)


@pytest.fixture
def task_output_tool(runtime: Runtime) -> TaskOutput:
    return TaskOutput(runtime)


@pytest.fixture
def task_stop_tool(runtime: Runtime, approval: Approval) -> TaskStop:
    return TaskStop(runtime, approval)


@pytest.fixture
def task_write_tool(runtime: Runtime) -> TaskWrite:
    return TaskWrite(runtime)


@pytest.fixture
def read_file_tool(runtime: Runtime) -> ReadFile:
    """Create a ReadFile tool instance."""
    return ReadFile(runtime)


@pytest.fixture
def recall_compacted_context_tool(runtime: Runtime) -> RecallCompactedContext:
    """Create a RecallCompactedContext tool instance."""
    return RecallCompactedContext(runtime)


@pytest.fixture
def read_media_file_tool(runtime: Runtime) -> ReadMediaFile:
    """Create a ReadMediaFile tool instance."""
    return ReadMediaFile(runtime)


@pytest.fixture
def glob_tool(runtime: Runtime) -> Glob:
    """Create a Glob tool instance."""
    return Glob(runtime)


@pytest.fixture
def grep_tool(runtime: Runtime) -> Grep:
    """Create a Grep tool instance."""
    return Grep(runtime)


@pytest.fixture
def write_file_tool(runtime: Runtime, approval: Approval) -> Generator[WriteFile]:
    """Create a WriteFile tool instance."""
    with tool_call_context("WriteFile"):
        yield WriteFile(runtime, approval)


@pytest.fixture
def edit_tool(runtime: Runtime, approval: Approval) -> Generator[EditTool]:
    """Create an Edit tool instance."""
    with tool_call_context("Edit"):
        yield EditTool(runtime, approval)


@pytest.fixture
def search_web_tool(config: Config, runtime: Runtime) -> SearchWeb:
    """Create a SearchWeb tool instance."""
    return SearchWeb(config, runtime)


@pytest.fixture
def fetch_url_tool(config: Config, runtime: Runtime) -> FetchURL:
    """Create a FetchURL tool instance."""
    return FetchURL(config, runtime)


# misc fixtures


@pytest.fixture
def outside_file() -> Generator[Path]:
    """Return a path to a file outside the working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "outside_file.txt"
