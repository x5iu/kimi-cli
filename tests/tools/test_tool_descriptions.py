from __future__ import annotations

# ruff: noqa

import platform
import pytest
from inline_snapshot import snapshot

from kimi_cli.tools.multiagent.create import CreateSubagent
from kimi_cli.tools.shell import Shell
from kimi_cli.tools.dmail import SendDMail
from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.context import RecallCompactedContext
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import EditTool
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.multiagent.task import Task
from kimi_cli.tools.think import Think
from kimi_cli.tools.todo import ExecuteTodo, SetTodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb


def test_task_description(task_tool: Task):
    """Test the description of Task tool."""
    assert task_tool.base.description == snapshot(
        """\
Spawn a subagent to perform a specific task. Subagent will be spawned with a fresh context without any history of yours.

**Context Isolation**

Context isolation is one of the key benefits of using subagents. By delegating tasks to subagents, you can keep your main context clean and focused on the main goal requested by the user.

Here are some scenarios you may want this tool for context isolation:

- You wrote some code and it did not work as expected. In this case you can spawn a subagent to fix the code, asking the subagent to return how it is fixed. This can potentially benefit because the detailed process of fixing the code may not be relevant to your main goal, and may clutter your context.
- When you need some latest knowledge of a specific library, framework or technology to proceed with your task, you can spawn a subagent to search on the internet for the needed information and return to you the gathered relevant information, for example code examples, API references, etc. This can avoid ton of irrelevant search results in your own context.

DO NOT directly forward the user prompt to Task tool. Do not blindly spawn one subagent for every todo item. Only use subagents for very specific and narrow tasks with clear inputs and expected outputs, such as fixing a compilation error or searching for a specific solution. It is appropriate to use `Task` for todo items that are narrow, independent, and easy to summarize back to the user.

When working with `SetTodoList`, the root agent should keep the todo list aligned with delegated work: mark delegated items `in_progress` when you start them, and update them to `done` or `blocked` after the subagent returns. If the todo already exists in the stored session todo list, prefer `ExecuteTodo` so the state update is handled for you.

**Parallel Multi-Tasking**

Parallel multi-tasking is another key benefit of this tool. When the user request involves multiple subtasks that are independent of each other, you can use Task tool multiple times in a single response to let subagents work in parallel for you.

Examples:

- User requests to code, refactor or fix multiple modules/files in a project, and they can be tested independently. In this case you can spawn multiple subagents each working on a different module/file.
- When you need to analyze a huge codebase (> hundreds of thousands of lines), you can spawn multiple subagents each exploring on a different part of the codebase and gather the summarized results.
- When you need to search the web for multiple queries, you can spawn multiple subagents for better efficiency.

**Available Subagents:**

- `mocker`: The mock agent for testing purposes.
"""
    )


def test_create_subagent_description(create_subagent_tool: CreateSubagent):
    """Test the description of CreateSubagent tool."""
    assert create_subagent_tool.base.description == snapshot(
        """\
Create a custom subagent with specific system prompt and name for reuse.

Usage:
- Define specialized agents with custom roles and boundaries
- Created agents can be referenced by name in the Task tool
- Use this when you need a specific agent type not covered by predefined agents
- The created agent configuration will be saved and can be used immediately

Example workflow:
1. Use CreateSubagent to define a specialized agent (e.g., 'code_reviewer')
2. Use the Task tool with agent='code_reviewer' to launch the created agent
"""
    )


def test_send_dmail_description(send_dmail_tool: SendDMail):
    """Test the description of SendDMail tool."""
    assert send_dmail_tool.base.description == snapshot(
        """\
Send a message to the past, just like sending a D-Mail in Steins;Gate.

This tool is provided to enable you to proactively manage the context. You can see some `user` messages with text `CHECKPOINT {checkpoint_id}` wrapped in `<system>` tags in the context. When you feel there is too much irrelevant information in the current context, you can send a D-Mail to revert the context to a previous checkpoint with a message containing only the useful information. When you send a D-Mail, you must specify an existing checkpoint ID from the before-mentioned messages.

Typical scenarios you may want to send a D-Mail:

- You read a file, found it very large and most of the content is not relevant to the current task. In this case you can send a D-Mail immediately to the checkpoint before you read the file and give your past self only the useful part.
- You searched the web, the result is large.
  - If you got what you need, you may send a D-Mail to the checkpoint before you searched the web and put only the useful result in the mail message.
  - If you did not get what you need, you may send a D-Mail to tell your past self to try another query.
- You wrote some code and it did not work as expected. You spent many struggling steps to fix it but the process is not relevant to the ultimate goal. In this case you can send a D-Mail to the checkpoint before you wrote the code and give your past self the fixed version of the code and tell yourself no need to write it again because you already wrote to the filesystem.

After a D-Mail is sent, the system will revert the current context to the specified checkpoint, after which, you will no longer see any messages which you can now see after that checkpoint. The message in the D-Mail will be appended to the end of the context. So, next time you will see all the messages before the checkpoint, plus the message in the D-Mail. You must make it very clear in the message, tell your past self what you have done/changed, what you have learned and any other information that may be useful, so that your past self can continue the task without confusion and will not repeat the steps you have already done.

You must understand that, unlike D-Mail in Steins;Gate, the D-Mail you send here will not revert the filesystem or any external state. That means, you are basically folding the recent messages in your context into a single message, which can significantly reduce the waste of context window.

When sending a D-Mail, DO NOT explain to the user. The user do not care about this. Just explain to your past self.
"""
    )


def test_think_description(think_tool: Think):
    """Test the description of Think tool."""
    assert think_tool.base.description == snapshot(
        "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.\n"
    )


def test_set_todo_list_description(set_todo_list_tool: SetTodoList):
    """Test the description of SetTodoList tool."""
    assert set_todo_list_tool.base.description == snapshot(
        """\
Update the whole todo list.

Todo list is a simple yet powerful tool to help you get things done. You typically want to use this tool when the given task involves multiple subtasks/milestones, or, multiple tasks are given in a single request. This tool can help you break down the work, track the progress, and coordinate delegated work.

This is the only todo list tool available to you. That said, each time you want to operate on the todo list, you need to update the whole. Make sure to maintain the todo items and their statuses properly. Valid statuses are `pending`, `in_progress`, `done`, and `blocked`.

Todo items can also act as a lightweight execution plan:

- Prefer `executor="task"` for narrow, independent work that a subagent can finish and summarize back.
- If multiple todo items are independent, you may launch multiple `Task` calls in the same response.
- Keep todo titles unique when you plan to execute them later via `ExecuteTodo`.
- Set `subagent_name` when you already know which subagent should do the work.
- Use `done_when` to record a short completion criterion when it helps keep the plan grounded.
- When a stored task-backed todo should run now, prefer `ExecuteTodo` over manually chaining `SetTodoList` -> `Task` -> `SetTodoList`.
- After each `Task` returns, update the whole todo list to reflect progress or blockers.

The root agent owns the todo list. Subagents must not update it directly.

Abusing this tool to track too small steps will just waste your time and make your context messy. For example, here are some cases you should not use this tool:

- When the user just simply ask you a question. E.g. "What language and framework is used in the project?", "What is the best practice for x?"
- When it only takes a few steps/tool calls to complete the task. E.g. "Fix the unit test function 'test_xxx'", "Refactor the function 'xxx' to make it more solid."
- When the user prompt is very specific and the only thing you need to do is brainlessly following the instructions. E.g. "Replace xxx to yyy in the file zzz", "Create a file xxx with content yyy."

However, do not get stuck in a rut. Be flexible. Sometimes, you may try to use todo list at first, then realize the task is too simple and you can simply stop using it; or, sometimes, you may realize the task is complex after a few steps and then you can start using todo list to break it down.
"""
    )


def test_execute_todo_description(execute_todo_tool: ExecuteTodo):
    """Test the description of ExecuteTodo tool."""
    assert execute_todo_tool.base.description == snapshot(
        """\
Execute a stored todo item via `Task`.

Use this when the current session already has a todo list and one todo item should now be delegated to a subagent. This tool reads the persisted todo list from session state, marks the matching item `in_progress`, runs `Task`, then updates the item to `done` or `blocked`.

Guidelines:

- Use `SetTodoList` first if the current session does not yet have the todo list you want to execute.
- Match the todo by exact title, and keep todo titles unique to avoid ambiguity.
- Provide a detailed `prompt`; this tool does not remove the need to give the subagent full background.
- Prefer this tool over manually chaining `SetTodoList` -> `Task` -> `SetTodoList` for one delegated todo item.
- Do not call `SetTodoList` and `ExecuteTodo` in the same response when `ExecuteTodo` depends on the newly written list, because tool calls may run in parallel.
- Use this only for work that should run through `Task`. For long-running shell work, use `Shell` with `run_in_background=true` instead.
"""
    )


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
def test_shell_description(shell_tool: Shell):
    """Test the description of Shell tool."""
    description = shell_tool.base.description
    assert "run_in_background=true" in description
    assert "TaskOutput" in description
    assert "TaskStop" in description
    assert "/task" in description
    assert "Prefer `run_in_background=true` for long-running builds" in description


def test_read_file_description(read_file_tool: ReadFile):
    """Test the description of ReadFile tool."""
    assert read_file_tool.base.description == snapshot(
        """\
Read text content from a file.

**Tips:**
- Make sure you follow the description of each tool parameter.
- A `<system>` tag will be given before the read file content.
- The system will notify you when there is anything wrong when reading the file.
- This tool is a tool that you typically want to use in parallel. Always read multiple files in one response when possible.
- This tool can only read text files. To read images or videos, use other appropriate tools. To list directories, use the Glob tool or `ls` command via the Shell tool. To read other file types, use appropriate commands via the Shell tool.
- If the file doesn't exist or path is invalid, an error will be returned.
- If you want to search for a certain content/pattern, prefer Grep tool over ReadFile.
- Content will be returned with a line number before each line like `cat -n` format.
- Use `line_offset` and `n_lines` parameters when you only need to read a part of the file.
- `line_offset` can be negative to count backward from the end of the file. For example, `line_offset=-200` reads the last 200 lines.
- The tool result message includes the file's total line count when it is known.
- The maximum number of lines that can be read at once is 1000.
- Any lines longer than 2000 characters will be truncated, ending with "...".
"""
    )


def test_recall_compacted_context_description(
    recall_compacted_context_tool: RecallCompactedContext,
):
    """Test the description of RecallCompactedContext tool."""
    assert recall_compacted_context_tool.base.description == snapshot(
        """\
Recall details from previously compacted conversation context for the current conversation trajectory.

Use this tool when the compaction summary is not enough and you need older context details without reading raw archive files directly.

What it does:
- Lists available compacted-context archives for the current trajectory
- Searches archived pre-compaction messages by targeted keywords
- Returns small, relevant excerpts instead of the whole archive

Guidelines:
- Prefer specific queries such as file paths, function names, error strings, IDs, or distinctive keywords
- Leave `query` empty to list available archives and their short summaries first
- Use `archive_id` to narrow the search when you already know which archive is relevant
- This tool only reads archives created by compaction for the current trajectory
- Returned excerpts are sanitized and may omit hidden thinking content
"""
    )


def test_read_media_file_description(read_media_file_tool: ReadMediaFile):
    """Test the description of ReadMediaFile tool."""
    assert read_media_file_tool.base.description == snapshot(
        """\
Read media content from a file.

**Tips:**
- Make sure you follow the description of each tool parameter.
- A `<system>` tag will be given before the read file content.
- The system will notify you when there is anything wrong when reading the file.
- This tool is a tool that you typically want to use in parallel. Always read multiple files in one response when possible.
- This tool can only read image or video files. To read other types of files, use the ReadFile tool. To list directories, use the Glob tool or `ls` command via the Shell tool.
- If the file doesn't exist or path is invalid, an error will be returned.
- The maximum size that can be read is 100MB. An error will be returned if the file is larger than this limit.
- The media content will be returned in a form that you can directly view and understand.

**Capabilities**
- This tool supports image and video files for the current model.
"""
    )


def test_glob_description(glob_tool: Glob):
    """Test the description of Glob tool."""
    assert glob_tool.base.description == snapshot(
        """\
Find files and directories using glob patterns. This tool supports standard glob syntax like `*`, `?`, and `**` for recursive searches.

**When to use:**
- Find files matching specific patterns (e.g., all Python files: `*.py`)
- Search for files recursively in subdirectories (e.g., `src/**/*.js`)
- Locate configuration files (e.g., `*.config.*`, `*.json`)
- Find test files (e.g., `test_*.py`, `*_test.go`)

**Example patterns:**
- `*.py` - All Python files in current directory
- `src/**/*.js` - All JavaScript files in src directory recursively
- `test_*.py` - Python test files starting with "test_"
- `*.config.{js,ts}` - Config files with .js or .ts extension

**Bad example patterns:**
- `**`, `**/*.py` - Any pattern starting with '**' will be rejected. Because it would recursively search all directories and subdirectories, which is very likely to yield large result that exceeds your context size. Always use more specific patterns like `src/**/*.py` instead.
- `node_modules/**/*.js` - Although this does not start with '**', it would still highly possible to yield large result because `node_modules` is well-known to contain too many directories and files. Avoid recursively searching in such directories, other examples include `venv`, `.venv`, `__pycache__`, `target`. If you really need to search in a dependency, use more specific patterns like `node_modules/react/src/*` instead.
"""
    )


def test_grep_description(grep_tool: Grep):
    """Test the description of Grep tool."""
    assert grep_tool.base.description == snapshot(
        """\
A powerful search tool based-on ripgrep.

**Tips:**
- ALWAYS use Grep tool instead of running `grep` or `rg` command with Shell tool.
- Use the ripgrep pattern syntax, not grep syntax. E.g. you need to escape braces like `\\\\{` to search for `{`.
"""
    )


def test_write_file_description(write_file_tool: WriteFile):
    """Test the description of WriteFile tool."""
    assert write_file_tool.base.description == snapshot(
        """\
Write content to a file.

**Tips:**
- When `mode` is not specified, it defaults to `overwrite`. Always write with caution.
- When the content to write is too long (e.g. > 100 lines), use this tool multiple times instead of a single call. Use `overwrite` mode at the first time, then use `append` mode after the first write.
"""
    )


def test_edit_description(edit_tool: EditTool):
    """Test the description of Edit tool."""
    assert edit_tool.base.description == snapshot(
        """\
Edit a text file using structured edit operations.

**Tips:**
- Only use this tool on text files.
- You can provide a single edit operation or a list of operations in one call.
- Supported edit kinds are `replace`, `append`, `prepend`, `delete`, `insert_before`, `insert_after`, `replace_lines`, and `patch`.
- Replace operations must use `kind: "replace"`.
- `replace_lines` uses 1-based inclusive line numbers; negative values count backward from the end of the file.
- `patch` accepts unified diff or hunk-only patch text and must apply cleanly to the current file.
- You should prefer this tool over WriteFile tool and Shell `sed` command when you want focused edits instead of rewriting the whole file.
"""
    )


def test_search_web_description(search_web_tool: SearchWeb):
    """Test the description of MoonshotSearch tool."""
    assert search_web_tool.base.description == snapshot(
        "WebSearch tool allows you to search on the internet to get latest information, including news, documents, release notes, blog posts, papers, etc.\n"
    )


def test_fetch_url_description(fetch_url_tool: FetchURL):
    """Test the description of FetchURL tool."""
    assert fetch_url_tool.base.description == snapshot(
        "Fetch a web page from a URL and extract main text content from it.\n"
    )
