from __future__ import annotations

# ruff: noqa

import platform
import pytest
from inline_snapshot import snapshot

from kimi_cli.tools.shell import Shell
from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.context import RecallCompactedContext
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import EditTool
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.think import Think
from kimi_cli.tools.todo import SetTodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb


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

Todo list is a simple yet powerful tool to help you get things done. You typically want to use this tool when the given task involves multiple subtasks/milestones, or, multiple tasks are given in a single request. This tool can help you break down the work and track the progress.

This is the only todo list tool available to you. That said, each time you want to operate on the todo list, you need to update the whole. Make sure to maintain the todo items and their statuses properly. Valid statuses are `pending`, `in_progress`, `done`, and `blocked`.

- Use `done_when` to record a short completion criterion when it helps keep the plan grounded.
- Use `executor="background_shell"` for long-running shell work, and `executor="main"` for work done directly.

Abusing this tool to track too small steps will just waste your time and make your context messy. For example, here are some cases you should not use this tool:

- When the user just simply ask you a question. E.g. "What language and framework is used in the project?", "What is the best practice for x?"
- When it only takes a few steps/tool calls to complete the task. E.g. "Fix the unit test function 'test_xxx'", "Refactor the function 'xxx' to make it more solid."
- When the user prompt is very specific and the only thing you need to do is brainlessly following the instructions. E.g. "Replace xxx to yyy in the file zzz", "Create a file xxx with content yyy."

However, do not get stuck in a rut. Be flexible. Sometimes, you may try to use todo list at first, then realize the task is too simple and you can simply stop using it; or, sometimes, you may realize the task is complex after a few steps and then you can start using todo list to break it down.
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
- If you want to search for a certain content/pattern, prefer Shell with `rg` (or the Grep tool if Shell is unavailable) over ReadFile.
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
A structured search tool wrapping ripgrep. Use as a fallback when the Shell tool with `rg` is unavailable or when you prefer a structured interface with built-in pagination.

**Tips:**
- When a `<system-hint>` gives you an `rg` path, prefer Shell with `rg` for maximum flexibility.
- Use this tool when Shell is not available, or for simple searches where structured output (files_with_matches, count_matches) is convenient.
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
        """\
Search the internet for latest information including news, documentation, release notes, blog posts, papers, and more.

**Tips:**
- Use specific, targeted queries — avoid vague or overly broad searches.
- When results don't contain what you need, refine the query rather than increasing `limit`.
- Set `include_content=true` only when you need page text; it consumes significant tokens.
- Prefer this tool over FetchURL when you don't have a specific URL.
"""
    )


def test_fetch_url_description(fetch_url_tool: FetchURL):
    """Test the description of FetchURL tool."""
    assert fetch_url_tool.base.description == snapshot(
        """\
Fetch a web page from a URL and extract its main text content.

**Tips:**
- Use this when you have a specific URL (from search results, documentation links, etc.).
- The tool extracts main content and strips navigation/ads — you get clean text.
- For very large pages, the content may be truncated.
"""
    )
