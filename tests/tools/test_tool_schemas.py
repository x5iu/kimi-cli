from __future__ import annotations

# ruff: noqa

from inline_snapshot import snapshot

from kimi_cli.tools.background import TaskList, TaskOutput, TaskStop
from kimi_cli.tools.shell import Shell
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.context import RecallCompactedContext
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import EditTool
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.todo import SetTodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb


def test_set_todo_list_params_schema(set_todo_list_tool: SetTodoList):
    """Test the schema of SetTodoList tool parameters."""
    assert set_todo_list_tool.base.parameters == snapshot(
        {
            "properties": {
                "todos": {
                    "description": "The updated todo list",
                    "items": {
                        "properties": {
                            "title": {
                                "description": "The title of the todo",
                                "minLength": 1,
                                "type": "string",
                            },
                            "status": {
                                "description": "The status of the todo",
                                "enum": ["pending", "in_progress", "done", "blocked"],
                                "type": "string",
                            },
                            "executor": {
                                "anyOf": [
                                    {
                                        "enum": ["main", "background_shell"],
                                        "type": "string",
                                    },
                                    {"type": "null"},
                                ],
                                "default": None,
                                "description": "How this todo should be executed. Use `main` for work done by the agent; use `background_shell` for long-running shell work.",
                            },
                            "done_when": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "description": "A short completion criterion for this todo.",
                            },
                        },
                        "required": ["title", "status"],
                        "type": "object",
                    },
                    "maxItems": 25,
                    "type": "array",
                }
            },
            "required": ["todos"],
            "type": "object",
        }
    )


def test_shell_params_schema(shell_tool: Shell):
    """Test the schema of Shell tool parameters."""
    assert shell_tool.base.parameters == snapshot(
        {
            "properties": {
                "command": {
                    "description": "The command to execute.",
                    "type": "string",
                },
                "timeout": {
                    "default": 60,
                    "description": "The timeout in seconds for the command to execute. If the command takes longer than this, it will be killed.",
                    "maximum": 86400,
                    "minimum": 1,
                    "type": "integer",
                },
                "run_in_background": {
                    "default": False,
                    "description": "Whether to run the command as a background task.",
                    "type": "boolean",
                },
                "description": {
                    "default": "",
                    "description": "A short description for the background task. Required when run_in_background=true.",
                    "type": "string",
                },
                "interactive": {
                    "default": False,
                    "description": "Whether the background task needs stdin interaction via TaskWrite. When true, use TaskWrite to send input to the task's stdin. Requires run_in_background=true.",
                    "type": "boolean",
                },
            },
            "required": ["command"],
            "type": "object",
        }
    )


def test_task_output_params_schema(task_output_tool: TaskOutput):
    assert task_output_tool.base.parameters == snapshot(
        {
            "properties": {
                "task_id": {
                    "description": "The background task ID to inspect.",
                    "type": "string",
                },
                "block": {
                    "default": True,
                    "description": "Whether to wait for the task to finish before returning.",
                    "type": "boolean",
                },
                "timeout": {
                    "default": 30,
                    "description": "Maximum number of seconds to wait when block=true.",
                    "maximum": 3600,
                    "minimum": 0,
                    "type": "integer",
                },
                "offset": {
                    "anyOf": [{"minimum": 0, "type": "integer"}, {"type": "null"}],
                    "default": None,
                    "description": "Line offset (0-based) to start reading output from. If not set, reads the last lines that fit within ~32 KiB (tail). Set to 0 to read from the beginning.",
                },
            },
            "required": ["task_id"],
            "type": "object",
        }
    )


def test_task_list_params_schema(task_list_tool: TaskList):
    assert task_list_tool.base.parameters == snapshot(
        {
            "properties": {
                "active_only": {
                    "default": True,
                    "description": "Whether to list only non-terminal background tasks.",
                    "type": "boolean",
                },
                "limit": {
                    "default": 20,
                    "description": "Maximum number of tasks to return.",
                    "maximum": 100,
                    "minimum": 1,
                    "type": "integer",
                },
            },
            "type": "object",
        }
    )


def test_task_stop_params_schema(task_stop_tool: TaskStop):
    assert task_stop_tool.base.parameters == snapshot(
        {
            "properties": {
                "task_id": {
                    "description": "The background task ID to stop.",
                    "type": "string",
                },
                "reason": {
                    "default": "Stopped by TaskStop",
                    "description": "Short reason recorded when the task is stopped.",
                    "type": "string",
                },
            },
            "required": ["task_id"],
            "type": "object",
        }
    )


def test_read_file_params_schema(read_file_tool: ReadFile):
    """Test the schema of ReadFile tool parameters."""
    assert read_file_tool.base.parameters == snapshot(
        {
            "properties": {
                "path": {
                    "description": "The path to the file to read. Absolute paths are required when reading files outside the working directory.",
                    "type": "string",
                },
                "line_offset": {
                    "default": 1,
                    "description": "The line number to start reading from. Positive values count from the beginning of the file. Negative values count backward from the end of the file, where -1 is the last line. By default read from the beginning of the file. Set this when the file is too large to read at once or when you want to read the tail of a file.",
                    "not": {"const": 0},
                    "type": "integer",
                },
                "n_lines": {
                    "default": 1000,
                    "description": "The number of lines to read. By default read up to 1000 lines, which is the max allowed value. Set this value when the file is too large to read at once.",
                    "minimum": 1,
                    "type": "integer",
                },
            },
            "required": ["path"],
            "type": "object",
        }
    )


def test_recall_compacted_context_params_schema(
    recall_compacted_context_tool: RecallCompactedContext,
):
    """Test the schema of RecallCompactedContext tool parameters."""
    assert recall_compacted_context_tool.base.parameters == snapshot(
        {
            "properties": {
                "query": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Targeted keywords to search for in compacted-context archives. Leave empty to list available archives and their summaries.",
                },
                "archive_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Optional archive ID like `c001` to restrict lookup to a single compacted-context archive.",
                },
                "max_results": {
                    "default": 3,
                    "description": "Maximum number of excerpts to return. Defaults to 3, max 5.",
                    "maximum": 5,
                    "minimum": 1,
                    "type": "integer",
                },
            },
            "type": "object",
        }
    )


def test_read_media_file_params_schema(read_media_file_tool: ReadMediaFile):
    """Test the schema of ReadMediaFile tool parameters."""
    assert read_media_file_tool.base.parameters == snapshot(
        {
            "properties": {
                "path": {
                    "description": "The path to the file to read. Absolute paths are required when reading files outside the working directory.",
                    "type": "string",
                }
            },
            "required": ["path"],
            "type": "object",
        }
    )
