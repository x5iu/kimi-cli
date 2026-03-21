import json
from typing import Any, TypedDict, cast

import streamingjson  # pyright: ignore[reportMissingTypeStubs]
from kaos.path import KaosPath
from kosong.utils.typing import JsonType

from kimi_cli.tools.todo_text import todo_label
from kimi_cli.utils.string import shorten_middle


class SkipThisTool(Exception):
    """Raised when a tool decides to skip itself from the loading process."""

    pass


class _TodoSummaryDict(TypedDict, total=False):
    title: object
    subagent_name: object
    status: object
    executor: object


def _todo_label_from_dict(todo: _TodoSummaryDict) -> str | None:
    title = todo.get("title")
    if not isinstance(title, str) or not title:
        return None
    subagent_name = todo.get("subagent_name")
    return todo_label(
        title,
        subagent_name if isinstance(subagent_name, str) and subagent_name else None,
    )


def _summarize_set_todo_list_argument(curr_args: dict[str, Any]) -> str | None:
    raw_todos = curr_args.get("todos")
    if not isinstance(raw_todos, list):
        return None
    todos = cast(list[object], raw_todos)
    if not todos:
        return "empty"

    ready_task_todos: list[str] = []
    n_in_progress = 0
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        typed_todo = cast(_TodoSummaryDict, todo)
        if typed_todo.get("status") == "in_progress":
            n_in_progress += 1
        label = _todo_label_from_dict(typed_todo)
        if typed_todo.get("executor") == "task" and typed_todo.get("status") == "pending" and label:
            ready_task_todos.append(label)

    summary = f"{len(todos)} todos"
    if len(ready_task_todos) == 1:
        return f"{summary}; ready: {ready_task_todos[0]}"
    if n_in_progress:
        return f"{summary}; active={n_in_progress}"
    return summary


def _summarize_execute_todo_argument(curr_args: dict[str, Any]) -> str | None:
    title = curr_args.get("title")
    if not title:
        return None
    key_argument = str(title)
    subagent_name = curr_args.get("subagent_name")
    if subagent_name:
        key_argument += f" @{subagent_name}"
    return key_argument


def extract_key_argument(json_content: str | streamingjson.Lexer, tool_name: str) -> str | None:
    if isinstance(json_content, streamingjson.Lexer):
        json_str = json_content.complete_json()
    else:
        json_str = json_content
    try:
        curr_args: JsonType = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not curr_args:
        return None
    key_argument: str = ""
    should_truncate = True
    match tool_name:
        case "Task":
            if not isinstance(curr_args, dict) or not curr_args.get("description"):
                return None
            key_argument = str(curr_args["description"])
        case "CreateSubagent":
            if not isinstance(curr_args, dict) or not curr_args.get("name"):
                return None
            key_argument = str(curr_args["name"])
        case "SendDMail":
            return None
        case "Think":
            if not isinstance(curr_args, dict) or not curr_args.get("thought"):
                return None
            key_argument = str(curr_args["thought"])
        case "SetTodoList":
            if not isinstance(curr_args, dict):
                return None
            summary = _summarize_set_todo_list_argument(curr_args)
            if summary is None:
                return None
            key_argument = summary
        case "ExecuteTodo":
            if not isinstance(curr_args, dict):
                return None
            summary = _summarize_execute_todo_argument(curr_args)
            if summary is None:
                return None
            key_argument = summary
        case "Shell":
            if not isinstance(curr_args, dict) or not curr_args.get("command"):
                return None
            key_argument = str(curr_args["command"])
            should_truncate = False
        case "TaskOutput":
            if not isinstance(curr_args, dict) or not curr_args.get("task_id"):
                return None
            key_argument = str(curr_args["task_id"])
        case "TaskList":
            if not isinstance(curr_args, dict):
                return None
            key_argument = "active" if curr_args.get("active_only", True) else "all"
        case "TaskStop":
            if not isinstance(curr_args, dict) or not curr_args.get("task_id"):
                return None
            key_argument = str(curr_args["task_id"])
        case "ReadFile":
            if not isinstance(curr_args, dict) or not curr_args.get("path"):
                return None
            key_argument = _normalize_path(str(curr_args["path"]))
        case "ReadMediaFile":
            if not isinstance(curr_args, dict) or not curr_args.get("path"):
                return None
            key_argument = _normalize_path(str(curr_args["path"]))
        case "Glob":
            if not isinstance(curr_args, dict) or not curr_args.get("pattern"):
                return None
            key_argument = str(curr_args["pattern"])
        case "Grep":
            if not isinstance(curr_args, dict) or not curr_args.get("pattern"):
                return None
            key_argument = str(curr_args["pattern"])
        case "RecallCompactedContext":
            if not isinstance(curr_args, dict):
                return None
            if curr_args.get("query"):
                key_argument = str(curr_args["query"])
            elif curr_args.get("archive_id"):
                key_argument = str(curr_args["archive_id"])
            else:
                return None
        case "WriteFile":
            if not isinstance(curr_args, dict) or not curr_args.get("path"):
                return None
            key_argument = _normalize_path(str(curr_args["path"]))
        case "Edit" | "StrReplaceFile":
            if not isinstance(curr_args, dict) or not curr_args.get("path"):
                return None
            key_argument = _normalize_path(str(curr_args["path"]))
        case "SearchWeb":
            if not isinstance(curr_args, dict) or not curr_args.get("query"):
                return None
            key_argument = str(curr_args["query"])
        case "FetchURL":
            if not isinstance(curr_args, dict) or not curr_args.get("url"):
                return None
            key_argument = str(curr_args["url"])
        case _:
            if isinstance(json_content, streamingjson.Lexer):
                # lexer.json_content is list[str] based on streamingjson source code
                raw_content = getattr(json_content, "json_content", [])
                content = cast(list[str], raw_content)
                key_argument = "".join(content)
            else:
                key_argument = json_content
    if should_truncate:
        key_argument = shorten_middle(key_argument, width=50)
    return key_argument


def _normalize_path(path: str) -> str:
    cwd = str(KaosPath.cwd().canonical())
    if path.startswith(cwd):
        path = path[len(cwd) :].lstrip("/\\")
    return path
