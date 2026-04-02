import json
from typing import Any, TypedDict, cast

import streamingjson  # pyright: ignore[reportMissingTypeStubs]
from kaos.path import KaosPath
from kosong.utils.typing import JsonType

from kimi_cli.tools.todo_text import todo_label


class SkipThisTool(Exception):
    """Raised when a tool decides to skip itself from the loading process."""

    pass


class _TodoSummaryDict(TypedDict, total=False):
    title: object
    status: object
    executor: object


def _todo_label_from_dict(todo: _TodoSummaryDict) -> str | None:
    title = todo.get("title")
    if not isinstance(title, str) or not title:
        return None
    return todo_label(title)


def _summarize_set_todo_list_argument(curr_args: dict[str, Any]) -> str | None:
    raw_todos = curr_args.get("todos")
    if not isinstance(raw_todos, list):
        return None
    todos = cast(list[object], raw_todos)
    if not todos:
        return "empty"

    n_in_progress = 0
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        typed_todo = cast(_TodoSummaryDict, todo)
        if typed_todo.get("status") == "in_progress":
            n_in_progress += 1

    summary = f"{len(todos)} todos"
    if n_in_progress:
        return f"{summary}; active={n_in_progress}"
    return summary


def extract_key_argument(json_content: str | streamingjson.Lexer, tool_name: str) -> str | None:
    if isinstance(json_content, streamingjson.Lexer):
        json_str = json_content.complete_json()
    else:
        json_str = json_content
    try:
        curr_args: JsonType = json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        return None
    if not curr_args:
        return None
    key_argument: str = ""
    match tool_name:
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
        case "Shell":
            if not isinstance(curr_args, dict) or not curr_args.get("command"):
                return None
            key_argument = str(curr_args["command"])
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
        case "Edit":
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
    return key_argument


def _normalize_path(path: str) -> str:
    cwd = str(KaosPath.cwd().canonical())
    if path.startswith(cwd):
        path = path[len(cwd) :].lstrip("/\\")
    return path
