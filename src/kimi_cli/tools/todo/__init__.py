import asyncio
from pathlib import Path
from typing import Literal, override

from llmkit.tooling import CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.loop.agent import Runtime
from kimi_cli.session import Session
from kimi_cli.session_state import TodoStateItem
from kimi_cli.tools.display import TodoDisplayBlock, TodoDisplayItem
from kimi_cli.tools.todo_text import todo_label
from kimi_cli.tools.utils import load_desc

TodoStatus = Literal["pending", "in_progress", "done", "blocked"]
TodoExecutor = Literal["main", "background_shell"]

_TODO_LOCKS: dict[str, asyncio.Lock] = {}


class Todo(BaseModel):
    title: str = Field(description="The title of the todo", min_length=1)
    status: TodoStatus = Field(description="The status of the todo")
    executor: TodoExecutor | None = Field(
        default=None,
        description=(
            "How this todo should be executed. Use `main` for work done by the "
            "agent; use `background_shell` for long-running shell work."
        ),
    )
    done_when: str | None = Field(
        default=None,
        description="A short completion criterion for this todo.",
    )


_MAX_TODOS = 25


class Params(BaseModel):
    todos: list[Todo] = Field(description="The updated todo list", max_length=_MAX_TODOS)


def _todo_lock(session: Session) -> asyncio.Lock:
    key = str(session.context_file.resolve())
    if key not in _TODO_LOCKS:
        _TODO_LOCKS[key] = asyncio.Lock()
    return _TODO_LOCKS[key]




def _todo_to_display_item(todo: Todo) -> TodoDisplayItem:
    return TodoDisplayItem(
        title=todo.title,
        status=todo.status,
        executor=todo.executor,
        done_when=todo.done_when,
    )


def _todo_to_state_item(todo: Todo) -> TodoStateItem:
    return TodoStateItem(
        title=todo.title,
        status=todo.status,
        executor=todo.executor,
        done_when=todo.done_when,
    )


def _todo_from_state_item(item: TodoStateItem) -> Todo:
    return Todo(
        title=item.title,
        status=item.status,
        executor=item.executor,
        done_when=item.done_when,
    )


def _load_todos_unlocked(session: Session) -> list[Todo]:
    return [_todo_from_state_item(item) for item in session.state.todos]


def _save_todos_unlocked(session: Session, todos: list[Todo]) -> None:
    session.state.todos = [_todo_to_state_item(todo) for todo in todos]
    session.save_state()


def _todo_display_block(todos: list[Todo]) -> TodoDisplayBlock:
    return TodoDisplayBlock(items=[_todo_to_display_item(todo) for todo in todos])


def _shorten_todo(todo: Todo) -> str:
    return todo_label(todo.title)


def _diff_todos(old: list[Todo], new: list[Todo]) -> str:
    """Return a short human-readable summary of what changed between two todo lists."""
    old_map: dict[str, TodoStatus] = {t.title: t.status for t in old}
    new_map: dict[str, TodoStatus] = {t.title: t.status for t in new}

    parts: list[str] = []
    transitions: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    unchanged = 0

    for title, status in new_map.items():
        if title not in old_map:
            added.append(f"'{_shorten_todo(Todo(title=title, status=status))}'")
        elif old_map[title] != status:
            transitions.append(
                f"'{_shorten_todo(Todo(title=title, status=status))}' "
                f"{old_map[title]} -> {status}"
            )
        else:
            unchanged += 1

    for title in old_map:
        if title not in new_map:
            removed.append(f"'{_shorten_todo(Todo(title=title, status=old_map[title]))}'")

    for t in transitions:
        parts.append(f"Marked {t}.")
    for a in added:
        parts.append(f"Added {a}.")
    for r in removed:
        parts.append(f"Removed {r}.")
    if unchanged:
        parts.append(f"{unchanged} item{'s' if unchanged != 1 else ''} unchanged.")

    return " ".join(parts) if parts else "No changes."


def _summarize_todos(todos: list[Todo]) -> str:
    if not todos:
        return "Todo list is empty."

    ordered_statuses: tuple[TodoStatus, ...] = ("pending", "in_progress", "done", "blocked")
    counts = {status: 0 for status in ordered_statuses}
    for todo in todos:
        counts[todo.status] += 1

    summary_parts = [f"{len(todos)} todos"]
    summary_parts.extend(
        f"{status}={counts[status]}" for status in ordered_statuses if counts[status]
    )
    lines = [", ".join(summary_parts)]

    in_progress = [_shorten_todo(todo) for todo in todos if todo.status == "in_progress"]
    if in_progress:
        suffix = "; ..." if len(in_progress) > 3 else ""
        lines.append(f"In progress: {'; '.join(in_progress[:3])}{suffix}")

    return "\n".join(lines)


class SetTodoList(CallableTool2[Params]):
    name: str = "SetTodoList"
    description: str = load_desc(Path(__file__).parent / "set_todo_list.md")
    params: type[Params] = Params

    def __init__(self, runtime: Runtime):
        super().__init__()
        self._runtime = runtime

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        async with _todo_lock(self._runtime.session):
            old_todos = _load_todos_unlocked(self._runtime.session)
            _save_todos_unlocked(self._runtime.session, params.todos)

        diff = _diff_todos(old_todos, params.todos)
        summary = _summarize_todos(params.todos)
        output = f"{summary}\nChanges: {diff}"

        return ToolReturnValue(
            is_error=False,
            output=output,
            message="Todo list updated",
            display=[_todo_display_block(params.todos)],
        )


