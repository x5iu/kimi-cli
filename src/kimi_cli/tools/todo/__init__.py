import asyncio
from pathlib import Path
from typing import Literal, override

from kosong.tooling import BriefDisplayBlock, CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.session import Session
from kimi_cli.session_state import TodoStateItem
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.display import TodoDisplayBlock, TodoDisplayItem
from kimi_cli.tools.multiagent.task import Params as TaskParams
from kimi_cli.tools.todo_text import (
    completed_todo_text,
    execute_todo_failure_message,
    execute_todo_suggested_next_step,
    next_state_text,
    ready_to_execute_todo_text,
    todo_label,
)
from kimi_cli.tools.utils import load_desc

TodoStatus = Literal["pending", "in_progress", "done", "blocked"]
TodoExecutor = Literal["main", "task", "background_shell"]

_TODO_LOCKS: dict[str, asyncio.Lock] = {}


class Todo(BaseModel):
    title: str = Field(description="The title of the todo", min_length=1)
    status: TodoStatus = Field(description="The status of the todo")
    executor: TodoExecutor | None = Field(
        default=None,
        description=(
            "How this todo should be executed. Prefer `task` for narrow, independent work "
            "that can be delegated to a subagent; use `main` for work done by the root "
            "agent; use `background_shell` for long-running shell work."
        ),
    )
    subagent_name: str | None = Field(
        default=None,
        description="The preferred subagent name when `executor` is `task`.",
    )
    done_when: str | None = Field(
        default=None,
        description="A short completion criterion for this todo.",
    )


class Params(BaseModel):
    todos: list[Todo] = Field(description="The updated todo list")


class ExecuteTodoParams(BaseModel):
    title: str = Field(
        description=(
            "The exact title of the todo item to execute from the current session todo list."
        )
    )
    description: str = Field(description="A short (3-5 word) description of the delegated task.")
    prompt: str = Field(
        description=(
            "The detailed prompt for the delegated Task call. You must still provide all "
            "necessary background because the subagent cannot see your context."
        )
    )
    subagent_name: str | None = Field(
        default=None,
        description=(
            "Optional override for the subagent name. Defaults to `todos[].subagent_name`."
        ),
    )
    mark_blocked_on_error: bool = Field(
        default=True,
        description="Whether to mark the todo as `blocked` when Task returns an error.",
    )


def _todo_lock(session: Session) -> asyncio.Lock:
    key = str(session.context_file.resolve())
    if key not in _TODO_LOCKS:
        _TODO_LOCKS[key] = asyncio.Lock()
    return _TODO_LOCKS[key]


def _ensure_root(runtime: Runtime) -> ToolError | None:
    if runtime.role != "root":
        return ToolError(
            message="Todo tools can only be managed by the root agent.",
            brief="Todo unavailable",
        )
    return None


def _todo_to_display_item(todo: Todo) -> TodoDisplayItem:
    return TodoDisplayItem(
        title=todo.title,
        status=todo.status,
        executor=todo.executor,
        subagent_name=todo.subagent_name,
        done_when=todo.done_when,
    )


def _todo_to_state_item(todo: Todo) -> TodoStateItem:
    return TodoStateItem(
        title=todo.title,
        status=todo.status,
        executor=todo.executor,
        subagent_name=todo.subagent_name,
        done_when=todo.done_when,
    )


def _todo_from_state_item(item: TodoStateItem) -> Todo:
    return Todo(
        title=item.title,
        status=item.status,
        executor=item.executor,
        subagent_name=item.subagent_name,
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
    return todo_label(todo.title, todo.subagent_name)


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

    task_todos = [
        _shorten_todo(todo)
        for todo in todos
        if todo.executor == "task" and todo.status not in {"done", "blocked"}
    ]
    if task_todos:
        suffix = "; ..." if len(task_todos) > 3 else ""
        lines.append(f"Use Task for: {'; '.join(task_todos[:3])}{suffix}")

    in_progress = [_shorten_todo(todo) for todo in todos if todo.status == "in_progress"]
    if in_progress:
        suffix = "; ..." if len(in_progress) > 3 else ""
        lines.append(f"In progress: {'; '.join(in_progress[:3])}{suffix}")

    ready_to_execute = [
        _shorten_todo(todo)
        for todo in todos
        if todo.executor == "task" and todo.status == "pending"
    ]
    if len(ready_to_execute) == 1:
        lines.append(ready_to_execute_todo_text(ready_to_execute[0]))

    return "\n".join(lines)


def _format_task_output(task_result: ToolReturnValue) -> str:
    parts: list[str] = []
    if task_result.message:
        parts.append(task_result.message)
    if isinstance(task_result.output, str) and task_result.output.strip():
        parts.append(task_result.output.strip())
    return "\n\n".join(parts)


def _merge_output(summary: str, details: str) -> str:
    details = details.strip()
    if not details:
        return summary
    return f"{summary}\n\n[Task output]\n{details}"


def _execute_todo_failure_summary(summary: str, *, next_state: str) -> str:
    return "\n".join(
        [
            summary,
            next_state_text(next_state),
            execute_todo_suggested_next_step(next_state),
        ]
    )


def _build_task_prompt(todo: Todo, prompt: str) -> str:
    lines = [f"Todo title: {todo.title}"]
    if todo.done_when:
        lines.append(f"Done when: {todo.done_when}")
    lines.append("")
    lines.append(prompt.strip())
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
        if err := _ensure_root(self._runtime):
            return err

        async with _todo_lock(self._runtime.session):
            _save_todos_unlocked(self._runtime.session, params.todos)

        return ToolReturnValue(
            is_error=False,
            output=_summarize_todos(params.todos),
            message="Todo list updated",
            display=[_todo_display_block(params.todos)],
        )


class ExecuteTodo(CallableTool2[ExecuteTodoParams]):
    name: str = "ExecuteTodo"
    description: str = load_desc(Path(__file__).parent / "execute_todo.md")
    params: type[ExecuteTodoParams] = ExecuteTodoParams

    def __init__(self, toolset: KimiToolset, runtime: Runtime):
        super().__init__()
        self._toolset = toolset
        self._runtime = runtime

    @staticmethod
    def _find_todo(title: str, todos: list[Todo]) -> tuple[int, Todo] | ToolError:
        matches = [(index, todo) for index, todo in enumerate(todos) if todo.title == title]
        if not matches:
            return ToolError(
                message=(
                    f'Todo "{title}" was not found in the current session. '
                    "Use SetTodoList first, or make sure the title matches exactly."
                ),
                brief="Todo not found",
            )
        if len(matches) > 1:
            return ToolError(
                message=(
                    f'Multiple todos share the title "{title}". '
                    "Use unique todo titles before calling ExecuteTodo."
                ),
                brief="Ambiguous todo",
            )
        return matches[0]

    async def __call__(self, params: ExecuteTodoParams) -> ToolReturnValue:
        if err := _ensure_root(self._runtime):
            return err

        task_tool = self._toolset.find("Task")
        if task_tool is None:
            return ToolError(
                message="Task tool is not available in the current agent.",
                brief="Task unavailable",
            )

        async with _todo_lock(self._runtime.session):
            todos = _load_todos_unlocked(self._runtime.session)
            if not todos:
                return ToolError(
                    message="Todo list is empty. Use SetTodoList before ExecuteTodo.",
                    brief="Todo unavailable",
                )
            match = self._find_todo(params.title, todos)
            if isinstance(match, ToolError):
                return match
            index, todo = match
            if todo.status == "done":
                return ToolError(
                    message=f'Todo "{todo.title}" is already marked done.',
                    brief="Todo already done",
                )
            if todo.status == "in_progress":
                return ToolError(
                    message=f'Todo "{todo.title}" is already in progress.',
                    brief="Todo already running",
                )
            if todo.executor == "background_shell" and params.subagent_name is None:
                return ToolError(
                    message=(
                        f'Todo "{todo.title}" is assigned to `background_shell`. '
                        "Use Shell with run_in_background=true instead, or provide a "
                        "subagent override if you really want to delegate it."
                    ),
                    brief="Executor mismatch",
                )
            subagent_name = params.subagent_name or todo.subagent_name
            if todo.executor == "main" and params.subagent_name is None and not todo.subagent_name:
                return ToolError(
                    message=(
                        f'Todo "{todo.title}" is assigned to `main`. '
                        "Do it yourself, or pass `subagent_name` to delegate it explicitly."
                    ),
                    brief="Executor mismatch",
                )
            if not subagent_name:
                return ToolError(
                    message=(
                        f'Todo "{todo.title}" does not have a subagent yet. '
                        "Set `todos[].subagent_name` or pass `subagent_name`."
                    ),
                    brief="Subagent missing",
                )
            todo.status = "in_progress"
            todo.executor = "task"
            todo.subagent_name = subagent_name
            todos[index] = todo
            _save_todos_unlocked(self._runtime.session, todos)

        task_result = await task_tool(
            TaskParams(
                description=params.description,
                subagent_name=subagent_name,
                prompt=_build_task_prompt(todo, params.prompt),
            )
        )

        async with _todo_lock(self._runtime.session):
            latest_todos = _load_todos_unlocked(self._runtime.session)
            latest_match = self._find_todo(params.title, latest_todos)
            if isinstance(latest_match, ToolError):
                latest_todos = todos
                latest_match = self._find_todo(params.title, latest_todos)
            assert not isinstance(latest_match, ToolError)
            latest_index, latest_todo = latest_match
            latest_todo.executor = "task"
            latest_todo.subagent_name = subagent_name
            latest_todo.status = (
                "blocked" if task_result.is_error and params.mark_blocked_on_error else "done"
            )
            if task_result.is_error and not params.mark_blocked_on_error:
                latest_todo.status = "in_progress"
            latest_todos[latest_index] = latest_todo
            _save_todos_unlocked(self._runtime.session, latest_todos)

        summary = _summarize_todos(latest_todos)
        nested_output = _format_task_output(task_result)
        if task_result.is_error:
            next_state = "blocked" if params.mark_blocked_on_error else "in_progress"
            return ToolReturnValue(
                is_error=True,
                output=_merge_output(
                    _execute_todo_failure_summary(summary, next_state=next_state),
                    nested_output,
                ),
                message=execute_todo_failure_message(params.title, next_state=next_state),
                display=[_todo_display_block(latest_todos), *task_result.display],
            )

        return ToolReturnValue(
            is_error=False,
            output=_merge_output(summary, nested_output),
            message=f'Todo "{params.title}" completed via Task.',
            display=[
                BriefDisplayBlock(text=completed_todo_text(params.title)),
                _todo_display_block(latest_todos),
                *task_result.display,
            ],
        )
