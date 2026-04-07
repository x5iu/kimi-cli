from typing import Literal

from pydantic import BaseModel

from llmkit.tooling import DisplayBlock

TodoStatus = Literal["pending", "in_progress", "done", "blocked"]
TodoExecutor = Literal["main", "background_shell"]


class DiffDisplayBlock(DisplayBlock):
    """Display block describing a file diff."""

    type: str = "diff"
    path: str
    old_text: str
    new_text: str
    old_start_line: int = 1
    new_start_line: int = 1


class TodoDisplayItem(BaseModel):
    title: str
    status: TodoStatus
    executor: TodoExecutor | None = None
    done_when: str | None = None


class TodoDisplayBlock(DisplayBlock):
    """Display block describing a todo list update."""

    type: str = "todo"
    items: list[TodoDisplayItem]


class ShellDisplayBlock(DisplayBlock):
    """Display block describing a shell command."""

    type: str = "shell"
    language: str
    command: str


class BackgroundTaskDisplayBlock(DisplayBlock):
    """Display block describing a background task."""

    type: str = "background_task"
    task_id: str
    kind: str
    status: str
    description: str
