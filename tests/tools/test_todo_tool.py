"""Tests for the todo tool."""

from __future__ import annotations

from kimi_cli.tools.display import TodoDisplayBlock
from kimi_cli.tools.todo import Params, SetTodoList, Todo


async def test_set_todo_list_returns_summary_and_persists_state(
    set_todo_list_tool: SetTodoList,
    runtime,
):
    params = Params(
        todos=[
            Todo(
                title="Audit todo flow",
                status="in_progress",
                executor="main",
                done_when="root agent has reviewed the returned summary",
            ),
            Todo(title="Share findings", status="pending", executor="main"),
        ]
    )

    result = await set_todo_list_tool(params)

    assert not result.is_error
    assert result.message == "Todo list updated"
    assert isinstance(result.output, str)
    assert "2 todos, pending=1, in_progress=1" in result.output
    assert "In progress: Audit todo flow" in result.output

    assert len(result.display) == 1
    display = result.display[0]
    assert isinstance(display, TodoDisplayBlock)
    assert display.items[0].executor == "main"
    assert display.items[0].done_when == "root agent has reviewed the returned summary"

    assert runtime.session.state.todos[0].title == "Audit todo flow"
    assert runtime.session.state.todos[0].status == "in_progress"
