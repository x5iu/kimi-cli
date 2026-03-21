"""Tests for the todo tool."""

from __future__ import annotations

from kosong.tooling import BriefDisplayBlock, ToolReturnValue

from kimi_cli.tools.display import TodoDisplayBlock
from kimi_cli.tools.todo import ExecuteTodoParams, Params, SetTodoList, Todo


class _FakeTaskTool:
    name = "Task"

    def __init__(self, result: ToolReturnValue):
        self.result = result
        self.calls = []

    async def __call__(self, params):
        self.calls.append(params)
        return self.result


async def test_set_todo_list_returns_summary_and_persists_state(
    set_todo_list_tool: SetTodoList,
    runtime,
):
    params = Params(
        todos=[
            Todo(
                title="Audit todo flow",
                status="in_progress",
                executor="task",
                subagent_name="coder",
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
    assert "Use Task for: Audit todo flow @coder" in result.output
    assert "In progress: Audit todo flow @coder" in result.output

    assert len(result.display) == 1
    display = result.display[0]
    assert isinstance(display, TodoDisplayBlock)
    assert display.items[0].executor == "task"
    assert display.items[0].subagent_name == "coder"
    assert display.items[0].done_when == "root agent has reviewed the returned summary"

    assert runtime.session.state.todos[0].title == "Audit todo flow"
    assert runtime.session.state.todos[0].status == "in_progress"


async def test_set_todo_list_adds_ready_to_execute_hint_for_single_pending_task_todo(
    set_todo_list_tool: SetTodoList,
):
    result = await set_todo_list_tool(
        Params(
            todos=[
                Todo(
                    title="Inspect parser",
                    status="pending",
                    executor="task",
                    subagent_name="coder",
                ),
                Todo(title="Share findings", status="pending", executor="main"),
            ]
        )
    )

    assert not result.is_error
    assert "Ready to ExecuteTodo: Inspect parser @coder" in result.output


async def test_set_todo_list_omits_ready_to_execute_hint_for_multiple_pending_task_todos(
    set_todo_list_tool: SetTodoList,
):
    result = await set_todo_list_tool(
        Params(
            todos=[
                Todo(
                    title="Inspect parser",
                    status="pending",
                    executor="task",
                    subagent_name="coder",
                ),
                Todo(
                    title="Inspect lexer",
                    status="pending",
                    executor="task",
                    subagent_name="coder",
                ),
            ]
        )
    )

    assert not result.is_error
    assert "Ready to ExecuteTodo:" not in result.output


async def test_execute_todo_runs_task_and_marks_todo_done(
    set_todo_list_tool: SetTodoList,
    execute_todo_tool,
    runtime,
    toolset,
):
    task_tool = _FakeTaskTool(
        ToolReturnValue(
            is_error=False,
            output="Subagent summary",
            message="Task finished",
            display=[],
        )
    )
    toolset.add(task_tool)

    await set_todo_list_tool(
        Params(
            todos=[
                Todo(
                    title="Inspect parser",
                    status="pending",
                    executor="task",
                    subagent_name="coder",
                    done_when="subagent reports the root cause",
                )
            ]
        )
    )

    result = await execute_todo_tool(
        ExecuteTodoParams(
            title="Inspect parser",
            description="inspect parser",
            prompt="Look at the parser module and summarize the issue.",
        )
    )

    assert not result.is_error
    assert result.message == 'Todo "Inspect parser" completed via Task.'
    assert "1 todos, done=1" in result.output
    assert "[Task output]" in result.output
    assert "Subagent summary" in result.output
    assert runtime.session.state.todos[0].status == "done"
    assert isinstance(result.display[0], BriefDisplayBlock)
    assert result.display[0].text == "Completed Todo: Inspect parser"
    assert len(task_tool.calls) == 1
    assert task_tool.calls[0].subagent_name == "coder"
    assert "Todo title: Inspect parser" in task_tool.calls[0].prompt
    assert "Done when: subagent reports the root cause" in task_tool.calls[0].prompt


async def test_execute_todo_marks_todo_blocked_when_task_fails(
    set_todo_list_tool: SetTodoList,
    execute_todo_tool,
    runtime,
    toolset,
):
    toolset.add(
        _FakeTaskTool(
            ToolReturnValue(
                is_error=True,
                output="Subagent failed",
                message="Task failed",
                display=[],
            )
        )
    )

    await set_todo_list_tool(
        Params(
            todos=[
                Todo(title="Fix build", status="pending", executor="task", subagent_name="coder")
            ]
        )
    )

    result = await execute_todo_tool(
        ExecuteTodoParams(
            title="Fix build",
            description="fix build",
            prompt="Investigate the build failure.",
        )
    )

    assert result.is_error
    assert result.message == "Blocked Todo: Fix build. Delegated Task failed."
    assert "1 todos, blocked=1" in result.output
    assert "Next state: blocked" in result.output
    assert (
        "Suggested next step: inspect [Task output], revise the prompt, or update the todo state before retrying."
        in result.output
    )
    assert runtime.session.state.todos[0].status == "blocked"


async def test_execute_todo_can_leave_todo_in_progress_after_task_failure(
    set_todo_list_tool: SetTodoList,
    execute_todo_tool,
    runtime,
    toolset,
):
    toolset.add(
        _FakeTaskTool(
            ToolReturnValue(
                is_error=True,
                output="Subagent failed",
                message="Task failed",
                display=[],
            )
        )
    )

    await set_todo_list_tool(
        Params(
            todos=[
                Todo(
                    title="Investigate flaky test",
                    status="pending",
                    executor="task",
                    subagent_name="coder",
                )
            ]
        )
    )

    result = await execute_todo_tool(
        ExecuteTodoParams(
            title="Investigate flaky test",
            description="investigate test",
            prompt="Look into the flaky test and report findings.",
            mark_blocked_on_error=False,
        )
    )

    assert result.is_error
    assert (
        result.message == "Todo Still In Progress: Investigate flaky test. Delegated Task failed."
    )
    assert "1 todos, in_progress=1" in result.output
    assert "Next state: in_progress" in result.output
    assert (
        "Suggested next step: inspect [Task output] before retrying ExecuteTodo or updating the todo state."
        in result.output
    )
    assert runtime.session.state.todos[0].status == "in_progress"
