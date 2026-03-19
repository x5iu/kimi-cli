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
