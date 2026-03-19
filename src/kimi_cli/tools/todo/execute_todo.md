Execute a stored todo item via `Task`.

Use this when the current session already has a todo list and one todo item should now be delegated to a subagent. This tool reads the persisted todo list from session state, marks the matching item `in_progress`, runs `Task`, then updates the item to `done` or `blocked`.

Guidelines:

- Use `SetTodoList` first if the current session does not yet have the todo list you want to execute.
- Match the todo by exact title, and keep todo titles unique to avoid ambiguity.
- Provide a detailed `prompt`; this tool does not remove the need to give the subagent full background.
- Prefer this tool over manually chaining `SetTodoList` -> `Task` -> `SetTodoList` for one delegated todo item.
- Do not call `SetTodoList` and `ExecuteTodo` in the same response when `ExecuteTodo` depends on the newly written list, because tool calls may run in parallel.
- Use this only for work that should run through `Task`. For long-running shell work, use `Shell` with `run_in_background=true` instead.
