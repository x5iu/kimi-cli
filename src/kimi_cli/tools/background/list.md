List background tasks from the current session.

Use this when you need to re-enumerate which background tasks still exist, especially after context compaction or when you are no longer confident which task IDs are still active.

Parameters:
- `active_only` (default true): If true, only non-terminal tasks are listed.
- `limit` (default 20, max 100): Maximum number of tasks returned.

Returns a numbered list of tasks with fields: task_id, kind, status, description, command, exit_code (if terminal), reason (if failed), elapsed_s.

Guidelines:
- Prefer the default `active_only=true` unless you specifically need completed or failed tasks.
- Use `TaskOutput` to inspect one task in detail after you have identified the correct task ID.
- Do not guess which tasks are still running when you can call this tool directly.
- This tool is read-only and safe to use in plan mode.
