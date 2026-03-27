Retrieve output from a running or completed background task.

Use this after `Shell(run_in_background=true)` when you need to inspect progress or explicitly wait for completion.

Guidelines:
- Prefer relying on automatic completion notifications. Use this tool only when you need task output before the automatic notification arrives.
- Use `block=true` to wait for completion or timeout.
- Use `block=false` for a non-blocking status and output check.
- `timeout` (default 30s, max 3600s): How long to wait when `block=true`. If the task does not finish within this time, `retrieval_status` will be `timeout` (task still running). With `block=false`, a still-running task yields `not_ready`.
- `offset`: By default, the output preview shows the **tail** (last ~32 KiB). Set `offset=0` to read from the beginning instead.
- This tool returns structured task metadata, a fixed-size output preview, and an `output_path` for the full log.
- When the preview is truncated, use `ReadFile` with the returned `output_path` to inspect the full log in pages.
- This tool works with the generic background task system and should remain the primary read path for future task types, not just bash.
