Retrieve output from a running or completed background task.

Use this after `Shell(run_in_background=true)` when you need to inspect progress or explicitly wait for completion.

Guidelines:
- Prefer relying on automatic completion notifications. Use this tool only when you need task output before the automatic notification arrives.
- For interactive tasks (`interactive=true`), `block=true` waits for the current turn to complete (not task termination). This is the expected workflow — completion notifications do not apply to interactive tasks because they never exit on their own.
- Use `block=true` to wait for completion or timeout.
- Use `block=false` for a non-blocking status and output check.
- `timeout` (default 30s, max 3600s): How long to wait when `block=true`. If the task does not finish within this time, `retrieval_status` will be `timeout` (task still running). With `block=false`, a still-running task yields `not_ready`.
- `offset`: 0-based **raw line** offset. By default, the output preview shows the **tail** (last lines fitting within ~32 KiB). Set `offset=0` to read from the beginning instead.
- `mode` (default `auto`) controls how the `[output]` block is rendered:
  - `auto`: structured logs (Claude `--output-format stream-json`, Codex JSONL, Kimi/agent NDJSON) are projected into a concise event stream; plain-text output is emitted as bounded raw lines (legacy behavior).
  - `summary`: always project; plain text becomes a bounded tail digest.
  - `raw`: emit the raw line window, bounded by `max_bytes` (raise `max_bytes` for larger reads). Combine with `tail_lines` to keep only the last N lines.
  - `none`: omit the `[output]` block entirely. Useful when you only want status/metadata.
- `max_bytes` (default 4096) caps the full rendered `[output]` payload (including summary/truncation headers). `max_lines` caps the number of summarized events. `tail_lines` keeps only the last N raw lines in `raw` mode.
- Even when summarized, `output_path` and `output_next_offset` continue to reference **raw** line numbers, so forward pagination is unaffected by `mode`.
- The internal preview is read in whole lines capped at 32 KiB. If a single line exceeds the cap it is omitted and a hint directs you to use `ReadFile` on the `output_path`.
- The response includes `output_preview_start_line`, `output_preview_end_line` (exclusive — the range is `[start, end)`), `output_has_before`, `output_has_after`, and `output_next_offset` for line-level pagination. Use `output_next_offset` as the next `offset` value to continue forward.
- When the preview is truncated or the summary drops detail, use `ReadFile` with the returned `output_path` to inspect the full raw log in pages.
- This tool works with the generic background task system and should remain the primary read path for future task types, not just bash.
- When context budget is tight, always pass `offset` (from the previous `output_next_offset`) to read only new lines instead of re-reading the full output.

