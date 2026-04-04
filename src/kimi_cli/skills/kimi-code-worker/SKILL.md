---
name: kimi-code-worker
description: Spawn Kimi Code CLI (`kimi`) as a sub-agent via shell commands to work on tasks. Use headless mode (`kimi --print`) for single-shot tasks. Supports parallel fan-out for independent subtasks like batch fixes, code review, migrations, or analysis.
---

# kimi-code-worker

Orchestrate Kimi Code CLI (`kimi`) as sub-agent workers in headless mode.

## Mandatory rules

1. **Must use Shell tool.** Always invoke `kimi` via the Shell tool — never
   ask the user to run it manually.
2. **Must run in background.** Always set `run_in_background=true` on the Shell
   call. This keeps the main conversation responsive while the worker runs.
   Use `TaskOutput` to poll progress or wait for completion.
3. **Must use `--output-format stream-json`.** This flag serialises events as
   incremental JSON to stdout. When you inspect the background task via
   `TaskOutput`, you can see exactly which step the worker is on (thinking,
   tool calls, partial results) without waiting for the full run to finish.
4. **Scope by directory.** When a task targets a specific directory, `cd` into
   it before launching `kimi`. This constrains the worker's scope to that
   directory.

Putting it together — every headless call looks like:

```
Shell(
  command='cd <target-dir> && kimi --print --output-format stream-json -p "prompt"',
  run_in_background=true,
  description="short task description"
)
```

Then check progress / collect results:

```
TaskOutput(task_id=<id>, block=false)   # non-blocking peek
TaskOutput(task_id=<id>, block=true)    # wait for completion
```

## Preflight check

```bash
command -v kimi
```

## Headless mode (`kimi --print`)

Preferred for single-shot, well-scoped tasks. Runs without TUI, streams
incremental JSON events to stdout via `--output-format stream-json`.
The `--print` flag implicitly adds `--yolo` (auto-approve all actions).

### Recommended sub-agent recipe

```bash
kimi --print \
  --output-format stream-json \
  -p "Fix the TypeError in src/auth/token.ts"
```

Launch this via Shell with `run_in_background=true`, then use `TaskOutput` to
monitor.

Key flags:

| Flag | Purpose |
|---|---|
| `--print` | Headless mode (non-interactive), implies `--yolo` |
| `-p` / `--prompt` | The task prompt |
| `--output-format stream-json` | Incremental JSON events for progress tracking |
| `--output-format text` | Plain text output (default for `--print`) |
| `--quiet` | Alias for `--print --output-format text --final-message-only` |
| `--final-message-only` | Only print the final assistant message |
| `--model <model>` | Override the LLM model |
| `-w` / `--work-dir <dir>` | Working directory for the agent |
| `--add-dir <dir>` | Grant access to additional directories |
| `--max-steps-per-turn <n>` | Limit maximum steps per turn |
| `-C` / `--continue` | Resume the most recent session |
| `-S` / `--session <id>` | Resume a specific session |
| `--no-thinking` | Disable thinking mode |

### Pipe input

```bash
cat error.log | kimi --print --output-format stream-json -p "Explain this error"
git diff HEAD~3 | kimi --print --output-format stream-json -p "Review this diff"
```

### Working directory control

```bash
cd /path/to/project && kimi --print --output-format stream-json -p "Fix lint errors"
# Or grant access to additional dirs:
kimi --print --output-format stream-json --add-dir ../shared-lib -p "Update imports"
```

### Parallel fan-out

Launch multiple background Shell calls — each becomes an independent background
task that you can monitor via `TaskOutput`:

```bash
# Each of these is a separate Shell(run_in_background=true) call:

kimi --print --output-format stream-json \
  -p "Fix the bug in src/auth/login.ts"

kimi --print --output-format stream-json \
  -p "Add input validation to src/api/users.ts"

kimi --print --output-format stream-json \
  -p "Write tests for src/utils/format.ts"
```

Then use `TaskList` to enumerate active tasks and `TaskOutput` to collect
results as they complete.

## Quick flag reference

| Flag | Description |
|---|---|
| `--print` | Headless mode (implies `--yolo`) |
| `-p` / `--prompt` | User prompt |
| `--quiet` | Headless + text output + final message only |
| `--output-format` | `text` or `stream-json` |
| `--final-message-only` | Only emit last assistant message |
| `--model` | LLM model override |
| `-w` / `--work-dir` | Working directory |
| `--add-dir` | Additional directory access |
| `-y` / `--yolo` | Auto-approve (already implied by `--print`) |
| `--max-steps-per-turn` | Step limit per turn |
| `--no-thinking` | Disable thinking mode |
| `-C` / `--continue` | Resume last session |
| `-S` / `--session` | Resume specific session |
