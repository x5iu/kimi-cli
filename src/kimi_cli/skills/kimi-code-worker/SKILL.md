---
name: kimi-code-worker
description: Spawn Kimi Code CLI (`kimi`) as a sub-agent via shell commands to work on tasks. Supports persistent interactive sessions via stdin/stdout streaming (preferred) and headless one-shot mode. Use when a task can be delegated — e.g. parallel fixes, code review, migrations, analysis, or any independent subtask that benefits from a separate agent with its own context window. **When the user asks to "discuss with Kimi", "talk to Kimi", or any phrasing that implies back-and-forth dialogue, always use the interactive multi-turn session (Mode 1).**
---

# kimi-code-worker

Orchestrate Kimi Code CLI (`kimi`) as sub-agent workers. Two modes
available — interactive stdin session (preferred) and headless one-shot.

## Mandatory rules

1. **Must use Shell tool.** Always invoke `kimi` via the Shell tool — never
   ask the user to run it manually.
2. **Must run in background.** Always set `run_in_background=true` on the Shell
   call. This keeps the main conversation responsive while the worker runs.
   Use `TaskOutput` to poll progress or wait for completion.
3. **Must use `--output-format stream-json`.** This flag serialises output as
   per-step NDJSON `Message` records to stdout. Output is **step-buffered**:
   each step's assistant message (and any tool results) is flushed as a
   complete JSON line when the step finishes — you do not get token-by-token
   streaming within a step. When you inspect the background task via
   `TaskOutput`, you can see which steps have completed so far.
4. **Must use `--input-format stream-json` for multi-turn sessions.** This flag
   is what enables Kimi to accept follow-up messages via stdin. Without it,
   there are two failure modes: (a) with `-p`, the process runs the single
   prompt and exits — `TaskWrite` messages are ignored; (b) without `-p`,
   text mode does a blocking `sys.stdin.read()` to EOF, so the process hangs
   indefinitely instead of processing line-delimited turns. **Every
   interactive session (`interactive=true`) must include this flag.** It is
   not needed for headless one-shot mode (Mode 2).
5. **Scope by directory.** When a task targets a specific directory, `cd` into
   it before launching `kimi`. This constrains the worker's scope to that
   directory.
6. **Must set timeout ≥ 3600; use higher for interactive sessions.** The default
   60 s timeout will kill the worker prematurely. For headless one-shot tasks
   (Mode 2), `timeout=3600` (1 hour) is usually sufficient. For **interactive
   multi-turn sessions (Mode 1), set `timeout=7200` or higher** — these
   sessions can span many turns over an extended period; if the timeout fires
   mid-conversation, all accumulated context is lost.
7. **"Discuss with Kimi" = interactive mode.** When the user asks to
   "discuss", "talk to", "consult", "brainstorm with", "ask Kimi about",
   or any phrasing that implies back-and-forth dialogue with Kimi, you
   **must** launch an interactive multi-turn session (Mode 1). Do NOT use
   headless one-shot mode for discussions — a single `kimi --print -p "prompt"`
   cannot receive follow-up messages. The whole point of a discussion is
   multiple exchanges, which requires `--input-format stream-json` +
   `interactive=true` so you can keep sending `TaskWrite` messages.
   Only use headless one-shot (Mode 2) when the task is a clear, self-
   contained instruction that needs no back-and-forth.

## Preflight check

```bash
command -v kimi
```


## Mode 1: Interactive session (preferred)

A **persistent Kimi subagent** that stays alive for the entire duration of a
complex task. Think of it as pairing with a senior engineer who has their own
editor open — you discuss the problem, they investigate and code, you review
their work, give feedback, and iterate until the task is done. The subagent
retains full conversation context across every turn, so you never need to
re-explain background or decisions.

Communicate via NDJSON on stdin/stdout: send user messages with `TaskWrite`,
read responses with `TaskOutput`. The process keeps running until you
explicitly stop it or it times out.

### Why keep it open

- **Context accumulates.** Every turn builds on the last. The subagent
  remembers what it already tried, what files it changed, and what you said.
  Re-spawning loses all of this.
- **Cheaper and faster.** A persistent session means cache hits on every turn.
- **Mid-flight steering.** You can jump in at any point — correct a wrong
  approach, add a forgotten requirement, or ask the subagent to explain its
  reasoning before it continues.

### When to use

- **Complex multi-step tasks**: bug investigation → root cause → fix → test →
  verify. Each step may need your guidance.
- **Exploratory work**: "look at this module and suggest how to refactor it" →
  discuss the proposal → "ok, do option B" → review the result.
- **Iterative code review**: send a diff → get feedback → fix → re-review,
  all within one session.
- **Debugging sessions**: share an error → subagent investigates → reports
  findings → you provide more context → subagent digs deeper.
- **Architecture discussions**: describe a design goal → subagent analyzes the
  codebase → proposes approaches → you pick one → subagent implements.
- **Any task where you'd normally go back and forth** with a human colleague.

### Step 1: Start the session

```
Shell(
  command='cd <target-dir> && kimi --print --output-format stream-json --input-format stream-json',
  run_in_background=true,
  interactive=true,
  timeout=7200,
  description="kimi: <short task description>"
)
```

Key differences from headless mode:
- `interactive=true` — keeps stdin pipe open for `TaskWrite`
- `--input-format stream-json` — accept NDJSON user messages on stdin
- **Usually omit `-p`** — prompts are sent via `TaskWrite` instead. You
  *can* pass `-p "initial prompt"` to seed the first turn; after that turn
  completes, the process falls back to reading follow-up messages from stdin.

### Step 2: Confirm ready, then send the first prompt

After Shell returns the task ID, the worker needs a moment to start. Confirm
the task is running before sending input:

```
TaskOutput(task_id="<id>", block=true, timeout=10)
# Check the metadata field "status: running" in the TaskOutput response.
# Do NOT expect any stdout content yet — in --input-format stream-json mode,
# Kimi produces NO output until it receives the first TaskWrite message.
```

Then send the first user message:

```
TaskWrite(
  task_id="<id>",
  input='{"role":"user","content":"Fix the TypeError in src/auth/token.ts"}'
)
```

The NDJSON user message format:

```json
{"role":"user","content":"<your prompt>"}
```

Each message **must** be a single line of valid JSON. `TaskWrite` appends a
newline automatically (`append_newline=true` by default). Max input size is
1 MiB.

**`input` must be a plain string, NOT a dict/object.** The JSON message is
passed as a string value to the `input` parameter. Do NOT pass a Python dict
or JSON object — always pass a quoted JSON string:

```python
# CORRECT — input is a string containing JSON
TaskWrite(task_id="<id>", input='{"role":"user","content":"Fix the bug"}')

# WRONG — input is a dict, not a string
TaskWrite(task_id="<id>", input={"role": "user", "content": "Fix the bug"})
```

### Step 3: Monitor progress

```
TaskOutput(task_id="<id>", block=false)
```

**Important `TaskOutput` defaults:** `block` defaults to `true` and `timeout`
defaults to 30s. For interactive sessions, always pass `block=false` for
non-blocking polling, or `block=true, timeout=120` (or higher) when you want
to wait for Kimi to finish a turn.

The response is a **Kimi structured output** containing metadata fields
(`retrieval_status`, `output_path`, `output_next_offset`, etc.) with the
Kimi NDJSON output embedded in the `[output]` section. The output consists
of per-step `Message` JSON objects (flushed after each step completes, not
token-by-token):

| Message `role` | Meaning |
|---|---|
| `assistant` (with `tool_calls`) | Model invoked tools (step complete) |
| `tool` | Tool result returned to Kimi |
| `assistant` (without `tool_calls`) | **Turn complete.** Final assistant response for this turn |

**A turn is complete when the last output line is an `assistant` message
without `tool_calls`.** After that, the worker blocks on stdin waiting for
the next message.

**Note:** `TaskOutput` defaults to a **tail** preview (~32 KiB). Use
`offset=0` to read from the beginning, or pass the `output_next_offset`
from a previous call to page forward. `output_next_offset` is only present
when there is more output beyond the current preview.

### Step 4: Send follow-up messages

After the turn completes, send the next user message:

```
TaskWrite(
  task_id="<id>",
  input='{"role":"user","content":"Good. Now add tests for the fix."}'
)
```

Then repeat Step 3 to monitor. Use the `output_next_offset` value from the
previous `TaskOutput` response as `offset` to read only new lines:

```
TaskOutput(task_id="<id>", block=false, offset=<output_next_offset from previous call>)
```

`offset` is a 0-based **line number**, not a byte offset.

### Step 5: Keep or end the session

**Keep the session open when:**

- The task has more phases ahead (e.g. you just finished the fix, tests are
  next). Staying in the same session preserves all context.
- You might need to come back with follow-up questions or adjustments based on
  results you haven't reviewed yet.
- The subagent has built up valuable context (read many files, understands the
  architecture). Re-spawning would lose all of this and require a cold start.
- You have a cluster of related small tasks that share the same codebase
  context — cheaper to reuse one session than spawn many.

**End the session when:**

- The task is fully complete and no follow-up is expected.
- You need to switch to a completely different project or directory where the
  current context is irrelevant. Start a fresh session instead.
- The conversation has grown very long (many tool calls, large outputs). Kimi
  will auto-compact, but a fresh session may be more efficient at that point.
- You are running low on background task slots (max 4 concurrent). End idle
  sessions to free slots for new work.
- The subagent is stuck or in an error state that repeated messages can't fix.

**How to end:**

```
TaskStop(task_id="<id>", reason="Task complete")
```

**If you don't explicitly end it**, the session will be killed when `timeout`
expires (default 3600s). This is fine for natural completion, but wastes a
background task slot if the work finished hours ago.

**Rule of thumb:** End the session when a logical unit of work is done. If in
doubt, keep it open — an idle session costs only a slot, but re-spawning costs
all the context.

### Multi-turn collaboration patterns

The real power of interactive mode is **conversation**, not just sequential
commands. Here are proven patterns for working with a persistent subagent:

**Pattern 1: Investigate → Discuss → Act**

Don't jump to "fix it". Start by asking the subagent to investigate, review
its findings, then decide together what to do.

```
You  → "There's a memory leak in the worker pool. Investigate — check for
        unclosed connections, goroutine leaks, and circular references.
        Report what you find before making any changes."
Kimi → (reads code, runs analysis, reports 3 potential causes)
You  → "Cause #2 looks most likely. Can you write a minimal repro test
        to confirm before we fix it?"
Kimi → (writes test, confirms the hypothesis)
You  → "Good. Fix it, and make sure the repro test passes."
Kimi → (implements fix, runs test)
```

**Pattern 2: Incremental implementation with checkpoints**

Break a large task into phases. Review each phase before proceeding.

```
You  → "We need to add Redis caching to the API layer. Let's do this in
        phases. Phase 1: add the Redis client and config. Show me what
        you plan to change before writing code."
Kimi → (proposes file changes)
You  → "Looks good, but use the existing config module instead of a new
        one. Go ahead with phase 1."
Kimi → (implements)
You  → "Phase 2: cache the /users endpoint. Use a 5-minute TTL."
Kimi → (implements)
You  → "The cache invalidation on PUT /users is missing. Add that."
Kimi → (fixes)
You  → "Phase 3: add cache metrics. Then we're done."
```

**Pattern 3: Pair debugging**

Share observations back and forth, like debugging with a colleague.

```
You  → "Tests pass locally but fail in CI. Here's the CI log: ...
        What do you think?"
Kimi → (analyzes log, suspects timezone issue)
You  → "Interesting. The CI server is in UTC. Can you check if any tests
        assume local timezone?"
Kimi → (finds 2 tests with hardcoded timezone assumptions)
You  → "Fix those. Also check if there are similar issues elsewhere."
Kimi → (fixes and audits, finds 1 more)
```

**Pattern 4: Design-first development**

Use the subagent as an architecture consultant before it writes code.

```
You  → "I want to add webhook support. Before coding, analyze the
        existing event system and propose 2-3 design options."
Kimi → (reads codebase, proposes 3 approaches with trade-offs)
You  → "Option B fits best, but I want to use the existing queue
        instead of adding a new dependency. Revise and implement."
Kimi → (implements revised design)
You  → "Review your own implementation for edge cases."
Kimi → (self-reviews, finds 2 issues, fixes them)
```

**Key principle:** Treat the subagent as a thinking partner, not a command
executor. The more context you share about *why* you want something, the
better the results. The persistent session means every exchange enriches
the subagent's understanding of your codebase and intent.

### Complete example: fix → test → review cycle

```python
# 1. Start interactive kimi session
Shell(
  command='cd /project && kimi --print --output-format stream-json --input-format stream-json',
  run_in_background=true,
  interactive=true,
  timeout=7200,
  description="kimi: fix auth bug and add tests"
)
# → returns task_id (e.g. "bash-abc123") plus status/next_steps metadata

# 2. Confirm worker is running
TaskOutput(task_id="bash-abc123", block=true, timeout=10)
# → look for "status: running" in output

# 3. Send initial task
TaskWrite(task_id="bash-abc123", input='{"role":"user","content":"Fix the TypeError in src/auth/token.ts. The error is: Cannot read property expiry of undefined."}')

# 4. Poll output (use output_next_offset from each response for the next call)
TaskOutput(task_id="bash-abc123", block=false)
# → check [output] section for assistant message without tool_calls to know turn is done
# → note output_next_offset for next read

# 5. After seeing turn complete, send follow-up
TaskWrite(task_id="bash-abc123", input='{"role":"user","content":"Now write unit tests for the fix you just made."}')

# 6. Check new output only (pass offset from step 4)
TaskOutput(task_id="bash-abc123", block=false, offset=<output_next_offset>)

# 7. Provide feedback on results
TaskWrite(task_id="bash-abc123", input='{"role":"user","content":"The test for expired tokens is missing an edge case: what if expiry is 0? Add that."}')

# 8. Final check
TaskOutput(task_id="bash-abc123", block=false, offset=<output_next_offset>)

# 9. Done — stop the session (requires user approval)
TaskStop(task_id="bash-abc123", reason="Task complete")
```

### Sending context with the prompt

You can include file contents, diffs, or error logs inline in the message text:

```
TaskWrite(
  task_id="<id>",
  input='{"role":"user","content":"Here is the error log:\\n\\nTypeError: Cannot read property...\\n\\nFix it."}'
)
```

For very large context, let Kimi read the files itself — just reference paths.

### Parallel interactive sessions

Launch multiple interactive sessions, each with its own task ID:

```python
# Session A: fix auth
Shell(command='cd /project && kimi --print --output-format stream-json --input-format stream-json', run_in_background=true, interactive=true, timeout=7200, description="kimi: fix auth")
# → bash-aaa

# Session B: fix API
Shell(command='cd /project && kimi --print --output-format stream-json --input-format stream-json', run_in_background=true, interactive=true, timeout=7200, description="kimi: fix api")
# → bash-bbb

# Send prompts to each
TaskWrite(task_id="bash-aaa", input='{"role":"user","content":"Fix auth bug"}')
TaskWrite(task_id="bash-bbb", input='{"role":"user","content":"Fix API validation"}')

# Monitor both
TaskOutput(task_id="bash-aaa", block=false)
TaskOutput(task_id="bash-bbb", block=false)
```

## Mode 2: Headless one-shot (`kimi --print -p "prompt"`)

For simple, fire-and-forget tasks where no follow-up is needed. The process
runs a single prompt and exits.

```
Shell(
  command='cd <target-dir> && kimi --print --output-format stream-json -p "Fix the lint errors"',
  run_in_background=true,
  timeout=3600,
  description="kimi: fix lint errors"
)
```

Then check progress / collect results:

```
TaskOutput(task_id=<id>, block=false)   # non-blocking peek
TaskOutput(task_id=<id>, block=true)    # wait for completion
```

### When to use

- Simple, well-scoped tasks that don't need guidance
- Parallel fan-out of independent subtasks
- Tasks where you just need the end result

### Pipe input

When `-p` is set, piped stdin data is **ignored** — the prompt argument is
used directly. To include piped content, either omit `-p` (stdin becomes
the prompt) or embed it in the prompt string:

```bash
# Option A: stdin IS the prompt (no -p)
echo "Explain this error: $(cat error.log)" | kimi --print --output-format stream-json

# Option B: embed in -p via command substitution
kimi --print --output-format stream-json -p "Review this diff: $(git diff HEAD~3)"
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
| `-p` / `--prompt` | User prompt; seeds first turn (can be used in both modes) |
| `--output-format stream-json` | **Mandatory.** Per-step NDJSON Message records on stdout |
| `--input-format stream-json` | Accept NDJSON user messages on stdin (Mode 1) |
| `--quiet` | Headless + text output + final message only (**incompatible with NDJSON monitoring**) |
| `--final-message-only` | Only emit last assistant message |
| `--model` | LLM model override |
| `-w` / `--work-dir` | Working directory |
| `--add-dir` | Additional directory access |
| `-y` / `--yolo` | Auto-approve (already implied by `--print`) |
| `--max-steps-per-turn` | Step limit per turn |
| `--no-thinking` | Disable thinking mode |
| `-C` / `--continue` | Resume last session |
| `-S` / `--session` | Resume specific session |

## NDJSON message reference

### User message (stdin → kimi)

```json
{"role":"user","content":"Your prompt here"}
```

The content field also accepts the array form:

```json
{"role":"user","content":[{"type":"text","text":"Your prompt here"}]}
```

**Parser behavior:** `_read_next_command()` reads stdin line-by-line. Each
line must be valid JSON that deserializes to a `Message` with `role=="user"`.
Blank lines, invalid JSON, and non-user-role messages are silently ignored.
Only `TextPart` content is used — non-text parts are discarded. Multiple
text parts are joined with `\n`. EOF closes the session gracefully.

### Key stdout events (kimi → stdout)

**Assistant response (with tool calls):**
```json
{"role":"assistant","content":"Let me check the code...","tool_calls":[{"id":"...","type":"function","function":{"name":"Shell","arguments":"{...}"}}]}
```

**Tool result:**
```json
{"role":"tool","content":"...","tool_call_id":"..."}
```

**Turn complete (assistant response without tool calls):**
```json
{"role":"assistant","content":"I've fixed the TypeError by adding a null check..."}
```

### Turn detection pattern

After each `TaskWrite`, poll `TaskOutput` and scan for the last JSON line.
When the last line is an `assistant` message **without** `tool_calls`, the
turn is complete and you can:
- Send the next `TaskWrite` for follow-up
- Or `TaskStop` to end the session

**Edge cases:** This detection works on the happy path. If the turn is
interrupted (e.g. SIGINT, max steps reached, or an LLM error), the final
assistant message may be absent or truncated. Also note that
`--final-message-only` changes the output shape (only the last assistant
text is emitted), so the detection pattern above assumes the default
`--output-format stream-json` without `--final-message-only`.


## Caveats and limitations

- **TaskStop requires approval**: Stopping a task triggers a user approval
  prompt. Plan for this in automated workflows.
- **stdin delivery latency**: `TaskWrite` queues input as a file; the worker
  relay polls every ~200ms. Input is not delivered instantly.
- **TaskWrite input limit**: Max 1 MiB per `TaskWrite` call. For larger
  context, let Kimi read files directly instead of inlining content.
- **stdin_ready race**: After `Shell` returns the task ID, the worker process
  needs a moment to initialize. Always confirm `status: running` via
  `TaskOutput` before the first `TaskWrite`.
- **`--quiet` is incompatible with NDJSON monitoring**: `--quiet` forces
  `--output-format text --final-message-only`, so you lose structured
  progress tracking. Do not use it with the workflows described here.
- **Step-buffered output**: Unlike token-level streaming, `stream-json`
  output is flushed per step. Long-running tool calls (e.g. a slow shell
  command) produce no output until the step finishes.
- **Pipe input with `-p`**: When `-p` is set, piped stdin data is ignored.
  Either omit `-p` to use stdin as the prompt, or use command substitution
  to embed piped content in the `-p` string.
