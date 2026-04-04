# Phase 3 Research Report: New System-Reminder Optimization Opportunities

**Date**: 2026-04-04
**Scope**: Quantitative + qualitative analysis of 50 most recent sessions in profile `af7f4ba88e1d2c18c278b399b92b9bf1`, plus codebase review of AttachmentProvider framework.

---

## 1. Data Summary

### 1.1 Corpus Overview

| Metric | Value |
|---|---|
| Sessions analyzed | 50 |
| Total turns | 336 |
| Total tool calls | 2,124 |
| Sessions with compaction | 22 (44%) |
| Avg tool calls per turn | 6.3 |
| Tool errors (is_error) | 0 (errors are in tool output text; see §1.4) |

### 1.2 Turn Length Distribution

| Category | Count | % |
|---|---|---|
| Short (<5 tool calls) | 204 | 60.7% |
| Medium (5–19) | 106 | 31.5% |
| Long (20–49) | 23 | 6.8% |
| Very Long (50+) | 3 | 0.9% |

The 3 very-long turns: session `0ff9e07c` (113 calls), `d9251661` (70 calls), `3042756e` (50 calls). All are dominated by Shell (55–70% of calls).

### 1.3 Tool Usage Distribution

| Tool | Calls | % | Notes |
|---|---|---|---|
| Shell | 1,186 | 55.8% | Dominant by far |
| ReadFile | 289 | 13.6% | 63% of "read" work done via Shell cat/head/tail instead |
| TaskOutput | 276 | 13.0% | Background task polling |
| SetTodoList | 125 | 5.9% | |
| Edit | 125 | 5.9% | |
| AskUserQuestion | 43 | 2.0% | |
| WriteFile | 43 | 2.0% | |
| RecallCompactedContext | **1** | 0.0% | Nearly unused despite 22 compaction sessions |

### 1.4 Key Anti-Pattern Metrics

| Pattern | Occurrences | Impact |
|---|---|---|
| **Shell storms** (5+ consecutive Shell) | 76 events, max streak 26 | Wastes steps exploring |
| **RG search storms** (5+ consecutive rg calls) | 8 sessions, max streak 10 | Unfocused searching |
| **Grep → full-file read** | 68 occurrences | ~99K wasted tokens |
| **Full-file reads >10K chars** | 57 (44.5% of all full reads) | Massive context waste |
| **Edit without verify** | 24/60 edit turns (40%) | Risk of shipping bugs |
| **Shell-for-reading** (cat/head/tail) | 497 vs 289 ReadFile | 63% of reads bypass ReadFile |
| **TaskOutput polling loops** (4+ polls same task) | 11 tasks across 7 sessions | Burns steps waiting |
| **Claude-in-Shell subagent calls** | 198 calls across 37 sessions | Each call is expensive |
| **Retry spirals** (same tool+args 3x) | 23 events, mostly claude subagent | Stuck in loop |
| **RecallCompactedContext unused** | 1 call across 50 sessions (22 had compaction) | Post-compaction info loss |

### 1.5 Compaction Observations

- 22/50 sessions (44%) hit compaction. These are the high-value, long sessions.
- RecallCompactedContext was used exactly **once** across all 50 sessions. The agent effectively never recalls compacted context, preferring to re-explore via Shell/ReadFile.
- Post-compaction, the agent typically continues without incident — P5 is working. But the agent never proactively recalls lost details.

---

## 2. Identified Opportunities

### P6: ToolStormBreaker — Interrupt Consecutive Same-Tool Sequences

**Problem**: The agent frequently enters "tool storms" — sequences of 5+ consecutive calls to the same tool (76 occurrences, max streak of 26 consecutive Shell calls). In session `c3ddccf4` Turn 3, the agent made 24 consecutive Shell calls doing rg searches. In session `990387f1` Turn 1, 26 consecutive Shell calls exploring git history.

**Trajectory evidence**:
- Session `c3ddccf4` Turn 3: 24 consecutive Shell (all rg searches across session directories)
- Session `d9251661` Turn 6: 16 consecutive Shell calls
- Session `990387f1` Turn 1: 16 consecutive Shell calls (git log/show exploration)
- Session `a4189fe1` Turn 0: 15 consecutive Shell calls
- Session `d9251661` Turn 4: 15+14 consecutive Shell calls in same turn

**Root cause**: The agent gets into a search/exploration loop without stepping back to synthesize findings. Each individual call is reasonable, but collectively they waste steps and context tokens.

**Proposed solution**: New `ToolStormBreakerAttachmentProvider`. Track consecutive calls to the same tool. After 8 consecutive calls to the same tool in a turn, inject a `<system-hint>`:

```
You've made {N} consecutive {tool_name} calls. Consider:
- Synthesize what you've learned so far before continuing
- Batch multiple operations into a single call (e.g., chain Shell commands with &&)
- If searching, narrow your query instead of broadening it
```

**Trigger**: `consecutive_same_tool >= 8` (checked per step, resets on tool change)
**Tag**: `<system-hint>` (non-binding suggestion)
**Cooldown**: Don't re-inject for another 8 calls after injection
**Expected impact**: Saves 3–5 tool calls per storm × 76 storms = ~300 saved tool calls across 50 sessions. Token cost: ~60 tokens per injection.

**Implementation**:
- New file: `src/kimi_cli/loop/attachments/tool_storm.py` (~60 lines)
- Register in `kimi_agent_loop.py` L232–237
- State tracking: count consecutive same-tool in history tail (no new agent_loop state needed)

---

### P7: GrepThenTargetedReadNudge — Prevent Full-File Reads After Search

**Problem**: After running `rg` or `grep`, the agent reads the **entire** file 29% of the time (68/232 grep→read sequences) instead of using `line_offset`+`n_lines` to read only the relevant section. This wastes ~99K tokens across 50 sessions.

**Trajectory evidence**:
- Session `e5828fee`: rg search → full ReadFile of `src/kimi_cli/tools/search.py` (entire file)
- Session `2d66dc4b`: rg search → full ReadFile of `src/kimi_cli/ui/shell/slash.py` (3 times!)
- Session `c3ddccf4`: rg search → full ReadFile of `src/kimi_cli/tools/todo/__init__.py` (twice)
- Full-file reads >10K chars: 57 occurrences (44.5% of all full reads), median 7,888 chars
- Largest: `src/kimi_cli/ui/shell/visualize.py` at 43,361 chars (~1000 lines)

**Root cause**: The grep output already shows line numbers, but the agent doesn't use them to construct targeted reads. The ReadFile tool description already mentions `line_offset` + `n_lines` but the agent ignores it after search results.

**Proposed solution**: New `GrepThenReadNudgeAttachmentProvider`. Detect when the most recent assistant step contained a grep/rg Shell call AND the current step is about to happen. Check if the previous tool result contained line-numbered output (pattern: `\d+:`). If so, inject:

```
Your recent search returned line-numbered results. When reading those files,
use ReadFile with line_offset and n_lines to read only the relevant sections
(±20 lines around matches) instead of the whole file.
```

**Trigger**: Previous step had Shell with `rg`/`grep`, tool result contains line numbers
**Tag**: `<system-hint>` (non-binding)
**Cooldown**: Once per turn (self-throttle)
**Expected impact**: Converts ~40% of the 68 full-file reads to targeted reads. Saves ~60K tokens across 50 sessions. Token cost: ~50 tokens per injection.

**Implementation**:
- New file: `src/kimi_cli/loop/attachments/grep_read_nudge.py` (~70 lines)
- Register in `kimi_agent_loop.py`
- Logic: scan last 2 assistant messages in history for Shell calls containing `rg ` or `grep `; check corresponding tool results for line-number patterns

---

### P8: TaskPollEscalation — Break TaskOutput Polling Loops

**Problem**: The agent repeatedly polls `TaskOutput(block=True)` on the same background task, escalating timeout from 10s → 120s → 180s → 300s → 600s, wasting an entire LLM step each time just to get "timeout / still running". Session `2d66dc4b` had **16 TaskOutput calls**, with task `bash-i7lamhql` polled 5 times in a row with 600s timeout each, and **every single one timed out**.

**Trajectory evidence**:
- Session `2d66dc4b`: 16 TaskOutput calls (all blocking), tasks `bash-gs8o2aqg` polled 5x, `bash-i7lamhql` polled 5x — all timed out
- Session `5e07a464`: 18 TaskOutput calls, task `bash-myov506t` polled 4x, `bash-2onlbpme` polled 4x — all timed out
- Session `e5828fee`: task `bash-y522e3pz` polled 5x, `bash-1tlg7llg` polled 4x
- 11 tasks across 7 sessions polled 4+ times

**Root cause**: The agent has no signal that polling is futile. Each poll consumes an LLM step (the LLM generates a tool call, waits for timeout, gets "still running", generates another poll). The agent escalates timeout but never gives up.

**Proposed solution**: New `TaskPollEscalationAttachmentProvider`. Track TaskOutput calls per task_id in the current turn. After the 3rd poll of the same task, inject:

```
You've polled task {task_id} {N} times and it's still running.
Stop polling — rely on the automatic completion notification instead.
Move on to other work or inform the user the task is long-running.
```

**Trigger**: 3+ TaskOutput calls to same task_id in current turn
**Tag**: `<system-reminder>` (authoritative — this is wasting real wall-clock time)
**Cooldown**: Once per task_id per turn
**Expected impact**: Saves 2–3 steps per affected task × 11 tasks = ~25 saved steps. More importantly, saves **wall-clock time** (each 600s blocking poll is 10 minutes of wait). Token cost: ~50 tokens per injection.

**Implementation**:
- New file: `src/kimi_cli/loop/attachments/task_poll.py` (~70 lines)
- Register in `kimi_agent_loop.py`
- State tracking: maintain `dict[str, int]` of task_id → poll count per turn, reset on turn change

---

### P9: EditVerificationReminder — Prompt Test/Lint After Multi-File Edits

**Problem**: In 40% of turns with edits (24/60), the agent modifies code files without running any tests, linting, or type-checking. This is especially risky in multi-file edits.

**Trajectory evidence**:
- Session `d67d3c4e`: Edited 4 files (`prompt1.txt` through `prompt3.txt`) without any verification
- Session `d67d3c4e`: Another turn edited 4 files (`w1.txt` through `w3.txt`) without verification
- Session `d67d3c4e`: Edited `bash.md` and `powershell.md` without verification
- Session `68761c85`: Edited 2 prompt files without verification
- Session `3042756e`: Edited 3 review prompt files without verification
- Overall: 60 turns had Edit/WriteFile calls; only 36 (60%) followed up with test/lint/check

**Root cause**: The agent focuses on completing edits and forgets to verify. This is especially common when the edit is the last action before the agent concludes ("premature conclusion").

**Proposed solution**: New `EditVerificationAttachmentProvider`. After detecting 2+ Edit/WriteFile calls to source code files (not /tmp/, not .md, not .txt) in the current turn WITHOUT any subsequent Shell call containing test/lint/check keywords, inject:

```
You've edited {N} source files in this turn without running tests or linting.
Consider running the project's test suite or linter to verify your changes
before concluding.
```

**Trigger**: ≥2 Edit/WriteFile to source files + no test/lint Shell call in the turn so far + agent is ≥5 steps into the turn
**Tag**: `<system-hint>` (non-binding — not all edits need tests, e.g., config files)
**Cooldown**: Once per turn
**Expected impact**: Converts some of the 24 unverified turns into verified ones. Token cost: ~40 tokens. Prevents bugs from reaching the user.

**Implementation**:
- New file: `src/kimi_cli/loop/attachments/edit_verify.py` (~80 lines)
- Register in `kimi_agent_loop.py`
- Logic: walk history from turn start, track Edit/WriteFile target files and Shell commands with test keywords

**Nuance**: Must exclude non-code files (/tmp/*, *.md, *.txt, *.json config) from triggering. Source files = .py, .go, .js, .ts, .rs, .java, .rb, .c, .cpp, .h, etc.

---

### P10: RecallNudgeAfterCompaction — Encourage RecallCompactedContext Usage

**Problem**: RecallCompactedContext was used exactly **1 time** across 50 sessions, despite 22 sessions having compaction events. The agent never proactively recalls lost details, instead re-exploring via Shell/ReadFile or making assumptions.

**Trajectory evidence**:
- 22 sessions had compaction, but only session `386d4255` ever called RecallCompactedContext
- Session `3042756e` post-compaction: user asked "你刚才说还有 Rust 实现，Rust 实现能删掉吗？" — referencing pre-compaction context. The agent could have used RecallCompactedContext but didn't.
- P5 (PostCompactionContinuity) already tells the agent to use RecallCompactedContext, but this fires only once right after compaction. The agent forgets about it within a few steps.

**Root cause**: The one-time P5 reminder fades from working memory as more steps execute. The agent never develops the habit of using RecallCompactedContext because it doesn't get reinforced.

**Proposed solution**: Enhance the existing PostCompactionContinuityAttachmentProvider or create a new `RecallNudgeAttachmentProvider`. After compaction, if the agent has gone 10+ steps without using RecallCompactedContext AND is doing exploration (Shell searches, ReadFile), inject:

```
Reminder: You have {N} compacted archives available. If you're looking
for information discussed earlier in this session, use
RecallCompactedContext with targeted keywords instead of re-searching.
```

**Trigger**: Post-compaction + ≥10 steps without RecallCompactedContext + agent is currently in search/exploration mode
**Tag**: `<system-hint>` (suggestion)
**Cooldown**: Once per 15 steps, max 2 times per session
**Expected impact**: Converts some of the re-exploration into targeted recall. Saves potentially 5–10 tool calls per affected session. Low token cost (~40 tokens).

**Implementation**:
- New file: `src/kimi_cli/loop/attachments/recall_nudge.py` (~60 lines)
- Register in `kimi_agent_loop.py`
- State: track steps since compaction and RecallCompactedContext usage via history scan

---

## 3. Priority Ranking

| Rank | ID | Name | Impact | Effort | Ratio |
|---|---|---|---|---|---|
| 1 | P8 | TaskPollEscalation | **High** (saves wall-clock time + steps) | Low (~70 lines) | ★★★★★ |
| 2 | P6 | ToolStormBreaker | **High** (76 storms, ~300 saved calls) | Low (~60 lines) | ★★★★★ |
| 3 | P7 | GrepThenTargetedRead | **Medium** (~99K saved tokens) | Low (~70 lines) | ★★★★ |
| 4 | P10 | RecallNudgeAfterCompaction | **Medium** (reduces re-exploration) | Low (~60 lines) | ★★★★ |
| 5 | P9 | EditVerificationReminder | **Medium** (quality improvement) | Medium (~80 lines) | ★★★ |

**Rationale**:
- **P8** is #1 because polling loops waste real wall-clock time (minutes of blocking), not just tokens. The fix is simple and the signal is unambiguous.
- **P6** is #2 because tool storms are the most frequent anti-pattern (76 occurrences) and the fix directly saves LLM steps.
- **P7** is #3 because it addresses a clear token waste pattern (99K tokens) with a simple detection heuristic.
- **P10** is #4 because RecallCompactedContext is nearly unused despite being available, and a periodic nudge could change this.
- **P9** is #5 because it's harder to get right (must exclude non-code files, tests may not exist) and the benefit is quality rather than efficiency.

---

## 4. Disqualified Ideas

### D1: SubagentRetryLimiter
**Idea**: Detect repeated `claude --bare -p` Shell calls and warn the agent to stop retrying subagent calls.
**Rejected because**: The retry pattern is caused by the subagent itself failing (timeouts, rate limits), not by the agent making bad decisions. The agent is correctly retrying. The right fix is at the subagent infrastructure level (better error handling, retry backoff), not via a system-reminder. A reminder would tell the agent to stop doing something the user explicitly asked for.

### D2: ShellForReadingDiscourager
**Idea**: Nudge agent to use ReadFile instead of `cat`/`head`/`tail` via Shell (63% of reads use Shell).
**Rejected because**: Shell-based reading is actually often more efficient — the agent chains `cat` with pipes, reads specific sections with `sed -n`, and combines with other commands. Forcing ReadFile would break these patterns. The issue is not Shell vs ReadFile but whether the read is targeted. P7 (GrepThenTargetedRead) already addresses the real problem.

### D3: TodoInflationWarning
**Idea**: Warn when SetTodoList is called with growing item counts.
**Rejected because**: Only 1 session (d9010128) showed todo inflation (14→14→14→4→3 items). The pattern is too rare to justify a new provider. The existing P1 GoalTracking already helps with scope creep.

### D4: AskUserQuestionThrottle
**Idea**: Warn when agent asks too many questions in a turn.
**Rejected because**: Only 1 session (5e07a464, 20 questions) showed this. That session was an intentional multi-step confirmation workflow. The pattern is too rare and context-dependent.

### D5: LargeFileReadWarning
**Idea**: Warn when ReadFile reads >10K chars.
**Rejected because**: This overlaps with P2 (ContextBudget) which already warns about context usage. Adding a per-read warning would be noisy. P7 addresses the more actionable version of this (reads after grep).

### D6: ScopeCreepDetector
**Idea**: Detect when agent starts editing files unrelated to the original request.
**Rejected because**: Defining "unrelated" requires understanding the semantic relationship between files and the user request, which is beyond what a simple pattern detector can do. P1 (GoalTracking) already addresses this more robustly.

### D7: CompactionTodoSync
**Idea**: After compaction, re-inject current todo state.
**Rejected because**: Already implemented — `compact_context()` at line 1603-1616 of `kimi_agent_loop.py` already injects todos after compaction.

---

## 5. Implementation Notes

### Available Signals in AttachmentProvider

From `get_attachments(history, agent_loop)`, providers can access:

| Signal | Access Path | Used By |
|---|---|---|
| Full message history | `history` parameter | P1, P5, P6, P7, P8, P9 |
| Context usage ratio | `agent_loop._context_usage` | P2 |
| Compaction generation | `agent_loop._compaction_generation` | P2, P5, P10 |
| Active turn ID | `agent_loop._active_turn_id` | — |
| Turn steer count | `agent_loop._turn_steer_count` | — |
| Runtime config | `agent_loop.runtime` | — |
| Toolset | `agent_loop.agent.toolset` | — |

All 5 proposed providers can be implemented using only `history` scanning + `_compaction_generation`. No new agent_loop state is required (providers can maintain their own state).

### History Scanning Patterns

For P6/P7/P8/P9, the key is efficiently scanning the tail of history to detect patterns. The approach:

1. Walk backward from end of `history`
2. Stop at the first real user turn-start message (this marks the current turn boundary)
3. Within the current turn, count/classify assistant tool calls and tool results
4. Apply the pattern-specific logic

This is the same pattern used by P1 (GoalTracking) and is efficient because it only scans the current turn, not the full history.

### Token Budget

All 5 providers inject ≤60 tokens each. At worst case, if all 5 fire in the same step, that's 300 tokens — well within the 1–2K typical attachment budget. In practice, co-firing is extremely unlikely because:
- P6 fires only during storms
- P7 fires only after grep
- P8 fires only during TaskOutput loops
- P9 fires only when edits lack verification
- P10 fires only post-compaction during exploration

---

## Appendix: Session Evidence Index

| Session ID | Key Pattern | Relevant To |
|---|---|---|
| `0ff9e07c` | 113-call turn, 77 Shell (rg storms + claude subagent) | P6, P7 |
| `d9251661` | 70-call turn, 53 Shell + 28-streak git exploration | P6 |
| `c3ddccf4` | 24-streak Shell rg storm | P6 |
| `990387f1` | 26-streak Shell git exploration | P6 |
| `2d66dc4b` | 16 TaskOutput calls, 5x poll of same task (all timeout) | P8 |
| `5e07a464` | 18 TaskOutput calls, 4x poll of 2 tasks (all timeout) | P8 |
| `e5828fee` | 5x poll + 4x poll of background tasks | P8 |
| `3042756e` | 28 full-file reads (0 targeted), 50-call turn | P7 |
| `386d4255` | Only session to use RecallCompactedContext (1 time) | P10 |
| `d67d3c4e` | 4-file edit without verify (twice in same session) | P9 |
| `efa15268` | 43K-char full-file read of visualize.py | P7 |
| `d9010128` | 28 claude-in-Shell calls, 7 retry spirals | (D1 — rejected) |
