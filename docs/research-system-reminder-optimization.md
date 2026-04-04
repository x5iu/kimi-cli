# Research Report: Optimizing Agent Behavior via Mid-Turn `<system-reminder>` Injection

**Date**: 2026-04-04
**Author**: Kimi Code CLI Deep Research
**Scope**: Analysis of local Kimi CLI trajectories, logs, and codebase to identify optimization opportunities for mid-turn `<system-reminder>` usage.

---

## 1. Executive Summary

This report analyzes the current `<system-reminder>` mechanism in Kimi CLI through examination of **2,930 sessions**, **17 log files**, and the complete codebase. The key finding is that the AttachmentProvider framework — designed for dynamic mid-turn prompt injection — is **severely underutilized**: only 1 provider exists out of a rich design capable of supporting many. We identify **5 high-impact optimization opportunities** that can be implemented through new AttachmentProviders and refinements to existing injection logic, with minimal code intrusion.

---

## 2. Current Architecture

### 2.1 Three-Tier Tag System (defined in `src/kimi_cli/agents/default/system.md:17-21`)

| Tag | Semantics | Agent Behavior |
|---|---|---|
| `<system>` | Supplementary context | Consider, non-binding |
| `<system-reminder>` | Authoritative system directive | **MUST follow**, may override normal behavior |
| `<system-hint>` | Non-binding suggestion | Consider but can be overridden by user |

### 2.2 Injection Points in the Agent Loop

The agent loop (`src/kimi_cli/loop/kimi_agent_loop.py`) has **4 distinct injection points** during a turn:

1. **Attachment Injection** (L1376-1387, every `_step()` call):
   - Collects from registered `AttachmentProvider` instances
   - Wraps content in `<system-hint>` or `<system-reminder>` based on `is_hint` flag
   - Appended as `internal_user_message` before the LLM call

2. **Notification Injection** (L1366-1374, every `_step()` call):
   - Delivers pending background task notifications
   - Uses `<notification>` tags (distinct from system-reminder)
   - Up to 4 notifications per step

3. **Steer Injection** (L385-402, L1340-1345, L1357-1358):
   - User sends mid-turn reminders via the shell UI
   - Queued via `steer()`, consumed between steps
   - Uses `<system-reminder>` tag with full instruction text (~100 tokens each)
   - If agent was about to stop (no_tool_calls), forces another LLM step

4. **Compaction Post-Injection** (L1563-1650):
   - After compaction: archive overview, todo state, active background tasks
   - Uses `<system>` tags (not `<system-reminder>`)

### 2.3 AttachmentProvider Framework

**Design** (`src/kimi_cli/loop/attachment.py`):
```python
@dataclass(frozen=True, slots=True)
class Attachment:
    type: str       # identifier
    content: str    # text content
    is_hint: bool = False  # True = <system-hint>, False = <system-reminder>

class AttachmentProvider(ABC):
    async def get_attachments(
        self, history: Sequence[Message], agent_loop: KimiAgentLoop,
    ) -> list[Attachment]: ...
```

**Current providers** (only 1):
- `PreferShellRgAttachmentProvider` (`src/kimi_cli/loop/attachments/prefer_shell_rg.py`):
  - Injects once per session when `rg` is available
  - Uses `is_hint=True` (system-hint)
  - Self-deduplicates by checking history for existing reminder

**Registration** (L223):
```python
self._attachment_providers: list[AttachmentProvider] = [PreferShellRgAttachmentProvider()]
```

---

## 3. Trajectory Data Analysis

### 3.1 Quantitative Findings (50 most recent sessions sampled)

| Metric | Value |
|---|---|
| Sessions with `<system-reminder>` | 40/50 (80%) |
| Sessions with `<system-hint>` | 9/50 (18%) |
| Total `<system-reminder>` in user messages | 52 |
| Total `<system-reminder>` in tool messages | 155 |
| Total `<system-reminder>` in assistant messages | 14 |
| Total `<system-hint>` across all roles | 85 |

### 3.2 Actual system-reminder Content in User Messages

| Content Pattern | Count | Source |
|---|---|---|
| `Prefer Shell with rg` | 19 | `PreferShellRgAttachmentProvider` |
| `The user sent a new reminder` | 5 | Steer mechanism |
| `Background tasks completed` | 4 | Shell UI idle trigger |

### 3.3 False Positives in Tool Messages (155 occurrences)

The vast majority of `<system-reminder>` in tool messages are **not actual injections** — they are source code strings read by the agent during file/grep operations. This means the agent sees `<system-reminder>` text inside tool results when reading its own codebase, which could potentially:
- Confuse the agent into treating code snippets as directives
- Create unintended behavioral shifts during code exploration tasks

### 3.4 Attachment Injection Frequency

The `PreferShellRgAttachmentProvider` injects **exactly once** per session (verified across 5 sessions):
- Session `d47e3768`: 1 hint, 8 assistant messages
- Session `e5828fee`: 1 hint, 60 assistant messages
- Session `2d66dc4b`: 1 hint, 75 assistant messages

This shows the attachment framework is called every step but only produces output once, meaning **there is zero ongoing mid-turn behavioral guidance** after the first step.

---

## 4. Identified Problems and Optimization Opportunities

### P1: No Goal-Tracking Reminder for Long Turns (HIGH IMPACT)

**Problem**: In long turns (30+ steps), the agent can drift from the original user request. The trajectory data shows sessions with 60-75 assistant messages in a single turn with no mid-turn reminders reinforcing the goal.

**Solution**: Create a `GoalTrackingAttachmentProvider` that:
- Activates after N steps (configurable, default 10)
- Injects the original user request as a `<system-reminder>` every M steps
- Uses concise framing: "Stay focused on: {original_request_summary}"
- Self-throttles to inject at most once every M steps

### P2: No Context Budget Awareness (MEDIUM IMPACT)

**Problem**: The agent has no visibility into how much context budget remains. It may continue verbose operations when the context is nearly full, triggering unexpected compaction.

**Solution**: Create a `ContextBudgetAttachmentProvider` that:
- Activates when context usage exceeds a threshold (e.g., 60%)
- Injects a `<system-hint>` with current usage percentage and guidance
- Escalates to `<system-reminder>` at higher thresholds (e.g., 80%)
- Suggests strategies: "Be concise", "Avoid reading large files", "Consider summarizing"

### P3: Steer Instruction Repetition Waste (MEDIUM IMPACT)

**Problem**: Every steer injection includes the full instruction text (~100 tokens):
```
"The user sent a new reminder during the current turn. 
Treat it as an additional user instruction for this task. 
Incorporate it into the ongoing turn, but do not stop, summarize, or conclude..."
```
In a turn with multiple steers, this identical instruction is repeated verbatim each time.

**Solution**: 
- First steer in a turn: include full instruction
- Subsequent steers: use abbreviated form: "Additional reminder (same handling rules as above):"
- Track steer count per turn in `_consume_pending_steers()`

### P4: Tool Result system-reminder Pollution (LOW-MEDIUM IMPACT)

**Problem**: When the agent reads source code containing `<system-reminder>` strings (155 occurrences in tool messages across 50 sessions), the model may interpret these as actual directives.

**Solution**: In `tool_result_to_message()` (`src/kimi_cli/loop/message.py`), sanitize tool output by escaping `<system-reminder>` and `<system-hint>` tags in tool results that come from file-reading or grep tools. Use the same approach as `sanitize_archive_text()` in `compaction_archive.py:130-134`:
```python
text.replace("<system-reminder>", "[system-reminder]")
text.replace("</system-reminder>", "[/system-reminder]")
```

### P5: Post-Compaction Continuity Reminder (MEDIUM IMPACT)

**Problem**: After compaction, the agent receives a structured summary and archive overview, but no explicit `<system-reminder>` directing it to continue the current task. The todo state is injected via `<system>` (informational), not `<system-reminder>` (authoritative).

**Solution**: Create a `PostCompactionContinuityAttachmentProvider` that:
- Detects when the first message is a compaction summary (starts with `<system>Previous context has been compacted`)
- On the first step after compaction, injects a `<system-reminder>`:
  "Context was compacted. Review the compaction summary and your todo list above carefully. Continue working on the current task without asking the user to repeat instructions."
- Fires only once after each compaction event

---

## 5. Implementation Plan

### Priority Order

1. **P1: GoalTrackingAttachmentProvider** — Highest impact, addresses the most common behavioral drift
2. **P4: Tool Result Sanitization** — Simple change, prevents subtle agent confusion
3. **P3: Steer Instruction Deduplication** — Reduces token waste in multi-steer turns
4. **P2: ContextBudgetAttachmentProvider** — Proactive context management
5. **P5: PostCompactionContinuityAttachmentProvider** — Better compaction recovery

### File Changes Required

| Change | Files |
|---|---|
| P1: GoalTrackingAttachmentProvider | `src/kimi_cli/loop/attachments/goal_tracking.py` (new), `src/kimi_cli/loop/kimi_agent_loop.py` (register) |
| P2: ContextBudgetAttachmentProvider | `src/kimi_cli/loop/attachments/context_budget.py` (new), `src/kimi_cli/loop/kimi_agent_loop.py` (register) |
| P3: Steer deduplication | `src/kimi_cli/loop/kimi_agent_loop.py` (modify `_build_steer_message`, `_consume_pending_steers`) |
| P4: Tool result sanitization | `src/kimi_cli/loop/message.py` (modify `tool_result_to_message`) |
| P5: PostCompactionContinuityAttachmentProvider | `src/kimi_cli/loop/attachments/post_compaction.py` (new), `src/kimi_cli/loop/kimi_agent_loop.py` (register) |

### Test Changes Required

| Change | Test Files |
|---|---|
| P1 | `tests/test_goal_tracking_attachment.py` (new) |
| P2 | `tests/test_context_budget_attachment.py` (new) |
| P3 | `tests/test_steer_deduplication.py` or modify existing steer tests |
| P4 | Modify existing tool result tests |
| P5 | `tests/test_post_compaction_attachment.py` (new) |

### Design Constraints

- All new AttachmentProviders MUST follow the existing pattern in `PreferShellRgAttachmentProvider`
- Each provider MUST handle its own throttling/deduplication
- Use `is_hint=True` (system-hint) for suggestions, `is_hint=False` (system-reminder) for directives
- Minimal changes to `kimi_agent_loop.py` — only register new providers
- Follow existing code style: line length 100, ruff rules (E, F, UP, B, SIM, I)

---

## 6. Expected Impact

| Optimization | Token Savings | Behavioral Improvement |
|---|---|---|
| P1: Goal Tracking | Slight increase (~20 tokens/10 steps) | Significant: prevents goal drift in long turns |
| P2: Context Budget | Slight increase (~15 tokens when triggered) | Medium: proactive context management |
| P3: Steer Dedup | ~80 tokens saved per additional steer | Medium: cleaner context for multi-steer turns |
| P4: Tool Sanitization | None | Medium: prevents false directive interpretation |
| P5: Post-Compaction | Slight increase (~30 tokens once) | Medium: smoother compaction recovery |

---

## 7. Raw Data References

- Sessions analyzed: `~/.kimi/sessions/af7f4ba88e1d2c18c278b399b92b9bf1/` (262 turns in most active session)
- Log files: `~/.kimi/logs/kimi.log` and 17 rotated logs
- High system-reminder sessions: `379520bc` (30), `4669bd14` (28), `5e07a464` (25)
- Codebase files analyzed: `kimi_agent_loop.py`, `attachment.py`, `prefer_shell_rg.py`, `message.py`, `compaction.py`, `compaction_archive.py`, `turns.py`, `notifications/llm.py`, `agents/default/system.md`
