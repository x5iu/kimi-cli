from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.loop.compaction_archive import load_compaction_archives
from kimi_cli.utils.metrics import emit_metric
from kimi_cli.utils.turns import is_real_user_turn_start_message
from llmkit.message import Message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop

_RECALL_NUDGE_TYPE = "recall_nudge_after_compaction"

_COMPACTION_PREFIX = "<system>Previous context has been compacted"

_MIN_STEPS_AFTER_COMPACTION = 5

_COOLDOWN_MESSAGES = 15

_MAX_FIRES_PER_COMPACTION = 3

_EXPLORATION_TOOLS = frozenset(
    {
        "Shell",
        "ReadFile",
        "SearchWeb",
        "FetchURL",
        "WriteFile",
        "Edit",
    }
)

_REFERENTIAL_RE = re.compile(
    r"(之前|上次|刚才|刚刚|那个文件|那个错误|讨论过|较早|先前|"
    r"earlier|previous|before|as we discussed|like before|what did|what was|"
    r"the file we|that error)",
    re.I,
)


class RecallNudgeAfterCompactionProvider(AttachmentProvider):
    def __init__(
        self,
        *,
        min_steps: int = _MIN_STEPS_AFTER_COMPACTION,
        cooldown: int = _COOLDOWN_MESSAGES,
        max_fires: int = _MAX_FIRES_PER_COMPACTION,
    ) -> None:
        self._min_steps = min_steps
        self._cooldown = cooldown
        self._max_fires = max_fires

        self._last_seen_generation: int = 0
        self._fire_count: int = 0
        self._last_fired_at_step: int = 0

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        if not history:
            return []

        gen = agent_loop._compaction_generation  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        if gen != self._last_seen_generation:
            self._last_seen_generation = gen
            self._fire_count = 0
            self._last_fired_at_step = 0

        if gen == 0:
            return []
        if not _is_compaction_summary(history[0]):
            return []

        if self._fire_count >= self._max_fires:
            return []

        assistant_step_count = 0
        has_recall = False
        has_exploration = False
        past_summary = False

        for msg in history:
            if not past_summary:
                if _is_compaction_summary(msg):
                    past_summary = True
                    continue
                continue
            if msg.role != "assistant":
                continue
            assistant_step_count += 1
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    if name == "RecallCompactedContext":
                        has_recall = True
                    if name in _EXPLORATION_TOOLS:
                        has_exploration = True

        if has_recall:
            return []

        if assistant_step_count < self._min_steps:
            return []

        referential = _recent_user_referential(history)
        if not has_exploration and not referential:
            return []

        if (
            self._last_fired_at_step > 0
            and assistant_step_count < self._last_fired_at_step + self._cooldown
        ):
            return []

        self._fire_count += 1
        self._last_fired_at_step = assistant_step_count
        reason = "referential" if referential and not has_exploration else "exploration"
        emit_metric("recall.nudge_fired", reason=reason, step=assistant_step_count)

        kw_hint = _archive_keyword_hint(agent_loop)
        base = (
            "You have compacted archives available. "
            "If you're looking for information "
            "discussed earlier in this session, use "
            "RecallCompactedContext with targeted "
            "keywords instead of re-searching."
        )
        if kw_hint:
            base += f" Try keywords related to: {kw_hint}."
        return [
            Attachment(
                type=_RECALL_NUDGE_TYPE,
                content=base,
                is_hint=True,
            )
        ]


def _recent_user_referential(history: Sequence[Message]) -> bool:
    seen = False
    user_chunks: list[str] = []
    for msg in history:
        if not seen:
            if _is_compaction_summary(msg):
                seen = True
            continue
        if msg.role == "user" and is_real_user_turn_start_message(msg):
            user_chunks.append(msg.extract_text(" "))
    tail = "\n".join(user_chunks[-4:])
    return bool(_REFERENTIAL_RE.search(tail))


def _archive_keyword_hint(agent_loop: KimiAgentLoop) -> str:
    try:
        ctx_path = agent_loop._context.file_backend  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        recs = load_compaction_archives(ctx_path)
        acc: list[str] = []
        for r in recs[-2:]:
            acc.extend(r.keywords[:4])
        uniq = list(dict.fromkeys(acc))[:12]
        return ", ".join(uniq)
    except Exception:
        return ""


def _is_compaction_summary(msg: Message) -> bool:
    if msg.role not in ("user", "assistant"):
        return False
    text = msg.extract_text(" ").strip()
    return text.startswith(_COMPACTION_PREFIX)
