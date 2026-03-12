from __future__ import annotations

import json
from pathlib import Path

from kimi_cli.web.api.sessions import truncate_context_at_turn


def _write_context(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_truncate_context_ignores_internal_user_messages(tmp_path: Path) -> None:
    context_path = tmp_path / "context.jsonl"
    _write_context(
        context_path,
        [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "working"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "<system-reminder>\nlater\n</system-reminder>"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "<system>Reminder: suggested skill</system>"}],
            },
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "next"},
        ],
    )

    truncated = truncate_context_at_turn(context_path, 0)

    assert len(truncated) == 5
    assert json.loads(truncated[-1])["content"] == "done"
