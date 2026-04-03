from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from kimi_cli.eventbus.types import ContentPart
from kimi_cli.utils.logging import logger


class _HistoryEntry(BaseModel):
    content: str


def _load_history_entries(history_file: Path) -> list[_HistoryEntry]:
    entries: list[_HistoryEntry] = []
    if not history_file.exists():
        return entries

    try:
        with history_file.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse user history line; skipping: {line}",
                        line=line,
                    )
                    continue
                try:
                    entry = _HistoryEntry.model_validate(record)
                    entries.append(entry)
                except ValidationError:
                    logger.warning(
                        "Failed to validate user history entry; skipping: {line}",
                        line=line,
                    )
                    continue
    except OSError as exc:
        logger.warning(
            "Failed to load user history file: {file} ({error})",
            file=history_file,
            error=exc,
        )

    return entries


class PromptMode(Enum):
    AGENT = "agent"
    SHELL = "shell"

    def toggle(self) -> PromptMode:
        return PromptMode.SHELL if self == PromptMode.AGENT else PromptMode.AGENT

    def __str__(self) -> str:
        return self.value


class UserInput(BaseModel):
    mode: PromptMode
    command: str
    """The plain text representation of the user input."""

    resolved_command: str = ""
    """The text command after UI-only placeholders are expanded.
    Defaults to `command` when not explicitly set."""

    def model_post_init(self, __context: Any) -> None:
        if not self.resolved_command:
            self.resolved_command = self.command

    content: list[ContentPart]
    """The rich content parts."""

    def __str__(self) -> str:
        return self.command

    def __bool__(self) -> bool:
        return bool(self.command)


@dataclass(frozen=True, slots=True)
class TurnSubmitResult:
    accepted: bool
    persist_history: bool = False
    feedback: str = ""

    @classmethod
    def accept(cls, *, persist_history: bool = False) -> TurnSubmitResult:
        return cls(accepted=True, persist_history=persist_history)

    @classmethod
    def reject(cls, feedback: str = "") -> TurnSubmitResult:
        return cls(accepted=False, feedback=feedback)


@dataclass(frozen=True, slots=True)
class InputBoxState:
    active: bool = False
    mode: Literal["default", "reminder", "approval", "question", "question_other"] = "default"
    hint: str = ""
