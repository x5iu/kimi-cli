from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import aiofiles
import aiofiles.os
from llmkit.message import Message
from pydantic import ValidationError

from kimi_cli.loop.compaction import estimate_text_tokens
from kimi_cli.loop.message import internal_user_message, system
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import next_available_rotation


class Context:
    def __init__(self, file_backend: Path):
        self._file_backend = file_backend
        self._history: list[Message] = []
        self._token_count: int = 0
        self._pending_token_estimate: int = 0
        self._next_checkpoint_id: int = 0
        """The ID of the next checkpoint, starting from 0, incremented after each checkpoint."""
        self._last_turn_checkpoint_id: int | None = None
        """The checkpoint ID of the most recent turn-level checkpoint (created by _turn())."""

    async def restore(self) -> bool:
        logger.debug("Restoring context from file: {file_backend}", file_backend=self._file_backend)
        if self._history:
            logger.error("The context storage is already modified")
            raise RuntimeError("The context storage is already modified")
        if not self._file_backend.exists():
            logger.debug("No context file found, skipping restoration")
            return False
        if self._file_backend.stat().st_size == 0:
            logger.debug("Empty context file, skipping restoration")
            return False

        messages_after_last_usage: list[Message] = []
        async with aiofiles.open(
            self._file_backend, encoding="utf-8", errors="replace"
        ) as f:
            line_no = 0
            async for line in f:
                line_no += 1
                if not line.strip():
                    continue
                line_json = self._parse_context_line(
                    line,
                    file_backend=self._file_backend,
                    line_no=line_no,
                )
                if line_json is None:
                    continue
                self._apply_context_record(
                    line_json,
                    history=self._history,
                    messages_after_last_usage=messages_after_last_usage,
                    file_backend=self._file_backend,
                    line_no=line_no,
                )

        self._pending_token_estimate = estimate_text_tokens(messages_after_last_usage)
        return True

    @property
    def history(self) -> Sequence[Message]:
        return self._history

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def token_count_with_pending(self) -> int:
        return self._token_count + self._pending_token_estimate

    @property
    def n_checkpoints(self) -> int:
        return self._next_checkpoint_id

    @property
    def last_turn_checkpoint_id(self) -> int | None:
        return self._last_turn_checkpoint_id

    @property
    def file_backend(self) -> Path:
        return self._file_backend

    async def checkpoint(self, add_user_message: bool, *, turn_start: bool = False):
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        logger.debug(
            "Checkpointing, ID: {id}, turn_start: {turn_start}",
            id=checkpoint_id,
            turn_start=turn_start,
        )

        record: dict[str, object] = {"role": "_checkpoint", "id": checkpoint_id}
        if turn_start:
            record["turn"] = True
            self._last_turn_checkpoint_id = checkpoint_id
        async with aiofiles.open(self._file_backend, "a", encoding="utf-8") as f:
            await f.write(json.dumps(record) + "\n")
        if add_user_message:
            await self.append_message(
                internal_user_message([system(f"CHECKPOINT {checkpoint_id}")])
            )

    async def revert_to(self, checkpoint_id: int):
        """
        Revert the context to the specified checkpoint.
        After this, the specified checkpoint and all subsequent content will be
        removed from the context. File backend will be rotated.

        Args:
            checkpoint_id (int): The ID of the checkpoint to revert to. 0 is the first checkpoint.

        Raises:
            ValueError: When the checkpoint does not exist.
            RuntimeError: When no available rotation path is found.
        """

        logger.debug("Reverting checkpoint, ID: {id}", id=checkpoint_id)
        if checkpoint_id >= self._next_checkpoint_id:
            logger.error("Checkpoint {checkpoint_id} does not exist", checkpoint_id=checkpoint_id)
            raise ValueError(f"Checkpoint {checkpoint_id} does not exist")

        # rotate the context file
        rotated_file_path = await next_available_rotation(self._file_backend)
        if rotated_file_path is None:
            logger.error("No available rotation path found")
            raise RuntimeError("No available rotation path found")
        await aiofiles.os.replace(self._file_backend, rotated_file_path)
        logger.debug(
            "Rotated context file: {rotated_file_path}", rotated_file_path=rotated_file_path
        )

        # restore the context until the specified checkpoint
        self._history.clear()
        self._token_count = 0
        self._next_checkpoint_id = 0
        self._last_turn_checkpoint_id = None
        messages_after_last_usage: list[Message] = []
        async with (
            aiofiles.open(
                rotated_file_path, encoding="utf-8", errors="replace"
            ) as old_file,
            aiofiles.open(self._file_backend, "w", encoding="utf-8") as new_file,
        ):
            line_no = 0
            async for line in old_file:
                line_no += 1
                if not line.strip():
                    continue

                line_json = self._parse_context_line(
                    line,
                    file_backend=rotated_file_path,
                    line_no=line_no,
                )
                if line_json is None:
                    continue
                if (
                    line_json.get("role") == "_checkpoint"
                    and line_json.get("id") == checkpoint_id
                ):
                    break

                keep_line = self._apply_context_record(
                    line_json,
                    history=self._history,
                    messages_after_last_usage=messages_after_last_usage,
                    file_backend=rotated_file_path,
                    line_no=line_no,
                )
                if keep_line:
                    await new_file.write(line)

        self._pending_token_estimate = estimate_text_tokens(messages_after_last_usage)

    async def clear(self) -> Path:
        """
        Clear the context history.
        This is almost equivalent to revert_to(0), but without relying on the assumption
        that the first checkpoint exists.
        File backend will be rotated.

        Raises:
            RuntimeError: When no available rotation path is found.
        """

        logger.debug("Clearing context")

        # rotate the context file
        rotated_file_path = await next_available_rotation(self._file_backend)
        if rotated_file_path is None:
            logger.error("No available rotation path found")
            raise RuntimeError("No available rotation path found")
        await aiofiles.os.replace(self._file_backend, rotated_file_path)
        self._file_backend.touch()
        logger.debug(
            "Rotated context file: {rotated_file_path}", rotated_file_path=rotated_file_path
        )

        self._history.clear()
        self._token_count = 0
        self._pending_token_estimate = 0
        self._next_checkpoint_id = 0
        self._last_turn_checkpoint_id = None
        return rotated_file_path

    async def append_message(
        self,
        message: Message | Sequence[Message],
        *,
        message_metadata: Sequence[dict[str, object]] | None = None,
    ):
        logger.debug("Appending message(s) to context: {message}", message=message)
        messages = [message] if isinstance(message, Message) else message
        self._history.extend(messages)
        self._pending_token_estimate += estimate_text_tokens(messages)

        async with aiofiles.open(self._file_backend, "a", encoding="utf-8") as f:
            for idx, message in enumerate(messages):
                if message_metadata is not None and idx < len(message_metadata):
                    extras = message_metadata[idx]
                    if extras:
                        data = json.loads(message.model_dump_json(exclude_none=True))
                        data.update(extras)
                        await f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        continue
                await f.write(message.model_dump_json(exclude_none=True) + "\n")

    async def update_token_count(self, token_count: int):
        logger.debug("Updating token count in context: {token_count}", token_count=token_count)
        self._token_count = token_count
        self._pending_token_estimate = 0

        async with aiofiles.open(self._file_backend, "a", encoding="utf-8") as f:
            await f.write(json.dumps({"role": "_usage", "token_count": token_count}) + "\n")

    def _parse_context_line(
        self,
        line: str,
        *,
        file_backend: Path,
        line_no: int,
    ) -> dict[str, Any] | None:
        try:
            line_json = json.loads(line, strict=False)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Skipping malformed context line {line_no} in {file}: {error}",
                line_no=line_no,
                file=file_backend,
                error=exc,
            )
            return None
        if not isinstance(line_json, dict):
            logger.warning(
                "Skipping non-object context line {line_no} in {file}",
                line_no=line_no,
                file=file_backend,
            )
            return None
        return cast(dict[str, Any], line_json)

    def _apply_context_record(
        self,
        line_json: dict[str, Any],
        *,
        history: list[Message],
        messages_after_last_usage: list[Message],
        file_backend: Path,
        line_no: int,
    ) -> bool:
        role = line_json.get("role")
        if not isinstance(role, str):
            logger.warning(
                "Skipping context line {line_no} in {file}: missing or invalid role",
                line_no=line_no,
                file=file_backend,
            )
            return False
        if role == "_usage":
            token_count = line_json.get("token_count")
            if not isinstance(token_count, int):
                logger.warning(
                    "Skipping invalid usage line {line_no} in {file}",
                    line_no=line_no,
                    file=file_backend,
                )
                return False
            self._token_count = token_count
            messages_after_last_usage.clear()
            return True
        if role == "_checkpoint":
            checkpoint_id = line_json.get("id")
            if not isinstance(checkpoint_id, int):
                logger.warning(
                    "Skipping invalid checkpoint line {line_no} in {file}",
                    line_no=line_no,
                    file=file_backend,
                )
                return False
            self._next_checkpoint_id = checkpoint_id + 1
            if line_json.get("turn"):
                self._last_turn_checkpoint_id = checkpoint_id
            return True
        try:
            message = Message.model_validate(line_json)
        except ValidationError as exc:
            logger.warning(
                "Skipping invalid context message line {line_no} in {file}: {error}",
                line_no=line_no,
                file=file_backend,
                error=exc,
            )
            return False
        history.append(message)
        messages_after_last_usage.append(message)
        return True
