from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import aiofiles
from pydantic import BaseModel, ConfigDict, ValidationError

from kimi_cli.eventbus.types import BusMessage, BusMessageEnvelope
from kimi_cli.utils.logging import logger

BUS_PROTOCOL_VERSION: str = "1.4"
BUS_PROTOCOL_LEGACY_VERSION: str = "1.1"


class EventLogMetadata(BaseModel):
    """Metadata header stored as the first line in wire.jsonl."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["metadata"] = "metadata"
    protocol_version: str


class BusMessageRecord(BaseModel):
    """The persisted record of a `BusMessage`."""

    model_config = ConfigDict(extra="ignore")

    timestamp: float
    message: BusMessageEnvelope

    @classmethod
    def from_wire_message(cls, msg: BusMessage, *, timestamp: float) -> BusMessageRecord:
        return cls(timestamp=timestamp, message=BusMessageEnvelope.from_wire_message(msg))

    def to_wire_message(self) -> BusMessage:
        return self.message.to_wire_message()


def parse_event_log_metadata(line: str) -> EventLogMetadata | None:
    """Parse a wire file metadata line; return None if the line is not metadata."""
    try:
        return EventLogMetadata.model_validate_json(line)
    except (ValidationError, ValueError):
        return None


def parse_event_log_line(line: str) -> EventLogMetadata | BusMessageRecord:
    """Parse a wire file line into metadata or a message record."""
    metadata = parse_event_log_metadata(line)
    if metadata is not None:
        return metadata
    return BusMessageRecord.model_validate_json(line)


@dataclass(slots=True)
class EventLog:
    path: Path
    protocol_version: str = BUS_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.path.exists():
            version = _load_protocol_version(self.path)
            self.protocol_version = version if version is not None else BUS_PROTOCOL_LEGACY_VERSION
        else:
            self.protocol_version = BUS_PROTOCOL_VERSION

    def __str__(self) -> str:
        return str(self.path)

    @property
    def version(self) -> str:
        return self.protocol_version

    def is_empty(self) -> bool:
        if not self.path.exists():
            return True
        try:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if parse_event_log_metadata(line) is not None:
                        continue
                    return False
        except OSError:
            logger.exception("Failed to read wire file {file}:", file=self.path)
            return False
        return True

    async def iter_records(self) -> AsyncIterator[BusMessageRecord]:
        if not self.path.exists():
            return
        try:
            async with aiofiles.open(self.path, encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = parse_event_log_line(line)
                    except Exception:
                        logger.exception(
                            "Failed to parse line in wire file {file}:", file=self.path
                        )
                        continue
                    if isinstance(parsed, EventLogMetadata):
                        continue
                    yield parsed
        except Exception:
            logger.exception("Failed to read wire file {file}:", file=self.path)

    async def append_message(self, msg: BusMessage, *, timestamp: float | None = None) -> None:
        record = BusMessageRecord.from_wire_message(
            msg,
            timestamp=time.time() if timestamp is None else timestamp,
        )
        await self.append_record(record)

    async def append_record(self, record: BusMessageRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        async with aiofiles.open(self.path, mode="a", encoding="utf-8") as f:
            if needs_header:
                metadata = EventLogMetadata(protocol_version=self.protocol_version)
                await f.write(_dump_line(metadata))
            await f.write(_dump_line(record))


def _dump_line(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False) + "\n"


def _load_protocol_version(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                metadata = parse_event_log_metadata(line)
                if metadata is None:
                    return None
                return metadata.protocol_version
    except OSError:
        logger.exception("Failed to read wire file {file}:", file=path)
    return None
