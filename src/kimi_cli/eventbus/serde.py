from __future__ import annotations

from typing import Any

from llmkit.utils.typing import JsonType

from kimi_cli.eventbus.types import BusMessage, BusMessageEnvelope


def serialize_bus_message(msg: BusMessage) -> dict[str, JsonType]:
    """
    Convert a `BusMessage` into a jsonifiable dict.
    """
    envelope = BusMessageEnvelope.from_wire_message(msg)
    return envelope.model_dump(mode="json")


def deserialize_bus_message(data: dict[str, JsonType] | Any) -> BusMessage:
    """
    Convert a jsonifiable dict into a `BusMessage`.

    Raises:
        ValueError: If the message type is unknown or the payload is invalid.
    """
    envelope = BusMessageEnvelope.model_validate(data)
    return envelope.to_wire_message()
