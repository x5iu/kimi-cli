import asyncio
from typing import Any

from kimi_cli.eventbus.types import ContentPart, StepBegin, TextPart
from kimi_cli.llm import ALL_MODEL_CAPABILITIES, ModelCapability
from kimi_cli.loop import StatusSnapshot, bus_send
from kimi_cli.ui.shell import Shell
from kimi_cli.utils.slashcmd import SlashCommand


class EchoAgentLoop:
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "EchoAgentLoop"

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def model_capabilities(self) -> set[ModelCapability]:
        return ALL_MODEL_CAPABILITIES

    @property
    def status(self) -> StatusSnapshot:
        return StatusSnapshot(context_usage=0.0)

    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        return []

    async def run(self, user_input: str | list[ContentPart]) -> None:
        bus_send(StepBegin(n=1))
        if isinstance(user_input, str):
            bus_send(TextPart(text=user_input))
        else:
            for part in user_input:
                bus_send(part)


if __name__ == "__main__":
    soul = EchoAgentLoop()
    ui = Shell(soul)
    asyncio.run(ui.run())
