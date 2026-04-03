import asyncio

from kaos.path import KaosPath
from rich import print

from kimi_cli.app import KimiCLI, enable_logging
from kimi_cli.session import Session


async def main():
    enable_logging()
    session = await Session.create(KaosPath.cwd())
    instance = await KimiCLI.create(session)
    user_input = "Hello!"

    async for msg in instance.run(
        user_input=user_input,
        cancel_event=asyncio.Event(),
        merge_bus_messages=True,
    ):
        print(msg)

    # print the last assistant message
    print(instance.agent_loop.context.history[-1])


if __name__ == "__main__":
    asyncio.run(main())
