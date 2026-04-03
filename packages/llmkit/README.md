# Kosong

Kosong is an LLM abstraction layer designed for modern AI agent applications. It unifies message structures, asynchronous tool orchestration, and pluggable chat providers so you can build agents with ease and avoid vendor lock-in.

> Kosong means "empty" in Malay and Indonesian.

## Installation

Kosong requires Python 3.12 or higher. We recommend using uv as the package manager.

Init your project with:

```bash
uv init --python 3.12  # or higher
```

Then add Kosong as a dependency:

```bash
uv add llmkit
```

To enable chat providers other than Kimi (e.g. Anthropic and Google Gemini), install the optional extra:

```bash
uv add 'llmkit[contrib]'
```

## Examples

### Simple chat completion

```python
import asyncio

import llmkit
from llmkit.chat_provider.kimi import Kimi
from llmkit.message import Message


async def main() -> None:
    kimi = Kimi(
        base_url="https://api.moonshot.ai/v1",
        api_key="your_kimi_api_key_here",
        model="kimi-k2-turbo-preview",
    )

    history = [
        Message(role="user", content="Who are you?"),
    ]

    result = await llmkit.generate(
        chat_provider=kimi,
        system_prompt="You are a helpful assistant.",
        tools=[],
        history=history,
    )
    print(result.message)
    print(result.usage)


asyncio.run(main())
```

### Streaming output

```python
import asyncio

import llmkit
from llmkit.chat_provider import StreamedMessagePart
from llmkit.chat_provider.kimi import Kimi
from llmkit.message import Message


async def main() -> None:
    kimi = Kimi(
        base_url="https://api.moonshot.ai/v1",
        api_key="your_kimi_api_key_here",
        model="kimi-k2-turbo-preview",
    )

    history = [
        Message(role="user", content="Who are you?"),
    ]

    def output(message_part: StreamedMessagePart):
        print(message_part)

    result = await llmkit.generate(
        chat_provider=kimi,
        system_prompt="You are a helpful assistant.",
        tools=[],
        history=history,
        on_message_part=output,
    )
    print(result.message)
    print(result.usage)


asyncio.run(main())
```

### Tool calling with `llmkit.step`

```python
import asyncio

from pydantic import BaseModel

import llmkit
from llmkit import StepResult
from llmkit.chat_provider.kimi import Kimi
from llmkit.message import Message
from llmkit.tooling import CallableTool2, ToolOk, ToolReturnValue
from llmkit.tooling.simple import SimpleToolset


class AddToolParams(BaseModel):
    a: int
    b: int


class AddTool(CallableTool2[AddToolParams]):
    name: str = "add"
    description: str = "Add two integers."
    params: type[AddToolParams] = AddToolParams

    async def __call__(self, params: AddToolParams) -> ToolReturnValue:
        return ToolOk(output=str(params.a + params.b))


async def main() -> None:
    kimi = Kimi(
        base_url="https://api.moonshot.ai/v1",
        api_key="your_kimi_api_key_here",
        model="kimi-k2-turbo-preview",
    )

    toolset = SimpleToolset()
    toolset += AddTool()

    history = [
        Message(role="user", content="Please add 2 and 3 with the add tool."),
    ]

    result: StepResult = await llmkit.step(
        chat_provider=kimi,
        system_prompt="You are a precise math tutor.",
        toolset=toolset,
        history=history,
    )
    print(result.message)
    print(await result.tool_results())


asyncio.run(main())
```

## Builtin Demo

Kosong comes with a builtin demo agent that you can run locally. To start the demo, run:

```sh
export KIMI_BASE_URL="https://api.moonshot.ai/v1"
export KIMI_API_KEY="your_kimi_api_key"

uv run python -m llmkit kimi --with-bash
```

## Development

To set up a development environment, clone the repository and install the dependencies:

```bash
git clone https://github.com/MoonshotAI/llmkit.git
cd llmkit
uv sync --all-extras

make check  # run lint and type checks
make test   # run tests
make format # format code
```
