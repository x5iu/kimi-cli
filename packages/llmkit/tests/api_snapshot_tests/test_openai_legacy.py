"""Snapshot tests for OpenAI Legacy (Chat Completions API) chat provider."""

from common import COMMON_CASES, Case, capture_request, run_test_cases
from inline_snapshot import snapshot

from llmkit.contrib.chat_provider.openai_legacy import OpenAILegacy
from llmkit.message import Message, TextPart, ThinkPart

TEST_CASES: dict[str, Case] = {**COMMON_CASES}


async def test_openai_legacy_message_conversion():
    provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False)
    results = await run_test_cases(None, provider, TEST_CASES, ("messages", "tools"))

    assert results == snapshot(
        {
            "simple_user_message": {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"},
                ],
                "tools": [],
            },
            "multi_turn_conversation": {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                    {"role": "user", "content": "And 3+3?"},
                ],
                "tools": [],
            },
            "multi_turn_with_system": {
                "messages": [
                    {"role": "system", "content": "You are a math tutor."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                    {"role": "user", "content": "And 3+3?"},
                ],
                "tools": [],
            },
            "image_url": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image.png",
                                    "id": None,
                                },
                            },
                        ],
                    }
                ],
                "tools": [],
            },
            "tool_definition": {
                "messages": [{"role": "user", "content": "Add 2 and 3"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "description": "Add two integers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {
                                        "type": "integer",
                                        "description": "First number",
                                    },
                                    "b": {
                                        "type": "integer",
                                        "description": "Second number",
                                    },
                                },
                                "required": ["a", "b"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer", "description": "First number"},
                                    "b": {"type": "integer", "description": "Second number"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    },
                ],
            },
            "tool_call_with_image": {
                "messages": [
                    {"role": "user", "content": "Add 2 and 3"},
                    {
                        "role": "assistant",
                        "content": "I'll add those numbers for you.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call_abc123",
                                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {"type": "text", "text": "5"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image.png",
                                    "id": None,
                                },
                            },
                        ],
                        "tool_call_id": "call_abc123",
                    },
                ],
                "tools": [],
            },
            "tool_call": {
                "messages": [
                    {"role": "user", "content": "Add 2 and 3"},
                    {
                        "role": "assistant",
                        "content": "I'll add those numbers for you.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call_abc123",
                                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": "5", "tool_call_id": "call_abc123"},
                ],
                "tools": [],
            },
            "parallel_tool_calls": {
                "messages": [
                    {"role": "user", "content": "Calculate 2+3 and 4*5"},
                    {
                        "role": "assistant",
                        "content": "I'll calculate both.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call_add",
                                "function": {
                                    "name": "add",
                                    "arguments": '{"a": 2, "b": 3}',
                                },
                            },
                            {
                                "type": "function",
                                "id": "call_mul",
                                "function": {
                                    "name": "multiply",
                                    "arguments": '{"a": 4, "b": 5}',
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "text",
                                "text": "<system-reminder>This is a system reminder"
                                "</system-reminder>",
                            },
                            {"type": "text", "text": "5"},
                        ],
                        "tool_call_id": "call_add",
                    },
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "text",
                                "text": "<system-reminder>This is a system reminder"
                                "</system-reminder>",
                            },
                            {"type": "text", "text": "20"},
                        ],
                        "tool_call_id": "call_mul",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "description": "Add two integers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer", "description": "First number"},
                                    "b": {"type": "integer", "description": "Second number"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer", "description": "First number"},
                                    "b": {"type": "integer", "description": "Second number"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    },
                ],
            },
        }
    )


async def test_openai_legacy_reasoning_content():
    provider = OpenAILegacy(
        model="deepseek-reasoner",
        api_key="test-key",
        stream=False,
        reasoning_key="reasoning_content",
    )
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[ThinkPart(think="Thinking..."), TextPart(text="4.")],
        ),
        Message(role="user", content="Thanks!"),
    ]
    body = await capture_request(None, provider, "", [], history)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "4.",
                "reasoning_content": "Thinking...",
            },
            {"role": "user", "content": "Thanks!"},
        ]
    )


async def test_openai_legacy_generation_kwargs():
    provider = OpenAILegacy(
        model="gpt-4.1", api_key="test-key", stream=False
    ).with_generation_kwargs(temperature=0.7, max_tokens=2048)
    body = await capture_request(None, provider, "", [], [Message(role="user", content="Hi")])
    assert (body["temperature"], body["max_tokens"]) == snapshot((0.7, 2048))


async def test_openai_legacy_with_thinking():
    provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False).with_thinking("high")
    body = await capture_request(None, provider, "", [], [Message(role="user", content="Think")])
    assert body["reasoning_effort"] == snapshot("high")
