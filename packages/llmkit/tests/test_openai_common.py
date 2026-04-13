import asyncio
from typing import Any

import httpx
import openai
import pytest

from llmkit.chat_provider import (
    APIConnectionError,
    APITimeoutError,
    ChatProviderError,
    openai_common,
)
from llmkit.chat_provider.openai_common import convert_error
from llmkit.contrib.chat_provider.openai_legacy import OpenAILegacy


def test_create_openai_client_does_not_inject_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_common, "AsyncOpenAI", FakeAsyncOpenAI)

    openai_common.create_openai_client(
        api_key="test-key",
        base_url="https://example.com/v1",
        client_kwargs={"timeout": 3},
    )

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["timeout"] == 3
    assert "max_retries" not in captured


@pytest.mark.asyncio
async def test_retry_recovery_does_not_close_shared_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = OpenAILegacy(
        model="gpt-4.1",
        api_key="test-key",
        http_client=http_client,
    )

    provider.on_retryable_error(APIConnectionError("Connection error."))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert provider.client._client is http_client  # type: ignore[reportPrivateUsage]
    assert http_client.is_closed is False
    await http_client.aclose()


_DUMMY_REQUEST = httpx.Request("POST", "https://api.test")


class TestConvertErrorBaseAPIError:
    @pytest.mark.parametrize(
        ("message", "expected_type"),
        [
            ("Network connection lost.", APIConnectionError),
            ("Connection error.", APIConnectionError),
            ("network error", APIConnectionError),
            ("disconnected from server", APIConnectionError),
            ("Request timed out.", APITimeoutError),
            ("timed out", APITimeoutError),
            ("connection timed out", APITimeoutError),
            ("Something completely unrelated", ChatProviderError),
        ],
    )
    def test_base_api_error_mapping(
        self, message: str, expected_type: type[ChatProviderError]
    ) -> None:
        err = openai.APIError(message=message, request=_DUMMY_REQUEST, body=None)
        result = convert_error(err)
        assert type(result) is expected_type

    def test_subclass_errors_still_match_first(self):
        conn_err = openai.APIConnectionError(request=_DUMMY_REQUEST)
        result = convert_error(conn_err)
        assert type(result) is APIConnectionError
        timeout_err = openai.APITimeoutError(request=_DUMMY_REQUEST)
        result = convert_error(timeout_err)
        assert type(result) is APITimeoutError

    def test_api_error_with_body_skips_heuristic(self):
        err = openai.APIError(
            message="Connection limit exceeded",
            request=_DUMMY_REQUEST,
            body={"error": {"message": "Connection limit exceeded"}},
        )
        result = convert_error(err)
        assert type(result) is ChatProviderError
