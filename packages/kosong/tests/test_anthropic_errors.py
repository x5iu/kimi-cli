"""Tests for _convert_error in the Anthropic chat provider."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    APIStatusError as AnthropicAPIStatusError,
    APITimeoutError as AnthropicAPITimeoutError,
    AuthenticationError as AnthropicAuthenticationError,
    PermissionDeniedError as AnthropicPermissionDeniedError,
    RateLimitError as AnthropicRateLimitError,
)

from kosong.chat_provider import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ChatProviderError,
)
from kosong.contrib.chat_provider.anthropic import _convert_error


# ---------------------------------------------------------------------------
# httpx error mapping
# ---------------------------------------------------------------------------


def test_httpx_timeout_maps_to_api_timeout_error():
    exc = httpx.ReadTimeout("read timed out")
    result = _convert_error(exc)
    assert isinstance(result, APITimeoutError)
    assert "read timed out" in str(result)


def test_httpx_connect_timeout_maps_to_api_timeout_error():
    exc = httpx.ConnectTimeout("connect timed out")
    result = _convert_error(exc)
    assert isinstance(result, APITimeoutError)


def test_httpx_network_error_maps_to_api_connection_error():
    exc = httpx.ConnectError("connection refused")
    result = _convert_error(exc)
    assert isinstance(result, APIConnectionError)
    assert "connection refused" in str(result)


def test_httpx_remote_protocol_error_maps_to_api_connection_error():
    exc = httpx.RemoteProtocolError("bad status line")
    result = _convert_error(exc)
    assert isinstance(result, APIConnectionError)
    assert "bad status line" in str(result)


def test_httpx_http_status_error_maps_to_api_status_error():
    response = MagicMock()
    response.status_code = 502
    request = MagicMock()
    exc = httpx.HTTPStatusError("bad gateway", request=request, response=response)
    result = _convert_error(exc)
    assert isinstance(result, APIStatusError)
    assert result.status_code == 502


def test_httpx_generic_http_error_maps_to_chat_provider_error():
    """Any httpx.HTTPError that is not a known subclass -> ChatProviderError."""
    exc = httpx.HTTPError("some generic error")
    result = _convert_error(exc)
    assert isinstance(result, ChatProviderError)
    # Must not be one of the specialised subclasses
    assert type(result) is ChatProviderError
    assert "httpx error" in str(result)


# ---------------------------------------------------------------------------
# Anthropic SDK error mapping
# ---------------------------------------------------------------------------


def _make_anthropic_status_error(
    status_code: int,
    cls: type[AnthropicAPIStatusError] = AnthropicAPIStatusError,
) -> AnthropicAPIStatusError:
    """Build an Anthropic SDK status error with a mocked httpx.Response."""
    response = httpx.Response(status_code=status_code, request=httpx.Request("POST", "https://x"))
    body: object = {"error": {"message": "test error"}}
    return cls(message="test error", response=response, body=body)


def test_anthropic_api_status_error():
    exc = _make_anthropic_status_error(500)
    result = _convert_error(exc)
    assert isinstance(result, APIStatusError)
    assert result.status_code == 500


def test_anthropic_authentication_error():
    exc = _make_anthropic_status_error(401, AnthropicAuthenticationError)
    result = _convert_error(exc)
    assert isinstance(result, APIStatusError)
    assert result.status_code == 401


def test_anthropic_permission_denied_error():
    exc = _make_anthropic_status_error(403, AnthropicPermissionDeniedError)
    result = _convert_error(exc)
    assert isinstance(result, APIStatusError)
    assert result.status_code == 403


def test_anthropic_rate_limit_error():
    exc = _make_anthropic_status_error(429, AnthropicRateLimitError)
    result = _convert_error(exc)
    assert isinstance(result, APIStatusError)
    assert result.status_code == 429


def test_anthropic_api_timeout_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = AnthropicAPITimeoutError(request)
    result = _convert_error(exc)
    assert isinstance(result, APITimeoutError)


def test_anthropic_api_connection_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = AnthropicAPIConnectionError(request=request)
    result = _convert_error(exc)
    assert isinstance(result, APIConnectionError)


def test_anthropic_timeout_before_connection(
):
    """AnthropicAPITimeoutError is a subclass of AnthropicAPIConnectionError.

    _convert_error must check for timeout *before* connection to avoid
    misclassifying timeouts as connection errors.
    """
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = AnthropicAPITimeoutError(request)
    # Verify the inheritance that makes ordering important
    assert isinstance(exc, AnthropicAPIConnectionError)
    result = _convert_error(exc)
    # Must be timeout, not connection
    assert isinstance(result, APITimeoutError)
    assert not isinstance(result, APIConnectionError)
