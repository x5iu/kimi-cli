from __future__ import annotations

from unittest.mock import AsyncMock, patch

from aiohttp import web
from inline_snapshot import snapshot

from kimi_cli.eventbus.types import ToolCall
from kimi_cli.loop.toolset import current_tool_call
from kimi_cli.tools.web.search import Params, SearchWeb


def _tool_call_token():
    """Helper to set a tool call context for SearchWeb tests."""
    return current_tool_call.set(
        ToolCall(
            id="test-call-id",
            function=ToolCall.FunctionBody(name="SearchWeb", arguments=None),
        )
    )


async def test_search_web_retries_then_fails_on_persistent_429(config, runtime) -> None:
    """429 responses are retried; after all retries exhausted the error is returned."""
    request_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal request_count
        request_count += 1
        return web.json_response(
            {"error": {"message": "quota exceeded", "type": "rate_limit"}},
            status=429,
        )

    app = web.Application()
    app.router.add_post("/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[reportOptionalSubscript]

    try:
        assert config.services.moonshot_search is not None
        config.services.moonshot_search.base_url = f"http://127.0.0.1:{port}/search"
        tool = SearchWeb(config, runtime)

        token = _tool_call_token()
        try:
            with patch("kimi_cli.tools.web.search.asyncio.sleep", new_callable=AsyncMock):
                result = await tool(Params(query="kimi cli", limit=3, include_content=False))
        finally:
            current_tool_call.reset(token)

        # Should have tried initial + 2 retries = 3 requests
        assert request_count == 3
        assert result.is_error
        assert result.message == snapshot(
            "Failed to search. Status: 429. The HTTP response body is included below."
        )
        assert result.output == snapshot(
            '{\n  "error": {\n    "message": "quota exceeded",\n    "type": "rate_limit"\n  }\n}'
        )
    finally:
        await runner.cleanup()


async def test_search_web_retries_5xx_then_succeeds(config, runtime) -> None:
    """A transient 503 followed by a 200 succeeds after one retry."""
    request_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return web.json_response({"error": "service unavailable"}, status=503)
        return web.json_response(
            {
                "search_results": [
                    {
                        "site_name": "example",
                        "title": "Example Result",
                        "url": "https://example.com",
                        "snippet": "An example result",
                    }
                ]
            },
            status=200,
        )

    app = web.Application()
    app.router.add_post("/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[reportOptionalSubscript]

    try:
        assert config.services.moonshot_search is not None
        config.services.moonshot_search.base_url = f"http://127.0.0.1:{port}/search"
        tool = SearchWeb(config, runtime)

        token = _tool_call_token()
        try:
            with patch("kimi_cli.tools.web.search.asyncio.sleep", new_callable=AsyncMock):
                result = await tool(Params(query="test", limit=5, include_content=False))
        finally:
            current_tool_call.reset(token)

        assert request_count == 2
        assert not result.is_error
        assert "Example Result" in result.output
    finally:
        await runner.cleanup()


async def test_search_web_403_returns_immediately(config, runtime) -> None:
    """HTTP 403 is not retried and returns a specific guidance message."""
    request_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal request_count
        request_count += 1
        return web.json_response({"error": "forbidden"}, status=403)

    app = web.Application()
    app.router.add_post("/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[reportOptionalSubscript]

    try:
        assert config.services.moonshot_search is not None
        config.services.moonshot_search.base_url = f"http://127.0.0.1:{port}/search"
        tool = SearchWeb(config, runtime)

        token = _tool_call_token()
        try:
            result = await tool(Params(query="blocked query", limit=5, include_content=False))
        finally:
            current_tool_call.reset(token)

        # 403 should NOT be retried
        assert request_count == 1
        assert result.is_error
        assert "HTTP 403" in result.message
        assert "FetchURL" in result.message
    finally:
        await runner.cleanup()


async def test_search_web_non_retryable_4xx_returns_immediately(config, runtime) -> None:
    """Non-retryable 4xx (e.g. 400) is returned immediately without retry."""
    request_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal request_count
        request_count += 1
        return web.json_response({"error": "bad request"}, status=400)

    app = web.Application()
    app.router.add_post("/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[reportOptionalSubscript]

    try:
        assert config.services.moonshot_search is not None
        config.services.moonshot_search.base_url = f"http://127.0.0.1:{port}/search"
        tool = SearchWeb(config, runtime)

        token = _tool_call_token()
        try:
            result = await tool(Params(query="bad query", limit=5, include_content=False))
        finally:
            current_tool_call.reset(token)

        # Should NOT be retried
        assert request_count == 1
        assert result.is_error
        assert "400" in result.message
    finally:
        await runner.cleanup()
