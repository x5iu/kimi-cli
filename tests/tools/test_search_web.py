from __future__ import annotations

from aiohttp import web
from inline_snapshot import snapshot

from kimi_cli.soul.toolset import current_tool_call
from kimi_cli.tools.web.search import Params, SearchWeb
from kimi_cli.wire.types import ToolCall


async def test_search_web_includes_http_error_body(config, runtime) -> None:
    async def handler(request: web.Request) -> web.Response:
        assert request.method == "POST"
        assert request.headers.get("Authorization") == "Bearer test-api-key"

        data = await request.json()
        assert data == {
            "text_query": "kimi cli",
            "limit": 3,
            "enable_page_crawling": False,
            "timeout_seconds": 30,
        }

        return web.json_response(
            {
                "error": {
                    "message": "quota exceeded",
                    "type": "rate_limit",
                }
            },
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

        token = current_tool_call.set(
            ToolCall(
                id="test-call-id",
                function=ToolCall.FunctionBody(name="SearchWeb", arguments=None),
            )
        )
        try:
            result = await tool(Params(query="kimi cli", limit=3, include_content=False))
        finally:
            current_tool_call.reset(token)

        assert result.is_error
        assert result.message == snapshot(
            "Failed to search. Status: 429. The HTTP response body is included below."
        )
        assert result.output == snapshot(
            '{\n  "error": {\n    "message": "quota exceeded",\n    "type": "rate_limit"\n  }\n}'
        )
    finally:
        await runner.cleanup()
