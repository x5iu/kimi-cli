import asyncio
import json
from pathlib import Path
from typing import override

import aiohttp
from pydantic import BaseModel, Field, ValidationError

from kimi_cli.config import Config
from kimi_cli.constant import USER_AGENT
from kimi_cli.loop.agent import Runtime
from kimi_cli.loop.toolset import get_current_tool_call_or_none
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.utils import ToolResultBuilder, load_desc
from kimi_cli.utils.aiohttp import new_client_session
from kimi_cli.utils.logging import logger
from llmkit.tooling import CallableTool2, ToolReturnValue

_MAX_RETRIES = 2
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class Params(BaseModel):
    query: str = Field(description="The query text to search for.")
    limit: int = Field(
        description=(
            "The number of results to return. "
            "Typically you do not need to set this value. "
            "When the results do not contain what you need, "
            "you probably want to give a more concrete query."
        ),
        default=5,
        ge=1,
        le=20,
    )
    include_content: bool = Field(
        description=(
            "Whether to include the content of the web pages in the results. "
            "It can consume a large amount of tokens when this is set to True. "
            "You should avoid enabling this when `limit` is set to a large value."
        ),
        default=False,
    )


class SearchWeb(CallableTool2[Params]):
    name: str = "SearchWeb"
    description: str = load_desc(Path(__file__).parent / "search.md", {})
    params: type[Params] = Params

    def __init__(self, config: Config, runtime: Runtime):
        super().__init__()
        if config.services.moonshot_search is None:
            raise SkipThisTool()
        self._runtime = runtime
        self._base_url = config.services.moonshot_search.base_url
        self._api_key = config.services.moonshot_search.api_key
        self._custom_headers = config.services.moonshot_search.custom_headers or {}

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        builder = ToolResultBuilder(max_line_length=None)

        api_key = self._api_key.get_secret_value()
        if not self._base_url or not api_key:
            return builder.error(
                "Search service is not configured. You may want to try other methods to search.",
                brief="Search service not configured",
            )

        tool_call = get_current_tool_call_or_none()
        assert tool_call is not None, "Tool call is expected to be set"

        last_error_result: ToolReturnValue | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with (
                    new_client_session() as session,
                    session.post(
                        self._base_url,
                        headers={
                            "User-Agent": USER_AGENT,
                            "Authorization": f"Bearer {api_key}",
                            "X-Msh-Tool-Call-Id": tool_call.id,
                            **self._custom_headers,
                        },
                        json={
                            "text_query": params.query,
                            "limit": params.limit,
                            "enable_page_crawling": params.include_content,
                            "timeout_seconds": 30,
                        },
                    ) as response,
                ):
                    if response.status != 200:
                        logger.warning(
                            "SearchWeb HTTP error: status={status}, query={query}",
                            status=response.status,
                            query=params.query,
                        )
                        error_body = _format_error_response_body(await response.text())

                        # HTTP 403: return immediately with specific guidance
                        if response.status == 403:
                            err_builder = ToolResultBuilder(max_line_length=None)
                            if error_body:
                                err_builder.write(error_body)
                            return err_builder.error(
                                "Search service denied the request (HTTP 403). "
                                "Use `FetchURL` with a specific URL instead.",
                                brief="HTTP 403 Forbidden",
                            )

                        # Retryable statuses: 429 and 5xx
                        if response.status in _RETRYABLE_STATUSES:
                            err_builder = ToolResultBuilder(max_line_length=None)
                            if error_body:
                                err_builder.write(error_body)
                            last_error_result = err_builder.error(
                                (
                                    f"Failed to search. Status: {response.status}."
                                    + (
                                        " The HTTP response body is included below."
                                        if error_body
                                        else ""
                                    )
                                ),
                                brief="Failed to search",
                            )
                            if attempt < _MAX_RETRIES:
                                await asyncio.sleep(2**attempt)
                                continue
                            return last_error_result

                        # Other 4xx: return immediately
                        err_builder = ToolResultBuilder(max_line_length=None)
                        if error_body:
                            err_builder.write(error_body)
                        return err_builder.error(
                            (
                                f"Failed to search. Status: {response.status}."
                                + (
                                    " The HTTP response body is included below."
                                    if error_body
                                    else ""
                                )
                            ),
                            brief="Failed to search",
                        )

                    try:
                        results = Response(**await response.json()).search_results
                    except ValidationError as e:
                        logger.warning(
                            "SearchWeb response parse error: {error}, query={query}",
                            error=e,
                            query=params.query,
                        )
                        return builder.error(
                            (
                                f"Failed to parse search results. Error: {e}. "
                                "This may indicates that the search service is currently "
                                "unavailable."
                            ),
                            brief="Failed to parse search results",
                        )

                    # Success — break out of the retry loop
                    break

            except TimeoutError:
                logger.warning("SearchWeb request timed out: query={query}", query=params.query)
                return builder.error(
                    "Search request timed out. The search service may be slow or unavailable.",
                    brief="Search request timed out",
                )
            except aiohttp.ClientError as e:
                logger.warning(
                    "SearchWeb network error: {error}, query={query}",
                    error=e,
                    query=params.query,
                )
                return builder.error(
                    f"Search request failed: {e}. The search service may be unavailable.",
                    brief="Search request failed",
                )
        else:
            # All retries exhausted (should not reach here, but safety net)
            assert last_error_result is not None
            return last_error_result

        for i, result in enumerate(results):
            if i > 0:
                builder.write("---\n\n")
            builder.write(
                f"Title: {result.title}\nDate: {result.date}\n"
                f"URL: {result.url}\nSummary: {result.snippet}\n\n"
            )
            if result.content:
                builder.write(f"{result.content}\n\n")

        return builder.ok()


def _format_error_response_body(body: str) -> str:
    body = body.strip()
    if not body:
        return ""

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body
    return json.dumps(parsed, ensure_ascii=False, indent=2)


class SearchResult(BaseModel):
    site_name: str
    title: str
    url: str
    snippet: str
    content: str = ""
    date: str = ""
    icon: str = ""
    mime: str = ""


class Response(BaseModel):
    search_results: list[SearchResult]
