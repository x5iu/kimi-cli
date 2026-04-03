import ipaddress
import json
import re
import socket
from pathlib import Path
from typing import override
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup, Tag
from llmkit.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.config import Config
from kimi_cli.constant import USER_AGENT
from kimi_cli.loop.agent import Runtime
from kimi_cli.loop.toolset import get_current_tool_call_or_none
from kimi_cli.tools.utils import ToolResultBuilder, load_desc
from kimi_cli.utils.aiohttp import new_client_session
from kimi_cli.utils.logging import logger


class Params(BaseModel):
    url: str = Field(description="The URL to fetch content from.")


class FetchURL(CallableTool2[Params]):
    name: str = "FetchURL"
    description: str = load_desc(Path(__file__).parent / "fetch.md", {})
    params: type[Params] = Params

    def __init__(self, config: Config, runtime: Runtime):
        super().__init__()
        self._runtime = runtime
        self._service_config = config.services.moonshot_fetch

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if self._service_config:
            ret = await self._fetch_with_service(params)
            if not ret.is_error:
                return ret
            logger.warning("Failed to fetch URL via service: {error}", error=ret.message)
            # fallback to local fetch if service fetch fails
            fallback_ret = await self.fetch_with_http_get(params)
            if fallback_ret.is_error and ret.output:
                return ret
            return fallback_ret
        return await self.fetch_with_http_get(params)

    @staticmethod
    async def fetch_with_http_get(params: Params) -> ToolReturnValue:
        builder = ToolResultBuilder(max_line_length=None)
        validation_error = _validate_url(params.url)
        if validation_error:
            return builder.error(validation_error, brief="URL blocked")
        try:
            # Fetching arbitrary web pages can take a while on large/slow sites.
            fetch_timeout = aiohttp.ClientTimeout(total=180, sock_read=60, sock_connect=15)
            async with (
                new_client_session(timeout=fetch_timeout) as session,
                session.get(
                    params.url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        ),
                    },
                ) as response,
            ):
                if response.status >= 400:
                    return builder.error(
                        (
                            f"Failed to fetch URL. Status: {response.status}. "
                            f"This may indicate the page is not accessible or the server is down."
                        ),
                        brief=f"HTTP {response.status} error",
                    )

                resp_text = await response.text()

                content_type = response.headers.get(aiohttp.hdrs.CONTENT_TYPE, "").lower()
                if content_type.startswith(("text/plain", "text/markdown")):
                    builder.write(resp_text)
                    return builder.ok("The returned content is the full content of the page.")
        except TimeoutError:
            return builder.error(
                "Failed to fetch URL: request timed out. The server may be slow or unreachable.",
                brief="Request timed out",
            )
        except aiohttp.ClientError as e:
            return builder.error(
                (
                    f"Failed to fetch URL due to network error: {e}. "
                    "This may indicate the URL is invalid or the server is unreachable."
                ),
                brief="Network error",
            )

        if not resp_text:
            return builder.ok(
                "The response body is empty.",
                brief="Empty response body",
            )

        extracted_text = _extract_text_from_html(resp_text)

        if not extracted_text:
            return builder.error(
                (
                    "Failed to extract meaningful content from the page. "
                    "This may indicate the page content is not suitable for text extraction, "
                    "or the page requires JavaScript to render its content."
                ),
                brief="No content extracted",
            )

        builder.write(extracted_text)
        return builder.ok("The returned content is the main text content extracted from the page.")

    async def _fetch_with_service(self, params: Params) -> ToolReturnValue:
        assert self._service_config is not None

        tool_call = get_current_tool_call_or_none()
        assert tool_call is not None, "Tool call is expected to be set"

        builder = ToolResultBuilder(max_line_length=None)
        api_key = self._service_config.api_key.get_secret_value()
        if not api_key:
            return builder.error(
                "Fetch service is not configured. You may want to try other methods to fetch.",
                brief="Fetch service not configured",
            )
        headers = {
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/markdown",
            "X-Msh-Tool-Call-Id": tool_call.id,
            **(self._service_config.custom_headers or {}),
        }

        try:
            async with (
                new_client_session() as session,
                session.post(
                    self._service_config.base_url,
                    headers=headers,
                    json={"url": params.url},
                ) as response,
            ):
                if response.status != 200:
                    error_body = _format_error_response_body(await response.text())
                    if error_body:
                        builder.write(error_body)
                    return builder.error(
                        (
                            f"Failed to fetch URL via service. Status: {response.status}."
                            + (" The HTTP response body is included below." if error_body else "")
                        ),
                        brief="Failed to fetch URL via fetch service",
                    )

                content = await response.text()
                builder.write(content)
                return builder.ok(
                    "The returned content is the main content extracted from the page."
                )
        except aiohttp.ClientError as e:
            return builder.error(
                (
                    f"Failed to fetch URL via service due to network error: {str(e)}. "
                    "This may indicate the service is unreachable."
                ),
                brief="Network error when calling fetch service",
            )


_REMOVE_TAGS = frozenset(
    ["script", "style", "noscript", "svg", "iframe", "object", "embed", "head"]
)
_MAIN_CONTENT_TAGS = frozenset(["main", "article"])
_BLOCK_TAGS = frozenset(
    [
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "nav",
        "aside",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "tr",
        "dt",
        "dd",
        "figcaption",
        "figure",
    ]
)
_COLLAPSE_NEWLINES_RE = re.compile(r"\n{3,}")


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML using BeautifulSoup.

    Strips boilerplate tags, preserves code blocks and table structure, and
    collapses excessive whitespace.  Returns empty string when no meaningful
    content is found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()

    # Prefer <main> or <article> when present
    root: Tag | None = None
    for name in _MAIN_CONTENT_TAGS:
        root = soup.find(name)  # type: ignore[assignment]
        if root is not None:
            break
    if root is None:
        root = soup.body or soup  # type: ignore[assignment]
    assert root is not None

    parts: list[str] = []
    _walk(root, parts)

    text = "\n".join(parts)
    text = _COLLAPSE_NEWLINES_RE.sub("\n\n", text).strip()
    return text


def _walk(element: Tag, parts: list[str]) -> None:
    """Recursively walk the DOM tree, emitting text lines into *parts*."""
    tag_name = element.name

    # --- <pre> / <code> blocks: preserve whitespace ---
    if tag_name in ("pre", "code"):
        code_text = element.get_text()
        if code_text.strip():
            parts.append("")
            parts.append(code_text.rstrip())
            parts.append("")
        return

    # --- <table>: render as simple aligned text ---
    if tag_name == "table":
        _render_table(element, parts)
        return

    # Insert a blank line before block elements for readability
    if tag_name in _BLOCK_TAGS:
        parts.append("")

    for child in element.children:
        if isinstance(child, Tag):
            _walk(child, parts)
        else:
            text = child.get_text()
            stripped = text.strip()
            if stripped:
                parts.append(stripped)

    if tag_name in _BLOCK_TAGS:
        parts.append("")


def _render_table(table: Tag, parts: list[str]) -> None:
    """Render an HTML <table> as pipe-separated plain text."""
    parts.append("")
    for tr in table.find_all("tr"):
        cells = [(cell.get_text(separator=" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        if any(cells):
            parts.append(" | ".join(cells))
    parts.append("")


def _validate_url(url: str) -> str | None:
    """Validate URL for security (SSRF prevention). Returns error message if blocked, None if OK."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are supported."

    hostname = parsed.hostname
    if not hostname:
        return "URL has no hostname."

    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return f"Could not resolve hostname '{hostname}'."

    for addrinfo in addrinfos:
        ip = ipaddress.ip_address(addrinfo[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return (
                "URL blocked for security reasons: "
                "requests to private/internal network addresses are not allowed."
            )

    return None


def _format_error_response_body(body: str) -> str:
    body = body.strip()
    if not body:
        return ""

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body
    return json.dumps(parsed, ensure_ascii=False, indent=2)
