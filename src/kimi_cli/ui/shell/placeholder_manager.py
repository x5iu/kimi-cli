from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from PIL import Image

from kimi_cli.utils.envvar import get_env_int
from kimi_cli.wire.types import ContentPart, TextPart

from .placeholder_cache import AttachmentCache, parse_attachment_kind

_IMAGE_PLACEHOLDER_RE = re.compile(
    r"\[(?P<type>[a-zA-Z0-9_\-]+):(?P<id>[a-zA-Z0-9_\-\.]+)"
    r"(?:,(?P<width>\d+)x(?P<height>\d+))?\]"
)
_PASTED_TEXT_PLACEHOLDER_RE = re.compile(
    r"\[Pasted text #(?P<id>\d+)(?: \+(?P<lines>\d+) lines?)?\]"
)

_TEXT_PASTE_CHAR_THRESHOLD = get_env_int("KIMI_CLI_PASTE_CHAR_THRESHOLD", 1000)
_TEXT_PASTE_LINE_THRESHOLD = get_env_int("KIMI_CLI_PASTE_LINE_THRESHOLD", 15)


def sanitize_surrogates(text: str) -> str:
    """Replace lone UTF-16 surrogates that cannot be encoded as UTF-8.

    Windows clipboard data sometimes contains unpaired surrogates from
    applications that use UTF-16 internally. Passing such strings to
    ``json.dumps`` or writing them to a UTF-8 file raises
    ``UnicodeEncodeError``, so we replace them with U+FFFD early.
    """
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def normalize_pasted_text(text: str) -> str:
    """Normalize pasted text into the same newline format used by prompt_toolkit."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def count_text_lines(text: str) -> int:
    if not text:
        return 1
    return text.count("\n") + 1


def should_placeholderize_pasted_text(text: str) -> bool:
    normalized = normalize_pasted_text(text)
    return (
        len(normalized) >= _TEXT_PASTE_CHAR_THRESHOLD
        or count_text_lines(normalized) >= _TEXT_PASTE_LINE_THRESHOLD
    )


def build_pasted_text_placeholder(paste_id: int, text: str) -> str:
    line_count = count_text_lines(text)
    if line_count <= 1:
        return f"[Pasted text #{paste_id}]"
    return f"[Pasted text #{paste_id} +{line_count} lines]"


@dataclass(slots=True)
class PlaceholderTokenMatch:
    start: int
    end: int
    raw: str
    handler: PlaceholderHandler
    match: re.Match[str]


class PlaceholderHandler(Protocol):
    def find_next(self, text: str, start: int = 0) -> PlaceholderTokenMatch | None: ...

    def resolve_content(self, match: PlaceholderTokenMatch) -> list[ContentPart] | None: ...

    def expand_text(self, match: PlaceholderTokenMatch) -> str | None: ...

    def serialize_for_history(self, match: PlaceholderTokenMatch) -> str | None: ...

    def expand_for_editor(self, match: PlaceholderTokenMatch) -> str | None: ...


@dataclass(slots=True)
class PastedTextEntry:
    paste_id: int
    text: str

    @property
    def token(self) -> str:
        return build_pasted_text_placeholder(self.paste_id, self.text)


class PastedTextPlaceholderHandler:
    def __init__(self) -> None:
        self._entries: dict[int, PastedTextEntry] = {}
        self._next_id = 1

    def create_placeholder(self, text: str) -> str:
        normalized = sanitize_surrogates(normalize_pasted_text(text))
        entry = PastedTextEntry(paste_id=self._next_id, text=normalized)
        self._entries[entry.paste_id] = entry
        self._next_id += 1
        return entry.token

    def maybe_placeholderize(self, text: str) -> str:
        normalized = normalize_pasted_text(text)
        if not should_placeholderize_pasted_text(normalized):
            return normalized
        return self.create_placeholder(normalized)

    def entry_for_id(self, paste_id: int) -> PastedTextEntry | None:
        return self._entries.get(paste_id)

    def iter_entries_for_command(
        self, command: str
    ) -> list[tuple[PlaceholderTokenMatch, PastedTextEntry]]:
        entries: list[tuple[PlaceholderTokenMatch, PastedTextEntry]] = []
        cursor = 0
        while match := self.find_next(command, cursor):
            paste_id = int(match.match.group("id"))
            entry = self.entry_for_id(paste_id)
            if entry is not None:
                entries.append((match, entry))
            cursor = match.end
        return entries

    def find_next(self, text: str, start: int = 0) -> PlaceholderTokenMatch | None:
        match = _PASTED_TEXT_PLACEHOLDER_RE.search(text, start)
        if match is None:
            return None
        return PlaceholderTokenMatch(
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
            handler=self,
            match=match,
        )

    def resolve_content(self, match: PlaceholderTokenMatch) -> list[ContentPart] | None:
        paste_id = int(match.match.group("id"))
        entry = self.entry_for_id(paste_id)
        if entry is None:
            return None
        return [TextPart(text=entry.text)]

    def expand_text(self, match: PlaceholderTokenMatch) -> str | None:
        paste_id = int(match.match.group("id"))
        entry = self.entry_for_id(paste_id)
        return None if entry is None else entry.text

    def serialize_for_history(self, match: PlaceholderTokenMatch) -> str | None:
        return self.expand_text(match)

    def expand_for_editor(self, match: PlaceholderTokenMatch) -> str | None:
        return self.expand_text(match)

    def refold_after_editor(self, edited_text: str, original_command: str) -> str:
        expanded_original, intervals = self._expanded_text_and_intervals(original_command)
        if not intervals:
            return edited_text

        opcodes = SequenceMatcher(
            a=expanded_original,
            b=edited_text,
            autojunk=False,
        ).get_opcodes()
        replacements: list[tuple[int, int, str]] = []
        for start, end, token, expected_text in intervals:
            mapped = self._map_interval(opcodes, start, end)
            if mapped is None:
                continue
            mapped_start, mapped_end = mapped
            if edited_text[mapped_start:mapped_end] != expected_text:
                continue
            replacements.append((mapped_start, mapped_end, token))

        result = edited_text
        for start, end, token in reversed(replacements):
            result = result[:start] + token + result[end:]
        return result

    def _expanded_text_and_intervals(
        self, command: str
    ) -> tuple[str, list[tuple[int, int, str, str]]]:
        parts: list[str] = []
        intervals: list[tuple[int, int, str, str]] = []
        cursor = 0
        expanded_cursor = 0
        for match, entry in self.iter_entries_for_command(command):
            literal = command[cursor : match.start]
            if literal:
                parts.append(literal)
                expanded_cursor += len(literal)
            start = expanded_cursor
            parts.append(entry.text)
            expanded_cursor += len(entry.text)
            intervals.append((start, expanded_cursor, match.raw, entry.text))
            cursor = match.end
        if cursor < len(command):
            parts.append(command[cursor:])
        return "".join(parts), intervals

    @staticmethod
    def _map_interval(
        opcodes: Sequence[tuple[str, int, int, int, int]], start: int, end: int
    ) -> tuple[int, int] | None:
        mapped_start: int | None = None
        mapped_end: int | None = None
        cursor = start
        for tag, i1, i2, j1, _j2 in opcodes:
            if i2 <= cursor:
                continue
            if i1 >= end:
                break
            overlap_start = max(i1, cursor, start)
            overlap_end = min(i2, end)
            if overlap_start >= overlap_end:
                continue
            if tag != "equal":
                return None
            segment_start = j1 + (overlap_start - i1)
            segment_end = j1 + (overlap_end - i1)
            if mapped_start is None:
                mapped_start = segment_start
            elif mapped_end != segment_start:
                return None
            mapped_end = segment_end
            cursor = overlap_end
        if cursor != end or mapped_start is None or mapped_end is None:
            return None
        return mapped_start, mapped_end


class ImagePlaceholderHandler:
    def __init__(self, attachment_cache: AttachmentCache) -> None:
        self._attachment_cache = attachment_cache

    def create_placeholder(self, image: Image.Image) -> str | None:
        cached = self._attachment_cache.store_image(image)
        if cached is None:
            return None
        return f"[image:{cached.attachment_id},{image.width}x{image.height}]"

    def find_next(self, text: str, start: int = 0) -> PlaceholderTokenMatch | None:
        match = _IMAGE_PLACEHOLDER_RE.search(text, start)
        if match is None:
            return None
        return PlaceholderTokenMatch(
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
            handler=self,
            match=match,
        )

    def resolve_content(self, match: PlaceholderTokenMatch) -> list[ContentPart] | None:
        kind = parse_attachment_kind(match.match.group("type"))
        if kind is None:
            return None
        return self._attachment_cache.load_content_parts(kind, match.match.group("id"))

    def expand_text(self, match: PlaceholderTokenMatch) -> str | None:
        return match.raw

    def serialize_for_history(self, match: PlaceholderTokenMatch) -> str | None:
        return match.raw

    def expand_for_editor(self, match: PlaceholderTokenMatch) -> str | None:
        return match.raw


@dataclass(slots=True)
class ResolvedPromptCommand:
    display_command: str
    resolved_text: str
    content: list[ContentPart]


class PromptPlaceholderManager:
    def __init__(self, attachment_cache: AttachmentCache | None = None) -> None:
        self._attachment_cache = attachment_cache or AttachmentCache()
        self._text_handler = PastedTextPlaceholderHandler()
        self._image_handler = ImagePlaceholderHandler(self._attachment_cache)
        self._handlers: tuple[PlaceholderHandler, ...] = (
            self._text_handler,
            self._image_handler,
        )

    @property
    def attachment_cache(self) -> AttachmentCache:
        return self._attachment_cache

    def maybe_placeholderize_pasted_text(self, text: str) -> str:
        return self._text_handler.maybe_placeholderize(text)

    def create_image_placeholder(self, image: Image.Image) -> str | None:
        return self._image_handler.create_placeholder(image)

    def resolve_command(self, command: str) -> ResolvedPromptCommand:
        content: list[ContentPart] = []
        resolved_chunks: list[str] = []
        cursor = 0

        while match := self._find_next_match(command, cursor):
            if match.start > cursor:
                literal = command[cursor : match.start]
                content.append(TextPart(text=literal))
                resolved_chunks.append(literal)

            resolved_content = match.handler.resolve_content(match)
            if resolved_content is None:
                content.append(TextPart(text=match.raw))
                resolved_chunks.append(match.raw)
            else:
                content.extend(resolved_content)
                expanded = match.handler.expand_text(match)
                resolved_chunks.append(match.raw if expanded is None else expanded)

            cursor = match.end

        if cursor < len(command):
            literal = command[cursor:]
            content.append(TextPart(text=literal))
            resolved_chunks.append(literal)

        return ResolvedPromptCommand(
            display_command=command,
            resolved_text="".join(resolved_chunks),
            content=content,
        )

    def serialize_for_history(self, command: str) -> str:
        return self._rewrite_command(
            command,
            lambda handler, match: handler.serialize_for_history(match),
        )

    def expand_for_editor(self, command: str) -> str:
        return self._rewrite_command(
            command,
            lambda handler, match: handler.expand_for_editor(match),
        )

    def refold_after_editor(self, edited_text: str, original_command: str) -> str:
        return self._text_handler.refold_after_editor(edited_text, original_command)

    def _find_next_match(self, text: str, start: int = 0) -> PlaceholderTokenMatch | None:
        earliest: PlaceholderTokenMatch | None = None
        for handler in self._handlers:
            match = handler.find_next(text, start)
            if match is None:
                continue
            if earliest is None or match.start < earliest.start:
                earliest = match
        return earliest

    def _rewrite_command(
        self,
        command: str,
        replacer: Callable[[PlaceholderHandler, PlaceholderTokenMatch], str | None],
    ) -> str:
        parts: list[str] = []
        cursor = 0

        while match := self._find_next_match(command, cursor):
            if match.start > cursor:
                parts.append(command[cursor : match.start])
            replacement = replacer(match.handler, match)
            parts.append(match.raw if replacement is None else replacement)
            cursor = match.end

        if cursor < len(command):
            parts.append(command[cursor:])

        return "".join(parts)
