from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import shlex
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from hashlib import md5, sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, override

from kaos.path import KaosPath
from PIL import Image
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    FuzzyCompleter,
    WordCompleter,
    merge_completers,
)
from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    Float,
    FloatContainer,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea
from pydantic import BaseModel, ValidationError
from rich.console import Console as RichConsole
from rich.console import RenderableType
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.text import Text as RichText

from kimi_cli.llm import ModelCapability
from kimi_cli.share import get_share_dir
from kimi_cli.soul import StatusSnapshot, format_context_status
from kimi_cli.ui.shell.console import console
from kimi_cli.ui.shell.keyboard import KeyEvent
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.utils.clipboard import (
    grab_media_from_clipboard,
    is_clipboard_available,
)
from kimi_cli.utils.logging import logger
from kimi_cli.utils.media_tags import wrap_media_part
from kimi_cli.utils.slashcmd import SlashCommand
from kimi_cli.utils.string import random_string
from kimi_cli.wire.types import ContentPart, ImageURLPart, StepInterrupted, TextPart

PROMPT_SYMBOL = "✨"
PROMPT_SYMBOL_SHELL = "$"
PROMPT_SYMBOL_THINKING = "💫"
PROMPT_SYMBOL_PLAN = "📋"
STEADY_INPUT_CURSOR = SimpleCursorShapeConfig(CursorShape.BEAM)

_DOTS_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_MOON_FRAMES = ("🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘")
_PAUSE_FRAMES = ("…",)
_INDICATOR_FRAMES: dict[str, tuple[str, ...]] = {
    "running": _DOTS_FRAMES,
    "thinking": _DOTS_FRAMES,
    "composing": _DOTS_FRAMES,
    "tool": _DOTS_FRAMES,
    "mcp": _DOTS_FRAMES,
    "compacting": _DOTS_FRAMES,
    "moon": _MOON_FRAMES,
    "approval": _PAUSE_FRAMES,
    "question": _PAUSE_FRAMES,
}
_INDICATOR_STYLES = {
    "running": "fg:#22c55e",
    "thinking": "fg:#22c55e",
    "composing": "fg:#22c55e",
    "tool": "fg:#22c55e",
    "mcp": "fg:#38bdf8",
    "compacting": "fg:#c084fc",
    "moon": "fg:#facc15",
    "approval": "fg:#f59e0b",
    "question": "fg:#22d3ee",
}
_TURN_UI_REFRESH_INTERVAL = 0.1
_TURN_UI_FULL_REPAINT_INTERVAL = 0.5


def _rich_from_ansi(text: str) -> RichText:
    return RichText.from_ansi(text)


def _rich_style_to_prompt_toolkit(style: RichStyle | None) -> str:
    if style is None:
        return ""
    parts: list[str] = []
    if style.color is not None:
        fg = style.color.get_truecolor()
        parts.append(f"fg:#{fg.red:02x}{fg.green:02x}{fg.blue:02x}")
    if style.bgcolor is not None:
        bg = style.bgcolor.get_truecolor()
        parts.append(f"bg:#{bg.red:02x}{bg.green:02x}{bg.blue:02x}")
    if style.bold:
        parts.append("bold")
    if style.italic:
        parts.append("italic")
    if style.underline:
        parts.append("underline")
    if style.strike:
        parts.append("strike")
    if style.reverse:
        parts.append("reverse")
    if style.blink:
        parts.append("blink")
    return " ".join(parts)


class _RichRenderableControl(UIControl):
    """Render Rich renderables directly with the actual width assigned by prompt_toolkit."""

    def __init__(self, get_renderable: Callable[[], RenderableType]) -> None:
        self._get_renderable = get_renderable
        self._console = RichConsole(force_terminal=True, color_system="truecolor", highlight=False)
        self._cache: dict[tuple[int, tuple[tuple[tuple[str, str], ...], ...]], UIContent] = {}

    def _render_lines(self, width: int) -> list[list[tuple[str, str]]]:
        renderable = self._get_renderable()
        options = self._console.options.update_width(max(20, width))
        lines = self._console.render_lines(renderable, options=options, pad=False, new_lines=False)
        rendered_lines: list[list[tuple[str, str]]] = []
        for line in lines or [[]]:
            fragments: list[tuple[str, str]] = []
            for segment in Segment.simplify(line):
                if segment.control or not segment.text:
                    continue
                fragments.append((_rich_style_to_prompt_toolkit(segment.style), segment.text))
            rendered_lines.append(fragments)
        return rendered_lines or [[]]

    def line_count(self, width: int) -> int:
        return len(self._render_lines(width))

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Any,
    ) -> int | None:
        return min(self.line_count(width), max_available_height)

    def create_content(self, width: int, height: int | None) -> UIContent:
        lines = self._render_lines(width)
        key_lines = tuple(tuple(line) for line in lines)
        key = (max(20, width), key_lines)

        def get_content() -> UIContent:
            return UIContent(
                get_line=lambda i: list(key_lines[i]),
                line_count=len(key_lines),
                show_cursor=False,
            )

        cached = self._cache.get(key)
        if cached is None:
            cached = get_content()
            self._cache = {key: cached}
        return cached


class SlashCommandCompleter(Completer):
    """
    A completer that:
    - Shows one line per slash command in the form: "/name (alias1, alias2)"
    - Fuzzy-matches by primary name or any alias while inserting the canonical "/name"
    - Only activates when the current token starts with '/'
    """

    def __init__(self, available_commands: Sequence[SlashCommand[Any]]) -> None:
        super().__init__()
        self._available_commands = list(available_commands)
        self._command_lookup: dict[str, list[SlashCommand[Any]]] = {}
        words: list[str] = []

        for cmd in sorted(self._available_commands, key=lambda c: c.name):
            if cmd.name not in self._command_lookup:
                self._command_lookup[cmd.name] = []
                words.append(cmd.name)
            self._command_lookup[cmd.name].append(cmd)
            for alias in cmd.aliases:
                if alias in self._command_lookup:
                    self._command_lookup[alias].append(cmd)
                else:
                    self._command_lookup[alias] = [cmd]
                    words.append(alias)

        self._word_pattern = re.compile(r"[^\s]+")
        self._fuzzy_pattern = r"^[^\s]*"
        self._word_completer = WordCompleter(words, WORD=False, pattern=self._word_pattern)
        self._fuzzy = FuzzyCompleter(self._word_completer, WORD=False, pattern=self._fuzzy_pattern)

    @override
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor

        # Only autocomplete when the input buffer has no other content.
        if document.text_after_cursor.strip():
            return

        # Only consider the last token (allowing future arguments after a space)
        last_space = text.rfind(" ")
        token = text[last_space + 1 :]
        prefix = text[: last_space + 1] if last_space != -1 else ""

        if prefix.strip():
            return
        if not token.startswith("/"):
            return

        typed = token[1:]
        if typed and typed in self._command_lookup:
            return
        mention_doc = Document(text=typed, cursor_position=len(typed))
        candidates = list(self._fuzzy.get_completions(mention_doc, complete_event))

        seen: set[str] = set()

        for candidate in candidates:
            commands = self._command_lookup.get(candidate.text)
            if not commands:
                continue
            for cmd in commands:
                if cmd.name in seen:
                    continue
                seen.add(cmd.name)
                yield Completion(
                    text=f"/{cmd.name}",
                    start_position=-len(token),
                    display=cmd.slash_name(),
                    display_meta=cmd.description,
                )


class LocalFileMentionCompleter(Completer):
    """Offer fuzzy `@` path completion by indexing workspace files."""

    _FRAGMENT_PATTERN = re.compile(r"[^\s@]+")
    _TRIGGER_GUARDS = frozenset((".", "-", "_", "`", "'", '"', ":", "@", "#", "~"))
    _IGNORED_NAME_GROUPS: dict[str, tuple[str, ...]] = {
        "vcs_metadata": (".DS_Store", ".bzr", ".git", ".hg", ".svn"),
        "tooling_caches": (
            ".build",
            ".cache",
            ".coverage",
            ".fleet",
            ".gradle",
            ".idea",
            ".ipynb_checkpoints",
            ".pnpm-store",
            ".pytest_cache",
            ".pub-cache",
            ".ruff_cache",
            ".swiftpm",
            ".tox",
            ".venv",
            ".vs",
            ".vscode",
            ".yarn",
            ".yarn-cache",
        ),
        "js_frontend": (
            ".next",
            ".nuxt",
            ".parcel-cache",
            ".svelte-kit",
            ".turbo",
            ".vercel",
            "node_modules",
        ),
        "python_packaging": (
            "__pycache__",
            "build",
            "coverage",
            "dist",
            "htmlcov",
            "pip-wheel-metadata",
            "venv",
        ),
        "java_jvm": (".mvn", "out", "target"),
        "dotnet_native": ("bin", "cmake-build-debug", "cmake-build-release", "obj"),
        "bazel_buck": ("bazel-bin", "bazel-out", "bazel-testlogs", "buck-out"),
        "misc_artifacts": (
            ".dart_tool",
            ".serverless",
            ".stack-work",
            ".terraform",
            ".terragrunt-cache",
            "DerivedData",
            "Pods",
            "deps",
            "tmp",
            "vendor",
        ),
    }
    _IGNORED_NAMES = frozenset(name for group in _IGNORED_NAME_GROUPS.values() for name in group)
    _IGNORED_PATTERN_PARTS: tuple[str, ...] = (
        r".*_cache$",
        r".*-cache$",
        r".*\.egg-info$",
        r".*\.dist-info$",
        r".*\.py[co]$",
        r".*\.class$",
        r".*\.sw[po]$",
        r".*~$",
        r".*\.(?:tmp|bak)$",
    )
    _IGNORED_PATTERNS = re.compile(
        "|".join(f"(?:{part})" for part in _IGNORED_PATTERN_PARTS),
        re.IGNORECASE,
    )

    def __init__(
        self,
        root: Path,
        *,
        refresh_interval: float = 2.0,
        limit: int = 1000,
    ) -> None:
        self._root = root
        self._refresh_interval = refresh_interval
        self._limit = limit
        self._cache_time: float = 0.0
        self._cached_paths: list[str] = []
        self._top_cache_time: float = 0.0
        self._top_cached_paths: list[str] = []
        self._fragment_hint: str | None = None

        self._word_completer = WordCompleter(
            self._get_paths,
            WORD=False,
            pattern=self._FRAGMENT_PATTERN,
        )

        self._fuzzy = FuzzyCompleter(
            self._word_completer,
            WORD=False,
            pattern=r"^[^\s@]*",
        )

    @classmethod
    def _is_ignored(cls, name: str) -> bool:
        if not name:
            return True
        if name in cls._IGNORED_NAMES:
            return True
        return bool(cls._IGNORED_PATTERNS.fullmatch(name))

    def _get_paths(self) -> list[str]:
        fragment = self._fragment_hint or ""
        if "/" not in fragment and len(fragment) < 3:
            return self._get_top_level_paths()
        return self._get_deep_paths()

    def _get_top_level_paths(self) -> list[str]:
        now = time.monotonic()
        if now - self._top_cache_time <= self._refresh_interval:
            return self._top_cached_paths

        entries: list[str] = []
        try:
            for entry in sorted(self._root.iterdir(), key=lambda p: p.name):
                name = entry.name
                if self._is_ignored(name):
                    continue
                entries.append(f"{name}/" if entry.is_dir() else name)
                if len(entries) >= self._limit:
                    break
        except OSError:
            return self._top_cached_paths

        self._top_cached_paths = entries
        self._top_cache_time = now
        return self._top_cached_paths

    def _get_deep_paths(self) -> list[str]:
        now = time.monotonic()
        if now - self._cache_time <= self._refresh_interval:
            return self._cached_paths

        paths: list[str] = []
        try:
            for current_root, dirs, files in os.walk(self._root):
                relative_root = Path(current_root).relative_to(self._root)

                # Prevent descending into ignored directories.
                dirs[:] = sorted(d for d in dirs if not self._is_ignored(d))

                if relative_root.parts and any(
                    self._is_ignored(part) for part in relative_root.parts
                ):
                    dirs[:] = []
                    continue

                if relative_root.parts:
                    paths.append(relative_root.as_posix() + "/")
                    if len(paths) >= self._limit:
                        break

                for file_name in sorted(files):
                    if self._is_ignored(file_name):
                        continue
                    relative = (relative_root / file_name).as_posix()
                    if not relative:
                        continue
                    paths.append(relative)
                    if len(paths) >= self._limit:
                        break

                if len(paths) >= self._limit:
                    break
        except OSError:
            return self._cached_paths

        self._cached_paths = paths
        self._cache_time = now
        return self._cached_paths

    @staticmethod
    def _extract_fragment(text: str) -> str | None:
        index = text.rfind("@")
        if index == -1:
            return None

        if index > 0:
            prev = text[index - 1]
            if prev.isalnum() or prev in LocalFileMentionCompleter._TRIGGER_GUARDS:
                return None

        fragment = text[index + 1 :]
        if not fragment:
            return ""

        if any(ch.isspace() for ch in fragment):
            return None

        return fragment

    def _is_completed_file(self, fragment: str) -> bool:
        candidate = fragment.rstrip("/")
        if not candidate:
            return False
        try:
            return (self._root / candidate).is_file()
        except OSError:
            return False

    @override
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        fragment = self._extract_fragment(document.text_before_cursor)
        if fragment is None:
            return
        if self._is_completed_file(fragment):
            return

        mention_doc = Document(text=fragment, cursor_position=len(fragment))
        self._fragment_hint = fragment
        try:
            # First, ask the fuzzy completer for candidates.
            candidates = list(self._fuzzy.get_completions(mention_doc, complete_event))

            # re-rank: prefer basename matches
            frag_lower = fragment.lower()

            def _rank(c: Completion) -> tuple[int, ...]:
                path = c.text
                base = path.rstrip("/").split("/")[-1].lower()
                if base.startswith(frag_lower):
                    cat = 0
                elif frag_lower in base:
                    cat = 1
                else:
                    cat = 2
                # preserve original FuzzyCompleter's order in the same category
                return (cat,)

            candidates.sort(key=_rank)
            yield from candidates
        finally:
            self._fragment_hint = None


class _HistoryEntry(BaseModel):
    content: str


def _load_history_entries(history_file: Path) -> list[_HistoryEntry]:
    entries: list[_HistoryEntry] = []
    if not history_file.exists():
        return entries

    try:
        with history_file.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse user history line; skipping: {line}",
                        line=line,
                    )
                    continue
                try:
                    entry = _HistoryEntry.model_validate(record)
                    entries.append(entry)
                except ValidationError:
                    logger.warning(
                        "Failed to validate user history entry; skipping: {line}",
                        line=line,
                    )
                    continue
    except OSError as exc:
        logger.warning(
            "Failed to load user history file: {file} ({error})",
            file=history_file,
            error=exc,
        )

    return entries


class PromptMode(Enum):
    AGENT = "agent"
    SHELL = "shell"

    def toggle(self) -> PromptMode:
        return PromptMode.SHELL if self == PromptMode.AGENT else PromptMode.AGENT

    def __str__(self) -> str:
        return self.value


class UserInput(BaseModel):
    mode: PromptMode
    command: str
    """The plain text representation of the user input."""
    content: list[ContentPart]
    """The rich content parts."""

    def __str__(self) -> str:
        return self.command

    def __bool__(self) -> bool:
        return bool(self.command)


@dataclass(frozen=True, slots=True)
class TurnSubmitResult:
    accepted: bool
    persist_history: bool = False
    feedback: str = ""

    @classmethod
    def accept(cls, *, persist_history: bool = False) -> TurnSubmitResult:
        return cls(accepted=True, persist_history=persist_history)

    @classmethod
    def reject(cls, feedback: str = "") -> TurnSubmitResult:
        return cls(accepted=False, feedback=feedback)


@dataclass(frozen=True, slots=True)
class InputBoxState:
    active: bool = False
    mode: Literal["default", "reminder", "approval", "question", "question_other"] = "default"
    hint: str = ""


_REFRESH_INTERVAL = 1.0


@dataclass(slots=True)
class _ToastEntry:
    topic: str | None
    """There can be only one toast of each non-None topic in the queue."""
    message: str
    expires_at: float


_toast_queues: dict[Literal["left", "right"], deque[_ToastEntry]] = {
    "left": deque(),
    "right": deque(),
}
"""The queue of toasts to show, including the one currently being shown (the first one)."""


def toast(
    message: str,
    duration: float = 5.0,
    topic: str | None = None,
    immediate: bool = False,
    position: Literal["left", "right"] = "left",
) -> None:
    queue = _toast_queues[position]
    duration = max(duration, _REFRESH_INTERVAL)
    entry = _ToastEntry(topic=topic, message=message, expires_at=time.monotonic() + duration)
    if topic is not None:
        # Remove existing toasts with the same topic
        for existing in list(queue):
            if existing.topic == topic:
                queue.remove(existing)
    if immediate:
        queue.appendleft(entry)
    else:
        queue.append(entry)


def _prune_toasts(position: Literal["left", "right"]) -> None:
    queue = _toast_queues[position]
    now = time.monotonic()
    while queue and queue[0].expires_at <= now:
        queue.popleft()


def _current_toast(position: Literal["left", "right"] = "left") -> _ToastEntry | None:
    _prune_toasts(position)
    queue = _toast_queues[position]
    if not queue:
        return None
    return queue[0]


def _build_toolbar_tips(clipboard_available: bool) -> list[str]:
    tips = [
        "ctrl-x: toggle mode",
        "shift-tab: plan mode",
        "ctrl-o: editor",
        "ctrl-j: newline",
    ]
    if clipboard_available:
        tips.append("ctrl-v: paste media")
    tips.append("@: mention files")
    return tips


_TIP_SEPARATOR = " | "


_ATTACHMENT_PLACEHOLDER_RE = re.compile(
    r"\[(?P<type>[a-zA-Z0-9_\-]+):(?P<id>[a-zA-Z0-9_\-\.]+)"
    r"(?:,(?P<width>\d+)x(?P<height>\d+))?\]"
)


def _guess_image_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    # fallback to PNG
    return "image/png"


def _build_image_part(image_bytes: bytes, mime_type: str) -> ImageURLPart:
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    return ImageURLPart(
        image_url=ImageURLPart.ImageURL(
            url=f"data:{mime_type};base64,{image_base64}",
        )
    )


type CachedAttachmentKind = Literal["image"]


@dataclass(slots=True)
class CachedAttachment:
    kind: CachedAttachmentKind
    attachment_id: str
    path: Path


class AttachmentCache:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path("/tmp/kimi")
        self._dir_map: dict[CachedAttachmentKind, str] = {"image": "images"}
        self._payload_map: dict[tuple[CachedAttachmentKind, str, str], CachedAttachment] = {}

    def _dir_for(self, kind: CachedAttachmentKind) -> Path:
        return self._root / self._dir_map[kind]

    def _ensure_dir(self, kind: CachedAttachmentKind) -> Path | None:
        path = self._dir_for(kind)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to create attachment cache dir: {dir} ({error})",
                dir=path,
                error=exc,
            )
            return None
        return path

    def _reserve_id(self, dir_path: Path, suffix: str) -> str:
        for _ in range(5):
            candidate = f"{random_string(8)}{suffix}"
            if not (dir_path / candidate).exists():
                return candidate
        return f"{random_string(12)}{suffix}"

    def store_bytes(
        self, kind: CachedAttachmentKind, suffix: str, payload: bytes
    ) -> CachedAttachment | None:
        dir_path = self._ensure_dir(kind)
        if dir_path is None:
            return None
        payload_hash = sha256(payload).hexdigest()
        cache_key = (kind, suffix, payload_hash)
        cached = self._payload_map.get(cache_key)
        if cached is not None:
            if cached.path.exists():
                return cached
            self._payload_map.pop(cache_key, None)

        attachment_id = self._reserve_id(dir_path, suffix)
        path = dir_path / attachment_id
        try:
            path.write_bytes(payload)
        except OSError as exc:
            logger.warning(
                "Failed to write cached attachment: {file} ({error})",
                file=path,
                error=exc,
            )
            return None
        cached = CachedAttachment(kind=kind, attachment_id=attachment_id, path=path)
        self._payload_map[cache_key] = cached
        return cached

    def store_image(self, image: Image.Image) -> CachedAttachment | None:
        png_bytes = BytesIO()
        image.save(png_bytes, format="PNG")
        return self.store_bytes("image", ".png", png_bytes.getvalue())

    def load_bytes(
        self, kind: CachedAttachmentKind, attachment_id: str
    ) -> tuple[Path, bytes] | None:
        path = self._dir_for(kind) / attachment_id
        if not path.exists():
            return None
        try:
            return path, path.read_bytes()
        except OSError as exc:
            logger.warning(
                "Failed to read cached attachment: {file} ({error})",
                file=path,
                error=exc,
            )
            return None

    def load_content_parts(
        self, kind: CachedAttachmentKind, attachment_id: str
    ) -> list[ContentPart] | None:
        if kind == "image":
            payload = self.load_bytes(kind, attachment_id)
            if payload is None:
                return None
            path, image_bytes = payload
            mime_type = _guess_image_mime(path)
            part = _build_image_part(image_bytes, mime_type)
            return wrap_media_part(part, tag="image", attrs={"path": str(path)})
        return None


def _parse_attachment_kind(raw_kind: str) -> CachedAttachmentKind | None:
    if raw_kind == "image":
        return "image"
    return None


def _sanitize_surrogates(text: str) -> str:
    """Sanitize UTF-16 surrogate characters that cannot be encoded to UTF-8.

    This is particularly common on Windows when copying text from applications
    that use UTF-16 internally and don't properly convert surrogate pairs.
    """
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


class CustomPromptSession:
    def __init__(
        self,
        *,
        status_provider: Callable[[], StatusSnapshot],
        model_capabilities: set[ModelCapability],
        model_name: str | None,
        thinking: bool,
        agent_mode_slash_commands: Sequence[SlashCommand[Any]],
        shell_mode_slash_commands: Sequence[SlashCommand[Any]],
        editor_command_provider: Callable[[], str] = lambda: "",
        plan_mode_toggle_callback: Callable[[], Awaitable[bool]] | None = None,
        input_box_state_provider: Callable[[], InputBoxState] | None = None,
    ) -> None:
        history_dir = get_share_dir() / "user-history"
        history_dir.mkdir(parents=True, exist_ok=True)
        work_dir_id = md5(str(KaosPath.cwd()).encode(encoding="utf-8")).hexdigest()
        self._history_file = (history_dir / work_dir_id).with_suffix(".jsonl")
        self._status_provider = status_provider
        self._editor_command_provider = editor_command_provider
        self._plan_mode_toggle_callback = plan_mode_toggle_callback
        self._input_box_state_provider = input_box_state_provider or (lambda: InputBoxState())
        self._model_capabilities = model_capabilities
        self._model_name = model_name
        self._last_history_content: str | None = None
        self._mode: PromptMode = PromptMode.AGENT
        self._thinking = thinking
        self._attachment_cache = AttachmentCache()
        self._tip_rotation_index: int = 0
        clipboard_available = is_clipboard_available()
        self._tips = _build_toolbar_tips(clipboard_available)

        history_entries = _load_history_entries(self._history_file)
        history = InMemoryHistory()
        for entry in history_entries:
            history.append_string(entry.content)
        self._history = history

        if history_entries:
            # for consecutive deduplication
            self._last_history_content = history_entries[-1].content

        # Build completers
        # Slash commands are only available at the top-level prompt. During an active
        # turn, the input box is reserved for reminders / approval answers, so keep
        # slash commands out of that UI entirely.
        self._file_mention_completer = LocalFileMentionCompleter(
            KaosPath.cwd().unsafe_to_local_path()
        )
        self._agent_mode_completer = merge_completers(
            [
                SlashCommandCompleter(agent_mode_slash_commands),
                # TODO(kaos): we need an async KaosFileMentionCompleter
                self._file_mention_completer,
            ],
            deduplicate=True,
        )
        self._turn_mode_completer = self._file_mention_completer
        self._shell_mode_completer = SlashCommandCompleter(shell_mode_slash_commands)

        # Build key bindings
        _kb = KeyBindings()

        @_kb.add("enter", filter=has_completions)
        def _(event: KeyPressEvent) -> None:
            """Accept the first completion when Enter is pressed and completions are shown."""
            buff = event.current_buffer
            if buff.complete_state and buff.complete_state.completions:
                # Get the current completion, or use the first one if none is selected
                completion = buff.complete_state.current_completion
                if not completion:
                    completion = buff.complete_state.completions[0]
                buff.apply_completion(completion)

        @_kb.add("c-x", eager=True)
        def _(event: KeyPressEvent) -> None:
            self._mode = self._mode.toggle()
            # Apply mode-specific settings
            self._apply_mode(event)
            # Redraw UI
            event.app.invalidate()

        @_kb.add("s-tab", eager=True)
        def _(event: KeyPressEvent) -> None:
            """Toggle plan mode with Shift+Tab."""
            if self._plan_mode_toggle_callback is not None:

                async def _toggle() -> None:
                    assert self._plan_mode_toggle_callback is not None
                    new_state = await self._plan_mode_toggle_callback()
                    if new_state:
                        toast("plan mode ON", topic="plan_mode", duration=3.0, immediate=True)
                    else:
                        toast("plan mode OFF", topic="plan_mode", duration=3.0, immediate=True)
                    event.app.invalidate()

                event.app.create_background_task(_toggle())
            event.app.invalidate()

        @_kb.add("escape", "enter", eager=True)
        @_kb.add("c-j", eager=True)
        def _(event: KeyPressEvent) -> None:
            """Insert a newline when Alt-Enter or Ctrl-J is pressed."""
            event.current_buffer.insert_text("\n")

        @_kb.add("c-o", eager=True)
        def _(event: KeyPressEvent) -> None:
            """Open current buffer in external editor."""
            self._open_in_external_editor(event)

        if clipboard_available:

            @_kb.add("c-v", eager=True)
            def _(event: KeyPressEvent) -> None:
                if self._try_paste_media(event):
                    return
                clipboard_data = event.app.clipboard.get_data()
                if clipboard_data is None:  # type: ignore[reportUnnecessaryComparison]
                    return
                event.current_buffer.paste_clipboard_data(clipboard_data)

            clipboard = PyperclipClipboard()
        else:
            clipboard = None

        self._clipboard = clipboard
        self._base_key_bindings = _kb

        self._session = PromptSession[str](
            message=self._render_message,
            prompt_continuation=self._render_prompt_continuation,
            placeholder=self._render_placeholder,
            rprompt=self._render_rprompt,
            cursor=STEADY_INPUT_CURSOR,
            completer=self._agent_mode_completer,
            complete_while_typing=True,
            key_bindings=_kb,
            clipboard=clipboard,
            history=history,
            bottom_toolbar=self._render_bottom_toolbar,
            style=Style.from_dict({"bottom-toolbar": "noreverse"}),
        )
        # PromptSession defaults to polling terminal size every 0.5s, which causes
        # needless redraws/flicker in our persistent TUI input box.
        self._session.app.terminal_size_polling_interval = None

        # Allow completion to be triggered when the text is changed,
        # such as when backspace is used to delete text.
        @self._session.default_buffer.on_text_changed.add_handler
        def _(buffer: Buffer) -> None:
            if buffer.complete_while_typing():
                buffer.start_completion()

        self._status_refresh_task: asyncio.Task[None] | None = None

    def _render_message(self) -> FormattedText:
        if self._mode == PromptMode.SHELL:
            return FormattedText([("bold", f"{PROMPT_SYMBOL_SHELL} ")])

        input_box_state = self._input_box_state_provider()
        if input_box_state.active:
            return self._render_active_input_box_message(input_box_state)

        status = self._status_provider()
        if status.plan_mode:
            return FormattedText([("fg:#00aaff", f"{PROMPT_SYMBOL_PLAN} ")])
        symbol = PROMPT_SYMBOL_THINKING if self._thinking else PROMPT_SYMBOL
        return FormattedText([("", f"{symbol} ")])

    @staticmethod
    def _input_box_appearance(state: InputBoxState) -> tuple[str, str, str]:
        appearances = {
            "reminder": ("REMINDER", "fg:#38bdf8 bold", "bg:#1d4ed8 #ffffff bold"),
            "approval": ("APPROVAL", "fg:#f59e0b bold", "bg:#f59e0b #111827 bold"),
            "question": ("QUESTION", "fg:#22d3ee bold", "bg:#0891b2 #ffffff bold"),
            "question_other": ("CUSTOM ANSWER", "fg:#f472b6 bold", "bg:#db2777 #ffffff bold"),
        }
        return appearances.get(state.mode, ("INPUT", "fg:#9ca3af bold", "bg:#374151 #ffffff bold"))

    @staticmethod
    def _truncate_text(text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len == 1:
            return "…"
        return text[: max_len - 1] + "…"

    def _render_active_input_box_message(self, state: InputBoxState) -> FormattedText:
        label, border_style, badge_style = self._input_box_appearance(state)
        app = get_app_or_none()
        columns = app.output.get_size().columns if app is not None else 80
        badge = f" {label} "
        used = 3 + len(badge) + 1  # ╭─ + badge + ╮
        filler = "─" * max(1, columns - used)
        return FormattedText(
            [
                (border_style, "╭─"),
                (badge_style, badge),
                (border_style, f"{filler}╮\n│ "),
            ]
        )

    def _render_prompt_continuation(
        self,
        width: int,
        line_number: int,
        wrap_count: int,
    ) -> FormattedText:
        del width, line_number, wrap_count
        state = self._input_box_state_provider()
        if state.active and self._mode != PromptMode.SHELL:
            _, border_style, _ = self._input_box_appearance(state)
            return FormattedText([(border_style, "│ ")])
        return FormattedText([("fg:#4d4d4d", "… ")])

    def _render_placeholder(self) -> FormattedText | str:
        state = self._input_box_state_provider()
        if not state.active or self._mode == PromptMode.SHELL:
            return ""
        hint = self._truncate_text(state.hint, 120)
        if not hint:
            return ""
        return FormattedText([("fg:#22d3ee italic", hint)])

    def _render_rprompt(self) -> FormattedText | str:
        state = self._input_box_state_provider()
        if not state.active or self._mode == PromptMode.SHELL:
            return ""
        _, border_style, _ = self._input_box_appearance(state)
        return FormattedText([(border_style, "│")])

    @staticmethod
    def _render_prompt_title() -> FormattedText:
        border_style = "fg:#38bdf8 bold"
        badge_style = "bg:#2563eb #ffffff bold"
        return FormattedText([(border_style, "─"), (badge_style, " PROMPT "), (border_style, "─")])

    def _render_frame_title(self) -> FormattedText:
        return self._render_prompt_title()

    def _render_textarea_prompt(self) -> FormattedText | str:
        if self._mode == PromptMode.SHELL:
            return FormattedText([("bold", f"{PROMPT_SYMBOL_SHELL} ")])
        return ""

    def _show_agent_input_frame(self) -> bool:
        return self._mode == PromptMode.AGENT

    @staticmethod
    def _should_route_live_navigation(live_view: Any, buffer_text: str) -> bool:
        if not live_view.has_pending_input_request:
            return False
        if live_view.input_mode == "question_other":
            return False
        return not buffer_text.strip()

    def _render_hint_line(self, text_area: TextArea) -> FormattedText | str:
        if self._mode != PromptMode.AGENT:
            return ""
        if text_area.buffer.text:
            return ""
        state = self._input_box_state_provider()
        if state.mode == "reminder":
            return ""
        hint = state.hint.strip()
        if not hint:
            return ""
        hint = self._truncate_text(hint, 160)
        return FormattedText([("fg:#22d3ee italic", hint)])

    def _render_footer_line(self) -> FormattedText:
        status = self._status_provider()
        mode_text = self._mode_text(status)
        right_text = self._render_right_span(status)
        app = get_app_or_none()
        columns = app.output.get_size().columns if app is not None else 80
        left_text = mode_text
        padding = max(1, columns - len(left_text) - len(right_text))
        return FormattedText(
            [
                ("fg:#38bdf8 bold", left_text),
                ("", " " * padding),
                ("fg:#9ca3af", right_text),
            ]
        )

    @staticmethod
    def _with_completion_menu(content):
        return FloatContainer(
            content=content,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                )
            ],
        )

    def _build_prompt_application(self) -> tuple[Application[str], TextArea]:
        text_area = TextArea(
            text="",
            multiline=True,
            completer=(
                self._agent_mode_completer
                if self._mode == PromptMode.AGENT
                else self._shell_mode_completer
            ),
            complete_while_typing=True,
            history=self._history,
            prompt=self._render_textarea_prompt,
            wrap_lines=True,
            height=Dimension(min=1, max=6),
            dont_extend_height=True,
        )

        @text_area.buffer.on_text_changed.add_handler
        def _(buffer: Buffer) -> None:
            if buffer.complete_while_typing():
                buffer.start_completion()
            app = get_app_or_none()
            if app is not None:
                app.invalidate()

        hint_window = Window(
            FormattedTextControl(lambda: self._render_hint_line(text_area)),
            height=1,
            dont_extend_height=True,
        )
        footer_window = Window(
            FormattedTextControl(self._render_footer_line),
            height=1,
            dont_extend_height=True,
        )
        spacer = Window(height=Dimension(weight=1), char=" ")
        agent_container = HSplit(
            [
                spacer,
                Frame(
                    HSplit(
                        [
                            ConditionalContainer(
                                hint_window,
                                filter=Condition(
                                    lambda: self._show_agent_input_frame()
                                    and bool(self._render_hint_line(text_area))
                                ),
                            ),
                            text_area,
                        ]
                    ),
                    title=self._render_frame_title,
                    style="fg:#38bdf8",
                ),
                footer_window,
            ]
        )
        shell_container = HSplit([spacer, text_area])
        container = self._with_completion_menu(
            DynamicContainer(
                lambda: agent_container if self._show_agent_input_frame() else shell_container
            )
        )

        accept_kb = KeyBindings()

        @accept_kb.add("enter", filter=~has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            event.app.exit(result=text_area.buffer.text)

        @accept_kb.add("c-d", eager=True)
        def _(event: KeyPressEvent) -> None:
            buffer = event.current_buffer
            if not buffer.text:
                event.app.exit(exception=EOFError)
                return
            buffer.delete()

        @accept_kb.add("c-c", eager=True)
        def _(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt)

        app = Application[str](
            layout=Layout(container, focused_element=text_area),
            key_bindings=merge_key_bindings([self._base_key_bindings, accept_kb]),
            clipboard=self._clipboard,
            cursor=STEADY_INPUT_CURSOR,
            style=Style.from_dict({"frame.border": "fg:#38bdf8", "frame.label": "bold"}),
            full_screen=False,
            erase_when_done=True,
            refresh_interval=None,
            terminal_size_polling_interval=None,
        )
        return app, text_area

    def _open_in_external_editor(self, event: KeyPressEvent) -> None:
        """Open the current buffer content in an external editor."""
        from prompt_toolkit.application.run_in_terminal import run_in_terminal

        from kimi_cli.utils.editor import edit_text_in_editor, get_editor_command

        configured = self._editor_command_provider()

        if get_editor_command(configured) is None:
            toast("No editor found. Set $VISUAL/$EDITOR or run /editor.")
            return

        buff = event.current_buffer
        original_text = buff.text

        async def _run_editor() -> None:
            result = await run_in_terminal(
                lambda: edit_text_in_editor(original_text, configured), in_executor=True
            )
            if result is not None:
                buff.document = Document(text=result, cursor_position=len(result))

        event.app.create_background_task(_run_editor())

    def _open_live_view_expansion(self, event: KeyPressEvent, live_view: Any) -> None:
        """Open the current approval/question body pager without fighting the PTK app."""
        from prompt_toolkit.application.run_in_terminal import run_in_terminal

        async def _run_pager() -> None:
            await run_in_terminal(live_view.show_more)
            event.app.invalidate()

        event.app.create_background_task(_run_pager())

    def _apply_mode(self, event: KeyPressEvent | None = None) -> None:
        # Apply mode to the active buffer (not the PromptSession itself)
        try:
            buff = event.current_buffer if event is not None else self._session.default_buffer
        except Exception:
            buff = None

        if self._mode == PromptMode.SHELL:
            if buff is not None:
                buff.completer = self._shell_mode_completer
        else:
            if buff is not None:
                buff.completer = self._agent_mode_completer

    def _render_signature(self) -> tuple[object, ...]:
        status = self._status_provider()
        input_box_state = self._input_box_state_provider()
        left_toast = _current_toast("left")
        right_toast = _current_toast("right")
        return (
            self._mode,
            self._thinking,
            self._model_name,
            input_box_state,
            status.context_usage,
            status.context_tokens,
            status.max_context_tokens,
            status.yolo_enabled,
            status.plan_mode,
            left_toast.message if left_toast is not None else None,
            right_toast.message if right_toast is not None else None,
        )

    @staticmethod
    def _live_indicator_frame(kind: str, *, now: float | None = None) -> str:
        frames = _INDICATOR_FRAMES.get(kind, _PAUSE_FRAMES)
        current = time.monotonic() if now is None else now
        return frames[int(current * 10) % len(frames)]

    @staticmethod
    def _force_turn_full_repaint(app: Application[Any]) -> None:
        app.renderer._last_screen = None  # type: ignore[reportPrivateUsage]
        app.invalidate()

    @staticmethod
    def _target_turn_body_bottom_scroll(
        body_window: Window,
        *,
        line_count: int,
        current_scroll: int,
    ) -> int | None:
        render_info = body_window.render_info
        if render_info is None:
            return None
        visible_height = max(1, render_info.window_height)
        target_scroll = max(0, line_count - visible_height)
        if current_scroll == target_scroll:
            return None
        return target_scroll

    @classmethod
    def _refresh_turn_application(
        cls,
        app: Application[Any],
        *,
        live_view: Any,
        last_full_repaint_at: float | None,
        now: float | None = None,
    ) -> float | None:
        if not getattr(live_view, "needs_periodic_refresh", False):
            return None
        current = time.monotonic() if now is None else now
        if last_full_repaint_at is None:
            app.invalidate()
            return current
        if current - last_full_repaint_at >= _TURN_UI_FULL_REPAINT_INTERVAL:
            cls._force_turn_full_repaint(app)
            return current
        app.invalidate()
        return last_full_repaint_at

    def _format_live_activity_status(self, live_view: Any) -> tuple[str, str] | None:
        indicator = getattr(live_view, "activity_indicator", None)
        if indicator is None:
            return None
        kind, text = indicator
        if not text:
            return None
        frame = self._live_indicator_frame(kind)
        return f"{frame} {text}", _INDICATOR_STYLES.get(kind, "fg:#22c55e")

    def _render_turn_activity(self, live_view: Any) -> FormattedText | str:
        live_status = self._format_live_activity_status(live_view)
        if live_status is None:
            return ""
        status_text, status_style = live_status
        return FormattedText(
            [
                ("fg:#6b7280", "│ "),
                (status_style, status_text),
            ]
        )

    def _render_turn_footer(
        self,
        columns: int,
        *,
        status: StatusSnapshot,
    ) -> FormattedText:
        mode_text = self._mode_text(status)
        right_text = self._render_right_span(status)
        fragments: list[tuple[str, str]] = [("fg:#38bdf8 bold", mode_text)]
        remaining = columns - len(mode_text) - len(right_text)
        fragments.append(("", " " * max(1, remaining)))
        fragments.append(("fg:#9ca3af", right_text))
        return FormattedText(fragments)

    def __enter__(self) -> CustomPromptSession:
        if self._status_refresh_task is not None and not self._status_refresh_task.done():
            return self

        async def _refresh(interval: float) -> None:
            last_signature: tuple[object, ...] | None = None
            try:
                while True:
                    app = get_app_or_none()
                    if app is not None:
                        signature = self._render_signature()
                        if signature != last_signature:
                            app.invalidate()
                            last_signature = signature

                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        logger.warning("No running loop found, exiting status refresh task")
                        self._status_refresh_task = None
                        break

                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                # graceful exit
                pass

        self._status_refresh_task = asyncio.create_task(_refresh(_REFRESH_INTERVAL))
        return self

    def __exit__(self, *_) -> None:
        if self._status_refresh_task is not None and not self._status_refresh_task.done():
            self._status_refresh_task.cancel()
        self._status_refresh_task = None

    def _try_paste_media(self, event: KeyPressEvent) -> bool:
        """Try to paste media from the clipboard.

        Reads the clipboard once and handles all detected content:
        non-image files (videos, PDFs, etc.) are inserted as paths,
        image files are cached and inserted as placeholders.
        Returns True if any media was detected.
        """
        result = grab_media_from_clipboard()
        if result is None:
            return False

        parts: list[str] = []

        # 1. Insert file paths (videos, PDFs, etc.)
        if result.file_paths:
            logger.debug("Pasted {count} file path(s) from clipboard", count=len(result.file_paths))
            for p in result.file_paths:
                text = str(p)
                if self._mode == PromptMode.SHELL:
                    text = shlex.quote(text)
                parts.append(text)

        # 2. Insert images via cache.
        if result.images:
            if "image_in" not in self._model_capabilities:
                console.print(
                    "[yellow]Image input is not supported by the selected LLM model[/yellow]"
                )
            else:
                for image in result.images:
                    cached = self._attachment_cache.store_image(image)
                    if cached is None:
                        continue
                    logger.debug(
                        "Pasted image from clipboard: {attachment_id}, {image_size}",
                        attachment_id=cached.attachment_id,
                        image_size=image.size,
                    )
                    parts.append(f"[image:{cached.attachment_id},{image.width}x{image.height}]")

        if parts:
            event.current_buffer.insert_text(" ".join(parts))
        event.app.invalidate()
        return bool(parts)

    def parse_user_input(self, command: str) -> UserInput:
        command = command.replace("\x00", "")
        command = _sanitize_surrogates(command)

        content: list[ContentPart] = []
        remaining_command = command
        while match := _ATTACHMENT_PLACEHOLDER_RE.search(remaining_command):
            start, end = match.span()
            if start > 0:
                content.append(TextPart(text=remaining_command[:start]))
            attachment_id = match.group("id")
            attachment_kind = _parse_attachment_kind(match.group("type"))
            part = None
            if attachment_kind is not None:
                part = self._attachment_cache.load_content_parts(attachment_kind, attachment_id)
            if part is not None:
                content.extend(part)
            else:
                logger.warning(
                    "Attachment placeholder found but no matching attachment part: {placeholder}",
                    placeholder=match.group(0),
                )
                content.append(TextPart(text=match.group(0)))
            remaining_command = remaining_command[end:]

        if remaining_command:
            content.append(TextPart(text=remaining_command))

        return UserInput(mode=self._mode, content=content, command=command)

    async def prompt(self) -> UserInput:
        app, _ = self._build_prompt_application()
        with patch_stdout(raw=True):
            command = str(await app.run_async()).strip()
        self._append_history_entry(command)
        self._tip_rotation_index += 1
        return self.parse_user_input(command)

    async def run_turn_ui(
        self,
        *,
        wire: Any,
        live_view: Any,
        submit_handler: Callable[[UserInput], TurnSubmitResult],
        cancel_handler: Callable[[], None],
    ) -> None:
        self._mode = PromptMode.AGENT
        feedback_message = ""
        body_vertical_scroll = 0
        body_window: Window | None = None
        reminder_window: Window | None = None
        last_layout_signature: tuple[object, ...] | None = None

        def _app_columns(app: Application[Any] | None = None) -> int:
            current_app = get_app_or_none() if app is None else app
            return current_app.output.get_size().columns if current_app is not None else 80

        text_area = TextArea(
            text="",
            multiline=True,
            completer=self._turn_mode_completer,
            complete_while_typing=True,
            history=self._history,
            wrap_lines=True,
            height=Dimension(min=1, max=6),
            dont_extend_height=True,
        )

        def _input_box_height() -> int:
            columns = max(1, _app_columns() - 4)
            return text_area.window.preferred_height(columns, 6).preferred

        @text_area.buffer.on_text_changed.add_handler
        def _(buffer: Buffer) -> None:
            if buffer.complete_while_typing():
                buffer.start_completion()
            app = get_app_or_none()
            if app is not None:
                _redraw_turn_view(app)

        def _render_title() -> FormattedText:
            return self._render_prompt_title()

        def _render_hint() -> FormattedText | str:
            if text_area.buffer.text:
                return ""
            if live_view.input_mode == "reminder":
                return ""
            hint = feedback_message or live_view.input_hint
            if not hint:
                return ""
            hint = self._truncate_text(hint, 160)
            return FormattedText([("fg:#22d3ee italic", hint)])

        def _render_footer() -> FormattedText:
            status = self._status_provider()
            return self._render_turn_footer(_app_columns(), status=status)

        body_control = _RichRenderableControl(
            lambda: live_view.compose_body(
                include_running_indicators=False,
                include_sticky_reminders=False,
            )
        )
        reminder_control = _RichRenderableControl(live_view.compose_sticky_reminders)

        def _turn_layout_signature() -> tuple[object, ...]:
            body_width = (
                body_window.render_info.window_width
                if body_window is not None and body_window.render_info is not None
                else _app_columns()
            )
            reminder_width = (
                reminder_window.render_info.window_width
                if reminder_window is not None and reminder_window.render_info is not None
                else _app_columns()
            )
            input_height = (
                text_area.window.render_info.window_height
                if text_area.window.render_info is not None
                else _input_box_height()
            )
            return (
                getattr(live_view, "render_revision", 0),
                body_width,
                body_control.line_count(body_width),
                reminder_width,
                reminder_control.line_count(reminder_width),
                live_view.has_sticky_reminders,
                bool(self._render_turn_activity(live_view)),
                bool(_render_hint()),
                input_height,
            )

        def _redraw_turn_view(app: Application[Any]) -> None:
            nonlocal last_layout_signature
            signature = _turn_layout_signature()
            if last_layout_signature is None:
                last_layout_signature = signature
                app.invalidate()
                return
            if signature != last_layout_signature:
                last_layout_signature = signature
                self._force_turn_full_repaint(app)
                return
            app.invalidate()

        def _refresh_turn_view(app: Application[Any]) -> None:
            _redraw_turn_view(app)

        def _render_activity() -> FormattedText | str:
            return self._render_turn_activity(live_view)

        key_bindings = KeyBindings()
        route_live_navigation = Condition(
            lambda: self._should_route_live_navigation(live_view, text_area.buffer.text)
        )

        @key_bindings.add("enter", filter=has_completions)
        def _(event: KeyPressEvent) -> None:
            buff = event.current_buffer
            if buff.complete_state and buff.complete_state.completions:
                completion = buff.complete_state.current_completion
                if not completion:
                    completion = buff.complete_state.completions[0]
                buff.apply_completion(completion)

        @key_bindings.add("enter", filter=route_live_navigation & ~has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.ENTER)
            _refresh_turn_view(event.app)

        @key_bindings.add("up", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.UP)
            _refresh_turn_view(event.app)

        @key_bindings.add("down", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.DOWN)
            _refresh_turn_view(event.app)

        @key_bindings.add("left", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.LEFT)
            _refresh_turn_view(event.app)

        @key_bindings.add("right", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.RIGHT)
            _refresh_turn_view(event.app)

        @key_bindings.add("tab", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.TAB)
            _refresh_turn_view(event.app)

        @key_bindings.add("space", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.SPACE)
            _refresh_turn_view(event.app)

        @key_bindings.add("escape", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            live_view.dispatch_keyboard_event(KeyEvent.ESCAPE)
            _refresh_turn_view(event.app)

        @key_bindings.add("enter", filter=~route_live_navigation & ~has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            nonlocal feedback_message
            command = event.current_buffer.text.strip()
            if not command:
                return
            if live_view.has_pending_input_request and live_view.is_expand_command(command):
                if live_view.can_expand_current_panel:
                    feedback_message = ""
                    event.current_buffer.document = Document(text="", cursor_position=0)
                    self._open_live_view_expansion(event, live_view)
                else:
                    feedback_message = live_view.input_hint
                    _refresh_turn_view(event.app)
                return
            user_input = self.parse_user_input(command)
            submit_result = submit_handler(user_input)
            if submit_result.accepted:
                if submit_result.persist_history:
                    self._append_history_entry(command)
                    self._tip_rotation_index += 1
                feedback_message = ""
                event.current_buffer.document = Document(text="", cursor_position=0)
            else:
                feedback_message = submit_result.feedback or live_view.input_hint
            _refresh_turn_view(event.app)

        @key_bindings.add("escape", "enter", eager=True)
        @key_bindings.add("c-j", eager=True)
        def _(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-o", eager=True)
        def _(event: KeyPressEvent) -> None:
            self._open_in_external_editor(event)

        if self._clipboard is not None:

            @key_bindings.add("c-v", eager=True)
            def _(event: KeyPressEvent) -> None:
                if self._try_paste_media(event):
                    return
                clipboard_data = event.app.clipboard.get_data()
                if clipboard_data is None:  # type: ignore[reportUnnecessaryComparison]
                    return
                event.current_buffer.paste_clipboard_data(clipboard_data)

        @key_bindings.add("c-c", eager=True)
        def _(event: KeyPressEvent) -> None:
            cancel_handler()
            _refresh_turn_view(event.app)

        body_window = Window(
            body_control,
            always_hide_cursor=True,
            get_vertical_scroll=lambda _: body_vertical_scroll,
        )
        reminder_window = Window(reminder_control, dont_extend_height=True)
        activity_window = Window(
            FormattedTextControl(_render_activity),
            height=1,
            dont_extend_height=True,
        )
        hint_window = Window(
            FormattedTextControl(_render_hint),
            height=1,
            dont_extend_height=True,
        )
        footer_window = Window(
            FormattedTextControl(_render_footer),
            height=1,
            dont_extend_height=True,
        )
        container = self._with_completion_menu(
            HSplit(
                [
                    body_window,
                    ConditionalContainer(
                        reminder_window,
                        filter=Condition(lambda: live_view.has_sticky_reminders),
                    ),
                    ConditionalContainer(
                        activity_window,
                        filter=Condition(lambda: bool(self._render_turn_activity(live_view))),
                    ),
                    Frame(
                        HSplit(
                            [
                                ConditionalContainer(
                                    hint_window,
                                    filter=Condition(lambda: bool(_render_hint())),
                                ),
                                text_area,
                            ]
                        ),
                        title=_render_title,
                        style="fg:#38bdf8",
                    ),
                    footer_window,
                ]
            )
        )
        app = Application[None](
            layout=Layout(container, focused_element=text_area),
            key_bindings=key_bindings,
            clipboard=self._clipboard,
            cursor=STEADY_INPUT_CURSOR,
            style=Style.from_dict({"frame.border": "fg:#38bdf8", "frame.label": "bold"}),
            full_screen=False,
            erase_when_done=True,
            refresh_interval=None,
            terminal_size_polling_interval=None,
        )
        last_layout_signature = _turn_layout_signature()

        def _follow_turn_output(_: Application[None]) -> None:
            nonlocal body_vertical_scroll, last_layout_signature
            signature = _turn_layout_signature()
            target_scroll = self._target_turn_body_bottom_scroll(
                body_window,
                line_count=body_control.line_count(signature[0]),
                current_scroll=body_vertical_scroll,
            )
            if target_scroll is not None:
                body_vertical_scroll = target_scroll
                last_layout_signature = signature
                self._force_turn_full_repaint(app)
                return
            if signature != last_layout_signature:
                last_layout_signature = signature
                self._force_turn_full_repaint(app)

        app.after_render += _follow_turn_output

        async def _consume_wire() -> None:
            nonlocal feedback_message
            while True:
                try:
                    msg = await wire.receive()
                except QueueShutDown:
                    live_view.cleanup(is_interrupt=False)
                    live_view.finish_turn()
                    _refresh_turn_view(app)
                    app.exit()
                    return

                if isinstance(msg, StepInterrupted):
                    live_view.cleanup(is_interrupt=True)
                    live_view.finish_turn()
                    _refresh_turn_view(app)
                    app.exit()
                    return

                live_view.dispatch_wire_message(msg)
                feedback_message = ""
                _refresh_turn_view(app)

        async def _animate() -> None:
            last_full_repaint_at: float | None = None
            while True:
                await asyncio.sleep(_TURN_UI_REFRESH_INTERVAL)
                last_full_repaint_at = self._refresh_turn_application(
                    app,
                    live_view=live_view,
                    last_full_repaint_at=last_full_repaint_at,
                )

        consume_task = asyncio.create_task(_consume_wire())
        animate_task = asyncio.create_task(_animate())
        try:
            with patch_stdout(raw=True):
                await app.run_async()
        finally:
            consume_task.cancel()
            animate_task.cancel()
            with suppress(asyncio.CancelledError):
                await consume_task
            with suppress(asyncio.CancelledError):
                await animate_task

        final_output = live_view.render_ansi(
            app.output.get_size().columns,
            include_status=False,
            include_running_indicators=False,
        ).strip()
        if final_output:
            console.print(_rich_from_ansi(final_output), end="")

    def _append_history_entry(self, text: str) -> None:
        entry = _HistoryEntry(content=text.strip())
        if not entry.content:
            return

        # skip if same as last entry
        if entry.content == self._last_history_content:
            return

        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with self._history_file.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json(ensure_ascii=False) + "\n")
            self._last_history_content = entry.content
        except OSError as exc:
            logger.warning(
                "Failed to append user history entry: {file} ({error})",
                file=self._history_file,
                error=exc,
            )

    def _mode_text(self, status: StatusSnapshot) -> str:
        mode = str(self._mode).lower()
        if self._mode == PromptMode.AGENT:
            mode_details: list[str] = []
            if self._model_name:
                mode_details.append(self._model_name)
            if self._thinking:
                mode_details.append("thinking")
            if mode_details:
                mode += f" ({', '.join(mode_details)})"
        flags: list[str] = []
        if status.yolo_enabled:
            flags.append("yolo")
        if status.plan_mode:
            flags.append("plan")
        if flags:
            mode += f" [{' / '.join(flags)}]"
        return mode

    def _rotated_tips_text(self, available: int) -> str | None:
        full_text = _TIP_SEPARATOR.join(self._tips)
        if len(full_text) <= available:
            return full_text

        n = len(self._tips)
        offset = self._tip_rotation_index % n
        rotated = self._tips[offset:] + self._tips[:offset]
        selected: list[str] = []
        total_len = 0
        for tip in rotated:
            needed = len(tip) + (len(_TIP_SEPARATOR) if selected else 0)
            if total_len + needed <= available:
                selected.append(tip)
                total_len += needed
        return _TIP_SEPARATOR.join(selected) if selected else None

    def _render_active_bottom_toolbar(
        self,
        columns: int,
        state: InputBoxState,
        status: StatusSnapshot,
        right_text: str,
    ) -> FormattedText:
        _, border_style, _ = self._input_box_appearance(state)
        mode_text = self._mode_text(status)
        footer_text = f"╰─ {mode_text}"
        padding = max(1, columns - len(footer_text) - len(right_text))
        return FormattedText(
            [
                (border_style, footer_text),
                ("", " " * padding),
                ("fg:#9ca3af", right_text),
            ]
        )

    def _render_bottom_toolbar(self) -> FormattedText:
        app = get_app_or_none()
        assert app is not None
        columns = app.output.get_size().columns
        status = self._status_provider()
        right_text = self._render_right_span(status)
        input_box_state = self._input_box_state_provider()
        if input_box_state.active:
            return self._render_active_bottom_toolbar(columns, input_box_state, status, right_text)

        fragments: list[tuple[str, str]] = []
        fragments.append(("fg:#4d4d4d", "─" * columns))
        fragments.append(("", "\n"))

        mode_text = self._mode_text(status)
        fragments.extend([("", mode_text), ("", "  ")])
        columns -= len(mode_text) + 2

        current_toast_left = _current_toast("left")
        if current_toast_left is not None:
            left_text = current_toast_left.message
        else:
            left_text = self._rotated_tips_text(columns - len(right_text) - 3)

        if left_text:
            left_text = self._truncate_text(left_text, max(0, columns - len(right_text) - 3))
            fragments.extend([("", left_text), ("", "  ")])
            columns -= len(left_text) + 2

        padding = max(1, columns - len(right_text))
        fragments.append(("", " " * padding))
        fragments.append(("fg:#9ca3af", right_text))
        return FormattedText(fragments)

    @staticmethod
    def _render_right_span(status: StatusSnapshot) -> str:
        current_toast = _current_toast("right")
        if current_toast is not None:
            return current_toast.message
        return format_context_status(
            status.context_usage,
            status.context_tokens,
            status.max_context_tokens,
        )
