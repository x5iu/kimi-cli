from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast, override

from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    FuzzyCompleter,
    WordCompleter,
)
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.layout import HSplit
from prompt_toolkit.layout.containers import ConditionalContainer, FloatContainer
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.utils import get_cwidth

from kimi_cli.utils.slashcmd import SlashCommand


class SlashCommandCompleter(Completer):
    """
    A completer that:
    - Shows one line per slash command using the canonical "/name"
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

    @staticmethod
    def should_complete(document: Document) -> bool:
        """Return whether slash command completion should be active for the current buffer."""
        text = document.text_before_cursor

        if document.text_after_cursor.strip():
            return False

        last_space = text.rfind(" ")
        token = text[last_space + 1 :]
        prefix = text[: last_space + 1] if last_space != -1 else ""

        return not prefix.strip() and token.startswith("/")

    @staticmethod
    def _typed_token(document: Document) -> str:
        text = document.text_before_cursor
        last_space = text.rfind(" ")
        token = text[last_space + 1 :]
        return token[1:] if token.startswith("/") else ""

    def is_exact_match(self, document: Document) -> bool:
        if not self.should_complete(document):
            return False
        typed = self._typed_token(document)
        return bool(typed) and typed in self._command_lookup

    @override
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        if not self.should_complete(document):
            return
        text = document.text_before_cursor
        last_space = text.rfind(" ")
        token = text[last_space + 1 :]

        typed = self._typed_token(document)
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
                    display=f"/{cmd.name}",
                    display_meta=cmd.description,
                )


def _truncate_to_width(text: str, width: int) -> str:
    if width <= 0:
        return ""

    total = 0
    chars: list[str] = []
    for ch in text:
        ch_width = get_cwidth(ch)
        if total + ch_width > width:
            break
        chars.append(ch)
        total += ch_width

    if total == get_cwidth(text):
        return text + (" " * max(0, width - total))

    ellipsis = "..."
    ellipsis_width = get_cwidth(ellipsis)
    if width <= ellipsis_width:
        return "." * width

    available = width - ellipsis_width
    total = 0
    chars = []
    for ch in text:
        ch_width = get_cwidth(ch)
        if total + ch_width > available:
            break
        chars.append(ch)
        total += ch_width
    return "".join(chars) + ellipsis + (" " * max(0, width - total - ellipsis_width))


def _wrap_to_width(text: str, width: int, *, max_lines: int | None = None) -> list[str]:
    if width <= 0:
        return []

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_words: list[str] = []
    current_width = 0
    index = 0

    while index < len(words):
        word = words[index]
        word_width = get_cwidth(word)
        separator_width = 1 if current_words else 0

        if current_words and current_width + separator_width + word_width <= width:
            current_words.append(word)
            current_width += separator_width + word_width
            index += 1
            continue

        if not current_words and word_width <= width:
            current_words.append(word)
            current_width = word_width
            index += 1
            continue

        if not current_words and word_width > width:
            current_words.append(_truncate_to_width(word, width).rstrip())
            current_width = get_cwidth(current_words[0])
            index += 1

        lines.append(" ".join(current_words))
        current_words = []
        current_width = 0

        if max_lines is not None and len(lines) == max_lines:
            remaining = " ".join(words[index:])
            if remaining:
                prefix = f"{lines[-1]} " if lines[-1] else ""
                lines[-1] = _truncate_to_width(prefix + remaining, width).rstrip()
            return lines

    if current_words:
        line = " ".join(current_words)
        if max_lines is not None and len(lines) + 1 > max_lines:
            if lines:
                lines[-1] = _truncate_to_width(f"{lines[-1]} {line}", width).rstrip()
            else:
                lines.append(_truncate_to_width(line, width).rstrip())
        else:
            lines.append(line)

    return lines


def _find_prompt_float_container(layout_container: object) -> FloatContainer | None:
    if not isinstance(layout_container, HSplit):
        return None

    for child in cast(Sequence[object], layout_container.children):
        float_container = _extract_float_container(child)
        if float_container is not None:
            return float_container
    return None


def _extract_float_container(container: object) -> FloatContainer | None:
    if isinstance(container, FloatContainer):
        return container
    if isinstance(container, ConditionalContainer):
        if isinstance(container.content, FloatContainer):
            return container.content
        if isinstance(container.alternative_content, FloatContainer):
            return container.alternative_content
    return None


class SlashCommandMenuControl(UIControl):
    """Render slash command completions as a full-width menu that matches the shell UI."""

    _FIXED_MENU_HEIGHT = 10

    @staticmethod
    def _has_meta_column(completions: Sequence[Completion]) -> bool:
        return any(completion.display_meta_text.strip() for completion in completions)

    def __init__(
        self,
        *,
        left_padding: Callable[[], int],
        scroll_offset: int = 1,
    ) -> None:
        self._left_padding = left_padding
        self._scroll_offset = scroll_offset

    def has_focus(self) -> bool:
        return False

    def preferred_width(self, max_available_width: int) -> int | None:
        return max_available_width

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Callable[..., AnyFormattedText] | None,
    ) -> int | None:
        app = get_app_or_none()
        complete_state = (
            getattr(app.current_buffer, "complete_state", None) if app is not None else None
        )
        if complete_state is None:
            return 0
        return min(max_available_height, self._FIXED_MENU_HEIGHT)

    def create_content(self, width: int, height: int) -> UIContent:
        app = get_app_or_none()
        complete_state = (
            getattr(app.current_buffer, "complete_state", None) if app is not None else None
        )
        if complete_state is None or not complete_state.completions:
            return UIContent()

        completions = complete_state.completions
        selected_index = complete_state.complete_index
        separator_rows = 1 if height >= 2 else 0
        detail_rows = 1 if height >= 3 else 0
        available_rows = max(1, height - separator_rows - detail_rows)

        menu_width = max(0, width - self._left_padding())
        marker_width = 2
        has_meta = self._has_meta_column(completions)
        command_width = self._command_column_width(
            completions,
            menu_width,
            marker_width,
            has_meta=has_meta,
        )
        gap_width = 0 if not has_meta else (3 if menu_width > command_width + 6 else 1)
        meta_width = max(0, menu_width - marker_width - command_width - gap_width)

        rendered_lines: list[FormattedText] = []
        if separator_rows:
            rendered_lines.append(
                FormattedText([("class:slash-completion-menu.separator", "─" * max(0, width))])
            )
        selected_line_index = 0

        if selected_index is None:
            start, end = 0, min(len(completions) - 1, available_rows - 1)
        else:
            start, end = self._visible_window_bounds(
                completion_count=len(completions),
                selected_index=selected_index,
                available_rows=available_rows,
                selected_item_height=1,
            )
            selected_line_index = separator_rows + (selected_index - start)

        for index in range(start, end + 1):
            completion = completions[index]
            rendered_lines.append(
                self._render_single_line_item(
                    width=width,
                    completion=completion,
                    marker_width=marker_width,
                    command_width=command_width,
                    meta_width=meta_width,
                    gap_width=gap_width,
                    is_current=index == selected_index,
                )
            )

        item_line_target = separator_rows + available_rows
        while len(rendered_lines) < item_line_target:
            rendered_lines.append(FormattedText([("class:slash-completion-menu", " " * width)]))

        if detail_rows:
            detail_text = ""
            if selected_index is not None:
                detail_text = completions[selected_index].display_meta_text
            rendered_lines.append(self._render_detail_line(width=width, text=detail_text))

        return UIContent(
            get_line=lambda i: rendered_lines[i],
            line_count=len(rendered_lines),
            cursor_position=Point(x=0, y=max(0, selected_line_index)),
        )

    def _visible_window_bounds(
        self,
        *,
        completion_count: int,
        selected_index: int,
        available_rows: int,
        selected_item_height: int,
    ) -> tuple[int, int]:
        selected_item_height = min(selected_item_height, available_rows)
        remaining_rows = max(0, available_rows - selected_item_height)

        before = min(self._scroll_offset, selected_index, remaining_rows)
        remaining_rows -= before
        after = min(completion_count - selected_index - 1, remaining_rows)
        remaining_rows -= after

        extra_before = min(selected_index - before, remaining_rows)
        before += extra_before
        remaining_rows -= extra_before

        extra_after = min(completion_count - selected_index - 1 - after, remaining_rows)
        after += extra_after

        return selected_index - before, selected_index + after

    def _command_column_width(
        self,
        completions: Sequence[Completion],
        menu_width: int,
        marker_width: int,
        *,
        has_meta: bool,
    ) -> int:
        if menu_width <= 0:
            return 0
        usable_width = max(0, menu_width - marker_width)
        if not has_meta:
            return usable_width
        longest = max((get_cwidth(c.display_text) for c in completions), default=0)
        preferred = longest + 2
        minimum = min(usable_width, 18)
        maximum = max(minimum, min(28, usable_width // 2))
        return max(minimum, min(preferred, maximum))

    def _render_single_line_item(
        self,
        *,
        width: int,
        completion: Completion,
        marker_width: int,
        command_width: int,
        meta_width: int,
        gap_width: int,
        is_current: bool,
    ) -> FormattedText:
        padding_width = max(0, width - marker_width - command_width - meta_width - gap_width)
        left_padding = min(self._left_padding(), padding_width)
        trailing_width = max(
            0,
            width - left_padding - marker_width - command_width - gap_width - meta_width,
        )

        command_style = (
            "class:slash-completion-menu.command.current"
            if is_current
            else "class:slash-completion-menu.command"
        )
        meta_style = (
            "class:slash-completion-menu.meta.current"
            if is_current
            else "class:slash-completion-menu.meta"
        )
        marker_style = (
            "class:slash-completion-menu.marker.current"
            if is_current
            else "class:slash-completion-menu.marker"
        )
        marker = "› " if is_current else "  "

        fragments: FormattedText = FormattedText()
        fragments.append(("class:slash-completion-menu", " " * left_padding))
        fragments.append((marker_style, marker.ljust(marker_width)))
        fragments.append(
            (command_style, _truncate_to_width(completion.display_text, command_width))
        )
        fragments.append(("class:slash-completion-menu", " " * gap_width))
        fragments.append((meta_style, _truncate_to_width(completion.display_meta_text, meta_width)))
        fragments.append(("class:slash-completion-menu", " " * trailing_width))
        return fragments

    def _render_detail_line(self, *, width: int, text: str) -> FormattedText:
        left_padding = min(self._left_padding(), max(0, width - 1))
        detail_width = max(0, width - left_padding)
        fragments: FormattedText = FormattedText()
        fragments.append(("class:slash-completion-menu", " " * left_padding))
        style = (
            "class:slash-completion-menu.meta.current"
            if text.strip()
            else "class:slash-completion-menu"
        )
        fragments.append((style, _truncate_to_width(text, detail_width)))
        return fragments


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
        self._root = root.resolve()
        self._refresh_interval = refresh_interval
        self._limit = limit
        self._cache_time: float = 0.0
        self._cached_paths: list[str] = []
        self._cached_paths_key: str = "."
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
        return self._get_deep_paths(fragment)

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

    def _resolve_search_root(self, fragment: str) -> tuple[Path, Path]:
        if "/" not in fragment:
            return self._root, Path()

        prefix = fragment.rsplit("/", 1)[0].strip("/")
        if not prefix:
            return self._root, Path()

        current = Path(prefix)
        while current != Path("."):
            try:
                candidate = (self._root / current).resolve(strict=False)
                if candidate.is_dir() and candidate.is_relative_to(self._root):
                    return candidate, current
            except OSError:
                break
            current = current.parent

        return self._root, Path()

    def _get_deep_paths(self, fragment: str) -> list[str]:
        now = time.monotonic()
        search_root, relative_prefix = self._resolve_search_root(fragment)
        cache_key = relative_prefix.as_posix() or "."
        if now - self._cache_time <= self._refresh_interval and cache_key == self._cached_paths_key:
            return self._cached_paths

        paths: list[str] = []
        try:
            if relative_prefix.parts:
                paths.append(relative_prefix.as_posix() + "/")

            for current_root, dirs, files in os.walk(search_root):
                relative_root = Path(current_root).relative_to(search_root)
                display_root = (
                    relative_prefix / relative_root if relative_prefix.parts else relative_root
                )

                dirs[:] = sorted(d for d in dirs if not self._is_ignored(d))

                if relative_root.parts and any(
                    self._is_ignored(part) for part in relative_root.parts
                ):
                    dirs[:] = []
                    continue

                if display_root.parts and relative_root.parts:
                    paths.append(display_root.as_posix() + "/")
                    if len(paths) >= self._limit:
                        break

                for file_name in sorted(files):
                    if self._is_ignored(file_name):
                        continue
                    relative = (
                        (relative_prefix / relative_root / file_name)
                        if relative_prefix.parts
                        else (relative_root / file_name)
                    ).as_posix()
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
        self._cached_paths_key = cache_key
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

    @classmethod
    def should_complete(cls, document: Document) -> bool:
        return cls._extract_fragment(document.text_before_cursor) is not None

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
            candidates = list(self._fuzzy.get_completions(mention_doc, complete_event))

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
                return (cat,)

            candidates.sort(key=_rank)
            yield from candidates
        finally:
            self._fragment_hint = None
