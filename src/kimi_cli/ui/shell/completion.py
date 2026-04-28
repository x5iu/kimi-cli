from __future__ import annotations

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

type MentionPathSortKey = tuple[int, int, int, int, str]


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
        if not text.strip():
            fragments.append(("class:slash-completion-menu", " " * detail_width))
            return fragments

        prefix = "╰─ "
        prefix_width = min(detail_width, get_cwidth(prefix))
        detail_text_width = max(0, detail_width - prefix_width)
        fragments.append(
            (
                "class:slash-completion-menu.detail.prefix",
                _truncate_to_width(prefix, prefix_width),
            )
        )
        fragments.append(
            (
                "class:slash-completion-menu.detail",
                _truncate_to_width(text, detail_text_width),
            )
        )
        return fragments


class LocalFileMentionCompleter(Completer):
    """Offer path-aware `@` file completion by indexing workspace files.

    File discovery and ignore rules are delegated to
    :mod:`kimi_cli.utils.file_filter` so that the web backend can reuse
    them.
    """

    _TRIGGER_GUARDS = frozenset((".", "-", "_", "`", "'", '"', ":", "@", "#", "~"))

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
        self._cache_scope: str | None = None
        self._top_cache_time: float = 0.0
        self._top_cached_paths: list[str] = []
        self._fragment_hint: str | None = None
        self._is_git: bool | None = None  # lazily detected
        self._git_index_path: Path | None = None  # cached .git/index path
        self._git_index_mtime: float | None = None

    @staticmethod
    def _basename_subsequence_anchored(basename: str, frag: str) -> bool:
        if not frag:
            return True
        fb = basename.casefold()
        fl = frag.casefold()
        if not fb:
            return False
        if fb[0] != fl[0]:
            return False
        fi = 1
        bi = 1
        while fi < len(fl) and bi < len(fb):
            if fl[fi] == fb[bi]:
                fi += 1
            bi += 1
        return fi == len(fl)

    @staticmethod
    def _segment_compact_subsequence(cand_seg: str, frag: str) -> bool:
        if not frag:
            return True
        c = cand_seg.casefold()
        f = frag.casefold()
        if len(f) < 2:
            return False
        i = 0
        first: int | None = None
        last: int | None = None
        for idx, ch in enumerate(c):
            if ch == f[i]:
                if first is None:
                    first = idx
                last = idx
                i += 1
                if i == len(f):
                    break
        if i < len(f):
            return False
        if first is None or last is None:
            return False
        span = last - first + 1
        return span <= len(f) + 2

    @staticmethod
    def _segment_matches(frag: str, cand_seg: str) -> bool:
        if not frag:
            return True
        f = frag.casefold()
        c = cand_seg.casefold()
        if c == f:
            return True
        if c.startswith(f):
            return True
        if len(f) >= 2 and f in c:
            return True
        stem = Path(cand_seg).stem.casefold()
        if stem == f or stem.startswith(f) or (len(f) >= 2 and f in stem):
            return True
        return LocalFileMentionCompleter._segment_compact_subsequence(cand_seg, frag)

    @staticmethod
    def _segment_path_match_tie(rel_n: str, frag_trim: str) -> tuple[int, int] | None:
        frag_parts = [p for p in frag_trim.split("/") if p]
        if not frag_parts:
            return None
        cand_parts = rel_n.split("/")
        j = 0
        first_idx: int | None = None
        last_idx: int | None = None
        for fp in frag_parts:
            while j < len(cand_parts):
                if LocalFileMentionCompleter._segment_matches(fp, cand_parts[j]):
                    if first_idx is None:
                        first_idx = j
                    last_idx = j
                    j += 1
                    break
                j += 1
            else:
                return None
        assert first_idx is not None and last_idx is not None
        extra = (last_idx - first_idx + 1) - len(frag_parts)
        return (extra, first_idx)

    @staticmethod
    def _ascii_alnum_char(ch: str) -> bool:
        return len(ch) == 1 and ch.isascii() and ch.isalnum()

    @staticmethod
    def _mention_path_sort_key(rel: str, fragment: str) -> MentionPathSortKey | None:
        frag_trim = fragment.rstrip("/")
        if not frag_trim:
            return None
        frag_l = frag_trim.casefold()
        rel_n = rel.rstrip("/")
        if not rel_n:
            return None
        rel_l = rel_n.casefold()
        segments = rel_n.split("/")
        base = segments[-1]
        bl = base.casefold()
        depth = max(0, len(segments) - 1)
        stem = Path(base).stem.casefold()

        cand: list[tuple[int, tuple[int, ...]]] = []

        if rel_n == frag_trim or rel_l == frag_l:
            cand.append((1, (0,)))
        if bl == frag_l:
            cand.append((2, (0,)))
        if stem == frag_l and bl != frag_l:
            cand.append((3, (0,)))
        if rel_l.startswith(frag_l):
            cand.append((4, (0,)))
        if bl.startswith(frag_l) and bl != frag_l:
            cand.append((5, (0,)))

        seg_idx: int | None = None
        for i, seg in enumerate(segments[:-1]):
            if seg.casefold().startswith(frag_l):
                seg_idx = i if seg_idx is None else min(seg_idx, i)
        if seg_idx is not None:
            cand.append((6, (seg_idx,)))

        pos7 = bl.find(frag_l)
        if pos7 >= 0:
            cand.append((7, (pos7,)))

        pos8 = rel_l.find(frag_l)
        if pos8 >= 0:
            cand.append((8, (pos8,)))

        if "/" in frag_trim:
            seg_tie = LocalFileMentionCompleter._segment_path_match_tie(rel_n, frag_trim)
            if seg_tie is not None:
                ex, fidx = seg_tie
                cand.append((9, (ex, fidx)))

        if "/" not in frag_trim and LocalFileMentionCompleter._basename_subsequence_anchored(
            base, frag_trim
        ):
            cand.append((10, (0,)))

        if not cand:
            return None

        best_tier, best_tie = min(cand, key=lambda x: (x[0], x[1]))
        if best_tie and len(best_tie) >= 2:
            tie0 = int(best_tie[0] * 1024 + best_tie[1])
        else:
            tie0 = int(best_tie[0]) if best_tie else 0
        return (best_tier, tie0, len(base), depth, rel.casefold())

    def _get_paths(self) -> list[str]:
        fragment = self._fragment_hint or ""
        if fragment == "":
            return self._get_top_level_paths()
        return self._get_deep_paths()

    def _get_top_level_paths(self) -> list[str]:
        from kimi_cli.utils.file_filter import is_ignored

        now = time.monotonic()
        if now - self._top_cache_time <= self._refresh_interval:
            return self._top_cached_paths

        entries: list[str] = []
        try:
            for entry in sorted(self._root.iterdir(), key=lambda p: p.name):
                name = entry.name
                if is_ignored(name):
                    continue
                entries.append(f"{name}/" if entry.is_dir() else name)
                if len(entries) >= self._limit:
                    break
        except OSError:
            return self._top_cached_paths

        self._top_cached_paths = entries
        self._top_cache_time = now
        return self._top_cached_paths

    def _git_index_stat_mtime(self) -> float | None:
        """Return .git/index mtime without spawning a subprocess."""
        if self._git_index_path is None:
            from kimi_cli.utils.file_filter import git_index_mtime

            # First call: use the subprocess-based helper to locate the path,
            # then cache it so subsequent calls are pure stat().
            mtime = git_index_mtime(self._root)
            # Derive and cache the index path for future O(1) checks.
            try:
                import subprocess

                res = subprocess.run(
                    ["git", "rev-parse", "--git-dir"],
                    cwd=self._root,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if res.returncode == 0:
                    gd = Path(res.stdout.strip())
                    if not gd.is_absolute():
                        gd = self._root / gd
                    self._git_index_path = gd / "index"
            except Exception:
                pass
            return mtime
        try:
            return self._git_index_path.stat().st_mtime
        except OSError:
            return None

    @staticmethod
    def _resolve_scope(root: Path, fragment: str) -> str | None:
        """Derive a scope directory from *fragment*, walking up if needed."""
        if "/" not in fragment:
            return None
        scope = fragment.rsplit("/", 1)[0]
        current = Path(scope)
        while str(current) not in ("", "."):
            if (root / current).is_dir():
                return current.as_posix()
            current = current.parent
        return None

    def _get_deep_paths(self) -> list[str]:
        from kimi_cli.utils.file_filter import (
            detect_git,
            list_files_git,
            list_files_walk,
        )

        fragment = self._fragment_hint or ""
        scope = self._resolve_scope(self._root, fragment)

        now = time.monotonic()
        cache_valid = (
            now - self._cache_time <= self._refresh_interval and self._cache_scope == scope
        )

        # Invalidate on .git/index mtime change (O(1) stat after first call).
        if cache_valid and self._is_git:
            mtime = self._git_index_stat_mtime()
            if mtime != self._git_index_mtime:
                cache_valid = False

        if cache_valid:
            return self._cached_paths

        # Lazily detect git.
        if self._is_git is None:
            self._is_git = detect_git(self._root)

        paths: list[str] | None = None
        if self._is_git:
            paths = list_files_git(self._root, scope)
            self._git_index_mtime = self._git_index_stat_mtime()

        if paths is None:
            paths = list_files_walk(self._root, scope, limit=self._limit)

        self._cached_paths = paths
        self._cache_scope = scope
        self._cache_time = now
        return self._cached_paths

    @staticmethod
    def _extract_fragment(text: str) -> str | None:
        index = text.rfind("@")
        if index == -1:
            return None

        if index > 0:
            prev = text[index - 1]
            if (
                LocalFileMentionCompleter._ascii_alnum_char(prev)
                or prev in LocalFileMentionCompleter._TRIGGER_GUARDS
            ):
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

    @override
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        _ = complete_event
        fragment = self._extract_fragment(document.text_before_cursor)
        if fragment is None:
            return

        self._fragment_hint = fragment
        try:
            paths = self._get_paths()
        finally:
            self._fragment_hint = None

        if fragment == "":
            for rel in sorted(paths)[: self._limit]:
                yield Completion(text=rel, start_position=0)
            return

        ranked: list[tuple[MentionPathSortKey, str]] = []
        for rel in paths:
            key = self._mention_path_sort_key(rel, fragment)
            if key is None:
                continue
            ranked.append((key, rel))

        ranked.sort(key=lambda item: item[0])
        seen: set[str] = set()
        start_pos = -len(fragment)
        for _, rel in ranked:
            if rel in seen:
                continue
            seen.add(rel)
            yield Completion(text=rel, start_position=start_pos)
            if len(seen) >= self._limit:
                break


wrap_to_width = _wrap_to_width
find_prompt_float_container = _find_prompt_float_container
