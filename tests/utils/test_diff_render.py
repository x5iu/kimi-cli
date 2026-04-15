from __future__ import annotations

from kimi_cli.utils.rich.diff_render import (
    _ADD_HL,
    _DEL_HL,
    DiffLineKind,
    _build_diff_lines,
    _highlight_hunk,
    _make_highlighter,
)


class TestInlineDiffTabs:
    """Inline highlight offsets must account for tab-to-space expansion."""

    def test_inline_diff_with_tabs(self) -> None:
        old = "\told_value = 1"
        new = "\tnew_value = 2"
        hunks = _build_diff_lines(old, new, 1, 1)
        hl = _make_highlighter("test.py")
        _highlight_hunk(hl, hunks[0])
        deletes = [dl for dl in hunks[0] if dl.kind == DiffLineKind.DELETE]
        adds = [dl for dl in hunks[0] if dl.kind == DiffLineKind.ADD]
        assert deletes[0].is_inline_paired
        assert adds[0].is_inline_paired
        assert deletes[0].content is not None
        assert adds[0].content is not None
        del_plain = deletes[0].content.plain
        add_plain = adds[0].content.plain
        assert "old_value" in del_plain
        assert "new_value" in add_plain
        # Verify the highlight spans cover the actual changed words,
        # not characters shifted by unexpanded-tab offsets.
        del_hl_spans = [
            (s.start, s.end) for s in deletes[0].content.spans if s.style == _DEL_HL
        ]
        add_hl_spans = [(s.start, s.end) for s in adds[0].content.spans if s.style == _ADD_HL]
        del_highlighted = "".join(del_plain[s:e] for s, e in del_hl_spans)
        add_highlighted = "".join(add_plain[s:e] for s, e in add_hl_spans)
        assert "old" in del_highlighted, f"expected 'old' in highlighted text: {del_highlighted!r}"
        assert "new" in add_highlighted, f"expected 'new' in highlighted text: {add_highlighted!r}"

    def test_inline_diff_tab_to_spaces(self) -> None:
        old = "\tvalue = 1"
        new = "    value = 1"
        hunks = _build_diff_lines(old, new, 1, 1)
        hl = _make_highlighter("test.py")
        _highlight_hunk(hl, hunks[0])
        deletes = [dl for dl in hunks[0] if dl.kind == DiffLineKind.DELETE]
        adds = [dl for dl in hunks[0] if dl.kind == DiffLineKind.ADD]
        assert deletes[0].is_inline_paired
        assert adds[0].is_inline_paired
        assert deletes[0].content is not None
        assert adds[0].content is not None
        del_hl_spans = [
            (s.start, s.end) for s in deletes[0].content.spans if s.style == _DEL_HL
        ]
        add_hl_spans = [(s.start, s.end) for s in adds[0].content.spans if s.style == _ADD_HL]
        assert del_hl_spans, "tab indentation should be highlighted in deleted line"
        assert add_hl_spans, "space indentation should be highlighted in added line"

    def test_inline_diff_mixed_tab_and_space(self) -> None:
        old = "a\t b"
        new = "a   b"
        hunks = _build_diff_lines(old, new, 1, 1)
        hl = _make_highlighter("test.txt")
        _highlight_hunk(hl, hunks[0])
        deletes = [dl for dl in hunks[0] if dl.kind == DiffLineKind.DELETE]
        adds = [dl for dl in hunks[0] if dl.kind == DiffLineKind.ADD]
        assert deletes[0].is_inline_paired
        assert adds[0].is_inline_paired
        assert deletes[0].content is not None
        assert adds[0].content is not None
        del_hl_spans = [
            (s.start, s.end) for s in deletes[0].content.spans if s.style == _DEL_HL
        ]
        add_hl_spans = [(s.start, s.end) for s in adds[0].content.spans if s.style == _ADD_HL]
        assert del_hl_spans, "tab+space region should be highlighted in deleted line"
        assert all(s < e for s, e in del_hl_spans), "highlight spans must have non-zero width"
        assert add_hl_spans, "space region should be highlighted in added line"
        assert all(s < e for s, e in add_hl_spans), "highlight spans must have non-zero width"

    def test_inline_diff_trailing_whitespace(self) -> None:
        old = "hello   "
        new = "hello"
        hunks = _build_diff_lines(old, new, 1, 1)
        hl = _make_highlighter("test.txt")
        _highlight_hunk(hl, hunks[0])
        deletes = [dl for dl in hunks[0] if dl.kind == DiffLineKind.DELETE]
        adds = [dl for dl in hunks[0] if dl.kind == DiffLineKind.ADD]
        assert deletes[0].is_inline_paired
        assert adds[0].is_inline_paired
        assert deletes[0].content is not None
        # Trailing spaces must be preserved in the rendered content
        assert deletes[0].content.plain == "hello   "
        del_hl_spans = [
            (s.start, s.end) for s in deletes[0].content.spans if s.style == _DEL_HL
        ]
        del_highlighted = "".join(deletes[0].content.plain[s:e] for s, e in del_hl_spans)
        assert "   " in del_highlighted, (
            f"trailing spaces should be highlighted, got: {del_highlighted!r}"
        )
