"""Tests for _find_committed_boundary and _ContentBlock incremental streaming."""

from __future__ import annotations

from unittest.mock import patch

from kimi_cli.ui.shell.blocks import ContentBlock, _find_committed_boundary, _get_md_parser

# ── _find_committed_boundary tests ──────────────────────────────────────────


class TestFindCommittedBoundary:
    """Tests for _find_committed_boundary.

    The function only counts self-closing markdown-it-py blocks (fence,
    code_block, hr, html_block, table) because paragraph_close /
    heading_close tokens carry ``map=None``.  At least 2 such blocks must
    be closed before a boundary is returned.
    """

    def test_empty_string_returns_none(self):
        assert _find_committed_boundary("") is None

    def test_single_paragraph_returns_none(self):
        """A single paragraph has no self-closing blocks."""
        assert _find_committed_boundary("Hello world") is None

    def test_two_paragraphs_returns_none(self):
        """Paragraphs are not self-closing blocks, so no boundary is found."""
        text = "First paragraph.\n\nSecond paragraph."
        assert _find_committed_boundary(text) is None

    def test_single_fence_plus_paragraph_returns_none(self):
        """One fence block is not enough — need at least 2 self-closing blocks."""
        text = "```python\nprint('hello')\n```\n\nA paragraph after the fence."
        assert _find_committed_boundary(text) is None

    def test_incomplete_fence_returns_none(self):
        """An unclosed fence block means zero complete self-closing blocks."""
        text = "```python\nprint('hello')\n"
        assert _find_committed_boundary(text) is None

    def test_two_fences_plus_paragraph(self):
        """Two complete fence blocks allow committing the first one."""
        text = "```py\nx=1\n```\n\n```py\ny=2\n```\n\nTrailing."
        boundary = _find_committed_boundary(text)
        assert boundary is not None
        committed = text[:boundary]
        pending = text[boundary:]
        assert "x=1" in committed
        assert "Trailing." not in committed
        assert "Trailing." in pending

    def test_fence_plus_hr_plus_paragraph(self):
        """A fence and an hr give 2 self-closing blocks."""
        text = "```py\nx=1\n```\n\n---\n\nTrailing."
        boundary = _find_committed_boundary(text)
        assert boundary is not None
        committed = text[:boundary]
        pending = text[boundary:]
        # The fence is committed (block_ends[-2])
        assert "x=1" in committed
        assert "---" not in committed
        assert "Trailing." in pending

    def test_hr_plus_fence_plus_paragraph(self):
        """An hr followed by a fence also yields 2 self-closing blocks."""
        text = "---\n\n```py\nx=1\n```\n\nTrailing."
        boundary = _find_committed_boundary(text)
        assert boundary is not None
        committed = text[:boundary]
        pending = text[boundary:]
        assert "---" in committed
        assert "Trailing." in pending

    def test_two_hrs_plus_paragraph(self):
        """Two hrs are enough to commit through the first."""
        text = "---\n\n---\n\nTrailing."
        boundary = _find_committed_boundary(text)
        assert boundary is not None
        committed = text[:boundary]
        assert "---" in committed
        assert "Trailing." not in committed

    def test_multiple_self_closing_blocks_commits_all_but_last(self):
        """With 3 self-closing blocks, the first 2 should be committed."""
        text = "---\n\n```py\ncode\n```\n\n---\n\nFinal."
        boundary = _find_committed_boundary(text)
        assert boundary is not None
        committed = text[:boundary]
        pending = text[boundary:]
        # First two blocks (hr + fence) should be committed
        assert "code" in committed
        # The last hr is the final self-closing block — pending
        assert "Final." in pending

    def test_heading_plus_paragraph_returns_none(self):
        """Heading and paragraph close tokens have map=None, so no boundary."""
        text = "# Title\n\nBody text here."
        assert _find_committed_boundary(text) is None

    def test_returns_none_when_markdown_it_unavailable(self):
        with (
            patch("kimi_cli.ui.shell.blocks._HAS_MARKDOWN_IT", False),
            patch("kimi_cli.ui.shell.blocks._md_parser_instance", None),
        ):
            assert _find_committed_boundary("---\n\n---\n\nParagraph.") is None


# ── _get_md_parser tests ────────────────────────────────────────────────────


class TestGetMdParser:
    def test_returns_parser_when_available(self):
        parser = _get_md_parser()
        assert parser is not None

    def test_returns_none_when_markdown_it_unavailable(self):
        with (
            patch("kimi_cli.ui.shell.blocks._HAS_MARKDOWN_IT", False),
            patch("kimi_cli.ui.shell.blocks._md_parser_instance", None),
        ):
            assert _get_md_parser() is None


# ── _ContentBlock incremental streaming tests ──────────────────────────────


class TestContentBlockIncremental:
    def test_small_append_does_not_trigger_boundary_check(self):
        """append() with small chunks (< 128 chars total) should NOT trigger boundary check."""
        block = ContentBlock(is_think=False)
        block.append("Hello")
        # With less than 128 chars, _last_boundary_check_len should still be 0
        assert block._last_boundary_check_len == 0
        assert block._committed_text == ""

    def test_large_append_triggers_boundary_check(self):
        """append() with enough chars (>= 128) should trigger boundary check."""
        block = ContentBlock(is_think=False)
        # Build text with 2 self-closing blocks so boundary is found once threshold is hit
        text = "```py\n" + "x" * 120 + "\n```\n\n---\n\nAnother paragraph."
        block.append(text)
        # The total length exceeds 128, so _last_boundary_check_len should be updated
        assert block._last_boundary_check_len > 0

    def test_committed_text_advances_with_two_self_closing_blocks(self):
        """_committed_text should advance when 2+ self-closing blocks are complete."""
        block = ContentBlock(is_think=False)
        # Use fence + hr (both self-closing) with enough chars to exceed 128 threshold
        chunk = "```py\n" + "x = 1  # " * 15 + "\n```\n\n---\n\nTrailing paragraph."
        block.append(chunk)

        # With 2 self-closing blocks, committed text should be non-empty
        assert block._committed_text != ""
        assert "x = 1" in block._committed_text
        # The trailing paragraph should NOT be committed
        assert "Trailing paragraph." not in block._committed_text

    def test_pending_text_returns_text_after_committed_boundary(self):
        """_pending_text should return text after committed boundary."""
        block = ContentBlock(is_think=False)
        # Use two fences + trailing paragraph (>= 128 chars to trigger check)
        chunk = "```py\n" + "a = 1\n" * 20 + "```\n\n---\n\nTrailing text."
        block.append(chunk)

        pending = block._pending_text
        # Committed text should be set (2 self-closing blocks)
        assert block._committed_text != ""
        assert pending == block.raw_text[len(block._committed_text) :]
        assert "Trailing text." in pending

    def test_pending_text_equals_full_text_when_nothing_committed(self):
        """When nothing is committed, _pending_text returns the full text."""
        block = ContentBlock(is_think=False)
        block.append("short text")
        assert block._pending_text == "short text"
        assert block._committed_text == ""

    def test_incremental_append_accumulates_chunks(self):
        """Multiple append() calls should accumulate into raw_text."""
        block = ContentBlock(is_think=False)
        block.append("Hello ")
        block.append("world")
        assert block.raw_text == "Hello world"

    def test_committed_renderable_set_on_advance(self):
        """_committed_renderable should be set when committed text advances."""
        block = ContentBlock(is_think=False)
        # Two self-closing blocks (fence + hr) with enough chars
        chunk = "```py\n" + "code_line\n" * 15 + "```\n\n---\n\nEnd block."
        block.append(chunk)

        # The two self-closing blocks should trigger a commit
        assert block._committed_text != ""
        assert block._committed_renderable is not None

    def test_no_committed_renderable_without_self_closing_blocks(self):
        """Without self-closing blocks, _committed_renderable stays None."""
        block = ContentBlock(is_think=False)
        # Only paragraphs (no self-closing blocks) — even if long
        chunk = "Para one.\n\n" + "x" * 200 + "\n\nPara three."
        block.append(chunk)

        assert block._committed_text == ""
        assert block._committed_renderable is None

    def test_throttle_prevents_repeated_boundary_checks(self):
        """Appending small increments after a check should not re-check until 128 more chars."""
        block = ContentBlock(is_think=False)
        # First: fill up past 128 chars to trigger initial check
        big_chunk = "```py\n" + "A" * 200 + "\n```\n\n---\n\nEnd."
        block.append(big_chunk)
        check_len_after_first = block._last_boundary_check_len

        # Now append a tiny chunk — should NOT trigger another check
        block.append("x")
        assert block._last_boundary_check_len == check_len_after_first

    def test_think_block_style(self):
        """Think blocks should have italic grey style."""
        block = ContentBlock(is_think=True)
        assert block.is_think is True
        assert block.status_text == "Thinking..."
        assert "italic" in block._mk_style()

    def test_non_think_block_style(self):
        """Non-think blocks should have empty style."""
        block = ContentBlock(is_think=False)
        assert block.is_think is False
        assert block.status_text == "Composing..."
        assert block._mk_style() == ""

    def test_raw_text_reflects_appended_content(self):
        """raw_text should always reflect the latest appended content."""
        block = ContentBlock(is_think=False)
        block.append("first")
        assert block.raw_text == "first"

        block.append(" second")
        # raw_text must include the new chunk even though cache was rebuilt
        assert block.raw_text == "first second"

    def test_raw_text_length_tracks_total(self):
        """_raw_text_length should track the total length of all appended chunks."""
        block = ContentBlock(is_think=False)
        block.append("abc")
        assert block._raw_text_length == 3
        block.append("de")
        assert block._raw_text_length == 5
