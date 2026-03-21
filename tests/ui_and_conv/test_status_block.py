"""Tests for _StatusBlock partial-update logic."""

from kimi_cli.ui.shell.blocks import StatusBlock
from kimi_cli.wire.types import StatusUpdate


def test_full_initial_status():
    """All three fields provided — should display percentage and token counts."""
    block = StatusBlock(
        StatusUpdate(
            context_usage=0.42,
            context_tokens=4200,
            max_context_tokens=10000,
        )
    )
    assert "42.0%" in block.text.plain
    assert "4.2k" in block.text.plain
    assert "10k" in block.text.plain


def test_partial_update_preserves_tokens():
    """Updating only context_usage should keep previous token values."""
    block = StatusBlock(
        StatusUpdate(
            context_usage=0.30,
            context_tokens=3000,
            max_context_tokens=10000,
        )
    )
    # Partial update: only context_usage changes
    block.update(StatusUpdate(context_usage=0.50))
    assert "50.0%" in block.text.plain
    # Token values should be preserved, not reset to 0
    assert "3k" in block.text.plain
    assert "10k" in block.text.plain


def test_update_tokens_only_rerenders_with_latest_counts():
    """Updating token counts should refresh the displayed status line."""
    block = StatusBlock(
        StatusUpdate(
            context_usage=0.30,
            context_tokens=3000,
            max_context_tokens=10000,
        )
    )

    block.update(StatusUpdate(context_tokens=5000))

    assert "30.0%" in block.text.plain
    assert "5k" in block.text.plain
    assert "10k" in block.text.plain


def test_all_none_update_is_noop():
    """An empty StatusUpdate should change nothing."""
    block = StatusBlock(
        StatusUpdate(
            context_usage=0.30,
            context_tokens=3000,
            max_context_tokens=10000,
        )
    )
    old_text = block.text.plain
    block.update(StatusUpdate())
    assert block.text.plain == old_text


def test_initial_all_none():
    """Initial status with all None fields — text should remain empty."""
    block = StatusBlock(StatusUpdate())
    assert block.text.plain == ""
