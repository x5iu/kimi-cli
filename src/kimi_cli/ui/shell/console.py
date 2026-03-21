from __future__ import annotations

from rich.console import Console, ConsoleDimensions
from rich.theme import Theme

# Reserve 2 columns on the right so wide characters (e.g. CJK) and
# line-wrapping artefacts never get clipped by the terminal's right edge.
_RIGHT_PADDING = 2

_NEUTRAL_MARKDOWN_THEME = Theme(
    {
        "markdown.paragraph": "none",
        "markdown.block_quote": "none",
        "markdown.hr": "none",
        "markdown.item": "none",
        "markdown.item.bullet": "none",
        "markdown.item.number": "none",
        "markdown.link": "none",
        "markdown.link_url": "none",
        "markdown.h1": "none",
        "markdown.h1.border": "none",
        "markdown.h2": "none",
        "markdown.h3": "none",
        "markdown.h4": "none",
        "markdown.h5": "none",
        "markdown.h6": "none",
        "markdown.em": "none",
        "markdown.strong": "none",
        "markdown.s": "none",
        "status.spinner": "none",
    },
    inherit=True,
)


class _PaddedConsole(Console):
    """Console subclass that reserves a right-side padding column."""

    @property
    def size(self) -> ConsoleDimensions:  # pyright: ignore[reportIncompatibleMethodOverride]
        dims = super().size
        return ConsoleDimensions(
            max(1, dims.width - _RIGHT_PADDING),
            dims.height,
        )


console = _PaddedConsole(highlight=False, theme=_NEUTRAL_MARKDOWN_THEME)


RIGHT_PADDING = _RIGHT_PADDING
