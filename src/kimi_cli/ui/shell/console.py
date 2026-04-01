from __future__ import annotations

import os
import pydoc

from rich.console import Console, ConsoleDimensions, PagerContext
from rich.pager import Pager
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


class _KimiPager(Pager):
    """Pager that ignores MANPAGER to avoid garbled output.

    ``pydoc.getpager()`` reads ``MANPAGER`` before ``PAGER``.  When the user
    sets ``MANPAGER`` to a man-specific pipeline (e.g.
    ``sh -c 'col -bx | bat -l man -p'``), that pipeline mangles the ANSI
    rich-text we emit.  This pager strips ``MANPAGER`` from the subprocess
    environment so only ``PAGER`` (or the default ``less``) is used.
    """

    def show(self, content: str) -> None:
        saved = os.environ.pop("MANPAGER", None)
        try:
            pydoc.pager(content)
        finally:
            if saved is not None:
                os.environ["MANPAGER"] = saved


class _PaddedConsole(Console):
    """Console subclass that reserves a right-side padding column."""

    @property
    def size(self) -> ConsoleDimensions:  # pyright: ignore[reportIncompatibleMethodOverride]
        dims = super().size
        return ConsoleDimensions(
            max(1, dims.width - _RIGHT_PADDING),
            dims.height,
        )

    def pager(
        self,
        pager: Pager | None = None,
        styles: bool = False,
        links: bool = False,
    ) -> PagerContext:
        if pager is None:
            pager = _KimiPager()
        return super().pager(pager=pager, styles=styles, links=links)


console = _PaddedConsole(highlight=False, theme=_NEUTRAL_MARKDOWN_THEME)


RIGHT_PADDING = _RIGHT_PADDING
