from __future__ import annotations

from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES as _ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

# Register focus-in/focus-out escape sequences so prompt_toolkit swallows
# them instead of treating the bytes as literal input.
_ANSI_SEQUENCES.setdefault("\x1b[I", Keys.Ignore)
_ANSI_SEQUENCES.setdefault("\x1b[O", Keys.Ignore)

_FOCUS_IN_SEQ = "\x1b[I"

PROMPT_SYMBOL = "✨"
PROMPT_SYMBOL_SHELL = "$"
PROMPT_SYMBOL_THINKING = "💫"
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

_REFRESH_INTERVAL = 1.0
_TURN_HISTORY_VIEW_EMPTY_POSITION = "0/0"
_TURN_UI_REFRESH_INTERVAL = 0.2
_TURN_UI_FAST_REFRESH_INTERVAL = 1 / 30
_TURN_UI_BURST_REFRESH_INTERVAL = 1 / 60
_TURN_UI_STREAM_BURST_SECONDS = 0.3
_TURN_UI_STREAM_RESUME_BURST_GAP = 0.15
_TURN_UI_BURST_STREAM_PUSH_PARTS = 1
_TURN_UI_BURST_STREAM_PUSH_CHARS = 32
_TURN_UI_COMPOSE_STREAM_PUSH_PARTS = 2
_TURN_UI_COMPOSE_STREAM_PUSH_CHARS = 64
_TURN_UI_FAST_STREAM_PUSH_PARTS = 4
_TURN_UI_FAST_STREAM_PUSH_CHARS = 128
_TURN_UI_STREAM_PUSH_PARTS = 8
_TURN_UI_STREAM_PUSH_CHARS = 256
_TERMINAL_SIZE_POLLING_INTERVAL = 1.0
_RESIZE_DEBOUNCE_SECONDS = 0.3
