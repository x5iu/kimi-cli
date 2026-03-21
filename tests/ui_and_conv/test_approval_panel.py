from __future__ import annotations

from io import StringIO

from rich.console import Console

from kimi_cli.ui.shell import panels
from kimi_cli.ui.shell.panels import _ApprovalRequestPanel
from kimi_cli.wire.types import ApprovalRequest, DiffDisplayBlock


def _render_to_str(panel: _ApprovalRequestPanel) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    console.print(panel.render())
    return buf.getvalue()


def test_approval_panel_renders_line_numbers_for_edit_diff() -> None:
    panel = _ApprovalRequestPanel(
        ApprovalRequest(
            id="req-1",
            tool_call_id="tool-1",
            sender="Edit",
            action="edit files",
            description="",
            display=[
                DiffDisplayBlock(
                    path="src/example.py",
                    old_text="before",
                    new_text="after",
                    old_start_line=42,
                    new_start_line=42,
                )
            ],
        )
    )

    rendered = _render_to_str(panel)

    assert "@@ -42 +42 @@" in rendered
    assert "42    │ -before" in rendered
    assert "42 │ +after" in rendered


def test_approval_panel_renders_line_numbers_for_writefile_diff() -> None:
    panel = _ApprovalRequestPanel(
        ApprovalRequest(
            id="req-2",
            tool_call_id="tool-2",
            sender="WriteFile",
            action="write files",
            description="",
            display=[
                DiffDisplayBlock(
                    path="src/example.py",
                    old_text="before",
                    new_text="after",
                    old_start_line=42,
                    new_start_line=42,
                )
            ],
        )
    )

    rendered = _render_to_str(panel)

    assert "@@ -42 +42 @@" in rendered
    assert "42    │ -before" in rendered
    assert "42 │ +after" in rendered


class _FakePagerConsole:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def pager(self, *, styles: bool = False):
        self.calls.append(("pager", styles))
        return self

    def screen(self):
        raise AssertionError("approval preview should not switch to the alternate screen")

    def print(self, renderable=None) -> None:
        self.calls.append(("print", renderable))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_show_approval_in_pager_avoids_alternate_screen(monkeypatch) -> None:
    panel = _ApprovalRequestPanel(
        ApprovalRequest(
            id="req-pager",
            tool_call_id="tool-pager",
            sender="Shell",
            action="run command",
            description="echo pager",
        )
    )
    fake_console = _FakePagerConsole()
    monkeypatch.setattr(panels, "console", fake_console)

    panels.show_approval_in_pager(panel)

    assert ("pager", True) in fake_console.calls
    assert sum(1 for kind, _ in fake_console.calls if kind == "print") >= 2
