from __future__ import annotations

from pathlib import Path

import kimi_cli.tools.shell as shell_module


def test_build_rg_preference_guidance_includes_path(monkeypatch) -> None:
    monkeypatch.setattr(shell_module, "find_existing_rg", lambda: Path("/tmp/tools/rg"))

    guidance = shell_module._build_rg_preference_guidance(is_powershell=False)

    assert "Preferred Content Search" in guidance
    assert "`/tmp/tools/rg`" in guidance
    assert "prefer Shell with `/tmp/tools/rg`" in guidance
    assert "`which rg`" in guidance


def test_build_rg_preference_guidance_empty_without_rg(monkeypatch) -> None:
    monkeypatch.setattr(shell_module, "find_existing_rg", lambda: None)

    assert shell_module._build_rg_preference_guidance(is_powershell=False) == ""
