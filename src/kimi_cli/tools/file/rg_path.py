from __future__ import annotations

import platform
import shlex
import shutil
from pathlib import Path

import kimi_cli
from kimi_cli.share import get_share_dir


def rg_binary_name() -> str:
    return "rg.exe" if platform.system() == "Windows" else "rg"


def find_existing_rg(bin_name: str | None = None) -> Path | None:
    resolved_bin_name = bin_name or rg_binary_name()

    share_bin = get_share_dir() / "bin" / resolved_bin_name
    if share_bin.is_file():
        return share_bin

    assert kimi_cli.__file__ is not None
    local_dep = Path(kimi_cli.__file__).parent / "deps" / "bin" / resolved_bin_name
    if local_dep.is_file():
        return local_dep

    system_rg = shutil.which("rg")
    if system_rg:
        return Path(system_rg)

    return None


def format_rg_command(path: Path, *, is_powershell: bool) -> str:
    path_str = str(path)
    if is_powershell:
        return f'& "{path_str}"'
    return shlex.quote(path_str)
