"""Common HTTP headers for Kimi API requests."""

from __future__ import annotations

import os
import platform
import socket
import sys
import uuid
from pathlib import Path

from kimi_cli.constant import VERSION
from kimi_cli.share import get_share_dir
from kimi_cli.utils.logging import logger


def _device_id_path() -> Path:
    return get_share_dir() / "device_id"


def _ensure_private_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.warning("Could not set permissions on %s: %s", path, exc)


def get_device_id() -> str:
    path = _device_id_path()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    device_id = uuid.uuid4().hex
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(device_id)
    except BaseException:
        raise
    _ensure_private_file(path)
    return device_id


def _device_model() -> str:
    system = platform.system()
    arch = platform.machine() or ""
    if system == "Darwin":
        version = platform.mac_ver()[0] or platform.release()
        if version and arch:
            return f"macOS {version} {arch}"
        if version:
            return f"macOS {version}"
        return f"macOS {arch}".strip()
    if system == "Windows":
        release = platform.release()
        if release == "10":
            try:
                build = sys.getwindowsversion().build  # type: ignore[attr-defined]
            except Exception:
                build = None
            if build and build >= 22000:
                release = "11"
        if release and arch:
            return f"Windows {release} {arch}"
        if release:
            return f"Windows {release}"
        return f"Windows {arch}".strip()
    if system:
        version = platform.release()
        if version and arch:
            return f"{system} {version} {arch}"
        if version:
            return f"{system} {version}"
        return f"{system} {arch}".strip()
    return "Unknown"


def _ascii_header_value(value: str, *, fallback: str = "unknown") -> str:
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        sanitized = value.encode("ascii", errors="ignore").decode("ascii").strip()
        return sanitized or fallback


def common_headers() -> dict[str, str]:
    """Return common HTTP headers for Kimi API requests."""
    device_name = platform.node() or socket.gethostname()
    device_model = _device_model()
    headers = {
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": VERSION,
        "X-Msh-Device-Name": device_name,
        "X-Msh-Device-Model": device_model,
        "X-Msh-Os-Version": platform.version(),
        "X-Msh-Device-Id": get_device_id(),
    }
    return {key: _ascii_header_value(value) for key, value in headers.items()}
