import json
import os
import subprocess
from pathlib import Path

import pytest

from kaos.path import KaosPath


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_trace(label: str, text: str) -> None:
    if os.getenv("KIMI_TEST_TRACE") == "1":
        print("-----")
        print(f"{label}: {text}")


def _send_json(process: subprocess.Popen[str], payload: dict[str, object]) -> None:
    assert process.stdin is not None
    line = json.dumps(payload)
    _print_trace("STDIN", line)
    process.stdin.write(line + "\n")
    process.stdin.flush()


def _parse_message_content(line: str) -> list[dict[str, object]]:
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _content_has_part(parts: list[dict[str, object]], part_type: str) -> bool:
    return any(part.get("type") == part_type for part in parts)


def _content_has_text(parts: list[dict[str, object]], text: str) -> bool:
    return any(part.get("type") == "text" and text in str(part.get("text", "")) for part in parts)


def _run_print_mode(
    config_path: Path,
    work_dir: Path,
    messages: list[dict[str, object]],
    share_dir: Path,
) -> tuple[int, list[str]]:
    cmd = [
        "uv",
        "run",
        "kimi",
        "--print",
        "--yolo",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--config-file",
        str(config_path),
        "--work-dir",
        str(work_dir),
    ]
    env = os.environ.copy()
    env["KIMI_SHARE_DIR"] = str(share_dir)
    process = subprocess.Popen(
        cmd,
        cwd=_repo_root(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert process.stdin is not None
    for msg in messages:
        _send_json(process, msg)
    process.stdin.close()

    stdout_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        _print_trace("STDOUT", line)
        stdout_lines.append(line)
    return process.wait(), stdout_lines


@pytest.mark.parametrize("mode", ["print"])
def test_scripted_echo_media_e2e(temp_work_dir: KaosPath, tmp_path: Path, mode: str) -> None:
    image_url = "data:image/png;base64,AAAA"
    video_url = "data:video/mp4;base64,AAAA"

    scripts = [
        "\n".join(
            [
                "id: scripted-1",
                'usage: {"input_other": 11, "output": 5}',
                "think: analyzing the image",
                "text: The image shows a simple scene.",
            ]
        ),
        "\n".join(
            [
                "id: scripted-2",
                'usage: {"input_other": 13, "output": 6}',
                "think: analyzing the video",
                "text: The video appears to be a short clip.",
            ]
        ),
    ]

    scripts_path = tmp_path / "scripts.json"
    scripts_path.write_text(json.dumps(scripts), encoding="utf-8")

    config_path = tmp_path / "config.json"
    trace_env = os.getenv("KIMI_SCRIPTED_ECHO_TRACE", "0")
    config_data = {
        "default_model": "scripted",
        "models": {
            "scripted": {
                "provider": "scripted_provider",
                "model": "scripted_echo",
                "max_context_size": 100000,
                "capabilities": ["image_in", "video_in", "thinking"],
            }
        },
        "providers": {
            "scripted_provider": {
                "type": "_scripted_echo",
                "base_url": "",
                "api_key": "",
                "env": {
                    "KIMI_SCRIPTED_ECHO_SCRIPTS": str(scripts_path),
                    "KIMI_SCRIPTED_ECHO_TRACE": trace_env,
                },
            }
        },
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    work_dir = temp_work_dir.unsafe_to_local_path()
    share_dir = tmp_path / "share"
    if mode == "print":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            },
        ]
        return_code, stdout_lines = _run_print_mode(config_path, work_dir, messages, share_dir)
        assert return_code == 0
        parsed_contents = [_parse_message_content(line) for line in stdout_lines]
        parsed_contents = [parts for parts in parsed_contents if parts]
        assert len(parsed_contents) >= 2
        assert _content_has_part(parsed_contents[0], "think")
        assert _content_has_text(parsed_contents[0], "The image shows a simple scene.")
        assert _content_has_part(parsed_contents[1], "think")
        assert _content_has_text(parsed_contents[1], "The video appears to be a short clip.")
    else:
        raise AssertionError(f"Unknown mode: {mode}")
