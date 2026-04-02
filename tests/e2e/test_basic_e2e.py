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


def _collect_stdout(process: subprocess.Popen[str]) -> list[str]:
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        line = line.rstrip("\n")
        _print_trace("STDOUT", line)
        lines.append(line)
    return lines


def _run_print_mode(config_path: Path, work_dir: Path, user_prompt: str) -> tuple[int, list[str]]:
    cmd = [
        "uv",
        "run",
        "kimi",
        "--print",
        "--yolo",
        "--input-format",
        "text",
        "--output-format",
        "stream-json",
        "--config-file",
        str(config_path),
        "--work-dir",
        str(work_dir),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=_repo_root(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    assert process.stdin is not None
    process.stdin.write(user_prompt)
    process.stdin.close()
    stdout_lines = _collect_stdout(process)
    return process.wait(), stdout_lines


def _run_shell_mode(config_path: Path, work_dir: Path, user_prompt: str) -> tuple[int, list[str]]:
    cmd = [
        "uv",
        "run",
        "kimi",
        "--yolo",
        "--prompt",
        user_prompt,
        "--config-file",
        str(config_path),
        "--work-dir",
        str(work_dir),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=_repo_root(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    if process.stdin is not None:
        process.stdin.close()
    stdout_lines = _collect_stdout(process)
    return process.wait(), stdout_lines
@pytest.mark.parametrize("mode", ["print", "shell"])
async def test_scripted_echo_kimi_cli_agent_e2e(
    temp_work_dir: KaosPath, tmp_path: Path, mode: str
) -> None:
    sample_js = "\n".join(
        [
            "function add(a, b) {",
            "  return a + b;",
            "}",
            "",
            "function main() {",
            "  const result = add(2, 3);",
            "  console.log(`2 + 3 = ${result}`);",
            "}",
            "",
            "main();",
            "",
        ]
    )
    await (temp_work_dir / "sample.js").write_text(sample_js)

    translated_py = "\n".join(
        [
            "def add(a, b):",
            "    return a + b",
            "",
            "def main():",
            "    result = add(2, 3)",
            '    print(f"2 + 3 = {result}")',
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )

    read_args = json.dumps({"path": "sample.js"})
    write_args = json.dumps(
        {
            "path": "translated.py",
            "content": translated_py,
            "mode": "overwrite",
        }
    )
    read_call = {"id": "ReadFile:0", "name": "ReadFile", "arguments": read_args}
    write_call = {"id": "WriteFile:1", "name": "WriteFile", "arguments": write_args}

    scripts = [
        "\n".join(
            [
                "id: scripted-1",
                'usage: {"input_other": 18, "output": 3}',
                f"tool_call: {json.dumps(read_call)}",
            ]
        ),
        "\n".join(
            [
                "id: scripted-2",
                'usage: {"input_other": 22, "output": 4}',
                f"tool_call: {json.dumps(write_call)}",
            ]
        ),
        "\n".join(
            [
                "id: scripted-3",
                'usage: {"input_other": 12, "output": 2}',
                "text: Translation completed successfully.",
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

    user_prompt = (
        "You are a code translation assistant.\n\n"
        "Task:\n"
        "- Read the file `sample.js` in the current working directory.\n"
        "- Translate it into idiomatic Python 3.\n"
        "- Write the translated code to `translated.py` in the current working directory.\n\n"
        "Rules:\n"
        "- You must read the file from disk; do not guess its contents.\n"
        "- Preserve behavior and output.\n"
        "- Write only Python code in translated.py (no Markdown).\n"
        "- Overwrite translated.py if it already exists.\n"
        "- After writing, reply with a single short ASCII confirmation sentence.\n"
    )

    work_dir = temp_work_dir.unsafe_to_local_path()
    _print_trace("USER INPUT", json.dumps(user_prompt))

    if mode == "print":
        return_code, stdout_lines = _run_print_mode(config_path, work_dir, user_prompt)
        assert return_code == 0
        assert any("Translation completed successfully." in line for line in stdout_lines)
    elif mode == "shell":
        return_code, stdout_lines = _run_shell_mode(config_path, work_dir, user_prompt)
        assert return_code == 0
        assert any("Translation completed successfully." in line for line in stdout_lines)
    else:
        raise AssertionError(f"Unknown mode: {mode}")

    translated_path = work_dir / "translated.py"
    assert translated_path.read_text(encoding="utf-8") == translated_py
