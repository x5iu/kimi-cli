from __future__ import annotations

import platform
import sys
from pathlib import Path

from inline_snapshot import snapshot


def test_pyinstaller_datas():
    from kimi_cli.utils.pyinstaller import datas

    project_root = Path(__file__).parent.parent.parent
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = f".venv/lib/python{python_version}/site-packages"
    rg_binary = "rg.exe" if platform.system() == "Windows" else "rg"
    has_rg_binary = (project_root / "src/kimi_cli/deps/bin" / rg_binary).exists()
    datas = [
        (
            Path(path)
            .relative_to(project_root)
            .as_posix()
            .replace(".venv/Lib/site-packages", site_packages),
            Path(dst).as_posix(),
        )
        for path, dst in datas
    ]

    expected_datas = [
        (
            f"{site_packages}/fastmcp/../fastmcp-2.12.5.dist-info/INSTALLER",
            "fastmcp/../fastmcp-2.12.5.dist-info",
        ),
        (
            f"{site_packages}/fastmcp/../fastmcp-2.12.5.dist-info/METADATA",
            "fastmcp/../fastmcp-2.12.5.dist-info",
        ),
        (
            f"{site_packages}/fastmcp/../fastmcp-2.12.5.dist-info/RECORD",
            "fastmcp/../fastmcp-2.12.5.dist-info",
        ),
        (
            f"{site_packages}/fastmcp/../fastmcp-2.12.5.dist-info/REQUESTED",
            "fastmcp/../fastmcp-2.12.5.dist-info",
        ),
        (
            f"{site_packages}/fastmcp/../fastmcp-2.12.5.dist-info/WHEEL",
            "fastmcp/../fastmcp-2.12.5.dist-info",
        ),
        (
            f"{site_packages}/fastmcp/../fastmcp-2.12.5.dist-info/entry_points.txt",
            "fastmcp/../fastmcp-2.12.5.dist-info",
        ),
        (
            f"{site_packages}/fastmcp/../fastmcp-2.12.5.dist-info/licenses/LICENSE",
            "fastmcp/../fastmcp-2.12.5.dist-info/licenses",
        ),
        (
            "src/kimi_cli/CHANGELOG.md",
            "kimi_cli",
        ),
        ("src/kimi_cli/agents/default/agent.yaml", "kimi_cli/agents/default"),
        ("src/kimi_cli/agents/default/system.md", "kimi_cli/agents/default"),
        ("src/kimi_cli/prompts/compact.md", "kimi_cli/prompts"),
        ("src/kimi_cli/prompts/init.md", "kimi_cli/prompts"),
        (
            "src/kimi_cli/skills/kimi-cli-help/SKILL.md",
            "kimi_cli/skills/kimi-cli-help",
        ),
        (
            "src/kimi_cli/skills/skill-creator/SKILL.md",
            "kimi_cli/skills/skill-creator",
        ),
        ("src/kimi_cli/tools/ask_user/description.md", "kimi_cli/tools/ask_user"),
        (
            "src/kimi_cli/tools/background/list.md",
            "kimi_cli/tools/background",
        ),
        (
            "src/kimi_cli/tools/background/output.md",
            "kimi_cli/tools/background",
        ),
        (
            "src/kimi_cli/tools/background/stop.md",
            "kimi_cli/tools/background",
        ),
        (
            "src/kimi_cli/tools/context/description.md",
            "kimi_cli/tools/context",
        ),
        (
            "src/kimi_cli/tools/file/edit.md",
            "kimi_cli/tools/file",
        ),
        (
            "src/kimi_cli/tools/file/glob.md",
            "kimi_cli/tools/file",
        ),
        (
            "src/kimi_cli/tools/file/grep.md",
            "kimi_cli/tools/file",
        ),
        (
            "src/kimi_cli/tools/file/read.md",
            "kimi_cli/tools/file",
        ),
        (
            "src/kimi_cli/tools/file/read_media.md",
            "kimi_cli/tools/file",
        ),
        (
            "src/kimi_cli/tools/file/write.md",
            "kimi_cli/tools/file",
        ),
        ("src/kimi_cli/tools/shell/bash.md", "kimi_cli/tools/shell"),
        ("src/kimi_cli/tools/shell/powershell.md", "kimi_cli/tools/shell"),
        (
            "src/kimi_cli/tools/think/think.md",
            "kimi_cli/tools/think",
        ),
        (
            "src/kimi_cli/tools/todo/set_todo_list.md",
            "kimi_cli/tools/todo",
        ),
        (
            "src/kimi_cli/tools/web/fetch.md",
            "kimi_cli/tools/web",
        ),
        (
            "src/kimi_cli/tools/web/search.md",
            "kimi_cli/tools/web",
        ),
    ]
    if has_rg_binary:
        expected_datas.append((f"src/kimi_cli/deps/bin/{rg_binary}", "kimi_cli/deps/bin"))

    assert sorted(datas) == snapshot([
    (
        ".venv/lib/python3.14/site-packages/fastmcp/../fastmcp-2.12.5.dist-info/INSTALLER",
        "fastmcp/../fastmcp-2.12.5.dist-info",
    ),
    (
        ".venv/lib/python3.14/site-packages/fastmcp/../fastmcp-2.12.5.dist-info/METADATA",
        "fastmcp/../fastmcp-2.12.5.dist-info",
    ),
    (
        ".venv/lib/python3.14/site-packages/fastmcp/../fastmcp-2.12.5.dist-info/RECORD",
        "fastmcp/../fastmcp-2.12.5.dist-info",
    ),
    (
        ".venv/lib/python3.14/site-packages/fastmcp/../fastmcp-2.12.5.dist-info/REQUESTED",
        "fastmcp/../fastmcp-2.12.5.dist-info",
    ),
    (
        ".venv/lib/python3.14/site-packages/fastmcp/../fastmcp-2.12.5.dist-info/WHEEL",
        "fastmcp/../fastmcp-2.12.5.dist-info",
    ),
    (
        ".venv/lib/python3.14/site-packages/fastmcp/../fastmcp-2.12.5.dist-info/entry_points.txt",
        "fastmcp/../fastmcp-2.12.5.dist-info",
    ),
    (
        ".venv/lib/python3.14/site-packages/fastmcp/../fastmcp-2.12.5.dist-info/licenses/LICENSE",
        "fastmcp/../fastmcp-2.12.5.dist-info/licenses",
    ),
    ("src/kimi_cli/CHANGELOG.md", "kimi_cli"),
    ("src/kimi_cli/agents/default/agent.yaml", "kimi_cli/agents/default"),
    ("src/kimi_cli/agents/default/system.md", "kimi_cli/agents/default"),
    ("src/kimi_cli/prompts/compact.md", "kimi_cli/prompts"),
    ("src/kimi_cli/prompts/init.md", "kimi_cli/prompts"),
    ("src/kimi_cli/prompts/skill_recommender.md", "kimi_cli/prompts"),
    ("src/kimi_cli/prompts/turn_end_question_detector.md", "kimi_cli/prompts"),
    ("src/kimi_cli/skills/kimi-cli-help/SKILL.md", "kimi_cli/skills/kimi-cli-help"),
    (
        "src/kimi_cli/skills/kimi-code-worker/SKILL.md",
        "kimi_cli/skills/kimi-code-worker",
    ),
    ("src/kimi_cli/skills/skill-creator/SKILL.md", "kimi_cli/skills/skill-creator"),
    ("src/kimi_cli/tools/ask_user/description.md", "kimi_cli/tools/ask_user"),
    ("src/kimi_cli/tools/background/list.md", "kimi_cli/tools/background"),
    ("src/kimi_cli/tools/background/output.md", "kimi_cli/tools/background"),
    ("src/kimi_cli/tools/background/stop.md", "kimi_cli/tools/background"),
    ("src/kimi_cli/tools/background/write.md", "kimi_cli/tools/background"),
    ("src/kimi_cli/tools/context/description.md", "kimi_cli/tools/context"),
    ("src/kimi_cli/tools/file/edit.md", "kimi_cli/tools/file"),
    ("src/kimi_cli/tools/file/glob.md", "kimi_cli/tools/file"),
    ("src/kimi_cli/tools/file/grep.md", "kimi_cli/tools/file"),
    ("src/kimi_cli/tools/file/read.md", "kimi_cli/tools/file"),
    ("src/kimi_cli/tools/file/read_media.md", "kimi_cli/tools/file"),
    ("src/kimi_cli/tools/file/write.md", "kimi_cli/tools/file"),
    ("src/kimi_cli/tools/shell/bash.md", "kimi_cli/tools/shell"),
    ("src/kimi_cli/tools/shell/powershell.md", "kimi_cli/tools/shell"),
    ("src/kimi_cli/tools/think/think.md", "kimi_cli/tools/think"),
    ("src/kimi_cli/tools/todo/set_todo_list.md", "kimi_cli/tools/todo"),
    ("src/kimi_cli/tools/web/fetch.md", "kimi_cli/tools/web"),
    ("src/kimi_cli/tools/web/search.md", "kimi_cli/tools/web"),
])


def test_pyinstaller_hiddenimports():
    from kimi_cli.utils.pyinstaller import hiddenimports

    assert sorted(hiddenimports) == snapshot(
        [
            "kimi_cli.tools",
            "kimi_cli.tools.ask_user",
            "kimi_cli.tools.background",
            "kimi_cli.tools.context",
            "kimi_cli.tools.context.recall_compacted",
            "kimi_cli.tools.display",
            "kimi_cli.tools.file",
            "kimi_cli.tools.file.glob",
            "kimi_cli.tools.file.grep_local",
            "kimi_cli.tools.file.read",
            "kimi_cli.tools.file.read_media",
            "kimi_cli.tools.file.replace",
            "kimi_cli.tools.file.rg_path",
            "kimi_cli.tools.file.utils",
            "kimi_cli.tools.file.write",
            "kimi_cli.tools.shell",
            "kimi_cli.tools.test",
            "kimi_cli.tools.think",
            "kimi_cli.tools.todo",
            "kimi_cli.tools.todo_text",
            "kimi_cli.tools.utils",
            "kimi_cli.tools.web",
            "kimi_cli.tools.web.fetch",
            "kimi_cli.tools.web.search",
            "setproctitle",
        ]
    )
