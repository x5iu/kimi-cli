"""Tests for the shell file mention completer."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from kimi_cli.ui.shell.prompt import LocalFileMentionCompleter


def _completion_texts(completer: LocalFileMentionCompleter, text: str) -> list[str]:
    document = Document(text=text, cursor_position=len(text))
    event = CompleteEvent(completion_requested=True)
    return [completion.text for completion in completer.get_completions(document, event)]


def test_top_level_paths_skip_ignored_names(tmp_path: Path):
    """Only surface non-ignored entries when completing the top level."""
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "README.md").write_text("hello")

    completer = LocalFileMentionCompleter(tmp_path)

    texts = _completion_texts(completer, "@")

    assert "src/" in texts
    assert "README.md" in texts
    assert "node_modules/" not in texts
    assert ".DS_Store" not in texts


def test_directory_completion_continues_after_slash(tmp_path: Path):
    """Continue descending when the fragment ends with a slash."""
    src = tmp_path / "src"
    src.mkdir()
    nested = src / "module.py"
    nested.write_text("print('hi')\n")

    completer = LocalFileMentionCompleter(tmp_path)

    texts = _completion_texts(completer, "@src/")

    assert "src/" in texts
    assert "src/module.py" in texts


def test_directory_completion_scans_only_matching_subtree(
    tmp_path: Path,
):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "pkg" / "module.py").write_text("print('hi')\n")
    (tmp_path / "docs" / "guide.md").write_text("# guide\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@src/")

    assert "src/" in texts
    assert "src/pkg/" in texts
    assert "src/pkg/module.py" in texts
    # docs should not appear when scoped to src/
    assert all("docs" not in t for t in texts)


def test_completed_exact_file_remains_visible(tmp_path: Path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Agents\n")

    nested_dir = tmp_path / "src" / "kimi_cli" / "agents"
    nested_dir.mkdir(parents=True)
    (nested_dir / "README.md").write_text("nested\n")

    completer = LocalFileMentionCompleter(tmp_path)

    texts = _completion_texts(completer, "@AGENTS.md")

    assert "AGENTS.md" in texts


def test_completed_nested_path_remains_visible(tmp_path: Path):
    nested = tmp_path / "src" / "kimi_cli" / "tools" / "web"
    nested.mkdir(parents=True)
    target = nested / "fetch.py"
    target.write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    rel = "src/kimi_cli/tools/web/fetch.py"
    texts = _completion_texts(completer, f"@{rel}")

    assert rel in texts


def test_limit_is_enforced(tmp_path: Path):
    """Respect the configured limit when building top-level candidates."""
    for index in range(10):
        (tmp_path / f"dir{index}").mkdir()
    for index in range(10):
        (tmp_path / f"file{index}.txt").write_text("x")

    limit = 8
    completer = LocalFileMentionCompleter(tmp_path, limit=limit)

    texts = _completion_texts(completer, "@")

    assert len(set(texts)) == limit


def test_at_guard_prevents_email_like_fragments(tmp_path: Path):
    """Ignore `@` that are embedded inside identifiers (e.g. emails)."""
    (tmp_path / "example.py").write_text("")

    completer = LocalFileMentionCompleter(tmp_path)

    texts = _completion_texts(completer, "email@example.com")

    assert not texts


def test_basename_prefix_is_ranked_first(tmp_path: Path):
    """Prefer basename prefix matches over cross-segment fuzzy matches.

    For query 'fetch', we want '.../fetch.py' to appear before paths that only
    match by spreading characters across segments like 'file/patch.py'.
    """
    # Build a small tree mimicking the real project structure
    (tmp_path / "src" / "kimi_cli" / "tools" / "web").mkdir(parents=True)
    (tmp_path / "src" / "kimi_cli" / "tools" / "file").mkdir(parents=True)

    fetch_py = tmp_path / "src" / "kimi_cli" / "tools" / "web" / "fetch.py"
    fetch_py.write_text("# fetch\n")
    patch_py = tmp_path / "src" / "kimi_cli" / "tools" / "file" / "patch.py"
    patch_py.write_text("# patch\n")

    completer = LocalFileMentionCompleter(tmp_path)

    texts = _completion_texts(completer, "@fetch")

    assert texts == ["src/kimi_cli/tools/web/fetch.py"]


def test_fetch_subsequence_basename_still_matches(tmp_path: Path):
    (tmp_path / "src" / "kimi_cli" / "tools" / "web").mkdir(parents=True)
    (tmp_path / "src" / "kimi_cli" / "tools" / "file").mkdir(parents=True)
    (tmp_path / "src" / "kimi_cli" / "tools" / "web" / "fetch.py").write_text("x\n")
    (tmp_path / "src" / "kimi_cli" / "tools" / "file" / "patch.py").write_text("y\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@fch")

    assert "src/kimi_cli/tools/web/fetch.py" in texts
    assert "src/kimi_cli/tools/file/patch.py" not in texts


def test_patch_query_skips_chat_provider_false_substring(tmp_path: Path):
    (tmp_path / "packages" / "llm").mkdir(parents=True)
    (tmp_path / "packages" / "llm" / "chat_provider.py").write_text("x\n")
    (tmp_path / "src" / "kimi_cli" / "loop").mkdir(parents=True)
    (tmp_path / "src" / "kimi_cli" / "loop" / "compaction_archive.py").write_text("x\n")
    (tmp_path / "patch_tool.py").write_text("y\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@patch")

    assert "patch_tool.py" in texts
    assert "packages/llm/chat_provider.py" not in texts
    assert "src/kimi_cli/loop/compaction_archive.py" not in texts


def test_two_char_fragment_matches_nested_completion_paths(tmp_path: Path):
    (tmp_path / "alpha" / "nested").mkdir(parents=True)
    (tmp_path / "beta" / "nested").mkdir(parents=True)
    (tmp_path / "alpha" / "nested" / "completion.py").write_text("x\n")
    (tmp_path / "beta" / "nested" / "compaction_helper.py").write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@co")

    assert "alpha/nested/completion.py" in texts
    assert "beta/nested/compaction_helper.py" in texts


def test_two_char_fragment_matches_nested_directory_names(tmp_path: Path):
    (tmp_path / "src" / "kimi_cli" / "ui" / "shell").mkdir(parents=True)
    (tmp_path / "src" / "kimi_cli" / "ui" / "shell" / "block.py").write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts_sh = _completion_texts(completer, "@sh")
    texts_ui = _completion_texts(completer, "@ui")

    assert "src/kimi_cli/ui/shell/" in texts_sh
    assert "src/kimi_cli/ui/" in texts_ui


def test_segment_query_matches_with_skipped_middle_directory(tmp_path: Path):
    nested = tmp_path / "src" / "kimi_cli" / "ui" / "shell"
    nested.mkdir(parents=True)
    (nested / "completion.py").write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@src/ui")

    assert "src/kimi_cli/ui/shell/completion.py" in texts


def test_segment_query_kimi_main_matches_nested_main_py(tmp_path: Path):
    (tmp_path / "src" / "kimi_cli").mkdir(parents=True)
    (tmp_path / "src" / "kimi_cli" / "main.py").write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@kimi/main")

    assert "src/kimi_cli/main.py" in texts


def test_segment_query_matches_deep_utils_file_filter_path(tmp_path: Path):
    utils_dir = tmp_path / "src" / "kimi_cli" / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "file_filter.py").write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@src/utils/file_filter.py")

    assert "src/kimi_cli/utils/file_filter.py" in texts


def test_cjk_character_before_at_still_triggers_file_completion(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "请看@src/main")

    assert "src/main.py" in texts


def test_ascii_email_fragment_suppressed(tmp_path: Path):
    (tmp_path / "example.py").write_text("")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "user@host.com")

    assert not texts


def test_anchored_subsequence_matches_completion_abbreviation(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "completion.py").write_text("x\n")

    completer = LocalFileMentionCompleter(tmp_path)
    texts = _completion_texts(completer, "@cmpltn")

    assert texts == ["pkg/completion.py"]
