from __future__ import annotations

import json

from kimi_cli.tools import extract_key_argument


class TestExtractKeyArgument:
    """Tests for extract_key_argument with string inputs."""

    def test_fetchurl(self):
        result = extract_key_argument('{"url": "https://example.com/a/b/c"}', "FetchURL")
        assert result is not None
        assert "example.com" in result

    def test_shell(self):
        result = extract_key_argument('{"command": "ls -la"}', "Shell")
        assert result == "ls -la"

    def test_readfile(self):
        result = extract_key_argument('{"path": "foo/bar.py"}', "ReadFile")
        assert result is not None
        assert "foo/bar.py" in result


    def test_recall_compacted_context(self):
        result = extract_key_argument('{"query": "foo.py traceback"}', "RecallCompactedContext")
        assert result == "foo.py traceback"

    def test_long_shell_command_not_truncated(self):
        long_command = "echo " + "x" * 80
        result = extract_key_argument(json.dumps({"command": long_command}), "Shell")
        assert result == long_command

    def test_invalid_json(self):
        result = extract_key_argument("invalid", "Shell")
        assert result is None

    def test_empty_json_object(self):
        result = extract_key_argument("{}", "Shell")
        assert result is None

    def test_long_content_not_truncated(self):
        long_url = "https://example.com/" + "a" * 200
        result = extract_key_argument(f'{{"url": "{long_url}"}}', "FetchURL")
        assert result is not None
        # shorten_middle truncation was removed; full URL is returned
        assert result == long_url

    def test_unknown_tool_returns_raw_content(self):
        result = extract_key_argument('{"a": 1}', "UnknownTool")
        assert result is not None
        assert result == '{"a": 1}'
