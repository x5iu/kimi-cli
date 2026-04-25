from __future__ import annotations

import math

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.compaction import estimate_text_tokens
from kimi_cli.loop.compaction_archive import stringify_tool_calls
from llmkit.message import Message, ToolCall


def test_estimate_text_tokens_cjk_counts_per_char() -> None:
    ascii_part = "a" * 7
    cjk = "你好世界"
    msg = Message(role="user", content=[TextPart(text=ascii_part + cjk)])
    cjk_n = sum(1 for c in ascii_part + cjk if "\u4e00" <= c <= "\u9fff")
    non_cjk = len(ascii_part + cjk) - cjk_n
    expected = 4 + math.ceil(non_cjk / 4) + cjk_n
    assert estimate_text_tokens([msg]) == expected


def test_estimate_text_tokens_includes_assistant_tool_calls() -> None:
    tc = ToolCall(
        id="c1",
        function=ToolCall.FunctionBody(name="ReadFile", arguments='{"path": "/x"}'),
    )
    msg = Message(role="assistant", content=[TextPart(text="hi")], tool_calls=[tc])
    tc_text = stringify_tool_calls([tc])
    from kimi_cli.loop.compaction import _estimate_text_fragment_tokens

    expected = 4 + math.ceil(2 / 4) + _estimate_text_fragment_tokens(tc_text)
    assert estimate_text_tokens([msg]) == expected


def test_estimate_text_tokens_hiragana_katakana_in_message_text() -> None:
    from kimi_cli.loop.compaction import _estimate_text_fragment_tokens

    text = "ひらがなとカタカナ"
    msg = Message(role="user", content=[TextPart(text=text)])
    assert estimate_text_tokens([msg]) == 4 + _estimate_text_fragment_tokens(text)


def test_estimate_text_tokens_hangul_in_tool_arguments() -> None:
    args = '{"q": "한글검색어"}'
    tc = ToolCall(
        id="c1",
        function=ToolCall.FunctionBody(name="search", arguments=args),
    )
    msg = Message(role="assistant", content=[], tool_calls=[tc])
    from kimi_cli.loop.compaction import _estimate_text_fragment_tokens

    tc_text = stringify_tool_calls([tc])
    assert estimate_text_tokens([msg]) == 4 + _estimate_text_fragment_tokens(tc_text)
