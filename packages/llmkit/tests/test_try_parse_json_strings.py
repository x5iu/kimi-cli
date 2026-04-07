# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportIndexIssue=false, reportCallIssue=false, reportPrivateUsage=false, reportUnknownVariableType=false, reportMissingTypeArgument=false
"""Tests for _try_parse_json_strings with model-aware str field skipping."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llmkit.tooling import _try_parse_json_strings


class _StrModel(BaseModel):
    text: str = ""
    data: dict = {}


class _StrNoneModel(BaseModel):
    query: str | None = None


class _AnnotatedStrModel(BaseModel):
    input: str = Field(default="", max_length=1_000_000)


# --- model=None (legacy behavior) ---


def test_no_model_parses_json_object_string():
    args = {"text": '{"type": "user"}'}
    result = _try_parse_json_strings(args)
    assert result["text"] == {"type": "user"}


def test_no_model_parses_json_array_string():
    args = {"items": "[1, 2, 3]"}
    result = _try_parse_json_strings(args)
    assert result["items"] == [1, 2, 3]


def test_no_model_leaves_plain_string():
    args = {"text": "hello world"}
    result = _try_parse_json_strings(args)
    assert result["text"] == "hello world"


def test_no_model_leaves_invalid_json():
    args = {"text": "{invalid json}"}
    result = _try_parse_json_strings(args)
    assert result["text"] == "{invalid json}"


# --- model with str field (new skip behavior) ---


def test_str_field_not_parsed():
    """A field annotated as `str` should NOT be parsed even if the value is valid JSON."""
    args = {"text": '{"type": "user", "message": "hello"}'}
    result = _try_parse_json_strings(args, model=_StrModel)
    assert isinstance(result["text"], str)
    assert result["text"] == '{"type": "user", "message": "hello"}'


def test_str_field_with_json_array_not_parsed():
    args = {"text": "[1, 2, 3]"}
    result = _try_parse_json_strings(args, model=_StrModel)
    assert isinstance(result["text"], str)
    assert result["text"] == "[1, 2, 3]"


def test_annotated_str_field_not_parsed():
    """Annotated[str, Field(...)] should also be skipped — Pydantic unwraps it."""
    args = {"input": '{"type":"user","message":{"role":"user","content":[{"type":"text"}]}}'}
    result = _try_parse_json_strings(args, model=_AnnotatedStrModel)
    assert isinstance(result["input"], str)


# --- non-str fields still parsed ---


def test_dict_field_still_parsed():
    """A field annotated as `dict` should still have its JSON string parsed."""
    args = {"data": '{"key": "value"}'}
    result = _try_parse_json_strings(args, model=_StrModel)
    assert result["data"] == {"key": "value"}


# --- unknown fields (not in model) fall through ---


def test_unknown_field_still_parsed():
    """Fields not present in the model should get the old auto-parse behavior."""
    args = {"unknown": '{"a": 1}'}
    result = _try_parse_json_strings(args, model=_StrModel)
    assert result["unknown"] == {"a": 1}


# --- str | None gap (documented behavior) ---


def test_str_none_field_is_parsed():
    """str | None fields are NOT detected as str — they get parsed.

    This is a known limitation: `field_info.annotation is str` is False for
    `str | None`. Documented here as expected behavior so any future change
    to handle this case will be caught.
    """
    args = {"query": '{"filter": "active"}'}
    result = _try_parse_json_strings(args, model=_StrNoneModel)
    # Current behavior: parsed into dict (not skipped)
    assert result["query"] == {"filter": "active"}


# --- non-dict input passthrough ---


def test_non_dict_passthrough():
    assert _try_parse_json_strings("hello") == "hello"
    assert _try_parse_json_strings(42) == 42
    assert _try_parse_json_strings([1, 2]) == [1, 2]
    assert _try_parse_json_strings(None) is None


# --- mixed fields in one call ---


def test_mixed_fields():
    """str field skipped, dict field parsed, unknown field parsed — all in one call."""
    args = {
        "text": '{"skip": true}',
        "data": '{"parse": true}',
        "extra": '{"also": "parsed"}',
    }
    result = _try_parse_json_strings(args, model=_StrModel)
    assert isinstance(result, dict)
    assert isinstance(result["text"], str)
    assert result["data"] == {"parse": True}
    assert result["extra"] == {"also": "parsed"}
