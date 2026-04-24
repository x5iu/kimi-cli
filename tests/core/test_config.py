from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from inline_snapshot import snapshot

from kimi_cli.config import (
    Config,
    get_default_config,
    load_config,
    load_config_from_string,
)
from kimi_cli.exception import ConfigError


def test_default_config():
    config = get_default_config()
    assert config == snapshot(Config())


def test_default_config_dump():
    config = get_default_config()
    assert config.model_dump() == snapshot(
        {
            "default_model": "",
            "default_thinking": False,
            "default_yolo": False,
            "default_editor": "",
            "models": {},
            "providers": {},
            "loop_control": {
                "max_steps_per_turn": 100,
                "max_retries_per_step": 3,
                "reserved_context_size": 50000,
                "max_preserved_messages": 2,
                "auto_compact_enabled": True,
                "compaction_trigger_ratio": 0.85,
                "turn_end_question_detection": True,
            },
            "background": {
                "max_running_tasks": 4,
                "read_max_bytes": 30000,
                "notification_tail_bytes": 3000,
                "wait_poll_interval_ms": 500,
                "worker_control_poll_interval_ms": 2000,
                "worker_heartbeat_interval_ms": 5000,
                "worker_stale_after_ms": 30000,
                "kill_grace_period_ms": 2000,
                "keep_alive_on_exit": True,
            },
            "notifications": {"claim_stale_after_ms": 15000},
            "services": {"moonshot_search": None, "moonshot_fetch": None},
            "mcp": {"tool_call_timeout_ms": 60000},
            "env": {},
        }
    )


def test_load_config_text_toml():
    config = load_config_from_string('default_model = ""\n')
    assert config == get_default_config()


def test_load_config_text_json():
    config = load_config_from_string('{"default_model": ""}')
    assert config == get_default_config()


def test_load_config_sets_source_file(tmp_path):
    config_file = tmp_path / "custom.toml"

    config = load_config(config_file)

    assert config.source_file == config_file.resolve()
    assert not config.is_from_default_location


def test_load_config_text_has_no_source_file():
    config = load_config_from_string('{"default_model": ""}')

    assert config.source_file is None


def test_load_config_text_invalid():
    with pytest.raises(ConfigError, match="Invalid configuration text"):
        load_config_from_string("not valid {")


def test_load_config_reserved_context_size():
    config = load_config_from_string('{"loop_control": {"reserved_context_size": 30000}}')
    assert config.loop_control.reserved_context_size == 30000


def test_load_config_max_steps_per_turn():
    config = load_config_from_string("[loop_control]\nmax_steps_per_turn = 42\n")
    assert config.loop_control.max_steps_per_turn == 42


def test_load_config_max_steps_per_run():
    config = load_config_from_string('{"loop_control": {"max_steps_per_run": 7}}')
    assert config.loop_control.max_steps_per_turn == 7


def test_load_config_mcp_client_legacy_timeout():
    config = load_config_from_string("[mcp.client]\ntool_call_timeout_ms = 123\n")
    assert config.mcp.tool_call_timeout_ms == 123


def test_load_config_mcp_flat_timeout():
    config = load_config_from_string("[mcp]\ntool_call_timeout_ms = 999\n")
    assert config.mcp.tool_call_timeout_ms == 999


def test_load_config_mcp_client_malformed_rejected():
    with pytest.raises(ConfigError, match="mcp.client must be a table"):
        load_config_from_string('{"mcp": {"client": 123}}')
    with pytest.raises(ConfigError, match="mcp.client must be a table"):
        load_config_from_string('{"mcp": {"client": "x"}}')
    with pytest.raises(ConfigError, match="mcp.client must be a table"):
        load_config_from_string('{"mcp": {"client": null}}')


def test_load_config_reserved_context_size_too_low():
    with pytest.raises(ConfigError, match="reserved_context_size"):
        load_config_from_string('{"loop_control": {"reserved_context_size": 500}}')


def test_load_config_compaction_trigger_ratio():
    config = load_config_from_string('{"loop_control": {"compaction_trigger_ratio": 0.8}}')
    assert config.loop_control.compaction_trigger_ratio == 0.8


def test_load_config_compaction_trigger_ratio_default():
    config = load_config_from_string("{}")
    assert config.loop_control.compaction_trigger_ratio == 0.85


def test_load_config_compaction_trigger_ratio_too_low():
    with pytest.raises(ConfigError, match="compaction_trigger_ratio"):
        load_config_from_string('{"loop_control": {"compaction_trigger_ratio": 0.3}}')


def test_load_config_compaction_trigger_ratio_too_high():
    with pytest.raises(ConfigError, match="compaction_trigger_ratio"):
        load_config_from_string('{"loop_control": {"compaction_trigger_ratio": 1.0}}')


def test_load_config_env():
    config = load_config_from_string('[env]\nFOO = "bar"\nBAZ = "qux"\n')
    assert config.env == {"FOO": "bar", "BAZ": "qux"}


def test_load_config_env_default():
    config = load_config_from_string("{}")
    assert config.env == {}


def _apply_config_env(config):
    """Mirror the env-injection logic from KimiCLI.create for testing."""
    from kimi_cli.utils.logging import logger

    for key, value in config.env.items():
        existing = os.environ.get(key)
        if existing:
            logger.warning("env var {} already set, skipping config override", key)
        else:
            os.environ[key] = value


def test_apply_config_env_sets_new_vars(monkeypatch):
    """New env vars from config should be injected into os.environ."""
    from kimi_cli.utils.logging import logger

    monkeypatch.delenv("_KIMI_TEST_NEW_VAR", raising=False)
    config = load_config_from_string('[env]\n_KIMI_TEST_NEW_VAR = "hello"\n')

    with patch.object(logger, "warning") as mock_warn:
        _apply_config_env(config)

    assert os.environ["_KIMI_TEST_NEW_VAR"] == "hello"
    mock_warn.assert_not_called()
    monkeypatch.delenv("_KIMI_TEST_NEW_VAR", raising=False)


def test_apply_config_env_does_not_overwrite_existing(monkeypatch):
    """Existing env vars must NOT be overwritten; a warning should be logged."""
    from kimi_cli.utils.logging import logger

    monkeypatch.setenv("_KIMI_TEST_EXISTING", "original")
    config = load_config_from_string('[env]\n_KIMI_TEST_EXISTING = "overridden"\n')

    with patch.object(logger, "warning") as mock_warn:
        _apply_config_env(config)

    assert os.environ["_KIMI_TEST_EXISTING"] == "original"
    mock_warn.assert_called_once_with(
        "env var {} already set, skipping config override", "_KIMI_TEST_EXISTING"
    )


def test_apply_config_env_overwrites_empty_value(monkeypatch):
    """Env vars that exist but are empty should be overwritten by config values."""
    from kimi_cli.utils.logging import logger

    monkeypatch.setenv("_KIMI_TEST_EMPTY", "")
    config = load_config_from_string('[env]\n_KIMI_TEST_EMPTY = "filled"\n')

    with patch.object(logger, "warning") as mock_warn:
        _apply_config_env(config)

    assert os.environ["_KIMI_TEST_EMPTY"] == "filled"
    mock_warn.assert_not_called()
