from __future__ import annotations

from tests_e2e.wire_helpers import (
    collect_until_response,
    make_home_dir,
    make_work_dir,
    normalize_response,
    read_response,
    send_initialize,
    start_wire,
    summarize_messages,
    write_scripted_config,
)


def _set_plan_mode(wire, *, msg_id: str, enabled: bool) -> tuple[dict, dict]:
    wire.send_json(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "set_plan_mode",
            "params": {"enabled": enabled},
        }
    )
    event = wire.read_json()
    response = normalize_response(read_response(wire, msg_id))
    return event, response


def test_wire_set_plan_mode_status_matches_subsequent_turns(tmp_path) -> None:
    scripts = ["text: first response", "text: second response"]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
        yolo=True,
    )
    try:
        send_initialize(wire, capabilities={"supports_plan_mode": True})

        event_on, response_on = _set_plan_mode(wire, msg_id="plan-on", enabled=True)
        assert summarize_messages([event_on]) == [
            {
                "method": "event",
                "type": "StatusUpdate",
                "payload": {
                    "context_usage": None,
                    "context_tokens": None,
                    "max_context_tokens": None,
                    "token_usage": None,
                    "message_id": None,
                    "plan_mode": True,
                },
            }
        ]
        assert response_on == {"result": {"status": "ok", "plan_mode": True}}

        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-on",
                "method": "prompt",
                "params": {"user_input": "hello"},
            }
        )
        prompt_on_resp, prompt_on_messages = collect_until_response(wire, "prompt-on")
        assert prompt_on_resp.get("result", {}).get("status") == "finished"
        assert any(
            msg["type"] == "StatusUpdate" and msg["payload"].get("plan_mode") is True
            for msg in summarize_messages(prompt_on_messages)
        )

        event_off, response_off = _set_plan_mode(wire, msg_id="plan-off", enabled=False)
        assert summarize_messages([event_off]) == [
            {
                "method": "event",
                "type": "StatusUpdate",
                "payload": {
                    "context_usage": None,
                    "context_tokens": None,
                    "max_context_tokens": None,
                    "token_usage": None,
                    "message_id": None,
                    "plan_mode": False,
                },
            }
        ]
        assert response_off == {"result": {"status": "ok", "plan_mode": False}}

        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-off",
                "method": "prompt",
                "params": {"user_input": "hello again"},
            }
        )
        prompt_off_resp, prompt_off_messages = collect_until_response(wire, "prompt-off")
        assert prompt_off_resp.get("result", {}).get("status") == "finished"
        assert any(
            msg["type"] == "StatusUpdate" and msg["payload"].get("plan_mode") is False
            for msg in summarize_messages(prompt_off_messages)
        )
    finally:
        wire.close()
