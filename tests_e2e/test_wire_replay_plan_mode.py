from __future__ import annotations

from inline_snapshot import snapshot

from tests_e2e.wire_helpers import (
    collect_until_response,
    make_home_dir,
    make_work_dir,
    send_initialize,
    start_wire,
    summarize_messages,
    write_scripted_config,
)


def test_replay_restores_plan_mode_status_events(tmp_path) -> None:
    scripts = ["text: hello in plan mode"]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
        extra_args=["--session", "replay-plan-mode"],
        yolo=True,
    )
    try:
        send_initialize(wire, capabilities={"supports_plan_mode": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "plan-on",
                "method": "set_plan_mode",
                "params": {"enabled": True},
            }
        )
        plan_on_event = wire.read_json()
        plan_on_resp = wire.read_json()
        assert plan_on_resp.get("id") == "plan-on"

        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "hello"},
            }
        )
        prompt_resp, _ = collect_until_response(wire, "prompt-1")
        assert prompt_resp.get("result", {}).get("status") == "finished"

        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "plan-off",
                "method": "set_plan_mode",
                "params": {"enabled": False},
            }
        )
        plan_off_event = wire.read_json()
        plan_off_resp = wire.read_json()
        assert plan_off_resp.get("id") == "plan-off"

        wire.send_json({"jsonrpc": "2.0", "id": "replay-1", "method": "replay"})
        replay_resp, replay_messages = collect_until_response(wire, "replay-1")
        assert replay_resp.get("result") == snapshot(
            {
                "status": "finished",
                "events": 7,
                "requests": 0,
            }
        )
        assert summarize_messages([plan_on_event, plan_off_event] + replay_messages) == snapshot(
            [
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
                },
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
                },
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
                },
                {"method": "event", "type": "TurnBegin", "payload": {"user_input": "hello"}},
                {"method": "event", "type": "StepBegin", "payload": {"n": 1}},
                {
                    "method": "event",
                    "type": "ContentPart",
                    "payload": {"type": "text", "text": "hello in plan mode"},
                },
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
                },
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
                },
                {"method": "event", "type": "TurnEnd", "payload": {}},
            ]
        )
    finally:
        wire.close()
