from __future__ import annotations

from tests_e2e.wire_helpers import (
    build_approval_response,
    build_shell_tool_call,
    collect_until_response,
    make_home_dir,
    make_work_dir,
    send_initialize,
    start_wire,
    summarize_messages,
    write_scripted_config,
)


def test_replay_restores_rejected_approval_history(tmp_path) -> None:
    scripts = [
        "\n".join(
            [
                "text: step1",
                build_shell_tool_call("tc-1", "echo ok"),
            ]
        ),
        "text: done",
    ]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
        extra_args=["--session", "approval-replay-session"],
        yolo=False,
    )
    try:
        send_initialize(wire)
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "run shell"},
            }
        )
        resp, _ = collect_until_response(
            wire,
            "prompt-1",
            request_handler=lambda msg: build_approval_response(msg, "reject"),
        )
        assert resp.get("result", {}).get("status") == "finished"

        wire.send_json({"jsonrpc": "2.0", "id": "replay-1", "method": "replay"})
        replay_resp, replay_messages = collect_until_response(wire, "replay-1")
        assert replay_resp.get("result") == {
            "status": "finished",
            "events": 8,
            "requests": 1,
        }

        summary = summarize_messages(replay_messages)
        approval_requests = [m for m in summary if m.get("type") == "ApprovalRequest"]
        assert len(approval_requests) == 1
        assert approval_requests[0]["payload"] == {
            "id": "<uuid>",
            "tool_call_id": "tc-1",
            "sender": "Shell",
            "action": "run command",
            "description": "Run command `echo ok`",
            "display": [{"type": "shell", "language": "bash", "command": "echo ok"}],
        }

        approval_responses = [m for m in summary if m.get("type") == "ApprovalResponse"]
        assert approval_responses == [
            {
                "method": "event",
                "type": "ApprovalResponse",
                "payload": {"request_id": "<uuid>", "response": "reject"},
            }
        ]

        tool_results = [m for m in summary if m.get("type") == "ToolResult"]
        assert len(tool_results) == 1
        rv = tool_results[0]["payload"]["return_value"]
        assert rv["is_error"] is True
        assert tool_results[0]["payload"]["tool_call_id"] == "tc-1"
        assert "rejected by the user" in rv["message"]
    finally:
        wire.close()


def test_replay_restores_approve_for_session_response(tmp_path) -> None:
    scripts = [
        "\n".join(
            [
                "text: step1",
                build_shell_tool_call("tc-1", "echo first"),
            ]
        ),
        "text: done",
        "\n".join(
            [
                "text: step1",
                build_shell_tool_call("tc-2", "echo second"),
            ]
        ),
        "text: done",
    ]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
        extra_args=["--session", "approval-replay-session-approve"],
        yolo=False,
    )
    try:
        send_initialize(wire)
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "run shell"},
            }
        )
        resp, _ = collect_until_response(
            wire,
            "prompt-1",
            request_handler=lambda msg: build_approval_response(msg, "approve_for_session"),
        )
        assert resp.get("result", {}).get("status") == "finished"

        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-2",
                "method": "prompt",
                "params": {"user_input": "run shell again"},
            }
        )
        resp, _ = collect_until_response(wire, "prompt-2")
        assert resp.get("result", {}).get("status") == "finished"

        wire.send_json({"jsonrpc": "2.0", "id": "replay-1", "method": "replay"})
        replay_resp, replay_messages = collect_until_response(wire, "replay-1")
        assert replay_resp.get("result") == {
            "status": "finished",
            "events": 23,
            "requests": 1,
        }

        summary = summarize_messages(replay_messages)
        approval_requests = [m for m in summary if m.get("type") == "ApprovalRequest"]
        assert len(approval_requests) == 1
        approval_responses = [m for m in summary if m.get("type") == "ApprovalResponse"]
        assert approval_responses == [
            {
                "method": "event",
                "type": "ApprovalResponse",
                "payload": {"request_id": "<uuid>", "response": "approve_for_session"},
            }
        ]
        second_turn_requests = [
            m
            for m in summary
            if m.get("type") == "ApprovalRequest" and m["payload"].get("tool_call_id") == "tc-2"
        ]
        assert second_turn_requests == []
    finally:
        wire.close()
