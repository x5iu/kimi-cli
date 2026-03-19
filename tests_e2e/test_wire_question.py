"""E2E tests for AskUserQuestion via the Wire protocol."""

from __future__ import annotations

import json
from typing import Any

from .wire_helpers import (
    build_ask_user_tool_call,
    build_question_response,
    collect_until_response,
    make_home_dir,
    make_work_dir,
    send_initialize,
    start_wire,
    summarize_messages,
    write_scripted_config,
)


def _make_question(
    question: str = "Which option?",
    options: list[dict[str, str]] | None = None,
    multi_select: bool = False,
) -> dict[str, Any]:
    if options is None:
        options = [
            {"label": "Alpha", "description": "First"},
            {"label": "Beta", "description": "Second"},
        ]
    return {
        "question": question,
        "header": "Test",
        "options": options,
        "multi_select": multi_select,
    }


def _question_request_handler(answers: dict[str, str]):
    """Return a request_handler that responds to QuestionRequest with the given answers."""

    def handler(msg: dict[str, Any]) -> dict[str, Any]:
        params = msg.get("params", {})
        msg_type = params.get("type")
        if msg_type == "QuestionRequest":
            return build_question_response(msg, answers)
        # For other request types (e.g. approval), just approve
        from .wire_helpers import build_approval_response

        return build_approval_response(msg, "approve")

    return handler


def test_question_request_answer(tmp_path) -> None:
    """Test normal question → answer flow through Wire protocol."""
    question = _make_question()
    scripts = [
        "\n".join(
            [
                "text: asking",
                build_ask_user_tool_call("tc-q1", [question]),
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
        yolo=True,  # yolo to skip approval for other tools
    )
    try:
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "ask me"},
            }
        )

        answers = {"Which option?": "Alpha"}
        resp, messages = collect_until_response(
            wire,
            "prompt-1",
            request_handler=_question_request_handler(answers),
        )
        assert resp.get("result", {}).get("status") == "finished"

        # Verify the QuestionRequest was sent
        summary = summarize_messages(messages)
        question_requests = [m for m in summary if m.get("type") == "QuestionRequest"]
        assert len(question_requests) == 1
        qr_payload = question_requests[0]["payload"]
        assert qr_payload["tool_call_id"] == "tc-q1"
        assert len(qr_payload["questions"]) == 1
        assert qr_payload["questions"][0]["question"] == "Which option?"

        # Verify the ToolResult contains the answers
        tool_results = [m for m in summary if m.get("type") == "ToolResult"]
        assert len(tool_results) >= 1
        for tr in tool_results:
            rv = tr["payload"]["return_value"]
            if tr["payload"]["tool_call_id"] == "tc-q1":
                assert not rv["is_error"]
                output = json.loads(rv["output"])
                assert output == {"answers": {"Which option?": "Alpha"}}
                break
        else:
            raise AssertionError("ToolResult for tc-q1 not found")
    finally:
        wire.close()


def test_turn_end_question_request_uses_heuristic_and_runs_follow_up(tmp_path) -> None:
    scripts = [
        "text: 如果你要，我可以继续直接做下去。",
        "text: 收到，我继续处理。",
    ]
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
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-heuristic-1",
                "method": "prompt",
                "params": {"user_input": "继续"},
            }
        )

        resp, messages = collect_until_response(
            wire,
            "prompt-heuristic-1",
            request_handler=_question_request_handler({"要我继续吗？": "继续"}),
        )
        assert resp.get("result", {}).get("status") == "finished"

        summary = summarize_messages(messages)
        question_requests = [m for m in summary if m.get("type") == "QuestionRequest"]
        assert len(question_requests) == 1
        question_payload = question_requests[0]["payload"]
        assert question_payload["tool_call_id"].startswith("turn-end-")
        assert question_payload["questions"] == [
            {
                "question": "要我继续吗？",
                "header": "",
                "options": [
                    {"label": "继续", "description": "继续按当前方案往下做"},
                    {"label": "先别", "description": "先不要继续"},
                ],
                "multi_select": False,
                "body": "如果你要，我可以继续直接做下去。",
                "other_label": "",
                "other_description": "",
            }
        ]

        follow_ups = [m for m in summary if m.get("type") == "FollowUpInput"]
        assert follow_ups == [
            {"method": "event", "type": "FollowUpInput", "payload": {"text": "继续"}}
        ]

        content_parts = [m for m in summary if m.get("type") == "ContentPart"]
        assert [part["payload"] for part in content_parts] == [
            {"type": "text", "text": "如果你要，我可以继续直接做下去。"},
            {"type": "text", "text": "收到，我继续处理。"},
        ]
    finally:
        wire.close()


def test_turn_end_question_request_handles_if_continue_phrase(tmp_path) -> None:
    scripts = [
        "text: 如果继续，我可以先处理 A。",
        "text: 收到，我先处理 A。",
    ]
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
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-heuristic-2",
                "method": "prompt",
                "params": {"user_input": "继续"},
            }
        )

        resp, messages = collect_until_response(
            wire,
            "prompt-heuristic-2",
            request_handler=_question_request_handler({"要我继续吗？": "继续"}),
        )
        assert resp.get("result", {}).get("status") == "finished"

        summary = summarize_messages(messages)
        question_requests = [m for m in summary if m.get("type") == "QuestionRequest"]
        assert len(question_requests) == 1
        question_payload = question_requests[0]["payload"]
        assert question_payload["tool_call_id"].startswith("turn-end-")
        assert question_payload["questions"] == [
            {
                "question": "要我继续吗？",
                "header": "",
                "options": [
                    {"label": "继续", "description": "继续按当前方案往下做"},
                    {"label": "先别", "description": "先不要继续"},
                ],
                "multi_select": False,
                "body": "如果继续，我可以先处理 A。",
                "other_label": "",
                "other_description": "",
            }
        ]

        follow_ups = [m for m in summary if m.get("type") == "FollowUpInput"]
        assert follow_ups == [
            {"method": "event", "type": "FollowUpInput", "payload": {"text": "继续"}}
        ]

        content_parts = [m for m in summary if m.get("type") == "ContentPart"]
        assert [part["payload"] for part in content_parts] == [
            {"type": "text", "text": "如果继续，我可以先处理 A。"},
            {"type": "text", "text": "收到，我先处理 A。"},
        ]
    finally:
        wire.close()


def test_turn_end_question_request_ignores_conditional_analysis(tmp_path) -> None:
    scripts = ["text: 如果继续这样做，风险会更高。"]
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
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-heuristic-3",
                "method": "prompt",
                "params": {"user_input": "继续"},
            }
        )

        resp, messages = collect_until_response(
            wire,
            "prompt-heuristic-3",
            request_handler=_question_request_handler({"要我继续吗？": "继续"}),
        )
        assert resp.get("result", {}).get("status") == "finished"

        summary = summarize_messages(messages)
        question_requests = [m for m in summary if m.get("type") == "QuestionRequest"]
        assert question_requests == []

        follow_ups = [m for m in summary if m.get("type") == "FollowUpInput"]
        assert follow_ups == []

        content_parts = [m for m in summary if m.get("type") == "ContentPart"]
        assert [part["payload"] for part in content_parts] == [
            {"type": "text", "text": "如果继续这样做，风险会更高。"}
        ]
    finally:
        wire.close()


def test_question_request_error_response(tmp_path) -> None:
    """Test that a JSON-RPC error response resolves to empty answers without crash."""
    question = _make_question()
    scripts = [
        "\n".join(
            [
                "text: asking",
                build_ask_user_tool_call("tc-q2", [question]),
            ]
        ),
        "text: done after error",
    ]
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
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "ask me"},
            }
        )

        def error_handler(msg: dict[str, Any]) -> dict[str, Any]:
            params = msg.get("params", {})
            msg_type = params.get("type")
            if msg_type == "QuestionRequest":
                return {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "error": {"code": -32000, "message": "User cancelled"},
                }
            from .wire_helpers import build_approval_response

            return build_approval_response(msg, "approve")

        resp, messages = collect_until_response(
            wire,
            "prompt-1",
            request_handler=error_handler,
        )
        # Should complete normally (not crash)
        assert resp.get("result", {}).get("status") == "finished"

        # Verify the ToolResult has empty answers (error response resolves to {})
        summary = summarize_messages(messages)
        tool_results = [m for m in summary if m.get("type") == "ToolResult"]
        for tr in tool_results:
            if tr["payload"]["tool_call_id"] == "tc-q2":
                rv = tr["payload"]["return_value"]
                assert not rv["is_error"]
                output = json.loads(rv["output"])
                assert output["answers"] == {}
                assert "dismissed" in output.get("note", "").lower()
                break
        else:
            raise AssertionError("ToolResult for tc-q2 not found")
    finally:
        wire.close()


def test_question_capability_negotiation(tmp_path) -> None:
    """Test that the server reports supports_question in its capabilities."""
    scripts = ["text: hello"]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
    )
    try:
        resp = send_initialize(wire, capabilities={"supports_question": True})
        result = resp.get("result", {})
        # Server should always report supports_question: True
        caps = result.get("capabilities", {})
        assert caps.get("supports_question") is True
    finally:
        wire.close()


def test_ask_user_tool_hidden_when_question_not_supported(tmp_path) -> None:
    """When question support is disabled, AskUserQuestion should not emit QuestionRequest."""
    question = _make_question()
    scripts = [
        "\n".join(
            [
                "text: asking",
                build_ask_user_tool_call("tc-q-hidden", [question]),
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
        yolo=True,
    )
    try:
        # Initialize WITHOUT supports_question (defaults to false)
        send_initialize(wire, capabilities={"supports_question": False})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "ask me"},
            }
        )

        resp, messages = collect_until_response(
            wire,
            "prompt-1",
            request_handler=_question_request_handler({}),
        )
        assert resp.get("result", {}).get("status") == "finished"

        # The client does not support QuestionRequest, so no QuestionRequest should be emitted.
        summary = summarize_messages(messages)
        question_requests = [m for m in summary if m.get("type") == "QuestionRequest"]
        assert len(question_requests) == 0, (
            "AskUserQuestion tool should be hidden when client does not support questions"
        )

        # The scripted AskUserQuestion call should complete with a tool error indicating
        # the client cannot handle interactive questions.
        tool_results = [m for m in summary if m.get("type") == "ToolResult"]
        for tr in tool_results:
            if tr["payload"]["tool_call_id"] != "tc-q-hidden":
                continue
            rv = tr["payload"]["return_value"]
            assert rv["is_error"] is True
            assert "does not support interactive questions" in rv["message"]
            assert "Do NOT call this tool again" in rv["message"]
            break
        else:
            raise AssertionError("ToolResult for tc-q-hidden not found")
    finally:
        wire.close()


def test_replay_restores_question_request_history(tmp_path) -> None:
    question = _make_question()
    scripts = [
        "\n".join(
            [
                "text: asking",
                build_ask_user_tool_call("tc-q-replay", [question]),
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
        extra_args=["--session", "question-replay-session"],
        yolo=True,
    )
    try:
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "ask me"},
            }
        )
        resp, _ = collect_until_response(
            wire,
            "prompt-1",
            request_handler=_question_request_handler({"Which option?": "Alpha"}),
        )
        assert resp.get("result", {}).get("status") == "finished"

        wire.send_json({"jsonrpc": "2.0", "id": "replay-1", "method": "replay"})
        replay_resp, replay_messages = collect_until_response(wire, "replay-1")
        assert replay_resp.get("result") == {
            "status": "finished",
            "events": 10,
            "requests": 1,
        }

        summary = summarize_messages(replay_messages)
        question_requests = [m for m in summary if m.get("type") == "QuestionRequest"]
        assert len(question_requests) == 1
        payload = question_requests[0]["payload"]
        assert payload["id"] == "<uuid>"
        assert payload["tool_call_id"] == "tc-q-replay"
        assert payload["questions"] == [
            {
                "question": "Which option?",
                "header": "Test",
                "options": [
                    {"label": "Alpha", "description": "First"},
                    {"label": "Beta", "description": "Second"},
                ],
                "multi_select": False,
                "body": "",
                "other_label": "",
                "other_description": "",
            }
        ]

        tool_results = [m for m in summary if m.get("type") == "ToolResult"]
        for tr in tool_results:
            if tr["payload"]["tool_call_id"] != "tc-q-replay":
                continue
            rv = tr["payload"]["return_value"]
            assert rv["is_error"] is False
            assert json.loads(rv["output"]) == {"answers": {"Which option?": "Alpha"}}
            assert rv["message"] == "User has answered."
            break
        else:
            raise AssertionError("ToolResult for tc-q-replay not found")
    finally:
        wire.close()
