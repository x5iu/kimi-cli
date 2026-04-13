from __future__ import annotations


class InputAction:
    __slots__ = ("kind", "args")

    BTW = "btw"
    QUEUE = "queue"
    SEND = "send"
    IGNORED = "ignored"

    def __init__(self, kind: str, args: str = "") -> None:
        self.kind = kind
        self.args = args


def classify_input(text: str, *, is_streaming: bool) -> InputAction:
    from kimi_cli.utils.slashcmd import parse_slash_command_call

    if (cmd := parse_slash_command_call(text.strip())) and cmd.name == "btw":
        if cmd.args.strip():
            return InputAction(InputAction.BTW, cmd.args.strip())
        return InputAction(InputAction.IGNORED, "Usage: /btw <question>")

    if is_streaming:
        return InputAction(InputAction.QUEUE)
    return InputAction(InputAction.SEND)
