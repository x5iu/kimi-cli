from __future__ import annotations


def todo_label(title: str, subagent_name: str | None = None) -> str:
    label = title
    if subagent_name:
        label += f" @{subagent_name}"
    return label


def ready_to_execute_todo_text(title: str, subagent_name: str | None = None) -> str:
    return f"Ready to ExecuteTodo: {todo_label(title, subagent_name)}"


def completed_todo_text(title: str) -> str:
    return f"Completed Todo: {title}"


def blocked_todo_text(title: str) -> str:
    return f"Blocked Todo: {title}"


def todo_still_in_progress_text(title: str) -> str:
    return f"Todo Still In Progress: {title}"


def next_state_text(next_state: str) -> str:
    return f"Next state: {next_state}"


def execute_todo_suggested_next_step(next_state: str) -> str:
    if next_state == "blocked":
        return (
            "Suggested next step: inspect [Task output], revise the prompt, "
            "or update the todo state before retrying."
        )
    return (
        "Suggested next step: inspect [Task output] before retrying "
        "ExecuteTodo or updating the todo state."
    )


def execute_todo_failure_message(title: str, *, next_state: str) -> str:
    label = (
        blocked_todo_text(title)
        if next_state == "blocked"
        else todo_still_in_progress_text(title)
    )
    return f"{label}. Delegated Task failed."
