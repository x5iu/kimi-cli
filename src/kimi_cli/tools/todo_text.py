from __future__ import annotations


def todo_label(title: str) -> str:
    return title


def completed_todo_text(title: str) -> str:
    return f"Completed Todo: {title}"


def blocked_todo_text(title: str) -> str:
    return f"Blocked Todo: {title}"


def todo_still_in_progress_text(title: str) -> str:
    return f"Todo Still In Progress: {title}"
