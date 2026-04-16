from enum import StrEnum


class FileActions(StrEnum):
    READ = "read file"
    EDIT = "edit file"
    EDIT_OUTSIDE = "edit file outside of working directory"


from .read import ReadFile  # noqa: E402
from .read_media import ReadMediaFile  # noqa: E402
from .replace import EditTool as Edit  # noqa: E402
from .write import WriteFile  # noqa: E402

__all__ = (
    "ReadFile",
    "ReadMediaFile",
    "WriteFile",
    "Edit",
)
