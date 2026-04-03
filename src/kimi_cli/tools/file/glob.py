"""Glob tool implementation."""

from pathlib import Path
from typing import override

from kaos.path import KaosPath
from llmkit.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.loop.agent import Runtime
from kimi_cli.tools.utils import load_desc
from kimi_cli.utils.path import is_within_directory, is_within_workspace

MAX_MATCHES = 1000


class Params(BaseModel):
    pattern: str = Field(description=("Glob pattern to match files/directories."))
    directory: str | None = Field(
        description=(
            "Absolute path to the directory to search in (defaults to working directory)."
        ),
        default=None,
    )
    include_dirs: bool = Field(
        description="Whether to include directories in results.",
        default=True,
    )


class Glob(CallableTool2[Params]):
    name: str = "Glob"
    description: str = load_desc(
        Path(__file__).parent / "glob.md",
        {
            "MAX_MATCHES": str(MAX_MATCHES),
        },
    )
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._skills_dirs = runtime.skills_dirs

    def _auto_correct_pattern(self, pattern: str) -> tuple[str, str | None]:
        """Auto-correct unsafe patterns.

        Returns (corrected_pattern, notice_or_none).
        """
        if pattern.startswith("**/"):
            corrected = pattern.replace("**/", "*/", 1)
            notice = (
                f"Pattern `{pattern}` is not allowed (would recursively scan all "
                "directories). Automatically changed to `{corrected}` (current "
                "directory only). Specify a directory like `src/**/*.ext` for "
                "deeper search."
            ).format(corrected=corrected)
            return corrected, notice
        if pattern.startswith("**"):
            # e.g. "**" or "**xyz" without slash
            corrected = pattern.replace("**", "*", 1)
            notice = (
                f"Pattern `{pattern}` is not allowed (would recursively scan all "
                "directories). Automatically changed to `{corrected}` (current "
                "directory only). Specify a directory like `src/**/*.ext` for "
                "deeper search."
            ).format(corrected=corrected)
            return corrected, notice
        return pattern, None

    async def _validate_directory(self, directory: KaosPath) -> ToolError | None:
        """Validate that the directory is safe to search."""
        resolved_dir = directory.canonical()

        # Allow directories within the workspace (work_dir or additional dirs)
        if is_within_workspace(resolved_dir, self._work_dir, self._additional_dirs):
            return None

        # Allow directories within any discovered skills root
        if any(is_within_directory(resolved_dir, d) for d in self._skills_dirs):
            return None

        return ToolError(
            message=(
                f"`{directory}` is outside the workspace. "
                "You can only search within the working directory, "
                "additional directories, and skills directories."
            ),
            brief="Directory outside workspace",
        )

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            # Auto-correct unsafe patterns instead of rejecting them
            pattern, correction_notice = self._auto_correct_pattern(params.pattern)

            dir_path = (
                KaosPath(params.directory).expanduser() if params.directory else self._work_dir
            )

            if not dir_path.is_absolute():
                return ToolError(
                    message=(
                        f"`{params.directory}` is not an absolute path. "
                        "You must provide an absolute path to search."
                    ),
                    brief="Invalid directory",
                )

            # Validate directory safety
            dir_error = await self._validate_directory(dir_path)
            if dir_error:
                return dir_error

            if not await dir_path.exists():
                return ToolError(
                    message=f"`{params.directory}` does not exist.",
                    brief="Directory not found",
                )
            if not await dir_path.is_dir():
                return ToolError(
                    message=f"`{params.directory}` is not a directory.",
                    brief="Invalid directory",
                )

            # Perform the glob search - users can use ** directly in pattern
            matches: list[KaosPath] = []
            async for match in dir_path.glob(pattern):
                matches.append(match)

            # Post-filter: ensure matched paths don't escape the base directory
            resolved_base = dir_path.canonical()
            matches = [m for m in matches if is_within_directory(m.canonical(), resolved_base)]

            # Filter out directories if not requested
            if not params.include_dirs:
                matches = [p for p in matches if await p.is_file()]

            # Sort for consistent output
            matches.sort()

            # Limit matches
            message = (
                f"Found {len(matches)} matches for pattern `{pattern}`."
                if len(matches) > 0
                else f"No matches found for pattern `{pattern}`."
            )
            if len(matches) > MAX_MATCHES:
                matches = matches[:MAX_MATCHES]
                message += (
                    f" Only the first {MAX_MATCHES} matches are returned. "
                    "You may want to use a more specific pattern."
                )

            output = "\n".join(str(p.relative_to(dir_path)) for p in matches)
            if correction_notice:
                output = correction_notice + "\n\n" + output if output else correction_notice

            return ToolOk(
                output=output,
                message=message,
            )

        except Exception as e:
            return ToolError(
                message=f"Failed to search for pattern {params.pattern}. Error: {e}",
                brief="Glob failed",
            )
