from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer

from kimi_cli.constant import VERSION

from .export import attach_export_command
from .mcp import cli as mcp_cli


class Reload(Exception):
    """Reload configuration."""

    def __init__(self, session_id: str | None = None):
        super().__init__("reload")
        self.session_id = session_id


cli = typer.Typer(
    epilog="""\b\
Documentation:        https://moonshotai.github.io/kimi-cli/\n
LLM friendly version: https://moonshotai.github.io/kimi-cli/llms.txt""",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Kimi, your next CLI agent.",
)

UIMode = Literal["shell", "print"]


class ExitCode:
    SUCCESS = 0
    FAILURE = 1
    RETRYABLE = 75  # EX_TEMPFAIL from sysexits.h


InputFormat = Literal["text", "stream-json"]
OutputFormat = Literal["text", "stream-json"]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"kimi, version {VERSION}")
        raise typer.Exit()


@cli.callback(invoke_without_command=True)
def kimi(
    ctx: typer.Context,
    # Meta
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Log debug information. Default: no.",
        ),
    ] = False,
    # Basic configuration
    local_work_dir: Annotated[
        Path | None,
        typer.Option(
            "--work-dir",
            "-w",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            writable=True,
            help="Working directory for the agent. Default: current directory.",
        ),
    ] = None,
    local_add_dirs: Annotated[
        list[Path] | None,
        typer.Option(
            "--add-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help=(
                "Add an additional directory to the workspace scope. "
                "Can be specified multiple times."
            ),
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session",
            "-S",
            help="Session ID to resume for the working directory. Default: new session.",
        ),
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option(
            "--continue",
            "-C",
            help="Continue the previous session for the working directory. Default: no.",
        ),
    ] = False,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Config TOML/JSON file to load. Default: ~/.kimi/config.toml.",
        ),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use. Default: default model set in config file.",
        ),
    ] = None,
    thinking: Annotated[
        bool | None,
        typer.Option(
            "--thinking/--no-thinking",
            help="Enable thinking mode. Default: default thinking mode set in config file.",
        ),
    ] = None,
    # Run mode
    yolo: Annotated[
        bool,
        typer.Option(
            "--yolo",
            "-y",
            help="Automatically approve all actions. Default: no.",
        ),
    ] = False,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="User prompt to the agent. Default: prompt interactively.",
        ),
    ] = None,
    print_mode: Annotated[
        bool,
        typer.Option(
            "--print",
            help=(
                "Run in print mode (non-interactive). Note: print mode implicitly adds `--yolo`."
            ),
        ),
    ] = False,
    input_format: Annotated[
        InputFormat | None,
        typer.Option(
            "--input-format",
            help=(
                "Input format to use. Must be used with `--print` "
                "and the input must be piped in via stdin. "
                "Default: text."
            ),
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat | None,
        typer.Option(
            "--output-format",
            help="Output format to use. Must be used with `--print`. Default: text.",
        ),
    ] = None,
    final_message_only: Annotated[
        bool,
        typer.Option(
            "--final-message-only",
            help="Only print the final assistant message (print UI).",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            help="Alias for `--print --output-format text --final-message-only`.",
        ),
    ] = False,
    # Customization
    agent_file: Annotated[
        Path | None,
        typer.Option(
            "--agent-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Custom agent specification file. Default: builtin default agent.",
        ),
    ] = None,
    mcp_config_file: Annotated[
        list[Path] | None,
        typer.Option(
            "--mcp-config-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help=(
                "MCP config file to load. Add this option multiple times to specify multiple MCP "
                "configs. Default: none."
            ),
        ),
    ] = None,
    local_skills_dir: Annotated[
        Path | None,
        typer.Option(
            "--skills-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Path to the skills directory. Overrides discovery.",
        ),
    ] = None,
    # Loop control
    max_steps_per_turn: Annotated[
        int | None,
        typer.Option(
            "--max-steps-per-turn",
            min=1,
            help="Maximum number of steps in one turn. Default: from config.",
        ),
    ] = None,
):
    """Kimi, your next CLI agent."""
    from kimi_cli.utils.proxy import normalize_proxy_env

    normalize_proxy_env()

    from kimi_cli.utils.proctitle import init_process_name

    init_process_name("Kimi Code")

    if ctx.invoked_subcommand is not None:
        return  # skip rest if a subcommand is invoked

    del version  # handled in the callback

    from kaos.path import KaosPath
    from kimi_cli.app import KimiCLI, enable_logging
    from kimi_cli.config import Config
    from kimi_cli.metadata import load_metadata, save_metadata
    from kimi_cli.session import Session
    from kimi_cli.utils.logging import logger, open_original_stderr, redirect_stderr_to_logger

    from .mcp import get_global_mcp_config_file

    # Don't redirect stderr yet. Our stderr redirector replaces fd=2 with a pipe, which
    # would swallow Click/Typer startup errors (e.g. config parsing / BadParameter).
    # We re-enable stderr redirection after KimiCLI.create() succeeds.
    enable_logging(debug, redirect_stderr=False)

    def _emit_fatal_error(message: str) -> None:
        # Prefer writing to the original stderr fd even if we later redirect fd=2.
        # This ensures fatal errors are visible to the user.
        with open_original_stderr() as stream:
            if stream is not None:
                stream.write((message.rstrip() + "\n").encode("utf-8", errors="replace"))
                stream.flush()
                return
        typer.echo(message, err=True)

    if session_id is not None:
        session_id = session_id.strip()
        if not session_id:
            raise typer.BadParameter("Session ID cannot be empty", param_hint="--session")

    if quiet:
        if output_format not in (None, "text"):
            raise typer.BadParameter(
                "Quiet mode implies `--output-format text`",
                param_hint="--quiet",
            )
        print_mode = True
        output_format = "text"
        final_message_only = True

    conflict_option_sets = [
        {
            "--print": print_mode,
        },
        {
            "--continue": continue_,
            "--session": session_id is not None,
        },
    ]
    for option_set in conflict_option_sets:
        active_options = [flag for flag, active in option_set.items() if active]
        if len(active_options) > 1:
            raise typer.BadParameter(
                f"Cannot combine {', '.join(active_options)}.",
                param_hint=active_options[0],
            )

    ui: UIMode = "shell"
    if print_mode:
        ui = "print"

    if prompt is not None:
        prompt = prompt.strip()
        if not prompt:
            raise typer.BadParameter("Prompt cannot be empty", param_hint="--prompt")

    if input_format is not None and ui != "print":
        raise typer.BadParameter(
            "Input format is only supported for print UI",
            param_hint="--input-format",
        )
    if output_format is not None and ui != "print":
        raise typer.BadParameter(
            "Output format is only supported for print UI",
            param_hint="--output-format",
        )
    if final_message_only and ui != "print":
        raise typer.BadParameter(
            "Final-message-only output is only supported for print UI",
            param_hint="--final-message-only",
        )

    config: Config | Path | None = None
    if config_file is not None:
        config = config_file

    file_configs = list(mcp_config_file or [])

    # Use default MCP config file if no MCP config is provided
    if not file_configs:
        default_mcp_file = get_global_mcp_config_file()
        if default_mcp_file.exists():
            file_configs.append(default_mcp_file)

    try:
        mcp_configs = [json.loads(conf.read_text(encoding="utf-8")) for conf in file_configs]
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON: {e}", param_hint="--mcp-config-file") from e

    skills_dir: KaosPath | None = None
    if local_skills_dir is not None:
        skills_dir = KaosPath.unsafe_from_local_path(local_skills_dir)

    work_dir = KaosPath.unsafe_from_local_path(local_work_dir) if local_work_dir else KaosPath.cwd()

    async def _run(session_id: str | None) -> tuple[Session, bool]:
        """
        Create/load session and run the CLI instance.

        Returns:
            The session and whether the run succeeded.
        """
        if session_id is not None:
            session = await Session.find(work_dir, session_id)
            if session is None:
                logger.info(
                    "Session {session_id} not found, creating new session", session_id=session_id
                )
                session = await Session.create(work_dir, session_id)
            logger.info("Switching to session: {session_id}", session_id=session.id)
        elif continue_:
            session = await Session.continue_(work_dir)
            if session is None:
                raise typer.BadParameter(
                    "No previous session found for the working directory",
                    param_hint="--continue",
                )
            logger.info("Continuing previous session: {session_id}", session_id=session.id)
        else:
            session = await Session.create(work_dir)
            logger.info("Created new session: {session_id}", session_id=session.id)

        # Add CLI-provided additional directories to session state
        if local_add_dirs:
            from kimi_cli.utils.path import is_within_directory

            canonical_work_dir = work_dir.canonical()
            changed = False
            for d in local_add_dirs:
                dir_path = KaosPath.unsafe_from_local_path(d).canonical()
                dir_str = str(dir_path)
                # Skip dirs within work_dir (already accessible)
                if is_within_directory(dir_path, canonical_work_dir):
                    logger.info(
                        "Skipping --add-dir {dir}: already within working directory",
                        dir=dir_str,
                    )
                    continue
                if dir_str not in session.state.additional_dirs:
                    session.state.additional_dirs.append(dir_str)
                    changed = True
            if changed:
                session.save_state()

        # Redirect stderr *before* KimiCLI.create() so that MCP server
        # subprocesses (e.g. mcp-remote OAuth debug logs) write to the log
        # file instead of polluting the user's terminal.  CLI argument
        # parsing has already succeeded at this point, so Typer/Click
        # startup errors are no longer a concern.  Fatal errors from
        # create() are still visible because _emit_fatal_error() writes to
        # the saved original stderr fd.
        redirect_stderr_to_logger()

        instance = await KimiCLI.create(
            session,
            config=config,
            model_name=model_name,
            thinking=thinking,
            yolo=yolo or (ui == "print"),  # print mode implies yolo
            agent_file=agent_file,
            mcp_configs=mcp_configs,
            skills_dir=skills_dir,
            max_steps_per_turn=max_steps_per_turn,
        )
        try:
            match ui:
                case "shell":
                    succeeded = await instance.run_shell(prompt)
                case "print":
                    exit_code = await instance.run_print(
                        input_format or "text",
                        output_format or "text",
                        prompt,
                        final_only=final_message_only,
                    )
                    succeeded = exit_code == ExitCode.SUCCESS
        except Reload as e:
            if e.session_id is None:
                raise Reload(session_id=session.id) from e
            raise
        finally:
            instance.shutdown_background_tasks()

        return session, succeeded

    async def _post_run(last_session: Session, succeeded: bool) -> None:
        if not succeeded:
            return

        metadata = load_metadata()

        # Update work_dir metadata with last session
        work_dir_meta = metadata.get_work_dir_meta(last_session.work_dir)

        if work_dir_meta is None:
            logger.warning(
                "Work dir metadata missing when marking last session, recreating: {work_dir}",
                work_dir=last_session.work_dir,
            )
            work_dir_meta = metadata.new_work_dir_meta(last_session.work_dir)

        if last_session.is_empty():
            logger.info(
                "Session {session_id} has empty context, removing it",
                session_id=last_session.id,
            )
            await last_session.delete()
            if work_dir_meta.last_session_id == last_session.id:
                work_dir_meta.last_session_id = None
        else:
            work_dir_meta.last_session_id = last_session.id

        save_metadata(metadata)

    async def _reload_loop(session_id: str | None) -> None:
        while True:
            try:
                last_session, succeeded = await _run(session_id)
                break
            except Reload as e:
                session_id = e.session_id
                continue
        await _post_run(last_session, succeeded)

    try:
        asyncio.run(_reload_loop(session_id))
    except (typer.BadParameter, typer.Exit):
        # Let Typer/Click format these errors (rich panel + correct exit code).
        raise
    except Exception as exc:
        import click

        if isinstance(exc, click.ClickException):
            # ClickException includes the errors Typer knows how to render; don't
            # wrap them, or we'd lose the standard error UI and exit codes.
            raise
        logger.exception("Fatal error when running CLI")
        if debug:
            import traceback

            # In debug mode, show full traceback for quick diagnosis.
            _emit_fatal_error(traceback.format_exc())
        else:
            from kimi_cli.share import get_share_dir

            log_path = get_share_dir() / "logs" / "kimi.log"
            # In non-debug mode, print a concise error and point users to logs.
            _emit_fatal_error(f"{exc}\nSee logs: {log_path}\nRun with --debug for full traceback.")
        raise typer.Exit(code=1) from exc


@cli.command(name="__background-task-worker", hidden=True)
def background_task_worker(
    task_dir: Annotated[Path, typer.Option("--task-dir")],
    heartbeat_interval_ms: Annotated[int, typer.Option("--heartbeat-interval-ms")] = 5000,
    control_poll_interval_ms: Annotated[int, typer.Option("--control-poll-interval-ms")] = 2000,
    kill_grace_period_ms: Annotated[int, typer.Option("--kill-grace-period-ms")] = 2000,
) -> None:
    """Run background task worker subprocess (internal)."""
    from kimi_cli.background import run_background_task_worker
    from kimi_cli.utils.proctitle import set_process_title

    set_process_title("kimi-code-bg-worker")

    from kimi_cli.app import enable_logging

    enable_logging(debug=False)
    asyncio.run(
        run_background_task_worker(
            task_dir,
            heartbeat_interval_ms=heartbeat_interval_ms,
            control_poll_interval_ms=control_poll_interval_ms,
            kill_grace_period_ms=kill_grace_period_ms,
        )
    )


attach_export_command(cli)

cli.add_typer(mcp_cli, name="mcp")


if __name__ == "__main__":
    if "kimi_cli.cli" not in sys.modules:
        sys.modules["kimi_cli.cli"] = sys.modules[__name__]

    sys.exit(cli())
