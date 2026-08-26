"""Command-line entry point: every local-llm command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console

from . import __version__
from .agents import AgentError, claude_env, copilot_env, exec_with_env, export_lines, resolve_model
from .doctor import run_checks
from .estimate import budget_bytes, estimate_bytes, human_gb
from .hardware import total_ram
from .logs import current_log, prune_logs, tail_lines
from .paths import Paths
from .preset import Preset, PresetError
from .router import Router, RouterError
from .settings import load_settings

app = typer.Typer(
    help="One llama.cpp router serving every model in models.ini.",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode=None,
)
out = Console(highlight=False, soft_wrap=True, markup=False)
err = Console(stderr=True, highlight=False, soft_wrap=True, markup=False)

# Hooks that tests replace.
_sleep = time.sleep
_open_url = webbrowser.open


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _run_editor(path: Path) -> int:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        return subprocess.call([*editor.split(), str(path)])
    if sys.platform == "darwin":
        return subprocess.call(["open", "-t", str(path)])
    return subprocess.call(["vi", str(path)])


def fail(message: str, code: int = 1) -> NoReturn:
    err.print(message)
    raise typer.Exit(code)


class State:
    """Paths, settings and factories for one invocation."""

    def __init__(self, overrides: dict[str, object] | None = None) -> None:
        self.paths = Paths.from_env()
        self.settings = load_settings(self.paths, overrides=overrides)

    def router(self) -> Router:
        return Router(self.paths, self.settings, log=out.print)

    def preset(self) -> Preset:
        try:
            return Preset.load(self.paths.preset)
        except PresetError as error:
            fail(str(error))


_state: State | None = None


def state() -> State:
    global _state
    if _state is None:
        _state = State()
    return _state


def complete_model(incomplete: str) -> list[str]:
    try:
        names = Preset.load(Paths.from_env().preset).sections()
    except PresetError:
        return []
    return [name for name in names if name.startswith(incomplete)]


def _split_agent_args(model: str | None, extra: list[str]) -> tuple[str | None, list[str]]:
    """A first argument starting with '-' belongs to the agent, not to us."""
    if model is not None and model.startswith("-"):
        return None, [model, *extra]
    return model, list(extra)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
    port: int | None = typer.Option(None, "--port", help="Router port (overrides settings and LOCAL_LLM_PORT)."),
    host: str | None = typer.Option(None, "--host", help="Router host (overrides settings and LOCAL_LLM_HOST)."),
) -> None:
    if version:
        out.print(f"local-llm {__version__}")
        raise typer.Exit()
    global _state
    _state = State({"port": port, "host": host})
    if ctx.invoked_subcommand is None:
        status()


# ---------------------------------------------------------------- router commands


def _print_up(st: State) -> None:
    s = st.settings
    out.print("Router is up.")
    out.print(f"  url:      {s.openai_base_url}")
    if s.ui:
        out.print(f"  web ui:   http://{s.host}:{s.port}/    (local-llm ui to open it)")
    out.print("  models:   local-llm models")
    out.print("  logs:     local-llm logs -f")


def _start(st: State, router: Router) -> None:
    try:
        result = router.start()
    except RouterError as error:
        fail(str(error))
    if result.already_running:
        out.print(f"Router is already running (pid {result.pid}).")
        out.print("  local-llm status    what is loaded right now")
        return
    _print_up(st)


def _stop(router: Router) -> None:
    result = router.stop()
    for line in result.refused:
        err.print(f"Refusing to stop {line} - does not look like llama-server")
    if not result.was_running:
        out.print("Router is not running.")
        return
    if result.orphans_cleaned:
        out.print(f"Cleaned up {result.orphans_cleaned} orphaned model server(s).")
    out.print("Router is down.")


@app.command()
def up(
    foreground: bool = typer.Option(False, "-f", "--foreground", help="Stay attached to the terminal."),
    ui: bool | None = typer.Option(None, "--ui/--no-ui", help="Serve llama.cpp's web UI as well."),
    max_models: int | None = typer.Option(
        None, "--max-models", min=0, help="How many models may stay loaded at once (0 = unlimited)."
    ),
) -> None:
    """Start the router in the background and write a timestamped log."""
    st = state()
    if ui is not None:
        st.settings.ui = ui
    if max_models is not None:
        st.settings.max_models = max_models
    router = st.router()
    if foreground:
        out.print("Starting router in the foreground. Ctrl-C to stop.")
        out.print(f"  config: {st.paths.preset}")
        out.print(f"  url:    {st.settings.openai_base_url}")
        out.print()
        try:
            code = router.run_foreground()
        except RouterError as error:
            fail(str(error))
        raise typer.Exit(code)
    _start(st, router)


@app.command()
def down() -> None:
    """Stop the router and any model servers under it."""
    _stop(state().router())


app.command(name="stop", hidden=True)(down)


@app.command()
def restart(
    ui: bool | None = typer.Option(None, "--ui/--no-ui", help="Change the web UI mode."),
    restore: bool = typer.Option(True, "--restore/--no-restore", help="Reload the models that were loaded."),
    max_models: int | None = typer.Option(
        None, "--max-models", min=0, help="How many models may stay loaded at once (0 = unlimited)."
    ),
) -> None:
    """Stop and start the router, keeping the UI mode and reloading resident models."""
    st = state()
    router = st.router()
    st.settings.ui = router.ui_state() if ui is None else ui
    if max_models is not None:
        st.settings.max_models = max_models
    keep = router.loaded_model_names() if restore else []
    _stop(router)
    _sleep(1)
    _start(st, router)
    if keep:
        out.print()
        out.print(f"Restoring {len(keep)} model(s) that were loaded:")
        for name in keep:
            try:
                ok = router.load_model(name).get("success") is True
            except RouterError:
                ok = False
            out.print(f"  {name} ... {'ok' if ok else 'failed'}")


@app.command()
def status() -> None:
    """Is it running, what is resident, what to run next."""
    st = state()
    router = st.router()
    s = st.settings
    pid = router.pid()
    out.print("Local LLM router")
    out.print()
    if pid is None:
        out.print("  state:    stopped")
        out.print(f"  config:   {st.paths.preset}")
        out.print()
        out.print("  local-llm up        start it")
        out.print("  local-llm models    what it would serve")
        return
    try:
        health = json.dumps(router.health(), separators=(",", ":"))
    except RouterError:
        health = "unavailable"
    out.print("  state:    running")
    out.print(f"  pid:      {pid}")
    out.print(f"  url:      {s.openai_base_url}")
    if router.ui_state():
        out.print(f"  web ui:   http://{s.host}:{s.port}/")
    else:
        out.print("  web ui:   off  (local-llm restart --ui)")
    out.print(f"  health:   {health}")
    out.print(f"  config:   {st.paths.preset}")
    out.print()
    kids = router.children()
    if kids:
        out.print("  loaded models:")
        for kid in kids:
            note = "  (asleep)" if kid.asleep else ""
            out.print(f"    {kid.model or '?':<32} pid {kid.pid}  {human_gb(kid.rss)}{note}")
    else:
        out.print("  loaded models:  none resident")
    out.print()
    out.print("  local-llm logs -f        follow the log")
    out.print("  local-llm models         list every model in models.ini")
    out.print("  local-llm load <model>   preload one so the first prompt is fast")


@app.command()
def logs(
    follow: bool = typer.Option(False, "-f", "--follow", help="Keep printing as the log grows."),
    lines: int = typer.Option(200, "-n", "--lines", help="How many lines to show first."),
) -> None:
    """Show the last lines of the current log, or follow it."""
    st = state()
    path = current_log(st.paths.log_dir)
    if path is None:
        fail(f"No log yet in {st.paths.log_dir}\nStart the router first: local-llm up")
    for line in tail_lines(path, lines):
        out.print(line)
    if not follow:
        out.print()
        out.print("  local-llm logs -f    follow live")
        return
    with path.open(errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        try:
            while True:
                chunk = handle.readline()
                if chunk:
                    out.print(chunk.rstrip("\n"))
                else:
                    _sleep(0.5)
        except KeyboardInterrupt:
            return


def _open(url: str) -> None:
    if not _open_url(url):
        err.print(f"Could not open a browser. Visit {url}")


@app.command()
def ui(yes: bool = typer.Option(False, "-y", "--yes", help="Restart without asking if needed.")) -> None:
    """Open the web UI in a browser, starting or reconfiguring the router as needed."""
    st = state()
    router = st.router()
    url = f"http://{st.settings.host}:{st.settings.port}/"
    if router.pid() is None:
        out.print("Router is not running. Starting it with the web UI.")
        st.settings.ui = True
        _start(st, router)
        _open(url)
        return
    if router.ui_state():
        out.print(f"Opening {url}")
        _open(url)
        return
    loaded = router.loaded_model_names()
    out.print("The router is running without the web UI, and it can only be enabled at startup.")
    out.print("Restarting takes a few seconds.")
    if loaded:
        out.print("Currently loaded, and will be reloaded afterwards:")
        for name in loaded:
            out.print(f"  {name}")
    out.print()
    if not yes:
        if not _interactive():
            fail("Not an interactive shell. Run: local-llm restart --ui")
        if not typer.confirm("Restart with the web UI now?", default=False):
            out.print("Left running as it is.")
            raise typer.Exit(1)
    # Called as a plain function: pass every parameter, or typer's Option objects leak in.
    restart(ui=True, restore=True, max_models=None)
    _open(url)


@app.command(name="models")
def models_cmd() -> None:
    """List every model name in models.ini."""
    st = state()
    preset = st.preset()
    names = preset.sections()
    if not names:
        fail(f"No models defined in {st.paths.preset}")
    out.print(f"Models in {st.paths.preset}")
    out.print()
    for name in names:
        try:
            size = human_gb(sum(preset.file_sizes(name)))
        except PresetError as error:
            size = f"(missing: {str(error).split(': ')[-1]})"
        out.print(f"  {name:<44} {size}")
    out.print()
    out.print('Pass one as the "model" field of any OpenAI request - no restart needed.')
    out.print("  local-llm load <model>    load it now instead of on first use")


app.command(name="ls", hidden=True)(models_cmd)


@app.command()
def edit() -> None:
    """Open models.ini in $EDITOR."""
    raise typer.Exit(_run_editor(state().paths.preset))


@app.command(name="prune-logs")
def prune_logs_cmd(
    days: int = typer.Argument(30, help="Delete rotated logs older than this many days."),
) -> None:
    """Delete rotated logs older than N days (default 30)."""
    st = state()
    st.paths.ensure_state_dirs()
    removed = prune_logs(st.paths.log_dir, days)
    out.print(f"Removed {removed} log file(s) older than {days} days.")


# ---------------------------------------------------------------- model commands (Task 15)
