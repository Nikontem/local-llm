"""Command-line entry point: every local-llm command."""

from __future__ import annotations

import difflib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console

from . import __version__
from .agents import AgentError, claude_env, copilot_env, exec_with_env, export_lines, resolve_model
from .discover import GROUPS, Candidate, gather
from .discover import search as discover_search
from .doctor import Check, Env, run_checks  # noqa: F401 - Check re-exported for tests
from .estimate import budget_bytes, estimate_bytes, human_gb
from .gguf import GgufError, read_header, refined_estimate
from .hardware import Machine, detect, total_ram
from .hub import Hub, HubCache, HubError, free_disk_bytes, hf_cache_dir
from .integrations.opencode import (
    agent_snippet,
    current_tiny_model,
    install_plugin,
    merge_agent,
    opencode_paths,
    plugin_source,
    plugin_status,
    tiny_agent,
)
from .logs import current_log, prune_logs, tail_lines
from .paths import Paths
from .preset import Preset, PresetError
from .quant import QuantError, QuantOption, find_option, quant_options, suggest
from .router import Router, RouterError
from .sampling import values_for
from .sections import build_section, local_name, section_name
from .settings import load_settings
from .setup import Io, SetupContext, run_setup, smallest_model
from .shellrc import (
    alias_lines,
    detect_shell,
    install_completion,
    rc_file,
    remove_block,
    retire_old_source,
    upsert_block,
)
from .uninstall import (
    KEYS,
    delete_model_file,
    inventory,
    remove_config,
    remove_integrations,
    remove_models,
    remove_state,
    tool_uninstall_hint,
)

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
    port: int | None = typer.Option(
        None, "--port", help="Router port (overrides settings and LOCAL_LLM_PORT)."
    ),
    host: str | None = typer.Option(
        None, "--host", help="Router host (overrides settings and LOCAL_LLM_HOST)."
    ),
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
    foreground: bool = typer.Option(
        False, "-f", "--foreground", help="Stay attached to the terminal."
    ),
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
    restore: bool = typer.Option(
        True, "--restore/--no-restore", help="Reload the models that were loaded."
    ),
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
    statuses: dict[str, str] = {}
    try:
        for entry in router.list_models().get("data", []):
            statuses[str(entry.get("id"))] = str((entry.get("status") or {}).get("value", ""))
    except RouterError:
        pass
    if kids:
        out.print("  loaded models:")
        for kid in kids:
            value = statuses.get(kid.model or "")
            if value == "loading":
                note = "  (loading)"
            elif value == "sleeping" or (value is None and kid.asleep):
                note = "  (asleep)"
            else:
                note = ""
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
def ui(
    yes: bool = typer.Option(False, "-y", "--yes", help="Restart without asking if needed."),
) -> None:
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
    width = max(len(name) for name in names)
    for name in names:
        try:
            size = human_gb(sum(preset.file_sizes(name)))
        except PresetError as error:
            size = f"(missing: {str(error).split(': ')[-1]})"
        out.print(f"  {name:<{width}}  {size}")
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


@app.command()
def load(
    models: list[str] = typer.Argument(  # noqa: B008 - typer needs the call as the default
        ..., autocompletion=complete_model, help="Model names from models.ini."
    ),
    force: bool = typer.Option(False, "--force", help="Load even if over the memory budget."),
) -> None:
    """Load one or more models now; refuses if they will not fit."""
    st = state()
    router = st.router()
    preset = st.preset()
    if router.pid() is None:
        fail("Router is not running. Start it first: local-llm up")
    for name in models:
        if name not in preset.sections():
            fail(f"Unknown model: {name}\n  local-llm models    to see what is available")
    limit = st.settings.max_models
    if limit and len(models) > limit:
        fail(
            f"Asked for {len(models)} models but the router holds at most {limit}.\n"
            "The first would be evicted as the last loaded. Raise it with:\n"
            f"  local-llm restart --max-models {len(models)}\n"
            f"  or set LOCAL_LLM_MAX_MODELS={len(models)}, or max_models in settings.toml"
        )
    budget = budget_bytes(total_ram(), st.settings.reserve_gb)
    out.print("Requested:")
    total = 0
    for name in models:
        try:
            sizes = preset.file_sizes(name)
        except PresetError as error:
            fail(f"  {name} - cannot find its model file, refusing to guess\n  {error}")
        header = None
        try:
            header = read_header(Path(preset.get(name, "model", fallback_to_star=False) or ""))
        except GgufError:
            header = None
        context = int(preset.get(name, "c") or preset.get(name, "ctx-size") or 0)
        if header is not None and context <= 0:
            context = header.context_length
        cache_type = preset.get(name, "cache-type-k") or "f16"
        if header is not None and context > 0:
            estimate = refined_estimate(sizes, header, context, cache_type)
            detail = f"weights + KV cache at c={context}"
        else:
            estimate = estimate_bytes(sizes)
            detail = "weights plus headroom"
        total += estimate
        out.print(f"  {name}  ~{human_gb(estimate)}  ({detail})")
    out.print()
    out.print(f"  estimated total:  {human_gb(total)}")
    if budget > 0:
        reserved = st.settings.reserve_gb
        out.print(f"  usable memory:    {human_gb(budget)}  (RAM minus {reserved} GB reserved)")
    if budget > 0 and total > budget:
        out.print()
        over = human_gb(total - budget)
        if force:
            out.print(f"Over budget by {over} - loading anyway because --force was given.")
            out.print("Expect swapping.")
        else:
            fail(
                f"Refusing to load: over budget by {over}.\n\n"
                "  load them one at a time, and let the idle one sleep, or\n"
                f"  local-llm load {' '.join(models)} --force    to do it anyway"
            )
    out.print()
    for name in models:
        try:
            reply = router.load_model(name)
        except RouterError as error:
            fail(str(error))
        out.print(f"loading {name} ... {json.dumps(reply, separators=(',', ':'))}")
    out.print()
    out.print("  local-llm status    confirm what is resident")


@app.command()
def unload(model: str = typer.Argument(..., autocompletion=complete_model)) -> None:
    """Release a model immediately rather than waiting for it to go idle."""
    st = state()
    router = st.router()
    if router.pid() is None:
        fail("Router is not running. Start it first: local-llm up")
    try:
        reply = router.unload_model(model)
    except RouterError as error:
        fail(str(error))
    out.print(json.dumps(reply, separators=(",", ":")))


# ---------------------------------------------------------------- agents

_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


@app.command(context_settings=_PASSTHROUGH)
def claude(
    ctx: typer.Context,
    model: str | None = typer.Argument(
        None, autocompletion=complete_model, help="Model name; default from settings."
    ),
) -> None:
    """Run Claude Code against the router. Arguments after -- go to claude."""
    st = state()
    preset = st.preset()
    model, extra = _split_agent_args(model, ctx.args)
    try:
        name = resolve_model(model, preset, st.settings)
        env = claude_env(name, preset, st.settings)
        context = env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "unknown")
        out.print(f"claude -> {name} (context {context}) at {st.settings.anthropic_base_url}")
        exec_with_env("claude", extra, env)
    except AgentError as error:
        fail(str(error))


@app.command(context_settings=_PASSTHROUGH)
def copilot(
    ctx: typer.Context,
    model: str | None = typer.Argument(
        None, autocompletion=complete_model, help="Model name; default from settings."
    ),
    online: bool = typer.Option(False, "--online/--offline", help="Let copilot reach the network."),
) -> None:
    """Run GitHub Copilot CLI against the router. Arguments after -- go to copilot."""
    st = state()
    preset = st.preset()
    model, extra = _split_agent_args(model, ctx.args)
    try:
        name = resolve_model(model, preset, st.settings)
        env = copilot_env(name, preset, st.settings, offline=not online)
        context = env.get("COPILOT_PROVIDER_MAX_PROMPT_TOKENS", "unknown")
        out.print(f"copilot -> {name} (context {context}, offline={'false' if online else 'true'})")
        exec_with_env("copilot", extra, env)
    except AgentError as error:
        fail(str(error))


@app.command()
def env(
    model: str | None = typer.Argument(
        None, autocompletion=complete_model, help="Model name; default from settings."
    ),
    shell: str = typer.Option("zsh", "--shell", help="zsh, bash or fish."),
) -> None:
    """Print export lines that point any OpenAI- or Anthropic-style tool at the router."""
    st = state()
    preset = st.preset()
    try:
        name = resolve_model(model, preset, st.settings)
    except AgentError as error:
        fail(str(error))
    out.print(export_lines(name, preset, st.settings, st.paths.preset, shell=shell), end="")


# ---------------------------------------------------------------- doctor


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Offer to run the safe fixes."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Check prerequisites, config and the port; print the fix for anything wrong."""
    st = state()
    checks = run_checks(st.paths, st.settings, router_pid=st.router().pid())
    failed = any(check.status == "fail" for check in checks)
    if json_out:
        out.print(json.dumps([asdict(check) for check in checks], indent=2))
        raise typer.Exit(1 if failed else 0)
    marks = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}
    for check in checks:
        out.print(f"  {marks[check.status]}  {check.name:<16} {check.detail}")
        if check.fix and check.status != "ok":
            out.print(f"        {'':<16} fix: {check.fix}")
    if fix:
        for check in checks:
            if check.fix_cmd and check.status != "ok":
                if typer.confirm(f"Run: {' '.join(check.fix_cmd)} ?", default=True):
                    subprocess.call(check.fix_cmd)
    raise typer.Exit(1 if failed else 0)


# ---------------------------------------------------------------- discovery


def _make_hub(st: State, refresh: bool = False) -> Hub:
    cache = HubCache(st.paths.hub_cache_file)
    if refresh:
        cache.clear()
    return Hub(cache=cache)


def _machine(st: State) -> Machine:
    return detect(st.settings.reserve_gb)


def _preset_or_none(st: State) -> Preset | None:
    try:
        return Preset.load(st.paths.preset)
    except PresetError:
        return None


def _marks(candidate: Candidate) -> str:
    marks = []
    if candidate.thinking:
        marks.append("thinking")
    if candidate.vision:
        marks.append("vision")
    if candidate.gated:
        marks.append("gated")
    return f" ({', '.join(marks)})" if marks else ""


def _state_note(candidate: Candidate) -> str:
    notes = []
    if candidate.configured:
        notes.append("in models.ini")
    elif candidate.downloaded:
        notes.append("downloaded")
    return f"  · {', '.join(notes)}" if notes else ""


def _candidate_line(candidate: Candidate, machine: Machine) -> str:
    option = candidate.suggested
    if option is None:
        return (
            f"{candidate.display_name}{_marks(candidate)}  {candidate.repo_id}  (no files listed)"
        )
    return (
        f"{candidate.display_name}{_marks(candidate)}  {candidate.repo_id}  {option.label}"
        f"  {human_gb(option.total)} on disk, ~{human_gb(candidate.estimate)} in memory"
        f"  {machine.fit_label(candidate.estimate)}  {candidate.lineage}{_state_note(candidate)}"
    )


def _candidate_json(candidate: Candidate) -> dict:
    return {
        "repo_id": candidate.repo_id,
        "base_model": candidate.base_model,
        "display_name": candidate.display_name,
        "lineage": candidate.lineage,
        "downloads": candidate.downloads,
        "likes": candidate.likes,
        "params": candidate.params,
        "groups": candidate.groups,
        "thinking": candidate.thinking,
        "vision": candidate.vision,
        "gated": candidate.gated,
        "suggested": candidate.suggested.label if candidate.suggested else None,
        "fit": candidate.fit,
        "estimate": candidate.estimate,
        "downloaded": candidate.downloaded,
        "configured": candidate.configured,
        "also_from": candidate.also_from,
        "options": [
            {"label": o.label, "tag": o.tag, "files": o.files, "size": o.size, "mmproj": o.mmproj,
             "total": o.total, "estimate": estimate_bytes([o.total])}
            for o in candidate.options
        ],
    }


def _choose_option(
    options: list[QuantOption], suggested: QuantOption | None, machine: Machine, *, yes: bool
) -> QuantOption:
    if suggested is None:
        fail("No quantization to choose from.")
    if yes:
        return suggested
    ordered = [suggested] + [o for o in options if o is not suggested]
    out.print("  quantizations available (suggested first):")
    for index, option in enumerate(ordered, start=1):
        estimate = estimate_bytes([option.total])
        marker = "  <- suggested for this machine" if option is suggested else ""
        out.print(
            f"  {index:>2}. {option.label:<12} {human_gb(option.total):>9} on disk"
            f"  ~{human_gb(estimate):>9}  {machine.fit_label(estimate)}{marker}"
        )
    choice = typer.prompt("Quantization", default="1")
    try:
        option = ordered[int(choice) - 1]
    except (ValueError, IndexError):
        fail(f"Pick a number between 1 and {len(ordered)}")
    estimate = estimate_bytes([option.total])
    if machine.fit(estimate) == "too_big":
        short = human_gb(estimate - machine.budget)
        if not typer.confirm(
            f"{option.label} needs about {human_gb(estimate)}, {short} more than the"
            f" {human_gb(machine.budget)} usable here. Continue anyway?",
            default=False,
        ):
            raise typer.Exit(1)
    return option


@app.command()
def recommend(
    use: str | None = typer.Option(None, "--use", help="coding, general, small or vision."),
    include_finetunes: bool = typer.Option(
        False, "--include-finetunes", help="Also show community fine-tunes and merges."
    ),
    limit: int = typer.Option(3, "--limit", help="Models per use case."),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore the 24-hour cache."),
    pick: bool = typer.Option(False, "--pick", help="Choose one interactively and download it."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Models that suit this machine, computed from live Hugging Face data."""
    st = state()
    if use is not None and use not in GROUPS:
        fail(f"--use must be one of: {', '.join(GROUPS)}")
    machine = _machine(st)
    hub = _make_hub(st, refresh)
    preset = _preset_or_none(st)

    def progress(repo: str) -> None:
        if not json_out and err.is_terminal:
            err.print(f"  looking at {repo}", end="\r")

    try:
        groups = gather(
            hub, machine, preset, include_finetunes=include_finetunes,
            limit_per_group=limit, on_progress=progress,
        )
    except HubError as error:
        note = ""
        if not st.paths.hub_cache_file.is_file():
            note = (
                f"\n  No cached listings yet at {st.paths.hub_cache_file};"
                " the first run needs the network."
            )
        fail(f"{error}{note}")
    wanted = [use] if use else list(GROUPS)
    if json_out:
        data = {g: [_candidate_json(c) for c in groups[g]] for g in wanted}
        out.print(json.dumps(data, indent=2))
        return
    out.print(machine.describe())
    out.print()
    numbered: list[Candidate] = []
    for group in wanted:
        out.print(group)
        if not groups[group]:
            out.print("  nothing found that fits")
        for candidate in groups[group]:
            numbered.append(candidate)
            out.print(f"  {len(numbered):>2}. {_candidate_line(candidate, machine)}")
        out.print()
    if not pick:
        out.print("  local-llm recommend --pick         choose one and download it")
        out.print("  local-llm pull <repo>[:QUANT]      or name it yourself")
        return
    if not numbered:
        fail("Nothing to pick from.")
    choice = typer.prompt("Number to download (q to quit)", default="q")
    if choice.strip().lower() == "q":
        raise typer.Exit()
    try:
        candidate = numbered[int(choice) - 1]
    except (ValueError, IndexError):
        fail(f"Pick a number between 1 and {len(numbered)}")
    option = _choose_option(candidate.options, candidate.suggested, machine, yes=False)
    _pull(st, hub, machine, candidate.repo_id, option, name=None, context=None, extra=[],
          no_tuning=False, yes=True)


@app.command()
def search(
    text: str = typer.Argument(..., help="Words to look for in repository names."),
    limit: int = typer.Option(20, "--limit", help="How many repositories to show."),
    author: str | None = typer.Option(None, "--author", help="Only this organization or user."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Search GGUF repositories on Hugging Face; shows what fits this machine."""
    st = state()
    machine = _machine(st)
    hub = _make_hub(st)
    try:
        results = discover_search(
            hub, machine, _preset_or_none(st), text=text, limit=limit, author=author
        )
    except HubError as error:
        fail(str(error))
    if json_out:
        out.print(json.dumps([_candidate_json(c) for c in results], indent=2))
        return
    if not results:
        out.print(f"No GGUF repositories match {text!r}.")
        return
    for candidate in results:
        out.print(
            f"{candidate.repo_id}  {candidate.downloads:,} downloads  {candidate.likes:,} likes"
            f"  {candidate.lineage}{_marks(candidate)}{_state_note(candidate)}"
        )
        for option in candidate.options:
            estimate = estimate_bytes([option.total])
            out.print(
                f"    {option.label:<12} {human_gb(option.total):>9}"
                f"  ~{human_gb(estimate):>9}  {machine.fit_label(estimate)}"
            )
        if candidate.suggested is not None:
            out.print(f"    local-llm pull {candidate.repo_id}:{candidate.suggested.label}")
        out.print()


# ---------------------------------------------------------------- pull, add, remove


def template_text() -> str:
    return resources.files("local_llm").joinpath("resources/models.template.ini").read_text()


def _preset_or_new(st: State) -> Preset:
    try:
        return Preset.load(st.paths.preset)
    except PresetError:
        return Preset.parse(template_text())


def _split_repo(text: str) -> tuple[str, str | None]:
    repo, _, tag = text.partition(":")
    if repo.count("/") != 1 or not all(repo.split("/")):
        fail(
            f"{text!r} is not a repository id. Expected org/repo or org/repo:QUANT, for"
            " example unsloth/Qwen3.8-27B-GGUF:Q4_K_XL"
        )
    return repo, (tag or None)


def _parse_sets(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            fail(f"--set expects KEY=VALUE, got {item!r}")
        pairs.append((key.strip(), value.strip()))
    return pairs


def _after_preset_change(st: State, section: str, *, added: bool = True) -> None:
    router = st.router()
    if router.pid() is None:
        if added:
            out.print("  local-llm up    to start serving it")
        return
    try:
        router.list_models(reload=True)
        out.print("  the running router reloaded its model list")
        if added:
            out.print(
                f"  local-llm load {shlex.quote(section)}    to load it ahead of the first request"
            )
    except RouterError:
        out.print("  local-llm restart    so the running router sees the change")


def _pull(
    st: State, hub: Hub, machine: Machine, repo_id: str, option: QuantOption, *,
    name: str | None, context: int | None, extra: list[tuple[str, str]], no_tuning: bool, yes: bool,
) -> None:
    preset = _preset_or_new(st)
    section = name or section_name(repo_id, option.primary)
    if preset.has_section(section):
        fail(
            f"Section [{section}] already exists in {st.paths.preset}.\n"
            f"  local-llm remove {shlex.quote(section)}    to replace it, or --name <other>"
        )
    files = list(option.files) + ([option.mmproj] if option.mmproj else [])
    estimate = estimate_bytes([option.total])
    out.print(f"{repo_id}  {option.label}")
    for file in files:
        out.print(f"  {file}")
    out.print(f"  size on disk:    {human_gb(option.total)}")
    out.print(
        f"  estimated need:  {human_gb(estimate)}  ({machine.fit_label(estimate)};"
        f" {human_gb(machine.budget)} usable here)"
    )
    out.print(f"  section name:    {section}")
    cache_dir = hf_cache_dir()
    needed = option.total * 105 // 100
    free = free_disk_bytes(cache_dir)
    if free < needed:
        fail(
            f"Not enough disk space in {cache_dir}: need {human_gb(needed)}, {human_gb(free)} free."
        )
    if not yes and not typer.confirm("Download and add it?", default=True):
        raise typer.Exit(1)
    try:
        paths = hub.download(repo_id, files, progress=lambda file: out.print(f"downloading {file}"))
    except HubError as error:
        fail(str(error))
    model_path = paths[0]
    mmproj_path = paths[len(option.files)] if option.mmproj else None
    header = None
    try:
        header = read_header(model_path)
    except GgufError as error:
        err.print(f"  could not read the GGUF header ({error}); using a default context")
    sampling = values_for(
        repo_id, preset_ini=hub.preset_ini(repo_id), card=hub.model_card(repo_id),
        no_tuning=no_tuning,
    )
    plan = build_section(
        name=section, model_path=model_path, mmproj_path=mmproj_path, total_bytes=option.total,
        header=header, machine=machine, sampling=sampling, context=context, extra=extra,
        no_tuning=no_tuning, origin=f"{repo_id} ({option.label}, {human_gb(option.size)})",
    )
    preset.add_section(plan.name, plan.keys, plan.comments)
    preset.save(st.paths.preset)
    out.print()
    out.print(f"Added [{plan.name}] to {st.paths.preset}")
    for comment in plan.comments[1:]:
        out.print(f"  {comment}")
    _after_preset_change(st, plan.name)


@app.command()
def pull(
    repo: str = typer.Argument(..., help="org/repo or org/repo:QUANT from huggingface.co"),
    quant: str | None = typer.Option(None, "--quant", help="Quantization label, e.g. UD-Q4_K_XL."),
    file: str | None = typer.Option(
        None, "--file", help="Exact file name when tags are ambiguous."
    ),
    name: str | None = typer.Option(
        None, "--name", help="Section name (default: the id llama-server uses)."
    ),
    context: int | None = typer.Option(
        None, "--context", help="Context length; default is computed for this machine."
    ),
    set_: list[str] = typer.Option(  # noqa: B008 - typer needs the call as the default
        [], "--set", help="KEY=VALUE written into the section; repeatable."
    ),
    no_tuning: bool = typer.Option(
        False, "--no-tuning", help="Only model, mmproj, c and n-predict."
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Accept the suggestion and do not ask."),
) -> None:
    """Download a model from Hugging Face and add a tuned section to models.ini."""
    st = state()
    repo_id, tag = _split_repo(repo)
    quant = quant or tag
    extra = _parse_sets(set_)
    machine = _machine(st)
    hub = _make_hub(st)
    try:
        files = hub.repo_files(repo_id)
    except HubError as error:
        fail(str(error))
    options = quant_options(files.files)
    if not options:
        fail(
            f"{repo_id} has no GGUF model files.\n"
            f"  local-llm search {shlex.quote(repo_id.split('/')[-1])}    to find a GGUF build"
        )
    try:
        if quant or file:
            option = find_option(options, quant=quant, file=file)
        else:
            suggested, _ = suggest(options, machine.budget)
            option = _choose_option(options, suggested, machine, yes=yes)
    except QuantError as error:
        fail(str(error))
    _pull(st, hub, machine, repo_id, option, name=name, context=context, extra=extra,
          no_tuning=no_tuning, yes=yes)


@app.command()
def add(
    path: Path = typer.Argument(  # noqa: B008 - typer needs the call as the default
        ..., exists=True, dir_okay=False, help="A .gguf file on this machine."
    ),
    mmproj: Path | None = typer.Option(  # noqa: B008 - typer needs the call as the default
        None, "--mmproj", exists=True, dir_okay=False, help="Vision projector file."
    ),
    name: str | None = typer.Option(None, "--name", help="Section name (default: the file name)."),
    context: int | None = typer.Option(None, "--context"),
    set_: list[str] = typer.Option(  # noqa: B008 - typer needs the call as the default
        [], "--set", help="KEY=VALUE written into the section; repeatable."
    ),
    no_tuning: bool = typer.Option(False, "--no-tuning"),
) -> None:
    """Add a GGUF file you already have to models.ini."""
    st = state()
    machine = _machine(st)
    preset = _preset_or_new(st)
    section = name or local_name(path)
    if preset.has_section(section):
        fail(f"Section [{section}] already exists in {st.paths.preset}. Use --name <other>.")
    total = path.stat().st_size + (mmproj.stat().st_size if mmproj else 0)
    header = None
    try:
        header = read_header(path)
    except GgufError as error:
        err.print(f"  could not read the GGUF header ({error}); using a default context")
    sampling = values_for(local_name(path), card="", no_tuning=no_tuning)
    plan = build_section(
        name=section, model_path=path.resolve(), mmproj_path=mmproj.resolve() if mmproj else None,
        total_bytes=total, header=header, machine=machine, sampling=sampling, context=context,
        extra=_parse_sets(set_), no_tuning=no_tuning, origin=str(path),
    )
    preset.add_section(plan.name, plan.keys, plan.comments)
    preset.save(st.paths.preset)
    out.print(f"Added [{plan.name}] to {st.paths.preset}")
    for comment in plan.comments[1:]:
        out.print(f"  {comment}")
    _after_preset_change(st, plan.name)


@app.command()
def remove(
    model: str = typer.Argument(..., autocompletion=complete_model),
    delete_files: bool = typer.Option(False, "--delete-files", help="Also delete the model files."),
    yes: bool = typer.Option(False, "-y", "--yes"),
) -> None:
    """Remove a section from models.ini, optionally deleting its files."""
    st = state()
    preset = st.preset()
    if not preset.has_section(model):
        fail(f"Unknown model: {model}\n  local-llm models    to see what is available")
    files = [
        Path(value) for key in ("model", "mmproj")
        if (value := preset.get(model, key, fallback_to_star=False))
    ]
    out.print(f"Removing [{model}] from {st.paths.preset}")
    if delete_files:
        for file in files:
            gone = "" if file.exists() or file.is_symlink() else "  (already gone)"
            out.print(f"  delete {file}{gone}")
    if not yes and not typer.confirm("Continue?", default=False):
        raise typer.Exit(1)
    preset.remove_section(model)
    preset.save(st.paths.preset)
    if delete_files:
        for file in files:
            try:
                delete_model_file(file)
            except OSError as error:
                err.print(f"  could not delete {file}: {error}")
    out.print("Removed.")
    _after_preset_change(st, model, added=False)


# ---------------------------------------------------------------- integrations


def _integrate_opencode(st: State, *, agent: bool, yes: bool) -> list[str]:
    lines: list[str] = []
    paths = opencode_paths()
    status = plugin_status(paths)
    if status == "same":
        lines.append(f"plugin already installed: {paths.plugin}")
    else:
        if status == "different" and not yes:
            diff = difflib.unified_diff(
                paths.plugin.read_text().splitlines(), plugin_source().splitlines(),
                fromfile=str(paths.plugin), tofile="shipped plugin", lineterm="",
            )
            out.print("\n".join(diff))
            if not typer.confirm(f"Replace {paths.plugin} with the shipped plugin?", default=True):
                lines.append("plugin left as it is")
            else:
                install_plugin(paths)
                lines.append(f"plugin installed: {paths.plugin}")
        else:
            install_plugin(paths)
            lines.append(f"plugin installed: {paths.plugin}")
    if not agent:
        return lines
    preset = _preset_or_none(st)
    model = smallest_model(preset) if preset else None
    if not model:
        lines.append("no model in models.ini yet, so the tiny helper agent was not added")
        return lines
    model_id = f"llamacpp/{model}"
    target = paths.config_file or paths.new_config
    text = target.read_text() if target.is_file() else ""
    if current_tiny_model(text) == model_id:
        lines.append(f"tiny agent already points at {model_id} in {target}")
        return lines
    merged = merge_agent(text, tiny_agent(model_id))
    if merged is None:
        lines.append(f"{target} has comments, so it is not rewritten. Paste this into it:")
        lines.append(agent_snippet(tiny_agent(model_id)))
        return lines
    if yes or typer.confirm(f"Add the tiny helper agent ({model_id}) to {target}?", default=True):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(merged)
        lines.append(f"tiny agent set to {model_id} in {target}")
    return lines


def _integrate_shell(st: State, shell: str, *, aliases: bool, yes: bool) -> list[str]:
    lines: list[str] = []
    completion_path = install_completion(shell)
    lines.append(f"{shell} completion written to {completion_path}")
    rc = rc_file(shell, Path.home())
    text = rc.read_text() if rc.is_file() else ""
    updated, retired = retire_old_source(text)
    updated = upsert_block(updated, alias_lines(shell)) if aliases else remove_block(updated)
    if updated != text:
        if yes or typer.confirm(f"Update {rc}?", default=True):
            rc.parent.mkdir(parents=True, exist_ok=True)
            rc.write_text(updated)
            if retired:
                lines.append(f"retired {retired} old line(s) that sourced local_llm.zsh in {rc}")
            added_or_removed = "added to" if aliases else "removed from"
            lines.append(f"aliases local_llm, claude_local, copilot_local {added_or_removed} {rc}")
    lines.append(f"open a new shell, or run:  source {rc}")
    return lines


integrate_app = typer.Typer(help="Wire other tools to the router.", rich_markup_mode=None)
app.add_typer(integrate_app, name="integrate")


@integrate_app.command("opencode")
def integrate_opencode_cmd(
    agent: bool = typer.Option(True, "--agent/--no-agent", help="Also add the tiny helper agent."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Do not ask."),
) -> None:
    """Install the opencode plugin that lists every model in models.ini."""
    for line in _integrate_opencode(state(), agent=agent, yes=yes):
        out.print(line)


completion_app = typer.Typer(help="Shell completion and aliases.", rich_markup_mode=None)
app.add_typer(completion_app, name="completion")


@completion_app.command("install")
def completion_install_cmd(
    shell: str | None = typer.Option(
        None, "--shell", help="zsh, bash or fish (default: detected)."
    ),
    aliases: bool = typer.Option(
        True, "--aliases/--no-aliases", help="Add local_llm, claude_local, copilot_local."
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Do not ask."),
) -> None:
    """Install shell completion, and the aliases from the old zsh setup."""
    chosen = shell or detect_shell()
    if chosen not in ("zsh", "bash", "fish"):
        fail(f"--shell must be zsh, bash or fish, got {chosen!r}")
    for line in _integrate_shell(state(), chosen, aliases=aliases, yes=yes):
        out.print(line)


# ---------------------------------------------------------------- setup


@app.command()
def setup(
    yes: bool = typer.Option(False, "-y", "--yes", help="Take every default without asking."),
    use: str | None = typer.Option(
        None, "--use", help="Recommend only coding, general, small or vision."
    ),
    model: list[str] = typer.Option(  # noqa: B008 - typer needs the call as the default
        [], "--model", help="org/repo[:QUANT] to download; repeatable."
    ),
) -> None:
    """Guided first run: prerequisites, this machine, models, settings, integrations, start."""
    st = state()
    if use is not None and use not in GROUPS:
        fail(f"--use must be one of: {', '.join(GROUPS)}")
    hub = _make_hub(st)
    io = Io(
        say=out.print,
        ask=lambda prompt, default: typer.prompt(prompt, default=default),
        confirm=lambda prompt, default: typer.confirm(prompt, default=default),
        run=lambda cmd: subprocess.call(cmd),
        yes=yes,
    )

    def do_pull(machine: Machine, repo: str, option: QuantOption) -> str:
        section = section_name(repo, option.primary)
        preset = _preset_or_none(st)
        if preset is not None and preset.has_section(section):
            out.print(f"  {section} already in models.ini, skipped")
            return section
        _pull(
            st, hub, machine, repo, option,
            name=None, context=None, extra=[], no_tuning=False, yes=True,
        )
        return section

    ctx = SetupContext(
        paths=st.paths,
        settings=st.settings,
        io=io,
        env=Env(),
        hub=hub,
        detect=lambda reserve: _machine(st),
        router=st.router,
        which=shutil.which,
        recommend=lambda machine, preset, use_: gather(
            hub, machine, preset, limit_per_group=3,
            on_progress=(
                lambda repo: err.print(f"  looking at {repo}", end="\r")
                if err.is_terminal else None
            ),
        ),
        search=lambda machine, preset, text: discover_search(
            hub, machine, preset, text=text, limit=10
        ),
        choose_quant=lambda options, suggested, machine: _choose_option(
            options, suggested, machine, yes=yes
        ),
        pull=do_pull,
        integrate_shell=lambda shell: _integrate_shell(st, shell, aliases=True, yes=True),
        integrate_opencode=lambda: _integrate_opencode(st, agent=True, yes=True),
    )
    raise typer.Exit(run_setup(ctx, use=use, models=list(model)))


# ---------------------------------------------------------------- uninstall


def _print_plan(inv, chosen: set[str]) -> None:
    for key in KEYS:
        if key not in chosen:
            continue
        out.print(f"{key}: {inv.summary(key)}")
        if key == "models":
            for entry in inv.models:
                out.print(f"  [{entry.section}]  {human_gb(entry.size)}")
                for file in entry.files:
                    out.print(f"    {file}")
        elif key == "integrations":
            candidates = [inv.plugin, inv.agent_config, *inv.completion_files]
            candidates += [*inv.rc_with_block, *inv.rc_with_bash_source]
            for path in candidates:
                if path:
                    out.print(f"    {path}")
        elif key == "state":
            for path in [inv.state_dir, inv.settings_file]:
                if path:
                    out.print(f"    {path}")
        elif key == "config":
            for path in inv.preset_files:
                out.print(f"    {path}")


def _pick_models(inv) -> list[str]:
    out.print("  models in models.ini:")
    for index, entry in enumerate(inv.models, start=1):
        out.print(f"   {index:>2}. {entry.section}  {human_gb(entry.size)}")
    answer = typer.prompt("  Numbers to remove (e.g. 1 3), a for all", default="a").strip().lower()
    if answer in ("a", "all"):
        return [entry.section for entry in inv.models]
    try:
        return [inv.models[int(token) - 1].section for token in answer.split()]
    except (ValueError, IndexError):
        fail(f"Pick numbers between 1 and {len(inv.models)}")


@app.command()
def uninstall(
    models: bool = typer.Option(
        False, "--models", help="Remove every model section and its files."
    ),
    integrations: bool = typer.Option(
        False, "--integrations", help="opencode plugin and agent, shell aliases and completion."
    ),
    state_: bool = typer.Option(False, "--state", help="pid, logs, Hub cache, settings.toml."),
    config: bool = typer.Option(False, "--config", help="models.ini and its backup."),
    all_: bool = typer.Option(False, "--all", help="Everything above."),
    restore_shell_line: bool = typer.Option(
        False, "--restore-shell-line", help="Put back the retired `source local_llm.zsh` line."
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Do not ask."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only print what would be removed."),
) -> None:
    """Undo what local-llm put on this machine, item by item. Lists everything before deleting."""
    st = state()
    inv = inventory(st.paths)
    chosen: set[str] = set(KEYS) if all_ else set()
    selections = (
        (models, "models"),
        (integrations, "integrations"),
        (state_, "state"),
        (config, "config"),
    )
    for flag, key in selections:
        if flag:
            chosen.add(key)
    sections = [entry.section for entry in inv.models]
    if not chosen:
        if dry_run:
            chosen = set(KEYS)
        elif yes:
            fail(
                "Choose what to remove: --models, --integrations, --state, --config, or --all"
                " (see --dry-run first)."
            )
        else:
            if inv.empty:
                out.print("Nothing of local-llm's is left on this machine.")
                out.print(f"  to remove the tool itself:  {tool_uninstall_hint()}")
                return
            for index, key in enumerate(KEYS, start=1):
                out.print(f"  {index}. {key:<13} {inv.summary(key)}")
            out.print("  5. everything")
            answer = typer.prompt("Numbers to remove (e.g. 1 3), q to quit", default="q")
            answer = answer.strip().lower()
            if answer in ("q", "quit", ""):
                raise typer.Exit()
            if answer in ("5", "a", "all"):
                chosen = set(KEYS)
            else:
                try:
                    chosen = {KEYS[int(token) - 1] for token in answer.split()}
                except (ValueError, IndexError):
                    fail("Pick numbers between 1 and 5")
            if "models" in chosen and inv.models:
                sections = _pick_models(inv)
    if inv.empty:
        out.print("Nothing of local-llm's is left on this machine.")
        out.print(f"  to remove the tool itself:  {tool_uninstall_hint()}")
        return
    _print_plan(inv, chosen)
    if dry_run:
        out.print("Dry run: nothing was removed.")
        return
    if not yes and not typer.confirm("Remove the items above?", default=False):
        raise typer.Exit(1)
    if chosen & {"models", "state", "config"}:
        router = st.router()
        if router.pid() is not None:
            _stop(router)
    if "models" in chosen and sections:
        for line in remove_models(st.paths, sections):
            out.print(f"  {line}")
    if "integrations" in chosen:
        for line in remove_integrations(inv, restore_retired=restore_shell_line):
            out.print(f"  {line}")
    if "state" in chosen:
        for line in remove_state(inv):
            out.print(f"  {line}")
    if "config" in chosen:
        for line in remove_config(inv):
            out.print(f"  {line}")
    out.print()
    out.print(f"To remove the tool itself:  {tool_uninstall_hint()}")
