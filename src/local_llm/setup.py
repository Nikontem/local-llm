"""The guided first run: prerequisites, this machine, models, settings, integrations, start.

Every question goes through an Io object so the whole flow is testable, and
`--yes` takes each step's default without asking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .discover import GROUPS, Candidate
from .doctor import Check, Env, run_checks
from .estimate import human_gb
from .hardware import Machine
from .hub import Hub, HubError
from .paths import Paths
from .preset import Preset, PresetError, smallest_model
from .quant import QuantError, QuantOption, find_option, quant_options, suggest
from .router import Router, RouterError
from .settings import Settings, save_settings


class SetupAbort(Exception):
    pass


@dataclass
class Io:
    say: Callable[[str], None]
    ask: Callable[[str, str], str]
    confirm: Callable[[str, bool], bool]
    run: Callable[[list[str]], int]
    yes: bool = False

    def ask_or_default(self, prompt: str, default: str) -> str:
        return default if self.yes else self.ask(prompt, default)

    def confirm_or_default(self, prompt: str, default: bool) -> bool:
        return default if self.yes else self.confirm(prompt, default)


@dataclass
class SetupContext:
    paths: Paths
    settings: Settings
    io: Io
    env: Env
    hub: Hub
    detect: Callable[[int], Machine]
    router: Callable[[], Router]
    which: Callable[[str], str | None]
    recommend: Callable[[Machine, Preset | None, str | None], dict[str, list[Candidate]]]
    search: Callable[[Machine, Preset | None, str], list[Candidate]]
    choose_quant: Callable[[list[QuantOption], QuantOption | None, Machine], QuantOption]
    pull: Callable[[Machine, str, QuantOption], str]
    integrate_shell: Callable[[str], list[str]]
    integrate_opencode: Callable[[], list[str]]
    shell: str | None = None  # None: detect from the environment


def load_preset(paths: Paths) -> Preset | None:
    try:
        return Preset.load(paths.preset)
    except PresetError:
        return None


def _header(io: Io, text: str) -> None:
    io.say("")
    io.say(text)


def _check_line(check: Check) -> str:
    mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}[check.status]
    return f"  {mark}  {check.name:<16} {check.detail}"


# ---------------------------------------------------------------- 1. prerequisites


def step_prerequisites(ctx: SetupContext) -> dict[str, Check]:
    io = ctx.io
    _header(io, "1. Prerequisites")
    checks = {c.name: c for c in run_checks(ctx.paths, ctx.settings, env=ctx.env)}
    for name in ("platform", "brew", "llama-server", "router support", "hf", "hf token"):
        if name in checks:
            io.say(_check_line(checks[name]))

    brew = checks["brew"]
    if brew.status == "fail":
        raise SetupAbort(
            "Homebrew is needed on macOS: llama.cpp is installed from it.\n"
            f"  {brew.fix}\n"
            "Then run local-llm setup again (or re-run install.sh)."
        )

    server = checks["llama-server"]
    if server.status == "fail":
        if server.fix_cmd and io.confirm_or_default(
            f"llama-server is missing. Install llama.cpp now with: "
            f"{' '.join(server.fix_cmd)} ?",
            True,
        ):
            if io.run(server.fix_cmd) != 0:
                raise SetupAbort(
                    f"{' '.join(server.fix_cmd)} failed. Fix that, then run local-llm setup again."
                )
            checks = {c.name: c for c in run_checks(ctx.paths, ctx.settings, env=ctx.env)}
            server = checks["llama-server"]
            io.say(_check_line(server))
            if server.status == "fail":
                raise SetupAbort(
                    "llama-server is still not on PATH after the install. "
                    "Open a new shell and run local-llm setup again."
                )
        else:
            raise SetupAbort(server.fix or "Install llama.cpp, then run local-llm setup again.")
    if checks.get("router support") and checks["router support"].status == "fail":
        raise SetupAbort(
            checks["router support"].detail + "\n  " + (checks["router support"].fix or "")
        )

    hf = checks["hf"]
    if hf.status == "warn" and hf.fix_cmd:
        io.say(
            "  The hf command is optional: this tool downloads with the same library. "
            "It is handy for `hf auth login`."
        )
        if io.confirm_or_default(f"Install it now with: {' '.join(hf.fix_cmd)} ?", True):
            io.run(hf.fix_cmd)

    token = checks["hf token"]
    if token.status == "warn" and token.fix:
        io.say(
            "  Some repositories on Hugging Face are gated: the author asks you to accept terms,"
        )
        io.say("  which needs a free account and a token. Everything else works without one.")
        if io.confirm_or_default(f"Log in now with: {token.fix} ?", False):
            io.run(token.fix.split())
    return checks


# ---------------------------------------------------------------- 2. machine


def step_machine(ctx: SetupContext) -> Machine:
    io = ctx.io
    _header(io, "2. This machine")
    machine = ctx.detect(ctx.settings.reserve_gb)
    io.say(f"  {machine.describe()}")
    answer = io.ask_or_default(
        "  Memory to keep free for the OS and your apps, in GB", str(ctx.settings.reserve_gb)
    )
    try:
        reserve = int(answer)
    except ValueError:
        reserve = ctx.settings.reserve_gb
    if reserve != ctx.settings.reserve_gb and reserve >= 0:
        ctx.settings.reserve_gb = reserve
        machine = ctx.detect(reserve)
        io.say(f"  {machine.describe()}")
    return machine


# ---------------------------------------------------------------- 3. models


def _resolve_preselected(ctx: SetupContext, machine: Machine, spec: str) -> tuple[str, QuantOption]:
    repo, _, tag = spec.partition(":")
    if repo.count("/") != 1:
        raise SetupAbort(f"--model expects org/repo or org/repo:QUANT, got {spec!r}")
    try:
        files = ctx.hub.repo_files(repo)
    except HubError as error:
        raise SetupAbort(str(error)) from None
    options = quant_options(files.files)
    if not options:
        raise SetupAbort(f"{repo} has no GGUF model files.")
    try:
        if tag:
            return repo, find_option(options, quant=tag)
    except QuantError as error:
        raise SetupAbort(str(error)) from None
    suggested, _ = suggest(options, machine.budget)
    return repo, ctx.choose_quant(options, suggested, machine)


def _candidate_line(candidate: Candidate, machine: Machine) -> str:
    option = candidate.suggested
    marks = ", ".join(
        m for m, on in (("thinking", candidate.thinking), ("vision", candidate.vision)) if on
    )
    marks = f" ({marks})" if marks else ""
    state = (
        "  · in models.ini"
        if candidate.configured
        else ("  · downloaded" if candidate.downloaded else "")
    )
    if option is None:
        return f"{candidate.display_name}{marks}  {candidate.repo_id}"
    return (
        f"{candidate.display_name}{marks}  {candidate.repo_id}  {option.label}"
        f"  {human_gb(option.total)} on disk, ~{human_gb(candidate.estimate)} in memory"
        f"  {machine.fit_label(candidate.estimate)}{state}"
    )


def step_models(
    ctx: SetupContext, machine: Machine, preset: Preset | None, *, use: str | None,
    preselected: list[str],
) -> list[tuple[str, QuantOption]]:
    io = ctx.io
    _header(io, "3. Models")
    if preselected:
        return [_resolve_preselected(ctx, machine, spec) for spec in preselected]
    if io.yes:
        io.say("  No --model given; nothing is downloaded without asking.")
        io.say("  Later: local-llm recommend    or    local-llm pull <org/repo>")
        return []

    io.say("  Looking at what is popular on Hugging Face and what fits here...")
    try:
        groups = ctx.recommend(machine, preset, use)
    except HubError as error:
        io.say(f"  Could not reach Hugging Face: {error}")
        return []
    wanted = [use] if use else list(GROUPS)
    numbered: list[Candidate] = []
    while True:
        for group in wanted:
            io.say(f"  {group}")
            for candidate in groups.get(group, []):
                if candidate in numbered:
                    continue
                numbered.append(candidate)
                io.say(f"   {len(numbered):>2}. {_candidate_line(candidate, machine)}")
        answer = io.ask(
            "  Numbers to download (e.g. 1 3), s <text> to search, n for none", "n"
        ).strip()
        if answer.lower() in ("n", "none", ""):
            return []
        if answer.lower().startswith("s "):
            try:
                results = ctx.search(machine, preset, answer[2:].strip())
            except HubError as error:
                io.say(f"  Search failed: {error}")
                continue
            groups = {"search": results}
            wanted = ["search"]
            continue
        chosen: list[tuple[str, QuantOption]] = []
        try:
            picks = [numbered[int(token) - 1] for token in answer.split()]
        except (ValueError, IndexError):
            io.say(f"  Pick numbers between 1 and {len(numbered)}.")
            continue
        for candidate in picks:
            if not candidate.options:
                try:
                    files = ctx.hub.repo_files(candidate.repo_id)
                    candidate.options = quant_options(files.files)
                    candidate.suggested, _ = suggest(candidate.options, machine.budget)
                except HubError as error:
                    io.say(f"  {candidate.repo_id}: {error}")
                    continue
            option = ctx.choose_quant(candidate.options, candidate.suggested, machine)
            chosen.append((candidate.repo_id, option))
        return chosen


# ---------------------------------------------------------------- 4. download


def step_download(
    ctx: SetupContext, machine: Machine, chosen: list[tuple[str, QuantOption]]
) -> list[str]:
    added: list[str] = []
    if chosen:
        _header(ctx.io, "4. Download and configure")
    for repo, option in chosen:
        try:
            added.append(ctx.pull(machine, repo, option))
        except HubError as error:
            ctx.io.say(f"  {repo}: {error}")
    return added


# ---------------------------------------------------------------- 5. settings


def step_settings(ctx: SetupContext, default_model: str) -> Path:
    _header(ctx.io, "5. Settings")
    ctx.settings.default_model = default_model
    written = save_settings(ctx.paths, ctx.settings)
    ctx.io.say(f"  wrote {written}")
    ctx.io.say(
        f"  port {ctx.settings.port}, {ctx.settings.reserve_gb} GB reserved, "
        f"{ctx.settings.max_models} model(s) resident at once, "
        f"default model: {default_model or 'none'}"
    )
    return written


# ---------------------------------------------------------------- 6. integrations


def step_integrations(ctx: SetupContext) -> None:
    io = ctx.io
    _header(io, "6. Shell and coding agents")
    from .shellrc import detect_shell

    shell = ctx.shell or detect_shell()
    if io.confirm_or_default(
        f"Install {shell} completion and the aliases local_llm, claude_local, copilot_local?", True
    ):
        for line in ctx.integrate_shell(shell):
            io.say(f"  {line}")
    if ctx.which("opencode") and io.confirm_or_default(
        "opencode is installed. Add the local-llm plugin so every model shows up there?", True
    ):
        for line in ctx.integrate_opencode():
            io.say(f"  {line}")
    for name in ("claude", "copilot"):
        if ctx.which(name):
            io.say(f"  {name} found: run it against the router with  local-llm {name} [model]")


# ---------------------------------------------------------------- 7. start


def step_start(ctx: SetupContext, preset: Preset) -> None:
    io = ctx.io
    _header(io, "7. Start")
    router = ctx.router()
    try:
        result = router.start()
    except RouterError as error:
        io.say(f"  {error}")
        return
    io.say("  Router is already running." if result.already_running else "  Router is up.")
    model = smallest_model(preset)
    if model and io.confirm_or_default(
        f"Send a test request to {model}? It loads the model first.", True
    ):
        try:
            reply = router.chat(model, "Say hello in three words.", max_tokens=12)
            content = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content")
            io.say(f"  {model} says: {content!s}" if content else f"  unexpected reply: {reply}")
        except RouterError as error:
            io.say(f"  {error}")
    io.say("")
    io.say(f"  OpenAI-style endpoint:     {ctx.settings.openai_base_url}")
    io.say(f"  Anthropic-style endpoint:  {ctx.settings.anthropic_base_url}")
    io.say("  local-llm status      what is running")
    io.say("  local-llm models      every model you can ask for")
    io.say("  local-llm claude      Claude Code against the router")


# ---------------------------------------------------------------- all together


def run_setup(ctx: SetupContext, *, use: str | None = None, models: list[str] = ()) -> int:
    io = ctx.io
    try:
        step_prerequisites(ctx)
        machine = step_machine(ctx)
        preset = load_preset(ctx.paths)
        chosen = step_models(ctx, machine, preset, use=use, preselected=list(models))
        added = step_download(ctx, machine, chosen)
        preset = load_preset(ctx.paths)
        default_model = added[0] if added else ctx.settings.default_model
        if preset is not None and not default_model:
            default_model = smallest_model(preset) or ""
        step_settings(ctx, default_model)
        step_integrations(ctx)
        if preset is None or not preset.sections():
            io.say("")
            io.say("No models configured yet. When you are ready:")
            io.say("  local-llm recommend          what fits this machine")
            io.say("  local-llm pull <org/repo>    download one and add it")
            io.say("  local-llm up                 then start serving")
            return 0
        step_start(ctx, preset)
        return 0
    except SetupAbort as error:
        io.say("")
        io.say(str(error))
        return 1
