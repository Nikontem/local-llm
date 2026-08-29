"""Is this machine ready? One record per check, each with the fix to run."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import hub, logs
from .paths import Paths
from .preset import Preset, PresetError
from .router import API_KEY_VARIABLE
from .settings import Settings

BREW_INSTALL = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
LINUX_LLAMA_HELP = (
    "Install llama.cpp one of these ways:\n"
    "    brew install llama.cpp                                  (Homebrew on Linux, CPU build)\n"
    "    https://github.com/ggml-org/llama.cpp/releases"
    "          (prebuilt binaries, incl. CUDA/Vulkan)\n"
    "    https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md   (build for your GPU)"
)
_TOOL_TIMEOUT = 10


@dataclass
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    fix: str | None = None  # what a person should run or read
    fix_cmd: list[str] | None = None  # a safe command `doctor --fix` may run after asking


def port_in_use(host: str, port: int) -> bool:
    address = "127.0.0.1" if host == "localhost" else host
    # AF_INET can't bind an IPv6 literal like "::1" (settings.py allows it as a
    # loopback host) - it raises socket.gaierror, an OSError subclass, which would
    # otherwise be mistaken for "port is in use". Resolve the address first and
    # bind with whatever family it actually is.
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((address, port))
        except OSError:
            return True
    return False


@dataclass
class Env:
    """Everything doctor asks the machine, injectable for tests."""

    system: str = field(default_factory=platform.system)
    machine: str = field(default_factory=platform.machine)
    which: Callable[[str], str | None] = shutil.which
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    token_status: Callable[[], hub.TokenStatus] = hub.token_status
    port_in_use: Callable[[str, int], bool] = port_in_use
    home: Path | None = None
    environ: Mapping[str, str] | None = None


def _output(env: Env, binary: str, flag: str) -> str:
    try:
        result = env.run([binary, flag], capture_output=True, text=True, timeout=_TOOL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + (result.stderr or "")


def _runs(help_text: str) -> bool:
    """False when the binary cannot even print its help (missing library, wrong architecture).

    Judged from help text the caller has already asked for, rather than asking again:
    three separate checks used to run --help, and one of them has to decide whether
    the answer means anything.
    """
    broken = ("error while loading", "cannot execute", "Exec format error", "not found")
    return bool(help_text.strip()) and not any(marker in help_text for marker in broken)


def _devices(env: Env, binary: str) -> str:
    lines = [line.strip() for line in _output(env, binary, "--list-devices").splitlines()]
    devices = [line for line in lines if line and not line.startswith("Available devices")]
    return ", ".join(devices)


def run_checks(
    paths: Paths, settings: Settings, env: Env | None = None, router_pid: int | None = None
) -> list[Check]:
    env = env or Env()
    mac = env.system == "Darwin"
    checks: list[Check] = []

    # platform
    if mac or env.system == "Linux":
        checks.append(Check("platform", "ok", f"{env.system} {env.machine}"))
    else:
        checks.append(
            Check("platform", "warn", f"{env.system}: untested; continue at your own risk")
        )

    # brew
    brew = env.which("brew")
    if brew:
        checks.append(Check("brew", "ok", brew))
    elif mac:
        checks.append(Check(
            "brew", "fail", "Homebrew not found; llama.cpp comes from Homebrew on macOS",
            fix=f"Install it: {BREW_INSTALL}",
        ))
    else:
        checks.append(Check("brew", "warn", "Homebrew not found (optional on Linux)"))

    # llama-server
    server = env.which("llama-server")
    help_text = _output(env, server, "--help") if server else ""
    if server and not _runs(help_text):
        first = (_output(env, server, "--version").strip().splitlines() or ["no output"])[0]
        checks.append(Check(
            "llama-server", "fail", f"{server} does not run: {first}",
            fix=(
                "reinstall llama.cpp; the binary is missing a shared library"
                " or was built for another machine"
            ),
        ))
    elif server:
        version = _output(env, server, "--version").strip().splitlines()
        detail = f"{server} - {version[0] if version else 'unknown version'}"
        # Only ask for devices when the flag exists (spec 4.2): an old build that
        # doesn't know --list-devices prints "error: invalid argument" plus a full
        # usage dump instead, which _devices would otherwise fold into this detail.
        if "--list-devices" in help_text:
            devices = _devices(env, server)
            if devices:
                detail += f"; devices: {devices}"
        checks.append(Check("llama-server", "ok", detail))
        if "--models-preset" in help_text:
            checks.append(Check("router support", "ok", "--models-preset available"))
        else:
            checks.append(Check(
                "router support", "fail",
                "this llama-server has no --models-preset; router presets need a build "
                "from December 2025 or later",
                fix="brew upgrade llama.cpp" if brew else LINUX_LLAMA_HELP,
            ))
    elif brew:
        checks.append(Check(
            "llama-server", "fail", "not found", fix="brew install llama.cpp",
            fix_cmd=[brew, "install", "llama.cpp"],
        ))
    else:
        checks.append(Check(
            "llama-server", "fail", "not found",
            fix=f"Install Homebrew first: {BREW_INSTALL}" if mac else LINUX_LLAMA_HELP,
        ))

    # api key. Outside the branches above on purpose: the run where the key is most
    # exposed is one whose --help could not be read at all, which is exactly the branch
    # that would otherwise say nothing. Which of the two ways the key travels is decided
    # by this same help text over in Router, so the person can see which one they get.
    environ = os.environ if env.environ is None else env.environ
    upgrade = "brew upgrade llama.cpp" if brew else LINUX_LLAMA_HELP
    if settings.api_key and API_KEY_VARIABLE in help_text:
        checks.append(Check(
            "api key", "ok",
            f"passed to llama-server in {API_KEY_VARIABLE}, not on its command line",
        ))
    elif settings.api_key and _runs(help_text):
        checks.append(Check(
            "api key", "warn",
            f"this llama-server does not read {API_KEY_VARIABLE}, so the key goes on its"
            " command line, where any program running as you can read it with ps, and the"
            " per-model servers it starts get no key at all",
            fix=upgrade,
        ))
    elif settings.api_key:
        checks.append(Check(
            "api key", "warn",
            f"this llama-server could not be asked whether it reads {API_KEY_VARIABLE}, so"
            " the key will go on its command line, where any program running as you can"
            " read it with ps",
            fix=upgrade,
        ))
    elif environ.get(API_KEY_VARIABLE):
        checks.append(Check(
            "api key", "fail",
            f"{API_KEY_VARIABLE} is set in your environment but LOCAL_LLM_API_KEY is not,"
            " so llama-server will ask every request for a key that local-llm never sends",
            fix=f"export LOCAL_LLM_API_KEY instead, or unset {API_KEY_VARIABLE}",
        ))

    # hf command
    hf_cli = env.which("hf")
    uv = env.which("uv")
    if hf_cli:
        checks.append(Check("hf", "ok", hf_cli))
    elif brew:
        checks.append(Check(
            "hf", "warn", "hf command not found (optional; downloads work without it)",
            fix="brew install hf", fix_cmd=[brew, "install", "hf"],
        ))
    elif uv:
        checks.append(Check(
            "hf", "warn", "hf command not found (optional; downloads work without it)",
            fix='uv tool install "huggingface_hub[cli]"',
            fix_cmd=[uv, "tool", "install", "huggingface_hub[cli]"],
        ))
    else:
        checks.append(Check(
            "hf", "warn", "hf command not found (optional; downloads work without it)",
            fix='pipx install "huggingface_hub[cli]"',
        ))

    # token
    token = env.token_status()
    if token.state == "valid":
        checks.append(Check("hf token", "ok", f"logged in as {token.username}"))
    elif token.state == "invalid":
        checks.append(
            Check("hf token", "warn", "token present but invalid", fix="hf auth login --force")
        )
    elif token.state == "absent":
        checks.append(Check(
            "hf token", "warn", "no token; gated repos unavailable, downloads rate-limited",
            fix="hf auth login"
        ))
    else:
        checks.append(Check(
            "hf token", "warn", "could not reach huggingface.co to validate the token"
        ))

    # config dir
    try:
        paths.config_dir.mkdir(parents=True, exist_ok=True)
        checks.append(Check("config dir", "ok", str(paths.config_dir)))
    except OSError as error:
        checks.append(Check("config dir", "fail", f"cannot create {paths.config_dir}: {error}"))

    # models.ini. The loaded preset is kept for the agents section further down, which
    # needs the same file and used to read and parse it a second time.
    configured_models: Preset | None = None
    if paths.preset.is_file():
        try:
            preset = configured_models = Preset.load(paths.preset)
            problems: list[str] = []
            for name in preset.sections():
                try:
                    preset.file_sizes(name)
                except PresetError as error:
                    text = str(error)
                    if "no model key" in text:
                        problems.append(f"[{name}]: no model key")
                    else:
                        problems.append(f"[{name}]: {text.split(': ')[-1]}")
            if problems:
                checks.append(Check("models.ini", "fail", "missing files - " + "; ".join(problems),
                                    fix="local-llm edit  (fix the paths) or local-llm pull <repo>"))
            else:
                checks.append(Check(
                    "models.ini", "ok", f"{len(preset.sections())} model(s), all files present"
                ))

            # Sections written before the context floor pin `c`, which stops
            # llama-server's fitter adjusting anything at all for that model.
            # Their files are never rewritten without being asked, so the only
            # thing to do is say so.
            star_context = preset.items("*").get("c")
            pinned = [
                name for name in preset.sections()
                if preset.get(name, "c", fallback_to_star=False) is not None
            ]
            if star_context is not None:
                detail = f"[*] pins c = {star_context} for every model"
                if pinned:
                    detail += f"; also set on {', '.join(pinned)}"
            elif pinned:
                detail = f"{len(pinned)} model(s) pin a fixed context: {', '.join(pinned)}"
            else:
                detail = ""
            if detail:
                checks.append(Check(
                    "context sizing", "ok", detail,
                    fix=(
                        "These load at exactly that context and llama.cpp will not adjust"
                        " them if memory is short. Replace `c = N` with `fit-ctx = N` in"
                        " models.ini to let the context be chosen at load time."
                    ),
                ))
        except PresetError as error:
            checks.append(Check("models.ini", "fail", str(error)))
    else:
        checks.append(Check(
            "models.ini", "warn", f"not created yet ({paths.preset})", fix="local-llm setup"
        ))

    # state dirs
    try:
        paths.ensure_state_dirs()
        checks.append(Check("state dir", "ok", str(paths.state_dir)))
    except OSError as error:
        checks.append(Check("state dir", "fail", f"cannot create {paths.state_dir}: {error}"))

    # A model that could not be fitted loaded anyway, at its full unreduced
    # size. llama.cpp only warns; nothing else in this tool would notice.
    log_path = logs.current_log(paths.log_dir)
    if log_path is not None:
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            text = ""
        if "failed to fit params" in text:
            checks.append(Check(
                "model fit", "warn",
                "a model loaded without fitting into free memory; it may swap or fail",
                fix=(
                    "Lower that model's fit-ctx in models.ini, raise reserve_gb in"
                    " settings.toml, or use a smaller quantisation."
                ),
            ))

    # port
    if not env.port_in_use(settings.host, settings.port):
        checks.append(Check("port", "ok", f"{settings.port} is free"))
    elif router_pid is not None:
        checks.append(Check(
            "port", "ok", f"{settings.port} is held by our router (pid {router_pid})"
        ))
    else:
        checks.append(Check(
            "port", "fail", f"{settings.port} is in use by another process",
            fix="pick another port: local-llm --port N <command>, or port = N in settings.toml",
        ))

    # agents
    from . import harnesses
    from .integrations import HarnessContext
    from .integrations.codex import codex_paths, configured_base_url

    installed, missing = harnesses.detect(env.which)
    context = HarnessContext(
        paths=paths,
        settings=settings,
        preset=configured_models,
        home=env.home or Path.home(),
        env=os.environ if env.environ is None else env.environ,
    )

    def titled(harness) -> str:
        """A provider says how it is configured; a launcher's alias is not doctor's business."""
        if harness.kind != harnesses.PROVIDER:
            return harness.title
        state = harnesses.safe_status(harness, lambda h: h.status(context))
        return f"{harness.title} ({harnesses.PROVIDER_WORDS.get(state, 'not configured')})"

    detail = "found: " + (", ".join(titled(h) for h in installed) or "none")
    if missing:
        detail += "; not found: " + ", ".join(h.title for h in missing)
    checks.append(Check("agents", "ok", detail))

    codex_file = codex_paths(home=env.home, env=env.environ)
    written = configured_base_url(codex_file)
    if written is not None and written != settings.openai_base_url:
        checks.append(Check(
            "codex config", "warn",
            f"{codex_file.config_file} points at {written}, not {settings.openai_base_url}",
            fix="local-llm integrate codex",
        ))
    return checks
