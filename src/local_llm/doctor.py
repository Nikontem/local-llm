"""Is this machine ready? One record per check, each with the fix to run."""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import hub
from .paths import Paths
from .preset import Preset, PresetError
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


def _runs(env: Env, binary: str) -> bool:
    """False when the binary cannot even print its help (missing library, wrong architecture)."""
    try:
        result = env.run([binary, "--help"], capture_output=True, text=True, timeout=_TOOL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return False
    text = (result.stdout or "") + (result.stderr or "")
    broken = ("error while loading", "cannot execute", "Exec format error", "not found")
    return bool(text.strip()) and not any(marker in text for marker in broken)


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
    if server and not _runs(env, server):
        first = (_output(env, server, "--version").strip().splitlines() or ["no output"])[0]
        checks.append(Check(
            "llama-server", "fail", f"{server} does not run: {first}",
            fix=(
                "reinstall llama.cpp; the binary is missing a shared library"
                " or was built for another machine"
            ),
        ))
    elif server:
        help_text = _output(env, server, "--help")
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

    # models.ini
    if paths.preset.is_file():
        try:
            preset = Preset.load(paths.preset)
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
    from .integrations.codex import codex_paths, configured_base_url

    installed, missing = harnesses.detect(env.which)
    detail = "found: " + (", ".join(h.title for h in installed) or "none")
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
