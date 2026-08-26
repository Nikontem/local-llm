"""Start, stop and inspect the llama-server router process.

All process access goes through a ProcessBackend so the logic can be tested
against an in-memory fake; PsutilBackend is the real one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .logs import new_run_log, tail_lines
from .paths import Paths
from .settings import Settings

ASLEEP_RSS = 500 * 1024 * 1024  # below this a child has released its weights
START_TIMEOUT = 15.0
STOP_TIMEOUT = 15.0
_POLL = 0.5


class RouterError(Exception):
    pass


@dataclass
class ProcInfo:
    pid: int
    cmdline: list[str]
    rss: int = 0


class ProcessBackend(Protocol):
    def find(self, name_fragment: str) -> list[ProcInfo]: ...
    def info(self, pid: int) -> ProcInfo | None: ...
    def children(self, pid: int) -> list[ProcInfo]: ...
    def listening(self, pid: int, port: int) -> bool: ...
    def terminate(self, pid: int) -> None: ...
    def kill(self, pid: int) -> None: ...
    def wait(self, pid: int, timeout: float) -> bool: ...
    def spawn(self, args: list[str], log_path: Path) -> int: ...


def is_llama_server(cmdline: list[str]) -> bool:
    return bool(cmdline) and "llama-server" in os.path.basename(cmdline[0])


class PsutilBackend:
    def find(self, name_fragment: str) -> list[ProcInfo]:
        import psutil

        found: list[ProcInfo] = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"] or []
                name = proc.info["name"] or ""
                if name_fragment in name or (cmdline and name_fragment in os.path.basename(cmdline[0])):
                    found.append(ProcInfo(proc.pid, list(cmdline)))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return found

    def info(self, pid: int) -> ProcInfo | None:
        import psutil

        try:
            proc = psutil.Process(pid)
            return ProcInfo(pid, list(proc.cmdline()), proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

    def children(self, pid: int) -> list[ProcInfo]:
        import psutil

        try:
            kids = psutil.Process(pid).children()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []
        out: list[ProcInfo] = []
        for kid in kids:
            try:
                out.append(ProcInfo(kid.pid, list(kid.cmdline()), kid.memory_info().rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return out

    def listening(self, pid: int, port: int) -> bool:
        import psutil

        try:
            proc = psutil.Process(pid)
            connections = getattr(proc, "net_connections", proc.connections)
            for conn in connections(kind="tcp"):
                if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return False

    def terminate(self, pid: int) -> None:
        import psutil

        try:
            psutil.Process(pid).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def kill(self, pid: int) -> None:
        import psutil

        try:
            psutil.Process(pid).kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def wait(self, pid: int, timeout: float) -> bool:
        import psutil

        try:
            psutil.Process(pid).wait(timeout)
            return True
        except psutil.NoSuchProcess:
            return True
        except psutil.TimeoutExpired:
            return False

    def spawn(self, args: list[str], log_path: Path) -> int:
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(
                args,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        return proc.pid


@dataclass
class ChildInfo:
    pid: int
    model: str | None
    rss: int

    @property
    def asleep(self) -> bool:
        return self.rss < ASLEEP_RSS


@dataclass
class StartResult:
    pid: int
    log_file: Path | None
    already_running: bool = False


@dataclass
class StopResult:
    was_running: bool
    orphans_cleaned: int = 0
    refused: list[str] = field(default_factory=list)


HttpFn = Callable[[str, str, dict | None], dict]


def _alias_of(cmdline: list[str]) -> str | None:
    for i, arg in enumerate(cmdline):
        if arg in ("--alias", "-a") and i + 1 < len(cmdline):
            return cmdline[i + 1]
    return None


class Router:
    def __init__(
        self,
        paths: Paths,
        settings: Settings,
        *,
        backend: ProcessBackend | None = None,
        http: HttpFn | None = None,
        binary: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.backend: ProcessBackend = backend or PsutilBackend()
        self.http: HttpFn = http or self._urllib_http
        self.binary = binary if binary is not None else shutil.which("llama-server")
        self._sleep = sleep
        self._log = log

    # ------------------------------------------------------------ arguments

    def server_args(self) -> list[str]:
        s = self.settings
        args = [
            self.binary or "llama-server",
            "--host", s.host,
            "--port", str(s.port),
            "--models-preset", str(self.paths.preset),
            "--models-max", str(s.max_models),
            "--models-autoload",
            "--ui" if s.ui else "--no-ui",
        ]
        if s.api_key:
            args += ["--api-key", s.api_key]
        return args

    def check_preconditions(self) -> None:
        if not self.binary:
            raise RouterError("llama-server not found in PATH.\nInstall it with: brew install llama.cpp")
        if not self.paths.preset.is_file():
            raise RouterError(
                f"Missing model config: {self.paths.preset}\n"
                "Every model lives in that file. Create it with: local-llm setup"
            )
        s = self.settings
        if not s.is_local and not (s.allow_remote and s.api_key):
            raise RouterError(
                f"Refusing to bind a non-local address: {s.host}\n"
                "This would expose the models to your network.\n"
                "If that is intentional, set both LOCAL_LLM_ALLOW_REMOTE=1 and LOCAL_LLM_API_KEY."
            )

    # ------------------------------------------------------------ discovery

    def _is_our_router(self, pid: int) -> bool:
        info = self.backend.info(pid)
        return info is not None and is_llama_server(info.cmdline) and self.backend.listening(pid, self.settings.port)

    def pid(self) -> int | None:
        pid_file = self.paths.pid_file
        if pid_file.is_file():
            try:
                recorded = int(pid_file.read_text().strip())
            except ValueError:
                recorded = 0
            if recorded and self._is_our_router(recorded):
                return recorded
            pid_file.unlink(missing_ok=True)  # stale
        for proc in self.backend.find("llama-server"):
            if self.backend.listening(proc.pid, self.settings.port):
                return proc.pid
        return None

    def is_running(self) -> bool:
        return self.pid() is not None

    # ------------------------------------------------------------ ui state

    def ui_state(self) -> bool:
        f = self.paths.ui_file
        return f.is_file() and f.read_text().strip() == "1"

    def write_ui_state(self, on: bool) -> None:
        self.paths.ensure_state_dirs()
        self.paths.ui_file.write_text("1\n" if on else "0\n")

    # ------------------------------------------------------------ http (Task 10)

    def _urllib_http(self, method: str, path: str, body: dict | None) -> dict:
        raise RouterError("HTTP client not implemented yet")
