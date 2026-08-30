"""Test doubles for process handling and the router HTTP API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from local_llm.router import ProcInfo, _alias_of


@dataclass
class FakeProc:
    pid: int
    cmdline: list[str]
    rss: int = 0
    parent: int | None = None
    listening: set[int] = field(default_factory=set)
    alive: bool = True


class FakeBackend:
    """Implements local_llm.router.ProcessBackend against an in-memory process table."""

    def __init__(self) -> None:
        self.procs: dict[int, FakeProc] = {}
        self.next_pid = 1000
        self.spawned: list[tuple[list[str], Path]] = []
        self.spawn_env: list[dict | None] = []
        self.terminated: list[int] = []
        self.killed: list[int] = []
        self.stubborn: set[int] = set()      # pids that ignore terminate()
        self.spawn_listening: set[int] = set()  # ports a spawned process listens on
        self.spawn_dies = False              # spawned process exits immediately
        self.children_survive_parent = False

    def add(self, pid, cmdline, rss=0, parent=None, listening=()) -> FakeProc:
        proc = FakeProc(pid, list(cmdline), rss, parent, set(listening))
        self.procs[pid] = proc
        return proc

    def drop_child(self, alias: str) -> None:
        """Take the child serving `alias` down, which is what an accepted unload does.

        A router that answers an unload and leaves the process running is not
        something any test should be describing. `tune` re-reads the process
        table after unloading, precisely because the reply on its own is not
        evidence that the memory came back, so the fake has to model the part
        that makes that check pass.
        """
        for proc in self.procs.values():
            if _alias_of(proc.cmdline) == alias:
                proc.alive = False

    def restore_child(self, alias: str) -> None:
        """The other half: an accepted load brings the child back."""
        for proc in self.procs.values():
            if _alias_of(proc.cmdline) == alias:
                proc.alive = True

    def _exit(self, pid: int) -> None:
        proc = self.procs.get(pid)
        if proc is None:
            return
        proc.alive = False
        if not self.children_survive_parent:
            for kid in self.procs.values():
                if kid.parent == pid:
                    kid.alive = False

    # -- ProcessBackend
    def find(self, name_fragment: str) -> list[ProcInfo]:
        return [
            ProcInfo(p.pid, p.cmdline, p.rss)
            for p in self.procs.values()
            if p.alive and p.cmdline and name_fragment in os.path.basename(p.cmdline[0])
        ]

    def info(self, pid: int) -> ProcInfo | None:
        proc = self.procs.get(pid)
        return ProcInfo(proc.pid, proc.cmdline, proc.rss) if proc and proc.alive else None

    def children(self, pid: int) -> list[ProcInfo]:
        return [
            ProcInfo(p.pid, p.cmdline, p.rss)
            for p in self.procs.values()
            if p.alive and p.parent == pid
        ]

    def listening(self, pid: int, port: int) -> bool:
        proc = self.procs.get(pid)
        return bool(proc and proc.alive and port in proc.listening)

    def terminate(self, pid: int) -> None:
        self.terminated.append(pid)
        if pid not in self.stubborn:
            self._exit(pid)

    def kill(self, pid: int) -> None:
        self.killed.append(pid)
        self._exit(pid)

    def wait(self, pid: int, timeout: float) -> bool:
        proc = self.procs.get(pid)
        return not (proc and proc.alive)

    def spawn(self, args: list[str], log_path: Path, env=None) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.spawned.append((list(args), log_path))
        # Kept as it came, so a test can tell an empty mapping from no mapping at all.
        self.spawn_env.append(None if env is None else dict(env))
        self.add(pid, args, listening=self.spawn_listening)
        if self.spawn_dies:
            self.procs[pid].alive = False
        return pid


class FakeHttp:
    """Callable matching Router's http hook: (method, path, body) -> dict."""

    def __init__(
        self,
        responses: dict[tuple[str, str], dict] | None = None,
        backend: FakeBackend | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.backend = backend

    def __call__(self, method: str, path: str, body: dict | None) -> dict:
        self.calls.append((method, path, body))
        reply = self.responses.get((method, path), {"success": True})
        # A load or unload the server accepted changes the process table too, and
        # code that checks the table afterwards has to see the change. A reply
        # carrying an "error" key is a refusal, and refusals change nothing.
        if self.backend is not None and isinstance(body, dict) and "error" not in reply:
            model = body.get("model")
            if model and path == "/models/unload":
                self.backend.drop_child(model)
            elif model and path == "/models/load":
                self.backend.restore_child(model)
        return reply


class FakeApi:
    """Stands in for huggingface_hub.HfApi in tests."""

    def __init__(self, models: dict | None = None, extra_files: dict | None = None) -> None:
        self.models = models or {}
        self.extra_files = extra_files or {}
        self.calls: list[tuple] = []

    def list_models(self, **kwargs):
        self.calls.append(("list_models", kwargs))
        infos = list(self.models.values())
        if kwargs.get("filter"):
            infos = [m for m in infos if kwargs["filter"] in (m.tags or [])]
        if kwargs.get("author"):
            infos = [m for m in infos if m.id.split("/")[0] == kwargs["author"]]
        if kwargs.get("search"):
            infos = [m for m in infos if kwargs["search"].lower() in m.id.lower()]
        if kwargs.get("pipeline_tag"):
            infos = [m for m in infos if getattr(m, "pipeline_tag", None) == kwargs["pipeline_tag"]]
        key = "trending_score" if kwargs.get("sort") == "trending_score" else "downloads"
        infos.sort(key=lambda m: -(getattr(m, key, 0) or 0))
        return infos[: kwargs.get("limit") or len(infos)]

    def model_info(self, repo_id, **kwargs):
        import httpx
        from huggingface_hub.errors import RepositoryNotFoundError

        self.calls.append(("model_info", repo_id, kwargs))
        if repo_id not in self.models:
            response = httpx.Response(404, request=httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}"))
            raise RepositoryNotFoundError(f"404 {repo_id}", response=response)
        return self.models[repo_id]

    def file_exists(self, repo_id, filename, **kwargs):
        return filename in self.extra_files.get(repo_id, set())


def fake_model(repo_id, *, downloads=0, likes=0, trending=0, tags=(), base_models=(), gguf=None,
               files=(), pipeline_tag=None, gated=False):
    from types import SimpleNamespace

    siblings = [SimpleNamespace(rfilename=name, size=size) for name, size in files]
    return SimpleNamespace(
        id=repo_id, author=repo_id.split("/")[0], downloads=downloads, likes=likes,
        trending_score=trending, pipeline_tag=pipeline_tag, gated=gated, tags=list(tags),
        base_models=list(base_models), gguf=gguf, siblings=siblings,
    )
