from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from local_llm import cli
from local_llm.paths import Paths
from local_llm.router import Router

from .fakes import FakeBackend, FakeHttp

ENV_VARS = (
    "LOCAL_LLM_CONFIG_DIR", "LOCAL_LLM_PRESET", "LOCAL_LLM_STATE_DIR", "LOCAL_LLM_LOG_DIR",
    "LOCAL_LLM_PORT", "LOCAL_LLM_HOST", "LOCAL_LLM_MAX_MODELS", "LOCAL_LLM_RESERVE_GB",
    "LOCAL_LLM_UI", "LOCAL_LLM_DEFAULT_MODEL", "LOCAL_LLM_ALLOW_REMOTE", "LOCAL_LLM_API_KEY",
    "XDG_CONFIG_HOME", "XDG_STATE_HOME", "EDITOR", "VISUAL",
)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A CLI wired to a fake process table and a fake router API, in a scratch HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    paths = Paths.from_env(env={"HOME": str(tmp_path)}, home=tmp_path)
    paths.config_dir.mkdir(parents=True)
    (tmp_path / "big.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "small.gguf").write_bytes(b"y" * 512)
    paths.preset.write_text(
        "[*]\nc = 8192\n"
        f"[big]\nmodel = {tmp_path}/big.gguf\nc = 65536\nn-predict = 32768\n"
        f"[small]\nmodel = {tmp_path}/small.gguf\n"
    )
    backend = FakeBackend()
    http = FakeHttp({("GET", "/health"): {"status": "ok"}})
    messages: list[str] = []

    def make_router(p, s, **kwargs):
        return Router(p, s, backend=backend, http=http, binary="/opt/bin/llama-server",
                      sleep=lambda seconds: None, log=messages.append)

    monkeypatch.setattr(cli, "Router", make_router)
    monkeypatch.setattr(cli, "_sleep", lambda seconds: None)
    opened: list[str] = []
    monkeypatch.setattr(cli, "_open_url", lambda url: opened.append(url) or True)
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    cli._state = None
    runner = CliRunner()

    def run(*args):
        return runner.invoke(cli.app, list(args))

    return SimpleNamespace(paths=paths, backend=backend, http=http, run=run, tmp=tmp_path,
                           messages=messages, opened=opened, monkeypatch=monkeypatch)
