import pytest

from local_llm.paths import Paths
from local_llm.router import Router, RouterError, is_llama_server
from local_llm.settings import Settings

from .fakes import FakeBackend


def make(tmp_path, backend=None, **settings):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.preset.write_text("[*]\njinja = true\n[m]\nmodel = /x.gguf\n")
    return paths, Router(paths, Settings(**settings), backend=backend or FakeBackend(),
                         binary="/opt/bin/llama-server", sleep=lambda s: None)


def test_is_llama_server_checks_the_program_name():
    assert is_llama_server(["/opt/homebrew/bin/llama-server", "--port", "5678"])
    assert not is_llama_server(["/usr/bin/python3", "llama-server"])
    assert not is_llama_server([])


def test_server_args_reflect_settings(tmp_path):
    paths, router = make(tmp_path, port=7000, max_models=3, ui=True, api_key="k")
    assert router.server_args() == [
        "/opt/bin/llama-server", "--host", "127.0.0.1", "--port", "7000",
        "--models-preset", str(paths.preset), "--models-max", "3", "--models-autoload",
        "--ui", "--api-key", "k",
    ]
    _, plain = make(tmp_path)
    assert "--no-ui" in plain.server_args() and "--api-key" not in plain.server_args()


def test_preconditions(tmp_path):
    paths, router = make(tmp_path)
    router.check_preconditions()  # everything present: no error
    router.binary = None
    with pytest.raises(RouterError, match="llama-server not found"):
        router.check_preconditions()
    router.binary = "/opt/bin/llama-server"
    paths.preset.unlink()
    with pytest.raises(RouterError, match="Missing model config"):
        router.check_preconditions()


def test_refuses_remote_bind_without_key(tmp_path):
    _, router = make(tmp_path, host="0.0.0.0")
    with pytest.raises(RouterError, match="Refusing to bind a non-local address"):
        router.check_preconditions()
    _, allowed = make(tmp_path, host="0.0.0.0", allow_remote=True, api_key="k")
    allowed.check_preconditions()


def test_pid_from_pidfile_when_that_process_is_our_server(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    paths, router = make(tmp_path, backend=backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    assert router.pid() == 42 and router.is_running()


def test_stale_pidfile_is_removed_and_port_scan_finds_the_router(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/usr/bin/sleep", "100"])  # pid reused by something else
    backend.add(77, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    backend.add(78, ["/opt/bin/llama-server", "--alias", "m"], parent=77, listening={40001})
    paths, router = make(tmp_path, backend=backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    assert router.pid() == 77
    assert not paths.pid_file.exists()


def test_not_running_when_nothing_listens(tmp_path):
    backend = FakeBackend()
    backend.add(78, ["/opt/bin/llama-server", "--alias", "m"], listening={40001})
    _, router = make(tmp_path, backend=backend)
    assert router.pid() is None and not router.is_running()


def test_ui_state_round_trip(tmp_path):
    _, router = make(tmp_path)
    assert router.ui_state() is False
    router.write_ui_state(True)
    assert router.ui_state() is True
    router.write_ui_state(False)
    assert router.ui_state() is False
