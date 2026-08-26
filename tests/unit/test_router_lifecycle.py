import pytest

from local_llm.paths import Paths
from local_llm.router import Router, RouterError
from local_llm.settings import Settings

from .fakes import FakeBackend


def make(tmp_path, backend, **settings):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.preset.write_text("[*]\njinja = true\n[m]\nmodel = /x.gguf\n")
    messages: list[str] = []
    router = Router(paths, Settings(**settings), backend=backend, binary="/opt/bin/llama-server",
                    sleep=lambda s: None, log=messages.append)
    return paths, router, messages


def test_start_spawns_writes_pidfile_and_log(tmp_path):
    backend = FakeBackend()
    backend.spawn_listening = {5678}
    paths, router, _ = make(tmp_path, backend)
    result = router.start()
    assert result.already_running is False
    assert result.pid == 1000
    args, log_path = backend.spawned[0]
    assert args == router.server_args()
    assert log_path.parent == paths.log_dir and log_path.is_file()
    assert result.log_file == log_path
    assert paths.pid_file.read_text().strip() == "1000"
    assert oct(paths.pid_file.stat().st_mode & 0o777) == "0o600"
    assert paths.ui_file.read_text().strip() == "0"
    assert router.pid() == 1000


def test_start_when_already_running_does_not_spawn(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    _, router, _ = make(tmp_path, backend)
    result = router.start()
    assert result.already_running and result.pid == 42
    assert backend.spawned == []


def test_start_failure_reports_log_tail_and_clears_pidfile(tmp_path):
    backend = FakeBackend()
    backend.spawn_dies = True
    paths, router, _ = make(tmp_path, backend)
    original_spawn = backend.spawn

    def spawn_and_log(args, log_path):
        pid = original_spawn(args, log_path)
        log_path.write_text("error: failed to load preset\n")
        return pid

    backend.spawn = spawn_and_log
    with pytest.raises(RouterError, match="failed to load preset"):
        router.start()
    assert not paths.pid_file.exists()


def test_start_timeout_when_it_never_listens(tmp_path):
    backend = FakeBackend()  # spawned process stays alive but never listens
    _, router, _ = make(tmp_path, backend)
    with pytest.raises(RouterError, match="Router failed to start"):
        router.start()


def test_stop_terminates_router_and_reports_children(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.add(43, ["/opt/bin/llama-server", "--alias", "m"], parent=42)
    paths, router, messages = make(tmp_path, backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    result = router.stop()
    assert result.was_running and result.orphans_cleaned == 0 and result.refused == []
    assert backend.terminated == [42] and backend.killed == []
    assert not paths.pid_file.exists()
    assert messages == ["Stopping router pid 42"]


def test_stop_sweeps_orphaned_children(tmp_path):
    backend = FakeBackend()
    backend.children_survive_parent = True
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.add(43, ["/opt/bin/llama-server", "--alias", "m"], parent=42)
    _, router, messages = make(tmp_path, backend)
    result = router.stop()
    assert result.orphans_cleaned == 1
    assert backend.terminated == [42, 43]
    assert messages == ["Stopping router pid 42", "Stopping model server pid 43"]


def test_stop_forces_a_stubborn_process(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.stubborn.add(42)
    _, router, messages = make(tmp_path, backend)
    router.stop()
    assert backend.terminated == [42] and backend.killed == [42]
    assert messages == ["Stopping router pid 42", "  did not exit in 15s, forcing"]


def test_stop_refuses_pids_that_are_not_llama_server(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/usr/bin/python3", "server.py"])
    paths, router, _ = make(tmp_path, backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    result = router.stop()
    assert result.was_running
    assert result.refused == ["pid 42: /usr/bin/python3 server.py"]
    assert backend.terminated == []


def test_stop_when_not_running(tmp_path):
    paths, router, _ = make(tmp_path, FakeBackend())
    paths.ensure_state_dirs()
    paths.pid_file.write_text("999\n")
    assert router.stop().was_running is False
    assert not paths.pid_file.exists()


def test_children_and_loaded_models(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.add(43, ["/opt/bin/llama-server", "--alias", "big"], rss=20 * 1024**3, parent=42)
    backend.add(44, ["/opt/bin/llama-server", "-a", "small"], rss=100 * 1024**2, parent=42)
    _, router, _ = make(tmp_path, backend)
    kids = router.children()
    assert [(k.pid, k.model, k.asleep) for k in kids] == [(43, "big", False), (44, "small", True)]
    assert router.loaded_model_names() == ["big", "small"]
    assert make(tmp_path, FakeBackend())[1].children() == []
