import pytest

from local_llm.paths import Paths
from local_llm.router import API_KEY_VARIABLE, Router, RouterError
from local_llm.settings import Settings

from .fakes import FakeBackend

#: What a current llama-server prints for --api-key. A build that has not learned to
#: read the key from the environment prints the same line without the (env: ...) part.
MODERN_HELP = f"  --api-key KEY   API key to use for authentication\n(env: {API_KEY_VARIABLE})\n"
ANCIENT_HELP = "  --api-key KEY   API key to use for authentication\n"


def make(tmp_path, backend, help_text=MODERN_HELP, **settings):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.preset.write_text("[*]\njinja = true\n[m]\nmodel = /x.gguf\n")
    messages: list[str] = []
    router = Router(paths, Settings(**settings), backend=backend, binary="/opt/bin/llama-server",
                    sleep=lambda s: None, log=messages.append,
                    help_text=lambda binary: help_text)
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

    def spawn_and_log(args, log_path, env=None):
        pid = original_spawn(args, log_path, env)
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


def test_the_api_key_never_reaches_the_command_line(tmp_path):
    """`ps` shows every argument of every process to everybody running as this user."""
    backend = FakeBackend()
    backend.spawn_listening = {5678}
    _, router, _ = make(tmp_path, backend, api_key="sk-secret")

    router.start()

    args, _ = backend.spawned[0]
    assert "--api-key" not in args and "sk-secret" not in args
    assert backend.spawn_env[0] == {API_KEY_VARIABLE: "sk-secret"}


def test_a_build_that_cannot_read_the_variable_still_gets_the_key(tmp_path):
    """Putting it only in the environment there would start a server with no key at all."""
    backend = FakeBackend()
    backend.spawn_listening = {5678}
    _, router, _ = make(tmp_path, backend, help_text=ANCIENT_HELP, api_key="sk-secret")

    router.start()

    args, _ = backend.spawned[0]
    assert args[args.index("--api-key") + 1] == "sk-secret"
    assert backend.spawn_env[0] == {}


def test_no_key_configured_adds_nothing_either_way(tmp_path):
    backend = FakeBackend()
    backend.spawn_listening = {5678}
    _, router, _ = make(tmp_path, backend)

    router.start()

    assert "--api-key" not in backend.spawned[0][0] and backend.spawn_env[0] == {}
    assert backend.spawn_env[0] is not None, "an empty mapping, not no mapping at all"


def test_the_binary_is_asked_about_the_variable_once(tmp_path):
    """It cannot change while this process runs, and asking costs a subprocess each time."""
    backend = FakeBackend()
    asked: list[str | None] = []
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.preset.write_text("[*]\njinja = true\n[m]\nmodel = /x.gguf\n")

    def record(binary):
        asked.append(binary)
        return MODERN_HELP

    router = Router(paths, Settings(api_key="sk-secret"), backend=backend,
                    binary="/opt/bin/llama-server", sleep=lambda s: None, help_text=record)
    router.server_args()
    router.server_env()
    router.server_args()

    assert asked == ["/opt/bin/llama-server"]


def test_context_and_thinking_overrides_become_router_flags(tmp_path):
    """A flag on the router process beats the same key in every models.ini
    section - verified against llama-server 0.4.0, whose merged preset shows the
    section's own `c` replaced - so these need no rewrite of the preset."""
    _, router, _ = make(tmp_path, FakeBackend())
    args = router.server_args()
    assert "--ctx-size" not in args and "--chat-template-kwargs" not in args
    _, router, _ = make(tmp_path, FakeBackend(), context=262144, no_thinking=True)
    args = router.server_args()
    assert args[args.index("--ctx-size") + 1] == "262144"
    assert args[args.index("--chat-template-kwargs") + 1] == '{"enable_thinking": false}'


def test_overrides_are_read_back_from_the_running_router(tmp_path):
    """`restart` and `status` ask the live process, not a state file, so what
    they report is what the models are actually being served with."""
    backend = FakeBackend()
    backend.spawn_listening = {5678}
    _, router, _ = make(tmp_path, backend, context=262144, no_thinking=True)
    assert router.overrides() == (0, False)          # not running yet
    router.start()
    assert router.overrides() == (262144, True)
    backend = FakeBackend()
    backend.spawn_listening = {5678}
    _, router, _ = make(tmp_path, backend)
    router.start()
    assert router.overrides() == (0, False)
