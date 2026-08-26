from pathlib import Path

from local_llm import cli


def running_router(h, ui=False, children=()):
    h.backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    for pid, alias, rss in children:
        h.backend.add(pid, ["/opt/bin/llama-server", "--alias", alias], rss=rss, parent=42)
    h.paths.ensure_state_dirs()
    h.paths.pid_file.write_text("42\n")
    h.paths.ui_file.write_text("1\n" if ui else "0\n")


def test_no_arguments_shows_stopped_status(harness):
    h = harness
    result = h.run()
    assert result.exit_code == 0
    assert "state:    stopped" in result.output
    assert str(h.paths.preset) in result.output
    assert "local-llm up" in result.output


def test_version(harness):
    result = harness.run("--version")
    assert result.exit_code == 0 and result.output.startswith("local-llm ")


def test_up_starts_and_reports(harness):
    h = harness
    h.backend.spawn_listening = {5678}
    result = h.run("up")
    assert result.exit_code == 0, result.output
    assert "Router is up." in result.output
    assert "url:      http://127.0.0.1:5678/v1" in result.output
    assert "web ui" not in result.output
    args, _ = h.backend.spawned[0]
    assert args[:5] == ["/opt/bin/llama-server", "--host", "127.0.0.1", "--port", "5678"]
    assert "--no-ui" in args
    assert h.paths.pid_file.read_text().strip() == "1000"
    again = h.run("up")
    assert "Router is already running (pid 1000)." in again.output
    assert len(h.backend.spawned) == 1


def test_up_with_ui_and_port_override(harness):
    h = harness
    h.backend.spawn_listening = {7001}
    result = h.run("--port", "7001", "up", "--ui")
    assert result.exit_code == 0, result.output
    args, _ = h.backend.spawned[0]
    assert "--ui" in args and "7001" in args
    assert "web ui:   http://127.0.0.1:7001/" in result.output
    assert h.paths.ui_file.read_text().strip() == "1"


def test_up_max_models_flag_and_default(harness):
    h = harness
    h.backend.spawn_listening = {5678}
    h.run("up")
    args, _ = h.backend.spawned[0]
    assert args[args.index("--models-max") + 1] == "1"
    h.run("down")
    h.run("up", "--max-models", "2")
    args, _ = h.backend.spawned[1]
    assert args[args.index("--models-max") + 1] == "2"


def test_up_failure_shows_log_tail(harness):
    h = harness
    h.backend.spawn_dies = True
    result = h.run("up")
    assert result.exit_code == 1
    assert "Router failed to start" in result.output


def test_down(harness):
    h = harness
    assert "Router is not running." in h.run("down").output
    running_router(h, children=[(43, "big", 0)])
    result = h.run("down")
    assert result.exit_code == 0
    assert "Router is down." in result.output
    assert h.messages == ["Stopping router pid 42"]
    assert not h.paths.pid_file.exists()


def test_stop_alias(harness):
    assert "Router is not running." in harness.run("stop").output


def test_status_when_running(harness):
    h = harness
    running_router(h, children=[(43, "big", 20 * 1024**3), (44, "small", 100 * 1024**2)])
    result = h.run("status")
    assert result.exit_code == 0
    assert "state:    running" in result.output and "pid:      42" in result.output
    assert 'health:   {"status":"ok"}' in result.output
    assert "web ui:   off  (local-llm restart --ui)" in result.output
    assert "big" in result.output and "20.0 GB" in result.output
    assert "small" in result.output and "(asleep)" in result.output


def test_restart_restores_loaded_models(harness):
    h = harness
    running_router(h, ui=True, children=[(43, "big", 0)])
    h.backend.spawn_listening = {5678}
    result = h.run("restart")
    assert result.exit_code == 0, result.output
    assert "Router is down." in result.output and "Router is up." in result.output
    assert "Restoring 1 model(s) that were loaded:" in result.output
    assert "  big ... ok" in result.output
    assert ("POST", "/models/load", {"model": "big"}) in h.http.calls
    assert "--ui" in h.backend.spawned[0][0]  # UI mode remembered


def test_restart_no_restore_and_no_ui(harness):
    h = harness
    running_router(h, ui=True, children=[(43, "big", 0)])
    h.backend.spawn_listening = {5678}
    result = h.run("restart", "--no-ui", "--no-restore")
    assert "Restoring" not in result.output
    assert "--no-ui" in h.backend.spawned[0][0]
    assert ("POST", "/models/load", {"model": "big"}) not in h.http.calls


def test_models_lists_names_and_sizes(harness):
    h = harness
    result = h.run("models")
    assert result.exit_code == 0
    assert f"Models in {h.paths.preset}" in result.output
    assert "big" in result.output and "small" in result.output and "0.0 GB" in result.output
    assert h.run("ls").exit_code == 0
    Path(h.tmp / "small.gguf").unlink()
    assert "missing" in h.run("models").output


def test_models_without_preset(harness):
    h = harness
    h.paths.preset.unlink()
    result = h.run("models")
    assert result.exit_code == 1 and "Missing model config" in result.output


def test_logs(harness):
    h = harness
    result = h.run("logs")
    assert result.exit_code == 1 and "No log yet" in result.output
    h.backend.spawn_listening = {5678}
    h.run("up")
    _, log_path = h.backend.spawned[0]
    log_path.write_text("line one\nline two\n")
    result = h.run("logs", "-n", "1")
    assert result.exit_code == 0
    assert "line two" in result.output and "line one" not in result.output
    assert "local-llm logs -f" in result.output


def test_prune_logs(harness):
    result = harness.run("prune-logs", "30")
    assert result.exit_code == 0
    assert "Removed 0 log file(s) older than 30 days." in result.output


def test_ui_starts_router_when_stopped(harness):
    h = harness
    h.backend.spawn_listening = {5678}
    result = h.run("ui")
    assert result.exit_code == 0, result.output
    assert "Starting it with the web UI" in result.output
    assert "--ui" in h.backend.spawned[0][0]
    assert h.opened == ["http://127.0.0.1:5678/"]


def test_ui_opens_when_already_serving_it(harness):
    h = harness
    running_router(h, ui=True)
    result = h.run("ui")
    assert result.exit_code == 0 and h.opened == ["http://127.0.0.1:5678/"]
    assert h.backend.spawned == []


def test_ui_refuses_to_restart_non_interactively(harness):
    h = harness
    running_router(h, ui=False, children=[(43, "big", 0)])
    result = h.run("ui")
    assert result.exit_code == 1
    assert "can only be enabled at startup" in result.output
    assert "  big" in result.output
    assert "Not an interactive shell" in result.output


def test_ui_yes_restarts_with_ui(harness):
    h = harness
    running_router(h, ui=False)
    h.backend.spawn_listening = {5678}
    result = h.run("ui", "-y")
    assert result.exit_code == 0, result.output
    assert "--ui" in h.backend.spawned[0][0] and h.opened == ["http://127.0.0.1:5678/"]


def test_edit_opens_the_preset(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(cli, "_run_editor", lambda path: seen.append(path) or 0)
    assert h.run("edit").exit_code == 0
    assert seen == [h.paths.preset]


def test_complete_model_reads_the_preset(harness):
    assert cli.complete_model("b") == ["big"]
    assert cli.complete_model("") == ["big", "small"]
