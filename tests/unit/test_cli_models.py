import json

from local_llm import cli
from local_llm.doctor import Check
from local_llm.estimate import GIB


def running_router(h):
    h.backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    h.paths.ensure_state_dirs()
    h.paths.pid_file.write_text("42\n")


def test_split_agent_args():
    assert cli._split_agent_args("big", ["--resume"]) == ("big", ["--resume"])
    assert cli._split_agent_args("-p", ["hi"]) == (None, ["-p", "hi"])
    assert cli._split_agent_args(None, []) == (None, [])


def test_load_requires_running_router(harness):
    result = harness.run("load", "big")
    assert result.exit_code == 1 and "Router is not running" in result.output


def test_load_rejects_unknown_model(harness):
    running_router(harness)
    result = harness.run("load", "nope")
    assert result.exit_code == 1 and "Unknown model: nope" in result.output


def test_load_two_models_within_budget(harness):
    h = harness
    running_router(h)
    h.monkeypatch.setenv("LOCAL_LLM_MAX_MODELS", "2")
    h.monkeypatch.setattr(cli, "total_ram", lambda: 48 * GIB)
    result = h.run("load", "big", "small")
    assert result.exit_code == 0, result.output
    assert "Requested:" in result.output
    assert "estimated total:" in result.output
    assert "usable memory:    38.0 GB  (RAM minus 10 GB reserved)" in result.output
    assert 'loading big ... {"success":true}' in result.output
    assert ("POST", "/models/load", {"model": "small"}) in h.http.calls


def test_load_more_than_max_models(harness):
    h = harness
    running_router(h)
    h.monkeypatch.setenv("LOCAL_LLM_MAX_MODELS", "1")
    result = h.run("load", "big", "small")
    assert result.exit_code == 1
    assert "holds at most 1" in result.output
    assert "local-llm restart --max-models 2" in result.output


def test_load_over_budget_refuses_unless_forced(harness):
    h = harness
    running_router(h)
    h.monkeypatch.setenv("LOCAL_LLM_MAX_MODELS", "2")
    h.monkeypatch.setattr(cli, "total_ram", lambda: 11 * GIB)  # 1 GB budget; each model needs ~1 GB
    result = h.run("load", "big", "small")
    assert result.exit_code == 1
    assert "Refusing to load: over budget by" in result.output
    assert "local-llm load big small --force" in result.output
    assert h.http.calls == []
    forced = h.run("load", "big", "small", "--force")
    assert forced.exit_code == 0, forced.output
    assert "loading anyway because --force was given" in forced.output
    assert len([c for c in h.http.calls if c[1] == "/models/load"]) == 2


def test_load_reports_missing_file(harness):
    h = harness
    running_router(h)
    (h.tmp / "big.gguf").unlink()
    result = h.run("load", "big")
    assert result.exit_code == 1 and "cannot find its model file" in result.output


def test_unload(harness):
    h = harness
    assert h.run("unload", "big").exit_code == 1
    running_router(h)
    result = h.run("unload", "big")
    assert result.exit_code == 0 and '{"success":true}' in result.output
    assert ("POST", "/models/unload", {"model": "big"}) in h.http.calls


def test_claude_execs_with_environment(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(
        cli, "exec_with_env", lambda program, args, env: seen.append((program, args, env))
    )
    result = h.run("claude", "big", "--", "--resume")
    assert result.exit_code == 0, result.output
    assert "claude -> big (context 65536) at http://127.0.0.1:5678" in result.output
    program, args, env = seen[0]
    assert program == "claude" and args == ["--resume"]
    assert env["ANTHROPIC_MODEL"] == "big" and env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5678"


def test_claude_default_model_and_passthrough_options(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(
        cli, "exec_with_env", lambda program, args, env: seen.append((program, args, env))
    )
    h.monkeypatch.setenv("LOCAL_LLM_DEFAULT_MODEL", "small")
    result = h.run("claude", "-p", "hi")
    assert result.exit_code == 0, result.output
    assert seen[0][1] == ["-p", "hi"] and seen[0][2]["ANTHROPIC_MODEL"] == "small"


def test_claude_without_a_model(harness):
    result = harness.run("claude")
    assert result.exit_code == 1 and "No model given" in result.output


def test_copilot_online_flag(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(
        cli, "exec_with_env", lambda program, args, env: seen.append((program, args, env))
    )
    result = h.run("copilot", "small", "--online", "--", "--foo")
    assert result.exit_code == 0, result.output
    program, args, env = seen[0]
    assert program == "copilot" and args == ["--foo"]
    assert env["COPILOT_OFFLINE"] == "false" and env["COPILOT_MODEL"] == "small"
    assert "copilot -> small (context 8192, offline=false)" in result.output


def test_env_prints_exports(harness):
    result = harness.run("env", "big", "--shell", "fish")
    assert result.exit_code == 0
    assert "set -gx ANTHROPIC_MODEL big\n" in result.output
    assert "set -gx OPENAI_BASE_URL http://127.0.0.1:5678/v1\n" in result.output


def test_doctor_output_and_exit_code(harness):
    h = harness
    checks = [
        Check("platform", "ok", "Darwin arm64"),
        Check("brew", "fail", "not found", fix="install it"),
    ]
    h.monkeypatch.setattr(cli, "run_checks", lambda paths, settings, router_pid=None: checks)
    result = h.run("doctor")
    assert result.exit_code == 1
    assert "ok    platform" in result.output and "FAIL  brew" in result.output
    assert "fix: install it" in result.output
    as_json = h.run("doctor", "--json")
    assert json.loads(as_json.output)[1]["fix"] == "install it"


def test_doctor_all_ok_exits_zero(harness):
    h = harness
    h.monkeypatch.setattr(
        cli, "run_checks", lambda paths, settings, router_pid=None: [Check("a", "ok", "fine")]
    )
    assert h.run("doctor").exit_code == 0
