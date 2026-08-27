from __future__ import annotations

from local_llm import cli


def capture_exec(monkeypatch):
    """Record what a launcher would have exec'd instead of replacing the process."""
    calls = []
    monkeypatch.setattr(
        cli, "exec_with_env", lambda program, args, env: calls.append((program, args, env))
    )
    return calls


def test_aider_launcher_prefixes_the_model(harness):
    calls = capture_exec(harness.monkeypatch)
    result = harness.run("aider", "small", "--", "--no-auto-commits")
    assert result.exit_code == 0, result.output
    program, args, env = calls[0]
    assert program == "aider"
    assert args == ["--model", "openai/small", "--no-auto-commits"]
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:5678/v1"
    assert "aider -> small" in result.output


def test_qwen_launcher_sets_all_three_variables(harness):
    calls = capture_exec(harness.monkeypatch)
    result = harness.run("qwen", "small")
    assert result.exit_code == 0, result.output
    _, _, env = calls[0]
    assert env["OPENAI_MODEL"] == "small" and env["OPENAI_BASE_URL"].endswith("/v1")


def test_launchers_refuse_an_unknown_model(harness):
    result = harness.run("aider", "nope")
    assert result.exit_code == 1 and "Unknown model" in result.output
