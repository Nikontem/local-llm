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


def fake_which(monkeypatch, *names):
    installed = set(names)
    monkeypatch.setattr(
        cli, "_which", lambda binary: f"/usr/local/bin/{binary}" if binary in installed else None
    )


def interactive(monkeypatch):
    """The shared fixture reports no terminal; these tests are about being asked."""
    monkeypatch.setattr(cli, "_interactive", lambda: True)


def test_integrate_menu_lists_and_configures(harness):
    h = harness
    fake_which(h.monkeypatch, "codex", "gemini")
    interactive(h.monkeypatch)
    result = h.run("integrate", input="1\ny\n")
    assert result.exit_code == 0, result.output
    assert "OpenAI Codex CLI" in result.output and "Not found:" in result.output
    assert "Gemini CLI" in result.output and "google-antigravity" not in result.output
    config = h.tmp / ".codex" / "config.toml"
    assert '[model_providers.local-llm]' in config.read_text()
    assert "experimental" in result.output


def test_integrate_menu_none_writes_nothing(harness):
    h = harness
    fake_which(h.monkeypatch, "codex")
    interactive(h.monkeypatch)
    result = h.run("integrate", input="n\n")
    assert result.exit_code == 0, result.output
    assert not (h.tmp / ".codex").exists()


def test_integrate_menu_without_a_terminal_only_prints(harness):
    h = harness
    fake_which(h.monkeypatch, "codex")
    result = h.run("integrate")  # the fixture reports a non-interactive terminal
    assert result.exit_code == 0, result.output
    assert "OpenAI Codex CLI" in result.output
    assert not (h.tmp / ".codex").exists()


def test_integrate_codex_subcommand(harness):
    h = harness
    result = h.run("integrate", "codex", "--yes")
    assert result.exit_code == 0, result.output
    assert '[profiles.local-llm]' in (h.tmp / ".codex" / "config.toml").read_text()


def test_integrate_opencode_subcommand_still_works(hubbed):
    result = hubbed.run("integrate", "opencode", "--yes")
    assert result.exit_code == 0, result.output


def test_integrate_help_carries_the_explanations(harness):
    result = harness.run("integrate", "--help")
    assert result.exit_code == 0
    assert "Gemini CLI cannot be pointed at the router" in result.output
    assert "google-antigravity" in result.output
    assert "project-level" in result.output


def test_integrate_warns_when_the_router_is_not_loopback(harness):
    h = harness
    fake_which(h.monkeypatch, "codex")
    interactive(h.monkeypatch)
    h.monkeypatch.setenv("LOCAL_LLM_HOST", "0.0.0.0")
    h.monkeypatch.setenv("LOCAL_LLM_ALLOW_REMOTE", "1")
    cli._state = None
    result = h.run("integrate", input="n\n")
    assert "not a loopback address" in result.output
