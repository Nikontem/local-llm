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


def test_setup_wires_harness_prompts_through_the_wizard(harness):
    """The context the wizard hands a harness must not reach typer or Rich directly."""
    h = harness
    fake_which(h.monkeypatch, "codex")
    seen = {}
    h.monkeypatch.setattr(
        cli.codex_integration, "configure", lambda ctx: seen.setdefault("ctx", ctx) and []
    )
    h.run("integrate", "codex", "--yes")
    assert seen["ctx"].yes is True


def test_menu_survives_a_harness_whose_status_cannot_be_read(harness):
    """A dangling symlink proved nothing: Path.is_file() follows links and answers False,
    so the guarded read was never reached. A file whose bytes are not UTF-8 does reach it."""
    h = harness
    fake_which(h.monkeypatch, "opencode")
    plugin = h.tmp / ".config" / "opencode" / "plugins" / "local-llm-models.js"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b"\xff\xfe not valid utf-8 at all")
    result = h.run("integrate")
    assert result.exit_code == 0, result.output
    assert "opencode" in result.output
    assert "cannot be read" in result.output, "the row says it could not be read"


def test_menu_reasks_after_an_answer_it_cannot_parse(harness):
    h = harness
    fake_which(h.monkeypatch, "codex")
    interactive(h.monkeypatch)
    result = h.run("integrate", input="oops\nn\n")
    assert result.exit_code == 0, result.output
    assert "Pick numbers between 1 and 1" in result.output
    assert not (h.tmp / ".codex").exists()


def test_menu_with_yes_configures_every_agent_found(harness):
    h = harness
    fake_which(h.monkeypatch, "codex", "claude")
    result = h.run("integrate", "--yes")
    assert result.exit_code == 0, result.output
    assert "[model_providers.local-llm]" in (h.tmp / ".codex" / "config.toml").read_text()
    assert "local-llm claude" in result.output


def test_the_fixture_isolates_other_tools_config_directories(harness, monkeypatch):
    """A developer with these exported must not have their real config rewritten."""
    import os

    assert "CODEX_HOME" not in os.environ
    assert "OPENCODE_CONFIG_DIR" not in os.environ


def test_a_plugin_that_is_not_utf8_does_not_end_the_menu(harness):
    """The whole run used to die here: Codex was configured, then a traceback."""
    h = harness
    fake_which(h.monkeypatch, "codex", "opencode", "claude")
    plugin = h.tmp / ".config" / "opencode" / "plugins" / "local-llm-models.js"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b"\xff\xfe not valid utf-8 at all")
    result = h.run("integrate", "--yes")
    assert result.exit_code == 0, result.output
    assert "[model_providers.local-llm]" in (h.tmp / ".codex" / "config.toml").read_text()
    assert "cannot be read" in result.output
    assert "local-llm claude" in result.output, "the harnesses after opencode still ran"
    assert plugin.read_bytes() == b"\xff\xfe not valid utf-8 at all"


def test_a_harness_that_raises_does_not_stop_the_ones_after_it(harness):
    """Anything a harness can throw is one printed line, never the end of the loop."""
    import dataclasses

    from local_llm import harnesses

    h = harness

    def explode(ctx):
        raise ValueError("something nobody predicted")

    broken = dataclasses.replace(harnesses.find("codex"), configure=explode)
    registry = tuple(broken if entry.key == "codex" else entry for entry in harnesses.REGISTRY)
    h.monkeypatch.setattr(harnesses, "REGISTRY", registry)
    fake_which(h.monkeypatch, "codex", "claude")
    result = h.run("integrate", "--yes")
    assert result.exit_code == 0, result.output
    assert "could not configure OpenAI Codex CLI: something nobody predicted" in result.output
    assert "local-llm claude" in result.output


def test_yes_before_the_subcommand_name_still_counts(harness):
    """--yes binds to the group callback, so `integrate --yes codex` set it there
    and the subcommand never saw it."""
    h = harness
    seen: list[bool] = []
    h.monkeypatch.setattr(
        cli.codex_integration, "configure", lambda ctx: seen.append(ctx.yes) or []
    )
    assert h.run("integrate", "--yes", "codex").exit_code == 0
    assert h.run("integrate", "codex", "--yes").exit_code == 0
    assert h.run("integrate", "codex").exit_code == 0
    assert seen == [True, True, False]
