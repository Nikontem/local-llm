from __future__ import annotations

from local_llm import cli
from local_llm.shellrc import MARK_BEGIN


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
    config = (h.tmp / ".codex" / "config.toml").read_text()
    assert "[model_providers.local-llm]" in config and "[profiles." not in config
    profile = (h.tmp / ".codex" / "local-llm.config.toml").read_text()
    assert 'model_provider = "local-llm"' in profile


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


def test_a_file_we_refuse_to_touch_is_not_touched_by_the_completion_installer(
    tmp_path, monkeypatch
):
    """typer's installer appends to the same rc file with no guard of any kind, so
    letting it run rewrote the very file the line above said was left exactly as it is."""
    from local_llm import cli

    rc = tmp_path / ".zshrc"
    rc.write_text(f"export PATH=/usr/local/bin\n{MARK_BEGIN}\nalias half=x\n")
    before = rc.read_bytes()
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))

    lines = cli._integrate_shell("zsh", aliases=True, yes=True)

    assert rc.read_bytes() == before, "the refused file was rewritten anyway"
    assert any("left exactly as it is" in line for line in lines)
    assert any("completion" in line and "not installed" in line for line in lines)
    assert not (tmp_path / ".zfunc").exists()


def test_a_shell_file_that_is_not_utf8_never_raises_at_the_person(tmp_path, monkeypatch):
    """typer's installer reads the rc file in UTF-8 and raises; nothing caught it."""
    from local_llm import cli

    rc = tmp_path / ".zshrc"
    rc.write_bytes("export CAFE=caf\xe9\n".encode("iso-8859-1"))
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))

    lines = cli._integrate_shell("zsh", aliases=True, yes=True)

    assert any("not UTF-8" in line for line in lines)
    assert rc.read_bytes() == "export CAFE=caf\xe9\n".encode("iso-8859-1")


def test_a_completion_installer_that_fails_is_reported_not_raised(tmp_path, monkeypatch):
    from local_llm import cli

    (tmp_path / ".zshrc").write_text("export A=1\n")
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))

    def refuse(shell):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(cli, "install_completion", refuse)
    lines = cli._integrate_shell("zsh", aliases=True, yes=True)

    assert any("could not install" in line for line in lines), lines


def test_our_block_goes_in_before_the_completion_installer_appends(tmp_path, monkeypatch):
    """The installer appends its own lines to the same rc file. Running it first meant
    our block was worked out from a copy of the file taken before those lines existed."""
    from local_llm import cli

    rc = tmp_path / ".zshrc"
    rc.write_text("export A=1\n")
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))

    order: list[str] = []

    def fake_install(shell):
        order.append("completion")
        rc.write_text(rc.read_text() + "fpath+=~/.zfunc; autoload -Uz compinit; compinit\n")
        return tmp_path / ".zfunc" / "_local-llm"

    monkeypatch.setattr(cli, "install_completion", fake_install)
    original_write = cli.atomic_write

    def watched(target, text, **kwargs):
        order.append("block")
        original_write(target, text, **kwargs)

    monkeypatch.setattr(cli, "atomic_write", watched)

    lines = cli._integrate_shell("zsh", aliases=True, yes=True)

    assert order == ["block", "completion"]
    assert any("completion written to" in line for line in lines)
    after = rc.read_text()
    assert MARK_BEGIN in after and "export A=1" in after
    assert "fpath+=~/.zfunc" in after, "the installer's lines were overwritten"


def test_an_alias_of_the_same_name_elsewhere_in_the_rc_file_is_reported(tmp_path, monkeypatch):
    """One of the two definitions silently does nothing, and nothing used to say so."""
    rc = tmp_path / ".zshrc"
    rc.write_text("alias local_llm='ollama run llama3'\nexport A=1\n")
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(cli, "install_completion", lambda shell: tmp_path / "completion")

    lines = cli._integrate_shell("zsh", aliases=True, yes=True)

    assert any("local_llm" in line and "ours wins" in line for line in lines), lines

    rc.write_text(rc.read_text() + "alias local_llm='ollama run llama3'\n")
    again = cli._integrate_shell("zsh", aliases=True, yes=True)

    assert any("that one wins" in line for line in again), again
