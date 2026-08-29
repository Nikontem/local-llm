import json
from pathlib import Path

from local_llm import cli
from local_llm.integrations.opencode import PLUGIN_NAME, plugin_source
from local_llm.shellrc import MARK_BEGIN, RETIRED_PREFIX


def test_integrate_opencode_installs_plugin_and_merges_agent(hubbed):
    h = hubbed
    oc = h.tmp / ".config" / "opencode"
    oc.mkdir(parents=True)
    (oc / "opencode.json").write_text('{"$schema": "https://opencode.ai/config.json"}\n')
    result = h.run("integrate", "opencode", "--yes")
    assert result.exit_code == 0, result.output
    assert (oc / "plugins" / PLUGIN_NAME).read_text() == plugin_source()
    data = json.loads((oc / "opencode.json").read_text())
    assert data["agent"]["tiny"]["model"] == "llamacpp/small"  # the smallest configured model
    assert "installed" in result.output and "tiny" in result.output
    again = h.run("integrate", "opencode", "--yes")
    assert "already" in again.output


def test_integrate_opencode_prints_snippet_for_jsonc(hubbed):
    h = hubbed
    oc = h.tmp / ".config" / "opencode"
    oc.mkdir(parents=True)
    (oc / "opencode.jsonc").write_text('{\n  // keep me\n  "agent": {}\n}\n')
    result = h.run("integrate", "opencode", "--yes")
    assert result.exit_code == 0, result.output
    assert "// keep me" in (oc / "opencode.jsonc").read_text()  # untouched
    assert '"tiny"' in result.output and "paste" in result.output.lower()


def test_integrate_opencode_without_agent_or_preset(harness):
    h = harness
    h.paths.preset.unlink()
    result = h.run("integrate", "opencode", "--no-agent", "--yes")
    assert result.exit_code == 0, result.output
    assert (h.tmp / ".config" / "opencode" / "plugins" / PLUGIN_NAME).is_file()


def test_completion_install_writes_block_and_retires_old_line(harness):
    h = harness
    written = []
    h.monkeypatch.setattr(
        cli, "install_completion", lambda shell: written.append(shell) or Path(f"/fake/_{shell}")
    )
    rc = h.tmp / ".zshrc"
    rc.write_text(
        'export A=1\n[[ -r "$HOME/.config/local-llm/local_llm.zsh" ]]'
        ' && source "$HOME/.config/local-llm/local_llm.zsh"\n'
    )
    result = h.run("completion", "install", "--shell", "zsh", "--yes")
    assert result.exit_code == 0, result.output
    assert written == ["zsh"]
    text = rc.read_text()
    assert MARK_BEGIN in text and "alias local_llm='local-llm'" in text
    assert RETIRED_PREFIX in text and "retired 1 old line" in result.output
    assert "/fake/_zsh" in result.output
    again = h.run("completion", "install", "--shell", "zsh", "--no-aliases", "--yes")
    assert again.exit_code == 0 and MARK_BEGIN not in rc.read_text()


# A shell startup file worth losing: a PATH export, somebody's own alias, and the
# line that makes pyenv work, under a begin marker whose end marker somebody deleted.
HALF_MARKED = (
    'export PATH="$HOME/bin:$PATH"\n'
    "alias gs='git status'\n"
    'eval "$(pyenv init -)"\n'
    f"{MARK_BEGIN}\n"
)


def test_a_half_marked_rc_file_is_left_exactly_as_it_is(harness):
    """The reviewer's reproduction: a fresh block was appended below the stray marker,
    and the next removal deleted every line between the two, emptying the file."""
    h = harness
    h.monkeypatch.setattr(cli, "install_completion", lambda shell: Path("/fake/_zsh"))
    rc = h.tmp / ".zshrc"
    rc.write_text(HALF_MARKED)

    result = h.run("completion", "install", "--shell", "zsh", "--yes")

    assert result.exit_code == 0, result.output
    assert rc.read_text() == HALF_MARKED, "the file was rewritten"
    assert str(rc) in result.output and "by hand" in result.output
    assert not (h.tmp / ".zshrc.local-llm.bak").exists(), "nothing was rewritten to back up"

    removing = h.run("completion", "install", "--shell", "zsh", "--no-aliases", "--yes")
    assert removing.exit_code == 0, removing.output
    assert rc.read_text() == HALF_MARKED
    assert "by hand" in removing.output


def test_a_shell_file_that_is_not_utf8_is_reported_and_never_rewritten(harness):
    """Read in the locale's encoding and written back in UTF-8, it would be transcoded."""
    h = harness
    h.monkeypatch.setattr(cli, "install_completion", lambda shell: Path("/fake/_zsh"))
    rc = h.tmp / ".zshrc"
    original = "export CAFE=caf\xe9\n".encode("iso-8859-1")
    rc.write_bytes(original)

    result = h.run("completion", "install", "--shell", "zsh", "--yes")

    assert result.exit_code == 0, result.output
    assert rc.read_bytes() == original, "the file was rewritten"
    assert "not UTF-8" in result.output and str(rc) in result.output
    assert not (h.tmp / ".zshrc.local-llm.bak").exists(), "nothing was rewritten to back up"
    # The completion installer appends its own lines to this same file and has no
    # guards of its own, so a file we refuse to touch is refused to it as well. On
    # this one it does not merely rewrite the file: it raises while reading it.
    assert "/fake/_zsh" not in result.output
    assert "completion was not installed" in result.output


def test_the_rc_file_is_copied_aside_before_it_is_rewritten(harness):
    h = harness
    h.monkeypatch.setattr(cli, "install_completion", lambda shell: Path("/fake/_zsh"))
    rc = h.tmp / ".zshrc"
    original = 'export PATH="$HOME/bin:$PATH"\nalias gs=\'git status\'\n'
    rc.write_text(original)
    rc.chmod(0o644)

    result = h.run("completion", "install", "--shell", "zsh", "--yes")

    assert result.exit_code == 0, result.output
    assert MARK_BEGIN in rc.read_text()
    assert (h.tmp / ".zshrc.local-llm.bak").read_text() == original
    assert rc.stat().st_mode & 0o777 == 0o644, "the file's own permissions were changed"
    assert not list(h.tmp.glob(".zshrc.*")) or all(
        p.name == ".zshrc.local-llm.bak" for p in h.tmp.glob(".zshrc.*")
    ), "no temporary file left behind"


def test_setup_yes_runs_the_wizard(hubbed):
    h = hubbed
    h.monkeypatch.setattr(cli, "install_completion", lambda shell: Path(f"/fake/_{shell}"))
    import local_llm.setup as setup_module

    def fake_run_checks(paths, settings, env=None, router_pid=None):
        return [
            cli.Check("platform", "ok", "Darwin arm64"),
            cli.Check("brew", "ok", "/opt/homebrew/bin/brew"),
            cli.Check("llama-server", "ok", "/opt/homebrew/bin/llama-server"),
            cli.Check("router support", "ok", "yes"),
            cli.Check("hf", "ok", "/opt/homebrew/bin/hf"),
            cli.Check("hf token", "ok", "logged in as nikos"),
        ]

    h.monkeypatch.setattr(setup_module, "run_checks", fake_run_checks)
    h.backend.spawn_listening = {5678}
    h.http.responses[("POST", "/v1/chat/completions")] = {
        "choices": [{"message": {"content": "Hi there friend"}}]
    }
    result = h.run("setup", "--yes", "--model", "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q8_0")
    assert result.exit_code == 0, result.output
    assert "1. Prerequisites" in result.output and "7. Start" in result.output
    assert "Hi there friend" in result.output
    assert (h.tmp / ".config" / "local-llm" / "settings.toml").is_file()
    assert "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q8_0" in h.paths.preset.read_text()


def test_setup_detects_agents_through_the_injectable_hook(hubbed):
    """Step 6 must not read the real executable search path, or the wizard would
    configure whatever the developer running the tests happens to have installed."""
    h = hubbed
    h.monkeypatch.setattr(cli, "_which", lambda binary: None)
    h.monkeypatch.setattr(cli, "install_completion", lambda shell: Path(f"/fake/_{shell}"))
    import local_llm.setup as setup_module

    h.monkeypatch.setattr(
        setup_module,
        "run_checks",
        lambda *a, **k: [
            cli.Check("brew", "ok", "/opt/homebrew/bin/brew"),
            cli.Check("llama-server", "ok", "/opt/homebrew/bin/llama-server"),
            cli.Check("hf", "ok", "/opt/homebrew/bin/hf"),
            cli.Check("hf token", "ok", "logged in as nikos"),
            cli.Check("router support", "ok", "yes"),
        ],
    )
    h.backend.spawn_listening = {5678}
    h.http.responses[("POST", "/v1/chat/completions")] = {
        "choices": [{"message": {"content": "Hi"}}]
    }
    result = h.run("setup", "--yes")
    assert result.exit_code == 0, result.output
    assert "No coding agents found on PATH." in result.output
    assert not (h.tmp / ".codex").exists()
