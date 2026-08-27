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
