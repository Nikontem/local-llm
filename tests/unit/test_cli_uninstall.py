from local_llm.preset import Preset

from .test_uninstall import populate


def test_dry_run_prints_the_plan_and_changes_nothing(harness):
    h = harness
    populate_paths = populate(
        h.tmp
    )  # re-populates the harness home (overwrites the fixture preset)
    before = sorted(str(p) for p in h.tmp.rglob("*"))
    result = h.run("uninstall", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "models" in result.output and "[a]" in result.output and "a-Q4_0.gguf" in result.output
    assert "integrations" in result.output and "local-llm-models.js" in result.output
    assert "state" in result.output and str(populate_paths.state_dir) in result.output
    assert "config" in result.output and "models.ini" in result.output
    assert "dry run" in result.output.lower()
    assert sorted(str(p) for p in h.tmp.rglob("*")) == before


def test_yes_without_a_selection_is_refused(harness):
    populate(harness.tmp)
    result = harness.run("uninstall", "--yes")
    assert result.exit_code == 1 and "--all" in result.output


def test_all_yes_removes_everything_and_prints_the_hint(harness):
    h = harness
    paths = populate(h.tmp)
    result = h.run("uninstall", "--all", "--yes")
    assert result.exit_code == 0, result.output
    assert not paths.preset.exists() and not paths.state_dir.exists()
    assert not (h.tmp / ".config" / "opencode" / "plugins" / "local-llm-models.js").exists()
    assert not (h.tmp / "a-Q4_0.gguf").exists() and not (h.tmp / "b-Q8_0.gguf").exists()
    assert "uninstall local-llm" in result.output
    again = h.run("uninstall", "--all", "--yes")
    assert again.exit_code == 0 and "nothing" in again.output.lower()


def test_models_only_stops_a_running_router_first(harness):
    h = harness
    paths = populate(h.tmp)
    h.backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    paths.pid_file.write_text("42\n")
    result = h.run("uninstall", "--models", "--yes")
    assert result.exit_code == 0, result.output
    assert "Router is down." in result.output
    assert Preset.load(paths.preset).sections() == []
    assert (h.tmp / ".config" / "opencode" / "plugins" / "local-llm-models.js").exists()


def test_interactive_menu_selects_models_and_a_subset(harness):
    h = harness
    paths = populate(h.tmp)
    result = h.run("uninstall", input="1\n2\ny\n")
    assert result.exit_code == 0, result.output
    assert Preset.load(paths.preset).sections() == ["a"]
    assert not (h.tmp / "b-Q8_0.gguf").exists() and (h.tmp / "a-Q4_0.gguf").exists()


def test_restore_shell_line_flag(harness):
    h = harness
    populate(h.tmp)
    result = h.run("uninstall", "--integrations", "--restore-shell-line", "--yes")
    assert result.exit_code == 0, result.output
    assert "retired by local-llm" not in (h.tmp / ".zshrc").read_text()
    assert "source" in (h.tmp / ".zshrc").read_text()
