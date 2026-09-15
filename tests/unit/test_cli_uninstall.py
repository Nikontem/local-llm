from local_llm.preset import Preset

from .test_uninstall import populate


def test_dry_run_prints_the_plan_and_changes_nothing(harness):
    h = harness
    populate_paths = populate(
        h.tmp
    )  # re-populates the harness home (overwrites the fixture preset)
    before = {str(p): p.read_bytes() for p in h.tmp.rglob("*") if p.is_file()}
    result = h.run("uninstall", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "models" in result.output and "[a]" in result.output and "a-Q4_0.gguf" in result.output
    assert "integrations" in result.output and "local-llm-models.js" in result.output
    assert "state" in result.output and str(populate_paths.state_dir) in result.output
    assert "config" in result.output and "models.ini" in result.output
    assert "dry run" in result.output.lower()
    assert {str(p): p.read_bytes() for p in h.tmp.rglob("*") if p.is_file()} == before


def test_yes_without_a_selection_is_refused(harness):
    populate(harness.tmp)
    result = harness.run("uninstall", "--yes")
    assert result.exit_code == 1 and "--all" in result.output


def test_all_yes_removes_everything_and_prints_the_hint(harness):
    h = harness
    paths = populate(h.tmp)
    result = h.run("uninstall", "--all", "--yes", "--delete-backups")
    assert result.exit_code == 0, result.output
    assert not paths.preset.exists() and not paths.state_dir.exists()
    assert not (h.tmp / ".config" / "opencode" / "plugins" / "local-llm-models.js").exists()
    assert not (h.tmp / "a-Q4_0.gguf").exists() and not (h.tmp / "b-Q8_0.gguf").exists()
    assert "uninstall local-llm" in result.output
    again = h.run("uninstall", "--all", "--yes", "--delete-backups")
    assert again.exit_code == 0
    assert "Nothing of local-llm's is left on this machine." in again.output


def test_a_shell_file_that_is_not_utf8_is_named_and_the_rest_still_goes(harness):
    """It used to raise out of inventory(), before the plan was even printed."""
    h = harness
    paths = populate(h.tmp)
    rc = h.tmp / ".zshrc"
    original = rc.read_bytes() + "export CAFE=caf\xe9\n".encode("iso-8859-1")
    rc.write_bytes(original)

    plan = h.run("uninstall", "--dry-run")
    assert plan.exit_code == 0, plan.output
    assert "not UTF-8" in plan.output and str(rc) in plan.output

    result = h.run("uninstall", "--all", "--yes", "--delete-backups")
    assert result.exit_code == 0, result.output
    assert "not UTF-8" in result.output
    assert rc.read_bytes() == original, "the file was touched"
    assert not paths.preset.exists(), "everything else was still removed"


def test_the_backups_we_own_are_in_the_plan_and_go_with_the_rest(harness):
    h = harness
    populate(h.tmp)
    agent_backup = h.tmp / ".config" / "opencode" / "opencode.json.local-llm.bak"
    agent_backup.write_text('{"provider": {"anthropic": {"options": {"apiKey": "sk-mine"}}}}')
    rc_backup = h.tmp / ".zshrc.local-llm.bak"
    rc_backup.write_text("an older .zshrc")

    plan = h.run("uninstall", "--dry-run")
    assert plan.exit_code == 0, plan.output
    assert str(agent_backup) in plan.output and str(rc_backup) in plan.output

    result = h.run("uninstall", "--integrations", "--yes", "--delete-backups")

    assert result.exit_code == 0, result.output
    assert not agent_backup.exists() and not rc_backup.exists()
    assert not (h.tmp / ".zshrc.local-llm.bak").exists(), "the copy this run made stayed"


def test_a_backup_of_your_own_file_is_never_taken_away_unasked(harness):
    """You never know when you will want back what a file said before we touched it."""
    h = harness
    populate(h.tmp)
    agent_backup = h.tmp / ".config" / "opencode" / "opencode.json.local-llm.bak"
    agent_backup.write_text('{"provider": {"anthropic": {"options": {"apiKey": "sk-mine"}}}}')

    unattended = h.run("uninstall", "--integrations", "--yes")

    assert unattended.exit_code == 0, unattended.output
    assert agent_backup.exists(), "an unattended run deleted it anyway"
    assert "kept" in unattended.output and str(agent_backup) in unattended.output

    kept = h.run("uninstall", "--integrations", input="y\nn\n")

    assert kept.exit_code == 0, kept.output
    assert "delete the backup copies" in kept.output.lower(), "the question was never put"
    assert agent_backup.exists()

    gone = h.run("uninstall", "--integrations", input="y\ny\n")

    assert gone.exit_code == 0, gone.output
    assert not agent_backup.exists()
    assert f"deleted {agent_backup}" in gone.output


def test_keep_backups_answers_the_question_without_being_asked(harness):
    h = harness
    populate(h.tmp)
    rc_backup = h.tmp / ".zshrc.local-llm.bak"
    rc_backup.write_text("an older .zshrc")

    result = h.run("uninstall", "--integrations", "--keep-backups", input="y\n")

    assert result.exit_code == 0, result.output
    assert "delete the backup copies" not in result.output.lower()
    assert rc_backup.exists()


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
    assert "[b]" in result.output  # the plan shows the chosen model only
    assert "a-Q4_0.gguf" not in result.output
    assert Preset.load(paths.preset).sections() == ["a"]
    assert not (h.tmp / "b-Q8_0.gguf").exists() and (h.tmp / "a-Q4_0.gguf").exists()


def test_restore_shell_line_flag(harness):
    h = harness
    populate(h.tmp)
    result = h.run("uninstall", "--integrations", "--restore-shell-line", "--yes")
    assert result.exit_code == 0, result.output
    assert "retired by local-llm" not in (h.tmp / ".zshrc").read_text()
    assert "source" in (h.tmp / ".zshrc").read_text()


def test_uninstall_leaves_a_half_marked_rc_file_alone_and_says_so(harness):
    """A stray begin marker used to pair with the end marker of an appended block, and
    everything between them - a PATH export, a pyenv line - went with the removal."""
    h = harness
    populate(h.tmp)
    rc = h.tmp / ".zshrc"
    original = (
        'export PATH="$HOME/bin:$PATH"\n'
        "alias gs='git status'\n"
        'eval "$(pyenv init -)"\n'
        "# >>> local-llm >>>\n"
    )
    rc.write_text(original)

    result = h.run("uninstall", "--integrations", "--yes", "--delete-backups")

    assert result.exit_code == 0, result.output
    assert rc.read_text() == original, "the file was rewritten"
    assert str(rc) in result.output and "by hand" in result.output
    assert not (h.tmp / ".zshrc.local-llm.bak").exists(), "nothing was rewritten to back up"


def test_uninstall_copies_an_rc_file_aside_and_takes_the_copy_with_it(harness):
    """The copy is what makes the rewrite survivable; a copy left behind is a file left behind."""
    h = harness
    populate(h.tmp)
    rc = h.tmp / ".zshrc"
    copy = h.tmp / ".zshrc.local-llm.bak"

    result = h.run("uninstall", "--integrations", "--yes", "--delete-backups")

    assert result.exit_code == 0, result.output
    assert "# >>> local-llm >>>" not in rc.read_text() and "export A=1" in rc.read_text()
    assert f"deleted {copy}" in result.output, "the rewrite went through a copy"
    assert not copy.exists()


def test_zero_and_out_of_range_numbers_are_refused(harness):
    h = harness
    paths = populate(h.tmp)
    zero_model = h.run("uninstall", input="1\n0\n")
    assert zero_model.exit_code == 1 and "Pick numbers between 1 and 2" in zero_model.output
    zero_menu = h.run("uninstall", input="0\n")
    assert zero_menu.exit_code == 1 and "Pick numbers between 1 and 4" in zero_menu.output
    assert Preset.load(paths.preset).sections() == ["a", "b"]


def test_uninstall_all_removes_the_codex_tables(harness):
    h = harness
    populate(h.tmp)
    assert h.run("uninstall", "--dry-run").output.count("config.toml") >= 1
    result = h.run("uninstall", "--all", "--yes", "--delete-backups")
    assert result.exit_code == 0, result.output
    text = (h.tmp / ".codex" / "config.toml").read_text()
    assert "local-llm" not in text and 'model = "gpt-5"' in text
    backup = h.tmp / ".codex" / "config.toml.local-llm.bak"
    assert not backup.exists(), "the copy this tool made was left beside their config"


def test_uninstall_removes_the_codex_profile_file_and_its_backup(harness):
    h = harness
    populate(h.tmp)
    codex = h.tmp / ".codex"
    profile = codex / "local-llm.config.toml"
    profile.write_text('model = "b"\nmodel_provider = "local-llm"\n')
    backup = codex / "local-llm.config.toml.local-llm.bak"
    backup.write_text('model = "a"\nmodel_provider = "local-llm"\n')

    plan = h.run("uninstall", "--dry-run").output
    assert str(profile) in plan and str(backup) in plan

    result = h.run("uninstall", "--integrations", "--yes", "--delete-backups")
    assert result.exit_code == 0, result.output
    assert not profile.exists() and not backup.exists()
    assert 'model = "gpt-5"' in (codex / "config.toml").read_text()


def test_a_codex_backup_of_ours_goes_even_when_the_config_is_left_alone(harness):
    """It is ours by its name, which nothing else on the machine ever writes."""
    h = harness
    populate(h.tmp)
    codex = h.tmp / ".codex"
    backup = codex / "config.toml.local-llm.bak"
    backup.write_text('model = "gpt-5"\n')
    (codex / "config.toml").write_text('model = "gpt-5"\nmodel_providers = [\n')

    plan = h.run("uninstall", "--dry-run")
    assert str(backup) in plan.output, "the plan never mentioned it"

    result = h.run("uninstall", "--integrations", "--yes", "--delete-backups")

    assert result.exit_code == 0, result.output
    assert not backup.exists(), "the plan said it would go"
    assert (codex / "config.toml").read_text() == 'model = "gpt-5"\nmodel_providers = [\n'


def test_an_orphaned_codex_backup_is_still_found_and_removed(harness):
    """Our tables are gone from the config, but the copy we made of it is not."""
    h = harness
    populate(h.tmp)
    codex = h.tmp / ".codex"
    (codex / "config.toml").write_text('model = "gpt-5"\n')
    backup = codex / "config.toml.local-llm.bak"
    backup.write_text('model = "gpt-5"\n\n[model_providers.local-llm]\nname = "local-llm"\n')

    result = h.run("uninstall", "--integrations", "--yes", "--delete-backups")

    assert result.exit_code == 0, result.output
    assert not backup.exists()
    assert (codex / "config.toml").read_text() == 'model = "gpt-5"\n' 
