import os

import pytest

from local_llm.integrations import HarnessContext, codex
from local_llm.preset import Preset
from local_llm.settings import Settings


def test_paths_default_and_codex_home(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    assert paths.config_dir == tmp_path / ".codex"
    assert paths.config_file == tmp_path / ".codex" / "config.toml"
    assert paths.backup == tmp_path / ".codex" / "config.toml.local-llm.bak"
    assert paths.profile_file == tmp_path / ".codex" / "local-llm.config.toml"
    assert paths.profile_backup == tmp_path / ".codex" / "local-llm.config.toml.local-llm.bak"
    moved = codex.codex_paths(home=tmp_path, env={"CODEX_HOME": str(tmp_path / "elsewhere")})
    assert moved.config_file == tmp_path / "elsewhere" / "config.toml"
    assert moved.profile_file == tmp_path / "elsewhere" / "local-llm.config.toml"


def test_provider_table_omits_env_key_without_an_api_key():
    plain = codex.provider_table(Settings(port=5678, host="127.0.0.1"))
    assert plain == {
        "name": "local-llm",
        "base_url": "http://127.0.0.1:5678/v1",
        "wire_api": "responses",
    }
    keyed = codex.provider_table(Settings(api_key="secret"))
    assert keyed["env_key"] == "LOCAL_LLM_API_KEY"


def test_status_reports_missing_same_different_and_unreadable(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    provider = codex.provider_table(Settings())
    profile = codex.profile_table("small")
    assert codex.status(paths, provider, profile) == "missing"
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text('model = "gpt-5"\n')
    assert codex.status(paths, provider, profile) == "missing"
    codex.write(paths, provider, profile)
    assert codex.status(paths, provider, profile) == "same"
    assert codex.status(paths, codex.provider_table(Settings(port=9999)), profile) == "different"
    paths.config_file.write_text("[oops\n")
    assert codex.status(paths, provider, profile) == "unreadable"
    assert codex.parse_problem(paths) is not None


EXISTING = """# my codex config
model = "gpt-5"

[model_providers.other]
name = "Other"
base_url = "https://example.invalid/v1"

[profiles.work]
model = "gpt-5"
"""


def test_write_preserves_everything_else_and_keeps_a_backup(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))
    text = paths.config_file.read_text()
    assert '[model_providers.local-llm]' in text and 'wire_api = "responses"' in text
    assert "[profiles.local-llm]" not in text, "the table Codex 0.134+ refuses"
    assert paths.profile_file.read_text() == 'model = "small"\nmodel_provider = "local-llm"\n'

    # Everything above the first table we own is byte for byte what it was.
    head = text[: text.index("[model_providers.local-llm]")]
    assert head == EXISTING[: EXISTING.index("[profiles.work]")]

    # And nothing of the person's below it was changed, dropped or reordered. The
    # comparison drops blank lines because tomlkit takes the blank line that separated
    # the last provider from [profiles.work] as the slot for our table: where our tables
    # land inside an existing file is a recorded, deliberately deferred decision, so this
    # test says what is true rather than asserting a layout the writer does not promise.
    remaining = [line for line in text.splitlines() if line.strip()]
    for line in (line for line in EXISTING.splitlines() if line.strip()):
        assert line in remaining, f"{line!r} did not survive the write"
        remaining = remaining[remaining.index(line) + 1 :]

    assert paths.backup.read_text() == EXISTING
    assert not list(paths.config_dir.glob(".config.toml.*")), "no temporary file left behind"


def test_write_creates_the_file_when_absent(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    codex.write(paths, codex.provider_table(Settings()), None)
    assert paths.config_file.read_text().startswith("[model_providers.local-llm]")
    assert not paths.backup.exists()


def test_drop_removes_only_our_tables(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))
    removed = codex.drop(paths)
    assert removed == ["[model_providers.local-llm]"]
    text = paths.config_file.read_text()
    assert "local-llm" not in text
    assert "# my codex config" in text and "[model_providers.other]" in text
    assert "[profiles.work]" in text
    assert codex.drop(paths) == []
    assert paths.profile_file.is_file(), "drop is config.toml only; drop_profile is the rest"


def test_drop_takes_out_a_legacy_profile_too(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(
        'profile = "local-llm"\n\n[model_providers.local-llm]\nname = "local-llm"\n\n'
        '[profiles.local-llm]\nmodel = "small"\n\n[profiles.work]\nmodel = "gpt-5"\n'
    )
    assert codex.drop(paths) == [
        "[model_providers.local-llm]",
        "[profiles.local-llm]",
        'profile = "local-llm"',
    ]
    text = paths.config_file.read_text()
    assert "local-llm" not in text and "[profiles.work]" in text


def test_drop_keeps_a_parent_table_that_holds_a_comment(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(
        "[model_providers]\n# mine, keep it\n\n"
        '[model_providers.local-llm]\nname = "local-llm"\n'
    )
    assert codex.drop(paths) == ["[model_providers.local-llm]"]
    assert "# mine, keep it" in paths.config_file.read_text()


def make_ctx(tmp_path, *, models=("small",), yes=True, answers=None):
    from local_llm.paths import Paths

    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    body = "[*]\nc = 8192\n"
    for index, name in enumerate(models):
        blob = tmp_path / f"{name}.gguf"
        blob.write_bytes(b"x" * (100 * (index + 1)))
        body += f"[{name}]\nmodel = {blob}\n"
    paths.preset.write_text(body)
    said: list[str] = []
    replies = list(answers or [])
    ctx = HarnessContext(
        paths=paths,
        settings=Settings(),
        preset=Preset.load(paths.preset) if models else None,
        home=tmp_path,
        env={},
        say=said.append,
        confirm=lambda prompt, default: replies.pop(0) if replies else default,
        yes=yes,
    )
    ctx.said = said  # type: ignore[attr-defined]
    return ctx


def test_configure_writes_then_reports_already(tmp_path):
    ctx = make_ctx(tmp_path)
    lines = codex.configure(ctx)
    assert any("written to" in line for line in lines)
    assert any("codex --profile local-llm" in line for line in lines)
    assert any("experimental" in line for line in lines)
    paths = codex.codex_paths(home=tmp_path, env={})
    assert 'model = "small"' in paths.profile_file.read_text()
    assert 'model = "small"' not in paths.config_file.read_text()
    assert any("already" in line for line in codex.configure(ctx))


def test_configure_writes_the_profile_as_its_own_file(tmp_path):
    """Codex 0.134+ reads a profile from <name>.config.toml with top-level keys, and
    refuses to start when a [profiles.<name>] table is in config.toml instead."""
    codex.configure(make_ctx(tmp_path))
    paths = codex.codex_paths(home=tmp_path, env={})
    assert paths.config_file.read_text() == (
        "[model_providers.local-llm]\n"
        'name = "local-llm"\n'
        'base_url = "http://127.0.0.1:5678/v1"\n'
        'wire_api = "responses"\n'
    )
    assert paths.profile_file.read_text() == 'model = "small"\nmodel_provider = "local-llm"\n'


def test_configure_moves_a_legacy_profile_out_of_config_toml(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    legacy = (
        'model = "gpt-5"\nprofile = "local-llm"\n\n'
        "[model_providers.local-llm]\n"
        'name = "local-llm"\nbase_url = "http://127.0.0.1:5678/v1"\nwire_api = "responses"\n\n'
        '[profiles.local-llm]\nmodel = "small"\nmodel_provider = "local-llm"\n'
    )
    paths.config_file.write_text(legacy)
    provider = codex.provider_table(Settings())
    assert codex.status(paths, provider, codex.profile_table("small")) == "different"

    refusing = make_ctx(tmp_path, yes=False, answers=[False])
    assert codex.configure(refusing) == ["left as it is"]
    assert paths.config_file.read_text() == legacy and not paths.profile_file.exists()

    lines = codex.configure(make_ctx(tmp_path, yes=False, answers=[True]))
    text = paths.config_file.read_text()
    assert "[profiles." not in text and "profile = " not in text
    assert 'model = "gpt-5"' in text and "[model_providers.local-llm]" in text
    assert paths.profile_file.read_text() == 'model = "small"\nmodel_provider = "local-llm"\n'
    moved = [line for line in lines if line.startswith("moved the profile out of")]
    assert moved and "[profiles.local-llm]" in moved[0] and 'profile = "local-llm"' in moved[0]
    assert paths.backup.read_text() == legacy
    assert any("already" in line for line in codex.configure(make_ctx(tmp_path)))


def test_status_spans_both_files(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    provider = codex.provider_table(Settings())
    profile = codex.profile_table("small")
    codex.write(paths, provider, profile)
    assert codex.status(paths, provider, profile) == "same"

    paths.profile_file.write_text('model = "other"\nmodel_provider = "local-llm"\n')
    assert codex.status(paths, provider, profile) == "different"

    paths.profile_file.unlink()
    assert codex.status(paths, provider, profile) == "different"

    codex.write(paths, provider, profile)
    paths.config_file.write_text(
        paths.config_file.read_text() + '\n[profiles.local-llm]\nmodel = "small"\n'
    )
    assert codex.status(paths, provider, profile) == "different", "a remnant is never same"

    paths.profile_file.write_text("[oops\n")
    assert codex.status(paths, provider, profile) == "unreadable"


def test_an_unparsable_profile_file_is_reported_not_rewritten(tmp_path):
    ctx = make_ctx(tmp_path)
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.profile_file.write_text("[oops\n")
    lines = codex.configure(ctx)
    assert paths.profile_file.read_text() == "[oops\n"
    assert not paths.config_file.exists(), "neither file is written when one is broken"
    assert any(str(paths.profile_file) in line and "not rewritten" in line for line in lines)
    assert any('model_provider = "local-llm"' in line for line in lines)


def test_a_symlinked_profile_file_is_written_through(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    real = tmp_path / "dotfiles-profile.toml"
    real.write_text('model = "old"\nmodel_provider = "local-llm"\n')
    paths.profile_file.symlink_to(real)

    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))

    assert paths.profile_file.is_symlink()
    assert real.read_text() == 'model = "small"\nmodel_provider = "local-llm"\n'
    assert paths.profile_backup.read_text() == 'model = "old"\nmodel_provider = "local-llm"\n'


def test_removal_deletes_the_profile_file_and_both_backups(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    provider, profile = codex.provider_table(Settings()), codex.profile_table("small")
    codex.write(paths, provider, profile)
    codex.write(paths, provider, codex.profile_table("other"))  # makes the profile backup
    assert paths.backup.is_file() and paths.profile_backup.is_file()

    lines = codex.removal_lines(paths)

    assert not paths.profile_file.exists() and not paths.profile_backup.exists()
    assert not paths.backup.exists()
    assert "local-llm" not in paths.config_file.read_text()
    assert any(line == f"deleted {paths.profile_file}" for line in lines)
    assert any(line == f"deleted {paths.profile_backup}" for line in lines)


def test_a_profile_file_naming_another_provider_is_left_alone(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    theirs = 'model = "gpt-5"\nmodel_provider = "openai"\n'
    paths.profile_file.write_text(theirs)
    assert codex.removal_lines(paths) == []
    assert paths.profile_file.read_text() == theirs

    paths.profile_file.write_text("[oops\n")
    lines = codex.removal_lines(paths)
    assert any("does not parse" in line and "by hand" in line for line in lines)
    assert paths.profile_file.read_text() == "[oops\n"


def test_configure_without_models_writes_the_provider_only(tmp_path):
    ctx = make_ctx(tmp_path, models=())
    lines = codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    assert "[profiles.local-llm]" not in paths.config_file.read_text()
    assert not paths.profile_file.exists()
    assert any("no model in models.ini" in line for line in lines)


def test_configure_asks_before_replacing_a_changed_table(tmp_path):
    ctx = make_ctx(tmp_path)
    codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_file.write_text(
        paths.config_file.read_text().replace("5678", "9999")
    )
    refusing = make_ctx(tmp_path, yes=False, answers=[False])
    assert codex.configure(refusing) == ["left as it is"]
    assert "9999" in paths.config_file.read_text()
    accepting = make_ctx(tmp_path, yes=False, answers=[True])
    codex.configure(accepting)
    assert "9999" not in paths.config_file.read_text()
    assert any("---" in said for said in accepting.said), "a difference was shown"


def test_configure_refuses_an_unparsable_file(tmp_path):
    ctx = make_ctx(tmp_path)
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text("[oops\n")
    lines = codex.configure(ctx)
    assert paths.config_file.read_text() == "[oops\n"
    assert any("not rewritten" in line for line in lines)
    assert any("[model_providers.local-llm]" in line for line in lines)


def test_a_failed_write_leaves_no_temporary_file_behind(tmp_path, monkeypatch):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)

    def refuse(source, destination):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(codex.os, "replace", refuse)
    with pytest.raises(OSError):
        codex.write(paths, codex.provider_table(Settings()), None)
    assert not list(paths.config_dir.glob(".config.toml.*"))
    assert paths.config_file.read_text() == EXISTING


def test_binary_and_unreadable_files_are_reported_not_raised(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    provider = codex.provider_table(Settings())
    profile = codex.profile_table("small")

    paths.config_file.write_bytes(b"\xff\xfe binary junk [model_providers.local-llm]")
    assert codex.status(paths, provider, profile) == "unreadable"
    assert codex.parse_problem(paths) is not None
    assert codex.has_tables(paths) is False
    assert codex.configured_base_url(paths) is None
    assert codex.current_render(paths) == ""
    assert any("not rewritten" in line for line in codex.removal_lines(paths))

    # The same file with nothing of ours in it is somebody else's problem entirely.
    paths.config_file.write_bytes(b"\xff\xfe binary junk")
    assert codex.unparsable_but_ours(paths) is False
    assert codex.removal_lines(paths) == []

    paths.config_file.write_text(EXISTING)
    paths.config_file.chmod(0o000)
    try:
        if os.access(paths.config_file, os.R_OK):  # pragma: no cover - running as root
            pytest.skip("this user can read a chmod 000 file")
        assert codex.status(paths, provider, profile) == "unreadable"
        assert codex.parse_problem(paths) is not None
        assert codex.has_tables(paths) is False
    finally:
        paths.config_file.chmod(0o600)


def test_configure_reports_a_write_that_fails(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)

    def refuse(source, destination):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(codex.os, "replace", refuse)
    lines = codex.configure(ctx)
    assert lines == [
        f"could not update {codex.codex_paths(home=tmp_path, env={}).config_file}:"
        " [Errno 30] Read-only file system"
    ]


def test_backup_never_writes_through_a_symlink(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    elsewhere = tmp_path / "somebody-elses-file"
    elsewhere.write_text("do not touch me\n")
    paths.backup.symlink_to(elsewhere)

    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))

    assert elsewhere.read_text() == "do not touch me\n", "the link target was overwritten"
    assert not paths.backup.is_symlink()
    assert paths.backup.read_text() == EXISTING


def test_our_backup_is_named_so_it_can_never_be_a_persons_own(tmp_path):
    """Uninstall deletes our backup. A hand-made config.toml.bak must survive it."""
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    theirs = paths.config_file.with_name("config.toml.bak")
    theirs.write_text("my own backup, from before local-llm existed\n")

    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))
    assert paths.backup.read_text() == EXISTING
    assert theirs.read_text() == "my own backup, from before local-llm existed\n"

    lines = codex.removal_lines(paths)
    assert any("removed" in line for line in lines)
    assert not paths.backup.exists(), "our own backup goes with the tables"
    assert theirs.read_text() == "my own backup, from before local-llm existed\n"


def test_a_backup_survives_a_removal_that_rewrote_nothing(tmp_path):
    """Nothing of ours in the file means nothing is written, so nothing is deleted."""
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    paths.backup.write_text("an older copy\n")
    assert codex.removal_lines(paths) == []
    assert paths.backup.read_text() == "an older copy\n"


def test_a_symlinked_config_stays_a_symlink(tmp_path):
    """A config.toml linked into a dotfiles repository must be written through, not replaced."""
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real = dotfiles / "codex-config.toml"
    real.write_text(EXISTING)
    paths.config_file.symlink_to(real)

    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))

    assert paths.config_file.is_symlink(), "the link was replaced by a regular file"
    assert paths.config_file.resolve() == real.resolve()
    assert "[model_providers.local-llm]" in real.read_text(), "the dotfiles copy is stale"
    assert "# my codex config" in real.read_text()
    assert not list(dotfiles.glob(".config.toml.*")), "no temporary file left behind"
    assert paths.backup.read_text() == EXISTING, "the backup sits beside the link"


def test_the_written_line_names_a_profile_only_when_one_was_written(tmp_path):
    lines = codex.configure(make_ctx(tmp_path, models=()))
    assert any(line.startswith("provider local-llm written to") for line in lines)
    assert not any("profile written to" in line for line in lines)
    assert any("no model in models.ini" in line for line in lines)

    with_model = codex.configure(make_ctx(tmp_path / "other", models=("small",)))
    written = [line for line in with_model if line.startswith("provider local-llm written to")]
    assert written and "profile written to" in written[0] and "local-llm.config.toml" in written[0]


def test_a_stray_profile_with_no_provider_still_counts_as_configured(tmp_path):
    """Judging on the provider table alone called this 'missing' and overwrote it
    with no difference shown and no question asked."""
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text('[profiles.local-llm]\nmodel = "small"\n')
    provider = codex.provider_table(Settings())
    assert codex.status(paths, provider, codex.profile_table("small")) == "different"

    refusing = make_ctx(tmp_path, yes=False, answers=[False])
    assert codex.configure(refusing) == ["left as it is"]
    assert paths.config_file.read_text() == '[profiles.local-llm]\nmodel = "small"\n'


def test_an_accented_comment_does_not_make_a_config_unparsable(tmp_path):
    """TOML is UTF-8 by definition; without saying so the locale decides, and a
    perfectly valid file is reported as broken."""
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text('# réglages Codex — à garder\nmodel = "gpt-5"\n', encoding="utf-8")
    assert codex.parse_problem(paths) is None
    codex.write(paths, codex.provider_table(Settings()), None)
    text = paths.config_file.read_text(encoding="utf-8")
    assert "# réglages Codex — à garder" in text and "[model_providers.local-llm]" in text


def test_a_remote_router_is_warned_about_wherever_the_write_happens(tmp_path):
    """The menu warned; `local-llm integrate codex` run on its own said nothing at all."""
    from local_llm.integrations import codex

    ctx = make_ctx(tmp_path)
    ctx.settings = Settings(host="192.168.1.40")

    lines = codex.configure(ctx)

    note = [line for line in lines if "192.168.1.40" in line and "loopback" in line]
    assert note, lines
    assert "http" in note[0] and "unencrypted" in note[0]

    local = make_ctx(tmp_path)
    assert not any("loopback" in line for line in codex.configure(local))


def test_a_config_that_held_only_our_tables_is_removed_not_emptied(tmp_path):
    """The file did not exist before this tool wrote it, so an uninstall that leaves a
    zero-byte one behind has not finished putting the machine back."""
    from local_llm.integrations import codex

    ctx = make_ctx(tmp_path)
    codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    assert paths.config_file.is_file()

    lines = codex.removal_lines(paths)

    assert not paths.config_file.exists(), paths.config_file.read_text()
    assert not paths.profile_file.exists(), "the profile file is ours from start to end"
    assert any("removed" in line and str(paths.config_file) in line for line in lines)
    assert not paths.backup.exists()


def test_a_config_with_anything_else_in_it_stays(tmp_path):
    from local_llm.integrations import codex

    ctx = make_ctx(tmp_path)
    codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_file.write_text('model = "gpt-5"\n' + paths.config_file.read_text())

    codex.removal_lines(paths)

    assert paths.config_file.read_text().strip() == 'model = "gpt-5"'


def test_a_config_left_holding_only_a_comment_stays(tmp_path):
    from local_llm.integrations import codex

    ctx = make_ctx(tmp_path)
    codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_file.write_text("# my own notes\n" + paths.config_file.read_text())

    codex.removal_lines(paths)

    assert paths.config_file.is_file() and "# my own notes" in paths.config_file.read_text()


def test_a_caller_that_has_already_warned_is_not_repeated(tmp_path):
    """The menu says a fuller version before asking, so each agent it then configures
    would otherwise add a paragraph of the same thing to the same screen."""
    from local_llm.integrations import codex

    ctx = make_ctx(tmp_path)
    ctx.settings = Settings(host="192.168.1.40")
    ctx.warned_remote = True

    assert not any("loopback" in line for line in codex.configure(ctx))


def test_a_scalar_where_a_table_belongs_never_crashes_anything(tmp_path):
    """`profiles = "work"` is the plausible typo for Codex's own top-level `profile`.

    It is valid TOML, so the file parses and the wrong shape reached straight through.
    """
    from local_llm.integrations import codex

    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True, exist_ok=True)

    for text in ('profiles = "work"\n', 'model_providers = "oops"\n', "profiles = [1, 2]\n"):
        paths.config_file.write_text(text)
        assert codex.has_tables(paths) is False
        assert codex.configured_base_url(paths) is None
        assert codex.current_render(paths) == ""
        assert codex.removal_lines(paths) == []
        assert paths.config_file.read_text() == text, "somebody's file was rewritten"

    ctx = make_ctx(tmp_path)
    paths.config_file.write_text('profiles = "work"\n')
    lines = codex.configure(ctx)
    assert lines and not any("Traceback" in line for line in lines)
