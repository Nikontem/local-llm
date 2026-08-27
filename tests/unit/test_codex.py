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
    moved = codex.codex_paths(home=tmp_path, env={"CODEX_HOME": str(tmp_path / "elsewhere")})
    assert moved.config_file == tmp_path / "elsewhere" / "config.toml"


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
    assert "# my codex config" in text
    assert '[model_providers.other]' in text and '[profiles.work]' in text
    assert '[model_providers.local-llm]' in text and 'wire_api = "responses"' in text
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
    assert removed == ["[model_providers.local-llm]", "[profiles.local-llm]"]
    text = paths.config_file.read_text()
    assert "local-llm" not in text
    assert "# my codex config" in text and "[model_providers.other]" in text
    assert codex.drop(paths) == []


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
    assert 'model = "small"' in paths.config_file.read_text()
    assert any("already" in line for line in codex.configure(ctx))


def test_configure_without_models_writes_the_provider_only(tmp_path):
    ctx = make_ctx(tmp_path, models=())
    lines = codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    assert "[profiles.local-llm]" not in paths.config_file.read_text()
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

    paths.config_file.write_bytes(b"\xff\xfe binary junk")
    assert codex.status(paths, provider, profile) == "unreadable"
    assert codex.parse_problem(paths) is not None
    assert codex.has_tables(paths) is False
    assert codex.configured_base_url(paths) is None
    assert codex.current_render(paths) == ""
    assert any("not rewritten" in line for line in codex.removal_lines(paths))

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
