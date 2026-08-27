import json
from pathlib import Path

from local_llm.integrations import HarnessContext
from local_llm.paths import Paths
from local_llm.preset import Preset
from local_llm.settings import Settings
from local_llm.shellrc import MARK_BEGIN, RETIRED_PREFIX, block_text
from local_llm.uninstall import (
    KEYS,
    delete_model_file,
    inventory,
    remove_config,
    remove_integrations,
    remove_models,
    remove_state,
    tool_uninstall_hint,
)


def ctx_for(home: Path) -> HarnessContext:
    """What a registry entry needs to find its own files, pinned to a scratch home."""
    return HarnessContext(
        paths=Paths.from_env(env={}, home=home),
        settings=Settings(),
        home=home,
        env={},
        yes=True,
    )


def populate(home: Path, *, jsonc: bool = False) -> Paths:
    """A home with everything the tool can leave behind."""
    paths = Paths.from_env(env={}, home=home)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    blob = home / "blob"
    blob.write_bytes(b"x" * 30)
    link = home / "a-Q4_0.gguf"
    link.symlink_to(blob)
    plain = home / "b-Q8_0.gguf"
    plain.write_bytes(b"y" * 10)
    mm = home / "mmproj-F16.gguf"
    mm.write_bytes(b"z" * 5)
    paths.preset.write_text(
        "version = 1\n\n[*]\njinja = true\n\n# keep me\n"
        f"[a]\nmodel = {link}\nmmproj = {mm}\n\n[b]\nmodel = {plain}\n"
    )
    (paths.config_dir / "models.ini.bak").write_text("old")
    paths.settings_file.write_text("port = 5678\n")
    paths.ensure_state_dirs()
    paths.pid_file.write_text("1\n")
    (paths.log_dir / "llm-router.2026.log").write_text("log")
    paths.hub_cache_file.write_text("{}")
    oc = home / ".config" / "opencode"
    (oc / "plugins").mkdir(parents=True)
    (oc / "plugins" / "local-llm-models.js").write_text("// plugin")
    agent = {
        "agent": {"tiny": {"model": "llamacpp/b"}, "other": {"mode": "primary"}},
        "$schema": "x",
    }
    if jsonc:
        (oc / "opencode.jsonc").write_text(
            '{\n  // c\n  "agent": {"tiny": {"model": "llamacpp/b"}}\n}\n'
        )
    else:
        (oc / "opencode.json").write_text(json.dumps(agent, indent=2) + "\n")
    (home / ".zfunc").mkdir()
    (home / ".zfunc" / "_local-llm").write_text("#compdef local-llm")
    (home / ".zshrc").write_text(
        "export A=1\n"
        f'{RETIRED_PREFIX}[[ -r "$HOME/.config/local-llm/local_llm.zsh" ]]'
        ' && source "$HOME/.config/local-llm/local_llm.zsh"\n'
        "fpath+=~/.zfunc; autoload -Uz compinit; compinit\n"
        + block_text(["alias local_llm='local-llm'"])
    )
    (home / ".bash_completions").mkdir()
    (home / ".bash_completions" / "local-llm.sh").write_text("complete")
    (home / ".bashrc").write_text(f"source {home}/.bash_completions/local-llm.sh\nexport B=2\n")
    codex_dir = home / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    (codex_dir / "config.toml").write_text(
        'model = "gpt-5"\n\n'
        "[model_providers.local-llm]\n"
        'name = "local-llm"\n'
        'base_url = "http://127.0.0.1:5678/v1"\n'
        'wire_api = "responses"\n\n'
        "[profiles.local-llm]\n"
        'model = "b"\n'
        'model_provider = "local-llm"\n'
    )
    return paths


def test_inventory_sees_everything(tmp_path):
    paths = populate(tmp_path)
    inv = inventory(paths, home=tmp_path, env={})
    assert [m.section for m in inv.models] == ["a", "b"]
    assert inv.models[0].size == 35 and len(inv.models[0].files) == 2
    assert inv.plugin and inv.plugin.name == "local-llm-models.js"
    assert inv.agent_config and inv.agent_config.name == "opencode.json"
    assert inv.tiny_model == "llamacpp/b" and inv.agent_editable
    assert {p.name for p in inv.completion_files} == {"_local-llm", "local-llm.sh"}
    assert [p.name for p in inv.rc_with_block] == [".zshrc"]
    assert [p.name for p in inv.rc_with_retired] == [".zshrc"]
    assert [p.name for p in inv.rc_with_bash_source] == [".bashrc"]
    assert inv.state_dir == paths.state_dir and inv.settings_file == paths.settings_file
    assert [p.name for p in inv.preset_files] == ["models.ini", "models.ini.bak"]
    assert not inv.empty and KEYS == ("models", "integrations", "state", "config")
    assert "2 model(s)" in inv.summary("models") and "0.0 GB" in inv.summary("models")
    assert "plugin" in inv.summary("integrations") and "aliases" in inv.summary("integrations")


def test_inventory_on_a_clean_home(tmp_path):
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    assert inv.empty and inv.models == [] and inv.plugin is None and inv.state_dir is None
    assert inv.summary("models") == "nothing"


def test_remove_models_deletes_files_and_keeps_the_rest(tmp_path):
    paths = populate(tmp_path)
    lines = remove_models(paths, ["a"])
    preset = Preset.load(paths.preset)
    assert preset.sections() == ["b"] and "# keep me" not in paths.preset.read_text()
    assert not (tmp_path / "a-Q4_0.gguf").exists() and not (tmp_path / "blob").exists()
    assert not (tmp_path / "mmproj-F16.gguf").exists() and (tmp_path / "b-Q8_0.gguf").exists()
    assert any("a-Q4_0.gguf" in line for line in lines) and any("[a]" in line for line in lines)
    assert remove_models(paths, ["nope"]) == ["no section [nope]"]


def test_delete_model_file_removes_symlink_and_blob(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    link = tmp_path / "m.gguf"
    link.symlink_to(blob)
    delete_model_file(link)
    assert not link.exists() and not blob.exists()
    delete_model_file(link)  # already gone: no error


def test_remove_integrations_with_and_without_restore(tmp_path):
    paths = populate(tmp_path)
    inv = inventory(paths, home=tmp_path, env={})
    lines = remove_integrations(inv, ctx_for(tmp_path), restore_retired=True)
    assert not (tmp_path / ".config" / "opencode" / "plugins" / "local-llm-models.js").exists()
    config = json.loads((tmp_path / ".config" / "opencode" / "opencode.json").read_text())
    assert "tiny" not in config["agent"] and config["agent"]["other"] == {"mode": "primary"}
    assert config["$schema"] == "x"
    zshrc = (tmp_path / ".zshrc").read_text()
    assert MARK_BEGIN not in zshrc and RETIRED_PREFIX not in zshrc
    assert '[[ -r "$HOME/.config/local-llm/local_llm.zsh" ]] && source' in zshrc
    assert "fpath+=~/.zfunc; autoload -Uz compinit; compinit" in zshrc and "export A=1" in zshrc
    assert (tmp_path / ".bashrc").read_text() == "export B=2\n"
    assert not (tmp_path / ".zfunc" / "_local-llm").exists()
    assert not (tmp_path / ".bash_completions" / "local-llm.sh").exists()
    assert any("restored" in line for line in lines)
    again = remove_integrations(inventory(paths, home=tmp_path, env={}), ctx_for(tmp_path))
    assert again == []


def test_remove_integrations_keeps_retired_line_by_default_and_skips_jsonc(tmp_path):
    paths = populate(tmp_path, jsonc=True)
    inv = inventory(paths, home=tmp_path, env={})
    assert inv.agent_config and inv.agent_config.name == "opencode.jsonc" and not inv.agent_editable
    lines = remove_integrations(inv, ctx_for(tmp_path))
    assert RETIRED_PREFIX in (tmp_path / ".zshrc").read_text()
    assert "// c" in (tmp_path / ".config" / "opencode" / "opencode.jsonc").read_text()
    assert any('remove the "tiny" agent' in line for line in lines)


def test_remove_state_and_config(tmp_path):
    paths = populate(tmp_path)
    inv = inventory(paths, home=tmp_path, env={})
    remove_state(inv)
    assert not paths.state_dir.exists() and not paths.settings_file.exists()
    (paths.config_dir / "unrelated.txt").write_text("mine")
    lines = remove_config(inventory(paths, home=tmp_path, env={}))
    assert not paths.preset.exists() and not (paths.config_dir / "models.ini.bak").exists()
    assert paths.config_dir.exists() and any("left in place" in line for line in lines)
    (paths.config_dir / "unrelated.txt").unlink()
    remove_config(inventory(paths, home=tmp_path, env={}))
    assert not paths.config_dir.exists()


def test_tool_uninstall_hint_by_executable_location():
    assert (
        tool_uninstall_hint("/Users/x/.local/share/uv/tools/local-llm/bin/python")
        == "uv tool uninstall local-llm"
    )
    assert (
        tool_uninstall_hint("/opt/homebrew/Cellar/local-llm/0.1.0/libexec/bin/python")
        == "brew uninstall local-llm"
    )
    assert (
        tool_uninstall_hint("/home/x/.local/pipx/venvs/local-llm/bin/python")
        == "pipx uninstall local-llm"
    )
    assert tool_uninstall_hint("/usr/bin/python3") == "python3 -m pip uninstall local-llm"


def test_model_outside_the_cache_and_tilde_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True)
    plain = tmp_path / "mine.gguf"
    plain.write_bytes(b"q" * 7)
    paths.preset.write_text("[t]\nmodel = ~/mine.gguf\n")
    inv = inventory(paths, home=tmp_path, env={"HOME": str(tmp_path)})
    assert inv.models[0].files == [plain] and inv.models[0].size == 7
    lines = remove_models(paths, ["t"])
    assert not plain.exists() and any("deleted" in line for line in lines)
    assert Preset.load(paths.preset).sections() == []


def test_symlink_to_a_directory_removes_only_the_link(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True)
    directory = tmp_path / "adir"
    directory.mkdir()
    (directory / "inner").write_text("x")
    link = tmp_path / "weird.gguf"
    link.symlink_to(directory)
    paths.preset.write_text(f"[w]\nmodel = {link}\n")
    lines = remove_models(paths, ["w"])
    assert not link.is_symlink() and (directory / "inner").exists()
    assert Preset.load(paths.preset).sections() == []
    assert not any("could not" in line for line in lines)


def test_state_dir_symlink_is_unlinked_not_followed(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    real = tmp_path / "realstate"
    real.mkdir()
    (real / "keep").write_text("k")
    paths.state_dir.parent.mkdir(parents=True)
    paths.state_dir.symlink_to(real)
    inv = inventory(paths, home=tmp_path, env={})
    remove_state(inv)
    assert not paths.state_dir.is_symlink() and (real / "keep").exists()


def test_restore_only_strips_prefixed_lines(tmp_path):
    paths = populate(tmp_path)
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text(zshrc.read_text() + 'echo "# retired by local-llm: nothing"\n')
    remove_integrations(
        inventory(paths, home=tmp_path, env={}), ctx_for(tmp_path), restore_retired=True
    )
    text = zshrc.read_text()
    assert 'echo "# retired by local-llm: nothing"' in text
    assert text.count(RETIRED_PREFIX) == 1 and "&& source" in text


def test_remove_config_also_removes_a_backup_created_after_inventory(tmp_path):
    paths = populate(tmp_path)
    (paths.config_dir / "models.ini.bak").unlink()
    inv = inventory(paths, home=tmp_path, env={})
    remove_models(paths, ["a", "b"])  # Preset.save recreates the backup
    assert (paths.config_dir / "models.ini.bak").exists()
    remove_state(inv)
    remove_config(inv)
    assert not (paths.config_dir / "models.ini.bak").exists() and not paths.config_dir.exists()


def test_a_shell_file_that_is_not_utf8_never_stops_the_uninstall(tmp_path):
    """It used to raise out of inventory(), before the plan was even printed."""
    paths = populate(tmp_path)
    zshrc = tmp_path / ".zshrc"
    original = zshrc.read_bytes() + "export CAFE=caf\xe9\n".encode("iso-8859-1")
    zshrc.write_bytes(original)

    inv = inventory(paths, home=tmp_path, env={})

    assert inv.rc_not_utf8 == [zshrc]
    assert zshrc not in inv.rc_with_block and zshrc not in inv.rc_with_retired

    lines = remove_integrations(inv, ctx_for(tmp_path), restore_retired=True)

    assert zshrc.read_bytes() == original, "the file was touched"
    assert not (tmp_path / ".zshrc.local-llm.bak").exists(), "nothing was rewritten to back up"
    assert any("not UTF-8" in line and str(zshrc) in line for line in lines)
    assert not (tmp_path / ".zfunc" / "_local-llm").exists(), "the rest of the removal still ran"
    assert (tmp_path / ".bashrc").read_text() == "export B=2\n"


def test_a_utf8_shell_file_keeps_its_bytes_through_an_uninstall(tmp_path):
    """On a machine whose locale is not UTF-8, reading and writing used to disagree."""
    paths = populate(tmp_path)
    zshrc = tmp_path / ".zshrc"
    zshrc.write_bytes("export CAFE=caf\u00e9\n".encode() + zshrc.read_bytes())

    remove_integrations(inventory(paths, home=tmp_path, env={}), ctx_for(tmp_path))

    assert zshrc.read_bytes().startswith("export CAFE=caf\u00e9\n".encode())


def test_the_backup_copies_we_own_are_listed_and_removed(tmp_path):
    """A rewrite leaves one beside every file it touches, and nothing else deletes them."""
    paths = populate(tmp_path)
    rc_backup = tmp_path / ".zshrc.local-llm.bak"
    rc_backup.write_text("an older .zshrc")
    agent_backup = tmp_path / ".config" / "opencode" / "opencode.json.local-llm.bak"
    agent_backup.write_text('{"provider": {"anthropic": {"options": {"apiKey": "sk-mine"}}}}')

    inv = inventory(paths, home=tmp_path, env={})

    assert inv.agent_backup == agent_backup and inv.rc_backups == [rc_backup]
    assert "backup" in inv.summary("integrations")

    lines = remove_integrations(inv, ctx_for(tmp_path))

    assert not agent_backup.exists(), "a second copy of a file that holds API keys"
    assert not rc_backup.exists()
    assert any(str(agent_backup) in line for line in lines)
    assert not (tmp_path / ".bashrc.local-llm.bak").exists(), "the copy this run made stayed"
    assert inventory(paths, home=tmp_path, env={}).rc_backups == []


def test_retired_line_alone_counts_as_nothing_left(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    (tmp_path / ".zshrc").write_text(f"{RETIRED_PREFIX}source x\n")
    inv = inventory(paths, home=tmp_path, env={})
    assert inv.rc_with_retired and inv.empty


def test_inventory_finds_the_codex_tables(tmp_path):
    populate(tmp_path)
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    assert inv.codex_config == tmp_path / ".codex" / "config.toml"
    assert "Codex" in inv.summary("integrations")


def test_remove_integrations_strips_only_our_codex_tables(tmp_path):
    populate(tmp_path)
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    lines = remove_integrations(inv, ctx_for(tmp_path))
    text = (tmp_path / ".codex" / "config.toml").read_text()
    assert "local-llm" not in text and 'model = "gpt-5"' in text
    assert any("model_providers.local-llm" in line for line in lines)


def test_an_unparsable_codex_file_is_listed_reported_and_left_byte_identical(tmp_path):
    """It cannot be edited safely, so it is named and the hand-edit lines are printed."""
    populate(tmp_path)
    config = tmp_path / ".codex" / "config.toml"
    broken = '[model_providers.local-llm]\nname = "local-llm"\n[oops\n'
    config.write_bytes(broken.encode())
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    assert inv.codex_config == config, "a file that names us must reach the plan"
    assert "Codex" in inv.summary("integrations") and not inv.empty

    lines = remove_integrations(inv, ctx_for(tmp_path))
    assert any("does not parse" in line and "by hand" in line for line in lines)
    assert config.read_bytes() == broken.encode(), "the file was touched"


def test_a_broken_codex_file_that_is_not_ours_is_never_mentioned(tmp_path):
    populate(tmp_path)
    config = tmp_path / ".codex" / "config.toml"
    config.write_text("[oops\n")
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    assert inv.codex_config is None, "nothing of ours is in it"
    lines = remove_integrations(inv, ctx_for(tmp_path))
    assert not any("does not parse" in line for line in lines)
    assert config.read_text() == "[oops\n"


def test_removal_goes_through_the_registry_not_a_hardcoded_list(tmp_path):
    """A provider added to the registry must be removed without editing uninstall.py."""
    import dataclasses

    from local_llm import harnesses

    populate(tmp_path)
    called: list[str] = []

    def remove_invented(ctx):
        called.append(str(ctx.home))
        return ["removed the invented provider"]

    invented = dataclasses.replace(
        harnesses.find("codex"), key="invented", title="Invented CLI", remove=remove_invented
    )
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    lines = remove_integrations(inv, ctx_for(tmp_path))
    assert not any("invented" in line for line in lines)

    monkey = (*harnesses.REGISTRY, invented)
    import local_llm.uninstall as uninstall_module

    original = uninstall_module.REGISTRY
    uninstall_module.REGISTRY = monkey
    try:
        again = remove_integrations(inv, ctx_for(tmp_path))
    finally:
        uninstall_module.REGISTRY = original
    assert "removed the invented provider" in again
    assert called == [str(tmp_path)], "the entry was handed the context, not the real home"


def test_a_provider_that_raises_never_aborts_the_uninstall(tmp_path):
    import dataclasses

    import local_llm.uninstall as uninstall_module
    from local_llm import harnesses

    populate(tmp_path)

    def explode(ctx):
        raise ValueError("nobody predicted this")

    broken = dataclasses.replace(harnesses.find("codex"), remove=explode)
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    original = uninstall_module.REGISTRY
    uninstall_module.REGISTRY = tuple(
        broken if entry.key == "codex" else entry for entry in harnesses.REGISTRY
    )
    try:
        lines = remove_integrations(inv, ctx_for(tmp_path))
    finally:
        uninstall_module.REGISTRY = original
    assert any("could not remove the OpenAI Codex CLI integration" in line for line in lines)
    assert any("deleted" in line for line in lines), "the rest of the removal still ran"


def test_a_tiny_agent_that_is_not_ours_survives_the_uninstall(tmp_path):
    """configure refuses to replace one; removing it anyway made the two contradict."""
    paths = populate(tmp_path)
    config = tmp_path / ".config" / "opencode" / "opencode.json"
    config.write_text(
        json.dumps({"agent": {"tiny": {"tools": {"bash": False}, "model": "anthropic/haiku"}}})
    )

    lines = remove_integrations(inventory(paths, home=tmp_path, env={}), ctx_for(tmp_path))

    still = json.loads(config.read_text())
    assert still["agent"]["tiny"]["model"] == "anthropic/haiku", "somebody's own agent was deleted"
    assert any("anthropic/haiku" in line and "left alone" in line for line in lines)


def test_the_agent_removal_writes_atomically_and_never_truncates(tmp_path, monkeypatch):
    """write_text emptied the whole opencode config when the write did not finish."""
    from local_llm import integrations

    paths = populate(tmp_path)
    config = tmp_path / ".config" / "opencode" / "opencode.json"
    original = config.read_text()

    def refuse(source, destination):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(integrations.os, "replace", refuse)
    lines = remove_integrations(inventory(paths, home=tmp_path, env={}), ctx_for(tmp_path))

    assert config.read_text() == original, "the config was rewritten in place"
    assert not list(config.parent.glob(".opencode.json.*")), "no temporary file left behind"
    assert any("could not update" in line and str(config) in line for line in lines)


def test_an_opencode_config_that_is_not_utf8_never_stops_the_uninstall(tmp_path):
    """It used to raise out of inventory(), before the plan was even printed."""
    paths = populate(tmp_path)
    config = tmp_path / ".config" / "opencode" / "opencode.json"
    original = '{"agent": {"tiny": {"model": "llamacpp/b"}}, "note": "caf\xe9"}\n'.encode(
        "iso-8859-1"
    )
    config.write_bytes(original)

    inv = inventory(paths, home=tmp_path, env={})

    assert inv.agent_config is None and inv.tiny_model is None

    lines = remove_integrations(inv, ctx_for(tmp_path), restore_retired=True)

    assert config.read_bytes() == original, "the file was touched"
    assert any("not UTF-8" in line and str(config) in line for line in lines)
    assert not (tmp_path / ".zfunc" / "_local-llm").exists(), "the rest of the removal still ran"
    assert (tmp_path / ".bashrc").read_text() == "export B=2\n"
