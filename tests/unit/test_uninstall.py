import json
from pathlib import Path

from local_llm.paths import Paths
from local_llm.preset import Preset
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
    lines = remove_integrations(inv, restore_retired=True)
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
    again = remove_integrations(inventory(paths, home=tmp_path, env={}))
    assert again == []


def test_remove_integrations_keeps_retired_line_by_default_and_skips_jsonc(tmp_path):
    paths = populate(tmp_path, jsonc=True)
    inv = inventory(paths, home=tmp_path, env={})
    assert inv.agent_config and inv.agent_config.name == "opencode.jsonc" and not inv.agent_editable
    lines = remove_integrations(inv)
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
    remove_integrations(inventory(paths, home=tmp_path, env={}), restore_retired=True)
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


def test_retired_line_alone_counts_as_nothing_left(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    (tmp_path / ".zshrc").write_text(f"{RETIRED_PREFIX}source x\n")
    inv = inventory(paths, home=tmp_path, env={})
    assert inv.rc_with_retired and inv.empty
