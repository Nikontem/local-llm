# local-llm — `uninstall` command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `local-llm uninstall` undoes what the tool put on the machine, item by item: models (sections and files), integrations (opencode plugin and agent, shell aliases and completion, optionally restoring the retired zsh line), state (pid, logs, cache, settings), config (`models.ini`), and prints the command that removes the tool itself. Everything is listed before it is deleted; `--dry-run` only prints.

**Architecture:** One new module `uninstall.py` that builds an `Inventory` of what exists and exposes one removal function per category, all pure functions over `Paths`/home so they are unit-tested against a scratch home. The CLI command selects categories by flags or an interactive menu, confirms per category, stops the router first when needed, and prints the tool-removal hint last.

**Spec:** `docs/superpowers/specs/2026-08-26-local-llm-tool-design.md` sections 12 (integrations), 13 (paths), 15 (safety). This plan adds the command to §4 as 4.11 (done in Task 2).

## Global Constraints

- Everything from the milestone plans' Global Constraints (dependencies, lint, tests, commit trailers, working directory).
- Never delete a directory the tool does not own: the config directory is removed only when empty after removing the tool's own files; the state directory is the tool's and is removed whole.
- Never edit a commented (non-strict-JSON) opencode config; print instructions instead.
- Every deletion is printed (path and size) before it happens; nothing outside the inventoried paths is touched.
- The generic `compinit` line typer added to `.zshrc` stays; only the tool's marked block, its completion file, and the bash `source …/local-llm.sh` line are removed.

---

### Task 1: `uninstall` module and command

**Files:**
- Create: `src/local_llm/uninstall.py`
- Modify: `src/local_llm/cli.py` (new command; `_delete_model_file` moves to the module), `tests/unit/conftest.py` (`run` accepts `input`)
- Test: `tests/unit/test_uninstall.py`, `tests/unit/test_cli_uninstall.py`

**Interfaces:**
- Produces in `uninstall.py`: dataclass `ModelEntry(section: str, files: list[Path], size: int)`; dataclass `Inventory(models: list[ModelEntry], plugin: Path | None, agent_config: Path | None, tiny_model: str | None, agent_editable: bool, completion_files: list[Path], rc_with_block: list[Path], rc_with_retired: list[Path], rc_with_bash_source: list[Path], state_dir: Path | None, settings_file: Path | None, preset_files: list[Path], config_dir: Path)` with property `empty: bool` and method `summary(key: str) -> str`; `KEYS = ("models", "integrations", "state", "config")`; `inventory(paths: Paths, home: Path | None = None, env: Mapping[str, str] | None = None) -> Inventory`; `delete_model_file(path: Path) -> None`; `remove_models(paths: Paths, sections: list[str]) -> list[str]`; `remove_integrations(inv: Inventory, *, restore_retired: bool = False) -> list[str]`; `remove_state(inv: Inventory) -> list[str]`; `remove_config(inv: Inventory) -> list[str]`; `tool_uninstall_hint(executable: str | None = None) -> str`.
- Produces in `cli.py`: command `uninstall` with options `--models`, `--integrations`, `--state`, `--config`, `--all`, `--restore-shell-line`, `-y/--yes`, `--dry-run`.
- `conftest.harness.run(*args, input=None)` passes `input` to `CliRunner.invoke`.

- [ ] **Step 1: Write the failing module tests**

`tests/unit/test_uninstall.py`:

```python
import json
import os
from pathlib import Path

from local_llm.estimate import GIB
from local_llm.paths import Paths
from local_llm.preset import Preset
from local_llm.shellrc import MARK_BEGIN, RETIRED_PREFIX, block_text
from local_llm.uninstall import (
    KEYS, delete_model_file, inventory, remove_config, remove_integrations, remove_models,
    remove_state, tool_uninstall_hint,
)


def populate(home: Path, *, jsonc: bool = False) -> Paths:
    """A home with everything the tool can leave behind."""
    paths = Paths.from_env(env={}, home=home)
    paths.config_dir.mkdir(parents=True)
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
    paths.settings_file.write_text('port = 5678\n')
    paths.ensure_state_dirs()
    paths.pid_file.write_text("1\n")
    (paths.log_dir / "llm-router.2026.log").write_text("log")
    paths.hub_cache_file.write_text("{}")
    oc = home / ".config" / "opencode"
    (oc / "plugins").mkdir(parents=True)
    (oc / "plugins" / "local-llm-models.js").write_text("// plugin")
    agent = {"agent": {"tiny": {"model": "llamacpp/b"}, "other": {"mode": "primary"}}, "$schema": "x"}
    if jsonc:
        (oc / "opencode.jsonc").write_text('{\n  // c\n  "agent": {"tiny": {"model": "llamacpp/b"}}\n}\n')
    else:
        (oc / "opencode.json").write_text(json.dumps(agent, indent=2) + "\n")
    (home / ".zfunc").mkdir()
    (home / ".zfunc" / "_local-llm").write_text("#compdef local-llm")
    (home / ".zshrc").write_text(
        "export A=1\n"
        f'{RETIRED_PREFIX}[[ -r "$HOME/.config/local-llm/local_llm.zsh" ]] && source "$HOME/.config/local-llm/local_llm.zsh"\n'
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
    assert any("remove the \"tiny\" agent" in line for line in lines)


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
    assert tool_uninstall_hint("/Users/x/.local/share/uv/tools/local-llm/bin/python") == "uv tool uninstall local-llm"
    assert tool_uninstall_hint("/opt/homebrew/Cellar/local-llm/0.1.0/libexec/bin/python") == "brew uninstall local-llm"
    assert tool_uninstall_hint("/home/x/.local/pipx/venvs/local-llm/bin/python") == "pipx uninstall local-llm"
    assert tool_uninstall_hint("/usr/bin/python3") == "python3 -m pip uninstall local-llm"
```

- [ ] **Step 2: Write the failing CLI tests** (and extend the harness)

In `tests/unit/conftest.py`, change the `run` helper inside the `harness` fixture to:

```python
    def run(*args, input=None):
        return runner.invoke(cli.app, list(args), input=input)
```

`tests/unit/test_cli_uninstall.py`:

```python
from pathlib import Path

from local_llm.preset import Preset

from .test_uninstall import populate


def test_dry_run_prints_the_plan_and_changes_nothing(harness):
    h = harness
    populate_paths = populate(h.tmp)  # re-populates the harness home (overwrites the fixture preset)
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_uninstall.py tests/unit/test_cli_uninstall.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.uninstall'`

- [ ] **Step 4: Write `src/local_llm/uninstall.py`**

```python
"""Undo what the tool put on this machine, item by item, listing everything first."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .estimate import human_gb
from .integrations.opencode import CONFIG_CANDIDATES, current_tiny_model, is_strict_json, opencode_paths
from .paths import Paths
from .preset import Preset, PresetError
from .shellrc import MARK_BEGIN, RETIRED_PREFIX, rc_file, remove_block

KEYS = ("models", "integrations", "state", "config")
_COMPLETION_FILES = {
    "zsh": Path(".zfunc") / "_local-llm",
    "bash": Path(".bash_completions") / "local-llm.sh",
    "fish": Path(".config") / "fish" / "completions" / "local-llm.fish",
}
_BASH_SOURCE = re.compile(r"^\s*(source|\.)\s+.*\.bash_completions/local-llm\.sh\s*$")


@dataclass
class ModelEntry:
    section: str
    files: list[Path]
    size: int


@dataclass
class Inventory:
    models: list[ModelEntry] = field(default_factory=list)
    plugin: Path | None = None
    agent_config: Path | None = None
    tiny_model: str | None = None
    agent_editable: bool = False
    completion_files: list[Path] = field(default_factory=list)
    rc_with_block: list[Path] = field(default_factory=list)
    rc_with_retired: list[Path] = field(default_factory=list)
    rc_with_bash_source: list[Path] = field(default_factory=list)
    state_dir: Path | None = None
    settings_file: Path | None = None
    preset_files: list[Path] = field(default_factory=list)
    config_dir: Path = Path(".")

    @property
    def empty(self) -> bool:
        return not any(self.summary(key) != "nothing" for key in KEYS)

    def summary(self, key: str) -> str:
        if key == "models":
            if not self.models:
                return "nothing"
            total = sum(m.size for m in self.models)
            return f"{len(self.models)} model(s), {human_gb(total)} on disk"
        if key == "integrations":
            parts = []
            if self.plugin:
                parts.append("opencode plugin")
            if self.tiny_model:
                parts.append("opencode tiny agent")
            if self.rc_with_block:
                parts.append("shell aliases")
            if self.completion_files or self.rc_with_bash_source:
                parts.append("completion")
            if self.rc_with_retired:
                parts.append("a retired zsh line")
            return ", ".join(parts) or "nothing"
        if key == "state":
            parts = []
            if self.state_dir:
                parts.append("state and logs")
            if self.settings_file:
                parts.append("settings.toml")
            return ", ".join(parts) or "nothing"
        if key == "config":
            return ", ".join(p.name for p in self.preset_files) or "nothing"
        raise KeyError(key)


def _dir_size(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def inventory(paths: Paths, home: Path | None = None, env: Mapping[str, str] | None = None) -> Inventory:
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or Path.home())
    inv = Inventory(config_dir=paths.config_dir)

    if paths.preset.is_file():
        try:
            preset = Preset.load(paths.preset)
            for section in preset.sections():
                files = [
                    Path(value) for key in ("model", "mmproj")
                    if (value := preset.get(section, key, fallback_to_star=False))
                ]
                existing = [f for f in files if f.exists() or f.is_symlink()]
                size = sum(f.stat().st_size for f in existing if f.exists())
                inv.models.append(ModelEntry(section, existing, size))
        except PresetError:
            pass
        inv.preset_files.append(paths.preset)
    backup = paths.preset.with_name(paths.preset.name + ".bak")
    if backup.is_file():
        inv.preset_files.append(backup)

    oc = opencode_paths(home=home, env=env)
    if oc.plugin.is_file():
        inv.plugin = oc.plugin
    for name in CONFIG_CANDIDATES:
        candidate = oc.config_dir / name
        if candidate.is_file():
            text = candidate.read_text()
            model = current_tiny_model(text)
            if model:
                inv.agent_config, inv.tiny_model = candidate, model
                inv.agent_editable = is_strict_json(text)
            break

    for shell, relative in _COMPLETION_FILES.items():
        completion = home / relative
        if completion.is_file():
            inv.completion_files.append(completion)
        rc = rc_file(shell, home)
        if rc.is_file():
            text = rc.read_text()
            if MARK_BEGIN in text and rc not in inv.rc_with_block:
                inv.rc_with_block.append(rc)
            if RETIRED_PREFIX in text and rc not in inv.rc_with_retired:
                inv.rc_with_retired.append(rc)
            if any(_BASH_SOURCE.match(line) for line in text.splitlines()) and rc not in inv.rc_with_bash_source:
                inv.rc_with_bash_source.append(rc)

    if paths.state_dir.is_dir():
        inv.state_dir = paths.state_dir
    if paths.settings_file.is_file():
        inv.settings_file = paths.settings_file
    return inv


def delete_model_file(path: Path) -> None:
    """Delete a model file; a Hugging Face cache entry is a symlink, so delete its blob too."""
    target = path.resolve() if path.is_symlink() else None
    path.unlink(missing_ok=True)
    if target is not None and target.exists():
        target.unlink()


def remove_models(paths: Paths, sections: list[str]) -> list[str]:
    lines: list[str] = []
    try:
        preset = Preset.load(paths.preset)
    except PresetError as error:
        return [str(error)]
    for section in sections:
        if not preset.has_section(section):
            lines.append(f"no section [{section}]")
            continue
        for key in ("model", "mmproj"):
            value = preset.get(section, key, fallback_to_star=False)
            if not value:
                continue
            file = Path(value)
            if file.exists() or file.is_symlink():
                size = file.stat().st_size if file.exists() else 0
                delete_model_file(file)
                lines.append(f"deleted {file} ({human_gb(size)})")
        preset.remove_section(section)
        lines.append(f"removed [{section}] from {paths.preset}")
    preset.save(paths.preset)
    return lines


def _remove_agent(config: Path) -> str:
    data = json.loads(config.read_text() or "{}")
    agents = data.get("agent")
    if isinstance(agents, dict) and "tiny" in agents:
        del agents["tiny"]
        if not agents:
            del data["agent"]
        config.write_text(json.dumps(data, indent=2) + "\n")
        return f"removed the tiny agent from {config}"
    return f"no tiny agent in {config}"


def remove_integrations(inv: Inventory, *, restore_retired: bool = False) -> list[str]:
    lines: list[str] = []
    if inv.plugin and inv.plugin.exists():
        inv.plugin.unlink()
        lines.append(f"deleted {inv.plugin}")
    if inv.agent_config and inv.tiny_model:
        if inv.agent_editable:
            lines.append(_remove_agent(inv.agent_config))
        else:
            lines.append(
                f'{inv.agent_config} has comments, so it is not rewritten: remove the "tiny" agent'
                " entry from it by hand"
            )
    for rc in {*inv.rc_with_block, *inv.rc_with_bash_source, *(inv.rc_with_retired if restore_retired else [])}:
        text = rc.read_text()
        updated = remove_block(text)
        updated = "".join(
            line for line in updated.splitlines(keepends=True) if not _BASH_SOURCE.match(line)
        )
        if restore_retired:
            updated = updated.replace(RETIRED_PREFIX, "")
        if updated != text:
            rc.write_text(updated)
            what = []
            if rc in inv.rc_with_block:
                what.append("aliases removed")
            if rc in inv.rc_with_bash_source:
                what.append("completion line removed")
            if restore_retired and rc in inv.rc_with_retired:
                what.append("retired zsh line restored")
            lines.append(f"{rc}: {', '.join(what)}")
    for completion in inv.completion_files:
        if completion.exists():
            completion.unlink()
            lines.append(f"deleted {completion}")
    return lines


def remove_state(inv: Inventory) -> list[str]:
    lines: list[str] = []
    if inv.state_dir and inv.state_dir.exists():
        size = _dir_size(inv.state_dir)
        shutil.rmtree(inv.state_dir)
        lines.append(f"deleted {inv.state_dir} ({human_gb(size)})")
    if inv.settings_file and inv.settings_file.exists():
        inv.settings_file.unlink()
        lines.append(f"deleted {inv.settings_file}")
    return lines


def remove_config(inv: Inventory) -> list[str]:
    lines: list[str] = []
    for file in inv.preset_files:
        if file.exists():
            file.unlink()
            lines.append(f"deleted {file}")
    directory = inv.config_dir
    if directory.is_dir():
        leftovers = [p.name for p in directory.iterdir()]
        if leftovers:
            lines.append(f"{directory} left in place: it still holds {', '.join(sorted(leftovers)[:5])}")
        else:
            directory.rmdir()
            lines.append(f"removed the empty {directory}")
    return lines


def tool_uninstall_hint(executable: str | None = None) -> str:
    where = executable or sys.executable
    if "/uv/tools/" in where:
        return "uv tool uninstall local-llm"
    if "/Cellar/" in where:
        return "brew uninstall local-llm"
    if "/pipx/" in where:
        return "pipx uninstall local-llm"
    return "python3 -m pip uninstall local-llm"
```

- [ ] **Step 5: Add the command to `src/local_llm/cli.py`**

Replace the `_delete_model_file` function and its use in `remove` with an import: `from .uninstall import KEYS, delete_model_file, inventory, remove_config, remove_integrations, remove_models, remove_state, tool_uninstall_hint` (delete the local `_delete_model_file`, call `delete_model_file(file)` in `remove`). Append after the `setup` command:

```python
# ---------------------------------------------------------------- uninstall


def _print_plan(inv, chosen: set[str]) -> None:
    for key in KEYS:
        if key not in chosen:
            continue
        out.print(f"{key}: {inv.summary(key)}")
        if key == "models":
            for entry in inv.models:
                out.print(f"  [{entry.section}]  {human_gb(entry.size)}")
                for file in entry.files:
                    out.print(f"    {file}")
        elif key == "integrations":
            for path in [inv.plugin, inv.agent_config, *inv.completion_files, *inv.rc_with_block, *inv.rc_with_bash_source]:
                if path:
                    out.print(f"    {path}")
        elif key == "state":
            for path in [inv.state_dir, inv.settings_file]:
                if path:
                    out.print(f"    {path}")
        elif key == "config":
            for path in inv.preset_files:
                out.print(f"    {path}")


def _pick_models(inv) -> list[str]:
    out.print("  models in models.ini:")
    for index, entry in enumerate(inv.models, start=1):
        out.print(f"   {index:>2}. {entry.section}  {human_gb(entry.size)}")
    answer = typer.prompt("  Numbers to remove (e.g. 1 3), a for all", default="a").strip().lower()
    if answer in ("a", "all"):
        return [entry.section for entry in inv.models]
    try:
        return [inv.models[int(token) - 1].section for token in answer.split()]
    except (ValueError, IndexError):
        fail(f"Pick numbers between 1 and {len(inv.models)}")


@app.command()
def uninstall(
    models: bool = typer.Option(False, "--models", help="Remove every model section and its files."),
    integrations: bool = typer.Option(False, "--integrations", help="opencode plugin and agent, shell aliases and completion."),
    state_: bool = typer.Option(False, "--state", help="pid, logs, Hub cache, settings.toml."),
    config: bool = typer.Option(False, "--config", help="models.ini and its backup."),
    all_: bool = typer.Option(False, "--all", help="Everything above."),
    restore_shell_line: bool = typer.Option(False, "--restore-shell-line", help="Put back the retired `source local_llm.zsh` line."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Do not ask."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only print what would be removed."),
) -> None:
    """Undo what local-llm put on this machine, item by item. Lists everything before deleting."""
    st = state()
    inv = inventory(st.paths)
    chosen: set[str] = set(KEYS) if all_ else set()
    for flag, key in ((models, "models"), (integrations, "integrations"), (state_, "state"), (config, "config")):
        if flag:
            chosen.add(key)
    sections = [entry.section for entry in inv.models]
    if not chosen:
        if dry_run:
            chosen = set(KEYS)
        elif yes:
            fail("Choose what to remove: --models, --integrations, --state, --config, or --all (see --dry-run first).")
        else:
            if inv.empty:
                out.print("Nothing of local-llm's is left on this machine.")
                out.print(f"  to remove the tool itself:  {tool_uninstall_hint()}")
                return
            for index, key in enumerate(KEYS, start=1):
                out.print(f"  {index}. {key:<13} {inv.summary(key)}")
            out.print("  5. everything")
            answer = typer.prompt("Numbers to remove (e.g. 1 3), q to quit", default="q").strip().lower()
            if answer in ("q", "quit", ""):
                raise typer.Exit()
            if answer in ("5", "a", "all"):
                chosen = set(KEYS)
            else:
                try:
                    chosen = {KEYS[int(token) - 1] for token in answer.split()}
                except (ValueError, IndexError):
                    fail("Pick numbers between 1 and 5")
            if "models" in chosen and inv.models:
                sections = _pick_models(inv)
    if inv.empty:
        out.print("Nothing of local-llm's is left on this machine.")
        out.print(f"  to remove the tool itself:  {tool_uninstall_hint()}")
        return
    _print_plan(inv, chosen)
    if dry_run:
        out.print("Dry run: nothing was removed.")
        return
    if not yes and not typer.confirm("Remove the items above?", default=False):
        raise typer.Exit(1)
    if chosen & {"models", "state", "config"}:
        router = st.router()
        if router.pid() is not None:
            _stop(router)
    if "models" in chosen and sections:
        for line in remove_models(st.paths, sections):
            out.print(f"  {line}")
    if "integrations" in chosen:
        for line in remove_integrations(inv, restore_retired=restore_shell_line):
            out.print(f"  {line}")
    if "state" in chosen:
        for line in remove_state(inv):
            out.print(f"  {line}")
    if "config" in chosen:
        for line in remove_config(inv):
            out.print(f"  {line}")
    out.print()
    out.print(f"To remove the tool itself:  {tool_uninstall_hint()}")
```

Note for the interactive test (`input="1\n2\ny\n"`): the menu prompt receives `1` (models), the model picker receives `2` (section `b`), the confirmation receives `y`.

- [ ] **Step 6: Run the whole suite and the linter**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass (about 225 tests), ruff clean. Wrap any line over 100 columns.

- [ ] **Step 7: Commit**

```bash
git add src/local_llm/uninstall.py src/local_llm/cli.py tests/unit/conftest.py tests/unit/test_uninstall.py tests/unit/test_cli_uninstall.py
git commit -m "feat: uninstall command that undoes models, integrations, state and config item by item"
```

---

### Task 2: Documentation

**Files:**
- Modify: `README.md` (new section "Uninstall" before "Files"), `docs/superpowers/specs/2026-08-26-local-llm-tool-design.md` (new §4.11 after §4.10), `CHANGELOG.md` (an "Unreleased" entry above 0.1.0)

- [ ] **Step 1: README "Uninstall" section**

```markdown
## Uninstall

`local-llm uninstall` undoes what the tool put on the machine, one category at
a time, and lists every path before deleting it:

```
$ local-llm uninstall
  1. models        5 model(s), 73.0 GB on disk
  2. integrations  opencode plugin, opencode tiny agent, shell aliases, completion, a retired zsh line
  3. state         state and logs, settings.toml
  4. config        models.ini, models.ini.bak
  5. everything
Numbers to remove (e.g. 1 3), q to quit [q]:
```

Choosing models lets you pick which ones; their sections leave `models.ini`
and their files (cache symlink and blob) are deleted. `--models`,
`--integrations`, `--state`, `--config` and `--all` select without the menu,
`--yes` skips the questions, `--dry-run` only prints the plan.
`--restore-shell-line` puts back a `source …/local_llm.zsh` line that
`completion install` had commented out. The config directory is removed only
if nothing else is left in it. The tool cannot delete itself while running,
so the last line prints the command for that (`uv tool uninstall local-llm`
for the standard install).
```

- [ ] **Step 2: Spec §4.11**

```markdown
### 4.11 `local-llm uninstall [--models] [--integrations] [--state] [--config] [--all] [--restore-shell-line] [--yes] [--dry-run]`

Inventories what the tool has put on the machine and removes the chosen
categories, listing every path first: models (sections and their files,
including a cache entry's blob), integrations (opencode plugin and `tiny`
agent — only from a strict-JSON config —, the shell aliases block, the
completion file and bash's source line; optionally restoring a retired
`source local_llm.zsh` line), state (the state directory and
`settings.toml`), config (`models.ini` and its backup; the config directory
only when empty afterwards). Without flags it shows a numbered menu; with
`--yes` and no selection it refuses. The router is stopped first when models,
state or config are removed. It ends by printing the command that removes
the tool itself, chosen from where the interpreter lives.
```

- [ ] **Step 3: CHANGELOG**

Add above `## 0.1.0`:

```markdown
## Unreleased

- `uninstall`: undo models, integrations, state and config item by item,
  listing everything first; `--dry-run`, `--all`, `--restore-shell-line`.
- `install.sh` is uv-only; Homebrew is used for llama.cpp and `hf` when present.
- Downloads use plain HTTP and give up with a resume hint after three minutes
  without progress.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-26-local-llm-tool-design.md CHANGELOG.md
git commit -m "docs: uninstall command"
```
