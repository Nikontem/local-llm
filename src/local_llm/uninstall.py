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
from .integrations.opencode import (
    CONFIG_CANDIDATES,
    current_tiny_model,
    is_strict_json,
    opencode_paths,
)
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


def inventory(
    paths: Paths, home: Path | None = None, env: Mapping[str, str] | None = None
) -> Inventory:
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or Path.home())
    inv = Inventory(config_dir=paths.config_dir)

    if paths.preset.is_file():
        try:
            preset = Preset.load(paths.preset)
            for section in preset.sections():
                files = [
                    Path(value)
                    for key in ("model", "mmproj")
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
            if (
                any(_BASH_SOURCE.match(line) for line in text.splitlines())
                and rc not in inv.rc_with_bash_source
            ):
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
    for rc in {
        *inv.rc_with_block,
        *inv.rc_with_bash_source,
        *(inv.rc_with_retired if restore_retired else []),
    }:
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
            lines.append(
                f"{directory} left in place: it still holds {', '.join(sorted(leftovers)[:5])}"
            )
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
