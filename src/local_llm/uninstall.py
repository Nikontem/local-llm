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
from .harnesses import PROVIDER, REGISTRY
from .integrations import HarnessContext, atomic_write, backup_path
from .integrations.codex import codex_paths, has_tables, unparsable_but_ours
from .integrations.opencode import (
    CONFIG_CANDIDATES,
    current_tiny_model,
    is_strict_json,
    opencode_paths,
)
from .paths import Paths
from .preset import Preset, PresetError
from .shellrc import (
    RETIRED_PREFIX,
    encoding_warning,
    marker_problem,
    marker_warning,
    mentions_block,
    rc_file,
    remove_block,
)

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
    agent_backup: Path | None = None
    completion_files: list[Path] = field(default_factory=list)
    rc_with_block: list[Path] = field(default_factory=list)
    rc_with_retired: list[Path] = field(default_factory=list)
    rc_with_bash_source: list[Path] = field(default_factory=list)
    rc_not_utf8: list[Path] = field(default_factory=list)
    rc_backups: list[Path] = field(default_factory=list)
    codex_config: Path | None = None
    codex_backup: Path | None = None
    state_dir: Path | None = None
    settings_file: Path | None = None
    preset_files: list[Path] = field(default_factory=list)
    config_dir: Path = Path(".")

    @property
    def empty(self) -> bool:
        """Nothing left to remove. A retired zsh line does not count: it is only ever restored."""
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
            if self.codex_config:
                parts.append("Codex provider and profile")
            if self.agent_backup or self.rc_backups:
                parts.append("the backup copies we made")
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


def _model_files(preset: Preset, section: str) -> list[Path]:
    return [
        Path(value).expanduser()
        for key in ("model", "mmproj")
        if (value := preset.get(section, key, fallback_to_star=False))
    ]


def _backup_of(preset_path: Path) -> Path:
    return preset_path.with_name(preset_path.name + ".bak")


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
                existing = [
                    f for f in _model_files(preset, section) if f.exists() or f.is_symlink()
                ]
                size = sum(f.stat().st_size for f in existing if f.is_file())
                inv.models.append(ModelEntry(section, existing, size))
        except PresetError:
            pass
        inv.preset_files.append(paths.preset)
    backup = _backup_of(paths.preset)
    if backup.is_file() or backup.is_symlink():
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
            # Ours by its name alone, whether or not a tiny agent is still in the file.
            # A config commonly holds the person's own API keys, so a copy of it left
            # beside the original is a second copy of their secrets.
            if backup_path(candidate).is_file():
                inv.agent_backup = backup_path(candidate)
            break

    cx = codex_paths(home=home, env=env)
    # A config that will not parse never reaches has_tables, which answers False on any
    # read failure. One that names us is still ours to report: nothing is touched, and
    # removal prints the lines to delete by hand.
    if has_tables(cx) or unparsable_but_ours(cx):
        inv.codex_config = cx.config_file
        if cx.backup.is_file():
            inv.codex_backup = cx.backup

    for shell, relative in _COMPLETION_FILES.items():
        completion = home / relative
        if completion.is_file():
            inv.completion_files.append(completion)
        rc = rc_file(shell, home)
        copy = backup_path(rc)
        if copy.is_file() and copy not in inv.rc_backups:
            inv.rc_backups.append(copy)
        if rc.is_file():
            try:
                content = rc.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Nothing here can be read, so nothing here can be edited. Reported
                # rather than skipped in silence, and never put in a list something
                # rewrites: this used to raise before the plan was even printed.
                if rc not in inv.rc_not_utf8:
                    inv.rc_not_utf8.append(rc)
                continue
            lines = content.splitlines()
            # Half a block counts here: it cannot be removed, but the person has to be
            # told it is there rather than have the file passed over in silence.
            if mentions_block(content) and rc not in inv.rc_with_block:
                inv.rc_with_block.append(rc)
            if (
                any(line.startswith(RETIRED_PREFIX) for line in lines)
                and rc not in inv.rc_with_retired
            ):
                inv.rc_with_retired.append(rc)
            if (
                any(_BASH_SOURCE.match(line) for line in lines)
                and rc not in inv.rc_with_bash_source
            ):
                inv.rc_with_bash_source.append(rc)

    if paths.state_dir.is_dir():
        inv.state_dir = paths.state_dir
    if paths.settings_file.is_file():
        inv.settings_file = paths.settings_file
    return inv


def delete_model_file(path: Path) -> None:
    """Delete a model file; a Hugging Face cache entry is a symlink, so delete its blob too.

    Only a regular file is ever deleted through a link; a link to a directory
    is removed as a link and the directory stays.
    """
    target = path.resolve() if path.is_symlink() else None
    path.unlink(missing_ok=True)
    if target is not None and target.is_file():
        target.unlink()


def remove_models(paths: Paths, sections: list[str]) -> list[str]:
    """Delete each section's files and drop the section; a file that cannot go keeps its section."""
    lines: list[str] = []
    try:
        preset = Preset.load(paths.preset)
    except PresetError as error:
        return [str(error)]
    for section in sections:
        if not preset.has_section(section):
            lines.append(f"no section [{section}]")
            continue
        failed = False
        for file in _model_files(preset, section):
            if not (file.exists() or file.is_symlink()):
                continue
            size = file.stat().st_size if file.is_file() else 0
            try:
                delete_model_file(file)
            except OSError as error:
                failed = True
                lines.append(f"could not delete {file}: {error}")
                continue
            lines.append(f"deleted {file} ({human_gb(size)})")
        if failed:
            lines.append(f"kept [{section}] in {paths.preset} because a file could not be deleted")
            continue
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


def _restore_retired(text: str) -> str:
    """Strip the retirement prefix from the lines that carry it, and only from those."""
    restored = []
    for line in text.splitlines(keepends=True):
        if line.startswith(RETIRED_PREFIX):
            line = line[len(RETIRED_PREFIX) :]
        restored.append(line)
    return "".join(restored)


def remove_integrations(
    inv: Inventory, context: HarnessContext, *, restore_retired: bool = False
) -> list[str]:
    """Undo the integrations in the inventory, plus whatever the registry's providers wrote.

    The context is what each registry entry needs to find its own files, and is
    required rather than defaulted so nothing can quietly read the real home.
    """
    lines: list[str] = []
    if inv.plugin and inv.plugin.exists():
        try:
            inv.plugin.unlink()
            lines.append(f"deleted {inv.plugin}")
        except OSError as error:
            lines.append(f"could not delete {inv.plugin}: {error}")
    if inv.agent_config and inv.tiny_model:
        if inv.agent_editable:
            lines.append(_remove_agent(inv.agent_config))
        else:
            lines.append(
                f'{inv.agent_config} has comments, so it is not rewritten: remove the "tiny" agent'
                " entry from it by hand"
            )
    # Every provider takes its own integration back out, so adding one to the registry
    # needs no edit here. Each entry does nothing when the file holds nothing of ours,
    # which is the same reading that put it in the plan a moment ago.
    for harness in REGISTRY:
        if harness.kind != PROVIDER or harness.remove is None:
            continue
        try:
            lines.extend(harness.remove(context))
        except Exception as error:  # a failed removal never aborts the run
            lines.append(f"could not remove the {harness.title} integration: {error}")
    # Ours by name: the ones found a moment ago, and the ones the rewrites below make.
    backups: set[Path] = {*inv.rc_backups}
    if inv.agent_backup:
        backups.add(inv.agent_backup)
    rc_files = {*inv.rc_with_block, *inv.rc_with_bash_source}
    if restore_retired:
        rc_files.update(inv.rc_with_retired)
    for rc in inv.rc_not_utf8:
        lines.append(encoding_warning(rc))
    for rc in sorted(rc_files):
        try:
            text = rc.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # re-saved between the plan and now
            lines.append(encoding_warning(rc))
            continue
        problem = marker_problem(text)
        if problem is not None:
            # Markers that do not pair up: the end marker our removal would delete up to
            # belongs to somebody else's lines, so this file is not touched at all.
            lines.append(marker_warning(rc, problem))
            continue
        updated = remove_block(text)
        updated = "".join(
            line for line in updated.splitlines(keepends=True) if not _BASH_SOURCE.match(line)
        )
        if restore_retired:
            updated = _restore_retired(updated)
        if updated != text:
            try:
                atomic_write(rc, updated)
            except OSError as error:  # a failed removal never aborts the run
                lines.append(f"could not update {rc}: {error}")
                continue
            backups.add(backup_path(rc))  # the copy that rewrite just made
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
            try:
                completion.unlink()
                lines.append(f"deleted {completion}")
            except OSError as error:
                lines.append(f"could not delete {completion}: {error}")
    # Last, because the rewrites above are what make most of them. A .local-llm.bak is
    # a file this tool wrote and nothing else ever will, so an uninstall that leaves one
    # behind leaves a second copy of a configuration - opencode's holds API keys.
    for copy in sorted(backups):
        if not (copy.exists() or copy.is_symlink()):
            continue
        try:
            copy.unlink()
            lines.append(f"deleted {copy}")
        except OSError as error:  # a failed removal never aborts the run
            lines.append(f"could not delete {copy}: {error}")
    return lines


def remove_state(inv: Inventory) -> list[str]:
    lines: list[str] = []
    if inv.state_dir and (inv.state_dir.exists() or inv.state_dir.is_symlink()):
        size = _dir_size(inv.state_dir)
        try:
            if inv.state_dir.is_symlink():
                inv.state_dir.unlink()
            else:
                shutil.rmtree(inv.state_dir)
            lines.append(f"deleted {inv.state_dir} ({human_gb(size)})")
        except OSError as error:
            lines.append(f"could not delete {inv.state_dir}: {error}")
    if inv.settings_file and inv.settings_file.exists():
        try:
            inv.settings_file.unlink()
            lines.append(f"deleted {inv.settings_file}")
        except OSError as error:
            lines.append(f"could not delete {inv.settings_file}: {error}")
    return lines


def remove_config(inv: Inventory) -> list[str]:
    """Delete models.ini and its backup, then the config directory if nothing else is in it.

    The backup path is resolved here, not taken from the inventory: removing
    models a moment earlier re-creates it through Preset.save.
    """
    lines: list[str] = []
    candidates: list[Path] = list(inv.preset_files)
    for file in inv.preset_files:
        if file.name.endswith(".ini"):
            candidates.append(_backup_of(file))
    seen: set[Path] = set()
    for file in candidates:
        if file in seen:
            continue
        seen.add(file)
        if file.exists() or file.is_symlink():
            try:
                file.unlink()
                lines.append(f"deleted {file}")
            except OSError as error:
                lines.append(f"could not delete {file}: {error}")
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
