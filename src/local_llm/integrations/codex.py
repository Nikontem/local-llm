"""Codex CLI: a named model provider and a profile inside Codex's own config.toml."""

from __future__ import annotations

import difflib
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError

from ..preset import smallest_model
from ..settings import Settings
from . import HarnessContext

PROVIDER_ID = "local-llm"
WIRE_API = "responses"  # Codex dropped the older "chat" wire format in February 2026
KEY_VARIABLE = "LOCAL_LLM_API_KEY"
PARENTS = ("model_providers", "profiles")
BACKUP_SUFFIX = ".local-llm.bak"
EXPERIMENTAL = (
    "experimental: needs a recent llama.cpp build; tool calls may fail on older ones"
)


@dataclass(frozen=True)
class CodexPaths:
    config_dir: Path
    config_file: Path

    @property
    def backup(self) -> Path:
        """Our own name, not the plain .bak somebody may have made by hand: uninstall
        deletes this file, and it must never be able to delete a person's own copy."""
        return self.config_file.with_name(self.config_file.name + BACKUP_SUFFIX)


def codex_paths(
    home: Path | None = None, env: Mapping[str, str] | None = None
) -> CodexPaths:
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or Path.home())
    directory = Path(env.get("CODEX_HOME") or home / ".codex")
    return CodexPaths(config_dir=directory, config_file=directory / "config.toml")


def paths_for(config_file: Path) -> CodexPaths:
    """The same record built from a file we already found, so nothing is re-derived."""
    return CodexPaths(config_dir=config_file.parent, config_file=config_file)


def provider_table(settings: Settings) -> dict[str, str]:
    """Codex needs base_url to carry the /v1 suffix, which openai_base_url already has."""
    table = {
        "name": PROVIDER_ID,
        "base_url": settings.openai_base_url,
        "wire_api": WIRE_API,
    }
    if settings.api_key:
        table["env_key"] = KEY_VARIABLE
    return table


def profile_table(model: str) -> dict[str, str]:
    return {"model": model, "model_provider": PROVIDER_ID}


def chosen_model(ctx: HarnessContext) -> str | None:
    """The configured default when it exists, else the smallest model, else nothing."""
    preset = ctx.preset
    if preset is None:
        return None
    default = ctx.settings.default_model
    if default and preset.has_section(default):
        return default
    return smallest_model(preset)


def _unwrap(value: object) -> dict | None:
    if value is None:
        return None
    unwrap = getattr(value, "unwrap", None)
    return unwrap() if unwrap is not None else dict(value)  # type: ignore[arg-type]


#: Everything that stops us reading config.toml: bad TOML (ParseError), bytes that are
#: not UTF-8 (UnicodeDecodeError), or a file we are not allowed to open (OSError).
UNREADABLE = (ParseError, OSError, UnicodeDecodeError)


def _document(paths: CodexPaths):
    """The parsed file, or None when it does not exist. Raises anything in UNREADABLE."""
    if not paths.config_file.is_file():
        return None
    return tomlkit.parse(paths.config_file.read_text())


def _ours(doc, parent_name: str) -> dict | None:
    parent = doc.get(parent_name)
    return None if parent is None else _unwrap(parent.get(PROVIDER_ID))


def parse_problem(paths: CodexPaths) -> str | None:
    """None when the file can be read (or is absent), else why it cannot be."""
    try:
        _document(paths)
    except ParseError as error:
        return f"line {error.line}, column {error.col}: {error}"
    except (OSError, UnicodeDecodeError) as error:
        return f"cannot be read: {error}"
    return None


def status(paths: CodexPaths, provider: dict, profile: dict | None) -> str:
    try:
        doc = _document(paths)
    except UNREADABLE:
        return "unreadable"
    if doc is None or _ours(doc, "model_providers") is None:
        return "missing"
    same_provider = _ours(doc, "model_providers") == provider
    same_profile = profile is None or _ours(doc, "profiles") == profile
    return "same" if same_provider and same_profile else "different"


def has_tables(paths: CodexPaths) -> bool:
    try:
        doc = _document(paths)
    except UNREADABLE:
        return False
    return doc is not None and any(_ours(doc, name) is not None for name in PARENTS)


def configured_base_url(paths: CodexPaths) -> str | None:
    try:
        doc = _document(paths)
    except UNREADABLE:
        return None
    ours = None if doc is None else _ours(doc, "model_providers")
    return None if ours is None else ours.get("base_url")


def _set(doc, parent_name: str, values: dict) -> None:
    parent = doc.get(parent_name)
    if parent is None:
        parent = tomlkit.table(is_super_table=True)
        doc[parent_name] = parent
    child = tomlkit.table()
    for name, value in values.items():
        child[name] = value
    parent[PROVIDER_ID] = child


def render(provider: dict, profile: dict | None) -> str:
    """Just the tables we own, for printing when the file must not be written."""
    doc = tomlkit.document()
    _set(doc, "model_providers", provider)
    if profile is not None:
        _set(doc, "profiles", profile)
    return tomlkit.dumps(doc)


def current_render(paths: CodexPaths) -> str:
    try:
        doc = _document(paths)
    except UNREADABLE:
        return ""
    provider = None if doc is None else _ours(doc, "model_providers")
    if provider is None:
        return ""
    return render(provider, _ours(doc, "profiles"))


def _atomic_write(paths: CodexPaths, text: str) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    # People commonly symlink ~/.codex/config.toml into a dotfiles repository. Writing
    # to the link path would replace the link with a regular file and leave the dotfiles
    # copy holding the old content, so the write goes through to what the link points at.
    # The temporary file has to sit beside that destination for os.replace to work
    # across a filesystem boundary, and the backup stays where the link is.
    target = paths.config_file.resolve() if paths.config_file.is_symlink() else paths.config_file
    if paths.config_file.is_file():
        if paths.backup.is_symlink():
            paths.backup.unlink()  # never write through a link somebody put there
        shutil.copy2(paths.config_file, paths.backup)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".config.toml.")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.replace(temporary, target)
    finally:
        # A failed write must not litter ~/.codex with hidden half-files.
        if os.path.exists(temporary):
            os.unlink(temporary)


def write(paths: CodexPaths, provider: dict, profile: dict | None) -> None:
    doc = _document(paths)
    if doc is None:
        doc = tomlkit.document()
    _set(doc, "model_providers", provider)
    if profile is not None:
        _set(doc, "profiles", profile)
    _atomic_write(paths, tomlkit.dumps(doc))


def drop(paths: CodexPaths) -> list[str]:
    """Delete our tables, leaving everything else alone. Names what it removed."""
    doc = _document(paths)
    if doc is None:
        return []
    removed: list[str] = []
    for parent_name in PARENTS:
        parent = doc.get(parent_name)
        if parent is None or PROVIDER_ID not in parent:
            continue
        del parent[PROVIDER_ID]
        removed.append(f"[{parent_name}.{PROVIDER_ID}]")
        # An emptied table goes too, unless the person left a comment inside it.
        if not len(parent) and not tomlkit.dumps(parent).strip():
            del doc[parent_name]
    if doc.get("profile") == PROVIDER_ID:
        del doc["profile"]
        removed.append(f'profile = "{PROVIDER_ID}"')
    if removed:
        _atomic_write(paths, tomlkit.dumps(doc))
    return removed


def configure(ctx: HarnessContext) -> list[str]:
    """Write our provider and profile into Codex's config, asking before replacing."""
    paths = codex_paths(home=ctx.home, env=ctx.env)
    provider = provider_table(ctx.settings)
    model = chosen_model(ctx)
    profile = profile_table(model) if model else None
    state = status(paths, provider, profile)

    if state == "unreadable":
        return [
            f"{paths.config_file} does not parse ({parse_problem(paths)}),"
            " so it is not rewritten. Paste this into it:",
            render(provider, profile).rstrip(),
        ]

    lines: list[str] = []
    if state == "same":
        lines.append(f"already configured in {paths.config_file}")
    else:
        if state == "different":
            ctx.say(
                "\n".join(
                    difflib.unified_diff(
                        current_render(paths).splitlines(),
                        render(provider, profile).splitlines(),
                        fromfile=str(paths.config_file),
                        tofile="local-llm",
                        lineterm="",
                    )
                )
            )
            if not ctx.ask(f"Replace the local-llm tables in {paths.config_file}?", True):
                return ["left as it is"]
        try:
            write(paths, provider, profile)
        except UNREADABLE as error:
            # A read-only home, a full disk or somebody else's file: say so, never raise.
            return [f"could not update {paths.config_file}: {error}"]
        lines.append(f"provider and profile local-llm written to {paths.config_file}")

    if profile is None:
        lines.append("no model in models.ini yet, so no profile was written")
    else:
        lines.append(f"run it with:  codex --profile {PROVIDER_ID}")
    lines.append(EXPERIMENTAL)
    return lines


def harness_status(ctx: HarnessContext) -> str:
    paths = codex_paths(home=ctx.home, env=ctx.env)
    model = chosen_model(ctx)
    return status(
        paths,
        provider_table(ctx.settings),
        profile_table(model) if model else None,
    )


def remove(ctx: HarnessContext) -> list[str]:
    """Everything a Codex integration consists of, taken back out. For uninstall."""
    return removal_lines(codex_paths(home=ctx.home, env=ctx.env))


def removal_lines(paths: CodexPaths) -> list[str]:
    """Delete our tables and say what happened, for uninstall."""
    problem = parse_problem(paths)
    if problem is not None:
        return [
            f"{paths.config_file} does not parse ({problem}), so it is not rewritten:"
            f" delete its [model_providers.{PROVIDER_ID}] and [profiles.{PROVIDER_ID}]"
            " tables by hand"
        ]
    try:
        removed = drop(paths)
    except OSError as error:
        return [f"could not update {paths.config_file}: {error}"]
    if not removed:
        return []
    lines = [f"removed {', '.join(removed)} from {paths.config_file}"]
    # drop() rewrote the file, so the backup beside it is the one we just made.
    # Nothing above this point deletes it, which is why a file left untouched -
    # unparsable, or holding nothing of ours - keeps whatever backup it had.
    if paths.backup.exists():
        try:
            paths.backup.unlink()
            lines.append(f"deleted {paths.backup}")
        except OSError as error:
            lines.append(f"could not delete {paths.backup}: {error}")
    return lines
