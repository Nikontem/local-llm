"""Codex CLI: a named model provider and a profile inside Codex's own config.toml."""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError

from ..preset import smallest_model
from ..settings import Settings
from . import HarnessContext, atomic_write, backup_path, remote_note

PROVIDER_ID = "local-llm"
WIRE_API = "responses"  # Codex dropped the older "chat" wire format in February 2026
KEY_VARIABLE = "LOCAL_LLM_API_KEY"
PARENTS = ("model_providers", "profiles")
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
        return backup_path(self.config_file)


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
    # TOML is UTF-8 by definition. Without saying so, a valid config with an accented
    # comment is read in the locale's encoding and reported as unparsable.
    return tomlkit.parse(paths.config_file.read_text(encoding="utf-8"))


def _ours(doc, parent_name: str) -> dict | None:
    """Our table under that parent, or None when there is not one to find.

    A file that parses is not a file of the shape we expect. `profiles = "work"` is
    valid TOML and the plausible typo for Codex's own top-level `profile`, and asking
    a string for a key used to end the whole command in a traceback. Anything that is
    not a mapping holds nothing of ours by definition.
    """
    parent = doc.get(parent_name)
    if not hasattr(parent, "get"):
        return None
    return _unwrap(parent.get(PROVIDER_ID))


def wrong_shape(paths: CodexPaths) -> str | None:
    """The top-level key that is not a table where we need one, or None when all is well.

    A file that parses is not a file we can write into. `profiles = "work"` is valid
    TOML and the plausible typo for Codex's own top-level `profile`; putting our table
    there would silently overwrite whatever the person meant by it.
    """
    try:
        doc = _document(paths)
    except UNREADABLE:
        return None  # reported as unreadable instead, by parse_problem
    if doc is None:
        return None
    return next(
        (
            name
            for name in PARENTS
            if (value := doc.get(name)) is not None and not hasattr(value, "get")
        ),
        None,
    )


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
    # Either table counts as "it exists". Judging on the provider alone called a stray
    # [profiles.local-llm] "missing" and overwrote it with no diff and no question.
    if doc is None or all(_ours(doc, name) is None for name in PARENTS):
        return "missing"
    same_provider = _ours(doc, "model_providers") == provider
    same_profile = profile is None or _ours(doc, "profiles") == profile
    return "same" if same_provider and same_profile else "different"


def mentions_us(paths: CodexPaths) -> bool:
    """Does the file name us at all? Read as bytes, because one of the ways a config
    fails to parse is not being UTF-8, and the answer is still a plain yes or no."""
    try:
        return PROVIDER_ID.encode() in paths.config_file.read_bytes()
    except OSError:
        return False


def unparsable_but_ours(paths: CodexPaths) -> bool:
    """A config we cannot read that nonetheless names us.

    Uninstall has to know about this file: it cannot be edited safely, so it must be
    listed and the lines to delete by hand printed, rather than passed over in silence.
    """
    return parse_problem(paths) is not None and mentions_us(paths)


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


def write(paths: CodexPaths, provider: dict, profile: dict | None) -> None:
    doc = _document(paths)
    if doc is None:
        doc = tomlkit.document()
    _set(doc, "model_providers", provider)
    if profile is not None:
        _set(doc, "profiles", profile)
    atomic_write(paths.config_file, tomlkit.dumps(doc))


def drop(paths: CodexPaths) -> list[str]:
    """Delete our tables, leaving everything else alone. Names what it removed."""
    doc = _document(paths)
    if doc is None:
        return []
    removed: list[str] = []
    for parent_name in PARENTS:
        parent = doc.get(parent_name)
        # Not a mapping means nothing of ours is under it, whatever it is. See _ours.
        if not hasattr(parent, "get") or PROVIDER_ID not in parent:
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
        rest = tomlkit.dumps(doc)
        if rest.strip() or paths.config_file.is_symlink():
            # Anything of theirs still in it - a setting, or a comment we left alone -
            # and the file stays. So does a symlink, which is somebody's dotfiles
            # arrangement: breaking the link would be a change they did not ask for.
            atomic_write(paths.config_file, rest)
        else:
            # Our tables were the whole file, so the file is ours too: it did not exist
            # before configure wrote it. A zero-byte config.toml left behind is not the
            # machine put back the way it was found.
            paths.config_file.unlink(missing_ok=True)
    return removed


def configure(ctx: HarnessContext) -> list[str]:
    """Write our provider and profile into Codex's config, asking before replacing.

    The remote note is added here rather than at each way out, so no path this
    function grows later can be the one that forgets it.
    """
    return _configure(ctx) + remote_note(ctx)


def _configure(ctx: HarnessContext) -> list[str]:
    paths = codex_paths(home=ctx.home, env=ctx.env)
    provider = provider_table(ctx.settings)
    model = chosen_model(ctx)
    profile = profile_table(model) if model else None
    state = status(paths, provider, profile)

    shape = wrong_shape(paths)
    if shape is not None:
        return [
            f"{paths.config_file} has a {shape} setting that is not a table, so it is"
            " not rewritten. Paste this into it:",
            render(provider, profile).rstrip(),
        ]

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
        written = "provider" if profile is None else "provider and profile"
        lines.append(f"{written} local-llm written to {paths.config_file}")

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
        if not mentions_us(paths):
            return []  # somebody else's broken config: nothing of ours in it to discuss
        return [
            f"{paths.config_file} does not parse ({problem}), so it is not rewritten:"
            f" delete its [model_providers.{PROVIDER_ID}] and [profiles.{PROVIDER_ID}]"
            " tables by hand"
        ]
    try:
        removed = drop(paths)
    except UNREADABLE as error:
        return [f"could not update {paths.config_file}: {error}"]
    if not removed:
        return []
    lines = [f"removed {', '.join(removed)} from {paths.config_file}"]
    if not paths.config_file.exists():
        lines.append(f"deleted {paths.config_file}: it held nothing but those tables")
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
