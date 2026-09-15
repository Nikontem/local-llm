"""Codex CLI: a named model provider in Codex's config.toml, and a profile file beside it.

Codex 0.134 moved profiles out of config.toml. A profile is now its own file,
`<name>.config.toml` in the same directory, holding top-level keys that layer over the
base config; a `[profiles.<name>]` table or a top-level `profile = "<name>"` selector
left in config.toml makes `codex --profile <name>` refuse to start at all. The provider
still lives in config.toml, the only place Codex looks for `model_providers`. Earlier
versions of this tool wrote the table, so configure moves it out and uninstall knows
to take both files back.
"""

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
PROFILE_NAME = f"{PROVIDER_ID}.config.toml"
WIRE_API = "responses"  # Codex dropped the older "chat" wire format in February 2026
KEY_VARIABLE = "LOCAL_LLM_API_KEY"
#: Every top-level table this tool has ever written into config.toml. `profiles` is
#: legacy: nothing writes it any more, but it is still ours to find and to remove.
PARENTS = ("model_providers", "profiles")
EXPERIMENTAL = (
    "experimental: needs a recent llama.cpp build; tool calls may fail on older ones"
)


@dataclass(frozen=True)
class CodexPaths:
    config_dir: Path
    config_file: Path
    profile_file: Path

    @property
    def backup(self) -> Path:
        """Our own name, not the plain .bak somebody may have made by hand: uninstall
        deletes this file, and it must never be able to delete a person's own copy."""
        return backup_path(self.config_file)

    @property
    def profile_backup(self) -> Path:
        return backup_path(self.profile_file)


def codex_paths(
    home: Path | None = None, env: Mapping[str, str] | None = None
) -> CodexPaths:
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or Path.home())
    directory = Path(env.get("CODEX_HOME") or home / ".codex")
    return CodexPaths(
        config_dir=directory,
        config_file=directory / "config.toml",
        profile_file=directory / PROFILE_NAME,
    )


def paths_for(config_file: Path) -> CodexPaths:
    """The same record built from a file we already found, so nothing is re-derived."""
    return CodexPaths(
        config_dir=config_file.parent,
        config_file=config_file,
        profile_file=config_file.parent / PROFILE_NAME,
    )


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
    """The whole of the profile file: it layers over config.toml, so it needs only what
    differs, and that is the model and the provider serving it."""
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


#: Everything that stops us reading a TOML file: bad TOML (ParseError), bytes that are
#: not UTF-8 (UnicodeDecodeError), or a file we are not allowed to open (OSError).
UNREADABLE = (ParseError, OSError, UnicodeDecodeError)


def _parse(file: Path):
    """The parsed file, or None when it does not exist. Raises anything in UNREADABLE."""
    if not file.is_file():
        return None
    # TOML is UTF-8 by definition. Without saying so, a valid config with an accented
    # comment is read in the locale's encoding and reported as unparsable.
    return tomlkit.parse(file.read_text(encoding="utf-8"))


def _document(paths: CodexPaths):
    return _parse(paths.config_file)


def _profile_document(paths: CodexPaths):
    return _parse(paths.profile_file)


def _problem(file: Path) -> str | None:
    try:
        _parse(file)
    except ParseError as error:
        return f"line {error.line}, column {error.col}: {error}"
    except (OSError, UnicodeDecodeError) as error:
        return f"cannot be read: {error}"
    return None


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


def legacy_remnants(doc) -> list[str]:
    """What an older version of this tool left in config.toml that Codex now refuses."""
    names = []
    if _ours(doc, "profiles") is not None:
        names.append(f"[profiles.{PROVIDER_ID}]")
    if doc.get("profile") == PROVIDER_ID:
        names.append(f'profile = "{PROVIDER_ID}"')
    return names


def legacy_in_config(paths: CodexPaths) -> list[str]:
    """The same, read from disk, for doctor. Nothing when the file cannot be read."""
    try:
        doc = _document(paths)
    except UNREADABLE:
        return []
    return [] if doc is None else legacy_remnants(doc)


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
    """None when config.toml can be read (or is absent), else why it cannot be."""
    return _problem(paths.config_file)


def profile_problem(paths: CodexPaths) -> str | None:
    """None when the profile file can be read (or is absent), else why it cannot be."""
    return _problem(paths.profile_file)


def current_profile(paths: CodexPaths) -> dict | None:
    """What the profile file says, or None when there is none or it cannot be read."""
    try:
        doc = _profile_document(paths)
    except UNREADABLE:
        return None
    return None if doc is None else _unwrap(doc)


def profile_is_ours(profile: Mapping | None) -> bool:
    return profile is not None and profile.get("model_provider") == PROVIDER_ID


def status(paths: CodexPaths, provider: dict, profile: dict | None) -> str:
    try:
        doc = _document(paths)
        current = _profile_document(paths)
    except UNREADABLE:
        return "unreadable"
    ours = None if doc is None else _ours(doc, "model_providers")
    legacy = [] if doc is None else legacy_remnants(doc)
    # Anything of ours in either file counts as "it exists". Judging on the provider
    # alone called a stray profile "missing" and overwrote it with no diff and no
    # question.
    if ours is None and not legacy and current is None:
        return "missing"
    same_provider = ours == provider
    same_profile = profile is None or (current is not None and _unwrap(current) == profile)
    # A legacy table is never "same", however well the rest matches: Codex will not
    # start until it is gone, and configure is what takes it out.
    return "same" if same_provider and same_profile and not legacy else "different"


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


def has_profile_file(paths: CodexPaths) -> bool:
    """Is there a profile file to take back? Ours by its name: nothing else writes
    local-llm.config.toml, and one that will not parse is still ours to report."""
    return paths.profile_file.is_file() or paths.profile_file.is_symlink()


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


def render_profile(profile: dict) -> str:
    doc = tomlkit.document()
    for name, value in profile.items():
        doc[name] = value
    return tomlkit.dumps(doc)


def render(provider: dict, profile: dict | None) -> str:
    """Just what we own, in both files, for a diff or for printing when a file must
    not be written. The profile file follows its name, since it is a file of its own."""
    doc = tomlkit.document()
    _set(doc, "model_providers", provider)
    text = tomlkit.dumps(doc)
    if profile is not None:
        text += f"\n# {PROFILE_NAME}:\n" + render_profile(profile)
    return text


def current_render(paths: CodexPaths) -> str:
    """What is on disk now, in the shape render() gives, so the two diff cleanly."""
    try:
        doc = _document(paths)
        current = _profile_document(paths)
    except UNREADABLE:
        return ""
    provider = None if doc is None else _ours(doc, "model_providers")
    if provider is None and current is None:
        return ""
    out = tomlkit.document()
    if doc is not None and doc.get("profile") == PROVIDER_ID:
        out["profile"] = PROVIDER_ID  # a scalar, so it has to come before any table
    if provider is not None:
        _set(out, "model_providers", provider)
    legacy = None if doc is None else _ours(doc, "profiles")
    if legacy is not None:
        _set(out, "profiles", legacy)
    text = tomlkit.dumps(out)
    if current is not None:
        text += f"\n# {PROFILE_NAME}:\n" + tomlkit.dumps(current)
    return text


def write_profile(paths: CodexPaths, profile: dict) -> None:
    atomic_write(paths.profile_file, render_profile(profile))


def write(paths: CodexPaths, provider: dict, profile: dict | None) -> list[str]:
    """Write the provider into config.toml and the profile into its own file.

    Any legacy profile table goes in the same rewrite: leaving it would make Codex
    refuse the very profile that was just written. Names what was taken out, so the
    caller can say the profile moved rather than silently vanished.
    """
    doc = _document(paths)
    if doc is None:
        doc = tomlkit.document()
    _set(doc, "model_providers", provider)
    moved = _drop_legacy(doc)
    atomic_write(paths.config_file, tomlkit.dumps(doc))
    if profile is not None:
        write_profile(paths, profile)
    return moved


def _drop_table(doc, parent_name: str) -> bool:
    parent = doc.get(parent_name)
    # Not a mapping means nothing of ours is under it, whatever it is. See _ours.
    if not hasattr(parent, "get") or PROVIDER_ID not in parent:
        return False
    del parent[PROVIDER_ID]
    # An emptied table goes too, unless the person left a comment inside it.
    if not len(parent) and not tomlkit.dumps(parent).strip():
        del doc[parent_name]
    return True


def _drop_legacy(doc) -> list[str]:
    removed = []
    if _drop_table(doc, "profiles"):
        removed.append(f"[profiles.{PROVIDER_ID}]")
    if doc.get("profile") == PROVIDER_ID:
        del doc["profile"]
        removed.append(f'profile = "{PROVIDER_ID}"')
    return removed


def drop(paths: CodexPaths) -> list[str]:
    """Delete our tables from config.toml, leaving everything else alone. Names what
    it removed."""
    doc = _document(paths)
    if doc is None:
        return []
    removed: list[str] = []
    if _drop_table(doc, "model_providers"):
        removed.append(f"[model_providers.{PROVIDER_ID}]")
    removed += _drop_legacy(doc)
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


def drop_profile(paths: CodexPaths) -> list[str]:
    """Delete the profile file and its backup, saying what happened. For uninstall.

    The file is ours by its name and is only ever read by `codex --profile local-llm`,
    which stops meaning anything once the provider is gone, so anything a person added
    to it does not keep it. One that names another provider is theirs, and stays. One
    that will not parse cannot be judged, so it is named for deleting by hand.
    """
    lines: list[str] = []
    problem = profile_problem(paths)
    if problem is not None:
        lines.append(f"{paths.profile_file} does not parse ({problem}): delete it by hand")
    elif profile_is_ours(current_profile(paths)):
        try:
            # A symlink is unlinked, never followed: the target is somebody's own file.
            paths.profile_file.unlink()
            lines.append(f"deleted {paths.profile_file}")
        except OSError as error:
            lines.append(f"could not delete {paths.profile_file}: {error}")
    if paths.profile_backup.exists() or paths.profile_backup.is_symlink():
        try:
            paths.profile_backup.unlink()
            lines.append(f"deleted {paths.profile_backup}")
        except OSError as error:
            lines.append(f"could not delete {paths.profile_backup}: {error}")
    return lines


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
        broken, problem = paths.config_file, parse_problem(paths)
        if problem is None:
            broken, problem = paths.profile_file, profile_problem(paths)
        return [
            f"{broken} does not parse ({problem}), so it is not rewritten. Paste this"
            " into it:",
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
            where = f"{paths.config_file} and {paths.profile_file}"
            if not ctx.ask(f"Replace the local-llm settings in {where}?", True):
                return ["left as it is"]
        try:
            moved = write(paths, provider, profile)
        except UNREADABLE as error:
            # A read-only home, a full disk or somebody else's file: say so, never raise.
            return [f"could not update {paths.config_file}: {error}"]
        if profile is None:
            lines.append(f"provider local-llm written to {paths.config_file}")
        else:
            lines.append(
                f"provider local-llm written to {paths.config_file};"
                f" profile written to {paths.profile_file}"
            )
        if moved:
            lines.append(
                f"moved the profile out of {paths.config_file} into"
                f" {paths.profile_file}: removed {', '.join(moved)}"
                " (Codex 0.134+ refuses the old table)"
            )

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
    """Delete our tables and the profile file, and say what happened, for uninstall.

    The two files are judged separately: a config.toml that will not parse is no
    reason to leave a profile file that is entirely ours, and the other way round.
    """
    return _removal_from_config(paths) + drop_profile(paths)


def _removal_from_config(paths: CodexPaths) -> list[str]:
    problem = parse_problem(paths)
    if problem is not None:
        if not mentions_us(paths):
            return []  # somebody else's broken config: nothing of ours in it to discuss
        return [
            f"{paths.config_file} does not parse ({problem}), so it is not rewritten:"
            f" delete its [model_providers.{PROVIDER_ID}] table, and any"
            f" [profiles.{PROVIDER_ID}] table or profile = \"{PROVIDER_ID}\" line, by hand"
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
