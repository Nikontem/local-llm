"""opencode integration: the plugin file and the optional `tiny` helper agent."""

from __future__ import annotations

import difflib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..preset import smallest_model
from ..shellrc import encoding_warning, unreadable_warning
from . import HarnessContext, atomic_write, remote_note

PLUGIN_NAME = "local-llm-models.js"
CONFIG_CANDIDATES = ("opencode.jsonc", "opencode.json", "config.json")
#: Every model this tool configures is served by the router, which the plugin registers
#: with opencode under this provider name. A model id without it names somebody else's.
MODEL_PREFIX = "llamacpp/"
#: The last resort for a config with comments in it, which no parser will read.
_TINY_MODEL = re.compile(r'"tiny"\s*:\s*\{[^{}]*?"model"\s*:\s*"([^"]+)"', re.DOTALL)


@dataclass
class OpencodePaths:
    config_dir: Path
    plugin: Path
    config_file: Path | None
    new_config: Path


def opencode_paths(home: Path | None = None, env: Mapping[str, str] | None = None) -> OpencodePaths:
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or Path.home())
    config_dir = Path(env.get("OPENCODE_CONFIG_DIR") or home / ".config" / "opencode")
    existing = next(
        (config_dir / name for name in CONFIG_CANDIDATES if (config_dir / name).is_file()), None
    )
    return OpencodePaths(
        config_dir=config_dir,
        plugin=config_dir / "plugins" / PLUGIN_NAME,
        config_file=existing,
        new_config=config_dir / CONFIG_CANDIDATES[0],
    )


def plugin_source() -> str:
    source = resources.files("local_llm").joinpath("resources/opencode-plugin.js")
    return source.read_text(encoding="utf-8")


def read_plugin(paths: OpencodePaths) -> tuple[str, str]:
    """How the installed plugin stands, and the exact text that decided it.

    The text comes back alongside the answer so that a caller wanting to show a diff
    does not read the file a second time. Between two reads the file can change, or
    stop being readable, and the second read had no guard around it at all.
    """
    if not paths.plugin.is_file():
        return "missing", ""
    try:
        current = paths.plugin.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # A plugin whose bytes are not valid UTF-8 is as unreadable as one we
        # are not allowed to open, and neither may reach the caller as an exception.
        return "unreadable", ""
    return ("same" if current == plugin_source() else "different"), current


def plugin_status(paths: OpencodePaths) -> str:
    return read_plugin(paths)[0]


def install_plugin(paths: OpencodePaths) -> Path:
    # No backup: the plugin file is ours from top to bottom, written from the copy
    # shipped inside the package and deleted again by uninstall. The atomic part still
    # matters, because half a plugin file is an opencode that will not start.
    atomic_write(paths.plugin, plugin_source(), backup=False)
    return paths.plugin


def tiny_agent(model_id: str) -> dict:
    return {
        "description": (
            "Cheap local helper for mechanical text work: summarising, renaming, reformatting,"
            " extracting a value. Use it for anything that does not need reasoning about the"
            " codebase."
        ),
        "mode": "subagent",
        "model": model_id,
        "tools": {"write": False, "edit": False, "bash": False},
    }


def is_strict_json(text: str) -> bool:
    try:
        json.loads(text or "{}")
    except ValueError:
        return False
    return True


def merge_agent(text: str, agent: dict, name: str = "tiny") -> str | None:
    """The config with this agent in it, or None when the file is not strict JSON.

    An agent already under that name is replaced. Whether replacing it is allowed is
    the caller's question to ask, not this one's: see foreign_tiny_model.
    """
    if not is_strict_json(text):
        return None
    data = json.loads(text or "{}")
    if not isinstance(data, dict):
        return None
    agents = data.setdefault("agent", {})
    if not isinstance(agents, dict):
        return None
    agents[name] = agent
    # ensure_ascii=False so a person's own accented text stays readable in their file
    # rather than coming back as \u escapes. The write is UTF-8, so it round-trips.
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def agent_snippet(agent: dict, name: str = "tiny") -> str:
    return f'"agent": {json.dumps({name: agent}, indent=2, ensure_ascii=False)}'


def _model_of(data: object) -> str | None:
    """agent.tiny.model out of a parsed config, giving up at the first thing of the wrong shape."""
    for key in ("agent", "tiny", "model"):
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data if isinstance(data, str) else None


def current_tiny_model(text: str) -> str | None:
    """The model the tiny agent points at, or None when the config has no tiny agent.

    A strict JSON file is parsed. A text match cannot tell the "tiny" under agent from
    a "tiny" under provider, and its character class stops at the first nested object,
    so it reads nothing at all from the shape people actually write - where "model"
    comes after "tools". A file with comments in it cannot be parsed, and nothing is
    ever written to one, so there the match is what there is.
    """
    try:
        data = json.loads(text or "{}")
    except ValueError:
        match = _TINY_MODEL.search(text)
        return match.group(1) if match else None
    return _model_of(data)


def foreign_tiny_model(text: str) -> str | None:
    """The tiny agent's model when somebody other than this tool configured it.

    A tiny agent pointing at a model the router does not serve is the person's own -
    a paid model, very possibly - and replacing it is a question, not a default.
    """
    model = current_tiny_model(text)
    return None if model is None or model.startswith(MODEL_PREFIX) else model


def configure(ctx: HarnessContext, *, agent: bool = True) -> list[str]:
    """Install the plugin, and by default the tiny helper agent.

    The remote note is added here rather than at each way out, so no path this
    function grows later can be the one that forgets it.
    """
    return _configure(ctx, agent=agent) + remote_note(ctx)


def _configure(ctx: HarnessContext, *, agent: bool = True) -> list[str]:
    paths = opencode_paths(home=ctx.home, env=ctx.env)
    lines: list[str] = []
    state, installed = read_plugin(paths)
    if state == "same":
        lines.append(f"plugin already installed: {paths.plugin}")
    elif state == "unreadable":
        lines.append(f"{paths.plugin} cannot be read, so it is left as it is.")
    else:
        replace = True
        if state == "different":
            ctx.say(
                "\n".join(
                    difflib.unified_diff(
                        installed.splitlines(),
                        plugin_source().splitlines(),
                        fromfile=str(paths.plugin),
                        tofile="shipped plugin",
                        lineterm="",
                    )
                )
            )
            replace = ctx.ask(f"Replace {paths.plugin} with the shipped plugin?", True)
        if not replace:
            lines.append("plugin left as it is")
        else:
            try:
                install_plugin(paths)
            except OSError as error:
                # A read-only home or a full disk: say so, never raise at the person.
                lines.append(f"could not write {paths.plugin}: {error}")
            else:
                lines.append(f"plugin installed: {paths.plugin}")

    if not agent:
        return lines
    model = smallest_model(ctx.preset) if ctx.preset is not None else None
    if not model:
        lines.append("no model in models.ini yet, so the tiny helper agent was not added")
        return lines
    model_id = f"{MODEL_PREFIX}{model}"
    target = paths.config_file or paths.new_config
    try:
        # Pinned to UTF-8, and never the locale's encoding. Reading in whatever the
        # locale happens to name and writing back as UTF-8 turns a person's own
        # accented text into mojibake in a file this tool does not own, and the
        # .local-llm.bak beside it is then the only copy of what it used to say.
        text = target.read_text(encoding="utf-8") if target.is_file() else ""
    except UnicodeDecodeError:
        lines.append(encoding_warning(target))
        return lines
    except OSError as error:
        lines.append(unreadable_warning(target, error))
        return lines
    if current_tiny_model(text) == model_id:
        lines.append(f"tiny agent already points at {model_id} in {target}")
        return lines
    theirs = foreign_tiny_model(text)
    if theirs is not None and ctx.yes:
        # Nothing can be asked here, and whether to give up a tiny agent somebody set up
        # themselves is not a question this tool may answer on their behalf.
        lines.append(
            f"the tiny agent in {target} is your own, on {theirs}, so it was left alone:"
            " run local-llm integrate opencode without --yes to replace it"
        )
        return lines
    merged = merge_agent(text, tiny_agent(model_id))
    if merged is None:
        lines.append(f"{target} has comments, so it is not rewritten. Paste this into it:")
        lines.append(agent_snippet(tiny_agent(model_id)))
        return lines
    if theirs is not None:
        allowed = ctx.confirm(
            f"The tiny agent in {target} is your own, on {theirs}, not one local-llm"
            f" wrote. Replace it with {model_id}?",
            False,
        )
    else:
        allowed = ctx.ask(f"Add the tiny helper agent ({model_id}) to {target}?", True)
    if not allowed:
        lines.append(
            f"tiny agent left on {theirs} in {target}"
            if theirs is not None
            else f"tiny agent not added to {target}"
        )
        return lines
    try:
        # The rest of this file is the person's own providers, keybindings and agents.
        atomic_write(target, merged)
    except OSError as error:
        lines.append(f"could not update {target}: {error}")
        return lines
    lines.append(f"tiny agent set to {model_id} in {target}")
    return lines


def harness_status(ctx: HarnessContext) -> str:
    """missing, same, different or unreadable, judged on the plugin file alone."""
    return plugin_status(opencode_paths(home=ctx.home, env=ctx.env))
