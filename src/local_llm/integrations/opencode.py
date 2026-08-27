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
from . import HarnessContext

PLUGIN_NAME = "local-llm-models.js"
CONFIG_CANDIDATES = ("opencode.jsonc", "opencode.json", "config.json")
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
    return resources.files("local_llm").joinpath("resources/opencode-plugin.js").read_text()


def plugin_status(paths: OpencodePaths) -> str:
    if not paths.plugin.is_file():
        return "missing"
    try:
        current = paths.plugin.read_text()
    except OSError:
        return "unreadable"
    return "same" if current == plugin_source() else "different"


def install_plugin(paths: OpencodePaths) -> Path:
    paths.plugin.parent.mkdir(parents=True, exist_ok=True)
    paths.plugin.write_text(plugin_source())
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
    if not is_strict_json(text):
        return None
    data = json.loads(text or "{}")
    if not isinstance(data, dict):
        return None
    agents = data.setdefault("agent", {})
    if not isinstance(agents, dict):
        return None
    agents[name] = agent
    return json.dumps(data, indent=2) + "\n"


def agent_snippet(agent: dict, name: str = "tiny") -> str:
    return f'"agent": {json.dumps({name: agent}, indent=2)}'


def current_tiny_model(text: str) -> str | None:
    match = _TINY_MODEL.search(text)
    return match.group(1) if match else None


def configure(ctx: HarnessContext, *, agent: bool = True) -> list[str]:
    """Install the plugin, and by default the tiny helper agent."""
    paths = opencode_paths(home=ctx.home, env=ctx.env)
    lines: list[str] = []
    state = plugin_status(paths)
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
                        paths.plugin.read_text().splitlines(),
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
            install_plugin(paths)
            lines.append(f"plugin installed: {paths.plugin}")

    if not agent:
        return lines
    model = smallest_model(ctx.preset) if ctx.preset is not None else None
    if not model:
        lines.append("no model in models.ini yet, so the tiny helper agent was not added")
        return lines
    model_id = f"llamacpp/{model}"
    target = paths.config_file or paths.new_config
    text = target.read_text() if target.is_file() else ""
    if current_tiny_model(text) == model_id:
        lines.append(f"tiny agent already points at {model_id} in {target}")
        return lines
    merged = merge_agent(text, tiny_agent(model_id))
    if merged is None:
        lines.append(f"{target} has comments, so it is not rewritten. Paste this into it:")
        lines.append(agent_snippet(tiny_agent(model_id)))
        return lines
    if ctx.ask(f"Add the tiny helper agent ({model_id}) to {target}?", True):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(merged)
        lines.append(f"tiny agent set to {model_id} in {target}")
    return lines


def harness_status(ctx: HarnessContext) -> str:
    """missing, same or different, judged on the plugin file alone."""
    return plugin_status(opencode_paths(home=ctx.home, env=ctx.env))
