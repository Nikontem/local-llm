"""Point coding agents at the router: environment for claude, copilot, and any tool."""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import NoReturn

from .preset import Preset
from .settings import ENV_KEYS, Settings


class AgentError(Exception):
    pass


def resolve_model(model: str | None, preset: Preset, settings: Settings) -> str:
    name = model or settings.default_model
    if not name:
        raise AgentError(
            "No model given and no default model configured.\n"
            "  local-llm models                        to see the names\n"
            "  LOCAL_LLM_DEFAULT_MODEL=<name>          or default_model in settings.toml"
        )
    if name not in preset.sections():
        raise AgentError(f"Unknown model: {name}\n  local-llm models    to see what is available")
    return name


def _context(model: str, preset: Preset) -> str | None:
    return preset.get(model, "c") or preset.get(model, "ctx-size")


def claude_env(model: str, preset: Preset, settings: Settings) -> dict[str, str]:
    env = {
        "ANTHROPIC_BASE_URL": settings.anthropic_base_url,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_API_KEY": settings.api_key or "dummy",
        # Claude Code's background work asks for a Haiku-class model by name; point it
        # at the same local model or the router is asked for something it has never
        # heard of. ANTHROPIC_SMALL_FAST_MODEL is the deprecated spelling.
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        # These two react to being set at all, whatever the value is. A setup kept
        # deliberately on this machine has no reason to make optional network calls,
        # so both the non-essential traffic and the telemetry are turned off.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    }
    context = _context(model, preset)
    if context:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = context
    return env


def aider_env(settings: Settings) -> dict[str, str]:
    """aider reads the endpoint from OPENAI_API_BASE; the model carries an openai/ prefix."""
    return {
        "OPENAI_API_BASE": settings.openai_base_url,
        "OPENAI_API_KEY": settings.api_key or "dummy",
    }


def aider_args(model: str, extra: list[str]) -> list[str]:
    return ["--model", f"openai/{model}", *extra]


def qwen_env(model: str, settings: Settings) -> dict[str, str]:
    """Qwen Code takes the OpenAI-compatible path only when all three are set."""
    return {
        "OPENAI_BASE_URL": settings.openai_base_url,
        "OPENAI_API_KEY": settings.api_key or "dummy",
        "OPENAI_MODEL": model,
    }


def copilot_env(
    model: str, preset: Preset, settings: Settings, offline: bool = True
) -> dict[str, str]:
    env = {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": settings.openai_base_url,
        "COPILOT_PROVIDER_API_KEY": settings.api_key or "no-key",
        "COPILOT_MODEL": model,
        "COPILOT_OFFLINE": "true" if offline else "false",
    }
    context = _context(model, preset)
    if context:
        env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = context
    output = preset.get(model, "n-predict")
    if output:
        env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = output
    return env


#: What these lines say instead of the key itself. The key can only ever have come from
#: this variable - it is the one setting settings.toml will not hold - so it is already
#: in the environment of any shell that would run this output, and pointing at it makes
#: `eval "$(local-llm env)"` work exactly as before.
KEY_REFERENCE = f'"${ENV_KEYS["api_key"]}"'


def export_lines(
    model: str, preset: Preset, settings: Settings, preset_path: Path, shell: str = "zsh"
) -> str:
    """Shell lines pointing any OpenAI- or Anthropic-style tool at the router.

    They are meant to be read by a person and pasted around, so they end up in
    scrollback, in shell history and in bug reports. No line reproduces the API key's
    value: where there is one, the variable holding it is named instead.
    """
    values = {
        "LOCAL_LLM_OPENAI_BASE_URL": settings.openai_base_url,
        "LOCAL_LLM_ANTHROPIC_BASE_URL": settings.anthropic_base_url,
        "LOCAL_LLM_PRESET": str(preset_path),
        "OPENAI_BASE_URL": settings.openai_base_url,
        "OPENAI_API_KEY": settings.api_key or "dummy",
        **claude_env(model, preset, settings),
    }

    def written(value: str) -> str:
        # A quoted reference, not a quoted value: shlex.quote would put single quotes
        # around it and the shell would hand the tool the variable's name.
        if settings.api_key and value == settings.api_key:
            return KEY_REFERENCE
        return shlex.quote(value)

    if shell == "fish":
        return "".join(f"set -gx {key} {written(value)}\n" for key, value in values.items())
    return "".join(f"export {key}={written(value)}\n" for key, value in values.items())


def exec_with_env(program: str, args: list[str], extra_env: dict[str, str]) -> NoReturn:
    path = shutil.which(program)
    if not path:
        raise AgentError(f"{program} not found in PATH.")
    env = {**os.environ, **extra_env}
    os.execve(path, [program, *args], env)
    # os.execve does not return in real usage; a test double may, so fall through quietly.
