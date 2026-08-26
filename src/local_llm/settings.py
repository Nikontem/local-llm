"""Settings that are not llama-server flags: port, host, memory reserve, defaults.

Precedence, highest first: command flag (passed as overrides), environment,
settings.toml, built-in default. The API key is environment-only.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .paths import Paths

ENV_KEYS: dict[str, str] = {
    "port": "LOCAL_LLM_PORT",
    "host": "LOCAL_LLM_HOST",
    "max_models": "LOCAL_LLM_MAX_MODELS",
    "reserve_gb": "LOCAL_LLM_RESERVE_GB",
    "ui": "LOCAL_LLM_UI",
    "default_model": "LOCAL_LLM_DEFAULT_MODEL",
    "allow_remote": "LOCAL_LLM_ALLOW_REMOTE",
    "api_key": "LOCAL_LLM_API_KEY",
}

# Keys that may live in settings.toml. The API key is deliberately absent.
FILE_KEYS: tuple[str, ...] = (
    "port", "host", "max_models", "reserve_gb", "ui", "default_model", "allow_remote",
)

_TYPES: dict[str, type] = {
    "port": int, "host": str, "max_models": int, "reserve_gb": int, "ui": bool,
    "default_model": str, "allow_remote": bool, "api_key": str,
}

_LOOPBACK = ("127.0.0.1", "localhost", "::1")


@dataclass
class Settings:
    port: int = 5678
    host: str = "127.0.0.1"
    max_models: int = 1
    reserve_gb: int = 10
    ui: bool = False
    default_model: str = ""
    allow_remote: bool = False
    api_key: str = ""

    @property
    def openai_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def anthropic_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_local(self) -> bool:
        return self.host in _LOOPBACK


def _coerce(name: str, value: object) -> object:
    kind = _TYPES[name]
    if kind is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if kind is int:
        return int(value)  # type: ignore[arg-type]
    return str(value)


def load_settings(
    paths: Paths,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Settings:
    env = os.environ if env is None else env
    values: dict[str, object] = {}
    if paths.settings_file.is_file():
        data = tomllib.loads(paths.settings_file.read_text())
        for key in FILE_KEYS:
            if key in data:
                values[key] = _coerce(key, data[key])
    for key, variable in ENV_KEYS.items():
        raw = env.get(variable)
        if raw not in (None, ""):
            values[key] = _coerce(key, raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = _coerce(key, value)
    return Settings(**values)  # type: ignore[arg-type]


def dump_settings(settings: Settings) -> str:
    lines = ["# local-llm settings. Environment variables (LOCAL_LLM_*) override these.", ""]
    for key in FILE_KEYS:
        value = getattr(settings, key)
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, int):
            text = str(value)
        else:
            text = json.dumps(value)  # a JSON string is a valid TOML basic string
        lines.append(f"{key} = {text}")
    return "\n".join(lines) + "\n"


def save_settings(paths: Paths, settings: Settings) -> Path:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.settings_file.write_text(dump_settings(settings))
    return paths.settings_file
