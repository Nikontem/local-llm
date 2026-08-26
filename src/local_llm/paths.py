"""Where local-llm keeps its files, with the LOCAL_LLM_* environment overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SERVICE = "llm-router"


@dataclass(frozen=True)
class Paths:
    config_dir: Path
    preset: Path
    settings_file: Path
    state_dir: Path
    log_dir: Path

    @property
    def pid_file(self) -> Path:
        return self.state_dir / f"{SERVICE}.pid"

    @property
    def ui_file(self) -> Path:
        return self.state_dir / f"{SERVICE}.ui"

    @property
    def hub_cache_file(self) -> Path:
        return self.state_dir / "hub-cache.json"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, home: Path | None = None) -> Paths:
        env = os.environ if env is None else env
        if home is None:
            home = Path(env.get("HOME") or Path.home())
        xdg_config = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
        xdg_state = Path(env.get("XDG_STATE_HOME") or home / ".local" / "state")
        config_dir = Path(env.get("LOCAL_LLM_CONFIG_DIR") or xdg_config / "local-llm")
        preset = Path(env.get("LOCAL_LLM_PRESET") or config_dir / "models.ini")
        state_dir = Path(env.get("LOCAL_LLM_STATE_DIR") or xdg_state / "local-llm")
        log_dir = Path(env.get("LOCAL_LLM_LOG_DIR") or state_dir / "logs")
        return cls(
            config_dir=config_dir,
            preset=preset,
            settings_file=config_dir / "settings.toml",
            state_dir=state_dir,
            log_dir=log_dir,
        )

    def ensure_state_dirs(self) -> None:
        """Create the state and log directories, readable by the owner only."""
        for directory in (self.state_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)
