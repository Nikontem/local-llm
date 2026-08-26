# local-llm — Milestone 1: Runtime Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every command that `local_llm.zsh` provides today (`up`, `down`, `restart`, `status`, `logs`, `ui`, `models`, `load`, `unload`, `edit`, `prune-logs`, the `claude`/`copilot` wrappers) works from a Python package called `local-llm`, against the author's existing `~/.config/local-llm/models.ini`, plus a `doctor` command and an `env` command.

**Architecture:** One Python package `local_llm` with small single-purpose modules (`paths`, `settings`, `preset`, `logs`, `estimate`, `hardware`, `router`, `agents`, `hub`, `doctor`) and a thin typer CLI in `cli.py`. All process handling goes through an injectable `ProcessBackend` so the router logic is unit-tested against a fake; all HTTP goes through an injectable callable. `llama-server` is started detached with a pid file and per-run log files, exactly the model the zsh script uses.

**Tech Stack:** Python 3.11+, uv (dev environment and lock file), hatchling (build), typer (CLI + completion), rich (console output), psutil (processes and memory), huggingface_hub (token check only in this milestone), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-local-llm-tool-design.md` — sections 3, 4.2, 4.3, 4.7, 10, 11, 13, 14, 15, 16.1 are implemented here.

## Global Constraints

- `requires-python = ">=3.11"`; the dev environment is pinned to CPython 3.13 with `uv python pin 3.13`.
- Runtime dependencies are exactly `typer`, `rich`, `huggingface_hub`, `psutil`. Nothing else may be added.
- Command name `local-llm`, package `local_llm`, `src/` layout, hatchling build backend.
- Environment variable names are the existing ones: `LOCAL_LLM_CONFIG_DIR`, `LOCAL_LLM_PRESET`, `LOCAL_LLM_STATE_DIR`, `LOCAL_LLM_LOG_DIR`, `LOCAL_LLM_PORT`, `LOCAL_LLM_HOST`, `LOCAL_LLM_MAX_MODELS`, `LOCAL_LLM_RESERVE_GB`, `LOCAL_LLM_UI`, `LOCAL_LLM_DEFAULT_MODEL`, `LOCAL_LLM_ALLOW_REMOTE`, `LOCAL_LLM_API_KEY`.
- Default paths: config `~/.config/local-llm/`, preset `CONFIG/models.ini`, settings `CONFIG/settings.toml`, state `~/.local/state/local-llm/`, logs `STATE/logs/`; `$XDG_CONFIG_HOME` and `$XDG_STATE_HOME` honoured.
- Settings precedence: command flag, then environment, then `settings.toml`, then default. Defaults: `port = 5678`, `host = "127.0.0.1"`, `max_models = 1`, `reserve_gb = 10`, `ui = false`, `default_model = ""`, `allow_remote = false`. The API key is env-only and never written to disk.
- Never signal a process whose command line's program name lacks `llama-server`. Never bind a non-loopback host without both `allow_remote` and an API key. Preset writes are atomic and keep a `.bak`.
- Every error message: what failed, likely cause, the command to run next. Errors go to stderr. Exit codes: 0 success, 1 handled failure, 2 usage error.
- Log files: `LOGS/llm-router.<YYYY-MM-DDTHH-MM-SS>.log` mode 600, with `LOGS/llm-router.log` a symlink to the current one (a text file `current` holding the path when symlinks are unavailable).
- The memory estimate is `sum(file sizes) × 1.15 + 1 GiB` (integer arithmetic: `sum * 115 // 100 + 1 GiB`). A child process below 500 MB resident is reported as asleep.
- Unit tests never touch the network and never start real processes. Tests live in `tests/unit/`. Run everything with `uv run pytest -q` and `uv run ruff check .`.
- Commit after every task with a conventional-commit message (`feat:`, `test:`, `chore:`), and add these trailer lines to every commit message:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01KxgyhTuV6MNeVp4bvjqXrp`.
- Working directory for every command in this plan: `~/.config/local-llm/local-llm` (the repo root).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | package metadata, dependencies, `local-llm` entry point, pytest and ruff config |
| `src/local_llm/__init__.py` | `__version__` read from package metadata |
| `src/local_llm/paths.py` | `Paths` record: where config, preset, settings, state and logs live; env overrides |
| `src/local_llm/settings.py` | `Settings` record; load with precedence; dump/save `settings.toml` |
| `src/local_llm/preset.py` | `Preset`: parse/dump the llama-server INI preserving comments; query keys; add/replace/remove sections; atomic save; file sizes per section |
| `src/local_llm/logs.py` | per-run log file creation, current-log pointer, pruning |
| `src/local_llm/estimate.py` | memory estimate, budget, fit classification, GB formatting |
| `src/local_llm/hardware.py` | `total_ram()` only (milestone 2 adds full detection) |
| `src/local_llm/router.py` | `ProcessBackend` protocol, `PsutilBackend`, `Router` (discover, start, stop, children, UI state, HTTP API) |
| `src/local_llm/agents.py` | environment for `claude`, `copilot`, `env`; exec helper |
| `src/local_llm/hub.py` | `token_status()` only (milestone 2 adds search/downloads) |
| `src/local_llm/doctor.py` | `Check` record and `run_checks()` |
| `src/local_llm/cli.py` | typer app: every command, model-name completion, output formatting |
| `tests/unit/fakes.py` | `FakeBackend` and `FakeHttp` used by router and CLI tests |
| `tests/unit/test_*.py` | one test file per module |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `LICENSE`, `README.md`, `src/local_llm/__init__.py`, `src/local_llm/cli.py`, `tests/unit/__init__.py`, `tests/unit/test_cli_version.py`

**Interfaces:**
- Produces: `local_llm.__version__: str`; `local_llm.cli.app` (a `typer.Typer`) that answers `local-llm --version`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "local-llm"
version = "0.1.0"
description = "Get llama.cpp serving models on your machine in one guided session, and wire your coding agents to it."
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [{ name = "nikosntemkas" }]
dependencies = [
  "typer>=0.15",
  "rich>=13",
  "huggingface_hub>=0.30",
  "psutil>=6",
]

[project.scripts]
local-llm = "local_llm.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/local_llm"]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.8"]

[tool.pytest.ini_options]
testpaths = ["tests/unit"]
markers = ["live: needs network access to huggingface.co"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 2: Write `.gitignore`, `LICENSE`, `README.md`**

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
dist/
build/
*.egg-info/
.pytest_cache/
.ruff_cache/
.DS_Store
```

`LICENSE`: the MIT license text with the line `Copyright (c) 2026 nikosntemkas`.

`README.md`:

```markdown
# local-llm

Work in progress. One `llama-server` router serving every model in `models.ini`,
plus a guided setup. The design is in `docs/superpowers/specs/`.
```

- [ ] **Step 3: Write the package init and a minimal CLI**

`src/local_llm/__init__.py`:

```python
"""local-llm: get llama.cpp serving models on your machine, and wire your agents to it."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-llm")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
```

`src/local_llm/cli.py` (this file is rewritten in Task 14; for now it only answers `--version`):

```python
"""Command-line entry point."""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(
    help="One llama.cpp router serving every model in models.ini.",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(f"local-llm {__version__}")
        raise typer.Exit()
```

- [ ] **Step 4: Write the failing test**

`tests/unit/__init__.py` is empty. `tests/unit/test_cli_version.py`:

```python
from typer.testing import CliRunner

from local_llm import __version__
from local_llm.cli import app


def test_version_flag_prints_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"local-llm {__version__}"
```

- [ ] **Step 5: Create the environment and run the test**

Run:

```bash
uv python pin 3.13
uv sync
uv run pytest -q
uv run ruff check .
uv run local-llm --version
```

Expected: `uv sync` creates `.venv` and `uv.lock`; pytest reports `1 passed`; ruff reports no errors; the last command prints `local-llm 0.1.0`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore LICENSE README.md src tests
git commit -m "chore: scaffold the local-llm package"
```

---

### Task 2: Paths

**Files:**
- Create: `src/local_llm/paths.py`
- Test: `tests/unit/test_paths.py`

**Interfaces:**
- Produces: `Paths` frozen dataclass with fields `config_dir, preset, settings_file, state_dir, log_dir: Path`; properties `pid_file`, `ui_file`, `hub_cache_file: Path`; classmethod `Paths.from_env(env: Mapping[str, str] | None = None, home: Path | None = None) -> Paths`; method `ensure_state_dirs() -> None`; constant `SERVICE = "llm-router"`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_paths.py`:

```python
from pathlib import Path

from local_llm.paths import SERVICE, Paths


def test_defaults_follow_xdg_layout(tmp_path):
    p = Paths.from_env(env={}, home=tmp_path)
    assert p.config_dir == tmp_path / ".config" / "local-llm"
    assert p.preset == tmp_path / ".config" / "local-llm" / "models.ini"
    assert p.settings_file == tmp_path / ".config" / "local-llm" / "settings.toml"
    assert p.state_dir == tmp_path / ".local" / "state" / "local-llm"
    assert p.log_dir == tmp_path / ".local" / "state" / "local-llm" / "logs"
    assert p.pid_file == p.state_dir / f"{SERVICE}.pid"
    assert p.ui_file == p.state_dir / f"{SERVICE}.ui"
    assert p.hub_cache_file == p.state_dir / "hub-cache.json"


def test_xdg_variables_are_honoured(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_STATE_HOME": str(tmp_path / "st")}
    p = Paths.from_env(env=env, home=tmp_path)
    assert p.config_dir == tmp_path / "cfg" / "local-llm"
    assert p.state_dir == tmp_path / "st" / "local-llm"


def test_local_llm_variables_override_everything(tmp_path):
    env = {
        "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
        "LOCAL_LLM_CONFIG_DIR": "/opt/llm",
        "LOCAL_LLM_PRESET": "/opt/llm/other.ini",
        "LOCAL_LLM_STATE_DIR": "/var/llm",
        "LOCAL_LLM_LOG_DIR": "/var/log/llm",
    }
    p = Paths.from_env(env=env, home=tmp_path)
    assert p.config_dir == Path("/opt/llm")
    assert p.preset == Path("/opt/llm/other.ini")
    assert p.settings_file == Path("/opt/llm/settings.toml")
    assert p.state_dir == Path("/var/llm")
    assert p.log_dir == Path("/var/log/llm")


def test_ensure_state_dirs_creates_private_dirs(tmp_path):
    p = Paths.from_env(env={}, home=tmp_path)
    p.ensure_state_dirs()
    assert p.state_dir.is_dir() and p.log_dir.is_dir()
    assert oct(p.state_dir.stat().st_mode & 0o777) == "0o700"
    assert oct(p.log_dir.stat().st_mode & 0o777) == "0o700"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_paths.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.paths'`

- [ ] **Step 3: Write `src/local_llm/paths.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_paths.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/paths.py tests/unit/test_paths.py
git commit -m "feat: paths with XDG defaults and LOCAL_LLM_* overrides"
```

---

### Task 3: Settings

**Files:**
- Create: `src/local_llm/settings.py`
- Test: `tests/unit/test_settings.py`

**Interfaces:**
- Consumes: `Paths` (Task 2).
- Produces: `Settings` dataclass (`port: int`, `host: str`, `max_models: int`, `reserve_gb: int`, `ui: bool`, `default_model: str`, `allow_remote: bool`, `api_key: str`) with properties `openai_base_url`, `anthropic_base_url: str` and `is_local: bool`; `load_settings(paths: Paths, env: Mapping[str, str] | None = None, overrides: Mapping[str, object] | None = None) -> Settings`; `dump_settings(settings: Settings) -> str`; `save_settings(paths: Paths, settings: Settings) -> Path`; constants `ENV_KEYS: dict[str, str]`, `FILE_KEYS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_settings.py`:

```python
import tomllib

from local_llm.paths import Paths
from local_llm.settings import Settings, dump_settings, load_settings, save_settings


def paths_for(tmp_path):
    return Paths.from_env(env={}, home=tmp_path)


def test_defaults_when_nothing_is_configured(tmp_path):
    s = load_settings(paths_for(tmp_path), env={})
    assert s == Settings()
    assert s.port == 5678 and s.host == "127.0.0.1" and s.max_models == 1
    assert s.reserve_gb == 10 and s.ui is False and s.default_model == ""
    assert s.allow_remote is False and s.api_key == ""
    assert s.openai_base_url == "http://127.0.0.1:5678/v1"
    assert s.anthropic_base_url == "http://127.0.0.1:5678"
    assert s.is_local


def test_file_then_env_then_overrides(tmp_path):
    p = paths_for(tmp_path)
    p.config_dir.mkdir(parents=True)
    p.settings_file.write_text('port = 6000\nmax_models = 3\ndefault_model = "a"\nui = true\n')
    env = {"LOCAL_LLM_PORT": "6001", "LOCAL_LLM_UI": "0", "LOCAL_LLM_API_KEY": "sekrit"}
    s = load_settings(p, env=env, overrides={"port": 6002, "host": None})
    assert s.port == 6002          # override wins over env and file
    assert s.max_models == 3       # from file
    assert s.default_model == "a"  # from file
    assert s.ui is False           # env "0" beats file true
    assert s.api_key == "sekrit"   # env only
    assert s.host == "127.0.0.1"   # None override means "not given"


def test_bool_parsing_from_env(tmp_path):
    p = paths_for(tmp_path)
    for raw, expected in [("1", True), ("true", True), ("YES", True), ("0", False), ("off", False)]:
        assert load_settings(p, env={"LOCAL_LLM_ALLOW_REMOTE": raw}).allow_remote is expected


def test_is_local_for_loopback_names_only():
    assert Settings(host="localhost").is_local
    assert Settings(host="::1").is_local
    assert not Settings(host="0.0.0.0").is_local


def test_dump_is_valid_toml_without_the_api_key():
    text = dump_settings(Settings(port=7000, default_model='q"uoted', api_key="never"))
    data = tomllib.loads(text)
    assert data["port"] == 7000 and data["default_model"] == 'q"uoted'
    assert "api_key" not in data and "never" not in text


def test_save_writes_file_and_round_trips(tmp_path):
    p = paths_for(tmp_path)
    written = save_settings(p, Settings(port=7001, reserve_gb=4))
    assert written == p.settings_file
    assert load_settings(p, env={}) == Settings(port=7001, reserve_gb=4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_settings.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.settings'`

- [ ] **Step 3: Write `src/local_llm/settings.py`**

```python
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
    max_models: int = 2
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_settings.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/settings.py tests/unit/test_settings.py
git commit -m "feat: settings with flag > env > settings.toml > default precedence"
```

---

### Task 4: Preset parser and queries

**Files:**
- Create: `src/local_llm/preset.py`
- Test: `tests/unit/test_preset.py`

**Interfaces:**
- Produces: `PresetError(Exception)`; class `Preset` with `Preset.parse(text: str) -> Preset`, `Preset.load(path: Path) -> Preset` (raises `PresetError` when missing), `dump() -> str`, `sections() -> list[str]` (every header except `*`, in file order, no duplicates), `has_section(name: str) -> bool`, `items(section: str) -> dict[str, str]`, `get(section: str, key: str, fallback_to_star: bool = True) -> str | None`, `file_sizes(section: str) -> list[int]` (sizes of the `model` and `mmproj` files; raises `PresetError` naming the missing file or the missing `model` key).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_preset.py`:

```python
import pytest

from local_llm.preset import Preset, PresetError

SAMPLE = """version = 1

[*]
jinja = true
n-gpu-layers = all   ; trailing comment
# Release weights after 5 minutes idle
sleep-idle-seconds = 300

[qwen-3.8-q4]
model = /models/Qwen3.8-27B-UD-Q4_K_XL.gguf
mmproj = /models/mmproj-F16.gguf
c = 65536
temp = 1.0

[tiny]
model = /models/tiny#1.gguf
c = 32768
"""


def test_round_trip_is_byte_identical():
    assert Preset.parse(SAMPLE).dump() == SAMPLE


def test_sections_excludes_star_and_keeps_order():
    assert Preset.parse(SAMPLE).sections() == ["qwen-3.8-q4", "tiny"]


def test_items_and_get_with_star_fallback():
    p = Preset.parse(SAMPLE)
    assert p.items("qwen-3.8-q4") == {
        "model": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "mmproj": "/models/mmproj-F16.gguf",
        "c": "65536",
        "temp": "1.0",
    }
    assert p.get("tiny", "c") == "32768"
    assert p.get("tiny", "jinja") == "true"                       # inherited from [*]
    assert p.get("tiny", "jinja", fallback_to_star=False) is None
    assert p.get("tiny", "nope") is None
    assert p.get("*", "n-gpu-layers") == "all"                    # inline comment stripped


def test_hash_inside_a_value_without_whitespace_is_kept():
    assert Preset.parse(SAMPLE).get("tiny", "model") == "/models/tiny#1.gguf"


def test_has_section():
    p = Preset.parse(SAMPLE)
    assert p.has_section("tiny") and not p.has_section("missing")


def test_load_missing_file_raises_preset_error(tmp_path):
    with pytest.raises(PresetError, match="Missing model config"):
        Preset.load(tmp_path / "none.ini")


def test_file_sizes_sums_model_and_mmproj(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x" * 10)
    (tmp_path / "p.gguf").write_bytes(b"y" * 5)
    p = Preset.parse(f"[a]\nmodel = {tmp_path}/m.gguf\nmmproj = {tmp_path}/p.gguf\n[b]\nmodel = {tmp_path}/m.gguf\n")
    assert p.file_sizes("a") == [10, 5]
    assert p.file_sizes("b") == [10]


def test_file_sizes_reports_missing_file_and_missing_key(tmp_path):
    p = Preset.parse(f"[a]\nmodel = {tmp_path}/gone.gguf\n[b]\nc = 1\n")
    with pytest.raises(PresetError, match="gone.gguf"):
        p.file_sizes("a")
    with pytest.raises(PresetError, match="no model key"):
        p.file_sizes("b")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_preset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.preset'`

- [ ] **Step 3: Write `src/local_llm/preset.py`**

```python
"""Read and edit llama-server preset files (INI) without losing comments or order.

The file is kept as a list of lines. Each line remembers what it is (header,
key = value, comment, blank) and which section it belongs to, so queries are
simple and writing back reproduces the file byte for byte.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_HEADER = re.compile(r"^\s*\[(?P<name>[^\]\n]+)\]\s*$")
# A comment starts with # or ; at the beginning of the line or after whitespace.
# A # glued to the previous character (as in a path) is part of the value.
_COMMENT = re.compile(r"(^|\s)[#;]")


class PresetError(Exception):
    pass


@dataclass
class _Line:
    raw: str
    kind: str  # "header" | "kv" | "comment" | "blank" | "other"
    section: str | None = None  # the header name, or the section a line sits in
    key: str | None = None
    value: str | None = None


def _strip_comment(line: str) -> str:
    match = _COMMENT.search(line)
    return line[: match.start()] if match else line


class Preset:
    def __init__(self, lines: list[_Line] | None = None) -> None:
        self._lines: list[_Line] = lines or []

    # ------------------------------------------------------------ parse / dump

    @classmethod
    def parse(cls, text: str) -> Preset:
        lines: list[_Line] = []
        current: str | None = None
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped == "":
                lines.append(_Line(raw, "blank", current))
                continue
            if stripped[0] in "#;":
                lines.append(_Line(raw, "comment", current))
                continue
            header = _HEADER.match(_strip_comment(raw))
            if header:
                current = header.group("name").strip()
                lines.append(_Line(raw, "header", current))
                continue
            body = _strip_comment(raw)
            if "=" in body:
                key, _, value = body.partition("=")
                lines.append(_Line(raw, "kv", current, key.strip(), value.strip()))
                continue
            lines.append(_Line(raw, "other", current))
        return cls(lines)

    @classmethod
    def load(cls, path: Path) -> Preset:
        try:
            text = path.read_text()
        except FileNotFoundError:
            raise PresetError(
                f"Missing model config: {path}\n"
                "Every model lives in that file. Create it with: local-llm setup"
            ) from None
        return cls.parse(text)

    def dump(self) -> str:
        return "".join(line.raw + "\n" for line in self._lines)

    # ------------------------------------------------------------ queries

    def sections(self) -> list[str]:
        names: list[str] = []
        for line in self._lines:
            if line.kind == "header" and line.section != "*" and line.section not in names:
                names.append(line.section)  # type: ignore[arg-type]
        return names

    def has_section(self, name: str) -> bool:
        return any(line.kind == "header" and line.section == name for line in self._lines)

    def items(self, section: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in self._lines:
            if line.kind == "kv" and line.section == section and line.key is not None:
                out[line.key] = line.value or ""
        return out

    def get(self, section: str, key: str, fallback_to_star: bool = True) -> str | None:
        own = self.items(section)
        if key in own:
            return own[key]
        if fallback_to_star:
            return self.items("*").get(key)
        return None

    def file_sizes(self, section: str) -> list[int]:
        """Sizes in bytes of the model file and, when present, the mmproj file."""
        model = self.get(section, "model", fallback_to_star=False)
        if not model:
            raise PresetError(f"Section [{section}] has no model key")
        sizes: list[int] = []
        for key in ("model", "mmproj"):
            value = self.get(section, key, fallback_to_star=False)
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_file():
                raise PresetError(f"Section [{section}]: {key} file is missing: {path}")
            sizes.append(path.stat().st_size)
        return sizes
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_preset.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/preset.py tests/unit/test_preset.py
git commit -m "feat: preset parser that preserves comments and order"
```

---

### Task 5: Preset editing and atomic save

**Files:**
- Modify: `src/local_llm/preset.py` (add methods to `Preset`)
- Test: `tests/unit/test_preset_edit.py`

**Interfaces:**
- Consumes: `Preset` (Task 4).
- Produces: `Preset.add_section(name: str, keys: Sequence[tuple[str, str]], comments: Sequence[str] = ()) -> None` (raises `PresetError` on a bad or duplicate name); `Preset.replace_section(name, keys, comments=()) -> None` (raises `PresetError` when absent); `Preset.remove_section(name: str) -> None` (raises when absent; also removes the comment lines directly above the header); `Preset.save(path: Path) -> None` (atomic; copies the previous file to `<name>.bak` first).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_preset_edit.py`:

```python
import pytest

from local_llm.preset import Preset, PresetError

BASE = "version = 1\n\n[*]\njinja = true\n\n[a]\nmodel = /m/a.gguf\nc = 4096\n"


def test_add_section_appends_with_comments_and_blank_line():
    p = Preset.parse(BASE)
    p.add_section("b", [("model", "/m/b.gguf"), ("c", "8192")], comments=["added by local-llm"])
    assert p.dump() == BASE + "\n# added by local-llm\n[b]\nmodel = /m/b.gguf\nc = 8192\n"
    assert p.sections() == ["a", "b"]
    assert p.get("b", "jinja") == "true"


def test_add_section_rejects_duplicates_and_bad_names():
    p = Preset.parse(BASE)
    with pytest.raises(PresetError, match="already exists"):
        p.add_section("a", [("model", "/x")])
    with pytest.raises(PresetError, match="Invalid section name"):
        p.add_section("bad]name", [("model", "/x")])


def test_replace_section_keeps_surroundings():
    text = BASE + "\n# old note\n[b]\nmodel = /m/b.gguf\nc = 1\n\n[c]\nmodel = /m/c.gguf\n"
    p = Preset.parse(text)
    p.replace_section("b", [("model", "/m/b2.gguf")], comments=["new note"])
    assert p.dump() == BASE + "\n# new note\n[b]\nmodel = /m/b2.gguf\n\n[c]\nmodel = /m/c.gguf\n"
    with pytest.raises(PresetError, match="No section"):
        p.replace_section("zzz", [])


def test_remove_section_takes_its_leading_comments():
    text = BASE + "\n# added by local-llm\n[b]\nmodel = /m/b.gguf\n\n[c]\nmodel = /m/c.gguf\n"
    p = Preset.parse(text)
    p.remove_section("b")
    assert p.dump() == BASE + "\n[c]\nmodel = /m/c.gguf\n"
    with pytest.raises(PresetError, match="No section"):
        p.remove_section("b")


def test_save_is_atomic_and_keeps_a_backup(tmp_path):
    target = tmp_path / "models.ini"
    target.write_text(BASE)
    p = Preset.load(target)
    p.add_section("b", [("model", "/m/b.gguf")])
    p.save(target)
    assert target.read_text().endswith("[b]\nmodel = /m/b.gguf\n")
    assert (tmp_path / "models.ini.bak").read_text() == BASE
    assert not list(tmp_path.glob("*.tmp"))


def test_save_creates_parent_directory(tmp_path):
    target = tmp_path / "deep" / "models.ini"
    p = Preset.parse("[*]\njinja = true\n")
    p.save(target)
    assert target.read_text() == "[*]\njinja = true\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_preset_edit.py -q`
Expected: FAIL with `AttributeError: 'Preset' object has no attribute 'add_section'`

- [ ] **Step 3: Add the editing methods to `Preset`** (append inside the class, after `file_sizes`)

```python
    # ------------------------------------------------------------ editing

    def _span(self, name: str) -> tuple[int, int]:
        """Index of the header line and the index just past the section."""
        start = next(
            (i for i, line in enumerate(self._lines) if line.kind == "header" and line.section == name),
            None,
        )
        if start is None:
            raise PresetError(f"No section [{name}] in the preset")
        end = next(
            (i for i in range(start + 1, len(self._lines)) if self._lines[i].kind == "header"),
            len(self._lines),
        )
        return start, end

    def _leading_comment_start(self, header_index: int) -> int:
        i = header_index
        while i > 0 and self._lines[i - 1].kind == "comment":
            i -= 1
        return i

    @staticmethod
    def _section_lines(
        name: str, keys: Sequence[tuple[str, str]], comments: Sequence[str]
    ) -> list[_Line]:
        lines = [_Line(f"# {c}", "comment", name) for c in comments]
        lines.append(_Line(f"[{name}]", "header", name))
        lines.extend(_Line(f"{k} = {v}", "kv", name, k, v) for k, v in keys)
        return lines

    def add_section(
        self, name: str, keys: Sequence[tuple[str, str]], comments: Sequence[str] = ()
    ) -> None:
        if not name.strip() or "]" in name or "\n" in name:
            raise PresetError(f"Invalid section name: {name!r}")
        if self.has_section(name):
            raise PresetError(f"Section [{name}] already exists in the preset")
        if self._lines and self._lines[-1].kind != "blank":
            self._lines.append(_Line("", "blank", self._lines[-1].section))
        self._lines.extend(self._section_lines(name, keys, comments))

    def replace_section(
        self, name: str, keys: Sequence[tuple[str, str]], comments: Sequence[str] = ()
    ) -> None:
        start, end = self._span(name)
        # Trailing blank lines belong to the layout, not to the section.
        body_end = end
        while body_end > start + 1 and self._lines[body_end - 1].kind == "blank":
            body_end -= 1
        lead = self._leading_comment_start(start)
        self._lines[lead:body_end] = self._section_lines(name, keys, comments)

    def remove_section(self, name: str) -> None:
        start, end = self._span(name)
        lead = self._leading_comment_start(start)
        del self._lines[lead:end]
        # Two blank lines now touching each other read as a hole; keep one.
        if lead > 0 and lead < len(self._lines):
            if self._lines[lead - 1].kind == "blank" and self._lines[lead].kind == "blank":
                del self._lines[lead]

    # ------------------------------------------------------------ save

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(self.dump())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_preset.py tests/unit/test_preset_edit.py -q`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/preset.py tests/unit/test_preset_edit.py
git commit -m "feat: preset editing with atomic save and backup"
```

---

### Task 6: Logs

**Files:**
- Create: `src/local_llm/logs.py`
- Test: `tests/unit/test_logs.py`

**Interfaces:**
- Consumes: `SERVICE` (Task 2).
- Produces: `new_run_log(log_dir: Path, service: str = SERVICE, now: datetime | None = None) -> Path` (creates the timestamped file, mode 600, repoints `<service>.log`); `current_log(log_dir: Path, service: str = SERVICE) -> Path | None`; `prune_logs(log_dir: Path, days: int, service: str = SERVICE, now: datetime | None = None) -> int`; `tail_lines(path: Path, count: int) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_logs.py`:

```python
import os
from datetime import datetime, timedelta

from local_llm.logs import current_log, new_run_log, prune_logs, tail_lines

T0 = datetime(2026, 8, 26, 10, 30, 5)


def test_new_run_log_creates_private_file_and_symlink(tmp_path):
    path = new_run_log(tmp_path / "logs", now=T0)
    assert path == tmp_path / "logs" / "llm-router.2026-08-26T10-30-05.log"
    assert path.is_file()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    link = tmp_path / "logs" / "llm-router.log"
    assert link.is_symlink() and link.resolve() == path.resolve()
    assert current_log(tmp_path / "logs") == path.resolve()


def test_second_run_repoints_the_symlink(tmp_path):
    new_run_log(tmp_path, now=T0)
    second = new_run_log(tmp_path, now=T0 + timedelta(minutes=1))
    assert current_log(tmp_path) == second.resolve()


def test_current_log_is_none_without_a_run(tmp_path):
    assert current_log(tmp_path) is None


def test_prune_removes_old_files_but_never_the_current_one(tmp_path):
    old = new_run_log(tmp_path, now=T0 - timedelta(days=40))
    stamp = (T0 - timedelta(days=40)).timestamp()
    os.utime(old, (stamp, stamp))
    current = new_run_log(tmp_path, now=T0)
    os.utime(current, (stamp, stamp))  # even an "old" current file survives
    assert prune_logs(tmp_path, days=30, now=T0) == 1
    assert not old.exists() and current.exists()


def test_tail_lines(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
    assert tail_lines(f, 3) == ["line 47", "line 48", "line 49"]
    assert tail_lines(f, 100)[0] == "line 0"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_logs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.logs'`

- [ ] **Step 3: Write `src/local_llm/logs.py`**

```python
"""One log file per router run, a pointer to the current one, and pruning."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from .paths import SERVICE


def new_run_log(log_dir: Path, service: str = SERVICE, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{service}.{now:%Y-%m-%dT%H-%M-%S}.log"
    path.touch()
    path.chmod(0o600)
    link = log_dir / f"{service}.log"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.name)
    except OSError:
        # No symlinks here (some filesystems, Windows without privileges): keep a pointer file.
        (log_dir / "current").write_text(str(path))
    return path


def current_log(log_dir: Path, service: str = SERVICE) -> Path | None:
    link = log_dir / f"{service}.log"
    if link.is_symlink():
        target = link.resolve()
        return target if target.is_file() else None
    pointer = log_dir / "current"
    if pointer.is_file():
        target = Path(pointer.read_text().strip())
        return target if target.is_file() else None
    return None


def prune_logs(
    log_dir: Path, days: int, service: str = SERVICE, now: datetime | None = None
) -> int:
    now = now or datetime.now()
    keep = current_log(log_dir, service)
    cutoff = now - timedelta(days=days)
    removed = 0
    for file in log_dir.glob(f"{service}.*.log"):
        if file.is_symlink():
            continue
        if keep is not None and file.resolve() == keep:
            continue
        if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
            file.unlink()
            removed += 1
    return removed


def tail_lines(path: Path, count: int) -> list[str]:
    with path.open(errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=count)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_logs.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/logs.py tests/unit/test_logs.py
git commit -m "feat: per-run log files with current pointer and pruning"
```

---

### Task 7: Estimate and total RAM

**Files:**
- Create: `src/local_llm/estimate.py`, `src/local_llm/hardware.py`
- Test: `tests/unit/test_estimate.py`

**Interfaces:**
- Produces: `GIB = 1024 ** 3`; `estimate_bytes(sizes: Iterable[int]) -> int`; `budget_bytes(total_ram: int, reserve_gb: int) -> int` (never negative); `fit(estimate: int, budget: int) -> str` returning `"comfortable"`, `"fits"`, `"too_big"`, or `"unknown"` when `budget <= 0`; `human_gb(n: int) -> str` like `"19.9 GB"`; `hardware.total_ram() -> int` (psutil).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_estimate.py`:

```python
from local_llm import hardware
from local_llm.estimate import GIB, budget_bytes, estimate_bytes, fit, human_gb


def test_estimate_matches_the_measured_rule():
    file_17_6_gb = int(17.6 * GIB)
    est = estimate_bytes([file_17_6_gb])
    assert est == file_17_6_gb * 115 // 100 + GIB
    assert 20.0 * GIB < est < 21.5 * GIB  # a 17.6 GB file settles near 19.9 GB resident; we err high


def test_estimate_sums_several_files():
    assert estimate_bytes([GIB, GIB]) == 2 * GIB * 115 // 100 + GIB


def test_budget_subtracts_reserve_and_floors_at_zero():
    assert budget_bytes(48 * GIB, 10) == 38 * GIB
    assert budget_bytes(8 * GIB, 10) == 0


def test_fit_thresholds():
    budget = 100 * GIB
    assert fit(60 * GIB, budget) == "comfortable"
    assert fit(61 * GIB, budget) == "fits"
    assert fit(100 * GIB, budget) == "fits"
    assert fit(101 * GIB, budget) == "too_big"
    assert fit(1, 0) == "unknown"


def test_human_gb():
    assert human_gb(int(19.94 * GIB)) == "19.9 GB"
    assert human_gb(0) == "0.0 GB"


def test_total_ram_is_positive_int():
    total = hardware.total_ram()
    assert isinstance(total, int) and total > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_estimate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.hardware'`

- [ ] **Step 3: Write both modules**

`src/local_llm/estimate.py`:

```python
"""How much memory a set of model files needs, and whether it fits a budget.

Weights on disk plus room for the KV cache and compute buffers. Measured
against real loads: a 17.6 GB file settles near 19.9 GB resident, a 1.1 GB
file near 2.0 GB. Deliberately errs high - refusing a load that would have fit
is cheaper than swapping.
"""

from __future__ import annotations

from collections.abc import Iterable

GIB = 1024**3


def estimate_bytes(sizes: Iterable[int]) -> int:
    return sum(sizes) * 115 // 100 + GIB


def budget_bytes(total_ram: int, reserve_gb: int) -> int:
    return max(0, total_ram - reserve_gb * GIB)


def fit(estimate: int, budget: int) -> str:
    if budget <= 0:
        return "unknown"
    if estimate * 10 <= budget * 6:
        return "comfortable"
    if estimate <= budget:
        return "fits"
    return "too_big"


def human_gb(n: int) -> str:
    return f"{n / GIB:.1f} GB"
```

`src/local_llm/hardware.py` (milestone 2 extends this file with full detection):

```python
"""Facts about this machine that decide what fits."""

from __future__ import annotations

import psutil


def total_ram() -> int:
    """Physical memory in bytes."""
    return int(psutil.virtual_memory().total)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_estimate.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/estimate.py src/local_llm/hardware.py tests/unit/test_estimate.py
git commit -m "feat: memory estimate, budget and fit classification"
```

---

### Task 8: Router discovery, arguments, preconditions, UI state

**Files:**
- Create: `src/local_llm/router.py`, `tests/unit/fakes.py`
- Test: `tests/unit/test_router_discovery.py`

**Interfaces:**
- Consumes: `Paths` (Task 2), `Settings` (Task 3), `new_run_log`/`tail_lines` (Task 6).
- Produces in `router.py`: `RouterError(Exception)`; dataclass `ProcInfo(pid: int, cmdline: list[str], rss: int = 0)`; protocol `ProcessBackend` with methods `find(name_fragment: str) -> list[ProcInfo]`, `info(pid: int) -> ProcInfo | None`, `children(pid: int) -> list[ProcInfo]`, `listening(pid: int, port: int) -> bool`, `terminate(pid: int) -> None`, `kill(pid: int) -> None`, `wait(pid: int, timeout: float) -> bool`, `spawn(args: list[str], log_path: Path) -> int`; class `PsutilBackend` implementing it; `is_llama_server(cmdline: list[str]) -> bool`; class `Router(paths, settings, *, backend=None, http=None, binary=None, sleep=time.sleep, log=lambda message: None)` with `server_args() -> list[str]`, `check_preconditions() -> None`, `pid() -> int | None`, `is_running() -> bool`, `ui_state() -> bool`, `write_ui_state(on: bool) -> None`. Constants `ASLEEP_RSS = 500 * 1024 * 1024`, `START_TIMEOUT = 15.0`, `STOP_TIMEOUT = 15.0`.
- Produces in `tests/unit/fakes.py`: `FakeProc`, `FakeBackend` (see code), `FakeHttp`.

- [ ] **Step 1: Write the fakes**

`tests/unit/fakes.py`:

```python
"""Test doubles for process handling and the router HTTP API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from local_llm.router import ProcInfo


@dataclass
class FakeProc:
    pid: int
    cmdline: list[str]
    rss: int = 0
    parent: int | None = None
    listening: set[int] = field(default_factory=set)
    alive: bool = True


class FakeBackend:
    """Implements local_llm.router.ProcessBackend against an in-memory process table."""

    def __init__(self) -> None:
        self.procs: dict[int, FakeProc] = {}
        self.next_pid = 1000
        self.spawned: list[tuple[list[str], Path]] = []
        self.terminated: list[int] = []
        self.killed: list[int] = []
        self.stubborn: set[int] = set()      # pids that ignore terminate()
        self.spawn_listening: set[int] = set()  # ports a spawned process listens on
        self.spawn_dies = False              # spawned process exits immediately
        self.children_survive_parent = False

    def add(self, pid, cmdline, rss=0, parent=None, listening=()) -> FakeProc:
        proc = FakeProc(pid, list(cmdline), rss, parent, set(listening))
        self.procs[pid] = proc
        return proc

    def _exit(self, pid: int) -> None:
        proc = self.procs.get(pid)
        if proc is None:
            return
        proc.alive = False
        if not self.children_survive_parent:
            for kid in self.procs.values():
                if kid.parent == pid:
                    kid.alive = False

    # -- ProcessBackend
    def find(self, name_fragment: str) -> list[ProcInfo]:
        return [
            ProcInfo(p.pid, p.cmdline, p.rss)
            for p in self.procs.values()
            if p.alive and p.cmdline and name_fragment in os.path.basename(p.cmdline[0])
        ]

    def info(self, pid: int) -> ProcInfo | None:
        proc = self.procs.get(pid)
        return ProcInfo(proc.pid, proc.cmdline, proc.rss) if proc and proc.alive else None

    def children(self, pid: int) -> list[ProcInfo]:
        return [ProcInfo(p.pid, p.cmdline, p.rss) for p in self.procs.values() if p.alive and p.parent == pid]

    def listening(self, pid: int, port: int) -> bool:
        proc = self.procs.get(pid)
        return bool(proc and proc.alive and port in proc.listening)

    def terminate(self, pid: int) -> None:
        self.terminated.append(pid)
        if pid not in self.stubborn:
            self._exit(pid)

    def kill(self, pid: int) -> None:
        self.killed.append(pid)
        self._exit(pid)

    def wait(self, pid: int, timeout: float) -> bool:
        proc = self.procs.get(pid)
        return not (proc and proc.alive)

    def spawn(self, args: list[str], log_path: Path) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.spawned.append((list(args), log_path))
        self.add(pid, args, listening=self.spawn_listening)
        if self.spawn_dies:
            self.procs[pid].alive = False
        return pid


class FakeHttp:
    """Callable matching Router's http hook: (method, path, body) -> dict."""

    def __init__(self, responses: dict[tuple[str, str], dict] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method: str, path: str, body: dict | None) -> dict:
        self.calls.append((method, path, body))
        return self.responses.get((method, path), {"success": True})
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_router_discovery.py`:

```python
import pytest

from local_llm.paths import Paths
from local_llm.router import Router, RouterError, is_llama_server
from local_llm.settings import Settings

from .fakes import FakeBackend


def make(tmp_path, backend=None, **settings):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.preset.write_text("[*]\njinja = true\n[m]\nmodel = /x.gguf\n")
    return paths, Router(paths, Settings(**settings), backend=backend or FakeBackend(),
                         binary="/opt/bin/llama-server", sleep=lambda s: None)


def test_is_llama_server_checks_the_program_name():
    assert is_llama_server(["/opt/homebrew/bin/llama-server", "--port", "5678"])
    assert not is_llama_server(["/usr/bin/python3", "llama-server"])
    assert not is_llama_server([])


def test_server_args_reflect_settings(tmp_path):
    paths, router = make(tmp_path, port=7000, max_models=3, ui=True, api_key="k")
    assert router.server_args() == [
        "/opt/bin/llama-server", "--host", "127.0.0.1", "--port", "7000",
        "--models-preset", str(paths.preset), "--models-max", "3", "--models-autoload",
        "--ui", "--api-key", "k",
    ]
    _, plain = make(tmp_path)
    assert "--no-ui" in plain.server_args() and "--api-key" not in plain.server_args()


def test_preconditions(tmp_path):
    paths, router = make(tmp_path)
    router.check_preconditions()  # everything present: no error
    router.binary = None
    with pytest.raises(RouterError, match="llama-server not found"):
        router.check_preconditions()
    router.binary = "/opt/bin/llama-server"
    paths.preset.unlink()
    with pytest.raises(RouterError, match="Missing model config"):
        router.check_preconditions()


def test_refuses_remote_bind_without_key(tmp_path):
    _, router = make(tmp_path, host="0.0.0.0")
    with pytest.raises(RouterError, match="Refusing to bind a non-local address"):
        router.check_preconditions()
    _, allowed = make(tmp_path, host="0.0.0.0", allow_remote=True, api_key="k")
    allowed.check_preconditions()


def test_pid_from_pidfile_when_that_process_is_our_server(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    paths, router = make(tmp_path, backend=backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    assert router.pid() == 42 and router.is_running()


def test_stale_pidfile_is_removed_and_port_scan_finds_the_router(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/usr/bin/sleep", "100"])  # pid reused by something else
    backend.add(77, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    backend.add(78, ["/opt/bin/llama-server", "--alias", "m"], parent=77, listening={40001})
    paths, router = make(tmp_path, backend=backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    assert router.pid() == 77
    assert not paths.pid_file.exists()


def test_not_running_when_nothing_listens(tmp_path):
    backend = FakeBackend()
    backend.add(78, ["/opt/bin/llama-server", "--alias", "m"], listening={40001})
    _, router = make(tmp_path, backend=backend)
    assert router.pid() is None and not router.is_running()


def test_ui_state_round_trip(tmp_path):
    _, router = make(tmp_path)
    assert router.ui_state() is False
    router.write_ui_state(True)
    assert router.ui_state() is True
    router.write_ui_state(False)
    assert router.ui_state() is False
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_router_discovery.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.router'`

- [ ] **Step 4: Write `src/local_llm/router.py`** (start, stop and the HTTP client are added in Tasks 9 and 10)

```python
"""Start, stop and inspect the llama-server router process.

All process access goes through a ProcessBackend so the logic can be tested
against an in-memory fake; PsutilBackend is the real one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .logs import new_run_log, tail_lines
from .paths import Paths
from .settings import Settings

ASLEEP_RSS = 500 * 1024 * 1024  # below this a child has released its weights
START_TIMEOUT = 15.0
STOP_TIMEOUT = 15.0
_POLL = 0.5


class RouterError(Exception):
    pass


@dataclass
class ProcInfo:
    pid: int
    cmdline: list[str]
    rss: int = 0


class ProcessBackend(Protocol):
    def find(self, name_fragment: str) -> list[ProcInfo]: ...
    def info(self, pid: int) -> ProcInfo | None: ...
    def children(self, pid: int) -> list[ProcInfo]: ...
    def listening(self, pid: int, port: int) -> bool: ...
    def terminate(self, pid: int) -> None: ...
    def kill(self, pid: int) -> None: ...
    def wait(self, pid: int, timeout: float) -> bool: ...
    def spawn(self, args: list[str], log_path: Path) -> int: ...


def is_llama_server(cmdline: list[str]) -> bool:
    return bool(cmdline) and "llama-server" in os.path.basename(cmdline[0])


class PsutilBackend:
    def find(self, name_fragment: str) -> list[ProcInfo]:
        import psutil

        found: list[ProcInfo] = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"] or []
                name = proc.info["name"] or ""
                if name_fragment in name or (cmdline and name_fragment in os.path.basename(cmdline[0])):
                    found.append(ProcInfo(proc.pid, list(cmdline)))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return found

    def info(self, pid: int) -> ProcInfo | None:
        import psutil

        try:
            proc = psutil.Process(pid)
            return ProcInfo(pid, list(proc.cmdline()), proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

    def children(self, pid: int) -> list[ProcInfo]:
        import psutil

        try:
            kids = psutil.Process(pid).children()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []
        out: list[ProcInfo] = []
        for kid in kids:
            try:
                out.append(ProcInfo(kid.pid, list(kid.cmdline()), kid.memory_info().rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return out

    def listening(self, pid: int, port: int) -> bool:
        import psutil

        try:
            proc = psutil.Process(pid)
            connections = getattr(proc, "net_connections", proc.connections)
            for conn in connections(kind="tcp"):
                if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return False

    def terminate(self, pid: int) -> None:
        import psutil

        try:
            psutil.Process(pid).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def kill(self, pid: int) -> None:
        import psutil

        try:
            psutil.Process(pid).kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def wait(self, pid: int, timeout: float) -> bool:
        import psutil

        try:
            psutil.Process(pid).wait(timeout)
            return True
        except psutil.NoSuchProcess:
            return True
        except psutil.TimeoutExpired:
            return False

    def spawn(self, args: list[str], log_path: Path) -> int:
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(
                args,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        return proc.pid


@dataclass
class ChildInfo:
    pid: int
    model: str | None
    rss: int

    @property
    def asleep(self) -> bool:
        return self.rss < ASLEEP_RSS


@dataclass
class StartResult:
    pid: int
    log_file: Path | None
    already_running: bool = False


@dataclass
class StopResult:
    was_running: bool
    orphans_cleaned: int = 0
    refused: list[str] = field(default_factory=list)


HttpFn = Callable[[str, str, dict | None], dict]


def _alias_of(cmdline: list[str]) -> str | None:
    for i, arg in enumerate(cmdline):
        if arg in ("--alias", "-a") and i + 1 < len(cmdline):
            return cmdline[i + 1]
    return None


class Router:
    def __init__(
        self,
        paths: Paths,
        settings: Settings,
        *,
        backend: ProcessBackend | None = None,
        http: HttpFn | None = None,
        binary: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.backend: ProcessBackend = backend or PsutilBackend()
        self.http: HttpFn = http or self._urllib_http
        self.binary = binary if binary is not None else shutil.which("llama-server")
        self._sleep = sleep
        self._log = log

    # ------------------------------------------------------------ arguments

    def server_args(self) -> list[str]:
        s = self.settings
        args = [
            self.binary or "llama-server",
            "--host", s.host,
            "--port", str(s.port),
            "--models-preset", str(self.paths.preset),
            "--models-max", str(s.max_models),
            "--models-autoload",
            "--ui" if s.ui else "--no-ui",
        ]
        if s.api_key:
            args += ["--api-key", s.api_key]
        return args

    def check_preconditions(self) -> None:
        if not self.binary:
            raise RouterError("llama-server not found in PATH.\nInstall it with: brew install llama.cpp")
        if not self.paths.preset.is_file():
            raise RouterError(
                f"Missing model config: {self.paths.preset}\n"
                "Every model lives in that file. Create it with: local-llm setup"
            )
        s = self.settings
        if not s.is_local and not (s.allow_remote and s.api_key):
            raise RouterError(
                f"Refusing to bind a non-local address: {s.host}\n"
                "This would expose the models to your network.\n"
                "If that is intentional, set both LOCAL_LLM_ALLOW_REMOTE=1 and LOCAL_LLM_API_KEY."
            )

    # ------------------------------------------------------------ discovery

    def _is_our_router(self, pid: int) -> bool:
        info = self.backend.info(pid)
        return info is not None and is_llama_server(info.cmdline) and self.backend.listening(pid, self.settings.port)

    def pid(self) -> int | None:
        pid_file = self.paths.pid_file
        if pid_file.is_file():
            try:
                recorded = int(pid_file.read_text().strip())
            except ValueError:
                recorded = 0
            if recorded and self._is_our_router(recorded):
                return recorded
            pid_file.unlink(missing_ok=True)  # stale
        for proc in self.backend.find("llama-server"):
            if self.backend.listening(proc.pid, self.settings.port):
                return proc.pid
        return None

    def is_running(self) -> bool:
        return self.pid() is not None

    # ------------------------------------------------------------ ui state

    def ui_state(self) -> bool:
        f = self.paths.ui_file
        return f.is_file() and f.read_text().strip() == "1"

    def write_ui_state(self, on: bool) -> None:
        self.paths.ensure_state_dirs()
        self.paths.ui_file.write_text("1\n" if on else "0\n")

    # ------------------------------------------------------------ http (Task 10)

    def _urllib_http(self, method: str, path: str, body: dict | None) -> dict:
        raise RouterError("HTTP client not implemented yet")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_router_discovery.py -q`
Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
git add src/local_llm/router.py tests/unit/fakes.py tests/unit/test_router_discovery.py
git commit -m "feat: router discovery, arguments and preconditions"
```

---

### Task 9: Router start, stop and children

**Files:**
- Modify: `src/local_llm/router.py` (add methods to `Router`)
- Test: `tests/unit/test_router_lifecycle.py`

**Interfaces:**
- Consumes: everything from Task 8; `FakeBackend`.
- Produces: `Router.start() -> StartResult` (raises `RouterError` with the log tail when the process dies or never listens); `Router.run_foreground() -> int`; `Router.stop() -> StopResult`; `Router.children() -> list[ChildInfo]`; `Router.loaded_model_names() -> list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_router_lifecycle.py`:

```python
import pytest

from local_llm.paths import Paths
from local_llm.router import Router, RouterError
from local_llm.settings import Settings

from .fakes import FakeBackend


def make(tmp_path, backend, **settings):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.preset.write_text("[*]\njinja = true\n[m]\nmodel = /x.gguf\n")
    messages: list[str] = []
    router = Router(paths, Settings(**settings), backend=backend, binary="/opt/bin/llama-server",
                    sleep=lambda s: None, log=messages.append)
    return paths, router, messages


def test_start_spawns_writes_pidfile_and_log(tmp_path):
    backend = FakeBackend()
    backend.spawn_listening = {5678}
    paths, router, _ = make(tmp_path, backend)
    result = router.start()
    assert result.already_running is False
    assert result.pid == 1000
    args, log_path = backend.spawned[0]
    assert args == router.server_args()
    assert log_path.parent == paths.log_dir and log_path.is_file()
    assert result.log_file == log_path
    assert paths.pid_file.read_text().strip() == "1000"
    assert oct(paths.pid_file.stat().st_mode & 0o777) == "0o600"
    assert paths.ui_file.read_text().strip() == "0"
    assert router.pid() == 1000


def test_start_when_already_running_does_not_spawn(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    _, router, _ = make(tmp_path, backend)
    result = router.start()
    assert result.already_running and result.pid == 42
    assert backend.spawned == []


def test_start_failure_reports_log_tail_and_clears_pidfile(tmp_path):
    backend = FakeBackend()
    backend.spawn_dies = True
    paths, router, _ = make(tmp_path, backend)
    original_spawn = backend.spawn

    def spawn_and_log(args, log_path):
        pid = original_spawn(args, log_path)
        log_path.write_text("error: failed to load preset\n")
        return pid

    backend.spawn = spawn_and_log
    with pytest.raises(RouterError, match="failed to load preset"):
        router.start()
    assert not paths.pid_file.exists()


def test_start_timeout_when_it_never_listens(tmp_path):
    backend = FakeBackend()  # spawned process stays alive but never listens
    _, router, _ = make(tmp_path, backend)
    with pytest.raises(RouterError, match="Router failed to start"):
        router.start()


def test_stop_terminates_router_and_reports_children(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.add(43, ["/opt/bin/llama-server", "--alias", "m"], parent=42)
    paths, router, messages = make(tmp_path, backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    result = router.stop()
    assert result.was_running and result.orphans_cleaned == 0 and result.refused == []
    assert backend.terminated == [42] and backend.killed == []
    assert not paths.pid_file.exists()
    assert messages == ["Stopping router pid 42"]


def test_stop_sweeps_orphaned_children(tmp_path):
    backend = FakeBackend()
    backend.children_survive_parent = True
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.add(43, ["/opt/bin/llama-server", "--alias", "m"], parent=42)
    _, router, messages = make(tmp_path, backend)
    result = router.stop()
    assert result.orphans_cleaned == 1
    assert backend.terminated == [42, 43]
    assert messages == ["Stopping router pid 42", "Stopping model server pid 43"]


def test_stop_forces_a_stubborn_process(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.stubborn.add(42)
    _, router, messages = make(tmp_path, backend)
    router.stop()
    assert backend.terminated == [42] and backend.killed == [42]
    assert messages == ["Stopping router pid 42", "  did not exit in 15s, forcing"]


def test_stop_refuses_pids_that_are_not_llama_server(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/usr/bin/python3", "server.py"])
    paths, router, _ = make(tmp_path, backend)
    paths.ensure_state_dirs()
    paths.pid_file.write_text("42\n")
    result = router.stop()
    assert result.was_running
    assert result.refused == ["pid 42: /usr/bin/python3 server.py"]
    assert backend.terminated == []


def test_stop_when_not_running(tmp_path):
    paths, router, _ = make(tmp_path, FakeBackend())
    paths.ensure_state_dirs()
    paths.pid_file.write_text("999\n")
    assert router.stop().was_running is False
    assert not paths.pid_file.exists()


def test_children_and_loaded_models(tmp_path):
    backend = FakeBackend()
    backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    backend.add(43, ["/opt/bin/llama-server", "--alias", "big"], rss=20 * 1024**3, parent=42)
    backend.add(44, ["/opt/bin/llama-server", "-a", "small"], rss=100 * 1024**2, parent=42)
    _, router, _ = make(tmp_path, backend)
    kids = router.children()
    assert [(k.pid, k.model, k.asleep) for k in kids] == [(43, "big", False), (44, "small", True)]
    assert router.loaded_model_names() == ["big", "small"]
    assert make(tmp_path, FakeBackend())[1].children() == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_router_lifecycle.py -q`
Expected: FAIL with `AttributeError: 'Router' object has no attribute 'start'`

- [ ] **Step 3: Add the lifecycle methods to `Router`** (insert before the `# --- http (Task 10)` marker)

```python
    # ------------------------------------------------------------ start

    def start(self) -> StartResult:
        self.check_preconditions()
        existing = self.pid()
        if existing is not None:
            return StartResult(existing, None, already_running=True)
        self.paths.ensure_state_dirs()
        log_file = new_run_log(self.paths.log_dir)
        pid = self.backend.spawn(self.server_args(), log_file)
        self.paths.pid_file.write_text(f"{pid}\n")
        self.paths.pid_file.chmod(0o600)
        self.write_ui_state(self.settings.ui)
        for _ in range(int(START_TIMEOUT / _POLL)):
            if self.backend.listening(pid, self.settings.port):
                return StartResult(pid, log_file)
            if self.backend.info(pid) is None:
                break
            self._sleep(_POLL)
        self.paths.pid_file.unlink(missing_ok=True)
        tail = "\n".join(tail_lines(log_file, 30))
        raise RouterError(f"Router failed to start. Last lines of the log:\n{tail}")

    def run_foreground(self) -> int:
        self.check_preconditions()
        return subprocess.call(self.server_args())

    # ------------------------------------------------------------ stop

    def stop(self) -> StopResult:
        parents: list[int] = []
        if self.paths.pid_file.is_file():
            try:
                parents.append(int(self.paths.pid_file.read_text().strip()))
            except ValueError:
                pass
        for proc in self.backend.find("llama-server"):
            if self.backend.listening(proc.pid, self.settings.port) and proc.pid not in parents:
                parents.append(proc.pid)
        parents = [p for p in parents if self.backend.info(p) is not None]
        if not parents:
            self.paths.pid_file.unlink(missing_ok=True)
            return StopResult(was_running=False)

        # Children hold the model weights. Note them before the parent dies, or
        # they become unreachable orphans still holding many GB of RAM.
        kids: list[int] = []
        for parent in parents:
            for kid in self.backend.children(parent):
                if kid.pid not in kids:
                    kids.append(kid.pid)

        result = StopResult(was_running=True)
        for parent in parents:
            self._stop_one(parent, "router", result)
        for kid in kids:
            if self.backend.info(kid) is not None:
                result.orphans_cleaned += 1
                self._stop_one(kid, "model server", result)
        self.paths.pid_file.unlink(missing_ok=True)
        return result

    def _stop_one(self, pid: int, label: str, result: StopResult) -> None:
        info = self.backend.info(pid)
        if info is None:
            return
        if not is_llama_server(info.cmdline):
            result.refused.append(f"pid {pid}: {' '.join(info.cmdline)}")
            return
        self._log(f"Stopping {label} pid {pid}")
        self.backend.terminate(pid)
        if self.backend.wait(pid, STOP_TIMEOUT):
            return
        self._log(f"  did not exit in {int(STOP_TIMEOUT)}s, forcing")
        self.backend.kill(pid)
        self.backend.wait(pid, 2.0)

    # ------------------------------------------------------------ children

    def children(self) -> list[ChildInfo]:
        pid = self.pid()
        if pid is None:
            return []
        return [ChildInfo(k.pid, _alias_of(k.cmdline), k.rss) for k in self.backend.children(pid)]

    def loaded_model_names(self) -> list[str]:
        return [child.model for child in self.children() if child.model]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_router_discovery.py tests/unit/test_router_lifecycle.py -q`
Expected: `18 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/router.py tests/unit/test_router_lifecycle.py
git commit -m "feat: router start, stop with orphan sweep, and child inspection"
```

---

### Task 10: Router HTTP client

**Files:**
- Modify: `src/local_llm/router.py` (replace the `_urllib_http` stub; add API methods)
- Test: `tests/unit/test_router_http.py`

**Interfaces:**
- Produces: `Router.health() -> dict`, `Router.list_models(reload: bool = False) -> dict`, `Router.load_model(name: str) -> dict`, `Router.unload_model(name: str) -> dict`, `Router.chat(model: str, prompt: str, max_tokens: int = 8) -> dict`. HTTP errors with a body come back as the parsed body; connection failures raise `RouterError`. Non-JSON bodies come back as `{"raw": text}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_router_http.py`:

```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from local_llm.paths import Paths
from local_llm.router import Router, RouterError
from local_llm.settings import Settings

from .fakes import FakeBackend, FakeHttp


def make(tmp_path, http=None, **settings):
    paths = Paths.from_env(env={}, home=tmp_path)
    return Router(paths, Settings(**settings), backend=FakeBackend(), http=http, binary="/x")


def test_api_methods_use_the_documented_endpoints(tmp_path):
    http = FakeHttp({("GET", "/health"): {"status": "ok"}})
    router = make(tmp_path, http=http)
    assert router.health() == {"status": "ok"}
    router.list_models()
    router.list_models(reload=True)
    router.load_model("m")
    router.unload_model("m")
    router.chat("m", "hi", max_tokens=3)
    assert http.calls == [
        ("GET", "/health", None),
        ("GET", "/models", None),
        ("GET", "/models?reload=1", None),
        ("POST", "/models/load", {"model": "m"}),
        ("POST", "/models/unload", {"model": "m"}),
        ("POST", "/v1/chat/completions",
         {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}),
    ]


class _Handler(BaseHTTPRequestHandler):
    seen: list[tuple[str, str, str | None, bytes]] = []

    def _reply(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        _Handler.seen.append(("GET", self.path, self.headers.get("Authorization"), b""))
        if self.path == "/health":
            self._reply(200, b'{"status":"ok"}')
        elif self.path == "/plain":
            self._reply(200, b"not json")
        else:
            self._reply(404, b'{"error":{"code":404,"message":"no"}}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _Handler.seen.append(("POST", self.path, self.headers.get("Authorization"), body))
        self._reply(200, b'{"success":true}')

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def server():
    _Handler.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()


def test_real_http_round_trip(tmp_path, server):
    router = make(tmp_path, port=server, api_key="k")
    assert router.health() == {"status": "ok"}
    assert router.load_model("m") == {"success": True}
    assert router.http("GET", "/missing", None) == {"error": {"code": 404, "message": "no"}}
    assert router.http("GET", "/plain", None) == {"raw": "not json"}
    get_health = _Handler.seen[0]
    post_load = _Handler.seen[1]
    assert get_health[2] == "Bearer k"
    assert json.loads(post_load[3]) == {"model": "m"}


def test_connection_refused_raises_router_error(tmp_path):
    router = make(tmp_path, port=1)  # nothing listens on port 1
    with pytest.raises(RouterError, match="Router is not answering"):
        router.health()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_router_http.py -q`
Expected: FAIL with `AttributeError: 'Router' object has no attribute 'health'`

- [ ] **Step 3: Replace the HTTP stub in `Router`**

```python
    # ------------------------------------------------------------ http

    def _urllib_http(self, method: str, path: str, body: dict | None) -> dict:
        url = f"http://{self.settings.host}:{self.settings.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if self.settings.api_key:
            request.add_header("Authorization", f"Bearer {self.settings.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                text = response.read().decode(errors="replace")
        except urllib.error.HTTPError as error:
            text = error.read().decode(errors="replace")
        except (urllib.error.URLError, OSError) as error:
            raise RouterError(
                f"Router is not answering at {url}: {error}\n"
                "  local-llm status    to see whether it is running"
            ) from None
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    def health(self) -> dict:
        return self.http("GET", "/health", None)

    def list_models(self, reload: bool = False) -> dict:
        return self.http("GET", "/models?reload=1" if reload else "/models", None)

    def load_model(self, name: str) -> dict:
        return self.http("POST", "/models/load", {"model": name})

    def unload_model(self, name: str) -> dict:
        return self.http("POST", "/models/unload", {"model": name})

    def chat(self, model: str, prompt: str, max_tokens: int = 8) -> dict:
        return self.http(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_router_http.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/router.py tests/unit/test_router_http.py
git commit -m "feat: router HTTP client for health, models, load and unload"
```

---

### Task 11: Agent environments

**Files:**
- Create: `src/local_llm/agents.py`
- Test: `tests/unit/test_agents.py`

**Interfaces:**
- Consumes: `Preset` (Task 4), `Settings` (Task 3).
- Produces: `AgentError(Exception)`; `resolve_model(model: str | None, preset: Preset, settings: Settings) -> str`; `claude_env(model: str, preset: Preset, settings: Settings) -> dict[str, str]`; `copilot_env(model: str, preset: Preset, settings: Settings, offline: bool = True) -> dict[str, str]`; `export_lines(model: str, preset: Preset, settings: Settings, preset_path: Path, shell: str = "zsh") -> str`; `exec_with_env(program: str, args: list[str], extra_env: dict[str, str]) -> NoReturn`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_agents.py`:

```python
from pathlib import Path

import pytest

from local_llm.agents import (
    AgentError, claude_env, copilot_env, exec_with_env, export_lines, resolve_model,
)
from local_llm.preset import Preset
from local_llm.settings import Settings

PRESET = Preset.parse("[*]\nc = 8192\n[big]\nmodel = /b.gguf\nc = 65536\nn-predict = 32768\n[small]\nmodel = /s.gguf\n")


def test_resolve_model_uses_default_and_validates():
    assert resolve_model(None, PRESET, Settings(default_model="big")) == "big"
    assert resolve_model("small", PRESET, Settings(default_model="big")) == "small"
    with pytest.raises(AgentError, match="No model given"):
        resolve_model(None, PRESET, Settings())
    with pytest.raises(AgentError, match="Unknown model: nope"):
        resolve_model("nope", PRESET, Settings())


def test_claude_env():
    env = claude_env("big", PRESET, Settings(port=7000))
    assert env == {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:7000",
        "ANTHROPIC_MODEL": "big",
        "ANTHROPIC_API_KEY": "dummy",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "65536",
    }
    assert claude_env("small", PRESET, Settings(api_key="k"))["ANTHROPIC_API_KEY"] == "k"
    assert claude_env("small", PRESET, Settings())["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "8192"


def test_copilot_env():
    env = copilot_env("big", PRESET, Settings(), offline=False)
    assert env == {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:5678/v1",
        "COPILOT_PROVIDER_API_KEY": "no-key",
        "COPILOT_MODEL": "big",
        "COPILOT_OFFLINE": "false",
        "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": "65536",
        "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": "32768",
    }
    small = copilot_env("small", PRESET, Settings())
    assert small["COPILOT_OFFLINE"] == "true"
    assert "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS" not in small


def test_export_lines_for_shells():
    zsh = export_lines("big", PRESET, Settings(), Path("/c/models.ini"), shell="zsh")
    assert "export OPENAI_BASE_URL=http://127.0.0.1:5678/v1\n" in zsh
    assert "export OPENAI_API_KEY=dummy\n" in zsh
    assert "export ANTHROPIC_BASE_URL=http://127.0.0.1:5678\n" in zsh
    assert "export ANTHROPIC_MODEL=big\n" in zsh
    assert "export LOCAL_LLM_OPENAI_BASE_URL=http://127.0.0.1:5678/v1\n" in zsh
    assert "export LOCAL_LLM_ANTHROPIC_BASE_URL=http://127.0.0.1:5678\n" in zsh
    assert "export LOCAL_LLM_PRESET=/c/models.ini\n" in zsh
    fish = export_lines("big", PRESET, Settings(), Path("/c/models.ini"), shell="fish")
    assert "set -gx ANTHROPIC_MODEL big\n" in fish
    quoted = export_lines("big", PRESET, Settings(api_key="a b"), Path("/c/x y.ini"))
    assert "export OPENAI_API_KEY='a b'\n" in quoted and "export LOCAL_LLM_PRESET='/c/x y.ini'\n" in quoted


def test_exec_with_env_replaces_process(monkeypatch):
    calls = []
    monkeypatch.setattr("local_llm.agents.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("local_llm.agents.os.execve", lambda path, argv, env: calls.append((path, argv, env)))
    monkeypatch.setenv("KEEP", "1")
    exec_with_env("claude", ["--resume"], {"ANTHROPIC_MODEL": "big"})
    path, argv, env = calls[0]
    assert path == "/bin/claude" and argv == ["claude", "--resume"]
    assert env["KEEP"] == "1" and env["ANTHROPIC_MODEL"] == "big"


def test_exec_with_env_reports_missing_program(monkeypatch):
    monkeypatch.setattr("local_llm.agents.shutil.which", lambda name: None)
    with pytest.raises(AgentError, match="copilot not found in PATH"):
        exec_with_env("copilot", [], {})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_agents.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.agents'`

- [ ] **Step 3: Write `src/local_llm/agents.py`**

```python
"""Point coding agents at the router: environment for claude, copilot, and any tool."""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import NoReturn

from .preset import Preset
from .settings import Settings


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
    }
    context = _context(model, preset)
    if context:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = context
    return env


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


def export_lines(
    model: str, preset: Preset, settings: Settings, preset_path: Path, shell: str = "zsh"
) -> str:
    values = {
        "LOCAL_LLM_OPENAI_BASE_URL": settings.openai_base_url,
        "LOCAL_LLM_ANTHROPIC_BASE_URL": settings.anthropic_base_url,
        "LOCAL_LLM_PRESET": str(preset_path),
        "OPENAI_BASE_URL": settings.openai_base_url,
        "OPENAI_API_KEY": settings.api_key or "dummy",
        **claude_env(model, preset, settings),
    }
    if shell == "fish":
        return "".join(f"set -gx {key} {shlex.quote(value)}\n" for key, value in values.items())
    return "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())


def exec_with_env(program: str, args: list[str], extra_env: dict[str, str]) -> NoReturn:
    path = shutil.which(program)
    if not path:
        raise AgentError(f"{program} not found in PATH.")
    env = {**os.environ, **extra_env}
    os.execve(path, [program, *args], env)
    raise AssertionError("unreachable")  # execve does not return
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_agents.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/agents.py tests/unit/test_agents.py
git commit -m "feat: agent environments for claude, copilot and env"
```

---

### Task 12: Hugging Face token status

**Files:**
- Create: `src/local_llm/hub.py`
- Test: `tests/unit/test_hub_token.py`

**Interfaces:**
- Produces: dataclass `TokenStatus(state: str, username: str | None = None)` where `state` is `"valid"`, `"invalid"`, `"absent"`, or `"unreachable"`; `token_status(timeout: float = 10.0) -> TokenStatus`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_hub_token.py`:

```python
import time

import huggingface_hub
from huggingface_hub.errors import HfHubHTTPError

from local_llm.hub import TokenStatus, token_status


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}


def _api_with(whoami):
    class Api:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return whoami()

    return Api


def test_absent_token(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert token_status() == TokenStatus("absent")


def test_valid_token(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(lambda: {"name": "nikos"}))
    assert token_status() == TokenStatus("valid", "nikos")


def test_invalid_token(monkeypatch):
    def boom():
        raise HfHubHTTPError("Invalid user token", response=_Response(401))

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(boom))
    assert token_status() == TokenStatus("invalid")


def test_unreachable_on_other_errors_and_timeouts(monkeypatch):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(offline))
    assert token_status() == TokenStatus("unreachable")

    def slow():
        time.sleep(0.3)
        return {"name": "late"}

    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(slow))
    assert token_status(timeout=0.05) == TokenStatus("unreachable")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_hub_token.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.hub'`

- [ ] **Step 3: Write `src/local_llm/hub.py`** (milestone 2 adds search and downloads to this file)

```python
"""Talking to the Hugging Face Hub. This milestone: is there a usable token?"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass


@dataclass
class TokenStatus:
    state: str  # "valid" | "invalid" | "absent" | "unreachable"
    username: str | None = None


def token_status(timeout: float = 10.0) -> TokenStatus:
    """Validate the locally stored token against the Hub, without hanging."""
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    token = huggingface_hub.get_token()
    if not token:
        return TokenStatus("absent")

    def whoami() -> dict:
        return huggingface_hub.HfApi(token=token).whoami()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(whoami)
        try:
            info = future.result(timeout=timeout)
        except FutureTimeout:
            return TokenStatus("unreachable")
        except HfHubHTTPError as error:
            response = getattr(error, "response", None)
            if response is not None and getattr(response, "status_code", None) in (401, 403):
                return TokenStatus("invalid")
            return TokenStatus("unreachable")
        except Exception:  # noqa: BLE001 - any network failure means "could not check"
            return TokenStatus("unreachable")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return TokenStatus("valid", info.get("name") if isinstance(info, dict) else None)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_hub_token.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/hub.py tests/unit/test_hub_token.py
git commit -m "feat: validate the Hugging Face token without hanging"
```

---

### Task 13: Doctor checks

**Files:**
- Create: `src/local_llm/doctor.py`
- Test: `tests/unit/test_doctor.py`

**Interfaces:**
- Consumes: `Paths`, `Settings`, `Preset`, `hub.TokenStatus`.
- Produces: dataclass `Check(name: str, status: str, detail: str, fix: str | None = None, fix_cmd: list[str] | None = None)`; dataclass `Env(system, machine, which, run, token_status, port_in_use)` with callable fields (defaults are the real functions); `run_checks(paths: Paths, settings: Settings, env: Env | None = None, router_pid: int | None = None) -> list[Check]`; `port_in_use(host: str, port: int) -> bool`; constants `BREW_INSTALL: str`, `LINUX_LLAMA_HELP: str`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_doctor.py`:

```python
import subprocess

from local_llm.doctor import BREW_INSTALL, Check, Env, run_checks
from local_llm.hub import TokenStatus
from local_llm.paths import Paths
from local_llm.settings import Settings


def fake_run(outputs):
    """Build a subprocess.run replacement answering from {(binary, flag): text}."""

    def run(args, **kwargs):
        text = outputs.get((args[0], args[1]), "")
        return subprocess.CompletedProcess(args, 0, stdout=text, stderr="")

    return run


def mac_env(**overrides):
    tools = {"brew": "/opt/homebrew/bin/brew", "llama-server": "/opt/homebrew/bin/llama-server",
             "hf": "/opt/homebrew/bin/hf", "claude": "/usr/local/bin/claude"}
    outputs = {
        ("/opt/homebrew/bin/llama-server", "--version"): "version: 0.3.0 (build 10621, commit c1d0e7a00)\n",
        ("/opt/homebrew/bin/llama-server", "--help"): "... --models-preset PATH ...\n",
        ("/opt/homebrew/bin/llama-server", "--list-devices"): "Available devices:\n  MTL0: Apple M4 Pro (38338 MiB, 38338 MiB free)\n",
    }
    env = Env(system="Darwin", machine="arm64", which=lambda name: tools.get(name), run=fake_run(outputs),
              token_status=lambda: TokenStatus("valid", "nikos"), port_in_use=lambda host, port: False)
    for key, value in overrides.items():
        setattr(env, key, value)
    return env


def by_name(checks):
    return {c.name: c for c in checks}


def healthy_paths(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True)
    (tmp_path / "m.gguf").write_bytes(b"x")
    paths.preset.write_text(f"[*]\njinja = true\n[m]\nmodel = {tmp_path}/m.gguf\n")
    return paths


def test_everything_ok_on_a_healthy_mac(tmp_path):
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=mac_env()))
    assert all(c.status == "ok" for c in checks.values()), checks
    assert "0.3.0" in checks["llama-server"].detail and "Apple M4 Pro" in checks["llama-server"].detail
    assert checks["router support"].status == "ok"
    assert checks["hf token"].detail == "logged in as nikos"
    assert checks["models.ini"].detail == "1 model(s), all files present"
    assert checks["port"].detail == "5678 is free"
    assert "claude" in checks["agents"].detail


def test_missing_brew_on_mac_is_fatal_and_optional_on_linux(tmp_path):
    no_brew = mac_env(which=lambda name: None)
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=no_brew))
    assert checks["brew"].status == "fail" and BREW_INSTALL in checks["brew"].fix
    assert checks["llama-server"].status == "fail" and BREW_INSTALL in checks["llama-server"].fix
    linux = mac_env(system="Linux", which=lambda name: None)
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=linux))
    assert checks["brew"].status == "warn"
    assert "releases" in checks["llama-server"].fix


def test_missing_llama_server_with_brew_offers_safe_fix(tmp_path):
    env = mac_env(which=lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=env))
    assert checks["llama-server"].fix_cmd == ["/opt/homebrew/bin/brew", "install", "llama.cpp"]
    assert checks["hf"].status == "warn"
    assert checks["hf"].fix_cmd == ["/opt/homebrew/bin/brew", "install", "hf"]
    assert "router support" not in checks


def test_old_llama_server_without_presets(tmp_path):
    env = mac_env(run=fake_run({("/opt/homebrew/bin/llama-server", "--help"): "no such flag\n"}))
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=env))
    assert checks["router support"].status == "fail"
    assert "December 2025" in checks["router support"].detail


def test_token_states(tmp_path):
    for status, expected_status, expected_fix in [
        (TokenStatus("absent"), "warn", "hf auth login"),
        (TokenStatus("invalid"), "warn", "hf auth login --force"),
        (TokenStatus("unreachable"), "warn", None),
    ]:
        env = mac_env(token_status=lambda s=status: s)
        check = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=env))["hf token"]
        assert check.status == expected_status and check.fix == expected_fix


def test_preset_problems(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    checks = by_name(run_checks(paths, Settings(), env=mac_env()))
    assert checks["models.ini"].status == "warn" and checks["models.ini"].fix == "local-llm setup"
    paths.config_dir.mkdir(parents=True)
    paths.preset.write_text("[a]\nmodel = /nope/a.gguf\n[b]\nc = 1\n")
    check = by_name(run_checks(paths, Settings(), env=mac_env()))["models.ini"]
    assert check.status == "fail"
    assert "[a]: /nope/a.gguf" in check.detail and "[b]: no model key" in check.detail


def test_port_states(tmp_path):
    paths = healthy_paths(tmp_path)
    busy = mac_env(port_in_use=lambda host, port: True)
    ours = by_name(run_checks(paths, Settings(), env=busy, router_pid=42))["port"]
    assert ours.status == "ok" and "pid 42" in ours.detail
    other = by_name(run_checks(paths, Settings(), env=busy))["port"]
    assert other.status == "fail" and "--port" in other.fix


def test_check_is_a_plain_record():
    assert Check("x", "ok", "fine").fix is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_doctor.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_llm.doctor'`

- [ ] **Step 3: Write `src/local_llm/doctor.py`**

```python
"""Is this machine ready? One record per check, each with the fix to run."""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

from . import hub
from .paths import Paths
from .preset import Preset, PresetError
from .settings import Settings

BREW_INSTALL = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
LINUX_LLAMA_HELP = (
    "Install llama.cpp one of these ways:\n"
    "    brew install llama.cpp                                  (Homebrew on Linux, CPU build)\n"
    "    https://github.com/ggml-org/llama.cpp/releases          (prebuilt binaries, incl. CUDA/Vulkan)\n"
    "    https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md   (build for your GPU)"
)
_TOOL_TIMEOUT = 10


@dataclass
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    fix: str | None = None  # what a person should run or read
    fix_cmd: list[str] | None = None  # a safe command `doctor --fix` may run after asking


def port_in_use(host: str, port: int) -> bool:
    address = "127.0.0.1" if host == "localhost" else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((address, port))
        except OSError:
            return True
    return False


@dataclass
class Env:
    """Everything doctor asks the machine, injectable for tests."""

    system: str = field(default_factory=platform.system)
    machine: str = field(default_factory=platform.machine)
    which: Callable[[str], str | None] = shutil.which
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    token_status: Callable[[], hub.TokenStatus] = hub.token_status
    port_in_use: Callable[[str, int], bool] = port_in_use


def _output(env: Env, binary: str, flag: str) -> str:
    try:
        result = env.run([binary, flag], capture_output=True, text=True, timeout=_TOOL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + (result.stderr or "")


def _devices(env: Env, binary: str) -> str:
    lines = [line.strip() for line in _output(env, binary, "--list-devices").splitlines()]
    devices = [line for line in lines if line and not line.startswith("Available devices")]
    return ", ".join(devices)


def run_checks(
    paths: Paths, settings: Settings, env: Env | None = None, router_pid: int | None = None
) -> list[Check]:
    env = env or Env()
    mac = env.system == "Darwin"
    checks: list[Check] = []

    # platform
    if mac or env.system == "Linux":
        checks.append(Check("platform", "ok", f"{env.system} {env.machine}"))
    else:
        checks.append(Check("platform", "warn", f"{env.system}: untested; continue at your own risk"))

    # brew
    brew = env.which("brew")
    if brew:
        checks.append(Check("brew", "ok", brew))
    elif mac:
        checks.append(Check("brew", "fail", "Homebrew not found; llama.cpp comes from Homebrew on macOS",
                            fix=f"Install it: {BREW_INSTALL}"))
    else:
        checks.append(Check("brew", "warn", "Homebrew not found (optional on Linux)"))

    # llama-server
    server = env.which("llama-server")
    if server:
        version = _output(env, server, "--version").strip().splitlines()
        detail = f"{server} - {version[0] if version else 'unknown version'}"
        devices = _devices(env, server)
        if devices:
            detail += f"; devices: {devices}"
        checks.append(Check("llama-server", "ok", detail))
        if "--models-preset" in _output(env, server, "--help"):
            checks.append(Check("router support", "ok", "--models-preset available"))
        else:
            checks.append(Check(
                "router support", "fail",
                "this llama-server has no --models-preset; router presets need a build from December 2025 or later",
                fix="brew upgrade llama.cpp" if brew else LINUX_LLAMA_HELP,
            ))
    elif brew:
        checks.append(Check("llama-server", "fail", "not found", fix="brew install llama.cpp",
                            fix_cmd=[brew, "install", "llama.cpp"]))
    else:
        checks.append(Check("llama-server", "fail", "not found",
                            fix=f"Install Homebrew first: {BREW_INSTALL}" if mac else LINUX_LLAMA_HELP))

    # hf command
    hf_cli = env.which("hf")
    uv = env.which("uv")
    if hf_cli:
        checks.append(Check("hf", "ok", hf_cli))
    elif brew:
        checks.append(Check("hf", "warn", "hf command not found (optional; downloads work without it)",
                            fix="brew install hf", fix_cmd=[brew, "install", "hf"]))
    elif uv:
        checks.append(Check("hf", "warn", "hf command not found (optional; downloads work without it)",
                            fix='uv tool install "huggingface_hub[cli]"',
                            fix_cmd=[uv, "tool", "install", "huggingface_hub[cli]"]))
    else:
        checks.append(Check("hf", "warn", "hf command not found (optional; downloads work without it)",
                            fix='pipx install "huggingface_hub[cli]"'))

    # token
    token = env.token_status()
    if token.state == "valid":
        checks.append(Check("hf token", "ok", f"logged in as {token.username}"))
    elif token.state == "invalid":
        checks.append(Check("hf token", "warn", "token present but invalid", fix="hf auth login --force"))
    elif token.state == "absent":
        checks.append(Check("hf token", "warn", "no token; gated repos will be unavailable", fix="hf auth login"))
    else:
        checks.append(Check("hf token", "warn", "could not reach huggingface.co to validate the token"))

    # config dir
    try:
        paths.config_dir.mkdir(parents=True, exist_ok=True)
        checks.append(Check("config dir", "ok", str(paths.config_dir)))
    except OSError as error:
        checks.append(Check("config dir", "fail", f"cannot create {paths.config_dir}: {error}"))

    # models.ini
    if paths.preset.is_file():
        try:
            preset = Preset.load(paths.preset)
            problems: list[str] = []
            for name in preset.sections():
                try:
                    preset.file_sizes(name)
                except PresetError as error:
                    text = str(error)
                    if "no model key" in text:
                        problems.append(f"[{name}]: no model key")
                    else:
                        problems.append(f"[{name}]: {text.split(': ')[-1]}")
            if problems:
                checks.append(Check("models.ini", "fail", "missing files - " + "; ".join(problems),
                                    fix="local-llm edit  (fix the paths) or local-llm pull <repo>"))
            else:
                checks.append(Check("models.ini", "ok", f"{len(preset.sections())} model(s), all files present"))
        except PresetError as error:
            checks.append(Check("models.ini", "fail", str(error)))
    else:
        checks.append(Check("models.ini", "warn", f"not created yet ({paths.preset})", fix="local-llm setup"))

    # state dirs
    try:
        paths.ensure_state_dirs()
        checks.append(Check("state dir", "ok", str(paths.state_dir)))
    except OSError as error:
        checks.append(Check("state dir", "fail", f"cannot create {paths.state_dir}: {error}"))

    # port
    if not env.port_in_use(settings.host, settings.port):
        checks.append(Check("port", "ok", f"{settings.port} is free"))
    elif router_pid is not None:
        checks.append(Check("port", "ok", f"{settings.port} is held by our router (pid {router_pid})"))
    else:
        checks.append(Check("port", "fail", f"{settings.port} is in use by another process",
                            fix="pick another port: local-llm --port N <command>, or port = N in settings.toml"))

    # agents
    present = [name for name in ("claude", "copilot", "opencode") if env.which(name)]
    absent = [name for name in ("claude", "copilot", "opencode") if not env.which(name)]
    detail = "found: " + (", ".join(present) or "none")
    if absent:
        detail += "; not found: " + ", ".join(absent)
    checks.append(Check("agents", "ok", detail))
    return checks
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_doctor.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/doctor.py tests/unit/test_doctor.py
git commit -m "feat: doctor checks with fixes"
```

---

### Task 14: CLI — router commands

**Files:**
- Modify: `src/local_llm/cli.py` (replace the Task 1 stub entirely)
- Test: `tests/unit/conftest.py`, `tests/unit/test_cli_router.py`

**Interfaces:**
- Consumes: everything from Tasks 2–13.
- Produces: `app` (typer application) with commands `status` (default), `up`, `down` (+ hidden alias `stop`), `restart`, `logs`, `ui`, `models` (+ hidden alias `ls`), `edit`, `prune-logs`; helpers `state() -> State`, `fail(message, code=1) -> NoReturn`, `complete_model(incomplete: str) -> list[str]`, `_split_agent_args(model: str | None, extra: list[str]) -> tuple[str | None, list[str]]`; module-level hooks tests replace: `Router`, `_sleep`, `_open_url`, `_run_editor`, `_interactive`, `total_ram`, `exec_with_env`, `run_checks`. Global options on every invocation: `--version`, `--port N`, `--host H`.

- [ ] **Step 1: Write the shared test fixture**

`tests/unit/conftest.py`:

```python
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from local_llm import cli
from local_llm.paths import Paths
from local_llm.router import Router

from .fakes import FakeBackend, FakeHttp

ENV_VARS = (
    "LOCAL_LLM_CONFIG_DIR", "LOCAL_LLM_PRESET", "LOCAL_LLM_STATE_DIR", "LOCAL_LLM_LOG_DIR",
    "LOCAL_LLM_PORT", "LOCAL_LLM_HOST", "LOCAL_LLM_MAX_MODELS", "LOCAL_LLM_RESERVE_GB",
    "LOCAL_LLM_UI", "LOCAL_LLM_DEFAULT_MODEL", "LOCAL_LLM_ALLOW_REMOTE", "LOCAL_LLM_API_KEY",
    "XDG_CONFIG_HOME", "XDG_STATE_HOME", "EDITOR", "VISUAL",
)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A CLI wired to a fake process table and a fake router API, in a scratch HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    paths = Paths.from_env(env={"HOME": str(tmp_path)}, home=tmp_path)
    paths.config_dir.mkdir(parents=True)
    (tmp_path / "big.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "small.gguf").write_bytes(b"y" * 512)
    paths.preset.write_text(
        "[*]\nc = 8192\n"
        f"[big]\nmodel = {tmp_path}/big.gguf\nc = 65536\nn-predict = 32768\n"
        f"[small]\nmodel = {tmp_path}/small.gguf\n"
    )
    backend = FakeBackend()
    http = FakeHttp({("GET", "/health"): {"status": "ok"}})
    messages: list[str] = []

    def make_router(p, s, **kwargs):
        return Router(p, s, backend=backend, http=http, binary="/opt/bin/llama-server",
                      sleep=lambda seconds: None, log=messages.append)

    monkeypatch.setattr(cli, "Router", make_router)
    monkeypatch.setattr(cli, "_sleep", lambda seconds: None)
    opened: list[str] = []
    monkeypatch.setattr(cli, "_open_url", lambda url: opened.append(url) or True)
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    cli._state = None
    runner = CliRunner()

    def run(*args):
        return runner.invoke(cli.app, list(args))

    return SimpleNamespace(paths=paths, backend=backend, http=http, run=run, tmp=tmp_path,
                           messages=messages, opened=opened, monkeypatch=monkeypatch)
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_cli_router.py`:

```python
from pathlib import Path

from local_llm import cli


def running_router(h, ui=False, children=()):
    h.backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    for pid, alias, rss in children:
        h.backend.add(pid, ["/opt/bin/llama-server", "--alias", alias], rss=rss, parent=42)
    h.paths.ensure_state_dirs()
    h.paths.pid_file.write_text("42\n")
    h.paths.ui_file.write_text("1\n" if ui else "0\n")


def test_no_arguments_shows_stopped_status(harness):
    h = harness
    result = h.run()
    assert result.exit_code == 0
    assert "state:    stopped" in result.output
    assert str(h.paths.preset) in result.output
    assert "local-llm up" in result.output


def test_version(harness):
    result = harness.run("--version")
    assert result.exit_code == 0 and result.output.startswith("local-llm ")


def test_up_starts_and_reports(harness):
    h = harness
    h.backend.spawn_listening = {5678}
    result = h.run("up")
    assert result.exit_code == 0, result.output
    assert "Router is up." in result.output
    assert "url:      http://127.0.0.1:5678/v1" in result.output
    assert "web ui" not in result.output
    args, _ = h.backend.spawned[0]
    assert args[:5] == ["/opt/bin/llama-server", "--host", "127.0.0.1", "--port", "5678"]
    assert "--no-ui" in args
    assert h.paths.pid_file.read_text().strip() == "1000"
    again = h.run("up")
    assert "Router is already running (pid 1000)." in again.output
    assert len(h.backend.spawned) == 1


def test_up_with_ui_and_port_override(harness):
    h = harness
    h.backend.spawn_listening = {7001}
    result = h.run("--port", "7001", "up", "--ui")
    assert result.exit_code == 0, result.output
    args, _ = h.backend.spawned[0]
    assert "--ui" in args and "7001" in args
    assert "web ui:   http://127.0.0.1:7001/" in result.output
    assert h.paths.ui_file.read_text().strip() == "1"


def test_up_max_models_flag_and_default(harness):
    h = harness
    h.backend.spawn_listening = {5678}
    h.run("up")
    assert h.backend.spawned[0][0][-5:-1] == ["--models-max", "1", "--models-autoload", "--no-ui"][:4] or "1" in h.backend.spawned[0][0]
    h.run("down")
    h.backend.procs.clear()
    h.run("up", "--max-models", "2")
    args, _ = h.backend.spawned[1]
    assert args[args.index("--models-max") + 1] == "2"


def test_up_failure_shows_log_tail(harness):
    h = harness
    h.backend.spawn_dies = True
    result = h.run("up")
    assert result.exit_code == 1
    assert "Router failed to start" in result.output


def test_down(harness):
    h = harness
    assert "Router is not running." in h.run("down").output
    running_router(h, children=[(43, "big", 0)])
    result = h.run("down")
    assert result.exit_code == 0
    assert "Router is down." in result.output
    assert h.messages == ["Stopping router pid 42"]
    assert not h.paths.pid_file.exists()


def test_stop_alias(harness):
    assert "Router is not running." in harness.run("stop").output


def test_status_when_running(harness):
    h = harness
    running_router(h, children=[(43, "big", 20 * 1024**3), (44, "small", 100 * 1024**2)])
    result = h.run("status")
    assert result.exit_code == 0
    assert "state:    running" in result.output and "pid:      42" in result.output
    assert 'health:   {"status":"ok"}' in result.output
    assert "web ui:   off  (local-llm restart --ui)" in result.output
    assert "big" in result.output and "20.0 GB" in result.output
    assert "small" in result.output and "(asleep)" in result.output


def test_restart_restores_loaded_models(harness):
    h = harness
    running_router(h, ui=True, children=[(43, "big", 0)])
    h.backend.spawn_listening = {5678}
    result = h.run("restart")
    assert result.exit_code == 0, result.output
    assert "Router is down." in result.output and "Router is up." in result.output
    assert "Restoring 1 model(s) that were loaded:" in result.output
    assert "  big ... ok" in result.output
    assert ("POST", "/models/load", {"model": "big"}) in h.http.calls
    assert "--ui" in h.backend.spawned[0][0]  # UI mode remembered


def test_restart_no_restore_and_no_ui(harness):
    h = harness
    running_router(h, ui=True, children=[(43, "big", 0)])
    h.backend.spawn_listening = {5678}
    result = h.run("restart", "--no-ui", "--no-restore")
    assert "Restoring" not in result.output
    assert "--no-ui" in h.backend.spawned[0][0]
    assert ("POST", "/models/load", {"model": "big"}) not in h.http.calls


def test_models_lists_names_and_sizes(harness):
    h = harness
    result = h.run("models")
    assert result.exit_code == 0
    assert f"Models in {h.paths.preset}" in result.output
    assert "big" in result.output and "small" in result.output and "0.0 GB" in result.output
    assert h.run("ls").exit_code == 0
    Path(h.tmp / "small.gguf").unlink()
    assert "missing" in h.run("models").output


def test_models_without_preset(harness):
    h = harness
    h.paths.preset.unlink()
    result = h.run("models")
    assert result.exit_code == 1 and "Missing model config" in result.output


def test_logs(harness):
    h = harness
    result = h.run("logs")
    assert result.exit_code == 1 and "No log yet" in result.output
    h.backend.spawn_listening = {5678}
    h.run("up")
    _, log_path = h.backend.spawned[0]
    log_path.write_text("line one\nline two\n")
    result = h.run("logs", "-n", "1")
    assert result.exit_code == 0
    assert "line two" in result.output and "line one" not in result.output
    assert "local-llm logs -f" in result.output


def test_prune_logs(harness):
    result = harness.run("prune-logs", "30")
    assert result.exit_code == 0
    assert "Removed 0 log file(s) older than 30 days." in result.output


def test_ui_starts_router_when_stopped(harness):
    h = harness
    h.backend.spawn_listening = {5678}
    result = h.run("ui")
    assert result.exit_code == 0, result.output
    assert "Starting it with the web UI" in result.output
    assert "--ui" in h.backend.spawned[0][0]
    assert h.opened == ["http://127.0.0.1:5678/"]


def test_ui_opens_when_already_serving_it(harness):
    h = harness
    running_router(h, ui=True)
    result = h.run("ui")
    assert result.exit_code == 0 and h.opened == ["http://127.0.0.1:5678/"]
    assert h.backend.spawned == []


def test_ui_refuses_to_restart_non_interactively(harness):
    h = harness
    running_router(h, ui=False, children=[(43, "big", 0)])
    result = h.run("ui")
    assert result.exit_code == 1
    assert "can only be enabled at startup" in result.output
    assert "  big" in result.output
    assert "Not an interactive shell" in result.output


def test_ui_yes_restarts_with_ui(harness):
    h = harness
    running_router(h, ui=False)
    h.backend.spawn_listening = {5678}
    result = h.run("ui", "-y")
    assert result.exit_code == 0, result.output
    assert "--ui" in h.backend.spawned[0][0] and h.opened == ["http://127.0.0.1:5678/"]


def test_edit_opens_the_preset(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(cli, "_run_editor", lambda path: seen.append(path) or 0)
    assert h.run("edit").exit_code == 0
    assert seen == [h.paths.preset]


def test_complete_model_reads_the_preset(harness):
    assert cli.complete_model("b") == ["big"]
    assert cli.complete_model("") == ["big", "small"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_router.py -q`
Expected: FAIL — `AttributeError: module 'local_llm.cli' has no attribute 'Router'` from the fixture.

- [ ] **Step 4: Rewrite `src/local_llm/cli.py`**

```python
"""Command-line entry point: every local-llm command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console

from . import __version__
from .agents import AgentError, claude_env, copilot_env, exec_with_env, export_lines, resolve_model
from .doctor import run_checks
from .estimate import budget_bytes, estimate_bytes, human_gb
from .hardware import total_ram
from .logs import current_log, prune_logs, tail_lines
from .paths import Paths
from .preset import Preset, PresetError
from .router import Router, RouterError
from .settings import load_settings

app = typer.Typer(
    help="One llama.cpp router serving every model in models.ini.",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode=None,
)
out = Console(highlight=False, soft_wrap=True, markup=False)
err = Console(stderr=True, highlight=False, soft_wrap=True, markup=False)

# Hooks that tests replace.
_sleep = time.sleep
_open_url = webbrowser.open


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _run_editor(path: Path) -> int:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        return subprocess.call([*editor.split(), str(path)])
    if sys.platform == "darwin":
        return subprocess.call(["open", "-t", str(path)])
    return subprocess.call(["vi", str(path)])


def fail(message: str, code: int = 1) -> NoReturn:
    err.print(message)
    raise typer.Exit(code)


class State:
    """Paths, settings and factories for one invocation."""

    def __init__(self, overrides: dict[str, object] | None = None) -> None:
        self.paths = Paths.from_env()
        self.settings = load_settings(self.paths, overrides=overrides)

    def router(self) -> Router:
        return Router(self.paths, self.settings, log=out.print)

    def preset(self) -> Preset:
        try:
            return Preset.load(self.paths.preset)
        except PresetError as error:
            fail(str(error))


_state: State | None = None


def state() -> State:
    global _state
    if _state is None:
        _state = State()
    return _state


def complete_model(incomplete: str) -> list[str]:
    try:
        names = Preset.load(Paths.from_env().preset).sections()
    except PresetError:
        return []
    return [name for name in names if name.startswith(incomplete)]


def _split_agent_args(model: str | None, extra: list[str]) -> tuple[str | None, list[str]]:
    """A first argument starting with '-' belongs to the agent, not to us."""
    if model is not None and model.startswith("-"):
        return None, [model, *extra]
    return model, list(extra)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
    port: int | None = typer.Option(None, "--port", help="Router port (overrides settings and LOCAL_LLM_PORT)."),
    host: str | None = typer.Option(None, "--host", help="Router host (overrides settings and LOCAL_LLM_HOST)."),
) -> None:
    if version:
        out.print(f"local-llm {__version__}")
        raise typer.Exit()
    global _state
    _state = State({"port": port, "host": host})
    if ctx.invoked_subcommand is None:
        status()


# ---------------------------------------------------------------- router commands


def _print_up(st: State) -> None:
    s = st.settings
    out.print("Router is up.")
    out.print(f"  url:      {s.openai_base_url}")
    if s.ui:
        out.print(f"  web ui:   http://{s.host}:{s.port}/    (local-llm ui to open it)")
    out.print("  models:   local-llm models")
    out.print("  logs:     local-llm logs -f")


def _start(st: State, router: Router) -> None:
    try:
        result = router.start()
    except RouterError as error:
        fail(str(error))
    if result.already_running:
        out.print(f"Router is already running (pid {result.pid}).")
        out.print("  local-llm status    what is loaded right now")
        return
    _print_up(st)


def _stop(router: Router) -> None:
    result = router.stop()
    for line in result.refused:
        err.print(f"Refusing to stop {line} - does not look like llama-server")
    if not result.was_running:
        out.print("Router is not running.")
        return
    if result.orphans_cleaned:
        out.print(f"Cleaned up {result.orphans_cleaned} orphaned model server(s).")
    out.print("Router is down.")


@app.command()
def up(
    foreground: bool = typer.Option(False, "-f", "--foreground", help="Stay attached to the terminal."),
    ui: bool | None = typer.Option(None, "--ui/--no-ui", help="Serve llama.cpp's web UI as well."),
    max_models: int | None = typer.Option(
        None, "--max-models", min=0, help="How many models may stay loaded at once (0 = unlimited)."
    ),
) -> None:
    """Start the router in the background and write a timestamped log."""
    st = state()
    if ui is not None:
        st.settings.ui = ui
    if max_models is not None:
        st.settings.max_models = max_models
    router = st.router()
    if foreground:
        out.print("Starting router in the foreground. Ctrl-C to stop.")
        out.print(f"  config: {st.paths.preset}")
        out.print(f"  url:    {st.settings.openai_base_url}")
        out.print()
        try:
            code = router.run_foreground()
        except RouterError as error:
            fail(str(error))
        raise typer.Exit(code)
    _start(st, router)


@app.command()
def down() -> None:
    """Stop the router and any model servers under it."""
    _stop(state().router())


app.command(name="stop", hidden=True)(down)


@app.command()
def restart(
    ui: bool | None = typer.Option(None, "--ui/--no-ui", help="Change the web UI mode."),
    restore: bool = typer.Option(True, "--restore/--no-restore", help="Reload the models that were loaded."),
    max_models: int | None = typer.Option(
        None, "--max-models", min=0, help="How many models may stay loaded at once (0 = unlimited)."
    ),
) -> None:
    """Stop and start the router, keeping the UI mode and reloading resident models."""
    st = state()
    router = st.router()
    st.settings.ui = router.ui_state() if ui is None else ui
    if max_models is not None:
        st.settings.max_models = max_models
    keep = router.loaded_model_names() if restore else []
    _stop(router)
    _sleep(1)
    _start(st, router)
    if keep:
        out.print()
        out.print(f"Restoring {len(keep)} model(s) that were loaded:")
        for name in keep:
            try:
                ok = router.load_model(name).get("success") is True
            except RouterError:
                ok = False
            out.print(f"  {name} ... {'ok' if ok else 'failed'}")


@app.command()
def status() -> None:
    """Is it running, what is resident, what to run next."""
    st = state()
    router = st.router()
    s = st.settings
    pid = router.pid()
    out.print("Local LLM router")
    out.print()
    if pid is None:
        out.print("  state:    stopped")
        out.print(f"  config:   {st.paths.preset}")
        out.print()
        out.print("  local-llm up        start it")
        out.print("  local-llm models    what it would serve")
        return
    try:
        health = json.dumps(router.health(), separators=(",", ":"))
    except RouterError:
        health = "unavailable"
    out.print("  state:    running")
    out.print(f"  pid:      {pid}")
    out.print(f"  url:      {s.openai_base_url}")
    if router.ui_state():
        out.print(f"  web ui:   http://{s.host}:{s.port}/")
    else:
        out.print("  web ui:   off  (local-llm restart --ui)")
    out.print(f"  health:   {health}")
    out.print(f"  config:   {st.paths.preset}")
    out.print()
    kids = router.children()
    if kids:
        out.print("  loaded models:")
        for kid in kids:
            note = "  (asleep)" if kid.asleep else ""
            out.print(f"    {kid.model or '?':<32} pid {kid.pid}  {human_gb(kid.rss)}{note}")
    else:
        out.print("  loaded models:  none resident")
    out.print()
    out.print("  local-llm logs -f        follow the log")
    out.print("  local-llm models         list every model in models.ini")
    out.print("  local-llm load <model>   preload one so the first prompt is fast")


@app.command()
def logs(
    follow: bool = typer.Option(False, "-f", "--follow", help="Keep printing as the log grows."),
    lines: int = typer.Option(200, "-n", "--lines", help="How many lines to show first."),
) -> None:
    """Show the last lines of the current log, or follow it."""
    st = state()
    path = current_log(st.paths.log_dir)
    if path is None:
        fail(f"No log yet in {st.paths.log_dir}\nStart the router first: local-llm up")
    for line in tail_lines(path, lines):
        out.print(line)
    if not follow:
        out.print()
        out.print("  local-llm logs -f    follow live")
        return
    with path.open(errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        try:
            while True:
                chunk = handle.readline()
                if chunk:
                    out.print(chunk.rstrip("\n"))
                else:
                    _sleep(0.5)
        except KeyboardInterrupt:
            return


def _open(url: str) -> None:
    if not _open_url(url):
        err.print(f"Could not open a browser. Visit {url}")


@app.command()
def ui(yes: bool = typer.Option(False, "-y", "--yes", help="Restart without asking if needed.")) -> None:
    """Open the web UI in a browser, starting or reconfiguring the router as needed."""
    st = state()
    router = st.router()
    url = f"http://{st.settings.host}:{st.settings.port}/"
    if router.pid() is None:
        out.print("Router is not running. Starting it with the web UI.")
        st.settings.ui = True
        _start(st, router)
        _open(url)
        return
    if router.ui_state():
        out.print(f"Opening {url}")
        _open(url)
        return
    loaded = router.loaded_model_names()
    out.print("The router is running without the web UI, and it can only be enabled at startup.")
    out.print("Restarting takes a few seconds.")
    if loaded:
        out.print("Currently loaded, and will be reloaded afterwards:")
        for name in loaded:
            out.print(f"  {name}")
    out.print()
    if not yes:
        if not _interactive():
            fail("Not an interactive shell. Run: local-llm restart --ui")
        if not typer.confirm("Restart with the web UI now?", default=False):
            out.print("Left running as it is.")
            raise typer.Exit(1)
    # Called as a plain function: pass every parameter, or typer's Option objects leak in.
    restart(ui=True, restore=True, max_models=None)
    _open(url)


@app.command(name="models")
def models_cmd() -> None:
    """List every model name in models.ini."""
    st = state()
    preset = st.preset()
    names = preset.sections()
    if not names:
        fail(f"No models defined in {st.paths.preset}")
    out.print(f"Models in {st.paths.preset}")
    out.print()
    for name in names:
        try:
            size = human_gb(sum(preset.file_sizes(name)))
        except PresetError as error:
            size = f"(missing: {str(error).split(': ')[-1]})"
        out.print(f"  {name:<44} {size}")
    out.print()
    out.print('Pass one as the "model" field of any OpenAI request - no restart needed.')
    out.print("  local-llm load <model>    load it now instead of on first use")


app.command(name="ls", hidden=True)(models_cmd)


@app.command()
def edit() -> None:
    """Open models.ini in $EDITOR."""
    raise typer.Exit(_run_editor(state().paths.preset))


@app.command(name="prune-logs")
def prune_logs_cmd(
    days: int = typer.Argument(30, help="Delete rotated logs older than this many days."),
) -> None:
    """Delete rotated logs older than N days (default 30)."""
    st = state()
    st.paths.ensure_state_dirs()
    removed = prune_logs(st.paths.log_dir, days)
    out.print(f"Removed {removed} log file(s) older than {days} days.")


# ---------------------------------------------------------------- model commands (Task 15)
```

Keep the trailing `# --- model commands (Task 15)` marker; Task 15 appends below it. The imports for `asdict`, `budget_bytes`, `estimate_bytes`, `total_ram`, `run_checks`, the agent helpers and `_split_agent_args` are used in Task 15; ruff will flag them as unused until then — that is expected, and Task 15 clears it. If the reviewer insists on a green ruff at this commit, add `# noqa: F401` to those import lines and remove the markers in Task 15.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_router.py tests/unit/test_cli_version.py -q`
Expected: `22 passed`

- [ ] **Step 6: Commit**

```bash
git add src/local_llm/cli.py tests/unit/conftest.py tests/unit/test_cli_router.py
git commit -m "feat: CLI router commands (up, down, restart, status, logs, ui, models, edit, prune-logs)"
```

---

### Task 15: CLI — load, unload, agents, env, doctor

**Files:**
- Modify: `src/local_llm/cli.py` (append below the Task 15 marker)
- Test: `tests/unit/test_cli_models.py`

**Interfaces:**
- Produces commands `load`, `unload`, `claude`, `copilot`, `env`, `doctor`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_cli_models.py`:

```python
import json

from local_llm import cli
from local_llm.doctor import Check
from local_llm.estimate import GIB


def running_router(h):
    h.backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    h.paths.ensure_state_dirs()
    h.paths.pid_file.write_text("42\n")


def test_split_agent_args():
    assert cli._split_agent_args("big", ["--resume"]) == ("big", ["--resume"])
    assert cli._split_agent_args("-p", ["hi"]) == (None, ["-p", "hi"])
    assert cli._split_agent_args(None, []) == (None, [])


def test_load_requires_running_router(harness):
    result = harness.run("load", "big")
    assert result.exit_code == 1 and "Router is not running" in result.output


def test_load_rejects_unknown_model(harness):
    running_router(harness)
    result = harness.run("load", "nope")
    assert result.exit_code == 1 and "Unknown model: nope" in result.output


def test_load_two_models_within_budget(harness):
    h = harness
    running_router(h)
    h.monkeypatch.setenv("LOCAL_LLM_MAX_MODELS", "2")
    h.monkeypatch.setattr(cli, "total_ram", lambda: 48 * GIB)
    result = h.run("load", "big", "small")
    assert result.exit_code == 0, result.output
    assert "Requested:" in result.output
    assert "estimated total:" in result.output
    assert "usable memory:    38.0 GB  (RAM minus 10 GB reserved)" in result.output
    assert 'loading big ... {"success":true}' in result.output
    assert ("POST", "/models/load", {"model": "small"}) in h.http.calls


def test_load_more_than_max_models(harness):
    h = harness
    running_router(h)
    h.monkeypatch.setenv("LOCAL_LLM_MAX_MODELS", "1")
    result = h.run("load", "big", "small")
    assert result.exit_code == 1
    assert "holds at most 1" in result.output and "local-llm restart --max-models 2" in result.output


def test_load_over_budget_refuses_unless_forced(harness):
    h = harness
    running_router(h)
    h.monkeypatch.setenv("LOCAL_LLM_MAX_MODELS", "2")
    h.monkeypatch.setattr(cli, "total_ram", lambda: 11 * GIB)  # 1 GB budget; each model needs ~1 GB
    result = h.run("load", "big", "small")
    assert result.exit_code == 1
    assert "Refusing to load: over budget by" in result.output
    assert "local-llm load big small --force" in result.output
    assert h.http.calls == []
    forced = h.run("load", "big", "small", "--force")
    assert forced.exit_code == 0, forced.output
    assert "loading anyway because --force was given" in forced.output
    assert len([c for c in h.http.calls if c[1] == "/models/load"]) == 2


def test_load_reports_missing_file(harness):
    h = harness
    running_router(h)
    (h.tmp / "big.gguf").unlink()
    result = h.run("load", "big")
    assert result.exit_code == 1 and "cannot find its model file" in result.output


def test_unload(harness):
    h = harness
    assert h.run("unload", "big").exit_code == 1
    running_router(h)
    result = h.run("unload", "big")
    assert result.exit_code == 0 and '{"success":true}' in result.output
    assert ("POST", "/models/unload", {"model": "big"}) in h.http.calls


def test_claude_execs_with_environment(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(cli, "exec_with_env", lambda program, args, env: seen.append((program, args, env)))
    result = h.run("claude", "big", "--", "--resume")
    assert result.exit_code == 0, result.output
    assert "claude -> big (context 65536) at http://127.0.0.1:5678" in result.output
    program, args, env = seen[0]
    assert program == "claude" and args == ["--resume"]
    assert env["ANTHROPIC_MODEL"] == "big" and env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5678"


def test_claude_default_model_and_passthrough_options(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(cli, "exec_with_env", lambda program, args, env: seen.append((program, args, env)))
    h.monkeypatch.setenv("LOCAL_LLM_DEFAULT_MODEL", "small")
    result = h.run("claude", "-p", "hi")
    assert result.exit_code == 0, result.output
    assert seen[0][1] == ["-p", "hi"] and seen[0][2]["ANTHROPIC_MODEL"] == "small"


def test_claude_without_a_model(harness):
    result = harness.run("claude")
    assert result.exit_code == 1 and "No model given" in result.output


def test_copilot_online_flag(harness):
    h = harness
    seen = []
    h.monkeypatch.setattr(cli, "exec_with_env", lambda program, args, env: seen.append((program, args, env)))
    result = h.run("copilot", "small", "--online", "--", "--foo")
    assert result.exit_code == 0, result.output
    program, args, env = seen[0]
    assert program == "copilot" and args == ["--foo"]
    assert env["COPILOT_OFFLINE"] == "false" and env["COPILOT_MODEL"] == "small"
    assert "copilot -> small (context 8192, offline=false)" in result.output


def test_env_prints_exports(harness):
    result = harness.run("env", "big", "--shell", "fish")
    assert result.exit_code == 0
    assert "set -gx ANTHROPIC_MODEL big\n" in result.output
    assert "set -gx OPENAI_BASE_URL http://127.0.0.1:5678/v1\n" in result.output


def test_doctor_output_and_exit_code(harness):
    h = harness
    checks = [Check("platform", "ok", "Darwin arm64"), Check("brew", "fail", "not found", fix="install it")]
    h.monkeypatch.setattr(cli, "run_checks", lambda paths, settings, router_pid=None: checks)
    result = h.run("doctor")
    assert result.exit_code == 1
    assert "ok    platform" in result.output and "FAIL  brew" in result.output
    assert "fix: install it" in result.output
    as_json = h.run("doctor", "--json")
    assert json.loads(as_json.output)[1]["fix"] == "install it"


def test_doctor_all_ok_exits_zero(harness):
    h = harness
    h.monkeypatch.setattr(cli, "run_checks", lambda paths, settings, router_pid=None: [Check("a", "ok", "fine")])
    assert h.run("doctor").exit_code == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_models.py -q`
Expected: FAIL — `AttributeError: module 'local_llm.cli' has no attribute '_split_agent_args'` is already defined, so the first failures are `No such command 'load'` (exit code 2) assertions.

- [ ] **Step 3: Append the commands to `src/local_llm/cli.py`** (below the Task 15 marker)

```python
@app.command()
def load(
    models: list[str] = typer.Argument(..., autocompletion=complete_model, help="Model names from models.ini."),
    force: bool = typer.Option(False, "--force", help="Load even if over the memory budget."),
) -> None:
    """Load one or more models now; refuses if they will not fit."""
    st = state()
    router = st.router()
    preset = st.preset()
    if router.pid() is None:
        fail("Router is not running. Start it first: local-llm up")
    for name in models:
        if name not in preset.sections():
            fail(f"Unknown model: {name}\n  local-llm models    to see what is available")
    limit = st.settings.max_models
    if len(models) > limit:
        fail(
            f"Asked for {len(models)} models but the router holds at most {limit}.\n"
            "The first would be evicted as the last loaded. Raise it with:\n"
            f"  local-llm restart --max-models {len(models)}\n"
            f"  or set LOCAL_LLM_MAX_MODELS={len(models)}, or max_models in settings.toml"
        )
    budget = budget_bytes(total_ram(), st.settings.reserve_gb)
    out.print("Requested:")
    total = 0
    for name in models:
        try:
            estimate = estimate_bytes(preset.file_sizes(name))
        except PresetError as error:
            fail(f"  {name} - cannot find its model file, refusing to guess\n  {error}")
        total += estimate
        out.print(f"  {name}  ~{human_gb(estimate)}")
    out.print()
    out.print(f"  estimated total:  {human_gb(total)}")
    if budget > 0:
        out.print(f"  usable memory:    {human_gb(budget)}  (RAM minus {st.settings.reserve_gb} GB reserved)")
    if budget > 0 and total > budget:
        out.print()
        over = human_gb(total - budget)
        if force:
            out.print(f"Over budget by {over} - loading anyway because --force was given.")
            out.print("Expect swapping.")
        else:
            fail(
                f"Refusing to load: over budget by {over}.\n\n"
                "  load them one at a time, and let the idle one sleep, or\n"
                f"  local-llm load {' '.join(models)} --force    to do it anyway"
            )
    out.print()
    for name in models:
        try:
            reply = router.load_model(name)
        except RouterError as error:
            fail(str(error))
        out.print(f"loading {name} ... {json.dumps(reply, separators=(',', ':'))}")
    out.print()
    out.print("  local-llm status    confirm what is resident")


@app.command()
def unload(model: str = typer.Argument(..., autocompletion=complete_model)) -> None:
    """Release a model immediately rather than waiting for it to go idle."""
    st = state()
    router = st.router()
    if router.pid() is None:
        fail("Router is not running.")
    try:
        reply = router.unload_model(model)
    except RouterError as error:
        fail(str(error))
    out.print(json.dumps(reply, separators=(",", ":")))


# ---------------------------------------------------------------- agents

_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


@app.command(context_settings=_PASSTHROUGH)
def claude(
    ctx: typer.Context,
    model: str | None = typer.Argument(None, autocompletion=complete_model, help="Model name; default from settings."),
) -> None:
    """Run Claude Code against the router. Arguments after -- go to claude."""
    st = state()
    preset = st.preset()
    model, extra = _split_agent_args(model, ctx.args)
    try:
        name = resolve_model(model, preset, st.settings)
        env = claude_env(name, preset, st.settings)
        context = env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "unknown")
        out.print(f"claude -> {name} (context {context}) at {st.settings.anthropic_base_url}")
        exec_with_env("claude", extra, env)
    except AgentError as error:
        fail(str(error))


@app.command(context_settings=_PASSTHROUGH)
def copilot(
    ctx: typer.Context,
    model: str | None = typer.Argument(None, autocompletion=complete_model, help="Model name; default from settings."),
    online: bool = typer.Option(False, "--online/--offline", help="Let copilot reach the network."),
) -> None:
    """Run GitHub Copilot CLI against the router. Arguments after -- go to copilot."""
    st = state()
    preset = st.preset()
    model, extra = _split_agent_args(model, ctx.args)
    try:
        name = resolve_model(model, preset, st.settings)
        env = copilot_env(name, preset, st.settings, offline=not online)
        context = env.get("COPILOT_PROVIDER_MAX_PROMPT_TOKENS", "unknown")
        out.print(f"copilot -> {name} (context {context}, offline={'false' if online else 'true'})")
        exec_with_env("copilot", extra, env)
    except AgentError as error:
        fail(str(error))


@app.command()
def env(
    model: str | None = typer.Argument(None, autocompletion=complete_model, help="Model name; default from settings."),
    shell: str = typer.Option("zsh", "--shell", help="zsh, bash or fish."),
) -> None:
    """Print export lines that point any OpenAI- or Anthropic-style tool at the router."""
    st = state()
    preset = st.preset()
    try:
        name = resolve_model(model, preset, st.settings)
    except AgentError as error:
        fail(str(error))
    out.print(export_lines(name, preset, st.settings, st.paths.preset, shell=shell), end="")


# ---------------------------------------------------------------- doctor


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Offer to run the safe fixes."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Check prerequisites, config and the port; print the fix for anything wrong."""
    st = state()
    checks = run_checks(st.paths, st.settings, router_pid=st.router().pid())
    failed = any(check.status == "fail" for check in checks)
    if json_out:
        out.print(json.dumps([asdict(check) for check in checks], indent=2))
        raise typer.Exit(1 if failed else 0)
    marks = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}
    for check in checks:
        out.print(f"  {marks[check.status]}  {check.name:<16} {check.detail}")
        if check.fix and check.status != "ok":
            out.print(f"        {'':<16} fix: {check.fix}")
    if fix:
        for check in checks:
            if check.fix_cmd and check.status != "ok":
                if typer.confirm(f"Run: {' '.join(check.fix_cmd)} ?", default=True):
                    subprocess.call(check.fix_cmd)
    raise typer.Exit(1 if failed else 0)
```

- [ ] **Step 4: Run the whole suite and the linter**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass (about 100), ruff clean (remove any `# noqa: F401` added in Task 14).

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/cli.py tests/unit/test_cli_models.py
git commit -m "feat: CLI load, unload, claude, copilot, env and doctor"
```

---

### Task 16: Acceptance on this Mac

This task runs against the author's real `~/.config/local-llm/models.ini` and real `llama-server`. Nothing is modified outside the repo. The router must not be running before this starts (`pgrep -fl llama-server` prints nothing).

**Files:**
- Create: `docs/superpowers/plans/2026-08-26-m1-acceptance.md` (the recorded outputs)

- [ ] **Step 1: Doctor and models**

Run:

```bash
uv run local-llm doctor; echo "exit=$?"
uv run local-llm models
```

Expected: every check `ok` except `hf token` (warn: token present but invalid — that is the true state of this machine) and possibly `agents`; exit=0. `models` lists the five sections (named `org/repo:QUANT`, e.g. `unsloth/Qwen3.8-27B-GGUF:Q4_K_XL`) with sizes near 22.4, 17.7, 18.8, 18.5 and 1.1 GB.

- [ ] **Step 2: Start, inspect, load, unload**

Run:

```bash
uv run local-llm up
uv run local-llm status
uv run local-llm load Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
sleep 5; uv run local-llm status
uv run local-llm unload Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
uv run local-llm logs -n 5
uv run local-llm env Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
```

Expected: `Router is up.` with the URL; status shows `running`, the pid, `health: {"status":"ok"}`; load prints the estimate (~2.2 GB), the budget (`38.0 GB`), and `{"success":true}`; the second status lists `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` with about 2 GB resident; unload returns `{"success":true}`; logs show llama-server lines; env prints eight export lines.

- [ ] **Step 3: Restart with restore, then stop**

Run:

```bash
uv run local-llm load Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
uv run local-llm restart
uv run local-llm status
uv run local-llm down
pgrep -fl llama-server; echo "leftover-check exit=$?"
```

Expected: restart prints `Router is down.`, `Router is up.`, `Restoring 1 model(s) that were loaded:` and `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M ... ok`; status shows it resident; down prints `Router is down.`; pgrep prints nothing and exits 1.

- [ ] **Step 4: Agent wrapper without launching an agent**

Run: `uv run local-llm claude Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M -- --version`
Expected: one line `claude -> Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M (context 32768) at http://127.0.0.1:5678` followed by Claude Code's own version output (the process is replaced by `claude`).

- [ ] **Step 5: Record and commit**

Write the actual outputs of steps 1–4 into `docs/superpowers/plans/2026-08-26-m1-acceptance.md` under a heading per step, noting any deviation from the expectations above. Then:

```bash
git add docs/superpowers/plans/2026-08-26-m1-acceptance.md
git commit -m "docs: milestone 1 acceptance on the author's Mac"
```

---

## Plan self-review (done at writing time)

- Spec coverage for this milestone: §4.2 doctor (Task 13, 15), §4.3 router commands (Tasks 8–10, 14, 15), §4.7 agents and `env` (Tasks 11, 15), §10 preset parser (Tasks 4, 5), §11 runtime (Tasks 8–10), §13 paths and settings (Tasks 2, 3), §14 logging (Task 6), §15 safety rules (Tasks 5, 8, 9), §16.1 unit tests (every task), §17 packaging basics (Task 1). Not in this milestone by design: §4.1 setup, §4.4–4.6, §4.8–4.9, §5–9, §12.2–12.3, §16.2–16.4, the rest of §17, §18.
- Names used across tasks were checked: `Paths.from_env`, `ensure_state_dirs`, `pid_file`, `ui_file`; `load_settings(paths, env, overrides)`; `Preset.parse/load/dump/sections/has_section/items/get/file_sizes/add_section/replace_section/remove_section/save`; `new_run_log/current_log/prune_logs/tail_lines`; `estimate_bytes/budget_bytes/fit/human_gb/GIB`; `hardware.total_ram`; `Router(paths, settings, backend=, http=, binary=, sleep=, log=)` with `server_args/check_preconditions/pid/is_running/ui_state/write_ui_state/start/run_foreground/stop/children/loaded_model_names/health/list_models/load_model/unload_model/chat`; `FakeBackend`, `FakeHttp`; `resolve_model/claude_env/copilot_env/export_lines/exec_with_env`; `hub.token_status/TokenStatus`; `doctor.Check/Env/run_checks/port_in_use`.
