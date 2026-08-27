# Coding-harness detection and configuration — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect which coding agents are installed on the machine and offer each the right kind of configuration — a provider written into the agent's own config file where the agent supports one, a launcher command otherwise — from one registry that the setup wizard, a new `local-llm integrate` menu, `doctor` and `uninstall` all read.

**Architecture:** A new `harnesses.py` holds one record per coding agent (key, title, binary to look for, kind, and for provider kinds a `configure` callable). Provider file handling lives in `integrations/<name>.py` modules that own both writing and deleting; they receive a `HarnessContext` carrying paths, settings, the model preset, and injectable prompt/print/home/env hooks, so nothing needs the CLI. Detection is `shutil.which` over the registry, injected everywhere so tests never depend on what the developer has installed.

**Tech Stack:** Python 3.11+, typer, rich, tomlkit (new), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-coding-harness-detection-design.md`

## Global Constraints

- Python `>=3.11`; line length 100; ruff lint rules `E, F, I, UP, B`; every module starts with `from __future__ import annotations`.
- Side-effecting integration functions return `list[str]` of finished, printable lines. They never print directly and never raise to the user; the caller does `for line in fn(...): out.print(line)`.
- `fail(message, code=1)` in `cli.py` is the only way a command aborts with a user-facing message.
- Path resolvers for another tool's files take `home: Path | None = None, env: Mapping[str, str] | None = None` and default to `Path.home()` / `os.environ`, so tests pass a scratch directory.
- Every question inside `setup.py` goes through the `Io` object (`say` / `ask` / `confirm` / `ask_or_default` / `confirm_or_default`), never `typer.prompt` directly.
- Writes into another tool's config file are atomic (temporary file in the same directory, then `os.replace`) and keep a `.bak` of the previous contents. A file that will not parse is never touched.
- Detection is by name on `PATH` only. Never run a harness binary to probe it.
- Never print, suggest or link a third-party proxy that patches Antigravity: per Antigravity's published terms that risks account termination.
- Commit after every task with a message in the repository's existing style (lower-case type prefix, present tense, the trailers below).

Commit trailers, on every commit:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
```

Set identity per commit (this repository uses a personal identity):

```bash
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F <file>
```

## Deviations from the spec, decided while planning

1. **Spec §4 gives each registry record a `remove` callable; this plan does not.** The integration *module* owns both writing and deleting (`codex.write` / `codex.drop`), and `uninstall.py` calls the module exactly as it already calls the opencode helpers. That keeps one place knowing what an integration consists of — the spec's actual intent — without rewriting the working, recently reviewed opencode removal path to fit a new signature.
2. **Spec §4 writes `configure(state, yes)`.** The real signature is `configure(ctx: HarnessContext)` with `ctx.yes` inside, because `State` lives in `cli.py` and importing it would make the integration modules depend on the command layer.
3. **`smallest_model` moves from `setup.py` to `preset.py`** (re-exported from `setup.py` so existing callers keep working). Both `setup.py` and the Codex integration need it, and `setup.py` will import the registry — importing it the other way would be a cycle.

## File structure

| File | Responsibility |
|---|---|
| `src/local_llm/integrations/__init__.py` | `HarnessContext`: what an integration may touch, all of it injectable |
| `src/local_llm/integrations/codex.py` | new — Codex's `config.toml`: paths, the two tables we own, status, write, drop |
| `src/local_llm/integrations/opencode.py` | gains `configure(ctx, agent)` — the plugin-and-agent logic moved out of `cli.py` |
| `src/local_llm/harnesses.py` | new — the registry, detection, menu rendering, answer parsing |
| `src/local_llm/preset.py` | gains `smallest_model(preset)` |
| `src/local_llm/agents.py` | gains `aider_env`, `aider_args`, `qwen_env`; Claude gains two variables |
| `src/local_llm/shellrc.py` | `alias_lines(shell, aliases)` takes the list; add-only merging helpers |
| `src/local_llm/setup.py` | step 6 becomes the grouped menu; `SetupContext` swaps one field for two |
| `src/local_llm/cli.py` | `integrate` menu callback, `integrate codex`, `aider`, `qwen`; wiring |
| `src/local_llm/uninstall.py` | inventory and removal of the Codex tables |
| `src/local_llm/doctor.py` | the agents check reads the registry; a stale-`base_url` check |
| `tests/unit/test_codex.py` | new |
| `tests/unit/test_harnesses.py` | new |
| `tests/unit/test_cli_harnesses.py` | new — the menu and the new launchers through the CLI |

---

### Task 1: The Codex integration module

**Files:**
- Modify: `pyproject.toml` (dependencies)
- Modify: `src/local_llm/preset.py` (add `smallest_model`)
- Modify: `src/local_llm/setup.py:66-76` (import `smallest_model` instead of defining it)
- Create: `src/local_llm/integrations/__init__.py` (replaces the empty file)
- Create: `src/local_llm/integrations/codex.py`
- Test: `tests/unit/test_codex.py`

**Interfaces:**
- Consumes: `Paths`, `Settings`, `Preset` from existing modules.
- Produces:
  - `HarnessContext(paths, settings, preset=None, home=Path.home(), env=os.environ, say=print, confirm=_no_questions, yes=False)` with method `ask(prompt: str, default: bool = True) -> bool`.
  - `preset.smallest_model(preset: Preset) -> str | None`.
  - `codex.codex_paths(home=None, env=None) -> CodexPaths` with fields `config_dir`, `config_file` and property `backup`; `codex.paths_for(config_file: Path) -> CodexPaths` for a file already located.
  - `codex.provider_table(settings: Settings) -> dict[str, str]`, `codex.profile_table(model: str) -> dict[str, str]`, `codex.chosen_model(ctx) -> str | None`.
  - `codex.status(paths, provider, profile) -> str` returning `"missing" | "same" | "different" | "unreadable"`.
  - `codex.render(provider, profile) -> str`, `codex.current_render(paths) -> str`, `codex.parse_problem(paths) -> str | None`.
  - `codex.write(paths, provider, profile) -> None`, `codex.drop(paths) -> list[str]`, `codex.has_tables(paths) -> bool`, `codex.configured_base_url(paths) -> str | None`.
  - `codex.configure(ctx) -> list[str]`, `codex.harness_status(ctx) -> str`, `codex.removal_lines(paths) -> list[str]`.
  - Constants `codex.PROVIDER_ID = "local-llm"`, `codex.WIRE_API = "responses"`, `codex.KEY_VARIABLE = "LOCAL_LLM_API_KEY"`.

- [ ] **Step 1: Add tomlkit and lock**

Edit `pyproject.toml`, adding one line to `dependencies`:

```toml
dependencies = [
  "typer>=0.15",
  "rich>=13",
  "huggingface_hub>=0.30",
  "psutil>=6",
  "tomlkit>=0.13",
]
```

Then run: `uv lock && uv sync`
Expected: `uv.lock` updated, tomlkit installed.

- [ ] **Step 2: Write the failing tests for paths, tables and status**

Create `tests/unit/test_codex.py`:

```python
from local_llm.integrations import HarnessContext
from local_llm.integrations import codex
from local_llm.preset import Preset
from local_llm.settings import Settings


def test_paths_default_and_codex_home(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    assert paths.config_dir == tmp_path / ".codex"
    assert paths.config_file == tmp_path / ".codex" / "config.toml"
    assert paths.backup == tmp_path / ".codex" / "config.toml.bak"
    moved = codex.codex_paths(home=tmp_path, env={"CODEX_HOME": str(tmp_path / "elsewhere")})
    assert moved.config_file == tmp_path / "elsewhere" / "config.toml"


def test_provider_table_omits_env_key_without_an_api_key():
    plain = codex.provider_table(Settings(port=5678, host="127.0.0.1"))
    assert plain == {
        "name": "local-llm",
        "base_url": "http://127.0.0.1:5678/v1",
        "wire_api": "responses",
    }
    keyed = codex.provider_table(Settings(api_key="secret"))
    assert keyed["env_key"] == "LOCAL_LLM_API_KEY"


def test_status_reports_missing_same_different_and_unreadable(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    provider = codex.provider_table(Settings())
    profile = codex.profile_table("small")
    assert codex.status(paths, provider, profile) == "missing"
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text('model = "gpt-5"\n')
    assert codex.status(paths, provider, profile) == "missing"
    codex.write(paths, provider, profile)
    assert codex.status(paths, provider, profile) == "same"
    assert codex.status(paths, codex.provider_table(Settings(port=9999)), profile) == "different"
    paths.config_file.write_text("[oops\n")
    assert codex.status(paths, provider, profile) == "unreadable"
    assert codex.parse_problem(paths) is not None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_codex.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_llm.integrations.codex'`

- [ ] **Step 4: Write `HarnessContext`**

Replace `src/local_llm/integrations/__init__.py` with:

```python
"""What a harness integration is allowed to touch, all of it injectable for tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import Paths
from ..preset import Preset
from ..settings import Settings


def _no_questions(prompt: str, default: bool) -> bool:
    """The default answer, used when nobody wired a real prompt."""
    return default


@dataclass
class HarnessContext:
    paths: Paths
    settings: Settings
    preset: Preset | None = None
    home: Path = field(default_factory=Path.home)
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    say: Callable[[str], None] = print
    confirm: Callable[[str, bool], bool] = _no_questions
    yes: bool = False

    def ask(self, prompt: str, default: bool = True) -> bool:
        """In yes-mode every question takes its default without being asked."""
        return default if self.yes else self.confirm(prompt, default)
```

- [ ] **Step 5: Move `smallest_model` into `preset.py`**

Append to `src/local_llm/preset.py`:

```python
def smallest_model(preset: Preset) -> str | None:
    """The section whose files are smallest on disk, or None when there are none."""
    sizes: dict[str, int] = {}
    for name in preset.sections():
        try:
            sizes[name] = sum(preset.file_sizes(name))
        except PresetError:
            continue
    if not sizes:
        return None
    return min(sizes, key=sizes.get)
```

In `src/local_llm/setup.py`, delete the local `smallest_model` definition (currently at lines 66-76) and add it to the existing preset import so every current caller keeps working:

```python
from .preset import Preset, PresetError, smallest_model
```

Run: `uv run pytest tests/unit/test_setup.py -q`
Expected: PASS — the move is behaviour-neutral.

- [ ] **Step 6: Write the Codex module**

Create `src/local_llm/integrations/codex.py`:

```python
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
EXPERIMENTAL = (
    "experimental: needs a recent llama.cpp build; tool calls may fail on older ones"
)


@dataclass(frozen=True)
class CodexPaths:
    config_dir: Path
    config_file: Path

    @property
    def backup(self) -> Path:
        return self.config_file.with_name(self.config_file.name + ".bak")


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


def _document(paths: CodexPaths):
    """The parsed file, or None when it does not exist. Raises ParseError."""
    if not paths.config_file.is_file():
        return None
    return tomlkit.parse(paths.config_file.read_text())


def _ours(doc, parent_name: str) -> dict | None:
    parent = doc.get(parent_name)
    return None if parent is None else _unwrap(parent.get(PROVIDER_ID))


def parse_problem(paths: CodexPaths) -> str | None:
    """None when the file parses (or is absent), else where it goes wrong."""
    try:
        _document(paths)
    except ParseError as error:
        return f"line {error.line}, column {error.col}: {error}"
    return None


def status(paths: CodexPaths, provider: dict, profile: dict | None) -> str:
    try:
        doc = _document(paths)
    except ParseError:
        return "unreadable"
    if doc is None or _ours(doc, "model_providers") is None:
        return "missing"
    same_provider = _ours(doc, "model_providers") == provider
    same_profile = profile is None or _ours(doc, "profiles") == profile
    return "same" if same_provider and same_profile else "different"


def has_tables(paths: CodexPaths) -> bool:
    try:
        doc = _document(paths)
    except ParseError:
        return False
    return doc is not None and any(_ours(doc, name) is not None for name in PARENTS)


def configured_base_url(paths: CodexPaths) -> str | None:
    try:
        doc = _document(paths)
    except ParseError:
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
    except ParseError:
        return ""
    provider = None if doc is None else _ours(doc, "model_providers")
    if provider is None:
        return ""
    return render(provider, _ours(doc, "profiles"))


def _atomic_write(paths: CodexPaths, text: str) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    if paths.config_file.is_file():
        shutil.copy2(paths.config_file, paths.backup)
    handle, temporary = tempfile.mkstemp(dir=paths.config_dir, prefix=".config.toml.")
    with os.fdopen(handle, "w") as stream:
        stream.write(text)
    os.replace(temporary, paths.config_file)


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
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_codex.py -q`
Expected: PASS

- [ ] **Step 8: Write the failing tests for writing, preserving and dropping**

Append to `tests/unit/test_codex.py`:

```python
EXISTING = """# my codex config
model = "gpt-5"

[model_providers.other]
name = "Other"
base_url = "https://example.invalid/v1"

[profiles.work]
model = "gpt-5"
"""


def test_write_preserves_everything_else_and_keeps_a_backup(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))
    text = paths.config_file.read_text()
    assert "# my codex config" in text
    assert '[model_providers.other]' in text and '[profiles.work]' in text
    assert '[model_providers.local-llm]' in text and 'wire_api = "responses"' in text
    assert paths.backup.read_text() == EXISTING
    assert not list(paths.config_dir.glob(".config.toml.*")), "no temporary file left behind"


def test_write_creates_the_file_when_absent(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    codex.write(paths, codex.provider_table(Settings()), None)
    assert paths.config_file.read_text().startswith("[model_providers.local-llm]")
    assert not paths.backup.exists()


def test_drop_removes_only_our_tables(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(EXISTING)
    codex.write(paths, codex.provider_table(Settings()), codex.profile_table("small"))
    removed = codex.drop(paths)
    assert removed == ["[model_providers.local-llm]", "[profiles.local-llm]"]
    text = paths.config_file.read_text()
    assert "local-llm" not in text
    assert "# my codex config" in text and "[model_providers.other]" in text
    assert codex.drop(paths) == []


def test_drop_keeps_a_parent_table_that_holds_a_comment(tmp_path):
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text(
        "[model_providers]\n# mine, keep it\n\n"
        '[model_providers.local-llm]\nname = "local-llm"\n'
    )
    assert codex.drop(paths) == ["[model_providers.local-llm]"]
    assert "# mine, keep it" in paths.config_file.read_text()
```

- [ ] **Step 9: Run them and verify they pass**

Run: `uv run pytest tests/unit/test_codex.py -q`
Expected: PASS — the module written in step 6 already covers this.

- [ ] **Step 10: Write the failing test for `configure`**

Append to `tests/unit/test_codex.py`:

```python
def make_ctx(tmp_path, *, models=("small",), yes=True, answers=None):
    from local_llm.paths import Paths

    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    body = "[*]\nc = 8192\n"
    for index, name in enumerate(models):
        blob = tmp_path / f"{name}.gguf"
        blob.write_bytes(b"x" * (100 * (index + 1)))
        body += f"[{name}]\nmodel = {blob}\n"
    paths.preset.write_text(body)
    said: list[str] = []
    replies = list(answers or [])
    ctx = HarnessContext(
        paths=paths,
        settings=Settings(),
        preset=Preset.load(paths.preset) if models else None,
        home=tmp_path,
        env={},
        say=said.append,
        confirm=lambda prompt, default: replies.pop(0) if replies else default,
        yes=yes,
    )
    ctx.said = said  # type: ignore[attr-defined]
    return ctx


def test_configure_writes_then_reports_already(tmp_path):
    ctx = make_ctx(tmp_path)
    lines = codex.configure(ctx)
    assert any("written to" in line for line in lines)
    assert any("codex --profile local-llm" in line for line in lines)
    assert any("experimental" in line for line in lines)
    paths = codex.codex_paths(home=tmp_path, env={})
    assert 'model = "small"' in paths.config_file.read_text()
    assert any("already" in line for line in codex.configure(ctx))


def test_configure_without_models_writes_the_provider_only(tmp_path):
    ctx = make_ctx(tmp_path, models=())
    lines = codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    assert "[profiles.local-llm]" not in paths.config_file.read_text()
    assert any("no model in models.ini" in line for line in lines)


def test_configure_asks_before_replacing_a_changed_table(tmp_path):
    ctx = make_ctx(tmp_path)
    codex.configure(ctx)
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_file.write_text(
        paths.config_file.read_text().replace("5678", "9999")
    )
    refusing = make_ctx(tmp_path, yes=False, answers=[False])
    assert codex.configure(refusing) == ["left as it is"]
    assert "9999" in paths.config_file.read_text()
    accepting = make_ctx(tmp_path, yes=False, answers=[True])
    codex.configure(accepting)
    assert "9999" not in paths.config_file.read_text()
    assert any("---" in said for said in accepting.said), "a difference was shown"


def test_configure_refuses_an_unparsable_file(tmp_path):
    ctx = make_ctx(tmp_path)
    paths = codex.codex_paths(home=tmp_path, env={})
    paths.config_dir.mkdir(parents=True)
    paths.config_file.write_text("[oops\n")
    lines = codex.configure(ctx)
    assert paths.config_file.read_text() == "[oops\n"
    assert any("not rewritten" in line for line in lines)
    assert any("[model_providers.local-llm]" in line for line in lines)
```

- [ ] **Step 11: Run to verify they fail**

Run: `uv run pytest tests/unit/test_codex.py -q -k configure`
Expected: FAIL — `AttributeError: module 'local_llm.integrations.codex' has no attribute 'configure'`

- [ ] **Step 12: Implement `configure`, `harness_status` and `removal_lines`**

Append to `src/local_llm/integrations/codex.py`:

```python
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
        write(paths, provider, profile)
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
    return [f"removed {', '.join(removed)} from {paths.config_file}"]
```

- [ ] **Step 13: Run the whole file plus ruff**

Run: `uv run pytest tests/unit/test_codex.py -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 14: Commit**

```bash
git add pyproject.toml uv.lock src/local_llm/integrations src/local_llm/preset.py src/local_llm/setup.py tests/unit/test_codex.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
feat(codex): write a local-llm provider and profile into Codex's config.toml

Parsed and rewritten with tomlkit so comments, key order and unrelated tables
survive, and a duplicate table can never reach Codex, which fails at startup
on one. Atomic write with a .bak; a file that does not parse is refused and
the snippet printed instead.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 2: Move opencode's configure into its own module

**Files:**
- Modify: `src/local_llm/integrations/opencode.py` (append `configure`, `harness_status`)
- Modify: `src/local_llm/cli.py:1090-1133` (`_integrate_opencode` becomes a wrapper)
- Test: `tests/unit/test_opencode.py` (append)

**Interfaces:**
- Consumes: `HarnessContext` and `preset.smallest_model` from Task 1; the existing `opencode_paths`, `plugin_status`, `install_plugin`, `plugin_source`, `tiny_agent`, `merge_agent`, `agent_snippet`, `current_tiny_model`.
- Produces: `opencode.configure(ctx: HarnessContext, *, agent: bool = True) -> list[str]`, `opencode.harness_status(ctx) -> str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_opencode.py`:

```python
def test_configure_installs_the_plugin_and_the_tiny_agent(tmp_path):
    from local_llm.integrations import HarnessContext
    from local_llm.integrations import opencode
    from local_llm.paths import Paths
    from local_llm.preset import Preset
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True)
    blob = tmp_path / "small.gguf"
    blob.write_bytes(b"x" * 10)
    paths.preset.write_text(f"[*]\nc = 8192\n[small]\nmodel = {blob}\n")
    config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True)
    (config_dir / "opencode.json").write_text("{}\n")
    ctx = HarnessContext(
        paths=paths,
        settings=Settings(),
        preset=Preset.load(paths.preset),
        home=tmp_path,
        env={},
        say=lambda line: None,
        yes=True,
    )
    lines = opencode.configure(ctx)
    assert (config_dir / "plugins" / PLUGIN_NAME).is_file()
    assert '"tiny"' in (config_dir / "opencode.json").read_text()
    assert any("plugin installed" in line for line in lines)
    assert opencode.harness_status(ctx) == "same"
    assert any("already" in line for line in opencode.configure(ctx))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_opencode.py -q -k configure`
Expected: FAIL — `AttributeError: module 'local_llm.integrations.opencode' has no attribute 'configure'`

- [ ] **Step 3: Move the logic into the module**

Append to `src/local_llm/integrations/opencode.py` (add `import difflib` and `from ..preset import smallest_model` and `from . import HarnessContext` to the imports at the top):

```python
def configure(ctx: HarnessContext, *, agent: bool = True) -> list[str]:
    """Install the plugin, and by default the tiny helper agent."""
    paths = opencode_paths(home=ctx.home, env=ctx.env)
    lines: list[str] = []
    state = plugin_status(paths)
    if state == "same":
        lines.append(f"plugin already installed: {paths.plugin}")
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
```

- [ ] **Step 4: Make `cli.py` call it**

In `src/local_llm/cli.py`, replace the whole body of `_integrate_opencode` (lines 1090-1133) with a wrapper, and add `from .integrations import HarnessContext` plus `from .integrations import codex as codex_integration` and `from .integrations import opencode as opencode_integration` to the imports:

```python
def _harness_context(st: State, *, yes: bool) -> HarnessContext:
    return HarnessContext(
        paths=st.paths,
        settings=st.settings,
        preset=_preset_or_none(st),
        home=Path.home(),
        env=os.environ,
        say=out.print,
        confirm=lambda prompt, default: typer.confirm(prompt, default=default),
        yes=yes,
    )


def _integrate_opencode(st: State, *, agent: bool, yes: bool) -> list[str]:
    return opencode_integration.configure(_harness_context(st, yes=yes), agent=agent)
```

Delete the now-unused imports from `.integrations.opencode` in `cli.py` that only `_integrate_opencode` used (`plugin_status`, `install_plugin`, `plugin_source`, `tiny_agent`, `merge_agent`, `agent_snippet`, `current_tiny_model`, `opencode_paths`) and the now-unused `difflib` import if nothing else uses it.

- [ ] **Step 5: Run the existing CLI tests unchanged**

Run: `uv run pytest tests/unit/test_opencode.py tests/unit/test_cli_setup.py -q && uv run ruff check src tests`
Expected: PASS — the three existing `integrate opencode` tests must still pass without modification.

- [ ] **Step 6: Commit**

```bash
git add src/local_llm/integrations/opencode.py src/local_llm/cli.py tests/unit/test_opencode.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
refactor(opencode): the integration owns its own configure step

Moves the plugin-and-agent logic out of the command layer and behind the same
HarnessContext the Codex integration takes, so a registry can call either one
without knowing which is which. No behaviour change.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 3: The harness registry

**Files:**
- Create: `src/local_llm/harnesses.py`
- Test: `tests/unit/test_harnesses.py`

**Interfaces:**
- Consumes: `HarnessContext`, `codex.configure`, `codex.harness_status`, `opencode.configure`, `opencode.harness_status`.
- Produces:
  - `Harness` frozen dataclass with fields `key, title, binary, kind, summary, note="", alias=None, configure=None, status=None`.
  - Constants `PROVIDER`, `LAUNCHER`, `INFORMATIONAL`; `REGISTRY: tuple[Harness, ...]`.
  - `find(key: str) -> Harness | None`.
  - `detect(which=shutil.which) -> tuple[list[Harness], list[Harness]]` returning `(installed, missing)` in registry order.
  - `numbered(installed) -> list[Harness]` — the rows a person can choose.
  - `render(installed, missing, status_of) -> list[str]` where `status_of: Callable[[Harness], str]`.
  - `parse_choice(answer: str, numbered: list[Harness]) -> list[Harness]` raising `ValueError` on anything out of range.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_harnesses.py`:

```python
import pytest

from local_llm import harnesses


def which_only(*names):
    installed = set(names)
    return lambda binary: f"/usr/local/bin/{binary}" if binary in installed else None


def test_registry_shape():
    keys = [h.key for h in harnesses.REGISTRY]
    assert keys == [
        "codex", "opencode", "claude", "copilot", "aider", "qwen", "gemini", "antigravity"
    ]
    for harness in harnesses.REGISTRY:
        assert harness.kind in (harnesses.PROVIDER, harnesses.LAUNCHER, harnesses.INFORMATIONAL)
        if harness.kind == harnesses.PROVIDER:
            assert harness.configure is not None and harness.status is not None
        if harness.kind == harnesses.INFORMATIONAL:
            assert harness.note and harness.configure is None
        if harness.kind == harnesses.LAUNCHER:
            assert harness.alias is not None
    assert harnesses.find("codex").binary == "codex"
    assert harnesses.find("nope") is None


def test_detect_splits_installed_from_missing():
    installed, missing = harnesses.detect(which_only("codex", "claude"))
    assert [h.key for h in installed] == ["codex", "claude"]
    assert "opencode" in [h.key for h in missing]


def test_numbered_excludes_informational_rows():
    installed, _ = harnesses.detect(which_only("codex", "claude", "gemini"))
    assert [h.key for h in harnesses.numbered(installed)] == ["codex", "claude"]


def test_render_groups_and_names_what_was_not_found():
    installed, missing = harnesses.detect(which_only("codex", "claude", "gemini"))
    lines = harnesses.render(installed, missing, lambda h: "missing")
    text = "\n".join(lines)
    assert "Configured inside the agent" in text and "Launched through local-llm" in text
    assert "Detected, but cannot use the router" in text
    assert "1. OpenAI Codex CLI" in text and "2. Claude Code" in text
    assert "Gemini CLI" in text and "3." not in text
    assert "Not found:" in text and "opencode" in text and "Qwen Code" in text


def test_render_says_nothing_was_found():
    installed, missing = harnesses.detect(which_only())
    text = "\n".join(harnesses.render(installed, missing, lambda h: "missing"))
    assert "No coding agents found on PATH" in text
    assert "local-llm env" in text


def test_parse_choice():
    installed, _ = harnesses.detect(which_only("codex", "claude", "aider"))
    rows = harnesses.numbered(installed)
    assert [h.key for h in harnesses.parse_choice("1 3", rows)] == ["codex", "aider"]
    assert [h.key for h in harnesses.parse_choice("a", rows)] == ["codex", "claude", "aider"]
    assert harnesses.parse_choice("n", rows) == []
    assert harnesses.parse_choice("", rows) == []
    for bad in ("0", "4", "x", "1 9"):
        with pytest.raises(ValueError):
            harnesses.parse_choice(bad, rows)


def test_the_informational_notes_stay_honest():
    gemini = harnesses.find("gemini").note
    antigravity = harnesses.find("antigravity").note
    assert "Qwen Code" in gemini and "Google" in gemini
    assert "google-antigravity" in antigravity
    for note in (gemini, antigravity):
        assert "proxy" not in note.lower(), "never point anyone at a proxy patch"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_harnesses.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_llm.harnesses'`

- [ ] **Step 3: Write the registry**

Create `src/local_llm/harnesses.py`:

```python
"""Every coding agent the tool knows: how to spot it, and what configuring it means."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from .integrations import HarnessContext
from .integrations import codex as codex_integration
from .integrations import opencode as opencode_integration

PROVIDER = "provider"
LAUNCHER = "launcher"
INFORMATIONAL = "informational"

GROUP_TITLES = {
    PROVIDER: "Configured inside the agent",
    LAUNCHER: "Launched through local-llm",
    INFORMATIONAL: "Detected, but cannot use the router",
}


@dataclass(frozen=True)
class Harness:
    key: str
    title: str
    binary: str
    kind: str
    summary: str
    note: str = ""
    alias: tuple[str, str] | None = None
    configure: Callable[[HarnessContext], list[str]] | None = None
    status: Callable[[HarnessContext], str] | None = None


GEMINI_NOTE = (
    "Gemini CLI cannot be pointed at the router: it has no setting for an"
    " OpenAI-compatible endpoint, and its GOOGLE_GEMINI_BASE_URL variable needs a"
    " server speaking Google's own request format, which llama-server does not."
    " Qwen Code continues the same codebase and does run local models:"
    " local-llm qwen"
)
ANTIGRAVITY_NOTE = (
    "Antigravity CLI is Google's replacement for Gemini CLI. It also takes only a"
    " Gemini-format endpoint, so it would need a translating server in front of the"
    " router, which this tool does not ship. To drive a local model from Antigravity,"
    " its Python SDK supports it officially: pip install google-antigravity, then"
    " LocalOpenAIAgentConfig(base_url=..., model=...)"
)

REGISTRY: tuple[Harness, ...] = (
    Harness(
        key="codex",
        title="OpenAI Codex CLI",
        binary="codex",
        kind=PROVIDER,
        summary="~/.codex/config.toml: provider and profile local-llm (experimental)",
        alias=("codex_local", "codex --profile local-llm"),
        configure=codex_integration.configure,
        status=codex_integration.harness_status,
    ),
    Harness(
        key="opencode",
        title="opencode",
        binary="opencode",
        kind=PROVIDER,
        summary="plugin listing every model, and a tiny helper agent",
        configure=partial(opencode_integration.configure, agent=True),
        status=opencode_integration.harness_status,
    ),
    Harness(
        key="claude",
        title="Claude Code",
        binary="claude",
        kind=LAUNCHER,
        summary="local-llm claude",
        alias=("claude_local", "local-llm claude"),
    ),
    Harness(
        key="copilot",
        title="GitHub Copilot CLI",
        binary="copilot",
        kind=LAUNCHER,
        summary="local-llm copilot",
        alias=("copilot_local", "local-llm copilot"),
    ),
    Harness(
        key="aider",
        title="aider",
        binary="aider",
        kind=LAUNCHER,
        summary="local-llm aider",
        alias=("aider_local", "local-llm aider"),
    ),
    Harness(
        key="qwen",
        title="Qwen Code",
        binary="qwen",
        kind=LAUNCHER,
        summary="local-llm qwen",
        alias=("qwen_local", "local-llm qwen"),
    ),
    Harness(
        key="gemini",
        title="Gemini CLI",
        binary="gemini",
        kind=INFORMATIONAL,
        summary="speaks only Google's own API format, which the router does not serve",
        note=GEMINI_NOTE,
    ),
    Harness(
        key="antigravity",
        title="Antigravity CLI",
        binary="agy",
        kind=INFORMATIONAL,
        summary="speaks only Google's own API format, which the router does not serve",
        note=ANTIGRAVITY_NOTE,
    ),
)


def find(key: str) -> Harness | None:
    return next((harness for harness in REGISTRY if harness.key == key), None)


def detect(
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[list[Harness], list[Harness]]:
    """(installed, missing), in registry order. Names on PATH only, nothing is run."""
    installed: list[Harness] = []
    missing: list[Harness] = []
    for harness in REGISTRY:
        (installed if which(harness.binary) else missing).append(harness)
    return installed, missing


def numbered(installed: Sequence[Harness]) -> list[Harness]:
    """The rows a person can actually choose."""
    return [h for h in installed if h.kind in (PROVIDER, LAUNCHER)]


def _status_note(harness: Harness, state: str) -> str:
    if harness.kind == PROVIDER:
        return {
            "same": "configured",
            "different": "configured, but differs from ours",
            "unreadable": "its config file does not parse",
        }.get(state, "not configured")
    if harness.kind == LAUNCHER:
        alias = harness.alias[0] if harness.alias else ""
        return f"alias {alias} installed" if state == "same" else f"alias {alias}"
    return ""


def render(
    installed: Sequence[Harness],
    missing: Sequence[Harness],
    status_of: Callable[[Harness], str],
) -> list[str]:
    """The whole menu as lines, ready to print. Numbering matches numbered()."""
    lines: list[str] = []
    choices = numbered(installed)
    if not installed:
        lines.append("  No coding agents found on PATH.")
        lines.append("  local-llm env prints the exports for any other tool.")
    for kind in (PROVIDER, LAUNCHER, INFORMATIONAL):
        group = [h for h in installed if h.kind == kind]
        if not group:
            continue
        lines.append("")
        lines.append(f"  {GROUP_TITLES[kind]}")
        for harness in group:
            note = _status_note(harness, status_of(harness))
            tail = f"   · {note}" if note else ""
            if kind == INFORMATIONAL:
                lines.append(f"      {harness.title:<20} {harness.summary}")
            else:
                number = choices.index(harness) + 1
                lines.append(f"   {number}. {harness.title:<20} {harness.summary}{tail}")
    if missing:
        lines.append("")
        lines.append("  Not found: " + ", ".join(h.title for h in missing))
    return lines


def parse_choice(answer: str, rows: Sequence[Harness]) -> list[Harness]:
    """'1 3' picks two, 'a' picks all, 'n' or empty picks none. Anything else raises."""
    answer = answer.strip().lower()
    if answer in ("", "n", "no", "none"):
        return []
    if answer in ("a", "all"):
        return list(rows)
    picked: list[Harness] = []
    for token in answer.split():
        if not token.isdigit():
            raise ValueError(f"Pick numbers between 1 and {len(rows)}, a for all, n for none")
        number = int(token)
        if number < 1 or number > len(rows):
            raise ValueError(f"Pick numbers between 1 and {len(rows)}, a for all, n for none")
        if rows[number - 1] not in picked:
            picked.append(rows[number - 1])
    return picked
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_harnesses.py -q && uv run ruff check src tests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/harnesses.py tests/unit/test_harnesses.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
feat(harnesses): one registry of the coding agents the tool supports

Detection by name on PATH, grouped rendering, and answer parsing, all with the
lookup injected so tests never depend on what is installed. Gemini CLI and
Antigravity CLI carry an explanation instead of a configuration step.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 4: The aider and Qwen Code launchers

**Files:**
- Modify: `src/local_llm/agents.py`
- Modify: `src/local_llm/cli.py` (two new commands next to `claude` and `copilot`)
- Test: `tests/unit/test_agents.py` (append; create if absent)
- Test: `tests/unit/test_cli_harnesses.py` (create)

**Interfaces:**
- Consumes: `resolve_model`, `_context`, `exec_with_env` from `agents.py`.
- Produces:
  - `agents.aider_env(model, preset, settings) -> dict[str, str]`
  - `agents.aider_args(model, extra) -> list[str]`
  - `agents.qwen_env(model, preset, settings) -> dict[str, str]`
  - `claude_env` additionally returns `ANTHROPIC_DEFAULT_HAIKU_MODEL` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`.
  - CLI commands `local-llm aider [MODEL] [-- ARGS...]` and `local-llm qwen [MODEL] [-- ARGS...]`.

- [ ] **Step 1: Write the failing tests**

Create or append to `tests/unit/test_agents.py`:

```python
from local_llm.agents import aider_args, aider_env, claude_env, qwen_env
from local_llm.preset import Preset
from local_llm.settings import Settings

PRESET = Preset.parse("[*]\nc = 8192\n[small]\nmodel = /tmp/small.gguf\nc = 32768\n")


def test_claude_env_sets_the_small_model_and_quietens_extra_traffic():
    env = claude_env("small", PRESET, Settings())
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5678"
    assert env["ANTHROPIC_MODEL"] == "small"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "small"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "32768"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_aider_env_and_args():
    env = aider_env(Settings())
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:5678/v1"
    assert env["OPENAI_API_KEY"] == "dummy"
    assert aider_args("small", ["--no-auto-commits"]) == [
        "--model", "openai/small", "--no-auto-commits",
    ]


def test_qwen_env_sets_all_three_or_qwen_ignores_them():
    env = qwen_env("small", Settings(api_key="secret"))
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:5678/v1"
    assert env["OPENAI_MODEL"] == "small"
    assert env["OPENAI_API_KEY"] == "secret"
    assert all(value for value in env.values()), "Qwen Code needs all three non-empty"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_agents.py -q`
Expected: FAIL — `ImportError: cannot import name 'aider_env'`

- [ ] **Step 3: Implement the environments**

In `src/local_llm/agents.py`, extend `claude_env` and append the two new builders:

```python
def claude_env(model: str, preset: Preset, settings: Settings) -> dict[str, str]:
    env = {
        "ANTHROPIC_BASE_URL": settings.anthropic_base_url,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_API_KEY": settings.api_key or "dummy",
        # Claude Code's background work asks for a Haiku-class model by name; point it
        # at the same local model or the router is asked for something it has never
        # heard of. ANTHROPIC_SMALL_FAST_MODEL is the deprecated spelling.
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        # Both react to being set at all, whatever the value. A local setup has no
        # reason to make optional network calls.
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
```

`aider_env` and `qwen_env` take fewer arguments than `claude_env` and
`copilot_env` on purpose: neither tool reads a context-window or output-token
variable, so passing the preset would be a parameter nothing uses.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_agents.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing CLI tests**

Create `tests/unit/test_cli_harnesses.py`:

```python
from local_llm import agents, cli


def capture_exec(monkeypatch):
    """Record what a launcher would have exec'd instead of replacing the process."""
    calls = []
    monkeypatch.setattr(
        cli, "exec_with_env", lambda program, args, env: calls.append((program, args, env))
    )
    return calls


def test_aider_launcher_prefixes_the_model(harness):
    calls = capture_exec(harness.monkeypatch)
    result = harness.run("aider", "small", "--", "--no-auto-commits")
    assert result.exit_code == 0, result.output
    program, args, env = calls[0]
    assert program == "aider"
    assert args == ["--model", "openai/small", "--no-auto-commits"]
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:5678/v1"
    assert "aider -> small" in result.output


def test_qwen_launcher_sets_all_three_variables(harness):
    calls = capture_exec(harness.monkeypatch)
    result = harness.run("qwen", "small")
    assert result.exit_code == 0, result.output
    _, _, env = calls[0]
    assert env["OPENAI_MODEL"] == "small" and env["OPENAI_BASE_URL"].endswith("/v1")


def test_launchers_refuse_an_unknown_model(harness):
    result = harness.run("aider", "nope")
    assert result.exit_code == 1 and "Unknown model" in result.output
```

- [ ] **Step 6: Run to verify they fail**

Run: `uv run pytest tests/unit/test_cli_harnesses.py -q`
Expected: FAIL — typer reports "No such command 'aider'"

- [ ] **Step 7: Add the commands**

In `src/local_llm/cli.py`, extend the agents import and add the two commands directly after `copilot`:

```python
from .agents import (
    AgentError,
    aider_args,
    aider_env,
    claude_env,
    copilot_env,
    exec_with_env,
    export_lines,
    qwen_env,
    resolve_model,
)
```

```python
@app.command(context_settings=_PASSTHROUGH)
def aider(
    ctx: typer.Context,
    model: str | None = typer.Argument(
        None, autocompletion=complete_model, help="Model name; default from settings."
    ),
) -> None:
    """Run aider against the router. Arguments after -- go to aider."""
    st = state()
    preset = st.preset()
    model, extra = _split_agent_args(model, ctx.args)
    try:
        name = resolve_model(model, preset, st.settings)
        env = aider_env(st.settings)
        out.print(f"aider -> {name} at {st.settings.openai_base_url}")
        exec_with_env("aider", aider_args(name, extra), env)
    except AgentError as error:
        fail(str(error))


@app.command(context_settings=_PASSTHROUGH)
def qwen(
    ctx: typer.Context,
    model: str | None = typer.Argument(
        None, autocompletion=complete_model, help="Model name; default from settings."
    ),
) -> None:
    """Run Qwen Code against the router. Arguments after -- go to qwen."""
    st = state()
    preset = st.preset()
    model, extra = _split_agent_args(model, ctx.args)
    try:
        name = resolve_model(model, preset, st.settings)
        env = qwen_env(name, st.settings)
        out.print(f"qwen -> {name} at {st.settings.openai_base_url}")
        exec_with_env("qwen", extra, env)
    except AgentError as error:
        fail(str(error))
```

- [ ] **Step 8: Run to verify they pass**

Run: `uv run pytest tests/unit/test_cli_harnesses.py tests/unit/test_agents.py -q && uv run ruff check src tests`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/local_llm/agents.py src/local_llm/cli.py tests/unit/test_agents.py tests/unit/test_cli_harnesses.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
feat(agents): launchers for aider and Qwen Code, and a Haiku model for Claude

aider needs the openai/ prefix on the model name to route through a custom
endpoint; Qwen Code ignores the endpoint unless all three variables are set.
Claude Code now also gets ANTHROPIC_DEFAULT_HAIKU_MODEL, so its background
work stops asking the router for a model it does not serve.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 5: Per-harness, add-only aliases

**Files:**
- Modify: `src/local_llm/shellrc.py`
- Modify: `src/local_llm/cli.py` (`_integrate_shell`)
- Test: `tests/unit/test_shellrc.py`

**Interfaces:**
- Consumes: `harnesses.Harness.alias`.
- Produces:
  - `shellrc.TOOL_ALIAS = ("local_llm", "local-llm")`
  - `shellrc.alias_lines(shell: str, aliases: Sequence[tuple[str, str]]) -> list[str]` — the signature changes; the old no-argument form is gone.
  - `shellrc.alias_name(line: str) -> str | None`
  - `shellrc.block_lines(text: str) -> list[str]`
  - `shellrc.merge_alias_lines(existing: list[str], wanted: list[str]) -> list[str]`

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_shellrc.py`, replace `test_alias_lines_per_shell` with:

```python
def test_alias_lines_take_the_pairs_they_are_given():
    pairs = [("local_llm", "local-llm"), ("claude_local", "local-llm claude")]
    assert alias_lines("zsh", pairs) == [
        "alias local_llm='local-llm'",
        "alias claude_local='local-llm claude'",
    ]
    assert alias_lines("fish", pairs)[0] == "alias local_llm 'local-llm'"
    assert TOOL_ALIAS == ("local_llm", "local-llm")


def test_aliases_are_add_only():
    existing = ["alias local_llm='local-llm'", "alias copilot_local='local-llm copilot'"]
    wanted = ["alias local_llm='local-llm'", "alias claude_local='local-llm claude'"]
    assert merge_alias_lines(existing, wanted) == [
        "alias local_llm='local-llm'",
        "alias copilot_local='local-llm copilot'",
        "alias claude_local='local-llm claude'",
    ]
    assert alias_name("alias claude_local='local-llm claude'") == "claude_local"
    assert alias_name("alias qwen_local 'local-llm qwen'") == "qwen_local"
    assert alias_name("export A=1") is None


def test_block_lines_reads_back_what_is_inside_the_markers():
    text = upsert_block("export A=1\n", ["alias x='y'", "alias z='w'"])
    assert block_lines(text) == ["alias x='y'", "alias z='w'"]
    assert block_lines("export A=1\n") == []
```

Add `TOOL_ALIAS`, `alias_name`, `block_lines`, `merge_alias_lines` to the import list at the top of that test file.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_shellrc.py -q`
Expected: FAIL — `ImportError: cannot import name 'TOOL_ALIAS'`

- [ ] **Step 3: Change `shellrc.py`**

Replace the `_ALIASES` constant and `alias_lines` in `src/local_llm/shellrc.py` with:

```python
TOOL_ALIAS = ("local_llm", "local-llm")
_ALIAS_NAME = re.compile(r"^\s*alias\s+([A-Za-z_][A-Za-z0-9_]*)")


def alias_lines(shell: str, aliases: Sequence[tuple[str, str]]) -> list[str]:
    if shell == "fish":
        return [f"alias {name} '{command}'" for name, command in aliases]
    return [f"alias {name}='{command}'" for name, command in aliases]


def alias_name(line: str) -> str | None:
    match = _ALIAS_NAME.match(line)
    return match.group(1) if match else None


def block_lines(text: str) -> list[str]:
    """The lines inside our marked block, or none when there is no block."""
    span = _block_span(text)
    if span is None:
        return []
    inner = text[span[0] : span[1]].splitlines()
    return [line for line in inner[1:-1]]


def merge_alias_lines(existing: list[str], wanted: list[str]) -> list[str]:
    """Add-only: an alias already there is never dropped, whatever we were asked for."""
    have = {alias_name(line) for line in existing}
    return [*existing, *(line for line in wanted if alias_name(line) not in have)]
```

Add `from collections.abc import Sequence` to the imports.

- [ ] **Step 4: Update `_integrate_shell` in `cli.py`**

```python
def _integrate_shell(
    st: State,
    shell: str,
    *,
    aliases: bool,
    yes: bool,
    extra_aliases: Sequence[tuple[str, str]] = (),
) -> list[str]:
    lines: list[str] = []
    completion_path = install_completion(shell)
    lines.append(f"{shell} completion written to {completion_path}")
    rc = rc_file(shell, Path.home())
    text = rc.read_text() if rc.is_file() else ""
    updated, retired = retire_old_source(text)
    if aliases:
        wanted = alias_lines(shell, [TOOL_ALIAS, *extra_aliases])
        merged = merge_alias_lines(block_lines(updated), wanted)
        updated = upsert_block(updated, merged)
    else:
        updated = remove_block(updated)
    if updated != text:
        if yes or typer.confirm(f"Update {rc}?", default=True):
            rc.parent.mkdir(parents=True, exist_ok=True)
            rc.write_text(updated)
            if retired:
                lines.append(f"retired {retired} old line(s) that sourced local_llm.zsh in {rc}")
            if aliases:
                names = ", ".join(name for name, _ in [TOOL_ALIAS, *extra_aliases])
                lines.append(f"aliases {names} added to {rc}")
            else:
                lines.append(f"aliases removed from {rc}")
    lines.append(f"open a new shell, or run:  source {rc}")
    return lines
```

Update the imports in `cli.py` to bring in `TOOL_ALIAS`, `block_lines`, `merge_alias_lines` and `Sequence`. Update the `--aliases` option help text on `completion install` to read `Add the local_llm alias and one per configured agent.`

- [ ] **Step 5: Fix the one existing test this changes**

`tests/unit/test_cli_setup.py::test_completion_install_writes_block_and_retires_old_line` asserts `claude_local` appears from a bare `completion install`. That is no longer true — aliases beyond `local_llm` now come from the harness menu. Change that assertion to:

```python
    assert MARK_BEGIN in text and "alias local_llm='local-llm'" in text
```

- [ ] **Step 6: Run the suite**

Run: `uv run pytest tests/unit -q && uv run ruff check src tests`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/local_llm/shellrc.py src/local_llm/cli.py tests/unit/test_shellrc.py tests/unit/test_cli_setup.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
feat(shell): one alias per configured agent, and never take one away

The alias block used to be a fixed three whatever was installed. Each harness
now owns its own alias name, and merging is add-only: an alias already in the
block survives a re-run even if that agent is gone, since only uninstall
should remove a line from someone's shell startup file.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 6: The menu, in setup and as `local-llm integrate`

**Files:**
- Modify: `src/local_llm/setup.py` (`SetupContext`, `step_integrations`)
- Modify: `src/local_llm/cli.py` (`integrate` callback, `integrate codex`, wiring)
- Test: `tests/unit/test_setup.py`, `tests/unit/test_cli_harnesses.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces:
  - `setup.SetupContext` loses `integrate_opencode`, gains `harness_status: Callable[[Harness], str]` and `configure_harness: Callable[[Harness], list[str]]`; keeps `which`.
  - `setup.choose_harnesses(io, installed, missing, status_of) -> list[Harness]`
  - `cli.integrate` with no subcommand runs the menu; `cli.integrate codex [--yes]` runs one entry.

- [ ] **Step 1: Write the failing setup test**

Append to `tests/unit/test_setup.py`:

```python
def test_step_six_offers_installed_agents_and_configures_the_chosen(tmp_path):
    from local_llm import harnesses
    from local_llm.setup import step_integrations

    script = Script(confirms=[True], asks=["1"])
    ctx = make_ctx(tmp_path, script, tools={
        "codex": "/usr/local/bin/codex",
        "claude": "/usr/local/bin/claude",
        "gemini": "/usr/local/bin/gemini",
    })
    done: list[str] = []
    ctx.configure_harness = lambda h: done.append(h.key) or [f"{h.key} done"]
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    text = script.text()
    assert "1. OpenAI Codex CLI" in text and "2. Claude Code" in text
    assert "Gemini CLI" in text and "Not found:" in text
    assert done == ["codex"] and "codex done" in text


def test_step_six_with_yes_configures_everything_installed(tmp_path):
    from local_llm.setup import step_integrations

    script = Script()
    ctx = make_ctx(tmp_path, script, yes=True, tools={"codex": "/x", "aider": "/y"})
    done: list[str] = []
    ctx.configure_harness = lambda h: done.append(h.key) or []
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    assert done == ["codex", "aider"]


def test_step_six_says_when_nothing_is_installed(tmp_path):
    from local_llm.setup import step_integrations

    script = Script(confirms=[True])
    ctx = make_ctx(tmp_path, script, tools={})
    ctx.configure_harness = lambda h: []
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    assert "No coding agents found on PATH" in script.text()
```

In `make_ctx`, drop `integrate_opencode`, add the two new `SetupContext` fields with harmless defaults, and give `integrate_shell` its second parameter:

```python
        integrate_shell=lambda shell, extra=(): integrations.append(f"shell:{shell}")
        or [f"completion for {shell} installed"],
        harness_status=lambda harness: "missing",
        configure_harness=lambda harness: [],
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_setup.py -q -k step_six`
Expected: FAIL — `TypeError: SetupContext.__init__() got an unexpected keyword argument 'harness_status'`

- [ ] **Step 3: Rewrite step 6**

In `src/local_llm/setup.py`, change the `SetupContext` fields:

```python
    integrate_shell: Callable[[str, Sequence[tuple[str, str]]], list[str]]
    harness_status: Callable[[Harness], str]
    configure_harness: Callable[[Harness], list[str]]
```

(import `from . import harnesses` and `from .harnesses import Harness`, and `from collections.abc import Sequence`), then replace `step_integrations` with:

```python
def choose_harnesses(io: Io, installed, missing, status_of) -> list[Harness]:
    """Print the grouped menu and return what was chosen. --yes takes everything."""
    for line in harnesses.render(installed, missing, status_of):
        io.say(line)
    rows = harnesses.numbered(installed)
    if not rows:
        return []
    io.say("")
    while True:
        answer = io.ask_or_default(
            "  Numbers to configure (e.g. 1 3), a for all, n for none", "a"
        )
        try:
            return harnesses.parse_choice(answer, rows)
        except ValueError as error:
            io.say(f"  {error}")


def step_integrations(ctx: SetupContext) -> None:
    io = ctx.io
    _header(io, "6. Shell and coding agents")
    from .shellrc import detect_shell

    shell = ctx.shell or detect_shell()
    installed, missing = harnesses.detect(ctx.which)
    chosen = choose_harnesses(io, installed, missing, ctx.harness_status)
    for harness in chosen:
        io.say(f"  {harness.title}")
        try:
            lines = ctx.configure_harness(harness)
        except OSError as error:  # one agent failing never stops the rest
            lines = [f"could not configure {harness.title}: {error}"]
        for line in lines:
            io.say(f"    {line}")
    for harness in installed:
        if harness.kind == harnesses.INFORMATIONAL:
            io.say("")
            io.say(f"  {harness.title}: {harness.note}")
    extra = [h.alias for h in chosen if h.alias]
    if io.confirm_or_default(
        f"Install {shell} completion and the aliases for what you chose?", True
    ):
        for line in ctx.integrate_shell(shell, extra):
            io.say(f"  {line}")
```

Note the ordering change: the harness menu now runs before the completion question, because the aliases depend on what was chosen.

- [ ] **Step 4: Run the setup tests**

Run: `uv run pytest tests/unit/test_setup.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing CLI menu tests**

Append to `tests/unit/test_cli_harnesses.py`:

```python
def fake_which(monkeypatch, *names):
    installed = set(names)
    monkeypatch.setattr(
        cli, "_which", lambda binary: f"/usr/local/bin/{binary}" if binary in installed else None
    )


def interactive(monkeypatch):
    """The shared fixture reports no terminal; these tests are about being asked."""
    monkeypatch.setattr(cli, "_interactive", lambda: True)


def test_integrate_menu_lists_and_configures(harness):
    h = harness
    fake_which(h.monkeypatch, "codex", "gemini")
    interactive(h.monkeypatch)
    result = h.run("integrate", input="1\n")
    assert result.exit_code == 0, result.output
    assert "OpenAI Codex CLI" in result.output and "Not found:" in result.output
    assert "Gemini CLI" in result.output and "google-antigravity" not in result.output
    config = h.tmp / ".codex" / "config.toml"
    assert '[model_providers.local-llm]' in config.read_text()
    assert "experimental" in result.output


def test_integrate_menu_none_writes_nothing(harness):
    h = harness
    fake_which(h.monkeypatch, "codex")
    interactive(h.monkeypatch)
    result = h.run("integrate", input="n\n")
    assert result.exit_code == 0, result.output
    assert not (h.tmp / ".codex").exists()


def test_integrate_menu_without_a_terminal_only_prints(harness):
    h = harness
    fake_which(h.monkeypatch, "codex")
    result = h.run("integrate")  # the fixture reports a non-interactive terminal
    assert result.exit_code == 0, result.output
    assert "OpenAI Codex CLI" in result.output
    assert not (h.tmp / ".codex").exists()


def test_integrate_codex_subcommand(harness):
    h = harness
    result = h.run("integrate", "codex", "--yes")
    assert result.exit_code == 0, result.output
    assert '[profiles.local-llm]' in (h.tmp / ".codex" / "config.toml").read_text()


def test_integrate_opencode_subcommand_still_works(hubbed):
    result = hubbed.run("integrate", "opencode", "--yes")
    assert result.exit_code == 0, result.output


def test_integrate_help_carries_the_explanations(harness):
    result = harness.run("integrate", "--help")
    assert result.exit_code == 0
    assert "Gemini CLI cannot be pointed at the router" in result.output
    assert "google-antigravity" in result.output
    assert "project-level" in result.output


def test_integrate_warns_when_the_router_is_not_loopback(harness):
    h = harness
    fake_which(h.monkeypatch, "codex")
    interactive(h.monkeypatch)
    h.monkeypatch.setenv("LOCAL_LLM_HOST", "0.0.0.0")
    h.monkeypatch.setenv("LOCAL_LLM_ALLOW_REMOTE", "1")
    cli._state = None
    result = h.run("integrate", input="n\n")
    assert "not a loopback address" in result.output
```

- [ ] **Step 6: Run to verify they fail**

Run: `uv run pytest tests/unit/test_cli_harnesses.py -q -k integrate`
Expected: FAIL — no `_which` attribute, and `integrate` with no subcommand prints help.

- [ ] **Step 7: Wire the CLI**

In `src/local_llm/cli.py`: add the detection hook next to the other test hooks near the top,

```python
_which = shutil.which
```

replace the `integrate_app` definition and add the callback and the Codex subcommand:

```python
INTEGRATE_HELP = f"""Wire coding agents and other tools to the router.

With no agent named, this shows every agent found on PATH, grouped by whether
it is configured inside the agent or launched through local-llm, and
configures the ones you pick.

Codex reads a project-level .codex/config.toml in preference to the one in
your home directory, so an agent that ignores the local provider inside one
repository is usually being overridden there.

{harnesses.GEMINI_NOTE}

{harnesses.ANTIGRAVITY_NOTE}
"""

integrate_app = typer.Typer(
    help=INTEGRATE_HELP,
    rich_markup_mode=None,
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(integrate_app, name="integrate")


@integrate_app.callback()
def integrate_menu(ctx: typer.Context, yes: bool = typer.Option(
    False, "-y", "--yes", help="Configure every agent found, without asking."
)) -> None:
    """With no agent named, show what is installed and configure what you pick."""
    if ctx.invoked_subcommand is not None:
        return
    st = state()
    installed, missing = harnesses.detect(_which)
    context = _harness_context(st, yes=yes)

    def status_of(harness: harnesses.Harness) -> str:
        return harness.status(context) if harness.status else "missing"

    for line in harnesses.render(installed, missing, status_of):
        out.print(line)
    if not st.settings.is_local and harnesses.numbered(installed):
        out.print("")
        out.print(
            f"  Note: {st.settings.host} is not a loopback address, so configuring an"
            " agent writes that address into its config file, where anything reading"
            " that file can see it."
        )
    rows = harnesses.numbered(installed)
    chosen: list[harnesses.Harness] = []
    if rows and yes:
        chosen = rows
    elif rows and _interactive():
        out.print("")
        answer = typer.prompt(
            "  Numbers to configure (e.g. 1 3), a for all, n for none", default="a"
        )
        try:
            chosen = harnesses.parse_choice(answer, rows)
        except ValueError as error:
            fail(str(error))
    for harness in chosen:
        out.print(f"  {harness.title}")
        try:
            lines = harness.configure(context) if harness.configure else _launcher_lines(harness)
        except OSError as error:
            lines = [f"could not configure {harness.title}: {error}"]
        for line in lines:
            out.print(f"    {line}")
    for harness in installed:
        if harness.kind == harnesses.INFORMATIONAL:
            out.print("")
            out.print(f"  {harness.title}: {harness.note}")
    extra = [h.alias for h in chosen if h.alias]
    if extra and (yes or _interactive()):
        shell = detect_shell()
        for line in _integrate_shell(st, shell, aliases=True, yes=yes, extra_aliases=extra):
            out.print(f"  {line}")


def _launcher_lines(harness: harnesses.Harness) -> list[str]:
    """A launcher writes nothing; it only reports how to run it."""
    alias = f", or the alias {harness.alias[0]}" if harness.alias else ""
    return [f"run it with:  {harness.summary}{alias}"]


@integrate_app.command("codex")
def integrate_codex_cmd(
    yes: bool = typer.Option(False, "-y", "--yes", help="Do not ask."),
) -> None:
    """Write the local-llm provider and profile into Codex's config.toml."""
    for line in codex_integration.configure(_harness_context(state(), yes=yes)):
        out.print(line)
```

Then wire the wizard: in the `setup` command's `SetupContext(...)`, replace `integrate_opencode=...` with

```python
        integrate_shell=lambda shell, extra=(): _integrate_shell(
            st, shell, aliases=True, yes=True, extra_aliases=extra
        ),
        harness_status=lambda harness: (
            harness.status(_harness_context(st, yes=yes)) if harness.status else "missing"
        ),
        configure_harness=lambda harness: (
            harness.configure(_harness_context(st, yes=yes))
            if harness.configure
            else _launcher_lines(harness)
        ),
```

and add `import shutil` / `from . import harnesses` / `from .shellrc import detect_shell` to the imports if not already present.

- [ ] **Step 8: Run everything**

Run: `uv run pytest tests/unit -q && uv run ruff check src tests`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/local_llm/setup.py src/local_llm/cli.py tests/unit/test_setup.py tests/unit/test_cli_harnesses.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
feat(integrate): a grouped menu of the coding agents found on this machine

Step 6 of setup and a bare `local-llm integrate` now show the same thing:
what is installed, grouped by whether it gets a provider written into its own
config or a launcher, what was not found, and why the Google agents cannot use
the router. Without a terminal it prints and configures nothing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 7: Uninstall removes what Codex was given

**Files:**
- Modify: `src/local_llm/uninstall.py`
- Modify: `src/local_llm/cli.py` (`_print_plan`)
- Test: `tests/unit/test_uninstall.py`, `tests/unit/test_cli_uninstall.py`

**Interfaces:**
- Consumes: `codex.codex_paths`, `codex.has_tables`, `codex.removal_lines`.
- Produces: `Inventory.codex_config: Path | None` and `Inventory.codex_backup: Path | None`, both filled by `inventory()` and consumed by `remove_integrations()` and `_print_plan`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_uninstall.py`, extend `populate()` to also leave a Codex file behind, just before it returns:

```python
    codex_dir = home / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    (codex_dir / "config.toml").write_text(
        'model = "gpt-5"\n\n'
        "[model_providers.local-llm]\n"
        'name = "local-llm"\n'
        'base_url = "http://127.0.0.1:5678/v1"\n'
        'wire_api = "responses"\n\n'
        "[profiles.local-llm]\n"
        'model = "b"\n'
        'model_provider = "local-llm"\n'
    )
```

Then append:

```python
def test_inventory_finds_the_codex_tables(tmp_path):
    populate(tmp_path)
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    assert inv.codex_config == tmp_path / ".codex" / "config.toml"
    assert "Codex" in inv.summary("integrations")


def test_remove_integrations_strips_only_our_codex_tables(tmp_path):
    populate(tmp_path)
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    lines = remove_integrations(inv)
    text = (tmp_path / ".codex" / "config.toml").read_text()
    assert "local-llm" not in text and 'model = "gpt-5"' in text
    assert any("model_providers.local-llm" in line for line in lines)


def test_remove_integrations_refuses_an_unparsable_codex_file(tmp_path):
    populate(tmp_path)
    (tmp_path / ".codex" / "config.toml").write_text("[oops\n")
    inv = inventory(Paths.from_env(env={}, home=tmp_path), home=tmp_path, env={})
    assert inv.codex_config is None, "a file we cannot read holds nothing of ours"
    assert (tmp_path / ".codex" / "config.toml").read_text() == "[oops\n"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_uninstall.py -q -k codex`
Expected: FAIL — `AttributeError: 'Inventory' object has no attribute 'codex_config'`

- [ ] **Step 3: Extend the inventory and the removal**

In `src/local_llm/uninstall.py`, add the import

```python
from .integrations.codex import codex_paths, has_tables, paths_for, removal_lines
```

add two fields to `Inventory`:

```python
    codex_config: Path | None = None
    codex_backup: Path | None = None
```

extend `summary("integrations")` so the parts list gains, after the opencode entries:

```python
            if self.codex_config:
                parts.append("Codex provider and profile")
```

fill them in `inventory()`, next to the opencode block:

```python
    cx = codex_paths(home=home, env=env)
    if has_tables(cx):
        inv.codex_config = cx.config_file
        if cx.backup.is_file():
            inv.codex_backup = cx.backup
```

and remove them in `remove_integrations()`, before the shell-rc loop:

```python
    if inv.codex_config:
        lines.extend(removal_lines(paths_for(inv.codex_config)))
        if inv.codex_backup and inv.codex_backup.exists():
            try:
                inv.codex_backup.unlink()
                lines.append(f"deleted {inv.codex_backup}")
            except OSError as error:
                lines.append(f"could not delete {inv.codex_backup}: {error}")
```

Note: `removal_lines` writes a fresh `.bak` as it edits, so delete the backup after it, not before.

- [ ] **Step 4: Show it in the plan**

In `src/local_llm/cli.py`, `_print_plan`'s integrations branch, extend the candidate list:

```python
            candidates = [inv.plugin, inv.agent_config, inv.codex_config, inv.codex_backup]
            candidates += [*inv.completion_files, *inv.rc_with_block, *inv.rc_with_bash_source]
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/test_uninstall.py tests/unit/test_cli_uninstall.py -q`
Expected: PASS

- [ ] **Step 6: Add the end-to-end assertion**

Append to `tests/unit/test_cli_uninstall.py`:

```python
def test_uninstall_all_removes_the_codex_tables(harness):
    h = harness
    populate(h.tmp)
    assert h.run("uninstall", "--dry-run").output.count("config.toml") >= 1
    result = h.run("uninstall", "--all", "--yes")
    assert result.exit_code == 0, result.output
    text = (h.tmp / ".codex" / "config.toml").read_text()
    assert "local-llm" not in text and 'model = "gpt-5"' in text
    assert not (h.tmp / ".codex" / "config.toml.bak").exists()
```

Run: `uv run pytest tests/unit -q && uv run ruff check src tests`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/local_llm/uninstall.py src/local_llm/cli.py tests/unit/test_uninstall.py tests/unit/test_cli_uninstall.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
feat(uninstall): take the Codex provider and profile back out

Listed in the plan like every other path, removed with tomlkit so the rest of
someone's Codex config survives untouched, and refused outright when the file
does not parse.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 8: Doctor reads the registry

**Files:**
- Modify: `src/local_llm/doctor.py:255-261`
- Test: `tests/unit/test_doctor.py`

**Interfaces:**
- Consumes: `harnesses.REGISTRY`, `harnesses.detect`, `codex.codex_paths`, `codex.configured_base_url`.
- Produces: the `agents` check reports every registry entry; a new `codex config` check appears only when a Codex provider is present and its `base_url` disagrees with settings.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_doctor.py`:

```python
def test_agents_check_names_every_harness_in_the_registry(tmp_path):
    from local_llm import harnesses
    from local_llm.doctor import Env, run_checks
    from local_llm.paths import Paths
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    env = Env(
        system="Darwin",
        machine="arm64",
        which=lambda name: "/usr/local/bin/codex" if name == "codex" else None,
        token_status=lambda: __import__(
            "local_llm.hub", fromlist=["TokenStatus"]
        ).TokenStatus("valid", "someone"),
        port_in_use=lambda host, port: False,
    )
    checks = {c.name: c for c in run_checks(paths, Settings(), env=env)}
    detail = checks["agents"].detail
    assert "OpenAI Codex CLI" in detail
    assert "Qwen Code" in detail and "Antigravity CLI" in detail


def test_codex_config_check_warns_when_the_address_moved(tmp_path):
    from local_llm.doctor import Env, run_checks
    from local_llm.integrations import codex
    from local_llm.paths import Paths
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    cx = codex.codex_paths(home=tmp_path, env={})
    codex.write(cx, codex.provider_table(Settings(port=5678)), None)
    env = Env(
        system="Darwin",
        machine="arm64",
        which=lambda name: None,
        token_status=lambda: __import__(
            "local_llm.hub", fromlist=["TokenStatus"]
        ).TokenStatus("valid", "someone"),
        port_in_use=lambda host, port: False,
        home=tmp_path,
        environ={},
    )
    ok = {c.name: c for c in run_checks(paths, Settings(port=5678), env=env)}
    assert "codex config" not in ok
    moved = {c.name: c for c in run_checks(paths, Settings(port=9999), env=env)}
    assert moved["codex config"].status == "warn"
    assert "integrate codex" in moved["codex config"].fix
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_doctor.py -q -k "agents_check or codex_config"`
Expected: FAIL — `TypeError: Env.__init__() got an unexpected keyword argument 'home'`

- [ ] **Step 3: Implement**

In `src/local_llm/doctor.py`, add two fields to `Env` so the check can be pointed at a scratch home:

```python
    home: Path | None = None
    environ: Mapping[str, str] | None = None
```

(import `Path` and `Mapping`), then replace the agents block at the end of `run_checks`:

```python
    # agents
    from . import harnesses
    from .integrations.codex import codex_paths, configured_base_url

    installed, missing = harnesses.detect(env.which)
    detail = "found: " + (", ".join(h.title for h in installed) or "none")
    if missing:
        detail += "; not found: " + ", ".join(h.title for h in missing)
    checks.append(Check("agents", "ok", detail))

    codex_file = codex_paths(home=env.home, env=env.environ)
    written = configured_base_url(codex_file)
    if written is not None and written != settings.openai_base_url:
        checks.append(Check(
            "codex config", "warn",
            f"{codex_file.config_file} points at {written}, not {settings.openai_base_url}",
            fix="local-llm integrate codex",
        ))
    return checks
```

The import is local to the function to keep `doctor` free of an import cycle with `harnesses`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit -q && uv run ruff check src tests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/doctor.py tests/unit/test_doctor.py
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
feat(doctor): report every agent in the registry, and a stale Codex address

The agents line was a hard-coded three, so anything added to the registry
would have gone unreported. A written provider can also go stale in a way a
launcher cannot, so a moved port is now named with the command that fixes it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md` (the "Coding agents" section, first-run step 6, the Files table, the Uninstall example)
- Modify: `CHANGELOG.md`
- Test: `tests/unit/test_install_sh.py` is unaffected; run the whole suite as the check.

**Interfaces:** none — prose only.

- [ ] **Step 1: Rewrite the "Coding agents" section of README.md**

Replace the whole section with:

````markdown
## Coding agents

`local-llm integrate` shows every coding agent it knows about that is
installed on this machine, grouped by what configuring it actually means, and
configures the ones you pick. The same menu is step 6 of `local-llm setup`.

```
  Configured inside the agent
   1. OpenAI Codex CLI    ~/.codex/config.toml: provider and profile local-llm (experimental)
   2. opencode            plugin listing every model, and a tiny helper agent   · configured

  Launched through local-llm
   3. Claude Code         local-llm claude   · alias claude_local

  Not found: GitHub Copilot CLI, aider, Qwen Code, Gemini CLI, Antigravity CLI

  Numbers to configure (e.g. 1 3), a for all, n for none [a]:
```

**Configured inside the agent** means a provider is written into the agent's
own configuration file, so it offers the router's models every time it starts,
with no help from this tool.

- **OpenAI Codex CLI** — writes `[model_providers.local-llm]` and
  `[profiles.local-llm]` into `~/.codex/config.toml` (or `$CODEX_HOME`), then
  run `codex --profile local-llm`, or the `codex_local` alias. The file is
  parsed and rewritten, so your comments and every other provider survive; a
  `.bak` is kept, and a file that does not parse is never touched.
  **This one is experimental.** Codex speaks only the OpenAI Responses API,
  and llama.cpp's `/v1/responses` endpoint does not yet match what Codex sends
  — the compatibility work is an open, unmerged llama.cpp pull request. Plain
  chat may work while tool calls fail, depending on how recent your
  `llama-server` is. Note too that Codex reads a project-level
  `.codex/config.toml` in preference to the one in your home directory, so if
  one repository ignores the local provider, look there first.
- **opencode** — copies `resources/opencode-plugin.js` to
  `~/.config/opencode/plugins/local-llm-models.js`, which builds a provider
  with one model per `models.ini` section at opencode's own start-up, and adds
  a `tiny` sub-agent bound to your smallest model with tools disabled. A
  config file with comments is never rewritten; the snippet is printed to
  paste. `local-llm integrate opencode --no-agent` installs the plugin alone.

**Launched through local-llm** means nothing is written into the agent's
configuration. A command sets the environment variables it reads and starts
it, and an alias is offered.

- **`local-llm claude [MODEL] [-- ARGS...]`** — `ANTHROPIC_BASE_URL`,
  `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, a dummy
  `ANTHROPIC_API_KEY`, and `CLAUDE_CODE_AUTO_COMPACT_WINDOW` from the model's
  context. Alias `claude_local`.
- **`local-llm copilot [MODEL] [--online|--offline]`** — the
  `COPILOT_PROVIDER_*` variables; `--offline` (the default) keeps Copilot from
  also reaching the network. The two token-limit variables it sets are
  best-effort: they are not in Copilot's published documentation.
  Alias `copilot_local`.
- **`local-llm aider [MODEL] [-- ARGS...]`** — `OPENAI_API_BASE` plus
  `--model openai/<name>`, which is what routes aider through a custom
  endpoint. Alias `aider_local`.
- **`local-llm qwen [MODEL] [-- ARGS...]`** — `OPENAI_BASE_URL`,
  `OPENAI_API_KEY` and `OPENAI_MODEL`, all three of which Qwen Code needs
  before it will use an OpenAI-compatible endpoint at all. Alias `qwen_local`.
- **`local-llm env [MODEL] [--shell zsh|bash|fish]`** — prints the `export`
  lines for both APIs, for any tool not listed here.

**Google's agents cannot be pointed at the router.** Gemini CLI has no setting
for an OpenAI-compatible endpoint — the feature request was closed and its
pull request never merged — and its `GOOGLE_GEMINI_BASE_URL` variable expects
a server speaking Google's own request format, which `llama-server` does not.
Gemini CLI is still actively released and still works with a paid Gemini or
Gemini Enterprise API key, or a Gemini Code Assist Standard or Enterprise
licence; access ended on 18 June 2026 for the free tier and for the paid
consumer plans (AI Pro and Ultra). Its successor for consumer accounts,
**Antigravity CLI** (`agy`), is in the same position. If you want a
Gemini-CLI-shaped tool that runs local models, Qwen Code continues the same
codebase and `local-llm qwen` configures it; if you want Antigravity itself to
drive a local model, its Python SDK supports that officially
(`pip install google-antigravity`, then `LocalOpenAIAgentConfig(base_url=...,
model=...)`).

**`local-llm completion install [--shell S] [--aliases/--no-aliases]`**
installs shell completion and the `local_llm` alias. Aliases are add-only: one
already in the block is never removed by a later run, only by
`local-llm uninstall`. Model names complete from `models.ini` even when the
router is down.

`MODEL` defaults to `default_model` from settings; naming one that is not in
`models.ini` is refused with the list of what is available.
````

- [ ] **Step 2: Update first-run step 6 in README.md**

Replace the step 6 paragraph (currently at lines 108-111) with:

```markdown
**6. Shell and coding agents** — detects which coding agents are installed and
offers each the right kind of configuration: a provider written into Codex's
or opencode's own config, or a launcher command plus a short alias for Claude
Code, Copilot CLI, aider and Qwen Code. Agents that cannot use the router are
named with the reason. Shell completion and the `local_llm` alias are offered
alongside. See [Coding agents](#coding-agents).
```

- [ ] **Step 3: Add the Files table row in README.md**

In the Files table, after the "Downloaded model files" row:

```markdown
| Codex provider (when configured) | `~/.codex/config.toml` | `CODEX_HOME` |
```

- [ ] **Step 4: Update the Uninstall example in README.md**

Change the integrations line of the example output to:

```
  2. integrations  opencode plugin, opencode tiny agent, Codex provider and profile, shell aliases, completion
```

- [ ] **Step 5: Add the CHANGELOG entry**

At the top of the Unreleased list in `CHANGELOG.md`:

```markdown
- Coding agents are detected and configured from one list: `local-llm
  integrate` (and step 6 of `setup`) group what is installed by whether it
  gets a provider written into its own config — Codex CLI, opencode — or a
  launcher — Claude Code, Copilot CLI, aider, Qwen Code — and say why Gemini
  CLI and Antigravity CLI cannot use the router. `integrate codex`, `aider`
  and `qwen` are new commands; `doctor` and `uninstall` read the same list.
```

- [ ] **Step 6: Verify**

Run: `uv run pytest tests/unit -q && uv run ruff check src tests`
Expected: PASS

Read the rewritten README section once from the top as someone who has never
seen this project: every term of art is either plain English or explained
where it first appears.

- [ ] **Step 7: Commit**

```bash
git add README.md CHANGELOG.md
git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F - <<'EOF'
docs: the coding-agent menu, the Codex provider, and why Google's agents cannot

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN
EOF
```

---

## Final verification

- [ ] Run the whole suite: `uv run pytest tests/unit -q`
- [ ] Lint: `uv run ruff check src tests`
- [ ] Exercise it for real, in a scratch home so nothing on this machine is touched:

```bash
HOME=$(mktemp -d) uv run local-llm integrate
HOME=$(mktemp -d) uv run local-llm doctor
```

Expected: the menu lists whatever agents this machine has, `doctor`'s agents
line names all eight, and nothing was written outside the scratch home.
