# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`local-llm` is a Typer CLI that wraps `llama-server` (llama.cpp) in *router mode*: one
process, one port, serving every model listed in `models.ini`. The tool detects the
machine, recommends and downloads models that fit, writes tuned INI sections, manages the
router process, and points coding agents (Claude Code, Codex CLI, opencode, Copilot CLI,
aider, Qwen Code) at it. The pages under `docs/` are the specification of user-facing
behaviour and are kept current: `docs/commands.md` for every command and flag,
`docs/setup.md`, `docs/agents.md`, `docs/configuration.md`, `docs/troubleshooting.md`.
`README.md` is the front page only — what the tool is, how to start, and links onward.

## Commands

```
uv sync                          # install into .venv from the committed lock file
uv run pytest -q                 # unit tests, no network (this is what CI runs)
uv run pytest tests/unit/test_cli_router.py::test_name   # a single test
uv run ruff check .              # lint (line-length 100, rules E/F/I/UP/B)
uv run local-llm --version       # smoke check, also in CI
uv run pytest tests/live -m live # one test against the real Hugging Face Hub
tests/e2e/run.sh                 # Docker: install.sh → doctor → pull → up → chat → down
```

CI (`.github/workflows/ci.yml`) runs ruff + unit tests on macOS and Linux, Python 3.11 and
3.13. The e2e and live-hub jobs run only on tags, weekly, or manually.

## Architecture

Three layers, bottom to top:

**Machine and model facts** — `hardware.py` (RAM, chip, GPU → a memory budget),
`estimate.py` and `gguf.py` (size estimates, refined from the GGUF header's KV-cache
numbers), `quant.py` (quantization tags and fit), `hub.py` (Hugging Face client, 24-hour
disk cache), `discover.py` (ranks candidates into the coding/general/small/vision groups),
`sampling.py` + `sampling.json` (per-family sampling defaults).

**State on disk** — `paths.py` resolves every location from the environment
(`Paths.from_env`, honouring `LOCAL_LLM_*` and XDG); `preset.py` is a comment-preserving
reader/writer for `models.ini`; `settings.py` holds `settings.toml`; `sections.py` builds a
tuned section for one model; `router.py` starts, stops and inspects the `llama-server`
process; `logs.py` handles the timestamped log files.

**Commands and integrations** — `cli.py` (1700 lines, every Typer command),
`setup.py` (the seven-step wizard), `doctor.py`, `uninstall.py`, `agents.py` (environment
variables for launcher-style agents), `harnesses.py` (the registry of every known coding
agent), `integrations/` (`codex.py`, `opencode.py`, and the shared `HarnessContext`),
`shellrc.py` (marked blocks and aliases in the user's shell startup file).

### Invariants worth knowing before you change things

- **Section names are not cosmetic.** `sections.section_name()` produces `org/repo:TAG` —
  exactly the id `llama-server` derives from the file name when it scans the Hugging Face
  cache. Naming a section anything else makes the router serve a *duplicate* entry with no
  context limit.
- **Every write to a user's file goes through `integrations.atomic_write()`.** It writes a
  temp file beside the destination, fsyncs, renames, carries the old permissions across,
  and follows symlinks through to the real file. It keeps a backup named
  `<name>.local-llm.bak` — a suffix nothing else uses, which is how `uninstall` knows a
  backup is its own and safe to offer for deletion. Never write a config file in place.
- **Settings precedence, highest first:** the command's own flag, then the `LOCAL_LLM_*`
  environment variable, then `settings.toml`, then the built-in default. The API key
  (`LOCAL_LLM_API_KEY`) is environment-only and is never written to disk or printed.
- **`max_models` defaults to 1 on purpose.** `llama-server` evicts resident models by count,
  never by memory pressure, so two large models can both stay loaded until the GPU runs out
  mid-request.
- **A harness integration only touches what its `HarnessContext` gives it** — paths,
  settings, preset, home, env, `say`, `confirm`. That is what makes them testable. Adding a
  coding agent means one `Harness` entry in `harnesses.REGISTRY` plus, for a provider-style
  agent, a module in `integrations/`; `doctor`, `integrate`, `setup` step 6 and `uninstall`
  all read that one registry.
- **A shell startup file whose markers do not pair up, or that is not UTF-8, is left
  completely alone** and named to the user — no aliases, no completion.

## Tests

`tests/unit/conftest.py` is the whole story. The `harness` fixture points `HOME` at
`tmp_path`, *deletes* every `LOCAL_LLM_*`, XDG, `EDITOR`, `CODEX_HOME` and
`OPENCODE_CONFIG_DIR` variable, and wires the CLI to `FakeBackend` (process table) and
`FakeHttp` (router API) from `tests/unit/fakes.py`. `hubbed` adds a fake Hugging Face Hub
and a fixed Mac. Scrubbing those variables is not tidiness: leave `CODEX_HOME` set and a
test run rewrites the developer's real Codex config.

Anything that shells out or hits the network gets a fake — `router.ProcessBackend` is a
Protocol precisely so `PsutilBackend` can be swapped out.

## Conventions

- Docstrings explain *why* a thing is the way it is, often at some length, and comments
  inside functions justify non-obvious choices (see `integrations/atomic_write`). Match
  that register rather than adding descriptive one-liners.
- Commit subjects are lowercase plain sentences describing the user-visible effect
  (`fix(uninstall): a backup of your own file is never taken away unasked`), not
  imperative summaries of the diff.
- User-visible changes go in `CHANGELOG.md` under `## Unreleased`, and in the `docs/` page
  that covers the command they change. Touch `README.md` only when the change alters
  something the front page itself claims.
- Design documents live in `docs/superpowers/specs/` and plans in `docs/superpowers/plans/`;
  per-task briefs and reports for in-flight work live in `.superpowers/sdd/<date-slug>/`.
