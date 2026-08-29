# Changelog

## Unreleased

- A newly written `models.ini` section gets `fit-ctx = N`, a floor, instead
  of a fixed `c = N`: `llama-server` now chooses the real context for a
  model when it loads, from whatever memory is actually free at that moment,
  never going below the floor. The same model can therefore load with a
  different context on different runs — more when the machine is idle, less
  when it is busy. `pull --context`/`add` still write `c` and pin the
  context exactly, switching off that adjustment for the section. `doctor`
  gained a `context sizing` note naming any section still pinned to `c`, and
  a `model fit` warning when the router log shows a model that loaded
  without fitting into free memory. `local-llm status` shows the context
  each resident model actually received.
- The documentation is split up: `README.md` is now a front page, and the
  reference material it used to carry lives in `docs/setup.md`,
  `docs/commands.md`, `docs/agents.md`, `docs/configuration.md` and
  `docs/troubleshooting.md`. No behaviour changed.
- Coding agents are detected and configured from one list: `local-llm
  integrate` (and step 6 of `setup`) group what is installed by whether it
  gets a provider written into its own config — Codex CLI, opencode — or a
  launcher — Claude Code, Copilot CLI, aider, Qwen Code — and say why Gemini
  CLI and Antigravity CLI cannot use the router. `integrate codex`, `aider`
  and `qwen` are new commands; `doctor` and `uninstall` read the same list.
- `uninstall`: undo models, integrations, state and config item by item,
  listing everything first; `--dry-run`, `--all`, `--restore-shell-line`.
- `install.sh` is uv-only; Homebrew is used for llama.cpp and `hf` when present.
- Downloads use plain HTTP and give up with a resume hint after three minutes
  without progress.

## 0.1.0 — 2026-08-26

First release.

- One `llama-server` router serving every model in `models.ini`, switched per
  request: `up`, `down`, `restart`, `status`, `logs`, `ui`, `models`, `load`,
  `unload`, `edit`, `prune-logs`. Loaded models are shown with the router's
  own status; `load` refuses combinations that do not fit and counts the KV
  cache for the configured context.
- `doctor`: Homebrew, `llama-server` (version, devices, router support), the
  `hf` command, token validity, config and state directories, `models.ini`
  paths, port, coding agents — each failure with the command that fixes it.
- `recommend`: computed from this machine (RAM, chip, GPU) and live Hugging
  Face data; no model list in the tool. Vendor releases are told from
  community remixes by their tags and names; the suggested quantization is
  the best one that fits comfortably; every choice can be overridden.
- `search`, `pull`, `add`, `remove`: download with a disk-space check and a
  stall watchdog, and write a tuned section named the way `llama-server`
  names cached files (`org/repo:TAG`), with context sized to the machine from
  the GGUF header and sampling values from the repo's `preset.ini`, its model
  card, or a family table.
- `setup`: a guided first run through prerequisites, machine, models,
  settings, shell completion and aliases, opencode plugin, start and a smoke
  request. Safe to re-run.
- `claude`, `copilot`, `env`: point Claude Code, GitHub Copilot CLI and any
  OpenAI- or Anthropic-style tool at the router; `integrate opencode` installs
  the plugin that lists every model in opencode.
- `install.sh`: installs the tool with `uv`; when Homebrew is present, `setup`
  installs llama.cpp and `hf` through it.
- macOS and Linux (CPU, and best effort on NVIDIA/AMD); Windows untested.
