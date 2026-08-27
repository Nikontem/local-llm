# Changelog

## Unreleased

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
