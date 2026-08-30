# Changelog

## Unreleased

- New command: `local-llm browse-models` opens the model menu from setup step
  3 on its own, so choosing models is no longer something you can only do
  during the first run. It takes several numbers at once, searches Hugging
  Face with `s <text>` without ending the menu, asks the quantization per
  pick, and downloads and configures each one. A model already in
  `models.ini` is skipped by name rather than downloaded twice, a download that
  fails is named and the run carries on to the picks after it rather than
  throwing away answers you have already given, an answer it cannot read is
  refused and asked again instead of dropping you back to the shell, and
  `settings.toml` is left alone. Without a terminal it refuses and
  names `recommend --json` and `pull` as the ways to do this in a script. The
  wizard's model step and the new command now run the same code, so they
  cannot drift apart.

- New command: `local-llm tune <model>` runs a model against a workload
  shaped like a coding agent's traffic and times it, varying batch size,
  micro-batch size, flash attention, and key-value cache precision one at a
  time against the model's current settings — everything else `local-llm`
  writes into `models.ini` is estimated, never measured. It unloads and
  reloads a model the router is already holding so a second copy in memory
  cannot skew the numbers, and writes nothing to `models.ini` unless you say
  so. It will not measure a cache-type change that the model could no longer
  fit at its configured context — the cache precision decides what every
  token in the conversation costs — and it names each combination it skips
  and why. The table shows the spread `llama-bench` saw across its own
  repetitions, and a winner less than 5% ahead of your current settings is
  reported as being within measurement noise rather than offered: on a busy
  machine the middle of this ranking does not reproduce, so only a visible
  margin is worth editing a configuration file for.

- The `looking at <repo>` line that `local-llm recommend` and the setup
  wizard's model step print while they scan the Hugging Face Hub is now
  padded to the width of the terminal and wiped when the scan ends. Short
  repository names used to leave the tail of a longer name behind them, so
  the two collided into an unreadable string that then stayed on screen
  underneath whatever was printed next.

- `local-llm claude`/`copilot` and `local-llm load`'s budget estimate now fall
  back to a section's `fit-ctx` when there is no `c` or `ctx-size` to read, so
  a model pulled with only a context floor still gets
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `COPILOT_PROVIDER_MAX_PROMPT_TOKENS`, and
  a `load` estimate that is not the model's uncapped trained context. `doctor`
  now also treats `ctx-size` as `c`'s equivalent in the `context sizing`
  check, names the model in the `model fit` warning when the router log says
  which one failed to fit, reads only the log's tail rather than the whole
  file, and no longer advises turning `c = N` straight into `fit-ctx = N` —
  that recreates the problem with an unreachable floor — suggesting a modest
  floor or deleting `c` and re-pulling instead.
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
