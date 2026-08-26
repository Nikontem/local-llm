# Milestone 1 acceptance — the author's Mac, 2026-08-26

Machine: Apple M4 Pro, 48 GB, macOS 26.5.2, llama-server 0.3.0 (build 10621, Homebrew).
Repo at commit `95fe37a` plus the `models` column-width fix. All commands run with
`uv run local-llm ...` from the repo, against the real `~/.config/local-llm/models.ini`
(five sections named `org/repo:QUANT`). No router was running beforehand.

## Step 1 — doctor and models

`doctor` exit 0. Every check `ok` except `hf token`: *warn — token present but
invalid, fix: `hf auth login --force`*, which is the true state of this machine.
`llama-server` line showed the version and both devices (`BLAS: Accelerate`,
`MTL0: Apple M4 Pro (38338 MiB)`); `router support` ok; `models.ini` "5 model(s),
all files present"; `port 5678 is free`; agents found: claude, copilot, opencode.

Note: `hf` resolved to `.venv/bin/hf` because `uv run` puts the project venv first
on PATH; an installed tool does not expose it, so the check stays honest there.

`models` listed the five sections with 20.8, 16.5, 17.5, 17.2 and 1.0 GB (GiB-based,
as the zsh script reported). The name column overflowed for the long names; fixed
by sizing the column to the longest name (commit after this file).

## Step 2 — up, status, load, unload, logs, env

- `up`: "Router is up." with `http://127.0.0.1:5678/v1`.
- `status`: running, pid 52995, `health: {"status":"ok"}`, "loaded models: none resident".
- `load Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`: estimate `~2.2 GB`, budget
  `38.0 GB (RAM minus 10 GB reserved)`, `{"success":true}`.
- `status` after 6 s: `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M pid 53010  2.0 GB`.
- `unload`: `{"success":true}`.
- `logs -n 5`: the five llama-server lines for the unload.
- `env`: nine export lines (OpenAI and Anthropic base URLs, dummy keys, model,
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW=32768`, the three `LOCAL_LLM_*` values).

## Step 3 — restart with restore, down

- `load` again, then `restart`: "Stopping router pid 52995", "Router is down.",
  "Router is up.", "Restoring 1 model(s) that were loaded: ... ok".
- `status` immediately afterwards showed the child at `0.1 GB (asleep)`: the
  restore call returns before the weights are read, and the label comes from
  resident memory, so a model still loading reads as asleep for a few seconds.
  Follow-up for a later milestone: label from `GET /models` status
  (`loading` / `loaded` / `sleeping`) instead of memory alone.
- `down`: "Stopping router pid 53106", "Router is down."; `pgrep -fl llama-server`
  found nothing (exit 1).

## Step 4 — agent wrapper

`claude Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M -- --version` printed
`claude -> Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M (context 32768) at http://127.0.0.1:5678`
followed by Claude Code's own `2.1.246 (Claude Code)`.

## Verdict

Every command in the milestone-1 plan behaves as specified on this machine.
Unit suite: 117 passed, ruff clean.
