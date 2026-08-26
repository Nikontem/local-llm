# local-llm — design

Date: 2026-08-26
Status: approved in discussion, written for review

## 1. Purpose

`local-llm` is a command-line tool that gets a person from "I have a Mac or a
Linux box" to "an OpenAI- and Anthropic-compatible endpoint is serving models on
my machine, and my coding agents use it" in one guided session, and then stays
out of the way.

It is a thin layer over stock `llama-server` from llama.cpp. llama.cpp is the
inference engine that runs models in the GGUF file format (a single file that
packages weights, tokenizer and metadata). `llama-server` has a *router mode*:
one process listens on one port, and each request names the model it wants in
the `model` field; the router loads that model on demand, keeps up to N loaded,
and puts idle ones to sleep. The router is configured by a *preset file*, a
plain INI file where each `[section]` is a model and each key is a
`llama-server` flag without its leading dashes.

The tool never replaces any of that. Its job is:

1. Check and install prerequisites (Homebrew, `llama-server`, the `hf` command).
2. Detect the machine (RAM, chip, GPU) and turn that into a memory budget.
3. Recommend models that fit that budget, and search Hugging Face for others.
4. Download models and write correct, tuned preset sections for them.
5. Start, stop and inspect the router; keep logs.
6. Wire coding agents (Claude Code, GitHub Copilot CLI, opencode) to the endpoint.

It packages, in a portable and testable form, the setup that exists today in
`~/.config/local-llm/local_llm.zsh` and `models.ini` on the author's Mac.

### Non-goals for version 1

- Windows. The code avoids Unix-only calls where that is free (pathlib, psutil),
  but nothing is tested there and the README says so.
- NVIDIA or AMD GPUs tested on real hardware. Detection is implemented and unit
  tested against recorded tool output; the README labels it best effort.
- Downloading `llama-server` binaries on Linux. Version 1 prints the exact
  routes (Homebrew on Linux, GitHub release binaries, build from source).
- MLX models, GUIs, speculative-decoding draft models, more than one router.

## 2. Decisions already taken

| Decision | Choice |
|---|---|
| Language | Python 3.11+, one package for runtime and wizard |
| Name | command `local-llm`, package `local_llm`, repo `Nikontem/local-llm` |
| Model files | downloaded with the `huggingface_hub` library into the standard Hugging Face cache (`~/.cache/huggingface/hub`); preset sections use absolute paths (`model = /abs/file.gguf`) |
| Recommendations | computed from this machine and live Hugging Face data; no model list in the tool; every suggestion overridable (repo, quantization, file, context, flags) |
| Install paths | `install.sh` offers Homebrew tap (preferred when `brew` exists) or `uv tool install`; a `Nikontem/homebrew-tap` formula is part of version 1 |
| Supervision | `local-llm up` starts a detached `llama-server` with a pid file, like today; no OS service in version 1 |
| License | MIT |
| Repo location on this machine | `~/.config/local-llm/local-llm/` |
| Commit identity | `nikosntemkas <ntemkasn@gmail.com>`, set locally in the repo |

## 3. Architecture

### 3.1 Dependencies

- `typer` — command-line parsing and shell completion (zsh, bash, fish).
- `rich` — tables, progress bars, coloured status lines.
- `huggingface_hub` — Hugging Face API (search, file lists, metadata, token) and
  downloads. This is the same library the `hf` command is built on.
- `psutil` — total memory, process lookup, child processes, resident memory.

No other runtime dependencies. Everything else is the standard library.

### 3.2 Modules

Each module has one job and is testable without the others.

| Module | Job | Depends on |
|---|---|---|
| `cli` | typer application; maps commands to functions; owns all printing of results | every module below |
| `paths` | platform paths (config, state, logs, HF cache) with `LOCAL_LLM_*` environment overrides | stdlib |
| `settings` | reads `settings.toml`, applies precedence (flag > env > file > default) | `paths` |
| `preset` | parses and edits the llama-server INI, preserving comments and order | stdlib |
| `hardware` | `Machine` record: OS, arch, chip name, total RAM, GPU kind and VRAM, usable budget in bytes | `psutil`, `subprocess` |
| `estimate` | memory estimate for a set of GGUF files; fit classification against a budget | stdlib |
| `hub` | search, repo file lists with sizes, GGUF metadata, token detection and validation, downloads with progress, a one-day on-disk cache of API answers | `huggingface_hub` |
| `discover` | candidate listing, lineage, use-case grouping, quantization choice, ranking (section 7) | `hub`, `estimate`, `gguf` |
| `gguf` | reads a GGUF file header (architecture, layers, KV heads, head size, context, chat template) and computes the KV-cache cost per token; no dependency | stdlib |
| `sampling` | sampling values for a model: the repo's `preset.ini`, else parsed from its model card, else the family table in `sampling.json` | `hub` |
| `router` | start, stop, status, restart, load, unload, API client for `llama-server` | `psutil`, `paths`, `preset` |
| `logs` | per-run log files, current-log pointer, pruning | `paths` |
| `agents` | builds the environment for `claude`, `copilot`, and `env`, then execs | `preset`, `settings` |
| `integrations.opencode` | installs the plugin file, merges optional agent config | stdlib |
| `doctor` | runs checks, prints fixes, optionally applies safe ones | everything above |
| `setup` | the guided first-run flow, composed from the modules above | everything above |

Package data: `sampling.json`, `resources/opencode-plugin.js`,
`resources/models.template.ini`.

### 3.3 Data flow

```
setup / recommend / search / pull
   hardware.detect() ──► Machine(budget_bytes)
   discover.candidates() ──► hub listings ──► lineage ──► groups ──► per-repo files + sizes
   estimate.for_files(files) [+ gguf.kv_bytes(c)] ──► fit(budget) ∈ {comfortable, fits, too_big}
   sampling.values_for(repo, card, arch) ──► {temp, top-k, ...} + source
   preset.add_section(name, keys) ──► models.ini (atomic write, .bak kept)
   router.reload_models() if running

up / down / status / load / unload
   router ──► llama-server process + HTTP API on 127.0.0.1:PORT
```

## 4. Command reference

Conventions for every command: exit code 0 on success, 1 on a handled failure,
2 on a usage error; errors go to stderr and always name what failed, the likely
cause, and the command to run next; `--json` on read-only commands prints
machine-readable output instead of tables.

### 4.1 `local-llm setup [--yes] [--use USE] [--model REPO[:QUANT]]...`

The guided first run. Section 5 describes it. `--yes` accepts every default
without prompting; `--model` preselects models (repeatable); `--use` limits
recommendations to one use case.

### 4.2 `local-llm doctor [--fix] [--json]`

Runs every check below and prints a line per check: `ok`, `warn`, or `fail`,
with the fix command on failures. `--fix` offers to run the fixes marked
"safe" (package installs) one at a time, asking before each. Exit code is 1 if
any check fails.

| Check | Pass condition | Fix printed |
|---|---|---|
| platform | macOS or Linux | Windows: "untested; continue at your own risk" |
| brew | `brew` on PATH | macOS: the brew.sh install line (fail). Linux: warn only — optional |
| llama-server | on PATH; version printed; `--list-devices` output shown when the flag exists | macOS: `brew install llama.cpp` (safe). Linux: `brew install llama.cpp` if brew exists, else the release and build routes |
| llama-server router support | `--help` mentions `--models-preset` | "upgrade llama.cpp; router presets need a build from December 2025 or later" |
| hf CLI | `hf` on PATH | `brew install hf` (safe) or `uv tool install "huggingface_hub[cli]"` (safe); warn only, downloads do not need it |
| HF token | token found *and* `whoami` succeeds; prints the username | absent: warn "no token; gated repos will be unavailable" with `hf auth login`. Present but invalid: warn with `hf auth login --force` |
| config dir | exists and writable | created automatically |
| models.ini | parses; every `model =` and `mmproj =` path exists | lists missing files per section |
| state and log dirs | exist, mode 700 | created automatically |
| port | free, or held by our own llama-server | "port N is used by pid P (name); pick another with `--port` or `settings.toml`" |
| opencode / claude / copilot | presence reported | informational |

### 4.3 Router commands (behaviour ported from `local_llm.zsh`)

- `up [-f|--foreground] [--ui|--no-ui] [--max-models N]` — starts the router detached, writes a
  timestamped log, waits up to 15 s for the port to listen, prints the URL and
  next commands. Already running → says so with the pid and exits 0. `-f` runs
  attached. `--ui` serves llama.cpp's web interface; the mode is remembered in
  state for `restart`.
- `down` — stops the router and its model children (section 11.3).
- `restart [--ui|--no-ui] [--no-restore] [--max-models N]` — down, up, then reloads every model
  that was resident before, unless `--no-restore`. `--max-models N` (also on `up`)
  sets how many models may stay loaded for this run; the default is 1 because
  llama-server evicts by count, never by memory, and two large models on a 48 GB
  machine both stay resident until the GPU runs out mid-request.
- `status` — default command. Stopped: state, config path, what to run.
  Running: pid, URL, UI mode, `/health`, config path, one line per model
  child with name, resident memory in GB, and `(asleep)` when under 500 MB.
- `logs [-f] [-n N]` — last 200 lines of the current log, `-f` follows.
- `ui [-y]` — opens the web UI, starting the router with `--ui` if stopped;
  if it runs without the UI, explains that the UI can only be enabled at
  startup, lists what will be reloaded, asks, and restarts with `--ui`.
- `models` (alias `ls`) — every section name in `models.ini`, plus size on
  disk and whether the file exists.
- `load NAME... [--force]` — estimates the combined memory need, refuses when it
  exceeds the budget unless `--force`, refuses when more models than
  `max_models` are requested, then calls `POST /models/load` for each.
- `unload NAME` — `POST /models/unload`.
- `edit` — opens `models.ini` in `$EDITOR` (falls back to `vi`; on macOS with no
  `$EDITOR`, `open -t`).
- `prune-logs [DAYS]` — deletes rotated logs older than DAYS (default 30), never
  the current one.

### 4.4 `local-llm recommend [--use coding|general|small|vision] [--include-finetunes] [--limit N] [--refresh] [--pick] [--json]`

Prints a table grouped by use case: model, quantizer repo, suggested
quantization, size on disk, estimated memory, fit (comfortable / fits / too
big), lineage, thinking/vision marks, and whether it is already downloaded or
in `models.ini`. Section 7 defines the rules; nothing in it is a fixed list.
`--pick` turns the table into a numbered menu: choose a model, then choose
its quantization from every available one (the suggestion is the default),
then the tool runs `pull` for it.

### 4.5 `local-llm search TEXT [--limit N] [--author A] [--json]`

Free-text search of GGUF repositories on Hugging Face sorted by downloads.
Each result shows repo id, downloads, likes, license, lineage (vendor release,
community derivative, unknown), and — for the top 10 — the quantizations
available with size and fit, plus the exact `pull` command for the suggested
one. Results with no GGUF files are skipped. A gated repo is marked `gated`.

### 4.6 `local-llm pull REPO[:QUANT] [--quant Q] [--file NAME] [--name NAME] [--context N] [--set KEY=VALUE]... [--no-tuning] [--yes]`

1. Resolves the repo. The quantization is `:QUANT` or `--quant`; `--file`
   names one exact file when tags are ambiguous (a repo carrying both
   `UD-Q4_K_XL` and `Q4_K_XL` files — the tool lists the ambiguity and asks).
   With none given, the machine-based suggestion (7.6) is shown as the default
   next to every other quantization with size and fit, and the person
   chooses (or `--yes` accepts the suggestion).
2. Shows the file(s), sizes, the estimate, the fit, and the destination; asks
   unless `--yes`.
3. Checks free disk space at the cache location (size + 5 % margin) before
   starting; refuses with the numbers if short.
4. Downloads with a progress bar; multi-shard files are all fetched; an
   `mmproj` file is fetched for vision models.
5. Writes a preset section (section 9): context from the GGUF header and
   this machine unless `--context` is given; sampling values by source
   precedence (9.2); every `--set KEY=VALUE` written as-is, last wins.
   Reports the section name.
6. If the router is running, calls `GET /models?reload=1` so the new model is
   available without a restart; otherwise says `local-llm up`.

`local-llm add PATH [--name NAME] [--mmproj PATH] [--no-tuning]` does steps 5–6
for a GGUF file already on disk. `local-llm remove NAME [--delete-files]`
removes the section (and, with the flag, the files it points at, after
listing them and asking).

### 4.7 Agent commands

- `local-llm claude [MODEL] [-- ARGS...]` — sets `ANTHROPIC_BASE_URL`,
  `ANTHROPIC_MODEL`, `ANTHROPIC_API_KEY` (the configured key or `dummy`),
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW` from the model's `c`, prints one line saying
  which model and endpoint, then execs `claude ARGS`.
- `local-llm copilot [MODEL] [--online|--offline] [-- ARGS...]` — sets the
  `COPILOT_PROVIDER_*`, `COPILOT_MODEL`, `COPILOT_OFFLINE` variables as the zsh
  wrapper does (max prompt tokens from `c`, max output from `n-predict`), then
  execs `copilot ARGS`.
- `local-llm env [MODEL] [--shell zsh|bash|fish]` — prints the `export` lines
  for both APIs so any other tool can be pointed at the endpoint.

MODEL defaults to `default_model` from settings; an unknown name is refused
with the list of known names.

### 4.8 `local-llm integrate opencode [--agent/--no-agent] [--yes]`

Section 12.2.

### 4.9 `local-llm completion install [--shell S] [--aliases/--no-aliases]`

Installs shell completion for the detected or given shell, and, when
`--aliases` is chosen (asked interactively, default yes), adds
`local_llm`, `claude_local`, and `copilot_local` as aliases to `local-llm`,
`local-llm claude`, and `local-llm copilot`, so existing muscle memory keeps
working. Model names complete from `models.ini` even when the router is down.

### 4.10 `local-llm --version`, `local-llm --help`

## 5. Setup flow

Each step prints what it found before asking anything. With `--yes`, every
question takes its default; a step whose default is "stop" still stops.

1. **Platform and Homebrew.** macOS without `brew`: print the brew.sh install
   line, explain that llama.cpp comes from Homebrew here, and stop. Linux
   without `brew`: note that brew is optional and continue.
2. **llama-server.** Missing: offer `brew install llama.cpp` when brew exists
   (default yes); on Linux without brew, print the GitHub release route and
   the CMake build route, and stop. Present: print version and, when the flag
   exists, the devices from `llama-server --list-devices`.
3. **hf CLI.** Missing: offer `brew install hf` when brew exists, else
   `uv tool install "huggingface_hub[cli]"` when uv exists, else skip (default:
   install when a route exists). The tool's own downloads do not need it.
4. **Hugging Face token.** Report one of: valid (username), present but
   invalid, absent. Explain in one sentence what gated repositories are (the
   author requires accepting terms, which needs a token). Offer to run
   `hf auth login` (or `hf auth login --force` when invalid) interactively;
   default no. Continue either way.
5. **Hardware.** One line, for example: `Apple M4 Pro, 20 GPU cores, 48 GB
   unified memory → 38 GB usable for models (10 GB reserved)`. Offer to change
   the reserve.
6. **Models.** Show recommendations (section 7) grouped by use case with
   fit; allow multi-select by number (each pick then offers its
   quantizations with the suggestion as default), `s` to search by text, or
   `n` to skip.
   Already-configured models are shown as such and not re-downloaded.
7. **Download and configure.** For each choice: disk check, download with
   progress, write the section. Write `[*]` from the template if
   `models.ini` does not exist; never rewrite an existing `[*]`.
8. **Settings.** Write `settings.toml` with port (default 5678), reserve GB,
   max models (default 1), UI (off), default model (the first chosen, or the
   existing `default_model`).
9. **Integrations.** Offer shell completion with aliases; offer the opencode
   plugin when `opencode` is on PATH; mention `local-llm claude` and
   `local-llm copilot` when those commands exist.
10. **Start and verify.** `up`, wait for `/health`, send one chat request of
    a few tokens to the configured model with the smallest file on disk (asks
    first; it loads the model), print the two base URLs and the three
    commands to remember:
    `local-llm status`, `local-llm models`, `local-llm claude`.

Re-running `setup` is safe: every step detects what is already done.

## 6. Hardware detection and memory budget

`hardware.detect()` returns a `Machine` record:

| Field | macOS source | Linux source |
|---|---|---|
| `total_ram` | `psutil.virtual_memory().total` | same |
| `chip` | `sysctl -n machdep.cpu.brand_string` | first `model name` in `/proc/cpuinfo` |
| `gpu_kind` | `apple` when `arch == arm64`, else `none` | `nvidia` when `nvidia-smi` runs, `amd` when `rocm-smi` runs, else `none` |
| `gpu_vram` | equals `total_ram` (unified memory) | `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` summed; `rocm-smi --showmeminfo vram --json`; else 0 |
| `gpu_cores` | `system_profiler SPDisplaysDataType` "Total Number of Cores" | not collected |

Budget rules (`reserve_gb` defaults to 10):

- `apple`: `budget = total_ram − reserve`. One number, because CPU and GPU share
  memory.
- `nvidia`/`amd`: `gpu_budget = gpu_vram − 1 GiB`; `ram_budget = total_ram −
  reserve`. A model that fits `gpu_budget` is "fits on GPU"; one that fits
  only `ram_budget` is "fits with partial CPU offload (slower)"; the estimate
  used for `load` refusals is `ram_budget`.
- `none`: `budget = total_ram − reserve`, with the label "CPU only; models
  above about 8B parameters will be slow".

All subprocess calls have a 5 s timeout and any failure degrades to `none`
rather than aborting.

## 7. Recommendations

### 7.1 Principle

Nothing about specific models is hardcoded. `recommend` is computed, every
time, from two inputs: this machine (section 6) and what the Hugging Face Hub
says is popular and well-founded right now. The tool contains policies —
how to tell a vendor release from a remix, which quantization is better than
which, how much memory a context costs — but no list of models. The
suggestion is a default the person can accept; they can always name the repo,
the quantization, the file, the context and any flag themselves (4.6).

### 7.2 Candidate discovery

Three listings, merged and de-duplicated by repo id, each through
`HfApi.list_models` with `filter="gguf"`, `expand` covering tags, downloads,
likes, pipeline tag, gated flag and the `gguf` summary (parameter count,
context length, architecture):

1. sorted by downloads, descending, 300 repos;
2. sorted by trending score, 100 repos;
3. `pipeline_tag="image-text-to-text"` sorted by downloads, 100 repos (vision).

Repos with no GGUF summary, or whose GGUF parameter count is unknown, are
kept but ranked last. Gated repos are kept and marked (they need a token).

### 7.3 Lineage: vendor release or community derivative

Every GGUF repo carries tags of the form `base_model:quantized:ORG/NAME`;
that base model in turn may carry `base_model:finetune:…` or
`base_model:merge:…` tags. The tool follows that chain (at most three steps,
each a cached `model_info` call). A model is a **vendor release** when every
fine-tune or merge step stays inside one organization (Qwen fine-tuning Qwen,
Google releasing Gemma); it is a **community derivative** when the chain
crosses organizations (someone else's "uncensored" merge of Qwen). A chain
that cannot be resolved (missing tags, private base) is **unknown**.

`recommend` shows vendor releases and unknowns; derivatives appear only with
`--include-finetunes`. `search` shows everything, with the lineage in a column.

### 7.4 One entry per model

Several quantizer repos usually carry the same base model (unsloth,
bartowski, lmstudio-community, the vendor itself). Candidates are grouped by
their base model; the most-downloaded GGUF repo represents the group, and
the others are listed under "also from" so the person can pick a different
quantizer explicitly.

### 7.5 Use-case groups, decided from data

- **vision**: the repo contains an `mmproj*.gguf` file, or its pipeline tag is
  `image-text-to-text`.
- **coding**: the base model's name contains `coder`, `code`, or `devstral`
  (case-insensitive). This is a word list about a category, not a model list.
- **small**: fewer than 5 billion parameters.
- **general**: everything else. A model whose chat template contains
  `<think>` or `enable_thinking` is marked *thinking* inside its group.

A model can be in more than one group (a small coder appears in both).

### 7.6 Choosing a quantization for this machine

Quantization is the precision the weights are stored at; higher is more
accurate and larger. The tags are read from file names exactly as
llama-server does (9.3). Files are grouped per tag; multi-shard files
(`…-00001-of-00003.gguf`) are summed; the `mmproj` file (vision projector) is
added to the total when present, preferring `mmproj-F16.gguf`, then
`mmproj-BF16.gguf`, then any `mmproj*.gguf`.

Quality order, best first (unsloth's `UD-` builds rank above the plain
build of the same level):

`Q8_0`, `UD-Q6_K_XL`, `Q6_K`, `UD-Q5_K_XL`, `Q5_K_M`, `UD-Q4_K_XL`, `Q4_K_XL`,
`Q4_K_M`, `Q4_K_S`, `IQ4_XS`, `UD-Q3_K_XL`, `Q3_K_M`, `IQ3_XXS`, `UD-IQ2_M`, `IQ2_M`.

The suggestion is the first tag in that order whose estimate (7.7) is
*comfortable* on this machine; if none is, the first that *fits*; if none
fits, the smallest, marked too big. `BF16`/`F16` and tags not in the list are
never suggested but always selectable. A repo whose files carry only unknown
tags (for example gpt-oss's `MXFP4`) gets the largest file that fits
comfortably, else the largest that fits.

Every pick shows all tags with size and fit so the person can choose any,
and a too-big choice is allowed after an explicit confirmation that names the
shortfall.

### 7.7 Estimate and fit

`estimate.for_files(sizes) = sum(sizes) × 1.15 + 1 GiB` is the rule measured
on real loads and used wherever the KV cost is unknown. When the model file
is local (after download, or for `load`), the estimate is refined with the
KV cache from the GGUF header (9.3): `sum(sizes) × 1.15 + 1 GiB + kv(c)`.

Fit against a budget `B`: `comfortable` when estimate ≤ 0.6 × B, `fits` when
≤ B, else `too_big`.

### 7.8 Ranking and display

Within each group: comfortable before fits; then larger parameter count
first (the biggest model this machine runs well); then downloads. The top
three per group are shown (`--limit N` for more), with the suggested tag, size
on disk, estimated memory, fit, lineage, thinking/vision marks, and "already
downloaded" when the file is in the cache. `--json` prints the full ranked
data.

### 7.9 Caching

Listings, file lists and lineage answers are cached in `STATE/hub-cache.json`
for 24 hours; `--refresh` bypasses the cache. Offline with a warm cache works;
offline with a cold cache prints one clear error naming the cache file.

## 8. Search

`search TEXT` runs the same pipeline as `recommend` on
`list_models(search=TEXT, filter="gguf", sort="downloads", direction=-1,
limit=N)` (default 20), keeps every lineage, and for the first ten results
fetches file lists so quantizations are shown with size and fit. It never
downloads anything. A result row ends with the exact `pull` command that
would fetch the suggested tag.

## 9. Preset generation

### 9.1 The global block

When `models.ini` does not exist, it is created from
`resources/models.template.ini`:

```ini
version = 1

[*]
jinja = true
n-gpu-layers = all
flash-attn = auto
no-context-shift = true
load-on-startup = false

# Release model weights after 5 minutes idle; wake costs ~3s from page cache.
# Per-model override: sleep-idle-seconds = -1 disables it for that section.
sleep-idle-seconds = 300
```

An existing file is never rewritten wholesale; sections are appended or
replaced individually.

### 9.2 Sampling values

Sources, first hit wins, and the section comment names which one applied:

1. `preset.ini` in the repo root (llama.cpp's shareable preset format): the
   keys of the section matching the model.
2. The model card (`README.md` of the repo, then of the base model): the first
   value found for each of `temperature`/`temp`, `top_p`, `top_k`, `min_p`,
   `presence_penalty`, `repeat_penalty`/`repetition_penalty`, written as
   `name = value`, `name: value`, or a markdown table row `| name | value |`,
   case-insensitive, `_`, `-` or space between the words. Only values in a
   plausible range are accepted (temperature 0–2, top_p and min_p 0–1, top_k
   0–1000, penalties 0–3).
3. The family table below (`sampling.json`), matched by repo name.
4. Nothing (`--no-tuning`, or no source matched): the section carries only
   `model`, `mmproj`, `c`, `n-predict`.

`reasoning-format = deepseek` is written whenever the GGUF chat template has
a think channel, whatever the sampling source. The family table, with a
`source` URL per profile:

| profile | keys | source |
|---|---|---|
| qwen3-thinking | temp 0.6, top-k 20, top-p 0.95, min-p 0, presence-penalty 0.0, repeat-penalty 1.0, reasoning-format deepseek | the author's Qwen3.6 section (unsloth model card) |
| qwen3.8 | temp 1.0, top-k 20, top-p 0.95, min-p 0, presence-penalty 0.0, repeat-penalty 1.0, reasoning-format deepseek, reasoning-effort medium | the author's Qwen3.8 section (unsloth model card). `reasoning-effort` is set because the GGUF chat template defaults it to `xhigh` when a client does not send it, and llama.cpp's web UI never does |
| qwen3-instruct | temp 0.7, top-k 20, top-p 0.8, min-p 0, repeat-penalty 1.05 | the author's Qwen3-Coder section |
| qwen2.5-small | temp 0.7, top-k 20, top-p 0.8, min-p 0, repeat-penalty 1.1 | the author's Qwen2.5 section |
| gemma | temp 1.0, top-k 64, top-p 0.95, min-p 0, repeat-penalty 1.0 | the author's Gemma section |
| llama3 | temp 0.6, top-p 0.9 | Meta generation defaults |
| mistral | temp 0.15 | Mistral model cards (Devstral, Small 3) |
| gpt-oss | temp 1.0, top-p 1.0, top-k 0, min-p 0.01, chat-template-kwargs {"reasoning_effort": "high"} | llama.cpp preset documentation example |
| deepseek-r1 | temp 0.6, top-p 0.95, reasoning-format deepseek | DeepSeek R1 model card |
| generic | temp 0.7, top-k 40, top-p 0.9, min-p 0.05 | fallback |

Matching rules, evaluated in order on the repo id, the `base_model` tags and
the GGUF `architecture` field, first match wins:

1. repo contains `Qwen3.8` → `qwen3.8`
2. repo contains `Qwen3` and (`Coder` or `Instruct`) → `qwen3-instruct`
3. repo contains `Qwen3` → `qwen3-thinking`
4. repo contains `Qwen2.5` → `qwen2.5-small`
5. repo contains `gemma` → `gemma`
6. repo contains `Llama-3` or `Llama3` → `llama3`
7. repo contains `Mistral`, `Devstral`, `Magistral`, `Ministral` → `mistral`
8. repo contains `gpt-oss` → `gpt-oss`
9. repo contains `DeepSeek-R1` → `deepseek-r1`
10. otherwise → `generic`

If the repo has a `preset.ini` in its root (llama.cpp's shareable preset
format), its keys for the matching model take precedence over the profile,
and the section comment says so.

### 9.3 Section template

```ini
# added by local-llm 2026-08-26 from unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_XL, 16.1 GB)
# sampling: profile "qwen3.8" — https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
[unsloth/Qwen3.8-27B-GGUF:Q4_K_XL]
model = /Users/you/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/<sha>/Qwen3.8-27B-UD-Q4_K_XL.gguf
mmproj = /Users/you/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/<sha>/mmproj-F16.gguf
c = 65536
n-predict = 32768
reasoning-format = deepseek
reasoning-effort = medium
temp = 1.0
top-k = 20
top-p = 0.95
min-p = 0
repeat-penalty = 1.0
presence-penalty = 0.0
cache-type-k = q8_0
cache-type-v = q8_0
```

Rules:

- Section name: `--name`, else the id llama-server itself gives the file.
  llama-server scans the Hugging Face cache on its own and lists every GGUF
  there as `org/repo:TAG`, next to the preset sections. TAG is derived from
  the file name exactly as llama.cpp's `get_gguf_split_info` does: strip
  `.gguf` and any `-NNNNN-of-NNNNN` shard suffix, take the last token after a
  `-` or `.`, upper-case it — so `Qwen3.8-27B-UD-Q4_K_XL.gguf` in
  `unsloth/Qwen3.8-27B-GGUF` is `unsloth/Qwen3.8-27B-GGUF:Q4_K_XL` and
  `qwen2.5-1.5b-instruct-q4_k_m.gguf` is `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`.
  Naming the section that way makes the preset entry *replace* the
  auto-discovered one instead of sitting next to it: a duplicate picked in
  the web UI would load with no `c` at all, which on a 30B model meant a
  212k-token context and a frozen machine (observed on the author's Mac,
  2026-08-26). For a file added from a plain path (`add`), the file stem
  lower-cased. Must be unique; a clash (two files in one repo sharing a TAG,
  such as `UD-Q4_K_XL` and `Q4_K_XL`) is an error naming the existing section
  and suggesting `--name`.
- `c`: `--context`, else computed for this machine from the GGUF header:
  the largest power of two, at least 4096 and at most the model's own
  `context_length`, such that `weights × 1.15 + 1 GiB + kv(c)` fits the
  budget (7.7). `kv(c) = attention_layers × 2 × kv_heads × head_size ×
  bytes_per_element × c`, with `bytes_per_element` 1.0625 for `q8_0` and 2
  for `f16`; `attention_layers` is `block_count / full_attention_interval`
  when the header has that key (hybrid architectures such as `qwen35`), else
  `block_count`. On the author's Mac this yields 65536 for Qwen3.8-27B and
  262144 for Qwen3.6-35B-A3B — the values in use today. When the header
  cannot be read, fall back to `min(context_length, 65536)`.
- `n-predict`: `32768` when `c ≥ 65536`, else `4096`.
- `cache-type-k` and `cache-type-v` = `q8_0` when the file total is 10 GB or
  more (the KV cache — the memory that grows with context — stored at 8-bit).
- `mmproj` only when a projector file was downloaded.
- `--no-tuning` writes only `model`, `mmproj`, `c`, `n-predict`.
- Values inherited from `[*]` are not repeated.

### 9.4 Writing

Writes are atomic: write `models.ini.tmp`, copy the previous file to
`models.ini.bak`, rename. Comments and blank lines elsewhere in the file are
preserved byte for byte.

## 10. Preset parser requirements

- Grammar: optional top-level `key = value` lines before any section, `[name]`
  headers, `key = value` lines, comments starting with `#` or `;` at line start
  or after whitespace, blank lines. `#` inside a value that is not preceded by
  whitespace is part of the value (paths can contain it).
- Round-trip: parse then write with no changes yields the identical file.
- Operations: `sections()`, `get(section, key, fallback_to_star=True)`,
  `add_section(name, ordered_keys, leading_comments)`,
  `replace_section(...)`, `remove_section(name)`.
- Section names may contain any character except `]` and newline.

## 11. Router runtime

Two facts about `llama-server` 0.3.0 shape this section. It evicts loaded
models only by count (`--models-max`, least recently used), never by memory
pressure, and idle sleep is time-based; so two large models can both stay
resident until Metal runs out of memory mid-request and every call fails with
"Compute error". That is why `max_models` defaults to 1 and why the `load`
budget check is described honestly as guarding explicit loads only — a
request naming an unloaded model, or the web UI, bypasses it. Second, the
router lists every GGUF it finds in the Hugging Face cache alongside the
preset sections, which is why sections are named with the cache id (9.3).

### 11.1 Start

`llama-server --host HOST --port PORT --models-preset INI --models-max N
--models-autoload [--ui|--no-ui] [--api-key KEY]`, started with
`subprocess.Popen(start_new_session=True, stdout=log, stderr=STDOUT)`. The pid
is written to `STATE/llm-router.pid` (mode 600), the UI mode to
`STATE/llm-router.ui`. Start is refused when `llama-server` is missing, the
preset is missing, or `host` is not loopback without both
`allow_remote = true` and an API key.

### 11.2 Identify

`router.pid()` reads the pid file, checks the process exists and its command
line contains `llama-server`; if not, it looks for the process listening on
the port (psutil) with the same check. Anything else is "not running", and a
stale pid file is removed.

### 11.3 Stop

Record the router's children first (they hold the weights). Send SIGTERM to
the router, wait up to 15 s for exit, then SIGKILL. Then any child still
alive gets the same treatment. A pid whose command line does not contain
`llama-server` is never signalled; the refusal is printed with the command
line. Reports how many orphans were cleaned.

### 11.4 API client

`GET /health`, `GET /models`, `GET /models?reload=1`, `POST /models/load`,
`POST /models/unload`, `POST /v1/chat/completions` (smoke test only), with a
120 s default timeout and the API key header when configured. Loaded model
names come from `GET /models` status. Each child process is matched to its
model by the `--alias NAME` argument on the child's command line (the router
starts every child that way); child memory comes from psutil.

### 11.5 Restart with restore

Names of resident models are read before `down`; after `up`, each is loaded
again with one status line per model.

## 12. Agents and integrations

### 12.1 Claude Code and Copilot

As in section 4.7. The context number given to the agent is read from the
model's section (`c`, falling back to `[*]`), never hard-coded.

### 12.2 opencode

opencode reads exactly one global config file: the first existing of
`~/.config/opencode/opencode.jsonc`, `opencode.json`, `config.json`. Plugins in
`~/.config/opencode/plugins/*.js` are loaded automatically.

`integrate opencode`:

1. Copies `resources/opencode-plugin.js` to
   `~/.config/opencode/plugins/local-llm-models.js` (asks before overwriting a
   differing file; shows a diff). The plugin reads `LOCAL_LLM_PRESET` and
   `LOCAL_LLM_OPENAI_BASE_URL`, defaulting to the standard paths, and builds a
   provider `llamacpp` with one model per section at opencode start-up, so the
   INI stays the single source of truth.
2. Optionally adds the `tiny` sub-agent (a cheap local helper bound to the
   smallest configured model, tools disabled). Target file: the first existing
   config file in the order above, else a new `opencode.jsonc`. If the target
   parses as strict JSON, the key is merged and the file rewritten with
   two-space indentation. If it contains comments or trailing commas, the tool
   prints the snippet and the path and asks the user to paste it — it never
   rewrites a commented file.

### 12.3 Shell completion and aliases

typer generates the completion script; `completion install` writes it to the
shell's standard location and appends one `source` line to the rc file (only
if not already present). Aliases go on one marked line in the same rc file.

## 13. Configuration and paths

| Item | macOS and Linux | Env override |
|---|---|---|
| config dir | `~/.config/local-llm/` (`$XDG_CONFIG_HOME` honoured) | `LOCAL_LLM_CONFIG_DIR` |
| preset | `CONFIG/models.ini` | `LOCAL_LLM_PRESET` |
| settings | `CONFIG/settings.toml` | — |
| state dir | `~/.local/state/local-llm/` (`$XDG_STATE_HOME` honoured) | `LOCAL_LLM_STATE_DIR` |
| logs | `STATE/logs/` | `LOCAL_LLM_LOG_DIR` |
| HF cache | `huggingface_hub`'s default (`HF_HOME`, `HF_HUB_CACHE` honoured) | those |

`settings.toml` keys and defaults: `port = 5678`, `host = "127.0.0.1"`,
`max_models = 1`, `reserve_gb = 10`, `ui = false`, `default_model = ""`,
`allow_remote = false`. The API key is env-only (`LOCAL_LLM_API_KEY`) and is
never written to disk. Environment variables keep today's names
(`LOCAL_LLM_PORT`, `LOCAL_LLM_HOST`, `LOCAL_LLM_MAX_MODELS`,
`LOCAL_LLM_RESERVE_GB`, `LOCAL_LLM_UI`, `LOCAL_LLM_DEFAULT_MODEL`,
`LOCAL_LLM_ALLOW_REMOTE`). Precedence: command flag, then environment, then
`settings.toml`, then default. `local-llm env` also exports
`LOCAL_LLM_OPENAI_BASE_URL` and `LOCAL_LLM_ANTHROPIC_BASE_URL` for the
opencode plugin and other tools.

## 14. Logging

One file per router run, `LOGS/llm-router.<YYYY-MM-DDTHH-MM-SS>.log`, mode
600, with `LOGS/llm-router.log` a symlink to the current one (on a filesystem
without symlinks, a text file `current` holds the path). `logs` reads through
the pointer. `prune-logs` never removes the file the pointer names. The tool's
own diagnostics go to stderr, not to the router log.

## 15. Error handling and safety

- Every error message: what failed, likely cause, the command to run.
- Never signal a process whose command line lacks `llama-server`.
- Never bind a non-loopback host without `allow_remote` and an API key.
- Never overwrite `models.ini` non-atomically; always keep `.bak`.
- Never rewrite a commented opencode config.
- Never start a download without a disk-space check.
- Never store the API key or the HF token.
- A gated or missing repo produces a message that names the repo and, for
  gated ones, says to accept the terms on the repo page and log in with
  `hf auth login`.

## 16. Testing and verification

Four layers, from cheapest to most realistic.

1. **Unit tests** (`pytest`, no network, run everywhere): preset round-trip
   with comments; estimate maths and fit thresholds; quantization matching
   including shards and mmproj; lineage rules against recorded tag sets;
   use-case grouping; quantization suggestion per budget; ranking against
   fake sizes; GGUF header parsing against a synthetic header and the KV
   formula against the author's INI values; model-card parsing fixtures;
   sampling source precedence; hardware parsing from recorded `sysctl`,
   `system_profiler`, `/proc/meminfo`, `/proc/cpuinfo`, `nvidia-smi` and
   `rocm-smi` output; router pid/child logic against a fake psutil; opencode
   merge rules; every CLI command through typer's `CliRunner` with the modules
   faked. Target: every module above 90 % line coverage.
2. **Docker end-to-end for Linux** (`tests/e2e/Dockerfile`, `tests/e2e/run.sh`):
   an `ubuntu:24.04` image that builds `llama-server` from a pinned llama.cpp
   tag (CPU only, cached as a layer), then runs `install.sh --uv`,
   `local-llm doctor`, `local-llm recommend`, `local-llm pull` of a model
   under 500 MB (`Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`), `up`, a chat
   request that must return text, `status` showing the loaded child, `down`
   with no `llama-server` process left, and `local-llm completion install`
   for bash. This is the clean-machine proof for Linux and answers "does the
   installer work from nothing". It runs on this Mac's Docker (arm64) and in
   CI. A second, manual target `run.sh --brew` installs Homebrew on Linux
   inside the image and takes the brew path; it is slow and not part of CI.
3. **macOS acceptance on this Mac**: `setup` end to end against the real
   Metal build and the models already in the cache; then `local-llm claude`
   with the default model; then the migration in section 18.4. A "pretend
   clean" run (`HOME` pointed at a scratch directory, `PATH` without
   `/opt/homebrew/bin`) exercises the missing-tool branches of `doctor` and
   `setup` without uninstalling anything.
4. **CI** (GitHub Actions): `ci.yml` runs `ruff` and the unit tests on
   `ubuntu-latest` and `macos-latest` on every push; `e2e.yml` runs the
   Docker end-to-end on `ubuntu-latest` on tags and on a weekly schedule, and
   a live discovery test with network (the pipeline returns at least one
   vendor-lineage model per use case).

What Docker cannot cover: Metal (macOS) and real NVIDIA/AMD hardware. Metal is
covered by layer 3; GPUs on Linux remain best effort until someone runs the
e2e on such a machine (the README asks for that report).

## 17. Packaging and distribution

- `pyproject.toml` with hatchling; `[project.scripts] local-llm =
  "local_llm.cli:app"`; `requires-python = ">=3.11"`; lower-bounded
  dependencies; `uv.lock` committed; `ruff` configured; version in
  `local_llm/__init__.py` starting at `0.1.0`.
- `install.sh` (POSIX sh): flags `--brew`, `--uv`, `-y`, `--upgrade`. Detects
  the OS. If `brew` exists, asks "Homebrew (recommended, brew manages
  updates) or uv?" defaulting to Homebrew; without `brew`, uses uv. The brew
  path runs `brew tap nikontem/tap && brew install local-llm`. The uv path
  installs uv with the official installer when missing, then
  `uv tool install git+https://github.com/Nikontem/local-llm` (with
  `--upgrade` when asked) and runs `uv tool update-shell` so `~/.local/bin`
  is on PATH. Both end with "run `local-llm setup`".
- Homebrew tap: repo `Nikontem/homebrew-tap`, `Formula/local-llm.rb` using
  `Language::Python::Virtualenv`, `depends_on "python@3.13"`, `"llama.cpp"`,
  `"hf"`; resources generated with `brew update-python-resources`; `url` is
  the `v0.1.0` release tarball with its sha256; `head` points at `main`;
  `test do` runs `local-llm --version`. Published after the main repo is
  tagged.

## 18. Repository, git, rollout

### 18.1 Layout

```
local-llm/
  pyproject.toml  uv.lock  README.md  LICENSE  CHANGELOG.md  install.sh
  src/local_llm/
    __init__.py cli.py paths.py settings.py preset.py hardware.py estimate.py
    hub.py catalog.py sampling.py router.py logs.py agents.py doctor.py setup.py
    integrations/opencode.py
    catalog.json sampling.json
    resources/opencode-plugin.js resources/models.template.ini
  tests/unit/ tests/live/ tests/e2e/
  docs/superpowers/specs/ docs/superpowers/plans/
  .github/workflows/ci.yml .github/workflows/e2e.yml
```

`models.ini`, `settings.toml`, and everything under `~/.config/local-llm/`
outside the repo folder are user configuration and are never committed. The
repo's `.gitignore` covers Python build output, `.venv`, and editor files.

### 18.2 README

Ported from the existing README: quick start, every command with a real
example, switching models by request, coding agents, memory (sleep and
budgeting), the settings table, troubleshooting, file locations. New sections:
install (both paths), `setup` walk-through, recommendations and search,
platform notes (macOS first class, Linux first class on CPU and best effort on
GPU, Windows untested), and a short "how this relates to Ollama, LM Studio and
llama.cpp's own router" paragraph so expectations are honest.

### 18.3 GitHub

The author runs `gh auth login -h github.com` for the `Nikontem` account. Then,
after explicit confirmation: `gh repo create Nikontem/local-llm --public
--source=. --remote=origin --push`, and later `Nikontem/homebrew-tap`. The
`v0.1.0` tag is created after the macOS acceptance passes.

### 18.4 Migration of this machine

After `install.sh` has installed the tool: replace the `.zshrc` line that
sources `local_llm.zsh` with the completion and aliases from
`local-llm completion install`; rename `local_llm.zsh` to
`local_llm.zsh.retired`; leave `models.ini` untouched (it is already valid);
write `settings.toml` with `default_model = "unsloth/Qwen3.8-27B-GGUF:Q4_K_XL"`,
`max_models = 1`, `reserve_gb = 10`, `port = 5678`; run `integrate opencode` (the plugin file
becomes the one shipped by the tool). Backups and retired router files are
left where they are.

### 18.5 Claude Code session

When the work in this session is finished, copy this session's transcript,
`~/.claude/projects/-Users-nikosntemkas--config-local-llm/44133a03-c00e-4a43-9970-6c373bbbdee9.jsonl`,
into `~/.claude/projects/-Users-nikosntemkas--config-local-llm-local-llm/` so
`claude --resume` from inside the repo lists it. The earlier session stays
where it is. Copy, not move, because the file is still being written.

## 19. Milestones

1. **Runtime parity** — scaffold, `paths`, `settings`, `preset`, `logs`,
   `router`, `agents`, `doctor`; every existing command works on this Mac
   against the existing `models.ini`.
2. **Models** — `hardware`, `estimate`, `hub`, `catalog`, `sampling`,
   `search`, `recommend`, `pull`, `add`, `remove`.
3. **First run** — `setup`, `integrate opencode`, `completion install`,
   `install.sh`, README, CI, Docker end-to-end.
4. **Release** — macOS acceptance, migration of this machine, GitHub push,
   `v0.1.0`, Homebrew tap, session copy.

## 20. Risks

- llama.cpp changes flag names or the preset format. Mitigation: `doctor`
  checks for `--models-preset`; unknown INI keys pass through untouched;
  router-controlled keys (host, port, api-key, alias) are never written.
- Hugging Face API shape changes. Mitigation: all access through
  `huggingface_hub`, no hand-built URLs.
- Hugging Face tag conventions (`base_model:*`) change or are missing.
  Mitigation: lineage falls back to "unknown", which is shown rather than
  hidden; the live test runs weekly.
- Model-card parsing picks a wrong number. Mitigation: plausibility ranges,
  the section comment names the source and the values, `--set` overrides any
  key, `--no-tuning` opts out entirely.
