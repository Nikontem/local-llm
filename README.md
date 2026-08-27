# local-llm

One `llama-server` process serving every model listed in `models.ini`, on one
port. `llama-server` is the inference server from
[llama.cpp](https://github.com/ggml-org/llama.cpp), the engine that runs
models packaged as **GGUF** files (a single file holding weights, tokenizer
and metadata). Since December 2025 it has a *router mode*: point it at an INI
file where each `[section]` names a model and the flags to run it, and one
endpoint serves all of them — a request's `model` field picks which one loads,
the router keeps a fixed number resident, and idle ones go to sleep. `local-llm`
is a thin layer on top of that: it detects your machine, recommends models
that actually fit, downloads and tunes them, starts and stops the router, and
points Claude Code, GitHub Copilot CLI and opencode at it. It runs on macOS
and Linux; Windows is untested.

Ollama and LM Studio also get a model running locally, but each hides
llama.cpp behind its own model format and app. `llama-server`'s own router
mode gives you the same one-endpoint-many-models setup this tool configures,
with no wizard, no hardware-aware suggestions, and no agent wiring. `local-llm`
stays thin and open on top of it: the config is a plain llama.cpp INI file you
can read and edit by hand, everything runs from the terminal, and every model
suggestion is computed from your actual machine rather than a fixed list.

## Install

```
curl -fsSL https://raw.githubusercontent.com/Nikontem/local-llm/main/install.sh | sh
```

The script installs the tool with [`uv`](https://astral.sh/uv), installing
`uv` itself first if needed, on macOS and Linux. Homebrew is not required for
the tool. It matters for what the tool installs *for you*: when Homebrew is
present, `local-llm setup` installs `llama.cpp` (for `llama-server`) and `hf`
(Hugging Face's CLI) through it, so they stay managed by Homebrew like the
rest of your machine; when it is absent, `setup` shows the routes — Homebrew
from [brew.sh](https://brew.sh), llama.cpp's release binaries, or a build.

To install by hand instead of running the script:

```
uv tool install git+https://github.com/Nikontem/local-llm
```

Upgrade later with `sh install.sh --upgrade` or
`uv tool install --upgrade git+https://github.com/Nikontem/local-llm`.

Either way, the next step is the same:

```
local-llm setup
```

## First run

`local-llm setup` is a guided session in seven numbered steps. Nothing is
downloaded or overwritten without asking, and it can be re-run any time —
each step detects what is already done and skips it. `--yes` accepts every
default without prompting; `--model org/repo[:QUANT]` preselects a model
(repeatable); `--use coding|general|small|vision` narrows step 3 to one use
case.

**1. Prerequisites** — runs the same checks as `local-llm doctor` and offers
to fix the ones it safely can (installing `llama.cpp` or `hf` via Homebrew).
Missing Homebrew on macOS stops the wizard, since `llama.cpp` is installed
from it there; on Linux it prints the release-binary and build-from-source
routes instead.

**2. This machine** — detects RAM, chip and GPU and turns that into a memory
budget, then asks how much memory to reserve for the OS and other apps
(default 10 GB). On the author's Mac this prints:

```
Apple M4 Pro, 20 GPU cores, 48 GB unified memory → 38 GB usable for models (10 GB reserved)
```

**3. Models** — with no `--model` given, it looks at what is popular on
Hugging Face right now and shows a numbered menu, grouped by use case:
coding, general, small, and vision. Every suggestion names a **quantization**
— the precision a model's weights are stored at; a lower-precision
quantization is a smaller file that runs faster but loses some accuracy — and
picks the quantization that best fits your machine's budget. Abridged from a
real run on the 48 GB Mac above:

```
coding
   1. Qwen3-Coder-30B-A3B-Instruct  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF  UD-Q4_K_XL
      16.5 GB on disk, ~19.9 GB in memory  comfortable  vendor  · in models.ini
general
   2. Qwen3-30B-A3B-Thinking-2507 (thinking)  unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF  UD-Q4_K_XL
      ...  comfortable  vendor
small
   4. Qwen3.5-4B (thinking, vision)  unsloth/Qwen3.5-4B-GGUF  Q8_0
      4.8 GB on disk, ~6.5 GB in memory  comfortable  vendor
```

The quantization suggested scales with the machine — small models get the
high-precision `Q8_0`, the 27–30B ones get `UD-Q4_K_XL` on a 38 GB budget.
Type numbers to download (`1 3`), `s <text>` to search for something else, or
`n` for nothing.

**4. Download and configure** — downloads what was picked and writes a tuned
section for each into `models.ini` (see [Models](#models) below).

**5. Settings** — writes `settings.toml` with the reserve from step 2 and a
default model (the first one just downloaded, or the smallest already
configured).

**6. Shell and coding agents** — detects which coding agents are installed and
offers each the right kind of configuration: a provider written into Codex's
or opencode's own config, or a launcher command plus a short alias for Claude
Code, Copilot CLI, aider and Qwen Code. Agents that cannot use the router are
named with the reason. Shell completion and the `local_llm` alias are offered
alongside. See [Coding agents](#coding-agents).

**7. Start** — starts the router and offers a one-line test chat against the
smallest configured model, then prints the endpoints:

```
  OpenAI-style endpoint:     http://127.0.0.1:5678/v1
  Anthropic-style endpoint:  http://127.0.0.1:5678
  local-llm status      what is running
  local-llm models      every model you can ask for
  local-llm claude      Claude Code against the router
```

If no models are configured yet — a bare `--yes` run, or nothing chosen in
step 3 — setup stops after step 6 and points at `local-llm recommend` and
`local-llm pull` instead of starting a router with nothing to serve.

## Every day

- **`local-llm up`** — starts the router detached and writes a timestamped
  log.

  ```
  Router is up.
    url:      http://127.0.0.1:5678/v1
    models:   local-llm models
    logs:     local-llm logs -f
  ```

- **`local-llm status`** — the default command; what is running and what is
  loaded right now.

  ```
  Local LLM router

    state:    running
    pid:      52995
    url:      http://127.0.0.1:5678/v1
    web ui:   off  (local-llm restart --ui)
    health:   {"status":"ok"}
    config:   /Users/you/.config/local-llm/models.ini

    loaded models:  none resident

    local-llm logs -f        follow the log
    local-llm models         list every model in models.ini
    local-llm load <model>   preload one so the first prompt is fast
  ```

- **`local-llm models`** (alias `ls`) — every section name in `models.ini`
  with its size on disk; pass any of them as the `model` field of an OpenAI
  request, no restart needed.
- **`local-llm load <name>...`** — preloads one or more models and refuses
  when the combined estimate exceeds the memory budget, unless `--force`.
- **`local-llm unload <name>`** — releases a model immediately instead of
  waiting for it to go idle.
- **`local-llm logs -f`** — follows the current log; `-n 5` shows the last
  five lines without following.
- **`local-llm down`** — stops the router and every model process under it,
  cleaning up any orphan it finds.
- **`local-llm restart`** — down, then up, then reloads every model that was
  resident before (`--no-restore` to skip that, `--ui`/`--no-ui` to change
  the web UI mode).
- **`local-llm ui`** — opens `llama-server`'s own web interface, starting the
  router with `--ui` first if it is not running that way.

`up` and `restart` both take `--max-models N`, capped at 1 by default. This
is not caution for its own sake: `llama-server` evicts loaded models only by
count, never by memory pressure, so two large models can both stay resident
until the GPU runs out of memory mid-request and every call fails. Raise it
only when you know two configured models together fit the budget.

## Models

- **`local-llm recommend [--use coding|general|small|vision]
  [--include-finetunes] [--pick]`** — the same table as setup step 3, on
  demand. "Computed from this machine" means every number reflects your RAM,
  chip and GPU (section 2 above), not a fixed list; the quantization
  suggested scales with what you have. Results favor **vendor releases** —
  the original publisher's own build, or a fine-tune that stays inside one
  organization — over **community derivatives** (someone else's remix);
  derivatives only show with `--include-finetunes`. `--pick` turns the table
  into a menu: choose a model, then its quantization (the suggestion is the
  default), and the tool runs `pull` for it.
- **`local-llm search TEXT [--limit N] [--author A]`** — free-text search of
  GGUF repositories on Hugging Face, sorted by downloads, with every
  quantization's size and fit and a ready-to-run `pull` command for the
  suggested one.
- **`local-llm pull REPO[:QUANT] [--quant Q] [--file NAME] [--context N]
  [--set KEY=VALUE]... [--no-tuning] [-y]`** — downloads a model and writes a
  tuned `models.ini` section for it. With no quantization named, it shows
  every one available with size and fit and asks; `-y` accepts the
  suggestion. Output from a 0.5 GB pull (from the milestone-2 acceptance
  run, paths shortened):

  ```
  Qwen/Qwen2.5-0.5B-Instruct-GGUF  Q8_0
    qwen2.5-0.5b-instruct-q8_0.gguf
    size on disk:    0.6 GB
    estimated need:  1.7 GB  (comfortable; 38.0 GB usable here)
    section name:    Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0

  Added [Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0] to /Users/you/.config/local-llm/models.ini
    context: 32768 suggested for this machine (38.0 GB usable)
    sampling: profile qwen2.5-small
  ```

  The section name (`org/repo:TAG`) is not cosmetic: it is exactly the id
  `llama-server` derives from the file name and gives that same file when it
  scans the Hugging Face cache on its own. Naming the section that way makes
  it *replace* the auto-discovered entry instead of sitting next to it under
  a different name — a duplicate loaded from the web UI gets no context
  limit at all, which on a 30B model meant a 212k-token context and a frozen
  machine on the author's Mac. `--quant`/`--file` pick a specific
  quantization or exact file when a repo's tags are ambiguous; `--context`
  overrides the size computed for your machine; `--set KEY=VALUE` writes any
  `llama-server` flag as-is (repeatable, last one wins); `--no-tuning` writes
  only `model`, `mmproj`, `c` and `n-predict`, skipping sampling values.
- **`local-llm add PATH [--mmproj PATH] [--name NAME]`** — the same tuning
  and section-writing as `pull`, for a GGUF file you already have on disk.
- **`local-llm remove NAME [--delete-files]`** — removes a section, and with
  the flag, the model file(s) it points at, after listing them and asking.
- **`local-llm edit`** — opens `models.ini` in `$EDITOR` directly; everything
  above is a convenience over hand-editing this plain INI file, which stays
  the source of truth.

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

  Detected, but cannot use the router
      Antigravity CLI     speaks only Google's own API format, which the router does not serve

  Not found: GitHub Copilot CLI, aider, Qwen Code, Gemini CLI

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
  context. It also sets `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` and
  `DISABLE_TELEMETRY`, because a setup kept deliberately on this machine has no
  reason to make optional network calls. Alias `claude_local`.
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
  lines for both APIs, for any tool not listed here. Run it through your
  shell with `eval "$(local-llm env)"`.

  These lines are meant to be read and pasted around, so none of them
  contains your API key. Where a key is set, they say
  `export OPENAI_API_KEY="$LOCAL_LLM_API_KEY"` — a reference to the variable
  you already have, not a copy of its value — so the key stays out of your
  scrollback, your shell history and anything you paste into a bug report.
  `eval` still gives the tool the real key. This works because the key can
  only ever come from `LOCAL_LLM_API_KEY` in your environment: it is the one
  setting `settings.toml` deliberately will not hold. With no key set, the
  lines say `dummy`, which is what a router that asks for no key expects.

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

## Memory

Idle models are not stopped, just put to sleep by `llama-server` after
`sleep-idle-seconds` (300 by default, set per-model with `-1` to disable it
for one section) — waking one back up costs a few seconds from the page
cache, not a full reload. Every size shown by `recommend`, `search`, `pull`
and `load` is an estimate, not a measurement: `weights × 1.15 + 1 GiB`, refined
once the file is on disk with the **KV cache** — the memory that grows with
how much context you ask for — read from the model's GGUF header:

```
load unsloth/Qwen3.8-27B-GGUF:Q4_K_XL   ~22.9 GB (weights + KV cache at c=65536)
```

`load`'s budget check only guards explicit loads, honestly: a chat request
naming an unloaded model, or the web UI, bypasses it, and `llama-server`
evicts resident models by *count* only, never by memory pressure — the
reason `--max-models` defaults to 1 (see [Every day](#every-day)).

## Settings

`settings.toml` lives next to `models.ini` and holds everything that is not
a `llama-server` flag:

| Key | Default | Meaning |
|---|---|---|
| `port` | `5678` | Router port |
| `host` | `"127.0.0.1"` | Router bind address |
| `max_models` | `1` | Models allowed resident at once |
| `reserve_gb` | `10` | Memory kept free for the OS and other apps |
| `ui` | `false` | Serve `llama-server`'s web UI by default |
| `default_model` | `""` | Model used when an agent command names none |
| `allow_remote` | `false` | Allow a non-loopback `host` (needs an API key too) |

Every key also has an environment variable, which wins over the file:
`LOCAL_LLM_PORT`, `LOCAL_LLM_HOST`, `LOCAL_LLM_MAX_MODELS`,
`LOCAL_LLM_RESERVE_GB`, `LOCAL_LLM_UI`, `LOCAL_LLM_DEFAULT_MODEL`,
`LOCAL_LLM_ALLOW_REMOTE`. Precedence, highest first: a command's own flag
(where one exists, e.g. `--max-models`), then the environment variable, then
`settings.toml`, then the built-in default. The API key
(`LOCAL_LLM_API_KEY`) is environment-only and never written to disk.

## Troubleshooting

- **`recommend` shows nothing / cannot reach huggingface.co** — listings are
  cached for 24 hours in `STATE/hub-cache.json`; offline with a warm cache
  still works. Offline with a cold cache names that file in the error. A
  proxy or firewall blocking `huggingface.co` produces the same symptom as
  being offline.
- **Port in use** — `local-llm doctor` reports the port taken, and names
  our own router's pid when it is ours; if it is not, find the other
  process with `lsof -i :5678` (swap in your port). `local-llm status`
  shows what our own router thinks is running. Pick another port with
  `local-llm --port N <command>` (the flag goes before the subcommand) or
  `port = N` in `settings.toml`.
- **A download stalls** — Hugging Face throttles unauthenticated downloads,
  and a CDN node can stall. `pull` gives up after three minutes without new
  data and says so; the partial file is kept, so running the same command
  again resumes. Logging in (`hf auth login`) lifts the limit. Downloads use
  plain HTTP by default; set `HF_HUB_DISABLE_XET=0` before running to use
  Hugging Face's xet backend instead.
- **Gated repo** — a repository whose weights need accepting the publisher's
  terms shows `gated` in `search`/`recommend`. Accept the terms on the
  repository's Hugging Face page, then `hf auth login`.
- **`hf token` warns "present but invalid"** — the cached token is stale;
  run `hf auth login --force`.
- **A model won't load / "Compute error" mid-request** — almost always two
  models resident at once on a GPU that only has room for one; see
  [Memory](#memory). Lower `--max-models` back to 1, or `local-llm unload`
  the one you are not using.
- **`llama-server` has no `--models-preset`** — the build is older than
  December 2025, before router-mode presets existed; `brew upgrade
  llama.cpp` on macOS, or rebuild from a current `llama.cpp` tag on Linux.
- **A coding agent still talks to the real Anthropic/OpenAI API** — it was
  started without going through `local-llm claude` / `local-llm copilot` /
  `local-llm env`, so it never got the `*_BASE_URL` variables.

## Uninstall

`local-llm uninstall` undoes what the tool put on the machine, one category at
a time, and lists every path before deleting it:

```
$ local-llm uninstall
  1. models        5 model(s), 73.0 GB on disk
  2. integrations  opencode plugin, opencode tiny agent, shell aliases, completion, Codex provider and profile
  3. state         state and logs, settings.toml
  4. config        models.ini, models.ini.bak
  5. everything
Numbers to remove (e.g. 1 3), q to quit [q]:
```

Choosing models lets you pick which ones; their sections leave `models.ini`
and their files (cache symlink and blob) are deleted. `--models`,
`--integrations`, `--state`, `--config` and `--all` select without the menu,
`--yes` skips the questions, `--dry-run` only prints the plan.
`--restore-shell-line` puts back a `source …/local_llm.zsh` line that
`completion install` had commented out. The config directory is removed only
if nothing else is left in it. The tool cannot delete itself while running,
so the last line prints the command for that (`uv tool uninstall local-llm`
for the standard install).

## Files

| What | macOS and Linux | Env override |
|---|---|---|
| Config directory | `~/.config/local-llm/` | `LOCAL_LLM_CONFIG_DIR` |
| Model preset | `CONFIG/models.ini` | `LOCAL_LLM_PRESET` |
| Settings | `CONFIG/settings.toml` | — |
| State directory | `~/.local/state/local-llm/` | `LOCAL_LLM_STATE_DIR` |
| Logs | `STATE/logs/` | `LOCAL_LLM_LOG_DIR` |
| Downloaded model files | the standard Hugging Face cache | `HF_HOME`, `HF_HUB_CACHE` |
| Codex provider (when configured) | `~/.codex/config.toml` | `CODEX_HOME` |

`$XDG_CONFIG_HOME` and `$XDG_STATE_HOME` are honoured for the first two rows
when set.

## Development

```
uv sync                          # install into .venv from the committed lock file
uv run pytest -q                 # unit tests, no network
uv run ruff check .              # lint
uv run pytest tests/live -m live # one test against the real Hugging Face Hub
tests/e2e/run.sh                 # Docker: install.sh, doctor, recommend, pull,
                                  # up, a chat request, status, down — from nothing
```
