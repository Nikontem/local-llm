# Commands

Every command, with its flags and what it prints. `local-llm --help` and
`local-llm <command> --help` say the same thing from the terminal.

## Running the router

- **`local-llm up`** — starts the router detached and writes a timestamped
  log.

  ```
  Router is up.
    url:      http://127.0.0.1:5678/v1
    models:   local-llm models
    logs:     local-llm logs -f
  ```

  Two flags override what `models.ini` says, for every model the router
  serves, for this run of the router only: `--context N` loads every model
  at exactly `N` tokens of context, and `--no-reasoning` turns reasoning
  (thinking) off for every model that has it. See [Overrides for one run](#overrides-for-one-run)
  below before reaching for the first one.

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

  When the router was started with `--context` or `--no-reasoning`, two more
  lines follow `config:` and name the override in effect (`context:  262144
  for every model (--context)`, `reasoning: off (--no-reasoning)`); they are
  read from the running process, not from a state file, so they are what the
  models are actually being served with. Once a model is actually resident,
  its line also shows the context it was given: `qwen2.5-0.5b  pid 52995  1.2 GB  ctx 16384`. That number comes
  straight from the router's own API, so it is what the model actually
  loaded at — not the floor written in `models.ini` (see
  [Memory](configuration.md#memory)) — and can differ between one load of
  the same model and the next.
- **`local-llm doctor`** — checks Homebrew, `llama-server`, the `hf` command
  and token, config and state directories, `models.ini`, port, and the
  coding agents this tool knows about, each failure paired with the command
  that fixes it. Two checks are specific to context sizing:
  - **`context sizing`** — names any section (or the `[*]` wildcard block)
    still using `c` or `ctx-size` to pin a fixed context instead of `fit-ctx`.
    This is informational (`ok`, not a failure): `c` still works, it just
    means that section never gets the load-time adjustment described in
    [Memory](configuration.md#memory). The fix text does not say to turn
    `c = N` straight into `fit-ctx = N` — that recreates the same problem
    with an unreachable floor, which makes the fitter give up entirely. It
    suggests a modest floor instead (`fit-ctx = 16384`), or deleting the `c`
    line and running `local-llm pull` again.
  - **`model fit`** — a warning when the current router log contains
    `failed to fit params to free device memory`, meaning some model loaded
    at its full, unreduced size instead of being adjusted to what was
    actually free, and may swap or fail under load. See
    [Troubleshooting](troubleshooting.md) for the same message and remedies.
- **`local-llm models`** (alias `ls`) — every section name in `models.ini`
  with its size on disk; pass any of them as the `model` field of an OpenAI
  request, no restart needed.
- **`local-llm load <name>...`** — preloads one or more models and refuses
  when the combined estimate exceeds the memory budget, unless `--force`. A
  router started with `--context N` serves every model at `N` whatever its
  section says, so the estimate uses `N` too and says so:
  `weights + KV cache at c=262144 (router --context)`.
- **`local-llm unload <name>`** — releases a model immediately instead of
  waiting for it to go idle.
- **`local-llm logs -f`** — follows the current log; `-n 5` shows the last
  five lines without following.
- **`local-llm down`** — stops the router and every model process under it,
  cleaning up any orphan it finds.
- **`local-llm restart`** — down, then up, then reloads every model that was
  resident before (`--no-restore` to skip that, `--ui`/`--no-ui` to change
  the web UI mode). A `--context` or `--no-reasoning` the running router was
  started with is carried forward unless you say otherwise: `--context 0`
  goes back to what `models.ini` says, `--reasoning` turns reasoning back on.
- **`local-llm ui`** — opens `llama-server`'s own web interface, starting the
  router with `--ui` first if it is not running that way.

`up` and `restart` both take `--max-models N`, capped at 1 by default. This
is not caution for its own sake: `llama-server` evicts loaded models only by
count, never by memory pressure, so two large models can both stay resident
until the GPU runs out of memory mid-request and every call fails. Raise it
only when you know two configured models together fit the budget.

### Overrides for one run

`up` and `restart` also take `--context N` and `--no-reasoning`, and both do
the same thing in the same way: they become flags on the `llama-server`
router process itself, which applies its own command line to every model it
spawns *ahead of* that model's section in `models.ini`. So `--context 262144`
loads every model at exactly 262144 tokens whether its section says `c =
65536` or `fit-ctx = 16384`, and `--no-reasoning` becomes `llama-server`'s own
`--reasoning off`, which tells every model's chat template not to think. `models.ini` is
not touched; `up` without the flag serves it exactly as written again. Both
are also environment variables, `LOCAL_LLM_CONTEXT` and
`LOCAL_LLM_NO_REASONING`, and neither is a `settings.toml` key on purpose
(see [Settings](configuration.md#settings)).

Two things follow from "every model". A pinned context switches off the
load-time fitting described in [Memory](configuration.md#memory) for every
section, so a number that suits the model you are about to load can be far
too large for another one the router autoloads later; `load` budgets
against the override and refuses when it does not fit, but a chat request
naming an unloaded model bypasses that check as it always has. And a router
flag *replaces* the same key in a section rather than merging with it, which
is why `--no-reasoning` uses `llama-server`'s dedicated `--reasoning off`
rather than `--chat-template-kwargs`: a section's own `chat-template-kwargs`
(at the time of writing only the gpt-oss sections, which use it for
`reasoning_effort`) is left intact.

`--no-reasoning` holds for every client of the router, including the coding
agents: a Codex session started with `codex --profile local-llm` against a
router running with `--no-reasoning` gets answers with no reasoning, even
though Codex asks for reasoning on every request, because the setting is
applied where the prompt is built rather than negotiated per request.

## Models

- **`local-llm browse-models [--use coding|general|small|vision]
  [--include-finetunes] [--limit N] [--refresh]`** — setup step 3 on its own:
  the numbered table of what fits this machine, a prompt that takes **several
  numbers at once** (`1 3` downloads two models), and `s <text>` to search
  Hugging Face without leaving the menu. Search results join the numbering
  rather than replacing it, so a model from the first listing is still
  pickable by the number it was given there. Each pick asks for its
  quantization, with the one suggested for this machine as the default, and is
  then downloaded and written into `models.ini` as its own tuned section — the
  same work `pull` does, once per pick. A model already in `models.ini` is
  named and skipped, not downloaded again, and a repository the Hub will not
  serve — gated, unreachable, or too large for the disk — is reported and the
  run goes on to the rest of the picks, naming what it could not add and
  exiting non-zero at the end. Nothing is
  written to `settings.toml`: the default model stays whatever setup left it.
  It is a menu and needs a terminal, so it refuses to run in a pipe or a
  script and points at `recommend --json` and `pull` instead.
- **`local-llm recommend [--use coding|general|small|vision]
  [--include-finetunes] [--pick]`** — the same table as setup step 3, on
  demand. "Computed from this machine" means every number reflects your RAM,
  chip and GPU (the machine detection in [setup](setup.md#first-run)), not a fixed
  list; the quantization
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
    context: chosen at load time to fit this machine, never below 16384
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
  pins the context to that exact number (writes `c`, not `fit-ctx`) instead of
  leaving it for `llama-server` to choose at load time — see
  [Memory](configuration.md#memory); `--set KEY=VALUE` writes any
  `llama-server` flag as-is (repeatable, last one wins); `--no-tuning` writes
  only `model`, `mmproj`, `fit-ctx` (or `c`, with `--context`) and
  `n-predict`, skipping sampling values.
- **`local-llm add PATH [--mmproj PATH] [--name NAME]`** — the same tuning
  and section-writing as `pull`, for a GGUF file you already have on disk.
- **`local-llm tune <model> [--repetitions N] [--json]
  [-y]`** — measure how fast a model actually answers on this machine, and offer
  to keep the settings that answered fastest. Everything else `local-llm` writes
  is estimated from the model file and your memory; this is the one command that
  runs the model and times it. It measures four settings — the batch size and
  micro-batch size, whether flash attention is used, and whether the key-value
  cache is stored at full or reduced precision — against a workload shaped like a
  coding agent's: four thousand tokens already in the conversation, four thousand
  more sent, and a short reply. Each setting is measured on its own against your
  current configuration, so a combination of two changes is never tried. Expect
  about five minutes for six settings on a 16 GB model, and well under a minute
  on a small one. If the router is holding a model it is unloaded for the
  duration and reloaded afterwards, because a second copy of a model in memory
  makes every number meaningless. Nothing is written to `models.ini` until you
  say so. Measurement runs through `llama-bench`, the program shipped alongside
  `llama-server`; `--repetitions` controls how many times each setting is
  measured (default 3); `-y`/`--yes` answers both of the questions this command
  can ask — unloading a resident model, and writing the winner — so an
  unattended run needs it; without it and without a terminal to ask in, it
  prints what it would write and changes nothing. It does not touch the number
  of GPU layers or the context size — those are decided elsewhere (see
  [Memory](configuration.md#memory)), because this command tunes speed, not
  fit. It will also decline to measure a setting it could not write: a
  `cache-type` change alters what your model's context costs in memory, so a
  cache the model could not fit at its configured context is named and skipped
  rather than offered. Output from a 1 GB model:

  ```
  measuring Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M with llama-bench, 5 settings, 3 runs each
    measuring baseline
    measuring batch 1024/256
    measuring batch 4096/1024
    measuring flash-attn off
    measuring cache q8_0

    baseline          b=2048 ub=512 fa=auto cache=f16                 1414 pp/s   134.2    ±0.4 tg/s     4.8 s
    batch 4096/1024   b=4096 ub=1024 fa=auto cache=f16                1417 pp/s   127.4    ±3.9 tg/s     4.9 s  +2.0%
    batch 1024/256    b=1024 ub=256 fa=auto cache=f16                 1351 pp/s   127.7    ±2.7 tg/s     5.0 s  +4.8%
    cache q8_0        b=2048 ub=512 fa=auto cache=q8_0                1275 pp/s   117.8    ±2.4 tg/s     5.4 s  +12.1%
    flash-attn off    b=2048 ub=512 fa=off cache=f16                  1202 pp/s   100.3    ±1.7 tg/s     6.0 s  +24.0%

    Each setting was measured against the baseline on its own. Combinations of two changed settings were not tried.
    The winner and the last place are what held across repeated runs; a few percent between the rows in between is not meaningful. A busy machine can move these numbers by more than that on its own, so a surprising result is worth measuring again.

  Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M is already the fastest of the settings tried. Nothing to change.
  ```

  Every percentage here reads as "slower than the baseline by this much" —
  `batch 4096/1024` at +2.0% is the closest thing to a rival, and it still
  lost. That closeness is exactly what the `±` column is for: its generation
  rate wobbled by ±3.9 tokens/second across its own three repeated runs, wider
  than the two-percent gap that supposedly separates it from the baseline, so
  the difference is noise rather than a finding. The baseline wins outright
  here, so `local-llm` says so and changes nothing. A closer race, where the
  fastest setting is not the baseline but still within five percent of it (the
  margin `tuning_report.MEANINGFUL_MARGIN` sets), gets a similar refusal: the
  command names the near-tie, explains that a margin that small does not
  survive a second run, and still writes nothing.

  When a setting clears that margin, the command instead names it as the
  fastest, lists the exact `models.ini` lines it would change, and asks
  before writing them; `--yes` writes them without asking, and with no
  terminal to ask in and no `--yes`, it prints what it would write and
  changes nothing instead. That did happen on a larger model: on a 16.5 GB
  model, `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL`, `batch
  4096/1024` measured at 12.47 seconds a turn against the baseline's 15.71 —
  comfortably past the five-percent margin — while turning flash attention off
  cost 17.64 seconds.

- **`local-llm remove NAME [--delete-files]`** — removes a section, and with
  the flag, the model file(s) it points at, after listing them and asking.
- **`local-llm edit`** — opens `models.ini` in `$EDITOR` directly; everything
  above is a convenience over hand-editing this plain INI file, which stays
  the source of truth.

## Uninstall

`local-llm uninstall` undoes what the tool put on the machine, one category at
a time, and lists every path before deleting it:

```
$ local-llm uninstall
  1. models        5 model(s), 73.0 GB on disk
  2. integrations  opencode plugin, opencode tiny agent, shell aliases, completion, Codex provider and profile, the backup copies we made
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
`completion install` had commented out.

Before editing any file of yours, this tool copies it aside as
`<name>.local-llm.bak`. Nothing else ever writes a file by that name, so
uninstall knows those copies are its own — but each one is the only record of
what your file said beforehand, so it never deletes them without asking. The
question comes after you confirm the rest; answer no and it prints where they
are so you can delete them yourself later. `--delete-backups` and
`--keep-backups` answer it in advance, and a run with `--yes` keeps them.
Worth clearing out once you are sure: the copy beside an opencode config holds
whatever API keys were in it. The config directory is removed only
if nothing else is left in it. The tool cannot delete itself while running,
so the last line prints the command for that (`uv tool uninstall local-llm`
for the standard install).
