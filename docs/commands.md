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

  Once a model is actually resident, its line also shows the context it was
  given: `qwen2.5-0.5b  pid 52995  1.2 GB  ctx 16384`. That number comes
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
