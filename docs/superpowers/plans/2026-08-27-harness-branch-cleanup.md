# Cleanup after the coding-harness branch — handoff

> **For agentic workers:** this is a handoff to a fresh session. Read this file
> and the two documents it names before touching anything. Steps use checkbox
> (`- [ ]`) syntax.

**Goal:** close the residual findings left by the reviews of the
`harness-detection` branch, plus one pre-existing key-exposure fix, so the
branch is fit to merge.

**Where things stand:** branch `harness-detection`, 27 commits ahead of `main`
at 506c5d8. `uv run pytest tests/unit -q` → 331 passed. `uv run ruff check src
tests` → clean. Nothing is pushed and nothing is merged. The feature itself
works: detection, the grouped menu, the Codex provider, the launchers, doctor
and uninstall were all exercised against scratch home directories by two
reviewers and a security reviewer.

**Read first:**
- `docs/superpowers/specs/2026-08-27-coding-harness-detection-design.md` — the
  binding design for the feature these fixes sit on.
- `.superpowers/sdd/2026-08-27-coding-harness-detection/progress.md` — the
  execution ledger: every ruling made during the build, why, and what it costs
  if wrong. It is git-ignored, so it exists on disk only. The per-task and
  per-fix-wave reports sit beside it.

## Global constraints

Copied from the branch's own plan; they still bind.

- Python `>=3.11`; ruff line-length 100; lint rules `E, F, I, UP, B`; every
  module starts with `from __future__ import annotations`.
- Side-effecting integration functions return `list[str]` of finished,
  printable lines. They never print and never raise to the user.
- Path resolvers for another tool's files take `home: Path | None = None,
  env: Mapping[str, str] | None = None`.
- Writes into another tool's config are atomic — temporary file in the same
  directory, then `os.replace` — and keep a `.local-llm.bak` we own. A file
  that will not parse, or that is not UTF-8, is never touched.
- Every question inside `setup.py` goes through the `Io` object.
- The suite must stay green and grow. Every new test is confirmed failing
  against the pre-fix source before it counts.
- Commit with:
  `git -c user.name=nikosntemkas -c user.email=ntemkasn@gmail.com commit -F <file>`
  and the trailers
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` /
  `Claude-Session: https://claude.ai/code/session_01H5uFXL89THd9ciHed8uuMN`.
- Stage files by name. Never `git add -A` — an untracked `graphify-out/`
  directory must stay out of every commit.

## DANGER — read before running the tool by hand

`LOCAL_LLM_STATE_DIR` and `LOCAL_LLM_LOG_DIR` are exported in this user's
shell and point at the real directories under `~/.local/state/local-llm`.
A scratch `HOME` alone does NOT isolate the tool: it honours those variables
and will act on the real paths. During the previous session this deleted the
user's real state directory with `uninstall --all --yes`.

Before any destructive command, clear the whole list the test fixture clears —
`LOCAL_LLM_CONFIG_DIR`, `LOCAL_LLM_PRESET`, `LOCAL_LLM_STATE_DIR`,
`LOCAL_LLM_LOG_DIR`, `LOCAL_LLM_PORT`, `LOCAL_LLM_HOST`, `LOCAL_LLM_API_KEY`,
`XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `CODEX_HOME`, `OPENCODE_CONFIG_DIR` —
and then prove the isolation before acting:

```bash
env -u XDG_CONFIG_HOME -u XDG_STATE_HOME -u LOCAL_LLM_CONFIG_DIR \
    -u LOCAL_LLM_PRESET -u LOCAL_LLM_STATE_DIR -u LOCAL_LLM_LOG_DIR \
    -u LOCAL_LLM_PORT -u LOCAL_LLM_HOST -u LOCAL_LLM_API_KEY \
    -u CODEX_HOME -u OPENCODE_CONFIG_DIR HOME=$SCRATCH \
  uv run local-llm uninstall --dry-run
```

Every path it prints must be under `$SCRATCH`. If one is not, stop.

---

### Task 1: uninstall is the weak half of the tool

The branch made `configure` careful about other people's files and left
`uninstall` where it was, so the two now contradict each other.

**Files:** `src/local_llm/uninstall.py`, `src/local_llm/integrations/opencode.py`,
`tests/unit/test_uninstall.py`, `tests/unit/test_cli_uninstall.py`

- [ ] **Step 1: A `tiny` agent that is not ours survives an uninstall.**
  `uninstall.py:266` `_remove_agent` deletes any `tiny` sub-agent from
  `opencode.json` whoever wrote it, while `opencode.configure` now refuses to
  replace one whose model does not start with `llamacpp/`. Reuse the same
  ownership reader `configure` uses — `opencode.foreign_tiny_model`, which
  parses the JSON rather than pattern-matching the text — and leave a foreign
  agent alone, reporting one line that says so and names the model. Only an
  agent on a `llamacpp/` model is ours to remove.

- [ ] **Step 2: That same write must be atomic.** `_remove_agent` calls
  `config.write_text(...)`, so a crash or a full disk mid-write truncates the
  person's entire opencode configuration — the hazard already fixed on the
  configure side. Route it through `integrations.atomic_write`, the shared
  helper both integrations now use, so it gets the temporary file, the owned
  backup and the rename.

- [ ] **Step 3: An opencode config that is not UTF-8 must not kill uninstall.**
  `uninstall.py:159` and `:266` read `opencode.json` with no encoding and no
  guard, so a non-UTF-8 file makes `inventory()` raise before the plan is even
  printed and *every* `uninstall` invocation dies with a traceback. Do exactly
  what the shell-rc reads now do: pass `encoding="utf-8"`, catch
  `UnicodeDecodeError`, report one line saying the file was left alone, and
  carry on with everything else.

- [ ] **Step 4:** Tests for all three — a foreign `tiny` agent surviving, an
  interrupted write not truncating, and an uninstall completing with a
  non-UTF-8 opencode config present. Run the suite and ruff, then commit.

### Task 2: the API key stops travelling on a command line

Pre-existing, outside the harness branch, and the one security finding worth
fixing on its own.

**Files:** `src/local_llm/router.py`, `src/local_llm/doctor.py`,
`tests/unit/test_router_lifecycle.py`

- [ ] **Step 1:** `router.py` (around line 217) passes the API key to
  `llama-server` as an `--api-key` argument, so any local process can read it
  with `ps`. Recent llama.cpp accepts `LLAMA_ARG_API_KEY` from the
  environment instead. Confirm the flag exists in the installed
  `llama-server` before relying on it — `llama-server --help | grep -i api.key`
  — then pass the key through the spawned process's environment rather than
  its argv.

- [ ] **Step 2:** Old llama.cpp builds do not read that variable. Decide and
  implement the fallback: either keep the argv form when the installed version
  is too old to know the variable, or require the newer build and say so in
  `doctor`. Whichever you choose, `doctor` should be able to tell the person
  which one is in effect.

- [ ] **Step 3:** A test that the key never appears in the spawned argv on the
  supported path. The fake process backend in `tests/unit/fakes.py` records
  every `spawn(args, log_path)`, so assert against `backend.spawned`.

### Task 3: the narrow correctness items

One commit, or one per bullet if they read better apart.

**Files:** `src/local_llm/shellrc.py`, `src/local_llm/cli.py`,
`src/local_llm/harnesses.py`, `src/local_llm/integrations/opencode.py`,
`src/local_llm/integrations/codex.py`

- [ ] Two complete marked blocks in one rc file — from an older version, or a
  paste — leave the second behind: `remove_block` removes the first and
  uninstall still reports "aliases removed". Verified live. Remove every
  block, or refuse and report, the way a half-marked file is now handled.
- [ ] `_integrate_shell` prints "left exactly as it is" about a file typer's
  completion installer appended two lines to moments earlier. Either install
  completion after the check, or say what actually happened.
- [ ] The non-loopback warning prints in the menu but not for `integrate
  codex` or `integrate opencode` run directly, and does not mention that the
  base URL is always plain `http`, so with a remote host prompts and code
  cross the network unencrypted. Move the note into the two `configure()`
  functions and add that clause.
- [ ] `opencode.configure` re-reads the plugin for its diff without a guard,
  keeping the time-of-check gap the Codex path closes by re-parsing inside its
  guarded write.
- [ ] `harnesses.parse_choice` duplicates `cli._numbers` and the two now give
  different messages for the same bad input. Pick one.
- [ ] Uninstall can leave a zero-byte `~/.codex/config.toml` when our tables
  were the whole file, rather than removing a file we created.

### Task 4: hardening with no realistic failure — do only if cheap

- [ ] `integrations/__init__.py` chmods the temporary file by path rather than
  `os.fchmod` on the descriptor still open a line above.
- [ ] No `fsync` before the rename: atomic against a crashing process, not
  against power loss.
- [ ] An alias defined elsewhere in the rc, outside our marked block, is
  silently shadowed by ours or shadows ours depending on file order.
- [ ] `doctor.run_checks` loads `models.ini` a second time to build its
  harness context.
- [ ] `_help_paragraph` in `cli.py` hard-wraps two help paragraphs at 76
  columns, so they do not reflow on a narrow terminal.

### Task 5: decide what `local-llm env` should do about the key

`local-llm env` prints shell `export` lines — both base URLs, the model, the
context window, and the API key's **value** — so that any OpenAI- or
Anthropic-compatible tool the registry does not cover can be pointed at the
router with `eval "$(local-llm env)"`. Printing the value is inherent to that
job; the exposure is that it lands in scrollback, shell history and pasted
terminal output.

- [ ] Decide and implement one of: leave it (documented), add `--no-key` for
  the common case where the router needs no key, or print
  `export OPENAI_API_KEY="$LOCAL_LLM_API_KEY"` so the value is referenced
  rather than reproduced. Whatever you choose, say it in the README where the
  command is documented.

---

## Finishing

- [ ] Full suite and ruff green.
- [ ] A whole-branch review of everything since 506c5d8, on the most capable
  model available, pointed at this file's task list so it can tell you what is
  still open.
- [ ] `superpowers:finishing-a-development-branch` to decide how
  `harness-detection` reaches `main`. Nothing has been pushed; the branch is
  local only.
