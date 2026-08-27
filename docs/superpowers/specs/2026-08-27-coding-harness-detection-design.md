# Coding-harness detection and configuration — design

Date: 2026-08-27
Status: approved in discussion, written for review

## 1. Purpose

A *coding harness* is a command-line program that drives a large language
model to write code — Claude Code, OpenAI's Codex CLI, opencode, GitHub
Copilot CLI, aider and others. Today `local-llm` knows about three of them
and treats each differently for historical reasons: opencode gets a plugin
written into its own configuration, Claude Code and Copilot CLI get a
launcher command that sets environment variables and starts the program, and
everything else is left to `local-llm env`. Nothing detects what the person
actually has installed, and the first-run wizard's sixth step hard-codes that
short list.

This feature adds one place that knows every harness the tool supports, uses
it to detect what is installed, and offers the right kind of configuration for
each — writing a provider into the harness's own configuration file where the
harness officially supports one, and otherwise offering a launcher command
that points the harness at the router for one session. The same list drives
the first-run wizard, a new interactive `local-llm integrate` menu, the
`local-llm doctor` report, and `local-llm uninstall`, so the four can never
disagree about which harnesses exist or what was written for them.

Nothing about the router, the model preset file, or the existing commands
changes. This is a layer on top of what is already there.

## 2. Terms used in this document

- **Harness** — a coding agent program a person runs in a terminal.
- **Provider integration** — `local-llm` writes a named model provider into
  the harness's own configuration file, so the harness offers the router's
  models every time it starts, with no help from this tool. Only harnesses
  that officially document such a file get this treatment. We can detect what
  we wrote and remove it again.
- **Launcher** — `local-llm` writes nothing into the harness's
  configuration. A command such as `local-llm claude` sets the environment
  variables the harness reads and then replaces itself with the harness
  process, so that one session talks to the router. Optionally a short shell
  alias is added to the person's shell startup file.
- **Informational entry** — the harness cannot be pointed at the router at
  all. `local-llm` writes nothing and launches nothing; it prints one short
  explanation of why, and where to look instead.
- **The router** — the single `llama-server` process this tool manages. It
  serves an OpenAI-style endpoint (a URL ending in `/v1`) and an
  Anthropic-style endpoint (the same host and port without `/v1`).

## 3. What ships in this release

| Harness | Binary on PATH | Kind | Mechanism |
|---|---|---|---|
| OpenAI Codex CLI | `codex` | provider | two tables in `~/.codex/config.toml`; run as `codex --profile local-llm` |
| opencode | `opencode` | provider | existing plugin file plus the optional `tiny` helper agent |
| Claude Code | `claude` | launcher | `local-llm claude`, alias `claude_local` |
| GitHub Copilot CLI | `copilot` | launcher | `local-llm copilot`, alias `copilot_local` |
| aider | `aider` | launcher | `local-llm aider`, alias `aider_local` |
| Qwen Code | `qwen` | launcher | `local-llm qwen`, alias `qwen_local` |
| Gemini CLI | `gemini` | informational | explanation only |
| Antigravity CLI | `agy` | informational | explanation only |

Deliberately excluded, with the reason recorded here so the decision is not
re-litigated by accident:

- **Charm Crush, Block Goose, Continue CLI, Factory Droid, Mistral Vibe** all
  do document a configuration file for a custom OpenAI-compatible provider and
  would qualify as provider integrations. They are left out of this release
  only because four of the five file formats were read from search summaries
  rather than confirmed against current documentation, and each one is a new
  module with its own removal logic and tests. The registry is shaped so each
  becomes one new entry later.
- **Cursor, Windsurf, Warp, Zed, Cline, Amp, Auggie, Kilo, Forge, Kimi,
  Junie, OpenHands** configure their endpoint through a graphical settings
  panel or an interactive menu that cannot be scripted, or their file format
  could not be confirmed. A menu row that detects a program and then tells the
  person to go and configure it by hand is noise, and every such row is a fact
  that will go stale. `local-llm env` already prints the exports for them.
- **Devin, AWS Kiro, Atlassian Rovo Dev** have no way to point at an arbitrary
  endpoint at all. Devin's interface is a proprietary session-based service;
  Kiro and Rovo Dev extend through MCP servers and a fixed model picker.
- **The Antigravity desktop application and the Antigravity IDE** are
  graphical applications, not terminal harnesses, and neither has a
  user-facing setting for a custom endpoint. Only the Antigravity CLI appears,
  as an informational entry.

## 4. The registry module

A new module, `src/local_llm/harnesses.py`, holds one record per harness:

```python
@dataclass(frozen=True)
class Harness:
    key: str                 # "codex", stable, used in commands and tests
    title: str               # "OpenAI Codex CLI", shown to people
    binary: str              # "codex", what we look for on PATH
    kind: str                # "provider" | "launcher" | "informational"
    summary: str             # one short line: what choosing this does
    note: str = ""           # informational entries: the explanation to print
    alias: tuple[str, str] | None = None   # ("claude_local", "local-llm claude")
    configure: Callable[[State, bool], list[str]] | None = None
    remove: Callable[[State], list[str]] | None = None
    status: Callable[[State], str] | None = None
```

`configure(state, yes)` and `remove(state)` return a list of finished,
printable lines, which is how every side-effecting function in this codebase
already reports itself; the caller only prints them. `status(state)` returns
one of `missing`, `same`, `different` or `unreadable` for provider entries,
reusing the vocabulary the opencode plugin check already uses, and
`configured` or `not configured` for launcher entries, which means only
whether that entry's alias line is present inside the marked block in the
shell startup file. Informational entries have none of the three callables and
never claim a status.

`detect(which)` runs the given lookup function (`shutil.which` in real use, a
fake in tests) over every entry and returns which are installed and which are
not. Detection is by name on the executable search path only: no version
probing, no running the harness, so a harness that hangs without a terminal
cannot hang our wizard.

The registry is a static catalogue, not saved state. Nothing records which
harnesses we have configured; everything is re-detected by reading files on
disk, which is exactly how `local-llm uninstall` already works.

Codex's file handling lives in a new `src/local_llm/integrations/codex.py`,
shaped like the existing `integrations/opencode.py`: a path resolver, a
pure-data builder for the two tables we own, a status function and
install/remove functions. The registry entry only points at them.

opencode's existing `--agent/--no-agent` flag has no place in a registry entry
that takes only a state object and a yes flag. Chosen from the menu, opencode
always includes the `tiny` helper agent, which is what the wizard does today
and what the flag already defaults to. Anyone who wants the plugin without the
helper agent runs `local-llm integrate opencode --no-agent`, which keeps
working.

A `tiny` agent is only ours to write over or take away when the model it names
begins with `llamacpp/`, the prefix under which the plugin registers every
model the router serves. Anything else — an agent the person set up themselves,
very possibly on a model they pay for — belongs to them. `configure` refuses to
replace one, and with `--yes`, where nothing can be asked, it does not even
offer to; uninstall refuses to delete one. Both report one line naming the
model instead, so the refusal is never silent.

## 5. Detection paths

`codex_paths(home=None, env=None)` follows the signature convention every
other path resolver in this codebase uses, so tests can pass a scratch
directory. It honours `CODEX_HOME` (Codex's own variable for relocating its
directory) and otherwise resolves `~/.codex/config.toml`.

Codex also reads a project-level configuration file — `./config.toml` or
`.codex/config.toml` walking up to the repository root — which takes
precedence over the file in the person's home directory when the directory is
trusted. That means a project can silently shadow the profile we wrote. The
tool cannot prevent this, so `local-llm integrate --help` and the README say
it plainly: if Codex is not using the local provider inside one particular
repository, look for a project-level configuration file before assuming
something is broken.

## 6. The menu

The same rendering and the same choices serve two entry points: step 6 of
`local-llm setup`, and `local-llm integrate` with no arguments.

```
6. Coding agents

  Configured inside the agent
   1. OpenAI Codex CLI    ~/.codex/config.toml: provider and profile local-llm   · not configured
                          experimental: needs a recent llama.cpp build
   2. opencode            plugin, and a tiny helper agent bound to your smallest model   · configured

  Launched through local-llm
   3. Claude Code         local-llm claude   · alias claude_local installed
   4. aider               local-llm aider    · alias aider_local

  Detected, but cannot use the router
      Gemini CLI          speaks only Google's own API format, which the router does not serve

  Not found: GitHub Copilot CLI, Qwen Code, Antigravity CLI

Numbers to configure (e.g. 1 3), a for all, n for none [a]:
```

Behaviour:

- Only installed harnesses get a numbered row. The "Not found" line names the
  rest so the person can see the whole supported list without reading the
  documentation.
- An informational harness appears without a number, since there is nothing to
  choose, and only when its binary is actually installed. Because most people
  will therefore never see the Gemini CLI explanation, the same text also
  appears in `local-llm integrate --help` and in the README.
- Number parsing reuses the existing helper from the uninstall menu: a
  space-separated list of numbers, `a` or `all` for every row, `n` or `none`
  to skip, out-of-range input refused with a message.
- With `--yes`, every detected provider and launcher entry is configured
  without asking, and informational rows are printed as text. This is the path
  the wizard uses.
- When no terminal is attached and `--yes` was not given, the menu is printed
  as a plain list, nothing is configured, and the command exits successfully.
  This keeps the wizard usable in a pipeline without it appearing to hang on a
  prompt nobody can answer.
- When no harness at all is detected, shell completion and the `local_llm`
  alias are still offered, and one line says no coding agents were found on
  the executable search path and that `local-llm env` prints the exports for
  any other tool.
- One harness failing — a directory that cannot be written, a configuration
  file that will not parse, a permissions error — prints one line and the loop
  continues to the next harness. This mirrors the discipline the uninstall
  command already follows, where a failed deletion never aborts the run.

Shell completion and the `local_llm` alias stay a separate question asked
before the harness menu, exactly as today.

The named subcommands remain: `local-llm integrate opencode
[--agent/--no-agent] [--yes]` keeps working because the README documents it,
and `local-llm integrate codex [--yes]` is added alongside it. Everything else
is reached through the menu. The bare `local-llm integrate` menu is added as a
callback that runs when no subcommand is named, and must not change how the
existing subcommands or `--help` behave.

## 7. Writing the Codex provider

The two tables `local-llm` owns in `~/.codex/config.toml`:

```toml
[model_providers.local-llm]
name = "local-llm"
base_url = "http://127.0.0.1:5678/v1"
wire_api = "responses"

[profiles.local-llm]
model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL"
model_provider = "local-llm"
```

Facts behind those values:

- `base_url` must include the `/v1` suffix, which `settings.openai_base_url`
  already provides.
- `wire_api` must be `"responses"`. Codex removed support for the older Chat
  Completions wire format on 1 February 2026, and `"chat"` is now a startup
  error rather than a fallback.
- `env_key` names an environment variable Codex reads the API key from. It is
  omitted when no API key is configured, in which case Codex sends no
  authorisation header, which is what a bare `llama-server` wants. When
  `LOCAL_LLM_API_KEY` is set — the router does enforce a key when one is
  configured — the line `env_key = "LOCAL_LLM_API_KEY"` is written so Codex
  sends it.
- The profile's `model` is the configured default model if there is one,
  otherwise the smallest model in `models.ini`, which is the same rule the
  opencode helper agent already uses. When `models.ini` holds no models at
  all, only the provider table is written and one printed line says the
  profile was skipped and why.
- No top-level `profile = "local-llm"` line is written. Making the local
  profile the machine-wide default for Codex is a bigger change to someone's
  setup than this feature should make on its own; `codex --profile local-llm`
  (or the offered `codex_local` alias) selects it per run.

**This integration is labelled experimental in the menu, in the printed
output, and in the README.** `llama-server` does implement `/v1/responses`,
but the work to make that endpoint behave the way Codex actually talks to it —
the full streaming event sequence, and accepting the tool definitions Codex
sends — is in an open, unmerged llama.cpp pull request. On an older
`llama-server` build, plain chat may work while tool calls fail. Writing the
configuration is still worth doing: it is correct, it costs nothing, and it
starts working the day that fix lands. Pretending it is as solid as the
opencode integration would not be honest.

Writing and removing the file:

1. `tomlkit` is added as a runtime dependency — the fifth, after `typer`,
   `rich`, `huggingface_hub` and `psutil`. Python's standard library can read
   TOML but not write it. `tomlkit` parses a file into a document that can be
   edited and written back with comments, key order and formatting intact.
   A hand-rolled marker-block approach is explicitly rejected here: a TOML
   table has several equivalent textual forms, so a header pattern can miss an
   existing table and emit a second one, and a duplicate table makes Codex
   fail at startup with a duplicate-key error or hang with no visible error.
   The shell startup file keeps its marker block, because there the target text
   has exactly one canonical form.
2. When the file does not exist, it is created holding only our two tables.
3. When it exists and holds neither of our tables, the tables are added and
   the rest of the file is written back unchanged. Missing parent tables are
   created in a way that does not emit a redundant `[model_providers]` header.
4. When our tables exist and already match what we would write, nothing is
   written and one line says so.
5. When our tables exist but differ — the person edited them, or the router's
   port changed — a difference is shown and replacing them is confirmed,
   bypassed by `--yes`. This is the same treatment the opencode plugin gets.
6. When the file cannot be parsed, it is never touched. The parse error's line
   and column are printed along with the exact TOML to paste in by hand, which
   mirrors how a commented opencode configuration is handled today.
7. Every write is atomic — written to a temporary file in the same directory
   and moved into place — and keeps a `.bak` copy of the previous contents,
   the same discipline `models.ini` gets. This file belongs to another tool
   and cannot be regenerated, so it earns the backup.

## 8. Launchers and aliases

`local-llm claude` and `local-llm copilot` keep their current shape. Two
additions to the Claude Code environment, both because a local endpoint is not
Anthropic's: `ANTHROPIC_DEFAULT_HAIKU_MODEL` is set to the chosen model,
replacing the now-deprecated `ANTHROPIC_SMALL_FAST_MODEL`, so Claude Code's
background work does not ask the router for a Haiku model it has never heard
of; and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` is set, because a
deliberately local setup has no reason to make optional network calls. Both
variables react to being set at all rather than to their value.

The two Copilot variables for prompt and output token limits are kept as they
are. They could not be found in current Copilot documentation, so they are
described in the README as best-effort rather than as documented settings.
Removing them would change behaviour for people already relying on them, and
setting a variable a program ignores is harmless.

Two new launcher commands:

- `local-llm aider [MODEL] [-- ARGS...]` sets `OPENAI_API_BASE` and
  `OPENAI_API_KEY` and runs `aider --model openai/<model> ARGS`. The
  `openai/` prefix is what tells aider to route through the custom endpoint.
  No configuration file is written: aider only warns that it does not
  recognise the model's context window and never enforces a token limit, so
  the optional model-metadata file is not worth the removal logic it would
  add.
- `local-llm qwen [MODEL] [-- ARGS...]` sets `OPENAI_API_KEY`,
  `OPENAI_BASE_URL` and `OPENAI_MODEL` and runs `qwen ARGS`. All three must be
  set and non-empty or Qwen Code ignores the OpenAI-compatible path entirely.
  Qwen Code is the community continuation of the original Gemini CLI codebase,
  which makes it the honest answer for anyone arriving from Gemini CLI wanting
  local models.

Aliases become per-harness and add-only. The fixed three-entry table in
`shellrc.py` becomes a list supplied by the caller: `local_llm` is always
offered as the tool's own short form, and each launcher entry owns its alias
(`claude_local`, `copilot_local`, `aider_local`, `qwen_local`). The Codex entry
owns `codex_local` for `codex --profile local-llm`, since a provider
integration still benefits from a short invocation. Only the aliases for
harnesses that are installed and chosen are added.

Add-only matters: an alias line already inside the marked block is never
removed by a later run of `setup` or `completion install`, even if that
harness has since been uninstalled from the machine. Otherwise someone who
installed all the aliases and later removed one harness would silently lose an
alias they still use. Only `local-llm uninstall` removes alias lines, and it
removes the whole marked block as it does today.

## 9. Uninstall

`local-llm uninstall` gains no new category. Under `integrations` it now also
finds and removes whatever the harness registry wrote.

The inventory grows one field per provider integration — for Codex, the path
of a `~/.codex/config.toml` that contains our tables, plus its `.bak` if we
made one — found by the same read-the-file detection the opencode fields
already use, with the path resolver threaded a scratch home and environment in
tests exactly as `opencode_paths` is.

Removal goes through each registry entry's `remove(state)` rather than being
re-implemented inside `uninstall.py`, so the knowledge of what a harness
integration consists of lives in exactly one place. For Codex that means:
parse the file with `tomlkit`; delete `model_providers.local-llm` and
`profiles.local-llm`; delete each parent table only if removing our key left
it empty and it carries no comments of the person's own; write atomically with
a backup. If a top-level `profile = "local-llm"` line is ever written by a
future version, remove that too.

If taking our tables out leaves nothing at all — no setting of theirs, not
even a comment — the file itself was ours: it did not exist before `configure`
wrote it, and a zero-byte `~/.codex/config.toml` left behind is not the
machine put back the way it was found. So the file is deleted rather than
written empty. The one exception is a config that is a symbolic link into a
dotfiles repository, where deleting the link would break an arrangement the
person made on purpose; there the emptied file is written through the link as
usual.

The opencode side follows the same rule for the part it owns. Uninstall
removes the `tiny` helper agent only when it is ours by the test in section 4,
writes the change atomically with a backup like every other write into a
foreign file, and leaves a config that is not valid UTF-8 alone with one line
saying so rather than failing the whole run.

Two deliberate positions:

- If the person edited our tables after we wrote them, uninstall removes them
  anyway and says what it removed. A half-owned provider table pointing at a
  port nothing is listening on is worse than removing an edit.
- If the file will not parse, nothing is touched and the exact lines to delete
  by hand are printed. This matches how a commented opencode configuration is
  reported today.

The plan printed before anything is deleted lists every affected path, as it
already does, so `--dry-run` shows the Codex file without touching it.

## 10. Doctor

`local-llm doctor`'s "agents" check currently hard-codes three names. It reads
the registry instead, so a harness added to the table appears in the report
without a second edit. The line keeps its shape — which harnesses were found,
which were not — and provider entries gain their configured status.

One new check, because a written configuration file can go stale in a way the
launcher path cannot: if `~/.codex/config.toml` holds our provider table and
its `base_url` no longer matches the router's current address, the check warns
and names `local-llm integrate codex` as the fix. The opencode plugin needs no
such check because it reads the address from the environment at startup rather
than baking it in.

## 11. Errors and safety

The existing rules in the tool's design still hold; these are the additions
this feature needs.

- Never write into a configuration file that cannot be parsed. Print the
  parse error and the snippet to paste instead.
- Never emit a second copy of a table we already own. Parse and replace.
- Never write a foreign configuration file non-atomically, and always keep a
  `.bak`.
- Never suggest a way to make Antigravity use the router other than its own
  Python software development kit. The community proxy patches that intercept
  Antigravity's internal API are, per Antigravity's published terms, grounds
  for account suspension. The tool must not point anyone at them.
- Every failure inside the menu loop is one printed line; the loop continues.
- When the router's host is not a loopback address, a provider integration
  writes that address into another tool's configuration file, which is a
  wider exposure than a launcher's per-session environment. It is said once,
  and by whichever part of the tool is in a position to say it first. The
  menu prints one line before asking which agents to configure, so the choice
  can be made knowing it, and it tells each agent it then configures that the
  warning has already been given. A provider configured on its own, by
  `local-llm integrate codex` or `local-llm integrate opencode`, is told
  nothing by any menu, so its own `configure` prints the line instead. Either
  way the person hears it exactly once, and the line says that the base
  address is always plain `http`: with a host that is not this machine, the
  agent's prompts and the code it sends cross the network unencrypted.

## 12. What the informational entries say

Gemini CLI:

> Gemini CLI cannot be pointed at the router. It has no setting for an
> OpenAI-compatible endpoint — the feature request for one was closed and its
> pull request was never merged — and its `GOOGLE_GEMINI_BASE_URL` variable
> needs a server that speaks Google's own Gemini request format, which
> `llama-server` does not. Gemini CLI is still actively released and still
> works with a paid Gemini or Gemini Enterprise API key, or a Gemini Code
> Assist Standard or Enterprise licence; access ended in June 2026 for the
> free tier and for the paid consumer plans. For a Gemini-CLI-shaped tool that
> does run local models, Qwen Code is a continuation of the same codebase and
> `local-llm qwen` configures it.

Antigravity CLI:

> Antigravity CLI is Google's replacement for Gemini CLI for consumer
> accounts. It accepts `GOOGLE_GEMINI_BASE_URL` for a Gemini-format endpoint,
> so it would need a translating proxy in front of the router, which this tool
> does not ship. If you want Antigravity to drive a local model, its Python
> software development kit supports that officially: `pip install
> google-antigravity`, then `LocalOpenAIAgentConfig(base_url=<the router's
> OpenAI URL>, model=<a model name>)`.

Wording in the code stays this short. Dated detail — transition dates, which
Google plans lost access — belongs in the README, where it can be corrected
without touching code, because everything in this paragraph has already
changed twice in one year.

## 13. Testing

Following the layers the tool's design already sets out, all of this is layer
one, unit tests through pytest with nothing real touched.

- The registry: detection with an injected lookup function returning a fixed
  set of installed binaries; grouping and ordering of the rendered menu; the
  "Not found" line; the no-harness-detected case.
- The menu: numbers, `a`, `n`, out-of-range input refused, `--yes` acting on
  everything, no-terminal printing without acting, one harness raising an
  error while the others still run.
- Codex file handling: first write into a missing file; write into a file with
  unrelated tables and comments, asserting those survive byte for byte;
  re-run reporting "already"; a differing table shown and replaced; an
  unparsable file refused with the snippet printed; the `.bak` written; the
  profile skipped when no models are configured; `env_key` present only when
  an API key is configured; `CODEX_HOME` honoured.
- Removal: our two tables gone, everything else identical, parent tables kept
  when they still hold other providers and dropped when empty.
- Launchers: the environment `local-llm aider` and `local-llm qwen` build,
  including the `openai/` model prefix aider needs and the three variables
  Qwen Code requires together.
- Aliases: only chosen harnesses' aliases added; a second run does not remove
  an alias for a harness no longer installed.
- Uninstall: the Codex file listed in the plan, `--dry-run` leaving it in
  place, and removal reported line by line.

Detection must be injected rather than read from the real executable search
path, or these tests pass or fail depending on what the developer happens to
have installed. The shared test fixture redirects `HOME`, so a Codex file
written during a test lands in a scratch directory for free, but it does not
fake the executable search path — the registry needs its own injectable
lookup, reachable from the object the commands already use.

## 14. Documentation to update

- `README.md`: the "Coding agents" section rewritten around the grouped menu,
  the Codex provider with its experimental caveat and the project-level
  shadowing note, the launcher list, and the Gemini CLI and Antigravity
  explanations with their dates; first-run step 6 rewritten; a row in the
  Files table for `~/.codex/config.toml` and `CODEX_HOME`; the uninstall
  example's integrations line listing Codex.
- `CHANGELOG.md`: one Unreleased bullet naming the command surface, in the
  existing terse style.
- The 2026-08-26 design document is left as it is. It is dated, approved, and
  describes a different scope; this document is the record for this feature,
  matching the repository's one-dated-file-per-feature convention.

## 15. Risks accepted

- The Codex integration may not work end to end on the `llama-server` build a
  person has today, for the reason in section 7. Mitigated by labelling it,
  not by holding it back.
- Breadth is a maintenance cost. Every fact about another tool in this
  document has a shelf life, and the Gemini and Antigravity situation changed
  twice during the research for it. Mitigated by keeping the strings in code
  short and dateless, and putting dated detail in the README.
- A model renamed in `models.ini` leaves both the Codex profile and the
  opencode helper agent pointing at a model that no longer exists, with no
  warning until the harness fails at run time. Out of scope here; worth a
  follow-up check that compares both against `models.ini`.
