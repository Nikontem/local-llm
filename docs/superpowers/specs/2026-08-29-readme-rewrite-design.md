# Rewriting the README as a front page, with the reference material moved to `docs/`

**Date:** 2026-08-29
**Status:** approved, ready to implement

## The problem

`README.md` is 487 lines and carries four different documents at once: a pitch for the
tool, a getting-started guide, a complete command reference, and a support manual. Roughly
300 of those lines are reference material — every flag of `pull`, every environment
variable each coding-agent command sets, the settings table, the file-location table, the
troubleshooting list.

The prose itself is accurate and readable. The problem is scope: a person who has just
found the project on GitHub has to scroll past the exact semantics of `--set KEY=VALUE`
before learning whether the tool is for them. That is what makes it read like a manual.

A second problem follows from the first. `CLAUDE.md` declares the README to be the
specification of user-facing behaviour and says it is kept current. So the reference detail
cannot simply be deleted — it has to keep a home, and that home has to be named in
`CLAUDE.md` instead.

## Decisions taken

Three questions were settled with the repository owner before this document was written:

1. **The reference material moves into topic pages under `docs/`** rather than being
   compressed away or collapsed into one long reference file. Nothing is lost; the front
   page stops being a manual.
2. **The moved prose is moved as-is**, with only the edits the move itself forces:
   corrected cross-links, adjusted heading levels, and rewritten sentences that referred to
   another part of the old single file ("see [Memory](#memory) above"). It is not rewritten.
   It was written for exactly this purpose and re-writing it risks silently dropping a
   documented behaviour.
3. **Only the README is written fresh.** The split is a mechanical operation with no
   judgement in it and is done directly; the new README is written by a subagent working
   from this document.

## The target file map

| File | Source | Budget |
|---|---|---|
| `README.md` | written fresh from this spec | ~120 lines |
| `docs/setup.md` | the old "Install" and "First run" sections | ~110 lines |
| `docs/commands.md` | the old "Every day", "Models" and "Uninstall" sections | ~190 lines |
| `docs/agents.md` | the old "Coding agents" section, entire | ~110 lines |
| `docs/configuration.md` | the old "Settings", "Files" and "Memory" sections | ~70 lines |
| `docs/troubleshooting.md` | the old "Troubleshooting" section | ~40 lines |

`docs/superpowers/` (design documents and implementation plans) is untouched and stays
where it is. The five new pages sit beside it at the top level of `docs/`.

## What the new README contains

Nine parts, in this order. The line budgets are guidance for keeping proportion, not a
target to hit exactly; the whole file should land near 120 lines.

**1. What this is** (about 8 lines). One paragraph. It has to answer, in the first two
sentences, what the tool does and who it is for: it runs a single `llama-server` process
that serves every model you have configured on one port, and it handles the parts that are
tedious by hand — working out which models fit your machine, downloading and tuning them,
and pointing coding agents at the result. Terms of art get defined in passing: `llama-server`
is llama.cpp's inference server, GGUF is the single-file model format it runs, router mode
is the December 2025 feature that lets one process serve many models. macOS and Linux;
Windows is untested.

**2. Why not Ollama or LM Studio** (about 4 lines). One short paragraph. The honest
distinction: those tools hide llama.cpp behind their own model format and application, and
llama.cpp's router mode already gives you the one-endpoint-many-models arrangement without
either. What this tool adds is the machine-aware model suggestions, the tuning, and the
agent wiring — on top of a plain INI file you can still read and edit by hand.

**3. Getting started** (about 15 lines). The install one-liner, then `local-llm setup`, in
copy-pasteable blocks. One sentence on what the installer does (installs the tool with
`uv`, installing `uv` first if needed) and one on the alternative for people who would
rather not pipe a script into a shell (`uv tool install git+...`). A link to
`docs/setup.md` for the Homebrew question, the upgrade path, and the seven steps in detail.

**4. What `setup` does to your machine** (about 6 lines). Prose, not a numbered list — the
numbered walkthrough lives in `docs/setup.md`. It should say plainly: it checks
prerequisites, measures your machine and turns that into a memory budget, suggests models
that fit and downloads the ones you pick, writes a tuned section for each into `models.ini`,
offers to wire up whichever coding agents it finds, and starts the router. Nothing is
downloaded or overwritten without asking, and the wizard can be re-run at any time.

**5. What you get** (about 12 lines). One terminal block showing a running router — adapted
from the current `local-llm status` output — so the reader sees the payoff rather than
reading about it. Follow it with the two endpoint URLs and a single sentence saying any
model named in `models.ini` can be asked for by name in the `model` field of a request, with
no restart.

**6. The commands** (about 25 lines). A two-column table: command, and one line saying what
it does. No flags, no options, no output samples — those are in `docs/commands.md`, which
the table's introduction links to. Group the rows with a blank-line break or a small heading
per group so the shape is legible: running the router (`up`, `down`, `restart`, `status`,
`logs`, `ui`), models (`models`, `load`, `unload`, `recommend`, `search`, `pull`, `add`,
`remove`, `edit`), agents (`integrate`, `claude`, `copilot`, `aider`, `qwen`, `env`), and
maintenance (`setup`, `doctor`, `completion install`, `prune-logs`, `uninstall`). Note that
`prune-logs` is missing from the current README and should appear; `stop` and `ls` are
hidden aliases and should not.

**7. Coding agents** (about 12 lines). Two sentences explaining the two ways an agent gets
wired — a provider written into the agent's own configuration file, or a launcher command
that sets the environment variables and starts it — then a table with one row per agent:
name, which of the two, and the command or alias to use. Codex CLI is marked experimental
in the table itself, in those words, because someone will otherwise try it and conclude the
tool is broken. Gemini CLI and Antigravity CLI get one row saying they cannot be pointed at
the router. Everything else — the exact variables, the reasoning, the Gemini history — is
in `docs/agents.md`.

**8. One warning** (about 6 lines). `--max-models` defaults to 1, and this is not caution
for its own sake: `llama-server` evicts resident models by count and never by memory
pressure, so two large models can both stay loaded until the GPU runs out mid-request and
every call fails. This is the one piece of reference material that earns its place on the
front page, because the failure it prevents is confusing and expensive.

**9. Documentation, development, licence** (about 12 lines). A short list linking the five
`docs/` pages with a few words each, the four development commands from the current
"Development" section, and the licence.

## Constraints the README must respect

- **Links must be absolute.** `pyproject.toml` sets `readme = "README.md"`, so this file is
  the package long description and will be rendered where relative paths do not resolve.
  Every link into `docs/` is written as a full
  `https://github.com/Nikontem/local-llm/blob/main/docs/<page>.md` URL.
- **No cross-references to itself.** The old file used anchors like `see [Memory](#memory)`.
  The new one is short enough not to need them.
- **Every factual claim comes from the current README or the code.** Nothing new is
  invented — model names, sizes, ports, variable names, flag names and output samples are
  taken from what is already documented and known correct.

## Voice: what "reads like a readme" means here

The failure mode being corrected is *exhaustiveness*, not tone. The rules for the new file:

- Say what a thing does and why it matters. Do not enumerate its options.
- One example where an example makes the point. Not one per feature.
- No parenthetical qualifications of edge cases. Those belong in `docs/`.
- A reader who wants the detail follows a link. A reader who does not is never made to
  scroll past it.
- Keep complete sentences and plain language — the fix is less material, not terser prose.

## Contract changes outside the documentation

- **`CLAUDE.md` line 11** currently says to read `README.md` for the user-facing behaviour
  of every command, and calls it the specification. That sentence now points at the `docs/`
  pages, naming `docs/commands.md` as the per-command reference.
- **`CLAUDE.md` line 96** says user-visible changes go in `README.md` if they change what a
  command does. That becomes: in the `docs/` page that covers the command, and in the
  README only when the change affects what the README itself says.
- **`CHANGELOG.md`** gets an entry under `## Unreleased` noting that the documentation was
  reorganised, since a reader following an old link needs to know where things went.

## Out of scope

No behaviour changes, no code changes, no test changes. No documentation site, no
navigation tooling, no rewriting of `docs/superpowers/`. The accuracy of the moved prose is
taken as given — this is a reorganisation, not an audit.
