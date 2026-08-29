# Coding agents

Which coding agents can talk to the router, how each one is wired up, and what
`local-llm integrate` writes where.

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

The aliases go inside a marked block in your shell startup file, and only
there. If your file already defines an alias of the same name outside that
block, the tool says so and tells you which of the two your shell will
actually use — whichever it reads last. A startup file whose markers do not
pair up, or that is not saved as UTF-8, is left completely alone and named:
shell completion is not installed into it either, because the installer would
append to the same file.

`MODEL` defaults to `default_model` from settings; naming one that is not in
`models.ini` is refused with the list of what is available.
