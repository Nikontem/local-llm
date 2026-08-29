# Configuration

Settings, environment variables, where files live, and how the router uses
memory.

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

A section in `models.ini` sets its context with one of two keys, and they mean
different things. `fit-ctx = N` is a **floor**, not a fixed number: it tells
`llama-server` the smallest context it may choose for that model, and the
server itself picks the real context when the model loads, from whatever
memory is actually free at that moment — never below the floor, but as high
above it as the machine allows. This is what `pull` and `add` now write for
a new section. The practical effect is that **the same model can load with a
different context on different runs of the router** — a bigger one when the
machine is idle, a smaller one when something else is using memory. `c = N`
still works and still means what it always did: it pins the context to
exactly `N` and switches off that load-time adjustment for the section, at
the cost of `llama-server` refusing to shrink it if memory is tight. `local-llm
doctor` names any section still using `c` (see
[Running the router](commands.md#running-the-router)) so it is easy to find
and switch over.

`load`'s budget check only guards explicit loads, honestly: a chat request
naming an unloaded model, or the web UI, bypasses it, and `llama-server`
evicts resident models by *count* only, never by memory pressure — the
reason `--max-models` defaults to 1 (see [Running the router](commands.md#running-the-router)).
