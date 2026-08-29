# Troubleshooting

Symptoms you are likely to hit, and what each one usually means.

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
  [Memory](configuration.md#memory). Lower `--max-models` back to 1, or `local-llm unload`
  the one you are not using.
- **`llama-server` has no `--models-preset`** — the build is older than
  December 2025, before router-mode presets existed; `brew upgrade
  llama.cpp` on macOS, or rebuild from a current `llama.cpp` tag on Linux.
- **A coding agent still talks to the real Anthropic/OpenAI API** — it was
  started without going through `local-llm claude` / `local-llm copilot` /
  `local-llm env`, so it never got the `*_BASE_URL` variables.
