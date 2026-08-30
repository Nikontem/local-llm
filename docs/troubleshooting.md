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
- **The log says `failed to fit params to free device memory`** —
  `llama-server`'s own fitter (see [Memory](configuration.md#memory)) could
  not find a context at or above the section's `fit-ctx` floor that fits in
  the memory free at that moment, and loaded the model anyway at its full,
  unreduced context instead of a reduced one — it is likely to swap or fail
  under load. `local-llm doctor` surfaces this as the `model fit` warning.
  Fix it with any of: lower that model's `fit-ctx` in `models.ini`, raise
  `reserve_gb` in `settings.toml` so more memory is left free, or use a
  smaller quantization of the model.
- **`llama-server` has no `--models-preset`** — the build is older than
  December 2025, before router-mode presets existed; `brew upgrade
  llama.cpp` on macOS, or rebuild from a current `llama.cpp` tag on Linux.
- **`tune` reports numbers far slower than expected** — it measures whatever
  the GPU actually delivers at that moment, and something else on the
  machine competing for it (another app, a second model still loaded, even
  a browser tab doing GPU work) will show up as a slow run with no error.
  Close other GPU users and run it again; `local-llm status` shows whether
  the router itself is holding a second model.
- **A candidate in `tune`'s table shows an error instead of a rate** — most
  often the `q8_0` cache-type candidate, because a quantized key-value cache
  can still need more memory than is free at that moment on top of whatever
  the baseline used. It is not a bug in the measurement; it means that
  candidate does not fit right now. Free up memory (see
  [Memory](configuration.md#memory)) and try again, or accept that this
  machine cannot use that setting for this model.
- **`tune` refuses to start, saying the model needs more than this machine
  has usable** — the workload it measures against (four thousand tokens of
  history, four thousand more, and a reply) needs its own memory on top of
  the model's weights, and this machine does not have enough free right now
  to hold both without swapping. `local-llm doctor` shows what else is using
  memory; freeing it, lowering `reserve_gb`, or using a smaller quantization
  are the ways out.
- **A coding agent still talks to the real Anthropic/OpenAI API** — it was
  started without going through `local-llm claude` / `local-llm copilot` /
  `local-llm env`, so it never got the `*_BASE_URL` variables.
