# Milestone 3 acceptance — 2026-08-26

Repo at `c144192`. Unit suite 212 passed, ruff clean.

## Docker end-to-end (Linux from nothing)

`tests/e2e/run.sh` with `E2E_MODEL=bartowski/SmolLM2-135M-Instruct-GGUF E2E_QUANT=Q8_0`
(see "network" below): image built from `ubuntu:24.04` with `llama-server`
compiled from tag `b10256`, the tool installed by `install.sh --uv` from the
copied checkout as a normal user. Every step passed:

```
=== version === doctor === recommend === pull === up === load === chat === status
=== completion === down === remove === ALL PASSED ===
```

`doctor` on Linux without Homebrew: brew warn (optional), llama-server ok with
`--models-preset`, hf warn (optional), token warn; `recommend --use small`
listed vendor models with fit against the container's 5.7 GB budget; `pull`
wrote `[bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0]`; the chat request returned
text; `status` showed the loaded child; `completion install --shell bash`
wrote `~/.bash_completions/local-llm.sh` and the aliases block; `down` left no
process; `remove --delete-files` deleted the section and the cache file.

Two defects found and fixed on the way:

1. The runtime image lacked `libgomp.so.1`, so `llama-server` could not start,
   and `doctor` reported that as "no router support". Now the image installs
   `libgomp1`, and `doctor` says "does not run: <error>" for a binary that
   cannot execute.
2. Downloads through the Hugging Face library's xet backend stalled a few
   megabytes short of the end, three times out of three on this network,
   while plain HTTP finished at full speed. The tool now downloads over plain
   HTTP by default (`HF_HUB_DISABLE_XET=0` opts back in), and a download that
   stops growing for three minutes fails with a message that says the partial
   file is kept and the command resumes — instead of hanging.

## Pretend-clean `setup` on this Mac

Scratch `HOME`, `HF_HOME` pointed at the existing cache, `env -i` so the
author's shell exports of `LOCAL_LLM_*` (from the old zsh script) do not
redirect config into the real directory — an earlier attempt without `env -i`
wrote a `settings.toml` into the real `~/.config/local-llm/`, which was
removed; `models.ini`, `.zshrc` and `opencode.jsonc` were never touched.

- Without Homebrew on `PATH`: step 1 stops with the brew.sh line and "run
  local-llm setup again", exit 1.
- With `--yes --model Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`: all seven steps.
  Prerequisites listed the real tools and devices; machine line
  `Apple M4 Pro, 20 GPU cores, 48 GB unified memory → 38 GB usable`; the model
  was a cache hit; the section got `c = 32768`, `n-predict = 4096`, the
  `qwen2.5-small` profile; `settings.toml` written with that default model and
  `max_models = 1`; zsh completion at `~/.zfunc/_local-llm` and the aliases
  block in `~/.zshrc`; opencode plugin installed and the `tiny` agent written
  into a new `opencode.jsonc` (in the run where `opencode` was on `PATH`);
  router started; the smoke request answered "Hello there!"; endpoints printed.
- Second run with the same `--model`: "already in models.ini, skipped", then
  every later step again, exit 0. `down` left nothing running.

## Network

Hugging Face's CDN throttled this address per repository after the same
0.5B file had been downloaded six times within the hour (100 KB/s on that
repo, 5 MB/s on others). That is why the Docker run used SmolLM2 and the Mac
run used a cached model. The `run.sh` script forwards `E2E_MODEL`,
`E2E_QUANT` and `HF_TOKEN` so CI can do the same.

## Verdict

Milestone 3 complete: install script, wizard, opencode and shell integration,
README, CI workflows, Docker end-to-end.
