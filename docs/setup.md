# Setup

How to install `local-llm` and what the guided first run does, step by step.

## Install

```
curl -fsSL https://raw.githubusercontent.com/Nikontem/local-llm/main/install.sh | sh
```

The script installs the tool with [`uv`](https://astral.sh/uv), installing
`uv` itself first if needed, on macOS and Linux. Homebrew is not required for
the tool. It matters for what the tool installs *for you*: when Homebrew is
present, `local-llm setup` installs `llama.cpp` (for `llama-server`) and `hf`
(Hugging Face's CLI) through it, so they stay managed by Homebrew like the
rest of your machine; when it is absent, `setup` shows the routes — Homebrew
from [brew.sh](https://brew.sh), llama.cpp's release binaries, or a build.

To install by hand instead of running the script:

```
uv tool install git+https://github.com/Nikontem/local-llm
```

Upgrade later with `sh install.sh --upgrade` or
`uv tool install --upgrade git+https://github.com/Nikontem/local-llm`.

Either way, the next step is the same:

```
local-llm setup
```

## First run

`local-llm setup` is a guided session in seven numbered steps. Nothing is
downloaded or overwritten without asking, and it can be re-run any time —
each step detects what is already done and skips it. `--yes` accepts every
default without prompting; `--model org/repo[:QUANT]` preselects a model
(repeatable); `--use coding|general|small|vision` narrows step 3 to one use
case.

**1. Prerequisites** — runs the same checks as `local-llm doctor` and offers
to fix the ones it safely can (installing `llama.cpp` or `hf` via Homebrew).
Missing Homebrew on macOS stops the wizard, since `llama.cpp` is installed
from it there; on Linux it prints the release-binary and build-from-source
routes instead.

**2. This machine** — detects RAM, chip and GPU and turns that into a memory
budget, then asks how much memory to reserve for the OS and other apps
(default 10 GB). On the author's Mac this prints:

```
Apple M4 Pro, 20 GPU cores, 48 GB unified memory → 38 GB usable for models (10 GB reserved)
```

**3. Models** — with no `--model` given, it looks at what is popular on
Hugging Face right now and shows a numbered menu, grouped by use case:
coding, general, small, and vision. Every suggestion names a **quantization**
— the precision a model's weights are stored at; a lower-precision
quantization is a smaller file that runs faster but loses some accuracy — and
picks the quantization that best fits your machine's budget. Abridged from a
real run on the 48 GB Mac above:

```
coding
   1. Qwen3-Coder-30B-A3B-Instruct  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF  UD-Q4_K_XL
      16.5 GB on disk, ~19.9 GB in memory  comfortable  vendor  · in models.ini
general
   2. Qwen3-30B-A3B-Thinking-2507 (thinking)  unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF  UD-Q4_K_XL
      ...  comfortable  vendor
small
   4. Qwen3.5-4B (thinking, vision)  unsloth/Qwen3.5-4B-GGUF  Q8_0
      4.8 GB on disk, ~6.5 GB in memory  comfortable  vendor
```

The quantization suggested scales with the machine — small models get the
high-precision `Q8_0`, the 27–30B ones get `UD-Q4_K_XL` on a 38 GB budget.
Type numbers to download (`1 3`), `s <text>` to search for something else, or
`n` for nothing. This same menu is available on its own afterwards as
[`local-llm browse-models`](commands.md#models), so adding a model later does
not mean running the wizard again.

**4. Download and configure** — downloads what was picked and writes a tuned
section for each into `models.ini` (see [Commands](commands.md#models)).

**5. Settings** — writes `settings.toml` with the reserve from step 2 and a
default model (the first one just downloaded, or the smallest already
configured).

**6. Shell and coding agents** — detects which coding agents are installed and
offers each the right kind of configuration: a provider written into Codex's
or opencode's own config, or a launcher command plus a short alias for Claude
Code, Copilot CLI, aider and Qwen Code. Agents that cannot use the router are
named with the reason. Shell completion and the `local_llm` alias are offered
alongside. See [Coding agents](agents.md).

**7. Start** — starts the router and offers a one-line test chat against the
smallest configured model, then prints the endpoints:

```
  OpenAI-style endpoint:     http://127.0.0.1:5678/v1
  Anthropic-style endpoint:  http://127.0.0.1:5678
  local-llm status      what is running
  local-llm models      every model you can ask for
  local-llm claude      Claude Code against the router
```

If no models are configured yet — a bare `--yes` run, or nothing chosen in
step 3 — setup stops after step 6 and points at `local-llm browse-models` and
`local-llm pull` instead of starting a router with nothing to serve.
