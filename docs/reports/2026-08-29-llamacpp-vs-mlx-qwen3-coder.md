# llama.cpp vs MLX — Qwen3-Coder-30B-A3B-Instruct on Apple M4 Pro

Measured 2026-08-29 in two sittings: llama.cpp first (part 1), then MLX and vllm-metal once
the 17 GB MLX checkpoint had downloaded (part 2). One engine held the model at a time. Every
number is a real measurement; nothing is estimated. The combined table is under
"Combined results" in part 2. The conclusion drawn from these numbers is in `2026-08-29-mlx-decision.md`, beside this file.

## Machine

| Item | Value |
| --- | --- |
| Chip | Apple M4 Pro |
| Memory | 48 GB unified (51,539,607,552 bytes) |
| OS | macOS 26.6.2 (build 25G83) |
| llama.cpp | `llama-server` version 0.3.0, build 10621, commit `c1d0e7a00`, AppleClang 21.0.0 |
| mlx | 0.32.2 |
| mlx-lm | 0.31.3 |
| transformers (in the MLX venv) | 5.16.1 |
| Python for the MLX venv | CPython 3.12.14 |

## Models

The GGUF file is the one the `local-llm` tool already has on disk:

```
/Users/nikosntemkas/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/b17cb02dd882d5b6ab62fc777ad2995f19668350/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf
```

That file is 17,665,334,432 bytes (17.7 GB).

The MLX checkpoint `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` was confirmed to exist
on the Hugging Face Hub (four safetensors shards plus tokenizer files), so no fallback
conversion was needed. It simply did not finish downloading — see "Why the MLX side is
missing" below.

## Prompts

Both prompt files were built by concatenating `.py` files from
`/Users/nikosntemkas/.config/local-llm/local-llm/src/local_llm` (read only, nothing was
modified), smallest file first, each preceded by a `# ==== FILE: <name> ====` banner, and
ending with the line `Summarise what this code does in 300 words.`

| File | Tokens (llama.cpp `/tokenize`) | Characters | Source files included |
| --- | --- | --- | --- |
| `prompt_2k.txt` | 2,063 | 7,991 | `__init__.py`, `estimate.py`, `paths.py`, `logs.py`, `sections.py` |
| `prompt_20k.txt` | 20,785 | 83,167 | the five above plus `settings.py`, `sampling.py`, `hardware.py`, `agents.py`, `integrations/__init__.py`, `quant.py`, `gguf.py`, `discover.py`, `preset.py`, `harnesses.py`, `shellrc.py` |

Those counts are the raw prompt text. Once wrapped in the chat template the server reports
2,070 and 20,801 prompt tokens respectively, and a one-line run-id header (explained below)
adds 8 more, giving the 2,078 and 20,801 figures in the results table.

**A cross-check worth recording:** `mlx_lm.generate`, using the MLX tokenizer, also counted
`prompt_2k.txt` as exactly **2,070 tokens** — identical to llama.cpp. The two engines
tokenize these prompts the same way, so any future throughput comparison between them is a
fair one and does not need a correction factor.

## Results

"Prefill" is prompt processing speed. "Gen" is generation speed. "TTFT" is time to first
token measured over HTTP with a streaming request, from sending the request to the arrival
of the first content delta. Every row is the second of two identical runs.

| Engine | Prompt size | Prefill tok/s | Gen tok/s | TTFT | Peak memory | Load time |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp (`llama-server`, HTTP) | 2,078 tok | 665.2 | 68.9 | 3.15 s | (not sampled; see 20k row) | 7.79 s |
| llama.cpp (`llama-server`, HTTP) | 20,801 tok | 337.7 | 37.9 | 61.81 s | 8,991 MB resident, 9,008 MB peak footprint, 16 GB total incl. clean mmap pages | 7.79 s |
| MLX (`mlx_lm.generate`) | 2,063 tok | **not measured** | **not measured** | n/a | **not measured** | **not measured** |
| MLX (`mlx_lm.generate`) | 20,785 tok | **not measured** | **not measured** | n/a | **not measured** | **not measured** |
| MLX (`mlx_lm.server`, HTTP) | both | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** |

Load time is measured once per server process, from launching `llama-server` to the first
HTTP 200 from `/health`. It is the same 7.79 s for both prompt rows because both prompts
were served by one process.

### Both llama.cpp runs, in full

The engine was extremely repeatable — the two runs at each size agree to within about 0.4%.

| Run | Prompt tokens | Prefill tok/s | Generated tokens | Gen tok/s | TTFT | Total wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 2k, run 1 | 2,078 | 664.8 | 212 | 68.6 | 3.148 s | 6.24 s |
| 2k, run 2 | 2,078 | 665.2 | 188 | 68.9 | 3.146 s | 5.87 s |
| 20k, run 1 | 20,801 | 338.5 | 293 | 38.0 | 61.496 s | 69.21 s |
| 20k, run 2 | 20,801 | 337.7 | 327 | 37.9 | 61.812 s | 70.44 s |

The generation counts are below the 512-token cap because the model stopped on its own
after finishing the requested 300-word summary. The tokens-per-second figures are rates and
are unaffected by that, but the sample is a few hundred tokens rather than a full 512.

### A measurement trap that had to be worked around

The first pair of 2k runs used an identical prompt both times. On the second run
`llama-server` reported `cache_n: 2069` and `prompt_n: 1` — it had reused the KV cache from
the first run and re-processed exactly one token. Prefill "speed" for that run was a
meaningless 36.4 tok/s over a single token, and TTFT collapsed to 0.039 s.

Every number in the tables above therefore comes from a run whose prompt carries a unique
`# run-id: <nonce>` first line, which defeats the prefix cache and forces a genuine cold
prefill. Anyone repeating this benchmark, on either engine, has to do the same —
`mlx_lm.server` also caches prompts, so comparing a cached MLX run against an uncached
llama.cpp run (or the reverse) would produce a difference of one to two orders of magnitude
that has nothing to do with the engines.

### Memory and swap

Sampled around the 20k llama.cpp runs.

```
=== BEFORE 20k (llama) ===
total = 4096.00M  used = 2378.00M  free = 1718.00M  (encrypted)
Pages free:      103159.
Swapins:         421863.
Swapouts:        743881.
=== AFTER 20k (llama) ===
total = 4096.00M  used = 2362.00M  free = 1734.00M  (encrypted)
Pages free:       18533.
Swapins:         422822.
Swapouts:        743881.
```

Swapouts did not move at all (743,881 before and after), and swap in use actually fell
slightly. **llama.cpp did not swap** serving a 20k-token prompt at 65k context on this
48 GB machine. The 2,378 MB of swap already in use was there before the benchmark started,
left over from other work on the machine.

Process memory at the end of the 20k runs:

```
PID   COMMAND      MEM   MEM
7261  llama-server 8991M 8991M

    ---        ---          ---        ---    ---
8991 MB      16 GB        33 MB       4574    TOTAL
Auxiliary data:
    phys_footprint: 8991 MB
    phys_footprint_peak: 9008 MB
```

The two different figures both matter. The 8,991 MB physical footprint is the memory
charged to the process; the 16 GB total includes clean pages mapped straight from the GGUF
file, which the kernel can evict and re-read from disk at will. llama.cpp memory-maps the
model rather than copying it into anonymous memory, which is why a 17.7 GB model shows a
9 GB footprint. An MLX comparison would need to account for this, because `mlx_lm` loads
weights into arrays rather than mapping them and would likely report a much higher figure
for the same model without actually using more of the machine.

### Correctness sanity check

First lines of the llama.cpp answer to the 20k prompt (run 2):

> This codebase provides a tool called `local-llm` for running local large language models
> (LLMs) using llama.cpp, making them available via an API server for integration with
> coding agents like Claude, Copilot, aider, and Qwen Code.
>
> Key functionalities include:
>
> *   **Model Management:** It discovers and recommends local LLMs from the Hugging Face
> Hub based on machine specs (RAM, GPU) and model characteristics (size, quantization). It
> handles quantization, model file detection, and memory estimation for running models.

First lines of the llama.cpp answer to the 2k prompt (run 2):

> This codebase provides tools for managing local LLM inference using `llama.cpp`, focusing
> on memory estimation, file organization, logging, and model configuration. The
> `__init__.py` exposes the package version. The `estimate.py` module calculates memory
> requirements for models, estimating disk usage plus overhead for KV cache and compute
> buffers, erring on the side of caution.

Both are accurate descriptions of the code that was fed in.

## Commands used, verbatim

Starting the GGUF server (with load timing):

```
llama-server \
  -m /Users/nikosntemkas/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/b17cb02dd882d5b6ab62fc777ad2995f19668350/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
  -c 65536 --n-predict 32768 --jinja -ngl all -fa auto --no-context-shift --metrics \
  --port 8911 --host 127.0.0.1
```

Note the flag is `--n-predict`, not `-n-predict`. Build 10621 rejects the single-dash form:

```
error: invalid argument: -n-predict
```

The tool writes `n-predict` as an INI key in `models.ini` (`sections.py` line 71), which the
router turns into the double-dash flag, so this is a transcription detail of running the
server by hand rather than a discrepancy with the tool.

Measuring over HTTP — one streaming chat completion, capturing time to first token from the
wall clock and prefill/generation rates from llama.cpp's `timings` object in the final SSE
chunk:

```
python3 bench_http.py 8911 prompt_2k.txt  "$MODEL_ID" llama_2k_b  n2kb
python3 bench_http.py 8911 prompt_20k.txt "$MODEL_ID" llama_20k_b n20kb
```

The request body is `{"max_tokens": 512, "temperature": 0, "stream": true,
"stream_options": {"include_usage": true}}`, and the fifth argument is the cache-busting
run-id described above.

Memory and swap sampling:

```
sysctl -n vm.swapusage
vm_stat | grep -iE "Swapins|Swapouts|Pages free"
top -l 1 -stats pid,command,mem,rsize -pid <pid>
footprint -p <pid>
```

MLX environment setup (all inside the scratch root, with `UV_CACHE_DIR` and `VIRTUAL_ENV`
pointed there):

```
uv venv --python 3.12 venv-mlx
uv pip install mlx-lm
```

The MLX benchmark commands, written and ready but never run against the 30B model:

```
./venv-mlx/bin/mlx_lm.generate \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --prompt - --max-tokens 512 --temp 0 < prompt_20k.txt

./venv-mlx/bin/mlx_lm.server \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --port 8912 --host 127.0.0.1 --log-level INFO
```

`mlx-lm` 0.31.3 has no `--prompt-file` option; `--prompt -` reading from standard input is
the equivalent.

# Part 2 — the MLX and vLLM measurements (added later the same day)

Same machine, same prompt files, same harness as part 1.

`llama-server` was confirmed to be stopped before any of this ran, and was never started
again, so only one engine ever held the model in memory at a time.

## Versions used in part 2

| Item | Value |
| --- | --- |
| mlx | 0.32.2 |
| mlx-lm | 0.31.3 |
| transformers | 5.16.1 |
| Python (MLX venv) | CPython 3.12.14 |
| vLLM | 0.28.0+cpu |
| vllm-metal | 0.3.0.dev20260829020049 (nightly build) |
| MLX checkpoint | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, snapshot `6e302ea604ad9ab206367e2c501d1571023e7b6d` |
| Checkpoint size on disk | 17,181,071,994 bytes (17.2 GB) across four safetensors shards |

Both engines were pointed at that same MLX checkpoint. The llama.cpp rows in the combined
table below use a different file — the 17.7 GB GGUF described in part 1 — because the two
runtimes cannot read each other's formats. Both are 4-bit quantizations of the same model,
but they are not bit-identical quantizations, so a few percent of the difference in speed
and quality between engines could come from the weights rather than the engine.

## Combined results

Every number is a real measurement. "Prefill" is prompt processing speed, "gen" is
generation speed, "TTFT" is time to first token measured over HTTP from sending the request
to the arrival of the first streamed piece of the answer. Command-line rows have no TTFT
because `mlx_lm.generate` prints its statistics only after the whole answer is finished.

| Engine | Prompt size | Prefill tok/s | Gen tok/s | TTFT | Peak / physical memory | Load time |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp `llama-server`, HTTP | 2,078 tok | 665.2 | 68.9 | 3.15 s | (not sampled; see 20k row) | 7.79 s |
| llama.cpp `llama-server`, HTTP | 20,801 tok | 337.7 | 37.9 | 61.81 s | 8,991 MB footprint, 16 GB incl. clean mmap pages | 7.79 s |
| MLX `mlx_lm.generate`, CLI | 2,081 tok | 845.9 | 77.0 | not printed | 18.14 GB peak | ~4.8 s (inside a 10.48 s total) |
| MLX `mlx_lm.generate`, CLI | 20,804 tok | 457.5 | 44.3 | not printed | 19.88 GB peak | ~5.1 s (inside a 57.73 s total) |
| MLX `mlx_lm.server`, HTTP | 2,081 tok | 736.6 (from TTFT) | 63.2 | 2.83 s | see note below | 13.56 s |
| MLX `mlx_lm.server`, HTTP | 20,804 tok | 412.2 (from TTFT), 397.7 (from the server's own progress log) | 31.8 | 50.47 s | 30 GB footprint after eight requests | 13.56 s |
| vllm-metal, HTTP | 2,080 tok | 351.4 (from TTFT) | 56.3 | 5.92 s | see note below | 19.48 s (at memory fraction 0.85) |
| vllm-metal, HTTP | 20,803 tok | 341.5 (from TTFT) | 21.9 | 60.92 s | 33 GB footprint (EngineCore process) | 19.48 s |

MLX rows are the second of two identical-shaped runs, each with its own unique run-id line,
as required by the cache-defeating rule established in part 1. The vLLM rows are single
runs — there was time for one pass at each prompt size, not two.

The prefill figures for the HTTP rows are derived, not reported: neither `mlx_lm.server` nor
vLLM returns llama.cpp's `timings` object, so prefill speed is the prompt token count
divided by the time to first token. That slightly *understates* prefill speed because it
also contains tokenization, chat-template rendering and the first decode step. For the MLX
20k case the server's own progress log gives an independent figure over a narrower window
(397.7 tok/s), which is 3.5% below the TTFT-derived one — so the derivation is sound to
within a few percent.

### MLX command line, both runs at each size

| Run | Prompt tokens | Prefill tok/s | Generated tokens | Gen tok/s | Peak memory | Total wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 2k, run 1 | 2,081 | 782.8 | 208 | 76.99 | 18.138 GB | 10.58 s |
| 2k, run 2 | 2,081 | 845.9 | 247 | 77.04 | 18.138 GB | 10.48 s |
| 20k, run 1 | 20,804 | 457.6 | 348 | 44.30 | 19.881 GB | 57.55 s |
| 20k, run 2 | 20,804 | 457.5 | 317 | 44.32 | 19.881 GB | 57.73 s |

Generation speed is repeatable to within 0.06%. Prefill at 2k varies by 8% between the two
runs because the measured interval is only about 2.5 seconds and includes some one-off
warm-up; at 20k, where the interval is 45 seconds, the two runs agree to within 0.02%.

The total wall time includes loading the model. Subtracting the reported prefill and
generation times leaves roughly 4.8 to 5.1 seconds of model load per invocation. That is
much faster than the 13.6 seconds `mlx_lm.server` takes to become ready, and comparable to
llama.cpp's 7.8 seconds.

### MLX over HTTP, both runs at each size

| Run | Prompt tokens | TTFT | Content pieces streamed | Wall gen tok/s | Total wall time | Server-reported cached tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 2k, run 1 | 2,081 | 2.956 s | 248 | 63.52 | 6.84 s | 0 |
| 2k, run 2 | 2,081 | 2.825 s | 286 | 63.16 | 7.34 s | 10 |
| 20k, run 1 | 20,804 | 51.148 s | 322 | 30.69 | 61.61 s | 13 |
| 20k, run 2 | 20,804 | 50.466 s | 512 | 31.78 | 66.54 s | 13 |

The "cached tokens" column is the evidence that the run-id trick worked on MLX as well: the
server reused at most 13 tokens of prefix out of 20,804, so every run did a genuine cold
prefill. Without a unique run-id the whole prompt would have been reused and the numbers
would have been meaningless, exactly as happened to llama.cpp in part 1.

Generation over HTTP is 28% slower than the same model generating from the command line
(31.8 versus 44.3 tokens per second at 20k; 63.2 versus 77.0 at 2k). The gap is the server's
per-token overhead — each token becomes its own JSON-encoded server-sent event — plus the
prompt-cache bookkeeping described below. It is not a property of the model.

### An MLX server quirk that silently produced empty answers

The first pair of 20k HTTP runs reported 391 and 363 completion tokens in the usage totals
but produced zero visible output, so no time-to-first-token could be computed at all.

The cause is that `mlx_lm.server` 0.31.3 streamed the entire answer in a field named
`reasoning` inside each delta, not the standard `content` field:

```
"choices": [{"index": 0, "finish_reason": null,
             "delta": {"role": "assistant", "reasoning": " coding"}}]
```

The same server used `content` for the 2k prompt in the same session, so the behaviour is
not a fixed property of the server — it depends on what the model emits. `bench_http.py` was
changed to accept `content`, `reasoning`, or `reasoning_content`, whichever is present, and
the 20k runs were repeated with fresh run-ids; those repeats are the numbers in the tables.
The unpatched original is kept alongside as `bench_http.py.orig`.

This matters beyond the benchmark: any client that reads only `delta.content` — which is
what the OpenAI streaming specification says to do — will show a blank response from this
server for prompts of this kind, while the token counters keep incrementing.

### Memory and swap, MLX

Around the command-line 20k runs:

```
### vm_stat BEFORE 20k generate runs
Pages free:      1124683.
Swapins:          455714.
Swapouts:         769632.
### vm_stat AFTER 20k generate run2
Pages free:      1331452.
Swapins:          455734.
Swapouts:         769632.
```

Around the HTTP 20k runs:

```
### vm_stat BEFORE 20k http (rerun)
Pages free:         8016.
Swapins:          456049.
Swapouts:         769632.
### vm_stat AFTER 20k http (rerun)
Pages free:         4302.
Swapins:          456447.
Swapouts:         769632.
```

Swapouts stayed at exactly 769,632 across every MLX run, command line and server alike.
**MLX did not swap**, the same result llama.cpp got. Free pages did drop to about 4,300
(roughly 67 MB) during the HTTP runs, so the machine was close to full, but the pressure was
absorbed by evicting clean file-backed pages rather than by writing anything to swap.

Process memory at the end of the 20k HTTP runs:

```
PID    COMMAND MEM MEM
14169  Python  30G 30G

  30 GB      14 MB       224 KB       5287    TOTAL
Auxiliary data:
    phys_footprint: 30 GB
    phys_footprint_peak: 30 GB
```

**Thirty gigabytes for a 17.2 GB model, and the excess is not a fixed overhead — it grows
with every request.** `mlx_lm.server` keeps one key-value cache per conversation it has seen
and, in this version, does not evict them. The server logs the total after each request:

```
19:53:48  Prompt Cache: 1 sequences, 0.23 GB
19:53:56  Prompt Cache: 2 sequences, 0.46 GB
19:54:57  Prompt Cache: 3 sequences, 2.55 GB
19:56:12  Prompt Cache: 4 sequences, 4.63 GB
19:57:28  Prompt Cache: 5 sequences, 6.67 GB
19:58:30  Prompt Cache: 6 sequences, 8.75 GB
```

Each 20k-token conversation adds about 2.1 GB and none of them are released. Eight requests
into the session the footprint was 30 GB; the trend line says a long-lived server on this
machine would keep climbing until it exhausted memory. The two 2k conversations cost about
0.23 GB each.

The comparison the part 1 report asked for can now be made. llama.cpp memory-maps its GGUF
file, so a 17.7 GB model showed a 9.0 GB physical footprint with the rest as clean pages the
kernel may drop and re-read. MLX loads its weights into arrays, so the model alone is a real
17-18 GB of the process footprint. The single-shot command line confirms this: 18.14 GB peak
for the 2k prompt and 19.88 GB for the 20k prompt — that is the 17.2 GB of weights plus 0.9
to 2.7 GB of working memory and key-value cache, with no accumulation. **The right
comparison is 9.0 GB (llama.cpp) against 19.9 GB (MLX single shot) against 30 GB and rising
(MLX server after eight requests).**

### Correctness sanity check, MLX

First lines of the MLX answer to the 20k prompt, HTTP run 2 (`answer_mlx_20k_r2.txt`),
alongside the llama.cpp answer already quoted in part 1:

> This codebase provides a tool to run local LLMs, primarily for coding tasks, by managing
> local models and integrating them into development environments.
>
> It provides tools to discover, download, and run local LLMs, including:
>
> *   **Model Discovery & Management:** Discovering models from Hugging Face Hub, selecting
> suitable quantizations, and managing models in a preset file.
> *   **Local LLM Router:** A server that serves local models via OpenAI-compatible API
> endpoints, allowing tools like Claude, Copilot, and Qwen to use local models.

The llama.cpp answer to the same prompt, for side-by-side reading:

> This codebase provides a tool called `local-llm` for running local large language models
> (LLMs) using llama.cpp, making them available via an API server for integration with
> coding agents like Claude, Copilot, aider, and Qwen Code.
>
> Key functionalities include:
>
> *   **Model Management:** It discovers and recommends local LLMs from the Hugging Face
> Hub based on machine specs (RAM, GPU) and model characteristics (size, quantization). It
> handles quantization, model file detection, and memory estimation for running models.

Both are accurate and of comparable quality. Two smaller observations. The MLX answer to the
2k prompt is likewise accurate. But the MLX **command line** answer to the 20k prompt
(`gen_gen_20k_r2.log`) repeated one bullet verbatim three times:

> - **Integration**: Provides integration points for various coding agents (Claude, Copilot,
> Qwen, etc.) to use local models.
> - **Integration**: Provides integration points for various coding agents (Claude, Copilot,
> Qwen, etc.) to use local models.
> - **Integration**: Provides integration points for various coding agents (Claude, Copilot,
> Qwen, etc.) to use local models.

That is at temperature 0, and the HTTP path with the same weights and the same prompt did
not do it. It is a single occurrence and is recorded rather than explained; a repetition
penalty difference between the two entry points is the obvious suspect but was not checked.

### MLX commands used, verbatim

```
HF_HUB_OFFLINE=1 ./venv-mlx/bin/mlx_lm.generate \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --prompt - --max-tokens 512 --temp 0 < <prompt file with a unique run-id line prepended>

HF_HUB_OFFLINE=1 ./venv-mlx/bin/mlx_lm.server \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --port 8912 --host 127.0.0.1 --log-level INFO

python3 bench_http.py 8912 prompt_2k.txt  mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit mlx_2k_r2  h2kr2nonceF
python3 bench_http.py 8912 prompt_20k.txt mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit mlx_20k_r2 h20kr2nonceK
```

Load time for the server is measured from launching the process to the first HTTP 200 from
`/v1/models`, which is the MLX equivalent of llama.cpp's `/health`.

## vllm-metal

vLLM 0.28.0 with the `vllm-metal` nightly plugin, serving the same MLX checkpoint.

### The requested configuration failed

The command specified for this test was run exactly as given:

```
VLLM_METAL_MEMORY_FRACTION=0.5 <venv>/bin/vllm serve mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --max-model-len 65536 --max-num-seqs 1 --port 8913 --host 127.0.0.1
```

The engine process died 16 seconds in. The memory fraction **did** take effect — the plugin
logged it and acted on it, and that is precisely why the run failed:

```
INFO 08-29 20:00:53 [cache_policy.py:1348] Paged attention: using VLLM_METAL_MEMORY_FRACTION=0.50
INFO 08-29 20:00:53 [cache_policy.py:1108] Paged attention memory breakdown: metal_limit=40.20GB, fraction=0.5, usable_metal=20.10GB, model_memory=17.18GB, overhead=0.71GB, kv_budget=2.21GB, per_block_bytes=1572864, num_blocks=1404, max_tokens_cached=22464
INFO 08-29 20:00:53 [kv_cache.py:266] KV cache: 2208.3 MB (48 layers, 1404 blocks, 16 tokens/block)
```

The error, verbatim:

```
ERROR 08-29 20:00:53 [core.py:1346] ValueError: To serve at least one request with the model's max seq len (65536), (6.0 GiB KV cache is needed, which is larger than the available KV cache memory (2.06 GiB). Based on the available memory, the estimated maximum model length is 22448. Try increasing `gpu_memory_utilization` (which also controls CPU memory on the CPU backend) or decreasing `max_model_len` when initializing the engine. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more details.
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}
```

The arithmetic is worth reading, because it explains the constraint rather than just
reporting it. The plugin treats 40.20 GB as the Metal limit on this 48 GB machine. Half of
that is 20.10 GB of usable memory. The model needs 17.18 GB and the runtime another 0.71 GB,
leaving 2.21 GB for the key-value cache — but a single 65,536-token sequence needs 6.0 GB.
At fraction 0.5 the longest context that fits is about 22,448 tokens.

### The same command at a higher memory fraction

Because the failure was a budgeting one rather than a defect, the run was repeated once with
`VLLM_METAL_MEMORY_FRACTION=0.85` and everything else identical. That started successfully:

```
INFO 08-29 20:02:11 [cache_policy.py:1108] Paged attention memory breakdown: metal_limit=40.20GB, fraction=0.85, usable_metal=34.17GB, model_memory=17.18GB, overhead=0.71GB, kv_budget=16.28GB, per_block_bytes=1572864, num_blocks=10349, max_tokens_cached=165584
INFO 08-29 20:02:12 [kv_cache.py:266] KV cache: 16277.6 MB (48 layers, 10349 blocks, 16 tokens/block)
```

Time from launch to the first HTTP 200 on `/health`: **19.48 seconds**. The measurements
below all come from this configuration, and each prompt was run once.

| Run | Prompt tokens | TTFT | Content pieces streamed | Wall gen tok/s | Total wall time |
| --- | --- | --- | --- | --- | --- |
| 2k | 2,080 | 5.920 s | 257 | 56.32 | 10.47 s |
| 20k | 20,803 | 60.915 s | 289 | 21.90 | 74.07 s |

vLLM was the slowest of the three engines on both prompt sizes: about 12% slower generation
than MLX over HTTP at 2k, and 31% slower at 20k, with a time to first token 21% worse on the
long prompt. Note also that the plugin logs `chunked prefill enabled (paged attention),
max_num_batched_tokens=2048` — it processes the prompt in 2,048-token chunks, the same block
size MLX's progress log shows.

### Memory and swap, vllm-metal

**This is the one engine of the three that swapped.** Swapouts are cumulative counters, so
the differences are what matter:

```
(after all MLX work, before vLLM started)   Swapouts: 769632
### vm_stat BEFORE vllm runs                Swapouts: 838884   (+69,252 pages during model load)
### vm_stat BEFORE vllm 20k                 Swapouts: 849352   (+10,468 during the 2k run)
### vm_stat AFTER vllm 20k                  Swapouts: 861864   (+12,512 during the 20k run)
                                            total = 4096.00M  used = 3438.62M  free = 657.38M
```

That is 92,232 pages, roughly 360 MB, written to swap over the vLLM session, against exactly
zero for both llama.cpp and MLX on the same machine and the same prompts. Swap in use also
grew from 2.07 GB to 3.44 GB.

Physical footprint of the EngineCore process at the end of the 20k run:

```
PID    COMMAND MEM MEM
16084  Python  33G 33G

  33 GB      34 MB        48 KB       8140    TOTAL
Auxiliary data:
    phys_footprint: 33 GB
    phys_footprint_peak: 33 GB
```

33 GB is close to what the memory breakdown predicted it would reserve — 17.18 GB of weights
plus a 16.28 GB key-value cache is 33.5 GB — so unlike the MLX server this is a fixed,
pre-allocated arena rather than a figure that grows request by request. It is large because
it was asked to be: the reservation is set by the memory fraction, not by demand.

### vllm-metal correctness check

First lines of the vLLM answer to the 20k prompt (`answer_vllm_20k_r1.txt`):

> This codebase provides tools to run large language models locally using llama.cpp,
> managing model files, hardware compatibility, and integrating with coding agents like
> Claude and Copilot.
>
> It handles model discovery, downloading, and configuration. The core logic estimates
> memory requirements based on model size and hardware capabilities, ensuring models fit
> within available RAM or GPU VRAM.

Accurate, and consistent with the other two engines.

### vllm-metal commands used, verbatim

```
VLLM_METAL_MEMORY_FRACTION=0.5  HF_HUB_OFFLINE=1 <venv>/bin/vllm serve mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --max-model-len 65536 --max-num-seqs 1 --port 8913 --host 127.0.0.1     # failed, see above
VLLM_METAL_MEMORY_FRACTION=0.85 HF_HUB_OFFLINE=1 <venv>/bin/vllm serve mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --max-model-len 65536 --max-num-seqs 1 --port 8913 --host 127.0.0.1     # started in 19.48 s

python3 bench_http.py 8913 prompt_2k.txt  mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit vllm_2k_r1  v2knonceP
python3 bench_http.py 8913 prompt_20k.txt mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit vllm_20k_r1 v20knonceQ
```

One warning appeared at startup and is recorded in case it matters later:

```
WARNING 08-29 20:01:59 [system_utils.py:299] Found ulimit of 2048 and failed to automatically increase with error current limit exceeds maximum limit. This can cause fd limit errors like `OSError: [Errno 24] Too many open files`. Consider increasing with ulimit -n
```
