# Two ways of measuring a model's speed — llama-bench against a real llama-server

Measured 2026-08-30 on an Apple M4 Pro. Every number below is a real measurement; nothing is
estimated. The question was narrow: `local-llm tune` needs one way to measure how fast a
model answers, two plausible ways existed, and this report is the evidence that settled which
one the tool ships. The conclusion is in "Outcome"; the decision rule was fixed in writing
before any of these numbers existed.

The design this implements is `docs/superpowers/specs/2026-08-30-empirical-tuning-design.md`,
whose section 8 records the same result in less detail.

## Machine

| Item | Value |
| --- | --- |
| Chip | Apple M4 Pro |
| Memory | 48 GB unified (51,539,607,552 bytes) |
| OS | macOS 26.6.2 (build 25G83) |
| llama.cpp | `llama-server` and `llama-bench` version 0.3.0, build 10621, commit `c1d0e7a00` |
| Install | Homebrew |

The router was stopped for every run and nothing else was loaded. Free memory was checked
before and between runs and never fell below 26.7 GB.

## Models

The comparison model, already on disk for the `local-llm` tool:

```
/Users/nikosntemkas/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/b17cb02dd882d5b6ab62fc777ad2995f19668350/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf
```

17,665,334,432 bytes (17.7 GB). A mixture-of-experts model, 30B total parameters with about
3B active per token, quantised to Q4_K_XL.

Development and the fixtures used a much smaller model, so that the expensive one was loaded
only when a question genuinely needed it:

```
/Users/nikosntemkas/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

1,117,320,736 bytes (1.12 GB).

## What was being measured, and what was deliberately left alone

Four settings in a `models.ini` section change how fast a model answers:

| Key | Meaning |
| --- | --- |
| `b` | batch size — how many tokens are submitted for processing at once |
| `ub` | micro-batch size — how that submission is subdivided; never larger than `b` |
| `flash-attn` | whether the fused attention kernel is used (`on`, `off`, `auto`) |
| `cache-type-k`, `cache-type-v` | the precision the per-token cache is stored at (`f16` or `q8_0`) |

Three things were **not** measured, on purpose:

- **The number of GPU layers.** The tool pins `n-gpu-layers = all`, and an earlier design
  established that pinning it is what makes llama.cpp reduce the context to fit rather than
  scatter layers onto the processor. Sweeping it would fight that decision.
- **The context.** llama.cpp's own fitting step chooses it at load time from measured free
  memory. That is the whole point of the previous design.
- **Thread count.** With every layer offloaded it makes almost no difference on this machine.

## The workload

One profile, shaped like a coding agent's traffic rather than a benchmark's defaults:

```
depth    = 4096   tokens already in the cache when the measurement starts
prompt   = 4096   tokens processed in this turn
generate =  256   tokens written in reply
```

Measuring at depth zero — `llama-bench`'s default — flatters every configuration equally and
says nothing about which to pick, so depth is part of the profile rather than left off.

Candidates are ranked by **turn time**, the seconds one reply takes:

```
turn_time = 4096 / prompt_rate + 256 / generation_rate
```

## The candidate list

The baseline is the model's effective settings today, read with the `[*]` global block taken
into account. Each setting is then varied against it exactly once — never two at a time — so
that a candidate being faster identifies which single setting caused it:

| Label | `b` | `ub` | `flash-attn` | cache |
| --- | --- | --- | --- | --- |
| `baseline` | 2048 | 512 | auto | f16 |
| `batch 1024/256` | 1024 | 256 | auto | f16 |
| `batch 4096/1024` | 4096 | 1024 | auto | f16 |
| `flash-attn off` | 2048 | 512 | **off** | f16 |
| `cache q8_0` | 2048 | 512 | auto | **q8_0** |

Three repetitions per candidate, `llama-bench`'s own warm-up left enabled.

## The two engines

**Engine A — `llama-bench`.** One invocation per candidate, loading the model directly with no
HTTP server involved. Two tests per invocation, kept apart because the combined `-pg` form
reports a single blended rate and turn time needs the two numbers separately. No `-ngl` is
passed: its default offloads every layer, confirmed by a run printing
`load_tensors: offloaded 29/29 layers to GPU`, which matches what the router pins.

**Engine B — a real `llama-server`.** One scratch server per candidate on a free port, with the
candidate's settings on the command line and no `models.ini` written. It sends a priming
request to fill the cache, then three measured requests with `cache_prompt` enabled, so that
the measured turn processes only the new tokens — the same thing `llama-bench`'s `-d` does.
Rates are computed from `timings.prompt_n` / `timings.prompt_ms` and
`timings.predicted_n` / `timings.predicted_ms` in the reply, never from the request, because
the synthetic prompt's token count is only approximate.

Engine B exists because engine A measures the inference engine in isolation and cannot see
anything that lives only in the server: request slots, continuous batching, cache reuse
between turns, the HTTP layer itself. Whether any of that changes which candidate wins is an
empirical question, and this report is how it was answered rather than argued.

## The decision rule, fixed before any numbers existed

> **Do the two engines name the same winner?**

A tuner only has to *rank* configurations; it never has to measure absolutely. If engine A
picks the same best candidate as a real server, then a consistent offset in its absolute
tokens per second changes no decision anyone makes, and it ships alone. Only a disagreement
about which candidate wins would justify carrying two engines.

## Results

Engine A, `llama-bench`, 305 seconds for all five candidates:

| Candidate | Turn | Prompt tok/s | Generation tok/s |
| --- | --- | --- | --- |
| **`batch 4096/1024`** | **12.47 s** | 486.3 | 63.3 |
| `cache q8_0` | 13.62 s | 443.9 | 58.3 |
| `batch 1024/256` | 14.33 s | 398.6 | 63.1 |
| `baseline` | 15.71 s | 351.3 | 63.3 |
| `flash-attn off` | 17.64 s | 379.1 | 37.5 |

Engine B, a real `llama-server`, 475 seconds:

| Candidate | Turn | Prompt tok/s | Generation tok/s |
| --- | --- | --- | --- |
| **`batch 4096/1024`** | **25.09 s** | 201.1 | 54.2 |
| `baseline` | 25.38 s | 198.2 | 54.4 |
| `batch 1024/256` | 25.92 s | 192.9 | 54.6 |
| `cache q8_0` | 26.65 s | 191.4 | 48.8 |
| `flash-attn off` | 37.60 s | 149.8 | 24.9 |

Engine B is uniformly slower in absolute terms, by a roughly consistent factor:

| Candidate | B/A prompt rate | B/A generation rate |
| --- | --- | --- |
| `batch 4096/1024` | ×0.41 | ×0.86 |
| `cache q8_0` | ×0.43 | ×0.84 |
| `batch 1024/256` | ×0.48 | ×0.86 |
| `baseline` | ×0.56 | ×0.86 |
| `flash-attn off` | ×0.40 | ×0.67 |

That gap is not explained here. The server does work `llama-bench` does not — tokenising over
HTTP, running the model's real sampling parameters, managing slots — and the offset is
consistent enough that it changes no ranking. It was not investigated further because the
decision rule does not depend on it.

## Outcome

**Both engines named `batch 4096/1024` the winner, and both put `flash-attn off` last.** The
rule was satisfied, so engine B was deleted and `llama-bench` is the only engine the tool
ships. The deleted code remains in this repository's history.

Two findings in the numbers themselves are worth keeping:

- **Turning flash attention off costs 42% of a turn** on this model and machine (17.64 s
  against a 12.47 s winner), almost all of it in generation, where the rate falls from 63.3 to
  37.5 tokens per second. The `auto` default is doing real work.
- **Raising the batch size is worth about 21%** against the baseline measured in the same run.

## Three things that went wrong, and what they cost

The comparison was worth running mostly because of this section. None of it was visible in
441 unit tests or in a smoke run against the 1.12 GB model.

**Engine B had never once measured a real model.** Its first run failed all five candidates
with `HTTP 503 Service Unavailable` and finished in 17 seconds instead of ten minutes. It
treated an open TCP port as readiness, but `llama-server` binds its port before the weights
have loaded and answers 503 in that window. On a 1.12 GB model the window is too narrow to
hit; on a 17.7 GB model it swallowed every candidate.

**The obvious fix was also wrong.** Polling `GET /health` looked correct and was not. Measured
directly against this model, `llama-server` answers `/health` with `{"status":"ok"}` about two
seconds after start, whether or not it can serve anything:

```
t=1  http=000  body=
t=2  http=200  body={"status":"ok"}
```

The scratch server's log showed each candidate being killed while still at 0.27 seconds of
`load_model: loading model`. Readiness had to be redefined as *a minimal completion request
succeeding*, which by construction cannot report ready before completions work.

**The middle of the ranking does not reproduce.** The two engines order the middle three
candidates differently. More tellingly, engine A reordered its own middle between two runs of
the same model on the same machine:

| Candidate | First run | Second run |
| --- | --- | --- |
| `batch 4096/1024` | 12.45 s | 12.47 s |
| `cache q8_0` | 13.42 s | 13.62 s |
| `batch 1024/256` | 14.32 s | 14.33 s |
| `baseline` | **13.09 s** | **15.71 s** |
| `flash-attn off` | 17.63 s | 17.64 s |

Every candidate reproduced within 1.5% except the baseline, which moved 20% — enough to take
it from second place to fourth. Within a single run the dispersion is tiny (`llama-bench`
reported a standard deviation of 0.41 tokens per second on a 1563 tokens-per-second
measurement), so this is machine state between runs, not measurement noise.

The winning margin and the run-to-run variance of the thing it is measured against are the
same size. That is why the shipped command now prints standard deviations, refuses to offer a
change unless the winner beats the baseline by at least 5%, and tells the user in as many
words that a few percent between the middle rows is not meaningful.

## A single-configuration cross-check

Before the full comparison, both programs were run once on the 1.12 GB model at the baseline
settings, to capture their output formats. Same profile, two repetitions:

| | Prompt tok/s | Generation tok/s |
| --- | --- | --- |
| `llama-bench` | 1563.46 (σ 0.41) | 151.91 (σ 2.79) |
| `llama-server` | 1011.61 | 100.33 |

The same direction and roughly the same magnitude of offset seen on the large model, from a
completely separate run.

## What the shipped command prints

The 1.12 GB model, three repetitions, after all fixes. The baseline wins here and the command
correctly declines to change anything:

```
  baseline          b=2048 ub=512 fa=auto cache=f16                 1414 pp/s   134.2    ±0.4 tg/s     4.8 s
  batch 4096/1024   b=4096 ub=1024 fa=auto cache=f16                1417 pp/s   127.4    ±3.9 tg/s     4.9 s  +2.0%
  batch 1024/256    b=1024 ub=256 fa=auto cache=f16                 1351 pp/s   127.7    ±2.7 tg/s     5.0 s  +4.8%
  cache q8_0        b=2048 ub=512 fa=auto cache=q8_0                1275 pp/s   117.8    ±2.4 tg/s     5.4 s  +12.1%
  flash-attn off    b=2048 ub=512 fa=off cache=f16                  1202 pp/s   100.3    ±1.7 tg/s     6.0 s  +24.0%

Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M is already the fastest of the settings tried. Nothing to change.
```

A percentage is positive when a candidate is *slower* than the baseline. Note that the two
knobs which mattered on the 30B model — batch size and flash attention — behave differently
here: on a 1.5B model raising the batch size is a small loss rather than a gain. Settings do
not transfer between models, which is the reason this command measures rather than assumes.

## Commands used, verbatim

The comparison:

```bash
uv run local-llm tune 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL' --engine bench  --json > /tmp/tune-bench.json
uv run local-llm tune 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL' --engine server --json > /tmp/tune-server.json
```

(`--engine` existed only while both engines did; it was removed with engine B.)

What engine A runs per candidate:

```bash
llama-bench -m <model.gguf> -o json -r 3 -d 4096 -p 4096 -n 256 \
  -b <batch> -ub <ubatch> -fa <on|off|auto> -ctk <f16|q8_0> -ctv <f16|q8_0>
```

What engine B ran per candidate:

```bash
llama-server -m <model.gguf> --host 127.0.0.1 --port <free> --no-ui \
  -c 9472 -ngl all -b <batch> -ub <ubatch> -fa <...> -ctk <...> -ctv <...>
```

The `/health` measurement that disproved the second readiness fix:

```bash
nohup llama-server -m <model.gguf> --host 127.0.0.1 --port 5701 --no-ui -c 9472 -ngl all >/tmp/probe.log 2>&1 &
for i in $(seq 1 40); do
  curl -s -o /tmp/h.body -w "%{http_code}" --max-time 2 http://127.0.0.1:5701/health
  sleep 1
done
```

Confirming `llama-bench` offloads every layer with no `-ngl` given:

```bash
llama-bench -m <model.gguf> -o json -r 1 -p 32 -n 8 -v 2>&1 >/dev/null | grep offloaded
# load_tensors: offloaded 29/29 layers to GPU
```

## What this does not establish

- **One machine, one build.** Everything here is an Apple M4 Pro running llama.cpp build
  10621. Neither `llama-bench`'s output format nor the fitter's behaviour is a documented
  stable interface. On a machine with a discrete graphics card the same four settings exist,
  but their relative importance almost certainly differs.
- **One model for the verdict.** The winner was confirmed on one 17.7 GB mixture-of-experts
  model. The 1.5B cross-check shows the *rankings themselves* do not transfer between models,
  only the finding that the two engines agree.
- **The absolute gap between the engines is unexplained.** It is consistent and it changes no
  ranking, so it was not pursued.
- **The workload profile is a guess about coding agents, not a recording of one.** The three
  constants were reasoned from how a coding agent's prompt is built. They are the most likely
  thing here to be wrong in a way that matters.
- **Interactions between settings were never explored.** Each was varied against the baseline
  alone. A combination faster than anything in the table can exist and would not be found.
