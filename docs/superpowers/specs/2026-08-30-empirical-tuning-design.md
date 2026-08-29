# Measuring a model instead of estimating it — design

Date: 2026-08-30
Status: approved in discussion, written for review

## 1. Purpose

`local-llm` has never measured anything. Every number it writes into
`models.ini` comes from arithmetic: file sizes, a memory budget, and a handful
of integers read out of the model file's header. That arithmetic is in
`estimate.py`, `gguf.py` and `quant.py`, and nothing in `src/` starts a
process, times a request, or counts tokens per second.

Until recently that gap was a safety problem. If the estimate was wrong, a
model could be configured in a way the machine could not actually run. It is
not a safety problem any more. The work recorded in
`2026-08-30-context-sizing-design.md` handed the safety question to
llama.cpp's own fitting step, which measures free device memory at the moment
a model loads and reduces the context until it fits. Fitting happens with real
measurements, on the real machine, at the real moment.

What is left is a *speed* question, and it is a different question with a
different answer. Several settings change how fast a model answers without
changing whether it fits at all: how many tokens are processed in one batch,
how that batch is subdivided, whether the attention step uses the fused
"flash attention" kernel, and whether the key-value cache is stored at full
precision or quantized. `local-llm` picks values for some of these and leaves
others at llama.cpp's defaults, and it has never had any evidence that its
picks are good ones.

This document proposes a command, `local-llm tune`, that measures those
settings on the machine in front of it and reports which one is fastest.

It also proposes something unusual: **two independent implementations of the
measurement, built at the same time and compared, with one of them expected to
be deleted.** Section 8 explains why, and states in advance what result would
justify keeping both.

## 2. Terms used in this document

- **The router** — the single `llama-server` process `local-llm` manages,
  serving every model in `models.ini` on one port.
- **The fitter** — a step inside `llama-server`, on by default, that measures
  free device memory before loading a model and reduces settings until it
  fits. It never overrides a setting the user has written explicitly.
- **`llama-bench`** — a separate program shipped in the same package as
  `llama-server`. It loads a model file directly and measures how fast it
  processes prompts and generates tokens, with no HTTP server involved.
- **Prompt processing** — reading the input. Measured in tokens per second,
  written `pp` in `llama-bench` output.
- **Token generation** — writing the reply, one token at a time. Written `tg`.
  It is much slower per token than prompt processing.
- **Depth** — how many tokens are already in the cache before a measurement
  starts. Both rates get slower as depth grows, so measuring at depth zero
  flatters every configuration equally and tells you nothing useful.
- **Batch size** (`b`) and **micro-batch size** (`ub`) — how many tokens are
  submitted for processing at once, and how that submission is subdivided.
  The micro-batch can never exceed the batch.
- **Flash attention** (`fa`) — a fused implementation of the attention step.
  Usually faster and usually uses less memory, but not on every backend, which
  is why llama.cpp's default is `auto` rather than `on`.
- **Key-value cache type** (`cache-type-k`, `cache-type-v`) — the precision the
  per-token cache is stored at. `q8_0` roughly halves its memory cost against
  `f16` and may cost some speed, or may gain some by moving less memory.
- **A candidate** — one combination of those four settings, which is what gets
  measured.
- **Turn time** — the derived number this design ranks on. See section 4.

## 3. What is being tuned, and what is deliberately not

The settings under test are exactly the ones that are (a) writable as
`models.ini` keys, (b) not owned by the fitter, and (c) plausibly worth
tokens per second:

| Key | What it is | Where its value comes from today |
| --- | --- | --- |
| `b` | batch size | never written; llama.cpp's default |
| `ub` | micro-batch size | never written; llama.cpp's default |
| `flash-attn` | flash attention | `auto`, in the `[*]` block of the template |
| `cache-type-k`, `cache-type-v` | key-value cache precision | `q8_0` for models over 10 GiB, otherwise unset, from `sections.py` |

Three things are explicitly **not** tuned.

**The number of GPU layers is not tuned.** `models.template.ini` pins
`n-gpu-layers = all`, and the context-sizing design established that pinning
it is what makes the fitter reduce the context rather than scatter layers onto
the processor. Sweeping it here would fight a decision this repository made
deliberately and recently.

**The context is not tuned.** It is chosen by the fitter at load time. That is
the entire point of the previous design.

**Thread count is not tuned.** With every layer offloaded it makes almost no
difference, and on a machine where it would matter the layer count is the
bigger lever — which the previous point rules out.

## 4. The workload, and how a winner is chosen

`llama-bench`'s defaults measure a 512-token prompt producing 128 tokens at
depth zero. Nobody this tool serves runs that. `local-llm` exists largely to
point coding agents at local models, and a coding agent's traffic is lopsided
and deep: a system prompt plus tool definitions plus file contents go in, a
few hundred tokens come out, and the conversation grows all session.

So there is one built-in profile, shaped like that traffic:

```
DEPTH   = 4096      tokens already in the cache
PROMPT  = 4096      tokens processed in this turn
GENERATE = 256      tokens written in reply
```

These are a judgement, not a measurement, and are named constants so they can
be revised — the same treatment `WORKING_MINIMUM` gets in the context-sizing
design. Two constraints pin them. Their total, 8448 tokens, must stay below
the 16384-token floor `suggest_context_floor` writes, or a candidate could
fail to run for reasons that have nothing to do with its speed. And the
wall-clock cost of the whole command scales with them roughly linearly.

**Turn time** is the single number a winner is chosen by:

```
turn_time = PROMPT / prompt_rate + GENERATE / generation_rate
```

It is the seconds a coding agent waits for one reply. Both engines report the
two rates separately; the derivation lives in `report.py` alone, so the
ranking rule is identical for both and exists in exactly one place.

## 5. The candidate list

The list is fixed and is not a search. The **baseline** is the model's
effective settings today, read through `Preset.get`, which falls back to the
`[*]` block — so `flash-attn = auto` is picked up as part of the baseline
rather than missed because it is not written in the model's own section.

Each knob is then varied against that baseline exactly once:

- batch and micro-batch as valid pairs: `(1024, 256)`, `(2048, 512)`,
  `(4096, 1024)` — skipping whichever pair the baseline already is;
- flash attention set to whichever of `on` / `off` the baseline is not;
- the key-value cache set to whichever of `f16` / `q8_0` the baseline is not,
  moving `cache-type-k` and `cache-type-v` together, since `sections.py`
  already treats them as one decision.

That is six candidates when the baseline's batch pair is not already one of
the three, and five when it is.

**Why a fixed list rather than a search.** The obvious alternative is to sweep
one knob, keep the winner, and carry it into the next sweep. That is
path-dependent: two measurement engines could diverge at the first step and
never evaluate the same candidate again, which would make section 8's
comparison meaningless. It also explores no more of the space than this does,
because it varies one knob at a time as well. A fixed list costs the same, is
reproducible, and lets the report state its own limitation plainly: **each
knob was measured against the baseline, and interactions between knobs were
not explored.** That sentence belongs in the output, not just in this
document.

Repetitions default to three rather than `llama-bench`'s five, because a run
is expected to take ten to twenty minutes on a large model. `--repetitions`
exposes it.

## 6. Structure

A new package, mirroring `integrations/` — several implementations of one idea
behind a shared context object:

```
src/local_llm/tuning/
  __init__.py   Candidate, Measurement, TuningContext
  profile.py    the workload constants and the candidate list
  bench.py      engine A: llama-bench
  server.py     engine B: a scratch llama-server
  report.py     turn time, ranking, the table, the models.ini lines
```

`cli.py` gains one command and does no work of its own beyond wiring and
printing — the division `doctor` already uses, where `doctor.py` builds
`Check` records and never prints and `cli.py` renders them.

**The contract is two dataclasses and one callable.** A `Candidate` holds the
four settings under test. A `Measurement` holds, for one candidate, the
prompt-processing rate, the generation rate, the derived turn time, and how
many repetitions succeeded. An engine is any callable taking a model path, the
profile, and a list of candidates, and returning a list of measurements.
Nothing downstream can tell which engine produced them. That is what makes
deleting the loser in section 8 a deletion rather than a rewrite.

**Every side effect is injected**, following `doctor.Env` and
`setup.SetupContext`: a `run` callable for subprocesses, the existing
`ProcessBackend` for spawning and stopping, the router's `http` callable,
and `say`, `confirm` and the module-level sleep that `tests/unit/conftest.py`
already stubs out. Neither engine imports `subprocess` or `psutil` directly.

### 6.1 The command

```
local-llm tune <model> [--engine bench|server] [--repetitions N] [--json] [-y]
```

`<model>` is a `models.ini` section name, resolved by the existing
`agents.resolve_model` and completed by `complete_model`, like every other
model-taking command. `--engine` exists for as long as two engines do; section
8 decides whether it survives.

## 7. The two engines

### 7.1 Engine A — `llama-bench`

One invocation per candidate, with two tests inside it:

```
llama-bench -m <path> -o json -r <reps> -p 4096 -d 4096 -n 256 <candidate flags>
```

The prompt-processing and generation tests are kept separate on purpose.
`llama-bench` also offers a combined `-pg` form, but it reports the pair as a
single blended rate, and turn time needs the two numbers apart.

**The JSON field names are not assumed.** The first task of the implementation
is a single run against the small model to capture a real JSON document, which
is then quoted here and committed as the test fixture. This follows the
precedent set by section 3 of the context-sizing design, which quotes observed
log lines rather than paraphrasing what the source appears to do. `llama-bench`
is not a stable documented interface any more than the fitter is.

One invocation per candidate means the model is loaded five or six times.
Candidates that differ only in batch sizes could share an invocation, since
those do not force a reload. That optimisation is deliberately left out until
a real run shows the loading time matters.

### 7.2 Engine B — a scratch `llama-server`

Not the router, and not router mode. Per candidate the engine spawns
`llama-server` directly with the model path and the candidate's settings as
command-line flags, on a free port found with the `port_in_use` check
`doctor.py` already has, through the injected `ProcessBackend`, waiting for it
with the same listen-poll loop `Router.start` uses. No `models.ini` is
written — not the user's, and not a scratch one.

Per candidate it sends one priming request, then `--repetitions` measured
requests, reading the `timings` block llama.cpp returns with each completion.

**Token counts come from the response, not from the request.** The synthetic
prompt is built to be about 4096 tokens, but "about" is not good enough to
divide by. The response reports `tokens_evaluated` and `tokens_predicted`, and
those are what the rates are computed from, so the arithmetic is exact even
when the prompt is approximate.

### 7.3 What engine B can see that engine A cannot

This is the reason for building both. `llama-bench` measures the inference
engine in isolation. The server adds things that only exist in the server:
request slots and continuous batching, the fitter's chosen context rather than
one implied by the test sizes, cache reuse between turns, and the HTTP layer
itself. Whether any of that changes which candidate wins is an empirical
question, and section 8 is how it gets answered instead of argued.

## 8. Two engines, one expected survivor

Both engines are built. Both run the same candidate list on the same model.
The comparison is then judged by one rule, fixed here before any numbers
exist:

> **Do they name the same winner?**

A tuner never has to measure absolutely; it only has to rank. If `llama-bench`
picks the same best candidate as a real server, then a consistent offset in
its absolute tokens per second changes no decision anyone makes, and it ships
alone. If the two disagree about which candidate wins, engine B is seeing
something engine A structurally cannot, and both stay behind `--engine`.

**The comparison runs once, on a large model, with the machine idle.** The
project's standing rule for experiments is to use
`Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`, about 1 GB, and that rule holds for
all development, all fixtures and all failure-path work. But it cannot settle
this question. A 1 GB model on a 38 GB machine sits so far inside the memory
budget that cache precision and flash attention may not move turn time at all,
and every candidate could tie within measurement noise. Two engines that agree
because neither measured anything have not agreed about anything. So the
verdict is taken from one deliberate run on a model of 17–25 GB already
present in `models.ini`, with the router stopped and nothing else loaded.

**If engine A wins**, `tuning/server.py` and its tests are deleted in a commit
that says why, `--engine` is removed, and the measured comparison — the
candidate list, both rankings, and the machine and build it ran on — is
written into this document as a new section. Git keeps the code; the spec
keeps the finding.

## 9. Output, and writing the result back

**The report is plain text, in `doctor`'s register.** One line per candidate
giving its settings, both rates, its turn time, and its change against the
baseline; then the winner named; then the exact `models.ini` lines. `--json`
dumps the measurements, matching `doctor`, `recommend` and `search`. The
sentence about knob interactions from section 5 is printed, not implied.

**The write-back is offered, never assumed.** A `typer.confirm`, then
`Preset.replace_section`, then `Preset.save`, which already writes through a
temporary file and leaves a `.bak`. Three details matter:

- **Only keys that actually differ from the effective baseline are written.**
  `Preset.get` falls back to `[*]`, so a value equal to the global default is
  not written into the section, and sections do not accumulate settings that
  change nothing.
- **If the baseline wins, nothing is offered.** The report says the current
  settings were the fastest of those tried, and the command exits having
  changed nothing.
- **Comments a user wrote inside the section must survive.**
  `replace_section` regenerates a section's whole body from the keys and
  comments it is handed, so the command reads the existing keys and comments
  first, merges the tuned values into them, and adds one line recording the
  date and the measured improvement.

There is no `--dry-run`: the report *is* the dry run. `-y/--yes` accepts the
write without asking, matching every other command that writes.

## 10. Running alongside the router

A benchmark competing for memory with a resident model produces confident
nonsense, which is the one output a tuning command must never produce.

Before measuring, the command reads what is resident from `Router.children()`
and asks permission to unload it. The models are unloaded through the existing
`POST /models/unload`, and **reloading them is registered as cleanup before
the first measurement runs**, so a crashed engine, a server that never comes
up, or an interrupted run still leaves the router holding what it held
before. The router process itself is never stopped, so the port keeps
answering throughout.

Before any of that, the model's `refined_estimate` is checked against the
machine budget, and the command declines rather than measure a configuration
that cannot fit.

## 11. Testing

Unit tests, which are what CI runs and which reach neither a process nor the
network:

- **Candidate list construction** from a section, including the case that
  matters most: a baseline value inherited from `[*]` rather than written in
  the section, which must be recognised as the baseline and not offered as a
  variation.
- **Turn time and ranking**, including a tie and a baseline win.
- **Engine A's parser**, against the JSON fixture captured from a real
  `llama-bench` run.
- **Engine B against `FakeBackend` and `FakeHttp`**: that a candidate becomes
  the right command-line flags, that rates are computed from the response's
  token counts, and — using the existing `spawn_dies` switch — that a server
  which never comes up still leaves the router restored.
- **The router restore path**, asserting the reload calls happen on the
  failure path and not only the success path.
- **The write-back's two hazards**: a hand-written comment inside a section
  survives, and a key equal to its `[*]` default is not written.
- **`--json` output shape**, matching the convention of the other three
  commands that have it.

One verification by hand, because no unit test can prove a number: the
comparison run of section 8.

## 12. Documentation

- `docs/commands.md` — the new command, its flags, and captured real output.
- `docs/configuration.md` — the four keys `tune` may write, and what each does.
- `docs/troubleshooting.md` — what to do when a run is slower than expected or
  a candidate fails to load.
- `CHANGELOG.md` — under `## Unreleased`.
- `README.md` — read against the finished behaviour, changed only if the front
  page's claims stop being true.

## 13. Risks and what this does not cover

**`llama-bench`'s output format is not a stable interface.** Neither is the
fitter's behaviour, and the previous design accepted the same risk. The
mitigation is the same: capture what was actually observed, quote it here,
record the build number, and keep a fixture so a format change fails a test
rather than producing wrong numbers silently.

**One machine, one build.** Everything will be measured on an Apple M4 Pro
running the Homebrew llama.cpp, build 10621. On a machine with a separate
graphics card the same four knobs exist but their relative importance almost
certainly differs, and the fixed candidate list may be a poor fit. Nothing
here is claimed to be tested on such a machine.

**The profile is a guess about coding agents, not a measurement of one.** The
three constants in section 4 are reasoned from how a coding agent's prompt is
built, not from recording real traffic. They are the most likely thing in this
design to be wrong in a way that matters, which is why they are named
constants and why the report states the workload it measured.

**Interactions between knobs are not explored**, by construction. A
combination that is faster than anything on the candidate list can exist and
will not be found. Section 5 explains why that trade was taken, and the report
says so out loud.

**A tuned section is tuned for the machine it was measured on.** `models.ini`
carries no record of which machine produced a value, so a configuration file
copied to a different machine carries speed settings chosen for the old one.
They remain safe — the fitter still governs what fits — but they may no longer
be fast. Teaching sections to record their provenance is worth doing and is
not covered here.
