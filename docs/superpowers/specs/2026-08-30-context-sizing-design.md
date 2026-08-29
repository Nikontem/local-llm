# Letting llama.cpp size the context, with a floor — design

Date: 2026-08-30
Status: approved in discussion, written for review

## 1. Purpose

When `local-llm` adds a model to `models.ini`, it writes a fixed context size
for that model — the `c` key, meaning "how many tokens of conversation this
model can hold at once". The number comes from `suggest_context()` in
`gguf.py`, which reads the model file's header, estimates how much memory the
key-value cache would need, and picks the largest power of two that fits the
machine while leaving two thirds of the free memory spare.

That code was written when `llama-server` had no opinion of its own. It now
does. Recent builds have a *fitting* step (the `-fit` option, on by default)
that measures how much memory each device actually has free at the moment a
model loads, and reduces the model's settings until it fits. This is strictly
better information than an estimate made when the model was first downloaded,
because it accounts for whatever else is running on the machine right now.

`local-llm` currently switches that safety net off without meaning to. The
shared `[*]` block in `models.template.ini` sets `n-gpu-layers = all`, and the
per-model section sets `c`. Between them they pin every setting the fitter is
allowed to change, so it gives up. Section 3 records what was measured; the
short version is that a model whose context does not fit produces a warning
and then attempts an allocation that cannot succeed.

This document proposes that `local-llm` stop writing a fixed context and
instead write a *floor* — the smallest context worth loading the model at —
leaving the fitter to choose the actual number. The tuning knowledge we have
is kept, but expressed as a lower bound rather than an exact value.

Nothing about the router process, the command set, or the harness
integrations changes.

## 2. Terms used in this document

- **The router** — the single `llama-server` process this tool manages, which
  serves every model named in `models.ini` on one port.
- **Child process** — when a request names a model, the router starts a
  separate `llama-server` process for that model and forwards the request to
  it. The child is what actually holds the model in memory.
- **Context** (the `c` key, `--ctx-size` on the command line) — how many
  tokens the model can hold at once. Larger contexts cost memory, roughly in
  a straight line: twice the context, twice the key-value cache.
- **Key-value cache** — the per-token memory a model keeps while generating.
  It is the part of a model's memory use that grows with the context, and the
  part `suggest_context()` estimates.
- **The fitter** (the `-fit` option) — a step inside `llama-server`, on by
  default, that measures free device memory before loading and reduces
  settings until the model fits. It only changes settings the user has not
  set explicitly.
- **Context floor** (the `fit-ctx` key, `--fit-ctx` on the command line) — the
  smallest context the fitter is permitted to choose. It does not request a
  context; it forbids going below one.
- **Offloading a layer** — running part of the model on the processor instead
  of the graphics chip. On a machine with a separate graphics card this frees
  graphics memory. On Apple Silicon, where both share the same physical
  memory, it frees nothing and only costs speed.

## 3. What was measured

All of the following was observed on one machine — an Apple M4 Pro reporting
38338 MiB of Metal memory — running the Homebrew build of llama.cpp, version
0.3.0, build 10621. Every claim in this section comes from a log line, not
from reading the source.

**The fitter never overrides a setting the user wrote.** Asked to load a model
with a deliberately impossible context of four million tokens, it reported
`context size set by user to 4000000 -> no change` and left the context
alone.

**With the layer count unpinned, the fitter sacrifices layers.** In the same
impossible-context run it moved 21 of the model's 29 layers off the graphics
chip to keep the context the user asked for. On Apple Silicon this is the
wrong trade — it saves no memory, because both live in the same pool, and
costs a great deal of speed.

**With the layer count pinned, as `local-llm` pins it, the fitter gives up
entirely.** The exact line is:

```
common_fit_params: failed to fit params to free device memory:
    n_gpu_layers already set by user to -2, abort
```

Loading then continues regardless. In the test run the process went on to
request a 109375 MiB key-value cache on a device with 38338 MiB. It is a
warning, not an error: the load is never refused.

**Removing the fixed context makes the pinned layer count an advantage.**
With `n-gpu-layers = all` kept, no `c` written, and a floor of 8192, the
fitter reduced the context instead and kept every layer where it was:

```
common_params_fit_impl: context size reduced from 32768 to 12032
    -> need 587 MiB less memory in total
common_params_fit_impl: entire model can be fit by reducing context
load_tensors: offloaded 29/29 layers to GPU
```

This is the behaviour the design wants, and it is available only because the
layer count is pinned. Pinning it turns out to be the right decision for the
wrong reason.

**The floor is honoured, and refuses to be crossed.** Given a floor equal to
the model's full trained context, the fitter declined to reduce below it:
`default model context size is 32768 which is <= the min. context size of
32768 -> no change`, and paid for the floor by offloading layers instead.

**When even the floor cannot be met, the fitter warns and proceeds.** With the
layer count pinned and the floor unreachable there is no lever left, and the
result is the `abort` warning above followed by a load at the full,
unreduced context. This is the one case where the machine genuinely cannot
run the model usefully, and it is currently invisible to `local-llm`.

**`fit-ctx` works as a `models.ini` key.** The router translates preset keys
into command-line options for the child — `c` becomes `--ctx-size` — and
`fit-ctx = 8192` duly arrived at the child as `--fit-ctx 8192`. The fitter
then ran and reported `will leave 36416 >= 1024 MiB of free device memory, no
changes needed`, and the model served its full 32768 tokens.

**Child processes inherit nothing from the router's own command line.** The
child receives exactly the `[*]` block plus its own section, translated. This
means the design cannot rely on passing anything to the parent process.

**Two things that were suspected and turned out to be fine.** The number of
server slots now defaults to automatic, which selects four, and a shared
key-value cache. The context is *not* divided between them — the child
reported `n_slots = 4, n_ctx_slot = 32768` for a 32768-token context. And the
`-cd` option named in the upstream discussion no longer exists; draft-model
options were renamed to `--spec-draft-*`. Neither needs any action.

## 4. What changes in the written configuration

`models.template.ini` is unchanged. `n-gpu-layers = all` stays, and section 3
explains why: it is what makes the fitter reduce the context rather than
scatter layers onto the processor.

`sections.py` changes one key. Where it writes

```ini
c = 65536
```

it will instead write

```ini
fit-ctx = 16384
```

This applies to sections whose context `local-llm` chose. A caller who named a
context explicitly still gets `c`; section 5.1 sets out all three cases.

Everything else about a generated section — the model path, the sampling
values, the cache types, the comment lines explaining where the section came
from — is untouched. The comment that currently reads `context: 65536
suggested for this machine (28.0 GB usable)` becomes a sentence explaining
that the context is chosen at load time and will not fall below the floor.

## 5. How the floor is chosen

The floor answers a different question from the one `suggest_context()`
answers today, so the arithmetic changes even though the inputs do not.

The governing constraint is that **an unreachable floor is worse than a low
one.** Section 3 shows what happens when the floor cannot be met: the fitter
abandons the attempt entirely and loads at the full context, which is the
unsafe behaviour this design exists to remove. A floor that is too high does
not produce a smaller context — it produces no fitting at all.

So the floor is the smaller of two numbers: what a person actually needs, and
what this machine can actually provide.

```
WORKING_MINIMUM = 16384

floor = min(WORKING_MINIMUM, suggest_context(...), header.context_length)
floor = max(floor, MIN_CONTEXT)          # MIN_CONTEXT is 4096
```

`WORKING_MINIMUM` is a judgement, not a measurement, and is written as a
single named constant so it can be revised. The reasoning: `local-llm` exists
largely to point coding agents at local models, and a coding agent spends
several thousand tokens on its system prompt and tool definitions before the
user has typed anything. Below roughly sixteen thousand tokens such a session
is not practical, so there is no point loading the model at all.

`suggest_context()` is kept exactly as it is, including its two-thirds margin
and its existing tests. Its role changes from "the answer" to "the ceiling on
the floor" — the guarantee that we never ask for a floor this machine cannot
reach. On a roomy machine the working minimum wins and the floor is 16384,
well below what the fitter will actually choose. On a machine too small for
that, the estimate wins and the floor drops to something achievable.
Clamping to the model's own trained context stops us setting a floor above
what the model was built for.

The new function lives beside `suggest_context()` in `gguf.py` and is the only
thing `sections.py` calls.

### 5.1 The three cases `build_section` already handles

`build_section()` in `sections.py` chooses a context in one of three ways, and
the floor only replaces one of them.

**The caller asked for a specific context** (the `context` argument is set,
today producing the comment `context: N (given)`). This keeps writing `c = N`,
exactly as it does now. Someone who names a number is making a request, not
offering a hint, and the current behaviour — the fitter stands down and the
model gets precisely that context — is what they asked for. The consequence
must be stated where the option is documented: naming a context also switches
off the load-time safety net, which is the behaviour every model has today.

**The model file's header was read** — the ordinary case, and the one section
5 describes. The floor is computed and written as `fit-ctx`.

**The header could not be read.** Today this falls back to a fixed context of
65536. Without the header there is no way to estimate what the key-value cache
will cost, so there is no way to tell whether any particular floor is
reachable, and section 5's governing constraint says an unreachable floor is
the worst outcome. The floor is therefore `MIN_CONTEXT`, 4096: low enough to
be reachable on any machine that can load the model at all, which leaves the
fitter free to choose the real number with the measurements we lack.
`FALLBACK_CONTEXT` stays in use here: with no header there is nothing to run
`suggest_context()` on either, so it is still what decides `n-predict` for
this case.

## 6. Reporting what actually happened

Delegating the choice to load time trades a number you can read in a file for
a number that varies. That trade is only acceptable if the actual number is
easy to see, so two reports change.

**`status` shows the real context of each resident model.** Today it lists
which models are loaded; it will also show what context each one received.

**`doctor` reports a failed fit.** The `failed to fit params to free device
memory` warning means a model loaded in a state we know to be unsafe.
`doctor` will scan the current log for it and report it as a finding, naming
the model and suggesting the remedies: a smaller floor, a larger
`reserve_gb`, or a smaller quantisation.

Both facts have now been checked against a live router (Apple M4 Pro, llama.cpp
0.3.0/build 10621), loading `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` through the
real `models.ini`. Both hold, so both reports go ahead.

**The router's API exposes a loaded child's context.** Loading the model and
requesting `GET /v1/models` returns, for that model's entry, a `meta` object
that is absent from every unloaded model's entry:

```json
"meta": {
    "vocab_type": true,
    "n_vocab": 151936,
    "n_ctx": 32768,
    "n_ctx_train": 32768,
    "n_embd": 1536,
    "n_params": 1777088000,
    "size": 1111370240,
    "ftype": "Q4_K - Medium"
}
```

The field is `n_ctx`, and it matches the `c = 32768` set for this model in
`models.ini`. `status` reads it from `/v1/models`; the child's own `/props`
endpoint was not needed as a fallback. Task 5 goes ahead using `meta.n_ctx`.

**The failed-fit warning is visible at the router's default log verbosity.**
Starting `llama-server` directly with no `-lv` flag at all — the same
invocation `doctor` will see in practice — logs its own verbosity as the
default:

```
0.00.056.283 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
```

Forcing an unfittable load (`--n-gpu-layers all --fit-target 38200 --fit-ctx
32768` against a machine reporting 38338 MiB of Metal memory) produced the
warning at that same default verbosity:

```
0.00.171.733 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to -2, abort
```

`grep -c 'failed to fit params' default_verbosity.log` returned `1`. The
observations in section 3 were made at `-lv 4`, but that turns out not to have
mattered — the line is a `W` (warning)-level log entry, and warnings survive
the default. `doctor` can scan the ordinary log for it without any change to
the router's verbosity. Task 6 goes ahead as specified.

## 7. Existing configuration files

Every `models.ini` already on disk pins `c` for every model, and those files
keep working exactly as they do today — llama.cpp still honours the key, and
the fitter still stands down. Nothing is rewritten automatically. Writing to
a user's configuration file without being asked would break the rule that
every such write is offered and consented to first.

New sections — from `setup`, `pull`, or any other command that adds a model —
use the floor. The two forms coexist in one file with no interaction; they
are separate keys on separate models.

`doctor` gains a note, not a finding, listing sections that still pin `c` and
explaining in one sentence what those models give up. Whether to offer an
actual migration is deliberately left out of this design: it is a separate
decision about rewriting user files, and the note is enough to make the
situation visible first.

## 8. Testing

Unit tests, which are what CI runs and which reach neither the network nor a
real `llama-server`:

- Floor selection across the range that matters: a roomy machine (floor is
  the working minimum), a machine too small for it (floor falls to the
  estimate), a model whose trained context is below the working minimum
  (floor is the model's own limit), and a machine so small the result clamps
  to `MIN_CONTEXT`. These extend `tests/unit/test_gguf.py`, which already has
  the model headers to test against.
- A generated section contains `fit-ctx` and does not contain `c`, with the
  rest of the section unchanged. This extends the existing section-building
  tests.
- The other two cases from section 5.1: a caller-supplied context still
  produces `c` and no `fit-ctx`, and an unreadable header produces a floor of
  `MIN_CONTEXT`.
- `doctor` reports the finding when a fake log contains the warning line, and
  stays quiet when it does not. The existing fake-backend fixtures cover
  this without launching anything.
- Reading a `models.ini` that pins `c` still works, and `doctor` produces the
  note rather than a finding.

One verification by hand, because no unit test can prove the key reaches
llama.cpp: write a section with a floor, start the router, load the model,
and confirm from the log that the child received `--fit-ctx` and that the
context it chose is at or above the floor. This is exactly the check already
performed while writing section 3, so the commands are known to work.

## 9. Documentation

- `docs/configuration.md` — the `models.ini` key changes, what a floor means,
  and the explicit statement that a model's context now varies between loads.
- `docs/commands.md` — the new `status` column and the new `doctor` finding.
- `docs/troubleshooting.md` — what `failed to fit params` means and what to do
  about it.
- `CHANGELOG.md` — under `## Unreleased`.
- `README.md` — checked, and changed only if needed. The front page says
  `setup` "writes a tuned section for each into `models.ini`", which stays
  true; on current reading no change is required, but the claim is re-read
  against the finished behaviour before the work is called done.

## 10. Risks and what this does not cover

**The evidence is from one machine and one build.** Everything in section 3
was observed on an Apple M4 Pro running build 10621. The fitter's behaviour
is not a documented interface and could change. The build is recorded here so
a future reader can tell whether the observations still apply.

**The behaviour on a separate graphics card is reasoned, not tested.** The
argument is that pinning the layer count leaves the fitter only the context
to reduce, which is the same lever it uses on Apple Silicon, so the outcome
should be the same or better than today — today the fitter aborts and the
load fails, whereas it would now reduce the context and succeed. This should
be confirmed on such a machine before the behaviour is described as
supported. It is not a reason to delay the change, because the current
behaviour on that hardware is worse.

**A model whose weights alone exceed memory is unchanged.** Pinning all
layers is unfittable in that case whatever the context, so the fitter aborts
exactly as it does today. This design does not improve that case; it is the
one `recommend` already exists to prevent.

**Contexts will generally get larger, not smaller.** The two-thirds margin in
`suggest_context()` is conservative, and the fitter's own margin is 1024 MiB
per device. Models will tend to load with more context than they do today.
That is the intent, but it is a behaviour change worth watching, and it is
the reason section 6 insists the real number be visible.

**Anything the system-wide llama.cpp configuration sets is invisible to this
tool.** llama.cpp now reads `~/.config/llama.cpp/config.ini` on startup, and
options there apply to our router. A `fit = off` written by hand there would
silently disable everything this design depends on. Teaching `doctor` to look
for that file is worth doing and is not covered here.
