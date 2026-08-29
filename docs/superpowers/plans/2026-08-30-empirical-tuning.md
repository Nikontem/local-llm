# Empirical Tuning (`local-llm tune`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `local-llm tune <model>`, a command that measures how fast a model actually answers under a handful of different settings on this machine, reports the results, and offers to write the fastest one into `models.ini`.

**Architecture:** A new `src/local_llm/tuning/` package holding a shared contract (what is measured, what came back), a workload profile and candidate list, a report/ranking module, and **two interchangeable measurement engines** — one driving the `llama-bench` program, one driving a scratch `llama-server` process. Nothing downstream of the contract can tell which engine produced a number. `cli.py` gains one command that wires the pieces together, unloads the router's resident models around the run, and offers the write-back. Every side effect is injected, so the whole package is unit-testable without starting a process or opening a socket.

**Tech Stack:** Python 3.11+, Typer, `rich.console` (as a plain print sink), `psutil` behind the existing `ProcessBackend` protocol, `subprocess`, `urllib`, pytest with `typer.testing.CliRunner`.

**Spec:** `docs/superpowers/specs/2026-08-30-empirical-tuning-design.md` — read it before Task 2. It records *why* each decision was made; this plan records *how*.

## Global Constraints

- **Python 3.11 and 3.13 both must pass.** CI runs ruff plus unit tests on macOS and Linux, on both versions.
- **Line length is 100.** Ruff rules in force: `E`, `F`, `I`, `UP`, `B`.
- **Lint with `uv run ruff check src/ tests/`, never `uv run ruff check .`** — `demo/recording/logpane.py` has three pre-existing errors that belong to the repository owner and are not yours to fix.
- **Never `git add demo/`, `graphify-out/`, or `docs/superpowers/handoff-2026-08-30.md`.** They are the owner's untracked work. Stage explicit paths, never `git add -A` or `git add .`.
- **No unit test may start a process, open a socket, or reach the network.** Everything goes through `FakeBackend`, `FakeHttp`, or an injected callable. `tests/unit/conftest.py` deletes every `LOCAL_LLM_*`, XDG, `EDITOR`, `CODEX_HOME` and `OPENCODE_CONFIG_DIR` variable for a reason: leaving them set makes a test run rewrite the developer's real configuration.
- **Every write to a user's file goes through `Preset.save()`**, which writes a temporary file beside the destination and renames it, leaving a `.bak`. Never write `models.ini` in place.
- **Commit subjects are lowercase plain sentences describing the user-visible effect**, e.g. `feat(tune): a model's speed settings are measured instead of guessed`. Not imperative summaries of the diff.
- **Docstrings explain why**, often at length; comments inside functions justify non-obvious choices. Match that register — do not add descriptive one-liners.
- **The workload constants are `DEPTH = 4096`, `PROMPT = 4096`, `GENERATE = 256`.** Their total, 8448, must stay below the 16384-token floor `gguf.suggest_context_floor` writes.
- **Repetitions default to 3.**
- **Experiments during development use `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` only** (about 1 GB). Every other model in the owner's `models.ini` is 17–25 GB and loading one will thrash the machine. The single exception is Task 9, which is explicitly authorised to use a large model.
- **`~/.config/local-llm/models.ini` is read, never written, during experiments.** Build a scratch preset in a temporary directory instead.
- **There is no `timeout` command on this macOS.** Start background servers with `nohup ... &` and wait with `until grep -q '<pattern>' logfile; do sleep 1; done`.
- **Always confirm `pgrep -fl llama-server` shows nothing left running when you finish.** Note that `pgrep -c` errors with a usage message when nothing matches, which looks like a failure but is not.

---

## File Structure

**Created:**

| File | Responsibility |
| --- | --- |
| `src/local_llm/tuning/__init__.py` | The contract: `Profile`, `Candidate`, `Measurement`, `turn_time`, `TuningContext`, `TuningError`. No engine logic. |
| `src/local_llm/tuning/profile.py` | The agent-shaped workload constants, reading the baseline out of a preset, and building the candidate list. |
| `src/local_llm/tuning/report.py` | Ranking, the printed table, the JSON shape, and which `models.ini` keys a win implies. |
| `src/local_llm/tuning/bench.py` | Engine A: build `llama-bench` arguments, run it, parse its JSON. |
| `src/local_llm/tuning/server.py` | Engine B: spawn a scratch `llama-server`, drive it, read its `timings`. |
| `tests/unit/fixtures/llama_bench.json` | A real `llama-bench` JSON document captured in Task 1. |
| `tests/unit/fixtures/llama_server_timings.json` | A real completion response captured in Task 1. |
| `tests/unit/test_tuning_profile.py` | Baseline reading and candidate list. |
| `tests/unit/test_tuning_report.py` | Ranking, table text, changed keys, JSON shape. |
| `tests/unit/test_tuning_bench.py` | Argument building and fixture parsing. |
| `tests/unit/test_tuning_server.py` | Argument building, timings parsing, and the failure path. |
| `tests/unit/test_cli_tune.py` | The command end to end against fakes. |

**Modified:**

| File | Change |
| --- | --- |
| `src/local_llm/preset.py` | Add a public `comments(section)` accessor. Nothing else. |
| `src/local_llm/cli.py` | One new `tune` command plus a module-level `_ENGINES` mapping. |
| `docs/commands.md`, `docs/configuration.md`, `docs/troubleshooting.md`, `CHANGELOG.md` | Task 8. |
| `docs/superpowers/specs/2026-08-30-empirical-tuning-design.md` | Task 1 adds the captured fixtures; Task 9 adds the comparison result. |

---

## Task 1: Capture what llama-bench and llama-server actually print

Nothing is written against a guess about an output format. The context-sizing design set this precedent — its section 3 quotes observed log lines rather than paraphrasing what the source appears to do — and `llama-bench`'s JSON is no more a stable documented interface than the fitter's behaviour is. Two later tasks parse these documents, and both write their tests *from these fixtures*, so capturing them first is what stops those tests from being written against fiction.

**Files:**
- Create: `tests/unit/fixtures/llama_bench.json`
- Create: `tests/unit/fixtures/llama_server_timings.json`
- Modify: `docs/superpowers/specs/2026-08-30-empirical-tuning-design.md` (new section 7.4)

**Interfaces:**
- Consumes: nothing.
- Produces: two committed fixture files. Task 5 parses `llama_bench.json`; Task 6 parses `llama_server_timings.json`.

- [ ] **Step 1: Find the small model's path without writing the owner's config**

```bash
grep -A2 '^\[Qwen/Qwen2.5-1.5B' ~/.config/local-llm/models.ini | grep '^model'
```

Expected: one line, `model = /path/to/....gguf`. Export it:

```bash
MODEL=$(grep -A2 '^\[Qwen/Qwen2.5-1.5B' ~/.config/local-llm/models.ini | grep '^model' | sed 's/^model *= *//')
ls -la "$MODEL"
```

- [ ] **Step 2: Confirm nothing is already running**

```bash
pgrep -fl llama-server || echo "nothing running"
```

Expected: `nothing running`. If a router is up, stop it with `local-llm down` before continuing — a resident model competes for memory and the numbers become noise.

- [ ] **Step 3: Capture a real llama-bench JSON document**

This is the exact shape Task 5 builds: two tests in one invocation, one for prompt processing and one for generation, both at depth.

```bash
mkdir -p tests/unit/fixtures
llama-bench -m "$MODEL" -o json -r 2 -d 4096 -p 4096 -n 256 \
  -b 2048 -ub 512 -fa auto -ctk f16 -ctv f16 \
  2>/tmp/bench-stderr.log >tests/unit/fixtures/llama_bench.json
echo "exit=$?"
```

- [ ] **Step 4: Read the captured document and note four facts**

```bash
python3 -c "
import json
rows = json.load(open('tests/unit/fixtures/llama_bench.json'))
print('rows:', len(rows))
for r in rows:
    print({k: r[k] for k in r if k in
          ('n_prompt','n_gen','n_depth','avg_ts','stddev_ts','test','n_batch','n_ubatch','flash_attn','type_k','type_v','n_gpu_layers')})
"
```

Write down, for use in Task 5:
1. the key holding the tokens-per-second average (expected `avg_ts`);
2. the keys distinguishing the prompt row from the generation row (expected `n_prompt` non-zero with `n_gen` zero, and the reverse);
3. whether `n_depth` is reported and equals 4096;
4. **what `n_gpu_layers` reports** — the plan deliberately passes no `-ngl`, on the assumption that `llama-bench`'s default offloads every layer, matching the `n-gpu-layers = all` the router pins. Confirm it. Also check `/tmp/bench-stderr.log` for a line like `load_tensors: offloaded 29/29 layers to GPU`.

If any expectation is wrong, that is a finding, not a failure — record the real key names; Task 5's parser is written against them.

- [ ] **Step 5: Capture a real llama-server completion response**

```bash
nohup llama-server -m "$MODEL" --host 127.0.0.1 --port 5699 --no-ui \
  -c 16384 -b 2048 -ub 512 -fa auto -ctk f16 -ctv f16 -ngl all \
  >/tmp/tune-server.log 2>&1 &
until grep -q 'server is listening' /tmp/tune-server.log; do sleep 1; done
echo "server up"
```

Send a priming request that fills the cache, then the measured one. `cache_prompt` is what makes the second request process only the new tokens, which is the same thing `llama-bench`'s `-d` does.

```bash
python3 - <<'PY'
import json, urllib.request
PRIME = " ".join(["word"] * 4096)
FOLLOW = PRIME + " " + " ".join(["next"] * 4096)
def post(body):
    req = urllib.request.Request("http://127.0.0.1:5699/completion",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())
post({"prompt": PRIME, "n_predict": 1, "cache_prompt": True})
second = post({"prompt": FOLLOW, "n_predict": 256, "cache_prompt": True})
second.pop("content", None)
open("tests/unit/fixtures/llama_server_timings.json", "w").write(json.dumps(second, indent=2))
print(json.dumps(second.get("timings", {}), indent=2))
print("tokens_evaluated:", second.get("tokens_evaluated"), "tokens_predicted:", second.get("tokens_predicted"))
PY
```

Note the exact `timings` keys (expected `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`) and confirm `prompt_n` is close to 4096 rather than 8192 — that is the proof prompt caching worked and the second request measured processing *at* depth, not from scratch.

- [ ] **Step 6: Stop the server and confirm nothing survives**

```bash
pkill -f 'llama-server.*--port 5699'
sleep 2
pgrep -fl llama-server || echo "clean"
```

Expected: `clean`.

- [ ] **Step 7: Record the observations in the spec**

Add a section `### 7.4 What the two programs actually printed` to `docs/superpowers/specs/2026-08-30-empirical-tuning-design.md`, after the existing `### 7.3` and before the `## 8` heading. Quote the real key names for both documents, the observed `n_gpu_layers` value, the observed `prompt_n`, and the llama.cpp build number from `llama-server --version`. Write it as prose in the register of the existing section 3 — quoting what was seen, naming the machine and build, not paraphrasing.

- [ ] **Step 8: Commit**

```bash
git add tests/unit/fixtures/llama_bench.json tests/unit/fixtures/llama_server_timings.json \
        docs/superpowers/specs/2026-08-30-empirical-tuning-design.md
git commit -m "docs(tune): what llama-bench and llama-server actually print, captured not assumed"
```

---

## Task 2: The contract every part of tuning shares

**Files:**
- Create: `src/local_llm/tuning/__init__.py`
- Test: `tests/unit/test_tuning_report.py` (the `turn_time` tests start here; the rest of the file arrives in Task 4)

**Interfaces:**
- Consumes: `ProcessBackend` and `PsutilBackend` from `local_llm.router`; `port_in_use` from `local_llm.doctor`.
- Produces:
  - `Profile(depth: int, prompt: int, generate: int)` with `.total` property.
  - `Candidate(batch: int, ubatch: int, flash_attn: str, cache_type: str, label: str)`, frozen, with `.pair -> tuple[int, int]` and `.preset_keys() -> list[tuple[str, str]]`.
  - `Measurement(candidate: Candidate, prompt_rate: float, generation_rate: float, repetitions: int, error: str | None = None)`, frozen, with `.ok -> bool`.
  - `turn_time(measurement: Measurement, profile: Profile) -> float`.
  - `TuningContext` dataclass with fields `bench_binary`, `server_binary`, `backend`, `run`, `post`, `port_in_use`, `say`, `sleep`.
  - `EngineFn` type alias, `BASELINE = "baseline"`, `TuningError`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tuning_report.py`:

```python
from local_llm.tuning import Candidate, Measurement, Profile, turn_time

AGENT = Profile(depth=4096, prompt=4096, generate=256)
BASE = Candidate(batch=2048, ubatch=512, flash_attn="auto", cache_type="f16", label="baseline")


def test_turn_time_is_prompt_plus_generation():
    # 4096 tokens in at 512/s is 8 seconds; 256 tokens out at 32/s is another 8.
    measurement = Measurement(BASE, prompt_rate=512.0, generation_rate=32.0, repetitions=3)
    assert turn_time(measurement, AGENT) == 16.0


def test_a_failed_measurement_sorts_last():
    failed = Measurement(BASE, prompt_rate=0.0, generation_rate=0.0, repetitions=0,
                         error="did not start")
    assert turn_time(failed, AGENT) == float("inf")
    assert not failed.ok


def test_a_candidate_names_the_preset_keys_it_implies():
    candidate = Candidate(batch=4096, ubatch=1024, flash_attn="off", cache_type="q8_0",
                          label="cache q8_0")
    assert candidate.preset_keys() == [
        ("b", "4096"),
        ("ub", "1024"),
        ("flash-attn", "off"),
        ("cache-type-k", "q8_0"),
        ("cache-type-v", "q8_0"),
    ]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_tuning_report.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'local_llm.tuning'`.

- [ ] **Step 3: Write the contract**

Create `src/local_llm/tuning/__init__.py`:

```python
"""What the two measurement engines have in common, and nothing else.

Two dataclasses and one callable shape. A `Candidate` is a combination of
settings being measured; a `Measurement` is what came back for one of them; an
engine is anything that turns a list of the first into a list of the second.

The important property is negative: nothing downstream of this module — the
ranking, the report, the write-back — can tell which engine produced a number.
The design document builds two engines on purpose and expects to delete one of
them once they have been compared, and that deletion is only a deletion, rather
than a rewrite of everything that consumes measurements, because of this
boundary.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..doctor import port_in_use
from ..router import ProcessBackend, PsutilBackend

BASELINE = "baseline"
_HTTP_TIMEOUT = 600


class TuningError(Exception):
    """A measurement could not be taken at all, as opposed to being taken and being slow."""


@dataclass(frozen=True)
class Profile:
    """The shape of the work a candidate is measured doing.

    Depth is how many tokens are already in the cache when the measurement
    starts. Measuring at depth zero flatters every candidate equally and so
    tells you nothing about which to pick, which is why it is part of the
    profile rather than left at the benchmark's default of nothing.
    """

    depth: int
    prompt: int
    generate: int

    @property
    def total(self) -> int:
        return self.depth + self.prompt + self.generate


@dataclass(frozen=True)
class Candidate:
    """One combination of the four settings under test."""

    batch: int
    ubatch: int
    flash_attn: str
    cache_type: str
    label: str

    @property
    def pair(self) -> tuple[int, int]:
        return (self.batch, self.ubatch)

    def preset_keys(self) -> list[tuple[str, str]]:
        """The models.ini keys this candidate stands for, in the order they should be written.

        The two cache-type keys always move together because `sections.py` already
        treats the key and value caches as one decision, and splitting them here
        would offer a choice nothing else in the tool can express.
        """
        return [
            ("b", str(self.batch)),
            ("ub", str(self.ubatch)),
            ("flash-attn", self.flash_attn),
            ("cache-type-k", self.cache_type),
            ("cache-type-v", self.cache_type),
        ]


@dataclass(frozen=True)
class Measurement:
    """What came back for one candidate. Rates only — the ranking lives in `turn_time`."""

    candidate: Candidate
    prompt_rate: float
    generation_rate: float
    repetitions: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.prompt_rate > 0 and self.generation_rate > 0


def turn_time(measurement: Measurement, profile: Profile) -> float:
    """Seconds a coding agent waits for one reply, and the only ranking rule there is.

    Kept here rather than on `Measurement` so that both engines, the table, the
    JSON output and the write-back all rank by arithmetic that exists once. A
    measurement that failed sorts last rather than being dropped, so the report
    can still say that a candidate was tried and did not work.
    """
    if not measurement.ok:
        return float("inf")
    return profile.prompt / measurement.prompt_rate + profile.generate / measurement.generation_rate


def _urllib_post(url: str, body: dict) -> dict:
    """The default HTTP for engine B, replaced wholesale in tests."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise TuningError(f"Request to {url} failed: {error}") from error


@dataclass
class TuningContext:
    """Every side effect an engine needs, injected.

    The same reasoning as `doctor.Env` and `setup.SetupContext`: an engine that
    reaches for `subprocess` or `psutil` directly cannot be tested without
    starting a real program, and a benchmark is the slowest possible thing to
    put in a unit test suite.
    """

    bench_binary: str | None = field(default_factory=lambda: shutil.which("llama-bench"))
    server_binary: str | None = field(default_factory=lambda: shutil.which("llama-server"))
    backend: ProcessBackend = field(default_factory=PsutilBackend)
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    post: Callable[[str, dict], dict] = _urllib_post
    port_in_use: Callable[[str, int], bool] = port_in_use
    say: Callable[[str], None] = lambda message: None
    sleep: Callable[[float], None] = time.sleep


EngineFn = Callable[[Path, Profile, Sequence[Candidate], TuningContext, int], list[Measurement]]
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run pytest tests/unit/test_tuning_report.py -v && uv run ruff check src/ tests/`
Expected: 3 passed, `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/tuning/__init__.py tests/unit/test_tuning_report.py
git commit -m "feat(tune): the shared shape of a measurement, independent of what measures it"
```

---

## Task 3: The workload and the candidate list

**Files:**
- Create: `src/local_llm/tuning/profile.py`
- Test: `tests/unit/test_tuning_profile.py`

**Interfaces:**
- Consumes: `Candidate`, `Profile`, `BASELINE` from `local_llm.tuning`; `Preset` from `local_llm.preset`.
- Produces:
  - `AGENT: Profile` — the single built-in profile.
  - `baseline(preset: Preset, section: str) -> Candidate`
  - `candidates(base: Candidate) -> list[Candidate]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tuning_profile.py`:

```python
from local_llm.preset import Preset
from local_llm.tuning import BASELINE
from local_llm.tuning.profile import AGENT, baseline, candidates


def test_the_profile_fits_under_the_context_floor():
    # gguf.suggest_context_floor writes 16384; a candidate that cannot be loaded
    # would fail for reasons that have nothing to do with its speed.
    assert AGENT.total < 16384


def test_baseline_falls_back_to_llama_cpp_defaults_when_nothing_is_written():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\n")
    base = baseline(preset, "m")
    assert (base.batch, base.ubatch) == (2048, 512)
    assert base.flash_attn == "auto"
    assert base.cache_type == "f16"
    assert base.label == BASELINE


def test_baseline_inherits_from_the_star_block():
    # flash-attn lives in the [*] block of the shipped template, never in a
    # model's own section, so a baseline that only reads the section misses it.
    preset = Preset.parse("[*]\nflash-attn = off\n\n[m]\nmodel = /tmp/m.gguf\ncache-type-k = q8_0\n")
    base = baseline(preset, "m")
    assert base.flash_attn == "off"
    assert base.cache_type == "q8_0"


def test_the_candidate_list_varies_each_knob_once():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\n")
    base = baseline(preset, "m")
    result = candidates(base)
    assert result[0] == base
    labels = [c.label for c in result]
    assert labels == [
        BASELINE,
        "batch 1024/256",
        "batch 4096/1024",
        "flash-attn off",
        "cache q8_0",
    ]
    # The baseline's own pair, 2048/512, is not offered a second time.
    assert [c.pair for c in result].count((2048, 512)) == 1


def test_a_baseline_outside_the_pair_table_keeps_all_three_pairs():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\nb = 512\nub = 128\n")
    base = baseline(preset, "m")
    result = candidates(base)
    assert len(result) == 6
    assert [c.pair for c in result[1:4]] == [(1024, 256), (2048, 512), (4096, 1024)]


def test_flipping_a_flash_attention_that_is_already_off():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\nflash-attn = off\n")
    result = candidates(baseline(preset, "m"))
    assert [c for c in result if c.label.startswith("flash-attn")][0].flash_attn == "on"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_tuning_profile.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'local_llm.tuning.profile'`.

- [ ] **Step 3: Write the profile module**

Create `src/local_llm/tuning/profile.py`:

```python
"""What is measured, and which combinations get measured.

The numbers in `AGENT` are a judgement, not a measurement, in the same way
`gguf.WORKING_MINIMUM` is. They are named constants so that revising them is a
one-line change with a visible diff rather than an archaeology exercise.
"""

from __future__ import annotations

from ..preset import Preset
from . import BASELINE, Candidate, Profile

# A coding agent's traffic is lopsided and deep: a system prompt plus tool
# definitions plus file contents go in, a few hundred tokens come out, and the
# conversation grows all session. llama-bench's own defaults (512 in, 128 out,
# nothing already cached) describe a workload nobody this tool serves runs.
AGENT = Profile(depth=4096, prompt=4096, generate=256)

# llama.cpp's own defaults, which is what a section that says nothing gets.
_DEFAULT_BATCH = 2048
_DEFAULT_UBATCH = 512
_DEFAULT_FLASH = "auto"
_DEFAULT_CACHE = "f16"

# Valid pairs only: a micro-batch can never exceed its batch.
_PAIRS = ((1024, 256), (2048, 512), (4096, 1024))


def _int(preset: Preset, section: str, key: str, default: int) -> int:
    value = preset.get(section, key)
    try:
        return int(value) if value is not None else default
    except ValueError:
        # A hand-edited section with nonsense in it should not stop the command;
        # measuring against llama.cpp's default is still a useful answer.
        return default


def baseline(preset: Preset, section: str) -> Candidate:
    """The settings this model effectively runs with today.

    Read through `Preset.get`, which falls back to the `[*]` block. That fallback
    is the whole point: `flash-attn = auto` lives in the shipped template's `[*]`
    and never in a model's own section, so a baseline built only from the
    section's own keys would report `auto` as "not set" and then offer it as a
    variation to try — measuring the current behaviour twice and one real
    alternative never.
    """
    return Candidate(
        batch=_int(preset, section, "b", _DEFAULT_BATCH),
        ubatch=_int(preset, section, "ub", _DEFAULT_UBATCH),
        flash_attn=preset.get(section, "flash-attn") or _DEFAULT_FLASH,
        cache_type=preset.get(section, "cache-type-k") or _DEFAULT_CACHE,
        label=BASELINE,
    )


def candidates(base: Candidate) -> list[Candidate]:
    """The baseline, then each knob varied against it exactly once.

    This is a fixed list and not a search. The obvious alternative — sweep one
    knob, keep the winner, carry it into the next sweep — is path-dependent, and
    two measurement engines that diverge at the first step never evaluate the
    same candidate again, which would make the comparison the design document
    rests on meaningless. It also explores no more of the space than this does,
    because it varies one knob at a time as well.
    """
    result = [base]
    for batch, ubatch in _PAIRS:
        if (batch, ubatch) == base.pair:
            continue
        result.append(
            Candidate(batch, ubatch, base.flash_attn, base.cache_type, f"batch {batch}/{ubatch}")
        )
    # `auto` almost always resolves to on, so the informative flip is to off.
    flipped = "on" if base.flash_attn == "off" else "off"
    result.append(
        Candidate(base.batch, base.ubatch, flipped, base.cache_type, f"flash-attn {flipped}")
    )
    other_cache = "q8_0" if base.cache_type == "f16" else "f16"
    result.append(
        Candidate(base.batch, base.ubatch, base.flash_attn, other_cache, f"cache {other_cache}")
    )
    return result
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run pytest tests/unit/test_tuning_profile.py -v && uv run ruff check src/ tests/`
Expected: 6 passed, `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/tuning/profile.py tests/unit/test_tuning_profile.py
git commit -m "feat(tune): a coding-agent-shaped workload, and the settings measured against it"
```

---

## Task 4: Ranking, the report, and which keys a win implies

`Preset` has no public way to read a section's comments — only the private `_leading_comment_start` — and `replace_section` regenerates a section's whole body from the keys and comments it is handed. Without an accessor, offering to write a tuned value would silently destroy any comment the user wrote in that section. Adding the accessor is therefore part of this task, not a separate concern.

One consequence must be stated where a user can see it: comments written *between* two keys can only survive by being re-emitted above the section header. Nothing is lost, but such a comment moves to the top of its section. The test below asserts exactly that, so the behaviour is a decision on the record rather than a surprise.

**Files:**
- Modify: `src/local_llm/preset.py` (add `comments()` after `items()`, around line 127)
- Create: `src/local_llm/tuning/report.py`
- Test: `tests/unit/test_tuning_report.py` (extend the file created in Task 2)

**Interfaces:**
- Consumes: `Candidate`, `Measurement`, `Profile`, `turn_time`, `BASELINE` from `local_llm.tuning`; `Preset`.
- Produces:
  - `Preset.comments(section: str) -> list[str]` — every comment belonging to a section, leading and interior, without the leading `#` and surrounding space.
  - `rank(measurements, profile) -> list[Measurement]`
  - `winner(measurements, profile) -> Measurement | None`
  - `table(measurements, profile) -> list[str]`
  - `as_json(measurements, profile) -> list[dict]`
  - `changed_keys(candidate, preset, section) -> list[tuple[str, str]]`
  - `merged_keys(preset, section, changes) -> list[tuple[str, str]]`
  - `INTERACTIONS_NOTE: str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tuning_report.py`:

```python
from local_llm.preset import Preset
from local_llm.tuning import BASELINE
from local_llm.tuning import report as report_module

FAST = Candidate(batch=4096, ubatch=1024, flash_attn="auto", cache_type="f16",
                 label="batch 4096/1024")


def _measured(candidate, prompt_rate, generation_rate):
    return Measurement(candidate, prompt_rate, generation_rate, repetitions=3)


def test_a_section_reports_every_comment_it_owns():
    preset = Preset.parse(
        "# added by local-llm\n"
        "# context chosen at load time\n"
        "[m]\n"
        "model = /tmp/m.gguf\n"
        "# hand-written note from the user\n"
        "b = 2048\n"
    )
    assert preset.comments("m") == [
        "added by local-llm",
        "context chosen at load time",
        "hand-written note from the user",
    ]


def test_a_hand_written_comment_survives_a_rewrite_by_moving_above_the_header():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\n# keep me\nb = 2048\n")
    preset.replace_section("m", [("model", "/tmp/m.gguf"), ("b", "4096")], preset.comments("m"))
    text = preset.dump()
    assert "# keep me" in text
    assert text.index("# keep me") < text.index("[m]")


def test_ranking_puts_the_shortest_turn_first_and_failures_last():
    slow = _measured(BASE, 400.0, 30.0)
    fast = _measured(FAST, 800.0, 30.0)
    broken = Measurement(FAST, 0.0, 0.0, repetitions=0, error="did not start")
    ordered = report_module.rank([slow, broken, fast], AGENT)
    assert [m.candidate.label for m in ordered] == [FAST.label, BASELINE, FAST.label]
    assert ordered[-1].error == "did not start"
    assert report_module.winner([slow, broken, fast], AGENT).candidate is FAST


def test_there_is_no_winner_when_everything_failed():
    broken = Measurement(BASE, 0.0, 0.0, repetitions=0, error="did not start")
    assert report_module.winner([broken], AGENT) is None


def test_the_table_names_the_baseline_and_the_change_against_it():
    rows = report_module.table([_measured(BASE, 400.0, 30.0), _measured(FAST, 800.0, 30.0)], AGENT)
    assert any(BASELINE in row and "b=2048" in row for row in rows)
    faster = [row for row in rows if FAST.label in row][0]
    assert "-" in faster and "%" in faster


def test_a_failed_row_says_so_instead_of_printing_a_rate():
    broken = Measurement(FAST, 0.0, 0.0, repetitions=0, error="did not start")
    rows = report_module.table([_measured(BASE, 400.0, 30.0), broken], AGENT)
    assert any("did not start" in row for row in rows)


def test_only_keys_that_differ_from_the_effective_value_are_written():
    # flash-attn = auto comes from [*]; writing it into the section would add a
    # line that changes nothing.
    preset = Preset.parse("[*]\nflash-attn = auto\n\n[m]\nmodel = /tmp/m.gguf\nb = 2048\nub = 512\n")
    changes = report_module.changed_keys(FAST, preset, "m")
    assert changes == [("b", "4096"), ("ub", "1024"), ("cache-type-k", "f16"),
                       ("cache-type-v", "f16")]


def test_a_winner_identical_to_the_file_implies_no_changes():
    preset = Preset.parse(
        "[m]\nmodel = /tmp/m.gguf\nb = 2048\nub = 512\nflash-attn = auto\n"
        "cache-type-k = f16\ncache-type-v = f16\n"
    )
    assert report_module.changed_keys(BASE, preset, "m") == []


def test_merging_keeps_the_keys_the_section_already_had():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\nfit-ctx = 16384\nb = 2048\n")
    merged = report_module.merged_keys(preset, "m", [("b", "4096"), ("ub", "1024")])
    assert merged[0] == ("model", "/tmp/m.gguf")
    assert ("fit-ctx", "16384") in merged
    assert ("b", "4096") in merged and ("b", "2048") not in merged
    assert ("ub", "1024") in merged


def test_the_json_shape_carries_the_rates_and_the_derived_turn():
    payload = report_module.as_json([_measured(BASE, 400.0, 30.0)], AGENT)
    assert payload[0]["label"] == BASELINE
    assert payload[0]["batch"] == 2048
    assert payload[0]["prompt_rate"] == 400.0
    assert payload[0]["generation_rate"] == 30.0
    assert round(payload[0]["turn_seconds"], 3) == round(4096 / 400.0 + 256 / 30.0, 3)
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/unit/test_tuning_report.py -v`
Expected: FAIL — `AttributeError: 'Preset' object has no attribute 'comments'` and `ModuleNotFoundError: No module named 'local_llm.tuning.report'`.

- [ ] **Step 3: Add the comments accessor to preset.py**

Insert after `items()` (which ends around `preset.py:127`):

```python
    def comments(self, section: str) -> list[str]:
        """Every comment belonging to a section, leading and interior, stripped of its marker.

        Needed because `replace_section` regenerates a section's whole body from
        the keys and comments it is handed, so a caller that rewrites one key
        must be able to hand back everything else the section had — otherwise a
        comment the user wrote is destroyed by a command that only meant to
        change a number.

        Interior comments come back in this list too, which means a caller that
        feeds the result to `replace_section` re-emits them above the header
        rather than where they were. Nothing is lost, but a comment written
        between two keys moves to the top of its section. That is the only shape
        `replace_section` can express, and losing the text would be worse than
        moving it.
        """
        start, end = self._span(section)
        lead = self._leading_comment_start(start)
        found = []
        for line in self._lines[lead : self._content_end(start, end)]:
            if line.kind == "comment":
                found.append(line.text.lstrip("#;").strip())
        return found
```

- [ ] **Step 4: Write the report module**

Create `src/local_llm/tuning/report.py`:

```python
"""Turning measurements into something a person reads and a file can hold.

Every engine's output arrives here and nothing here knows which engine produced
it. The ranking rule is `turn_time`, imported rather than reimplemented, so the
table, the JSON and the write-back cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..preset import Preset
from . import BASELINE, Candidate, Measurement, Profile, turn_time

INTERACTIONS_NOTE = (
    "Each setting was measured against the baseline on its own. "
    "Combinations of two changed settings were not tried."
)


def rank(measurements: Sequence[Measurement], profile: Profile) -> list[Measurement]:
    """Shortest turn first. Failures sort last rather than vanishing."""
    return sorted(measurements, key=lambda m: turn_time(m, profile))


def winner(measurements: Sequence[Measurement], profile: Profile) -> Measurement | None:
    usable = [m for m in rank(measurements, profile) if m.ok]
    return usable[0] if usable else None


def _settings(candidate: Candidate) -> str:
    return (
        f"b={candidate.batch} ub={candidate.ubatch} "
        f"fa={candidate.flash_attn} cache={candidate.cache_type}"
    )


def _baseline_time(measurements: Sequence[Measurement], profile: Profile) -> float | None:
    for measurement in measurements:
        if measurement.candidate.label == BASELINE and measurement.ok:
            return turn_time(measurement, profile)
    return None


def table(measurements: Sequence[Measurement], profile: Profile) -> list[str]:
    """One line per candidate, in `doctor`'s register: plain text, no markup, no colour."""
    base_time = _baseline_time(measurements, profile)
    rows = []
    for measurement in rank(measurements, profile):
        label = f"  {measurement.candidate.label:<18}{_settings(measurement.candidate):<44}"
        if not measurement.ok:
            rows.append(f"{label}{measurement.error or 'no result'}")
            continue
        seconds = turn_time(measurement, profile)
        change = ""
        if base_time and measurement.candidate.label != BASELINE:
            percent = (seconds - base_time) / base_time * 100
            change = f"  {percent:+.0f}%"
        rows.append(
            f"{label}{measurement.prompt_rate:8.0f} pp/s"
            f"{measurement.generation_rate:8.1f} tg/s{seconds:8.1f} s{change}"
        )
    return rows


def as_json(measurements: Sequence[Measurement], profile: Profile) -> list[dict]:
    return [
        {
            "label": m.candidate.label,
            "batch": m.candidate.batch,
            "ubatch": m.candidate.ubatch,
            "flash_attn": m.candidate.flash_attn,
            "cache_type": m.candidate.cache_type,
            "prompt_rate": m.prompt_rate,
            "generation_rate": m.generation_rate,
            "turn_seconds": turn_time(m, profile),
            "repetitions": m.repetitions,
            "error": m.error,
        }
        for m in rank(measurements, profile)
    ]


def changed_keys(
    candidate: Candidate, preset: Preset, section: str
) -> list[tuple[str, str]]:
    """The keys this candidate would actually change in the file.

    Compared against the *effective* value, which `Preset.get` resolves through
    the `[*]` block. A value equal to the global default is not written, so a
    section does not slowly fill up with lines that change nothing and hide the
    ones that do.
    """
    return [
        (key, value)
        for key, value in candidate.preset_keys()
        if preset.get(section, key) != value
    ]


def merged_keys(
    preset: Preset, section: str, changes: Sequence[tuple[str, str]]
) -> list[tuple[str, str]]:
    """The section's own keys with the tuned values folded in, order preserved.

    `replace_section` writes exactly what it is handed, so everything the section
    already had — the model path, the context floor, the sampling values — has to
    be handed back or it is dropped.
    """
    keys = dict(preset.items(section))
    for key, value in changes:
        keys[key] = value
    return list(keys.items())
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `uv run pytest tests/unit/test_tuning_report.py tests/unit/test_preset.py -v && uv run ruff check src/ tests/`
Expected: all pass, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/local_llm/preset.py src/local_llm/tuning/report.py tests/unit/test_tuning_report.py
git commit -m "feat(tune): the results become a table, and a comment in a section survives a rewrite"
```

---

## Task 5: Engine A — llama-bench

**Files:**
- Create: `src/local_llm/tuning/bench.py`
- Test: `tests/unit/test_tuning_bench.py`

**Interfaces:**
- Consumes: the contract from Task 2, `tests/unit/fixtures/llama_bench.json` from Task 1.
- Produces: `bench_args(...) -> list[str]`, `parse(text: str) -> tuple[float, float]`, and `measure(...) -> list[Measurement]` matching `EngineFn`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tuning_bench.py`. **Before writing it, open `tests/unit/fixtures/llama_bench.json` and read the two rows.** The assertions below use the field names Task 1 expected; if the captured document names them differently, use the captured names in both the test and the parser — the fixture is the authority, not this plan.

```python
import json
import subprocess
from pathlib import Path

from local_llm.tuning import Candidate, Profile, TuningContext
from local_llm.tuning import bench

FIXTURE = Path(__file__).parent / "fixtures" / "llama_bench.json"
AGENT = Profile(depth=4096, prompt=4096, generate=256)
BASE = Candidate(batch=2048, ubatch=512, flash_attn="auto", cache_type="f16", label="baseline")


def test_the_arguments_ask_for_both_tests_at_depth():
    args = bench.bench_args("/opt/bin/llama-bench", Path("/m.gguf"), AGENT, BASE, 3)
    assert args[0] == "/opt/bin/llama-bench"
    joined = " ".join(args)
    assert "-m /m.gguf" in joined
    assert "-o json" in joined
    assert "-r 3" in joined
    assert "-d 4096" in joined and "-p 4096" in joined and "-n 256" in joined
    assert "-b 2048" in joined and "-ub 512" in joined
    assert "-fa auto" in joined
    assert "-ctk f16" in joined and "-ctv f16" in joined
    # No -ngl: llama-bench's default offloads every layer, which is what the
    # router pins with n-gpu-layers = all. Task 1 confirmed this.
    assert "-ngl" not in joined


def test_parsing_a_real_llama_bench_document():
    prompt_rate, generation_rate = bench.parse(FIXTURE.read_text())
    rows = json.loads(FIXTURE.read_text())
    expected_pp = [r["avg_ts"] for r in rows if r["n_prompt"] and not r["n_gen"]][0]
    expected_tg = [r["avg_ts"] for r in rows if r["n_gen"] and not r["n_prompt"]][0]
    assert prompt_rate == expected_pp
    assert generation_rate == expected_tg


def test_a_document_missing_a_row_is_an_error_not_a_zero():
    from local_llm.tuning import TuningError

    try:
        bench.parse('[{"n_prompt": 4096, "n_gen": 0, "avg_ts": 500.0}]')
    except TuningError as error:
        assert "generation" in str(error)
    else:
        raise AssertionError("expected TuningError")


def test_measure_runs_once_per_candidate_and_records_a_failure_without_stopping():
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[args.index("-b") + 1] == "4096":
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="out of memory")
        return subprocess.CompletedProcess(args, 0, stdout=FIXTURE.read_text(), stderr="")

    context = TuningContext(bench_binary="/opt/bin/llama-bench", run=fake_run)
    others = [
        BASE,
        Candidate(4096, 1024, "auto", "f16", "batch 4096/1024"),
        Candidate(2048, 512, "off", "f16", "flash-attn off"),
    ]
    results = bench.measure(Path("/m.gguf"), AGENT, others, context, 3)
    assert len(results) == 3
    assert len(calls) == 3
    assert results[0].ok and results[2].ok
    assert not results[1].ok
    assert "out of memory" in results[1].error


def test_a_missing_binary_is_refused_before_anything_runs():
    from local_llm.tuning import TuningError

    context = TuningContext(bench_binary=None, run=lambda *a, **k: None)
    try:
        bench.measure(Path("/m.gguf"), AGENT, [BASE], context, 3)
    except TuningError as error:
        assert "llama-bench" in str(error)
    else:
        raise AssertionError("expected TuningError")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_tuning_bench.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'local_llm.tuning.bench'`.

- [ ] **Step 3: Write the engine**

Create `src/local_llm/tuning/bench.py`:

```python
"""Engine A: measure with llama-bench, the program shipped beside llama-server.

llama-bench loads a model file directly and times it, with no HTTP server in the
way. That makes it the cheap engine — one subprocess per candidate and no
process lifecycle to get wrong — and also the limited one: it cannot see
anything that exists only in the server, such as request slots, continuous
batching, or the context the fitter chooses. Whether that limitation changes
which candidate wins is the question `server.py` exists to answer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from . import Candidate, Measurement, Profile, TuningContext, TuningError

_TIMEOUT = 1800


def bench_args(
    binary: str, model_path: Path, profile: Profile, candidate: Candidate, repetitions: int
) -> list[str]:
    """One invocation asking for two tests: prompt processing, then generation, both at depth.

    They are kept apart on purpose. llama-bench also offers a combined `-pg`
    form, but it reports the pair as one blended rate, and `turn_time` needs the
    two numbers separately.

    No `-ngl` is passed. llama-bench's default offloads every layer, which is
    exactly what the router pins with `n-gpu-layers = all`, and passing a number
    here would risk measuring a split the router will never run.
    """
    return [
        binary,
        "-m", str(model_path),
        "-o", "json",
        "-r", str(repetitions),
        "-d", str(profile.depth),
        "-p", str(profile.prompt),
        "-n", str(profile.generate),
        "-b", str(candidate.batch),
        "-ub", str(candidate.ubatch),
        "-fa", candidate.flash_attn,
        "-ctk", candidate.cache_type,
        "-ctv", candidate.cache_type,
    ]


def parse(text: str) -> tuple[float, float]:
    """Pull the two rates out of a llama-bench JSON document.

    The prompt row is the one with tokens to process and none to generate; the
    generation row is the reverse. Both are required — a document with only one
    of them means the invocation did not run what was asked, and silently
    reporting a zero would put a fabricated number into a ranking.
    """
    try:
        rows = json.loads(text)
    except ValueError as error:
        raise TuningError(f"llama-bench did not return JSON: {error}") from error
    prompt = [r for r in rows if r.get("n_prompt") and not r.get("n_gen")]
    generation = [r for r in rows if r.get("n_gen") and not r.get("n_prompt")]
    if not prompt:
        raise TuningError("llama-bench returned no prompt-processing row")
    if not generation:
        raise TuningError("llama-bench returned no generation row")
    return float(prompt[0]["avg_ts"]), float(generation[0]["avg_ts"])


def measure(
    model_path: Path,
    profile: Profile,
    candidates: Sequence[Candidate],
    context: TuningContext,
    repetitions: int,
) -> list[Measurement]:
    """Run llama-bench once per candidate. One failure does not end the run.

    A candidate that cannot load — usually because its cache type costs more
    memory than the machine has left — is recorded as a failed measurement and
    the rest carry on. Abandoning the whole run would throw away the results
    already paid for, which on a large model is several minutes each.
    """
    if not context.bench_binary:
        raise TuningError(
            "llama-bench was not found on PATH.\n"
            "  It ships with llama.cpp, in the same package as llama-server.\n"
            "  brew install llama.cpp"
        )
    results = []
    for candidate in candidates:
        args = bench_args(context.bench_binary, model_path, profile, candidate, repetitions)
        context.say(f"  measuring {candidate.label}")
        try:
            completed = context.run(args, capture_output=True, text=True, timeout=_TIMEOUT)
        except OSError as error:
            results.append(Measurement(candidate, 0.0, 0.0, 0, error=str(error)))
            continue
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            results.append(
                Measurement(candidate, 0.0, 0.0, 0, error=detail[-1] if detail else "failed")
            )
            continue
        try:
            prompt_rate, generation_rate = parse(completed.stdout)
        except TuningError as error:
            results.append(Measurement(candidate, 0.0, 0.0, 0, error=str(error)))
            continue
        results.append(Measurement(candidate, prompt_rate, generation_rate, repetitions))
    return results
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run pytest tests/unit/test_tuning_bench.py -v && uv run ruff check src/ tests/`
Expected: 5 passed, `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/tuning/bench.py tests/unit/test_tuning_bench.py
git commit -m "feat(tune): llama-bench measures a candidate, and a candidate that fails is recorded not fatal"
```

---

## Task 6: Engine B — a scratch llama-server

**Files:**
- Create: `src/local_llm/tuning/server.py`
- Test: `tests/unit/test_tuning_server.py`

**Interfaces:**
- Consumes: the contract from Task 2, `ProcessBackend` through `TuningContext.backend`, `tests/unit/fixtures/llama_server_timings.json` from Task 1.
- Produces: `server_args(...) -> list[str]`, `free_port(context) -> int`, `rates(response: dict) -> tuple[float, float]`, `measure(...) -> list[Measurement]` matching `EngineFn`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tuning_server.py`. As in Task 5, **read `tests/unit/fixtures/llama_server_timings.json` first** and use the key names it actually contains.

```python
import json
from pathlib import Path

from local_llm.tuning import Candidate, Profile, TuningContext, TuningError
from local_llm.tuning import server

from .fakes import FakeBackend

FIXTURE = Path(__file__).parent / "fixtures" / "llama_server_timings.json"
AGENT = Profile(depth=4096, prompt=4096, generate=256)
BASE = Candidate(batch=2048, ubatch=512, flash_attn="auto", cache_type="f16", label="baseline")


def test_the_server_is_started_with_the_candidate_and_room_for_the_profile():
    args = server.server_args("/opt/bin/llama-server", Path("/m.gguf"), AGENT, BASE, 5699)
    joined = " ".join(args)
    assert "--port 5699" in joined
    assert "-m /m.gguf" in joined
    assert "-b 2048" in joined and "-ub 512" in joined
    assert "-fa auto" in joined
    assert "-ctk f16" in joined and "-ctv f16" in joined
    assert "-ngl all" in joined
    assert "--no-ui" in joined
    # The context must hold depth + prompt + generation with room to spare, or the
    # measured request is truncated and the rate is measured on the wrong work.
    size = int(args[args.index("-c") + 1])
    assert size > AGENT.total


def test_rates_come_from_the_response_not_from_the_request():
    response = json.loads(FIXTURE.read_text())
    prompt_rate, generation_rate = server.rates(response)
    timings = response["timings"]
    assert prompt_rate == timings["prompt_n"] / (timings["prompt_ms"] / 1000)
    assert generation_rate == timings["predicted_n"] / (timings["predicted_ms"] / 1000)


def test_a_response_without_timings_is_an_error():
    try:
        server.rates({"content": "hello"})
    except TuningError as error:
        assert "timings" in str(error)
    else:
        raise AssertionError("expected TuningError")


def _context(backend, posts):
    def post(url, body):
        posts.append((url, body))
        return json.loads(FIXTURE.read_text())

    return TuningContext(
        server_binary="/opt/bin/llama-server",
        backend=backend,
        post=post,
        port_in_use=lambda host, port: False,
        sleep=lambda seconds: None,
    )


def test_each_candidate_gets_its_own_server_which_is_stopped_afterwards():
    backend = FakeBackend()
    backend.spawn_listening = {5699}
    posts = []
    context = _context(backend, posts)
    candidates = [BASE, Candidate(4096, 1024, "auto", "f16", "batch 4096/1024")]
    results = server.measure(Path("/m.gguf"), AGENT, candidates, context, 2)
    assert len(results) == 2 and all(m.ok for m in results)
    assert len(backend.spawned) == 2
    # One priming request plus two measured requests, per candidate.
    assert len(posts) == 6
    assert posts[0][1]["n_predict"] == 1
    assert posts[1][1]["n_predict"] == AGENT.generate
    assert all(body["cache_prompt"] for _, body in posts)
    # Every server started was also stopped.
    assert len(backend.terminated) == 2


def test_a_server_that_never_comes_up_is_recorded_and_the_run_continues():
    backend = FakeBackend()
    backend.spawn_dies = True
    context = _context(backend, [])
    results = server.measure(Path("/m.gguf"), AGENT, [BASE], context, 2)
    assert len(results) == 1
    assert not results[0].ok
    assert "did not start" in results[0].error


def test_a_missing_binary_is_refused_before_anything_runs():
    context = TuningContext(server_binary=None, backend=FakeBackend())
    try:
        server.measure(Path("/m.gguf"), AGENT, [BASE], context, 2)
    except TuningError as error:
        assert "llama-server" in str(error)
    else:
        raise AssertionError("expected TuningError")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_tuning_server.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'local_llm.tuning.server'`.

- [ ] **Step 3: Write the engine**

Create `src/local_llm/tuning/server.py`:

```python
"""Engine B: measure by driving a real llama-server, one per candidate.

Not the router, and not router mode. The model path and the candidate's settings
go straight onto a command line, on a port nothing else is using, and no
`models.ini` is written — not the user's, and not a scratch one. What this buys
over `bench.py` is everything that exists only in the server: request slots,
continuous batching, prompt caching between turns, and the HTTP layer itself.
Whether any of that changes which candidate wins is the empirical question the
design document leaves open, and this engine is half of the answer.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from ..logs import tail_lines
from . import Candidate, Measurement, Profile, TuningContext, TuningError

_HOST = "127.0.0.1"
_FIRST_PORT = 5699
_PORT_ATTEMPTS = 40
_START_TIMEOUT = 180.0
_POLL = 0.5
# Room above the profile so the measured request is never truncated, which would
# time a shorter piece of work than the one being compared.
_CONTEXT_SLACK = 1024


def free_port(context: TuningContext) -> int:
    """A port nothing is listening on, well away from the router's own."""
    for offset in range(_PORT_ATTEMPTS):
        port = _FIRST_PORT + offset
        if not context.port_in_use(_HOST, port):
            return port
    raise TuningError(f"No free port between {_FIRST_PORT} and {_FIRST_PORT + _PORT_ATTEMPTS}")


def server_args(
    binary: str, model_path: Path, profile: Profile, candidate: Candidate, port: int
) -> list[str]:
    return [
        binary,
        "--host", _HOST,
        "--port", str(port),
        "--no-ui",
        "-m", str(model_path),
        "-c", str(profile.total + _CONTEXT_SLACK),
        "-ngl", "all",
        "-b", str(candidate.batch),
        "-ub", str(candidate.ubatch),
        "-fa", candidate.flash_attn,
        "-ctk", candidate.cache_type,
        "-ctv", candidate.cache_type,
    ]


def _filler(word: str, tokens: int) -> str:
    """Text of roughly `tokens` tokens. Roughly is fine — see `rates`."""
    return " ".join([word] * tokens)


def rates(response: dict) -> tuple[float, float]:
    """Tokens per second in and out, computed from what the server says it did.

    The counts come from the response, never from the request. The prompt is
    built to be about four thousand tokens, but "about" is not a number you can
    divide by, and a tokenizer that splits the filler word differently would
    quietly skew every rate by the same unknown factor.
    """
    timings = response.get("timings")
    if not timings:
        raise TuningError("The server's reply carried no timings block")
    try:
        prompt_ms = float(timings["prompt_ms"])
        predicted_ms = float(timings["predicted_ms"])
        prompt_n = float(timings["prompt_n"])
        predicted_n = float(timings["predicted_n"])
    except (KeyError, TypeError, ValueError) as error:
        raise TuningError(f"The server's timings block was not readable: {error}") from error
    if prompt_ms <= 0 or predicted_ms <= 0:
        raise TuningError("The server reported a zero-length measurement")
    return prompt_n / (prompt_ms / 1000), predicted_n / (predicted_ms / 1000)


def _why(log_path: Path) -> str:
    """The last thing the scratch server said, so a failure names its own cause."""
    try:
        lines = [line.strip() for line in tail_lines(log_path, 5) if line.strip()]
    except OSError:
        return ""
    return f": {lines[-1]}" if lines else ""


def _wait_for_listening(context: TuningContext, pid: int, port: int, log_path: Path) -> None:
    waited = 0.0
    while waited < _START_TIMEOUT:
        if context.backend.info(pid) is None:
            raise TuningError(f"The server did not start{_why(log_path)}")
        if context.backend.listening(pid, port):
            return
        context.sleep(_POLL)
        waited += _POLL
    raise TuningError(f"The server did not start listening in time{_why(log_path)}")


def _measure_one(
    model_path: Path,
    profile: Profile,
    candidate: Candidate,
    context: TuningContext,
    repetitions: int,
) -> Measurement:
    port = free_port(context)
    args = server_args(context.server_binary or "llama-server", model_path, profile,
                       candidate, port)
    # A real log rather than /dev/null: when a candidate fails to load, the reason
    # is in the server's own output and nowhere else, and Task 9 depends on being
    # able to say why a measurement is missing.
    log_path = Path(tempfile.gettempdir()) / f"local-llm-tune-{port}.log"
    pid = context.backend.spawn(args, log_path)
    try:
        _wait_for_listening(context, pid, port, log_path)
        url = f"http://{_HOST}:{port}/completion"
        prime = _filler("word", profile.depth)
        follow = prime + " " + _filler("next", profile.prompt)
        # The priming turn fills the cache. With cache_prompt on, the measured
        # turn then processes only the new tokens, which is the same thing
        # llama-bench's -d option does and the only way the two engines are
        # measuring comparable work.
        context.post(url, {"prompt": prime, "n_predict": 1, "cache_prompt": True})
        prompt_rates, generation_rates = [], []
        for _ in range(repetitions):
            response = context.post(
                url,
                {"prompt": follow, "n_predict": profile.generate, "cache_prompt": True},
            )
            prompt_rate, generation_rate = rates(response)
            prompt_rates.append(prompt_rate)
            generation_rates.append(generation_rate)
        return Measurement(
            candidate,
            sum(prompt_rates) / len(prompt_rates),
            sum(generation_rates) / len(generation_rates),
            repetitions,
        )
    except TuningError as error:
        return Measurement(candidate, 0.0, 0.0, 0, error=str(error))
    finally:
        # Stopping is in a finally because a server left running holds the whole
        # model in memory, and the next candidate would then be measured against
        # a machine the previous candidate is still using.
        context.backend.terminate(pid)
        context.backend.wait(pid, 15.0)
        if context.backend.info(pid) is not None:
            context.backend.kill(pid)


def measure(
    model_path: Path,
    profile: Profile,
    candidates: Sequence[Candidate],
    context: TuningContext,
    repetitions: int,
) -> list[Measurement]:
    if not context.server_binary:
        raise TuningError(
            "llama-server was not found on PATH.\n"
            "  local-llm doctor    will say how to install it"
        )
    results = []
    for candidate in candidates:
        context.say(f"  measuring {candidate.label}")
        results.append(_measure_one(model_path, profile, candidate, context, repetitions))
    return results
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run pytest tests/unit/test_tuning_server.py -v && uv run ruff check src/ tests/`
Expected: 6 passed, `All checks passed!`.

If `test_a_server_that_never_comes_up_is_recorded_and_the_run_continues` fails on the error text, check `FakeBackend.spawn_dies`: it marks the process dead immediately, so `backend.info(pid)` returns `None` and `_wait_for_listening` raises `"The server did not start"`. Match the assertion to that string rather than loosening the check.

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/tuning/server.py tests/unit/test_tuning_server.py
git commit -m "feat(tune): a scratch server measures the same candidates, so the two can be compared"
```

---

## Task 7: The `tune` command

**Files:**
- Modify: `src/local_llm/cli.py` (add `_ENGINES` beside the other module-level hooks near line 92, and the command after `unload`, around line 599)
- Test: `tests/unit/test_cli_tune.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6; `agents.resolve_model`, `complete_model`, `fail`, `state`, `Router.loaded_model_names`, `Router.unload_model`, `Router.load_model`, `gguf.read_header`, `gguf.refined_estimate`, `hardware.detect`, `estimate.human_gb`.
- Produces: the `tune` command and `cli._ENGINES`, a `dict[str, EngineFn]` that tests replace.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_cli_tune.py`:

```python
from pathlib import Path

from local_llm import cli
from local_llm.preset import Preset
from local_llm.tuning import Candidate, Measurement


def _fake_engine(results):
    def engine(model_path, profile, candidates, context, repetitions):
        return [
            Measurement(c, results[i][0], results[i][1], repetitions)
            for i, c in enumerate(candidates)
        ]

    return engine


def _flat(rate=400.0):
    # Every candidate identical, so the baseline wins.
    return [(rate, 30.0)] * 8


def _second_is_fastest():
    return [(400.0, 30.0), (900.0, 30.0)] + [(300.0, 25.0)] * 6


def running_router(h):
    h.backend.add(4242, ["llama-server", "--models-preset", str(h.paths.preset)],
                  listening={5678})
    h.paths.state_dir.mkdir(parents=True, exist_ok=True)
    h.paths.pid_file.write_text("4242")


def test_tune_refuses_an_unknown_model(harness):
    result = harness.run("tune", "nope")
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_tune_reports_a_table_and_names_the_winner(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small")
    assert result.exit_code == 0
    assert "baseline" in result.output
    assert "batch 1024/256" in result.output
    assert "fastest" in result.output
    assert "Combinations of two changed settings were not tried." in result.output


def test_tune_offers_nothing_when_the_baseline_wins(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_flat()))
    result = harness.run("tune", "small")
    assert result.exit_code == 0
    assert "already the fastest" in result.output
    assert Preset.load(harness.paths.preset).get("small", "b") is None


def test_tune_writes_only_the_keys_that_change(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 0
    preset = Preset.load(harness.paths.preset)
    assert preset.get("small", "b") == "1024"
    assert preset.get("small", "ub") == "256"
    # The model path the section already had is still there.
    assert preset.get("small", "model", fallback_to_star=False).endswith("small.gguf")


def test_tune_leaves_the_file_alone_without_yes_when_not_interactive(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small")
    assert Preset.load(harness.paths.preset).get("small", "b") is None
    assert "--yes" in result.output


def test_tune_json_carries_every_candidate(harness):
    import json

    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small", "--json")
    payload = json.loads(result.output)
    assert len(payload) >= 5
    assert payload[0]["label"] == "batch 1024/256"
    assert "turn_seconds" in payload[0]


def test_tune_unloads_a_resident_model_and_puts_it_back(harness):
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 0
    assert ("POST", "/models/unload", {"model": "big"}) in harness.http.calls
    assert ("POST", "/models/load", {"model": "big"}) in harness.http.calls
    assert harness.http.calls.index(("POST", "/models/unload", {"model": "big"})) < \
        harness.http.calls.index(("POST", "/models/load", {"model": "big"}))


def test_a_resident_model_is_reloaded_even_when_the_engine_explodes(harness):
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)

    def exploding(model_path, profile, candidates, context, repetitions):
        raise RuntimeError("boom")

    harness.monkeypatch.setitem(cli._ENGINES, "bench", exploding)
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code != 0
    assert ("POST", "/models/load", {"model": "big"}) in harness.http.calls


def test_the_engine_flag_selects_the_server_engine(harness):
    seen = []

    def engine(model_path, profile, candidates, context, repetitions):
        seen.append("server")
        return [Measurement(c, 400.0, 30.0, repetitions) for c in candidates]

    harness.monkeypatch.setitem(cli._ENGINES, "server", engine)
    result = harness.run("tune", "small", "--engine", "server")
    assert result.exit_code == 0
    assert seen == ["server"]


def test_an_unknown_engine_is_refused(harness):
    result = harness.run("tune", "small", "--engine", "guess")
    assert result.exit_code == 1
    assert "guess" in result.output
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_cli_tune.py -v`
Expected: FAIL, `AttributeError: module 'local_llm.cli' has no attribute '_ENGINES'`.

- [ ] **Step 3: Add the engine registry to cli.py**

Beside the other module-level hooks tests replace (`_sleep`, `_open_url`, `_which`, near `cli.py:90`):

```python
# The two measurement engines, as a mapping so a test can replace one without a
# subprocess and so `--engine` has exactly one place to look them up.
_ENGINES: dict[str, tuning.EngineFn] = {
    "bench": tuning_bench.measure,
    "server": tuning_server.measure,
}
```

with the imports beside the existing ones at the top of the file:

```python
from . import tuning
from .tuning import bench as tuning_bench
from .tuning import profile as tuning_profile
from .tuning import report as tuning_report
from .tuning import server as tuning_server
```

`cli.py` imports its own package relatively (`from .agents import ...`), so these
match. Note `cli.py` already imports a different `Candidate`, from `.discover`;
the tuning one is only ever reached as `tuning.Candidate`, so there is no clash
to resolve — do not rename either.

- [ ] **Step 4: Write the command**

Add after the `unload` command (around `cli.py:599`):

```python
@app.command()
def tune(
    model: str = typer.Argument(..., autocompletion=complete_model, help="Model to measure."),
    engine: str = typer.Option(
        "bench", "--engine", help="Which measurement to use: bench (llama-bench) or server."
    ),
    repetitions: int = typer.Option(
        3, "--repetitions", help="How many times each setting is measured."
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the measurements as JSON."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Write the winning settings without asking."),
) -> None:
    """Measure how fast a model answers under different settings, and offer to keep the best."""
    if engine not in _ENGINES:
        fail(f"Unknown engine: {engine}\n  choose one of: {', '.join(sorted(_ENGINES))}")
    st = state()
    preset = st.preset()
    try:
        name = resolve_model(model, preset, st.settings)
    except AgentError as error:
        fail(str(error))
    raw_path = preset.get(name, "model")
    if not raw_path:
        fail(f"[{name}] has no model path, so there is nothing to measure.")
    model_path = Path(raw_path)
    if not model_path.exists():
        fail(f"The model file is missing: {model_path}")

    base = tuning_profile.baseline(preset, name)
    candidates = tuning_profile.candidates(base)
    profile = tuning_profile.AGENT
    _refuse_if_it_cannot_fit(st, preset, name, model_path, profile)

    context = tuning.TuningContext(say=(lambda message: None) if json_out else out.print)
    if not json_out:
        out.print(f"measuring {name} with {engine}, {len(candidates)} settings, "
                  f"{repetitions} runs each")
    with _router_out_of_the_way(st, yes, json_out):
        try:
            measurements = _ENGINES[engine](model_path, profile, candidates, context, repetitions)
        except tuning.TuningError as error:
            fail(str(error))

    if json_out:
        out.print(json.dumps(tuning_report.as_json(measurements, profile), indent=2))
        return

    out.print("")
    for row in tuning_report.table(measurements, profile):
        out.print(row)
    out.print("")
    out.print(f"  {tuning_report.INTERACTIONS_NOTE}")
    out.print("")

    best = tuning_report.winner(measurements, profile)
    if best is None:
        fail("Nothing could be measured. See the errors above.")
    if best.candidate.label == tuning.BASELINE:
        out.print(f"{name} is already the fastest of the settings tried. Nothing to change.")
        return
    changes = tuning_report.changed_keys(best.candidate, preset, name)
    if not changes:
        out.print(f"{name} is already the fastest of the settings tried. Nothing to change.")
        return

    out.print(f"fastest: {best.candidate.label}")
    for key, value in changes:
        out.print(f"  {key} = {value}")
    if not yes:
        if not _interactive():
            out.print(f"\nRe-run with --yes to write these into [{name}].")
            return
        if not typer.confirm(f"\nWrite these into [{name}] in {st.paths.preset}?", default=True):
            return
    preset.replace_section(name, tuning_report.merged_keys(preset, name, changes),
                           _tuning_comments(preset, name, best, profile))
    preset.save(st.paths.preset)
    out.print(f"written to {st.paths.preset}")
    _after_preset_change(st)
```

and the three helpers it uses, immediately above it:

```python
def _refuse_if_it_cannot_fit(st: State, preset: Preset, name: str, model_path: Path,
                             profile: tuning.Profile) -> None:
    """Decline rather than measure a model this machine cannot hold.

    A benchmark that swaps produces confident nonsense, which is the one output a
    tuning command must never produce.
    """
    try:
        header = read_header(model_path)
    except GgufError:
        return  # No header, no estimate; the engines will fail honestly if it cannot load.
    machine = detect(reserve_gb=st.settings.reserve_gb)
    needed = refined_estimate(preset.file_sizes(name), header, profile.total, "f16")
    if needed > machine.budget:
        fail(
            f"{name} needs about {human_gb(needed)} to measure and this machine has "
            f"{human_gb(machine.budget)} usable.\n"
            f"  local-llm doctor    for what is taking the rest"
        )


@contextmanager
def _router_out_of_the_way(st: State, yes: bool, quiet: bool):
    """Unload whatever the router is holding, and put it back whatever happens.

    The reload is registered before the first measurement runs, so a crashed
    engine, a server that never comes up, or an interrupted run all still leave
    the router holding what it held before. The router process itself is never
    stopped, so its port keeps answering throughout.
    """
    router = st.router()
    resident: list[str] = []
    if router.is_running():
        resident = [name for name in router.loaded_model_names() if name]
    if resident and not quiet:
        out.print(f"the router is holding {', '.join(resident)}; unloading while measuring")
    if resident and not yes and _interactive():
        if not typer.confirm("Unload and reload afterwards?", default=True):
            raise typer.Exit(1)
    for name in resident:
        try:
            router.unload_model(name)
        except RouterError as error:
            fail(f"Could not unload {name}: {error}")
    try:
        yield
    finally:
        for name in resident:
            try:
                router.load_model(name)
            except RouterError as error:
                err.print(f"warning: {name} could not be reloaded: {error}")


def _tuning_comments(preset: Preset, name: str, best, profile: tuning.Profile) -> list[str]:
    """The section's existing comments plus one line saying where the new numbers came from."""
    seconds = tuning.turn_time(best, profile)
    note = (
        f"tuned by local-llm on {date.today().isoformat()}: {best.candidate.label}, "
        f"{seconds:.1f}s for a {profile.prompt}-token turn at depth {profile.depth}"
    )
    return [*preset.comments(name), note]
```

Add `from contextlib import contextmanager` and `from datetime import date` to the imports if they are not already present, and confirm `read_header`, `GgufError`, `refined_estimate`, `detect`, `human_gb`, `RouterError`, `AgentError` and `resolve_model` are imported — `cli.py` already uses all of them.

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `uv run pytest tests/unit/test_cli_tune.py -v && uv run ruff check src/ tests/`
Expected: 10 passed, `All checks passed!`.

- [ ] **Step 6: Run the whole suite, because this task touched shared code**

Run: `uv run pytest -q`
Expected: every test passes. If a `doctor` or `preset` test broke, the `comments()` addition from Task 4 or the new import block is the cause — fix it rather than adjusting the test.

- [ ] **Step 7: Commit**

```bash
git add src/local_llm/cli.py tests/unit/test_cli_tune.py
git commit -m "feat(tune): a model's speed settings are measured on this machine instead of guessed"
```

---

## Task 8: Documentation

**Files:**
- Modify: `docs/commands.md`, `docs/configuration.md`, `docs/troubleshooting.md`, `CHANGELOG.md`
- Modify: `README.md` — only if reading it shows a claim that has stopped being true

**Interfaces:**
- Consumes: the finished command from Task 7.
- Produces: nothing code depends on.

- [ ] **Step 1: Capture real output to quote**

The docs quote captured output, never invented output. Run the command against the small model with the router down:

```bash
local-llm tune 'Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M' --repetitions 2 | tee /tmp/tune-output.txt
pgrep -fl llama-server || echo "clean"
```

- [ ] **Step 2: Add the command to docs/commands.md**

Under `## Models`, matching the shape of the surrounding entries — a bolded invocation with its flags inline, an em-dashed one-sentence purpose, then prose, then a fenced block of the real output from step 1:

```markdown
- **`local-llm tune <model> [--engine bench|server] [--repetitions N] [--json]
  [-y]`** — measure how fast a model actually answers on this machine, and offer
  to keep the settings that answered fastest. Everything else `local-llm` writes
  is estimated from the model file and your memory; this is the one command that
  runs the model and times it. It measures four settings — the batch size and
  micro-batch size, whether flash attention is used, and whether the key-value
  cache is stored at full or reduced precision — against a workload shaped like a
  coding agent's: four thousand tokens already in the conversation, four thousand
  more sent, and a short reply. Each setting is measured on its own against your
  current configuration, so a combination of two changes is never tried. Expect
  ten to twenty minutes on a large model. If the router is holding a model it is
  unloaded for the duration and reloaded afterwards, because a second copy of a
  model in memory makes every number meaningless. Nothing is written to
  `models.ini` until you say so.
```

- [ ] **Step 3: Add the four keys to docs/configuration.md**

In the `models.ini` key reference, describe `b`, `ub`, `flash-attn` and the `cache-type-k` / `cache-type-v` pair: what each does, that `local-llm` leaves the first two at llama.cpp's default unless `tune` writes them, and that a value written by `tune` was measured on the machine it ran on — safe anywhere, because the fitter still governs what fits, but not necessarily fast on a different machine.

- [ ] **Step 4: Add an entry to docs/troubleshooting.md**

Cover the three things that actually go wrong: a run is far slower than expected because something else is using the GPU; a candidate reports an error instead of a rate, which usually means its cache type needed more memory than was free; and `tune` declines before starting because the model plus the profile does not fit the budget.

- [ ] **Step 5: Add the change to CHANGELOG.md**

Under `## Unreleased`, one entry in the repository's voice — a plain sentence describing the user-visible effect.

- [ ] **Step 6: Re-read README.md against what now exists**

The front page claims `setup` "writes a tuned section for each into `models.ini`". That word now means two different things. If reading the paragraph cold is misleading, fix it in one sentence; if it still reads correctly, change nothing and say so in the commit body.

- [ ] **Step 7: Commit**

```bash
git add docs/commands.md docs/configuration.md docs/troubleshooting.md CHANGELOG.md
git commit -m "docs: what tune measures, what it writes, and what to do when a run looks wrong"
```

(Add `README.md` to that `git add` only if step 6 changed it. Never add `demo/` or `graphify-out/`.)

---

## Task 9: The comparison, and deleting the engine that loses

This is the task the whole design exists to reach. Everything before it was built so that this question could be answered with numbers instead of argument.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-30-empirical-tuning-design.md` (fill in section 8's result)
- Delete, if engine A wins: `src/local_llm/tuning/server.py`, `tests/unit/test_tuning_server.py`, `tests/unit/fixtures/llama_server_timings.json`
- Modify, if engine A wins: `src/local_llm/cli.py` (drop `--engine` and the registry entry), `docs/commands.md`

**Interfaces:**
- Consumes: the finished command.
- Produces: a settled codebase and a recorded finding.

- [ ] **Step 1: Confirm the machine is quiet**

```bash
local-llm down
pgrep -fl llama-server || echo "clean"
```

Expected: `clean`. Close anything else using the GPU. **This step is not optional** — the entire experiment is a comparison of two rankings, and a background process that shifts memory pressure partway through can reorder one of them on its own.

- [ ] **Step 2: Pick the large model and confirm the authorisation**

The standing rule is that experiments use the 1.5 GB model only. **This task is the single documented exception**, because a 1 GB model on a 38 GB machine leaves cache precision and flash attention with nothing to push against, and two engines that agree because neither measured anything have not agreed about anything.

```bash
local-llm models
```

Choose one model between 17 and 25 GB. Record its name — it goes in the spec.

- [ ] **Step 3: Run both engines and keep the output**

```bash
local-llm tune '<model>' --engine bench  --json > /tmp/tune-bench.json
local-llm tune '<model>' --engine server --json > /tmp/tune-server.json
pgrep -fl llama-server || echo "clean"
```

- [ ] **Step 4: Compare the two rankings**

```bash
python3 - <<'PY'
import json
for path in ("/tmp/tune-bench.json", "/tmp/tune-server.json"):
    rows = json.load(open(path))
    print(path)
    for r in rows:
        print(f"  {r['label']:<20} {r['turn_seconds']:8.2f}s  "
              f"pp {r['prompt_rate']:8.1f}  tg {r['generation_rate']:6.1f}  {r['error'] or ''}")
    print("  winner:", rows[0]["label"])
    print()
PY
```

The rule, fixed in the spec before any of these numbers existed: **do the two engines name the same winner?** Absolute rates being offset between them changes no decision anyone makes and is not a reason to keep both.

- [ ] **Step 5: Write the result into the spec**

Section 8 currently states the rule and both possible outcomes but records no result. Add the result to it: the model, the machine, the llama.cpp build from `llama-server --version`, both full rankings as they printed, and the verdict. Write it in the register of section 3 of the context-sizing design — quoting what was seen, not paraphrasing it.

- [ ] **Step 6a: If the two engines named the same winner, delete engine B**

```bash
git rm src/local_llm/tuning/server.py tests/unit/test_tuning_server.py \
       tests/unit/fixtures/llama_server_timings.json
```

Then in `cli.py`: remove the `tuning_server` import, reduce `_ENGINES` to the single `bench` entry or drop the mapping entirely, and remove the `--engine` option and its unknown-engine check. Remove `test_the_engine_flag_selects_the_server_engine` and `test_an_unknown_engine_is_refused` from `tests/unit/test_cli_tune.py`. Remove `--engine` from the `docs/commands.md` entry.

```bash
uv run pytest -q && uv run ruff check src/ tests/
git add -A src/local_llm tests/unit docs CHANGELOG.md
git commit -m "refactor(tune): one way of measuring, because both ways picked the same settings"
```

- [ ] **Step 6b: If they named different winners, keep both and say why**

Leave `--engine` in place. Document in `docs/commands.md` what each engine measures and when the difference matters, using the observed disagreement as the example. Add a `CHANGELOG.md` entry.

```bash
uv run pytest -q && uv run ruff check src/ tests/
git add docs/commands.md CHANGELOG.md docs/superpowers/specs/2026-08-30-empirical-tuning-design.md
git commit -m "docs(tune): the two ways of measuring disagree, so both stay and the difference is named"
```

- [ ] **Step 7: Confirm the machine is as you found it**

```bash
pgrep -fl llama-server || echo "clean"
local-llm status
```

Restart the router if it was running when you started, and confirm `git status --short` still shows `demo/` and `graphify-out/` as untracked and unstaged.
