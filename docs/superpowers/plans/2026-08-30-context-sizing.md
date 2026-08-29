# Context Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop writing a fixed context size into generated `models.ini` sections and write a context *floor* instead, so `llama-server`'s own fitter chooses the real context from the memory actually free at load time.

**Architecture:** One new function in `gguf.py` computes the floor by capping a working minimum with what the machine can reach and what the model was trained for. `sections.py` writes it as the `fit-ctx` key instead of `c`. `doctor` and `status` are taught to show what the fitter actually did, so a number that now varies stays visible.

**Tech Stack:** Python 3.11+, Typer, pytest. No new dependencies. Tests run with `uv run pytest -q` and never touch the network or a real `llama-server`.

**Spec:** `docs/superpowers/specs/2026-08-30-context-sizing-design.md`

## Global Constraints

- Line length 100; lint with `uv run ruff check .` (rules E/F/I/UP/B).
- Docstrings explain *why*, at length. Comments inside functions justify non-obvious choices. Do not add descriptive one-liners.
- Commit subjects are lowercase plain sentences describing the user-visible effect, e.g. `fix(uninstall): a backup of your own file is never taken away unasked`. Not imperative summaries of the diff.
- `WORKING_MINIMUM = 16384`. `MIN_CONTEXT = 4096` already exists in `gguf.py`.
- Never rewrite a user's existing `models.ini`. Sections that already pin `c` keep working untouched.
- User-visible changes go in `CHANGELOG.md` under `## Unreleased` and in the `docs/` page covering the command they change.
- Every test added here runs under the existing `harness` fixture in `tests/unit/conftest.py`, which scrubs all `LOCAL_LLM_*`, XDG, `EDITOR` and `CODEX_HOME` variables. Do not bypass it.

---

### Task 1: Confirm the two facts section 6 depends on

Section 6 of the spec proposes two new reports and openly says both rest on facts nobody has established. This task establishes them **before** any code is written for Tasks 5 and 6, because a negative answer deletes those tasks rather than changing them.

This is an investigation, not a TDD cycle. Its deliverable is a decision written into the spec.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-30-context-sizing-design.md` (section 6 only)

**Interfaces:**
- Consumes: nothing.
- Produces: a decision recorded in the spec that Tasks 5 and 6 read to know whether they exist.

- [ ] **Step 1: Check whether the router API reports a loaded model's context**

Start the router on a spare port against the real config, load the smallest model, and dump what the models endpoint returns:

```bash
cd "$(mktemp -d)"
llama-server --host 127.0.0.1 --port 5690 \
  --models-preset ~/.config/local-llm/models.ini \
  --models-max 1 --models-autoload --no-ui > router.log 2>&1 &
until curl -s -m 2 http://127.0.0.1:5690/health >/dev/null; do sleep 1; done
curl -s -m 300 http://127.0.0.1:5690/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' >/dev/null
curl -s http://127.0.0.1:5690/v1/models | python3 -m json.tool
```

Look for any field naming the context — `n_ctx`, `context_length`, `ctx_size` — in the entry for the loaded model. If nothing is there, try the child's own props endpoint; the child's port appears in `router.log` on the `proxying request to model ... on port NNNNN` line:

```bash
grep -o 'on port [0-9]*' router.log | tail -1
curl -s http://127.0.0.1:<child-port>/props | python3 -m json.tool | grep -i ctx
```

- [ ] **Step 2: Check whether the failed-fit warning is visible at default verbosity**

The observations in the spec were all made at `-lv 4`. `doctor` will read the router's ordinary log, so the question is whether the warning survives at the default. Force an unfittable load by pinning layers and demanding an impossible margin, with no verbosity flag at all:

```bash
M=$(python3 - <<'EOF'
import configparser, pathlib
c = configparser.ConfigParser()
c.read(pathlib.Path.home() / ".config/local-llm/models.ini")
print(c["Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"]["model"])
EOF
)
llama-server -m "$M" --host 127.0.0.1 --port 5691 \
  --n-gpu-layers all --fit-target 38200 --fit-ctx 32768 \
  --no-ui --no-warmup > default_verbosity.log 2>&1 &
until grep -qiE 'listening on|error' default_verbosity.log; do sleep 1; done
grep -c 'failed to fit params' default_verbosity.log
```

A count of 1 or more means `doctor` can see it. A count of 0 means it cannot.

- [ ] **Step 3: Stop every test server**

```bash
pkill -f 'port 569'
pgrep -c llama-server   # expect 0
```

- [ ] **Step 4: Record the decision in the spec**

Replace the final two paragraphs of section 6 (the ones beginning "Both depend on a fact this document does not establish") with what was actually found. State the answer to each question, the evidence, and whether Task 5 and Task 6 go ahead. If the router API exposes no context, say so and record that Task 5 falls back to reading `n_ctx_slot` from the log — or is dropped, if that is judged too fragile. If the warning is invisible at default verbosity, record that Task 6 is dropped rather than silently raising the router's verbosity, which is a decision the spec explicitly refuses to make on its own.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-30-context-sizing-design.md
git commit -m "docs: the context spec records what the router API and the default log actually show"
```

---

### Task 2: The floor calculation

**Files:**
- Modify: `src/local_llm/gguf.py` (add `WORKING_MINIMUM` beside `MIN_CONTEXT` at line 18, add `suggest_context_floor()` after `suggest_context()` at line 196)
- Test: `tests/unit/test_gguf.py`

**Interfaces:**
- Consumes: `suggest_context(header, weights_bytes, budget, cache_type) -> int` and `MIN_CONTEXT` (4096), both already in `gguf.py`.
- Produces: `suggest_context_floor(header: GgufHeader | None, weights_bytes: int, budget: int, cache_type: str = "q8_0") -> int` and the module constant `WORKING_MINIMUM = 16384`. Task 3 calls this and nothing else.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_gguf.py`, and add `suggest_context_floor` to the existing `from local_llm.gguf import (...)` block at the top of the file:

```python
def test_context_floor_is_the_working_minimum_when_the_machine_has_room():
    # The fitter will choose far more than this at load time; the floor only
    # says the model is not worth loading below it.
    assert suggest_context_floor(header_from_kv(QWEN38), int(17.2 * GIB), BUDGET) == 16384
    assert suggest_context_floor(header_from_kv(CODER), int(16.5 * GIB), BUDGET) == 16384


def test_context_floor_drops_to_what_a_small_machine_can_reach():
    # An unreachable floor is the one bad outcome: the fitter abandons the
    # attempt entirely and loads at full context.
    tiny_budget = 2 * GIB
    assert suggest_context_floor(header_from_kv(QWEN38), int(17.2 * GIB), tiny_budget) == 4096


def test_context_floor_never_exceeds_what_the_model_was_trained_for():
    short = header_from_kv({**SMALL, "qwen2.context_length": 8192})
    assert suggest_context_floor(short, GIB, BUDGET) == 8192


def test_context_floor_without_a_header_is_the_minimum():
    # No header means no way to estimate the cache cost, so no way to know
    # whether a higher floor is reachable.
    assert suggest_context_floor(None, GIB, BUDGET) == 4096
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gguf.py -k floor -v`
Expected: FAIL — `ImportError: cannot import name 'suggest_context_floor'`

- [ ] **Step 3: Write the implementation**

In `src/local_llm/gguf.py`, add beside `MIN_CONTEXT` at line 18:

```python
WORKING_MINIMUM = 16384
```

And after `suggest_context()`:

```python
def suggest_context_floor(
    header: GgufHeader | None, weights_bytes: int, budget: int, cache_type: str = "q8_0"
) -> int:
    """The smallest context worth loading this model at, for llama.cpp's --fit-ctx.

    A floor is not a request. llama-server's fitter picks the real context from
    the memory free at the moment of loading; the floor only forbids it going
    lower. The one outcome to avoid is a floor this machine cannot reach: the
    fitter then abandons the attempt altogether and loads at the full,
    unreduced context, which is the unsafe state this whole approach exists to
    remove. So the working minimum is capped twice - by what the machine can
    actually reach, and by what the model was trained for.

    Without a header there is no way to estimate what the cache will cost, and
    so no way to tell whether any floor is reachable. The minimum is the only
    honest answer; the fitter has measurements we lack and can be left to it.
    """
    if header is None:
        return MIN_CONTEXT
    reachable = suggest_context(header, weights_bytes, budget, cache_type)
    trained_for = header.context_length or WORKING_MINIMUM
    return max(MIN_CONTEXT, min(WORKING_MINIMUM, reachable, trained_for))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gguf.py -v && uv run ruff check .`
Expected: PASS, including the four pre-existing `suggest_context` tests, which must not change.

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/gguf.py tests/unit/test_gguf.py
git commit -m "feat: a model gets a context floor instead of a fixed context"
```

---

### Task 3: Sections write the floor

Note a gap in the spec found while planning, and resolved here: `sections.py:70` sets `n-predict` from the chosen context (`"32768" if chosen >= 65536 else "4096"`). A floor gives no chosen context to derive that from. The resolution is that `n-predict` keeps using `suggest_context()` — the estimate — because how long a reply may be is a separate judgement from how much memory is safe, and the estimate still represents what the machine can probably do. This also means `FALLBACK_CONTEXT` stays in use for the unreadable-header case, contrary to the last sentence of the spec's section 5.1; amend that sentence when this task is done.

**Files:**
- Modify: `src/local_llm/sections.py:11` (import), `:53-71` (the context branches)
- Test: `tests/unit/test_sections.py`

**Interfaces:**
- Consumes: `suggest_context_floor()` and `suggest_context()` from Task 2.
- Produces: sections whose keys contain `fit-ctx` rather than `c`, except when the caller passed `context=N`.

- [ ] **Step 1: Write the failing tests**

Update the four existing assertions in `tests/unit/test_sections.py`. In `test_full_section_for_a_big_thinking_vision_model`, the expected keys and comments become:

```python
    assert plan.keys == [
        ("model", "/hf/Qwen3.8-27B-UD-Q4_K_XL.gguf"), ("mmproj", "/hf/mmproj-F16.gguf"),
        ("fit-ctx", "16384"), ("n-predict", "32768"), ("reasoning-format", "deepseek"),
        ("temp", "1.0"), ("top-k", "20"), ("reasoning-effort", "medium"),
        ("cache-type-k", "q8_0"), ("cache-type-v", "q8_0"),
    ]
    assert plan.comments == [
        "added by local-llm 2026-08-26 from unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_XL, 17.2 GB)",
        "context: chosen at load time to fit this machine, never below 16384",
        "sampling: profile qwen3.8",
    ]
```

In `test_small_model_without_kv_quantization_and_short_output`:

```python
    assert keys["fit-ctx"] == "16384" and keys["n-predict"] == "4096"
    assert "c" not in keys
```

In `test_overrides_no_tuning_and_missing_header`, the given-context case keeps `c` and the other two move to the floor:

```python
    assert keys["c"] == "65536" and keys["temp"] == "0.2" and keys["top-p"] == "0.9"
    assert "fit-ctx" not in keys
    assert plan.comments[1] == "context: 65536 (given; fixed, not adjusted at load time)"
```

```python
    assert [k for k, _ in bare.keys] == ["model", "fit-ctx", "n-predict"]
```

```python
    assert dict(unreadable.keys)["fit-ctx"] == "4096"
    assert "GGUF header not readable" in unreadable.comments[1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_sections.py -v`
Expected: FAIL — the built sections still contain `c`.

- [ ] **Step 3: Write the implementation**

In `src/local_llm/sections.py`, change the import on line 11:

```python
from .gguf import GgufHeader, suggest_context, suggest_context_floor
```

Replace the context branches (lines 53-71, from `if context is not None:` through the `keys["n-predict"] = ...` line) with:

```python
    keys: dict[str, str] = {"model": str(model_path)}
    if mmproj_path is not None:
        keys["mmproj"] = str(mmproj_path)

    if context is not None:
        # A named number is a request, not a hint. Writing `c` is also what
        # switches llama.cpp's own fitter off for this model - it never
        # overrides a setting the user set - so the person gets exactly the
        # context they asked for and none of the load-time adjustment.
        keys["c"] = str(context)
        expected = context
        note = f"context: {context} (given; fixed, not adjusted at load time)"
    else:
        floor = suggest_context_floor(header, total_bytes, machine.budget, cache_type)
        keys["fit-ctx"] = str(floor)
        if header is not None:
            expected = suggest_context(header, total_bytes, machine.budget, cache_type)
            note = f"context: chosen at load time to fit this machine, never below {floor}"
        else:
            expected = FALLBACK_CONTEXT
            note = (
                f"context: chosen at load time, never below {floor}"
                " (GGUF header not readable)"
            )

    # How long a reply may be is a separate judgement from how much memory is
    # safe, so it still follows the estimate rather than the floor.
    keys["n-predict"] = "32768" if expected >= 65536 else "4096"
    comments = [f"added by local-llm {today.isoformat()} from {origin}".rstrip(), note]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_sections.py -v && uv run pytest -q && uv run ruff check .`
Expected: PASS. The full suite matters here — `test_cli_pull.py` and `test_cli_setup.py` assert on written sections and will need the same `c` to `fit-ctx` change. Update every failure that is this rename; investigate any that is not.

- [ ] **Step 5: Amend the spec sentence this task disproved**

In section 5.1 of the spec, delete the final sentence ("The `FALLBACK_CONTEXT` constant becomes unused in this path and is removed if nothing else references it.") and replace it with a note that `FALLBACK_CONTEXT` is still used to decide `n-predict` when the header cannot be read.

- [ ] **Step 6: Commit**

```bash
git add src/local_llm/sections.py tests/unit/ docs/superpowers/specs/2026-08-30-context-sizing-design.md
git commit -m "feat: a new model's context is chosen when it loads, not when it is added"
```

---

### Task 4: Doctor notes sections that still pin a context

Section 7 of the spec: existing files are never rewritten, but the situation should be visible.

**Files:**
- Modify: `src/local_llm/doctor.py` (inside the `models.ini` block, around line 267)
- Test: `tests/unit/test_doctor.py`

**Interfaces:**
- Consumes: `Preset.sections()` and `Preset.get(section, key, fallback_to_star=False)`, both existing.
- Produces: a `Check` named `"context sizing"` with status `"ok"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_doctor.py`. This file does **not** use the `harness` fixture — it uses its own `healthy_paths(tmp_path)`, `Settings()`, `mac_env()` and `by_name()` helpers defined at the top of the file. Follow that pattern exactly:

```python
def test_doctor_notes_sections_that_still_pin_a_fixed_context(tmp_path):
    paths = healthy_paths(tmp_path)
    paths.preset.write_text(
        f"[*]\njinja = true\n"
        f"[a/b:Q4]\nmodel = {tmp_path}/m.gguf\nc = 65536\n"
        f"[c/d:Q4]\nmodel = {tmp_path}/m.gguf\nfit-ctx = 16384\n"
    )
    note = by_name(run_checks(paths, Settings(), env=mac_env()))["context sizing"]
    assert note.status == "ok"
    assert "a/b:Q4" in note.detail and "c/d:Q4" not in note.detail


def test_doctor_names_a_context_pinned_for_every_model_at_once(tmp_path):
    # A `c` in the wildcard block pins the context for every model, which is a
    # worse case than one section pinning its own.
    paths = healthy_paths(tmp_path)
    paths.preset.write_text(f"[*]\nc = 8192\n[m]\nmodel = {tmp_path}/m.gguf\n")
    note = by_name(run_checks(paths, Settings(), env=mac_env()))["context sizing"]
    assert note.status == "ok" and "every model" in note.detail


def test_doctor_says_nothing_when_no_section_pins_a_context(tmp_path):
    paths = healthy_paths(tmp_path)
    paths.preset.write_text(
        f"[*]\njinja = true\n[c/d:Q4]\nmodel = {tmp_path}/m.gguf\nfit-ctx = 16384\n"
    )
    assert "context sizing" not in by_name(run_checks(paths, Settings(), env=mac_env()))
```

`healthy_paths()` already writes a preset with no `c` anywhere, so the existing `test_everything_ok_on_a_healthy_mac` stays green — check that it still passes.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_doctor.py -k context_sizing -v`
Expected: FAIL with `KeyError: 'context sizing'`

- [ ] **Step 3: Write the implementation**

In `src/local_llm/doctor.py`, immediately after the `models.ini` check appends its own result (after the `except PresetError` branch, still inside `if paths.preset.is_file():`):

```python
            # Sections written before the context floor pin `c`, which stops
            # llama-server's fitter adjusting anything at all for that model.
            # Their files are never rewritten without being asked, so the only
            # thing to do is say so.
            star_context = preset.items("*").get("c")
            pinned = [
                name for name in preset.sections()
                if preset.get(name, "c", fallback_to_star=False) is not None
            ]
            if star_context is not None:
                detail = f"[*] pins c = {star_context} for every model"
                if pinned:
                    detail += f"; also set on {', '.join(pinned)}"
            elif pinned:
                detail = f"{len(pinned)} model(s) pin a fixed context: {', '.join(pinned)}"
            else:
                detail = ""
            if detail:
                checks.append(Check(
                    "context sizing", "ok", detail,
                    fix=(
                        "These load at exactly that context and llama.cpp will not adjust"
                        " them if memory is short. Replace `c = N` with `fit-ctx = N` in"
                        " models.ini to let the context be chosen at load time."
                    ),
                ))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_doctor.py -v && uv run ruff check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/doctor.py tests/unit/test_doctor.py
git commit -m "feat(doctor): models pinned to a fixed context are named"
```

---

### Task 5: Status shows the context a model actually got

**Skip this task entirely if Task 1 found no way to read a loaded model's context.** Record the skip in the plan rather than inventing a source.

**Files:**
- Modify: `src/local_llm/router.py` (extend `list_models()`'s consumer, or add a helper beside `loaded_model_names()` at line 409), `src/local_llm/cli.py:334-350`
- Test: `tests/unit/test_cli_router.py`

**Interfaces:**
- Consumes: `Router.list_models() -> dict` (existing), whose `data` entries Task 1 established do or do not carry a context field.
- Produces: the resident-model lines in `status` gain a context column.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_cli_router.py`, following the file's existing pattern for seeding `FakeHttp` responses, assert that `status` prints the context. Use the exact field name Task 1 found:

`FakeHttp` answers from a `{(method, path): response}` dictionary — seed it by assigning into `harness.http.responses`. `FakeBackend` exposes `add(pid, cmdline, rss=0, parent=None, listening=())`, not an `add_child` helper. Copy how `tests/unit/test_cli_router.py` already builds a running router with children rather than inventing a shape:

```python
def test_status_shows_the_context_each_resident_model_received(harness):
    harness.http.responses[("GET", "/v1/models")] = {"data": [
        {"id": "big", "status": {"value": "ready"}, "n_ctx": 40960},
    ]}
    # Seed the router pid file and a child process exactly as the existing
    # status tests in this file do, with the child's alias set to "big".
    result = harness.run("status")
    assert "40960" in result.output
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_cli_router.py -k context_each_resident -v`
Expected: FAIL — the context does not appear in the output.

- [ ] **Step 3: Write the implementation**

In `src/local_llm/cli.py`, the loop at line 334 already builds a `statuses` dictionary from `list_models()`. Collect the context in the same pass and print it:

```python
    contexts: dict[str, int] = {}
    try:
        for entry in router.list_models().get("data", []):
            name = str(entry.get("id"))
            statuses[name] = str((entry.get("status") or {}).get("value", ""))
            ctx = entry.get("n_ctx")
            if isinstance(ctx, int) and ctx > 0:
                contexts[name] = ctx
    except RouterError:
        pass
```

and in the line that prints each child, append the context when it is known:

```python
            ctx = contexts.get(kid.model or "")
            size = f"  ctx {ctx}" if ctx else ""
            out.print(
                f"    {kid.model or '?':<32} pid {kid.pid}  {human_gb(kid.rss)}{size}{note}"
            )
```

Replace `n_ctx` throughout with whatever field name Task 1 actually found.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_router.py -v && uv run ruff check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/cli.py tests/unit/test_cli_router.py
git commit -m "feat(status): each resident model shows the context it actually got"
```

---

### Task 6: Doctor reports a failed fit

**Skip this task entirely if Task 1 found the warning is invisible at the router's default log verbosity.** Do not raise the router's verbosity to make this work — the spec refuses that decision, and it belongs to a separate discussion.

**Files:**
- Modify: `src/local_llm/doctor.py` (a new check after the state-dir check, around line 281)
- Test: `tests/unit/test_doctor.py`

**Interfaces:**
- Consumes: `logs.current_log(log_dir) -> Path | None` from `src/local_llm/logs.py:29`.
- Produces: a `Check` named `"model fit"` with status `"warn"`.

- [ ] **Step 1: Write the failing test**

Log files are not named by hand. `logs.new_run_log(log_dir)` creates `llm-router.<timestamp>.log` **and** the `llm-router.log` symlink that `current_log()` follows; without that symlink `current_log()` returns `None` and the test would pass or fail for the wrong reason. Use it:

```python
from local_llm.logs import new_run_log


def test_doctor_reports_a_model_that_failed_to_fit(tmp_path):
    paths = healthy_paths(tmp_path)
    log = new_run_log(paths.log_dir)
    log.write_text(
        "[51937] I srv load_model: loading model 'a/b:Q4'\n"
        "[51937] W common_fit_params: failed to fit params to free device memory:"
        " n_gpu_layers already set by user to -2, abort\n"
    )
    check = by_name(run_checks(paths, Settings(), env=mac_env()))["model fit"]
    assert check.status == "warn"
    assert "reserve_gb" in (check.fix or "")


def test_doctor_is_quiet_when_every_model_fit(tmp_path):
    paths = healthy_paths(tmp_path)
    new_run_log(paths.log_dir).write_text(
        "[51937] I common_fit_params: successfully fit params\n"
    )
    assert "model fit" not in by_name(run_checks(paths, Settings(), env=mac_env()))
```

Note `test_everything_ok_on_a_healthy_mac` asserts every check is `"ok"`. It writes no log, so `current_log()` returns `None` and this check never fires there — confirm that still holds.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_doctor.py -k failed_to_fit -v`
Expected: FAIL with `KeyError: 'model fit'`

- [ ] **Step 3: Write the implementation**

In `src/local_llm/doctor.py`, add `from . import logs` to the imports and this check after the state-dir check:

```python
    # A model that could not be fitted loaded anyway, at its full unreduced
    # size. llama.cpp only warns; nothing else in this tool would notice.
    log_path = logs.current_log(paths.log_dir)
    if log_path is not None:
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            text = ""
        if "failed to fit params" in text:
            checks.append(Check(
                "model fit", "warn",
                "a model loaded without fitting into free memory; it may swap or fail",
                fix=(
                    "Lower that model's fit-ctx in models.ini, raise reserve_gb in"
                    " settings.toml, or use a smaller quantisation."
                ),
            ))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_doctor.py -v && uv run ruff check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_llm/doctor.py tests/unit/test_doctor.py
git commit -m "feat(doctor): a model that loaded without fitting is reported"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/configuration.md`, `docs/commands.md`, `docs/troubleshooting.md`, `CHANGELOG.md`
- Read and decide: `README.md`

- [ ] **Step 1: Update `docs/configuration.md`**

In the section describing `models.ini` keys, replace the description of `c` with both keys. Say plainly: `fit-ctx` is the smallest context a model will load at, `llama-server` chooses the real context from the memory free at the time, and therefore **a model's context can differ between loads** — more when the machine is idle, less when it is busy. Note that `c` still works and pins the context exactly, at the cost of switching off that adjustment.

- [ ] **Step 2: Update `docs/commands.md`**

Under `doctor`, document the `context sizing` note from Task 4 and, if Task 6 shipped, the `model fit` warning. Under `status`, if Task 5 shipped, document the context column.

- [ ] **Step 3: Update `docs/troubleshooting.md`**

Add an entry for `failed to fit params to free device memory` explaining that the model loaded without being adjusted to available memory and is likely to swap or fail, with the same three remedies as the `doctor` fix text.

- [ ] **Step 4: Update `CHANGELOG.md`**

Under `## Unreleased`, one entry per user-visible change, in the repository's existing voice.

- [ ] **Step 5: Decide on `README.md`**

Read the front page and check its claims against the new behaviour. The sentence "writes a tuned section for each into `models.ini`" is expected to remain true, so the expected outcome is no change. Change it only if a specific sentence has become false, and say which one in the commit message. Do not add a description of the floor to the front page — that belongs in `docs/configuration.md`.

- [ ] **Step 6: Commit**

```bash
git add docs/ CHANGELOG.md README.md
git commit -m "docs: a model's context is now chosen when it loads, and the pages say so"
```

---

### Task 8: End-to-end verification against a real llama-server

The unit tests prove the key is written. Only this proves llama.cpp accepts it.

**Files:** none modified.

- [ ] **Step 1: Build a scratch preset using the new key**

```bash
cd "$(mktemp -d)"
MODEL=$(python3 - <<'EOF'
import configparser, pathlib
c = configparser.ConfigParser()
c.read(pathlib.Path.home() / ".config/local-llm/models.ini")
print(c["Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"]["model"])
EOF
)
cat > scratch.ini <<EOF
version = 1

[*]
jinja = true
n-gpu-layers = all
flash-attn = auto
no-context-shift = true
load-on-startup = false

[Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M]
model = $MODEL
fit-ctx = 16384
n-predict = 4096
EOF
```

- [ ] **Step 2: Start the router and load the model**

```bash
llama-server --host 127.0.0.1 --port 5692 --models-preset scratch.ini \
  --models-max 1 --models-autoload --no-ui -lv 4 > verify.log 2>&1 &
until grep -q 'listening on http' verify.log; do sleep 1; done
curl -s -m 300 http://127.0.0.1:5692/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

- [ ] **Step 3: Confirm the three things that must be true**

```bash
grep -A1 'fit-ctx' verify.log          # the child received --fit-ctx 16384
grep 'fit params' verify.log           # the fitter ran and did not abort
grep 'n_ctx_slot' verify.log           # the context chosen is >= 16384
```

Expected: the child's argument list contains `--fit-ctx` followed by `16384`; a line reading `successfully fit params to free device memory` and **no** line containing `failed to fit`; and `n_ctx_slot` at or above 16384. If the fitter aborted, the floor was unreachable on this machine and `WORKING_MINIMUM` needs revisiting — record that rather than working around it.

- [ ] **Step 4: Stop the server**

```bash
pkill -f 'port 5692'
pgrep -c llama-server   # expect 0
```

- [ ] **Step 5: Run the whole suite one last time**

Run: `uv run pytest -q && uv run ruff check . && uv run local-llm --version`
Expected: all pass. This is what CI runs.

---

## Self-Review

**Spec coverage.** Section 3 (evidence) needs no task. Section 4 (the key changes) is Task 3. Section 5 and 5.1 (floor selection, all three cases) are Tasks 2 and 3. Section 6 (reporting) is Tasks 1, 5 and 6, with Task 1 gating the other two. Section 7 (existing files) is Task 4. Section 8 (testing) is spread across Tasks 2-6 and Task 8. Section 9 (documentation) is Task 7. Section 10 is risk narrative with nothing to build.

**Two amendments this plan makes to the spec**, both recorded as explicit steps rather than left as drift: `n-predict` keeps following `suggest_context()` and so `FALLBACK_CONTEXT` survives, contrary to the last line of section 5.1 (Task 3, Step 5); and section 6's open questions are answered in the spec itself before any code depends on them (Task 1, Step 4).

**Type consistency.** `suggest_context_floor(header, weights_bytes, budget, cache_type)` is defined in Task 2 and called with exactly that shape in Task 3. `WORKING_MINIMUM` and `MIN_CONTEXT` are the only two constants introduced or reused. Check names — `"context sizing"`, `"model fit"` — are each used in exactly one task and its tests. The `n_ctx` field name in Task 5 is explicitly marked as provisional on Task 1's finding, in both the test and the implementation step.
