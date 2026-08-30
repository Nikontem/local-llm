"""Measuring through llama-bench, the program shipped beside llama-server.

llama-bench loads a model file directly and times it, with no HTTP server in the
way. That makes it the cheap way to measure — one subprocess per candidate and
no process lifecycle to get wrong — and also the limited one: it cannot see
anything that exists only in the server, such as request slots, continuous
batching, or the context the fitter chooses at load time.

Whether that limitation changes which candidate wins was an open question, and
section 8 of the design document answered it by measuring one 16.5 GB model
both ways — here, and through a real `llama-server` answering real completions.
Both named the same winner and the same last place. The server-based way cost
half again as much wall clock for the same answer and two bugs of its own, so
it was deleted and this is what remains.
"""

from __future__ import annotations

import json
import subprocess
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


def parse(text: str) -> tuple[float, float, float, float]:
    """Pull the two rates, and the spread around each, out of a llama-bench JSON document.

    Returned in the order the prompt row comes first: prompt rate, prompt
    spread, generation rate, generation spread.

    The prompt row is the one with tokens to process and none to generate; the
    generation row is the reverse. Both are required — a document with only one
    of them means the invocation did not run what was asked, and silently
    reporting a zero would put a fabricated number into a ranking.

    `stddev_ts` is llama-bench's own spread across the repetitions it ran, and
    is the one field here that is read defensively: it is zero when only one
    repetition was asked for, and absent altogether from documents written by
    builds old enough not to report it. Nothing is allowed to depend on it, so
    a missing spread becomes zero rather than ending a run.
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
    return (
        float(prompt[0]["avg_ts"]),
        float(prompt[0].get("stddev_ts") or 0.0),
        float(generation[0]["avg_ts"]),
        float(generation[0].get("stddev_ts") or 0.0),
    )


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

    "Failure" here has to include a candidate that never comes back. A run that
    exceeds `_TIMEOUT` raises `subprocess.TimeoutExpired`, which is a
    `SubprocessError` and not an `OSError`, so catching only the latter would
    let exactly the slowest candidate on the slowest machine escape this loop
    and discard every measurement taken before it — the opposite of what the
    paragraph above promises.
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
        except subprocess.TimeoutExpired:
            # Its own message repeats the whole argument list, which is a table
            # row nobody can read. The number of minutes is the useful part.
            results.append(
                Measurement(candidate, 0.0, 0.0, 0, error=f"gave up after {_TIMEOUT // 60} minutes")
            )
            continue
        except (OSError, subprocess.SubprocessError) as error:
            results.append(Measurement(candidate, 0.0, 0.0, 0, error=str(error)))
            continue
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            results.append(
                Measurement(candidate, 0.0, 0.0, 0, error=detail[-1] if detail else "failed")
            )
            continue
        try:
            prompt_rate, prompt_stddev, generation_rate, generation_stddev = parse(completed.stdout)
        except TuningError as error:
            results.append(Measurement(candidate, 0.0, 0.0, 0, error=str(error)))
            continue
        results.append(
            Measurement(
                candidate,
                prompt_rate,
                generation_rate,
                repetitions,
                prompt_stddev=prompt_stddev,
                generation_stddev=generation_stddev,
            )
        )
    return results
