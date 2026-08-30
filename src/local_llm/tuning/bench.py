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
