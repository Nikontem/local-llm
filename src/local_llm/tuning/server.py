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
