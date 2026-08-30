"""The vocabulary a measurement engine shares with everything that reads its numbers.

Two dataclasses and one callable shape. A `Candidate` is a combination of
settings being measured; a `Measurement` is what came back for one of them; an
engine is anything that turns a list of the first into a list of the second.

There were two engines once. Section 8 of the design document
(`docs/superpowers/specs/2026-08-30-empirical-tuning-design.md`) built a second
one on top of a real `llama-server` beside the `llama-bench` one, ran both
against the same 16.5 GB model, found that they named the same winner and the
same last place, and deleted the slower of the two. That removal was a deletion
and not a rewrite of everything downstream precisely because of the boundary
this module draws: the ranking, the report and the write-back never learn how a
number was produced, so taking an engine away touched nothing outside its own
file. The boundary stays for the reason it was worth having in the first place
— it is what made that comparison cheap enough to run, and what would make a
third engine cheap to try against the one that survived.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BASELINE = "baseline"


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
    """What came back for one candidate. Rates only — the ranking lives in `turn_time`.

    The two standard deviations are the spread the engine saw across its own
    repetitions of this one candidate. They are carried so that the table can
    show how steady a number was, and nothing more is done with them: no test
    of significance is computed anywhere, because the variation this command
    actually suffers from is between runs rather than within one — the design
    document's comparison run measured the same baseline at 13.09 seconds once
    and 15.71 seconds a while later, which is what else the machine was doing,
    not something a within-run spread can see.

    They default to zero so that a caller with no spread to report — a test, or
    an engine that times a single pass — builds a `Measurement` exactly as
    before.
    """

    candidate: Candidate
    prompt_rate: float
    generation_rate: float
    repetitions: int
    error: str | None = None
    prompt_stddev: float = 0.0
    generation_stddev: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and self.prompt_rate > 0 and self.generation_rate > 0


def turn_time(measurement: Measurement, profile: Profile) -> float:
    """Seconds a coding agent waits for one reply, and the only ranking rule there is.

    Kept here rather than on `Measurement` so that the engine, the table, the
    JSON output and the write-back all rank by arithmetic that exists once. A
    measurement that failed sorts last rather than being dropped, so the report
    can still say that a candidate was tried and did not work.
    """
    if not measurement.ok:
        return float("inf")
    return profile.prompt / measurement.prompt_rate + profile.generate / measurement.generation_rate


@dataclass
class TuningContext:
    """Every side effect an engine needs, injected.

    The same reasoning as `doctor.Env` and `setup.SetupContext`: an engine that
    reaches for `subprocess` directly cannot be tested without starting a real
    program, and a benchmark is the slowest possible thing to put in a unit
    test suite.
    """

    bench_binary: str | None = field(default_factory=lambda: shutil.which("llama-bench"))
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    say: Callable[[str], None] = lambda message: None
    # Waiting is a side effect like any other. The caller uses this to let the
    # operating system reclaim an unloaded model's pages before the first
    # candidate is measured; a unit test replaces it and the wait costs nothing.
    sleep: Callable[[float], None] = time.sleep


EngineFn = Callable[[Path, Profile, Sequence[Candidate], TuningContext, int], list[Measurement]]
