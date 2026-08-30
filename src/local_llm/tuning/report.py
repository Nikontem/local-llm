"""Turning measurements into something a person reads and a file can hold.

Measurements arrive here as plain data, and nothing here knows how they were
taken. The ranking rule is `turn_time`, imported rather than reimplemented, so
the table, the JSON and the write-back cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..preset import Preset
from . import BASELINE, Candidate, Measurement, Profile, turn_time
from .profile import DEFAULTS

INTERACTIONS_NOTE = (
    "Each setting was measured against the baseline on its own. "
    "Combinations of two changed settings were not tried."
)

REPRODUCIBILITY_NOTE = (
    "The winner and the last place are what held across repeated runs; a few percent between "
    "the rows in between is not meaningful. A busy machine can move these numbers by more than "
    "that on its own, so a surprising result is worth measuring again."
)

# How much faster than the baseline the fastest candidate has to be before the
# write-back is offered at all.
#
# This is not a significance threshold and nothing here computes a statistic.
# The design document's comparison run (section 8) found the middle of the
# ranking simply not reproducible: two ways of measuring ordered the middle
# three candidates differently, and one of them did not reproduce its own middle
# between two runs of the same model on the same machine — its baseline moved
# from 13.09 seconds to 15.71 seconds, twenty percent, from nothing but what
# else the machine was doing at the time. A test over the repetitions inside one
# run cannot see that, because the thing that moved was between the runs.
#
# So this is a judgement, in the same spirit as `gguf.WORKING_MINIMUM`: the
# point below which offering to edit somebody's configuration file would be
# claiming more than was measured.
MEANINGFUL_MARGIN = 0.05


def rank(measurements: Sequence[Measurement], profile: Profile) -> list[Measurement]:
    """Shortest turn first. Failures sort last rather than vanishing."""
    return sorted(measurements, key=lambda m: turn_time(m, profile))


def winner(measurements: Sequence[Measurement], profile: Profile) -> Measurement | None:
    usable = [m for m in rank(measurements, profile) if m.ok]
    return usable[0] if usable else None


def improvement(
    best: Measurement, measurements: Sequence[Measurement], profile: Profile
) -> float | None:
    """How much faster the winner is than the baseline, as a fraction of the baseline time.

    Positive means faster. `None` means the comparison cannot be made at all —
    the baseline failed to measure, or the winner did — and that is deliberately
    not the same value as zero: a caller must not read a missing comparison as a
    small win and go on to offer the write-back on the strength of it.
    """
    base_time = _baseline_time(measurements, profile)
    if base_time is None or not best.ok:
        return None
    return (base_time - turn_time(best, profile)) / base_time


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
    """One line per candidate, in `doctor`'s register: plain text, no markup, no colour.

    The generation rate carries the spread the engine saw across its own
    repetitions, written `±0.4`, so that a row a tenth of a percent ahead of the
    next one is visibly not ahead of it by anything. The spread is omitted
    rather than printed as `±0.0` when there is none to report, which is what a
    single repetition gives.

    The change against the baseline is printed to one decimal place. Rounded to
    whole percent, a candidate half a percent faster renders as `-0%`, which
    reads as a bug in the tool rather than as the tiny difference it is.
    """
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
            change = f"  {percent:+.1f}%"
        spread = f"±{measurement.generation_stddev:.1f}" if measurement.generation_stddev else ""
        rows.append(
            f"{label}{measurement.prompt_rate:8.0f} pp/s"
            f"{measurement.generation_rate:8.1f}{spread:>8} tg/s{seconds:8.1f} s{change}"
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
            "prompt_stddev": m.prompt_stddev,
            "generation_rate": m.generation_rate,
            "generation_stddev": m.generation_stddev,
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
    the `[*]` block, falling back to llama.cpp's own default (`profile.DEFAULTS`)
    when neither the section nor `[*]` sets the key at all. A value equal to the
    global default is not written, so a section does not slowly fill up with
    lines that change nothing and hide the ones that do.
    """
    def effective(key: str) -> str | None:
        set_value = preset.get(section, key)
        return set_value if set_value is not None else DEFAULTS.get(key)

    return [
        (key, value)
        for key, value in candidate.preset_keys()
        if effective(key) != value
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
