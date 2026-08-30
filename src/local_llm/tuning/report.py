"""Turning measurements into something a person reads and a file can hold.

Every engine's output arrives here and nothing here knows which engine produced
it. The ranking rule is `turn_time`, imported rather than reimplemented, so the
table, the JSON and the write-back cannot drift apart.
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
