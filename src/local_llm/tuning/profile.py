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

# The same defaults, keyed by the models.ini setting they apply to, so that
# `report.changed_keys` can tell "unset" from "set to something else" without
# duplicating these values a second time. A key missing from a section (and not
# supplied by the `[*]` block either) runs at this default, so a candidate equal
# to it changes nothing and must not be written.
DEFAULTS: dict[str, str] = {
    "b": str(_DEFAULT_BATCH),
    "ub": str(_DEFAULT_UBATCH),
    "flash-attn": _DEFAULT_FLASH,
    "cache-type-k": _DEFAULT_CACHE,
    "cache-type-v": _DEFAULT_CACHE,
}

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
    knob, keep the winner, carry it into the next sweep — explores no more of
    the combination space than this does, because it also varies one knob at a
    time, and it gives up two things in exchange for nothing. It is path
    dependent: one measurement that happens to land badly, on a machine that
    was briefly busy, steers every measurement after it. And its table can no
    longer be read the way the printed one is read, as "here is what each
    setting on its own did to the configuration you have today", because after
    the first stage the rows are no longer being compared against that.
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
