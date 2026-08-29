"""Build the preset section for a model: name, context, output length, KV cache type, sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .estimate import GIB
from .gguf import GgufHeader, suggest_context, suggest_context_floor
from .hardware import Machine
from .quant import llama_tag
from .sampling import SamplingResult

BIG_MODEL = 10 * GIB
FALLBACK_CONTEXT = 65536


@dataclass
class SectionPlan:
    name: str
    keys: list[tuple[str, str]]
    comments: list[str] = field(default_factory=list)


def section_name(repo_id: str, filename: str) -> str:
    """The id llama-server itself would list for this file: org/repo:TAG."""
    tag = llama_tag(filename)
    return f"{repo_id}:{tag}" if tag else repo_id


def local_name(path: Path) -> str:
    stem = path.name[:-5] if path.name.lower().endswith(".gguf") else path.name
    return stem.lower()


def build_section(
    *,
    name: str,
    model_path: Path,
    mmproj_path: Path | None,
    total_bytes: int,
    header: GgufHeader | None,
    machine: Machine,
    sampling: SamplingResult,
    context: int | None = None,
    extra: Sequence[tuple[str, str]] = (),
    no_tuning: bool = False,
    origin: str = "",
    today: date | None = None,
) -> SectionPlan:
    today = today or date.today()
    big = total_bytes >= BIG_MODEL
    cache_type = "q8_0" if big and not no_tuning else "f16"

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

    if not no_tuning:
        if header is not None and header.thinking:
            keys["reasoning-format"] = "deepseek"
        for key, value in sampling.values.items():
            keys[key] = value
        if big:
            keys.setdefault("cache-type-k", "q8_0")
            keys.setdefault("cache-type-v", "q8_0")
        comments.append(f"sampling: {sampling.source}")

    for key, value in extra:
        keys[key] = value
    return SectionPlan(name, list(keys.items()), comments)
