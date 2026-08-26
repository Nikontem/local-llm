"""Build the preset section for a model: name, context, output length, KV cache type, sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .estimate import GIB, human_gb
from .gguf import GgufHeader, suggest_context
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

    if context is not None:
        chosen = context
        note = f"context: {chosen} (given)"
    elif header is not None:
        chosen = suggest_context(header, total_bytes, machine.budget, cache_type)
        note = f"context: {chosen} suggested for this machine ({human_gb(machine.budget)} usable)"
    else:
        chosen = FALLBACK_CONTEXT
        note = f"context: {chosen} (GGUF header not readable; default)"

    keys: dict[str, str] = {"model": str(model_path)}
    if mmproj_path is not None:
        keys["mmproj"] = str(mmproj_path)
    keys["c"] = str(chosen)
    keys["n-predict"] = "32768" if chosen >= 65536 else "4096"
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
