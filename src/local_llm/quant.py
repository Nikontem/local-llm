"""Quantization tags, file groups, and which quantization suits this machine.

Tags are read from file names exactly as llama.cpp does (get_gguf_split_info):
strip ".gguf" and a "-NNNNN-of-NNNNN" shard suffix, take the last token after
"-" or ".", upper-case it. So "Qwen3.8-27B-UD-Q4_K_XL.gguf" has tag Q4_K_XL and
label UD-Q4_K_XL; the label keeps unsloth's "UD-" marker, the tag is what
llama-server uses in model ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .estimate import estimate_bytes, fit

QUALITY_ORDER = [
    "Q8_0", "UD-Q6_K_XL", "Q6_K", "UD-Q5_K_XL", "Q5_K_M", "UD-Q4_K_XL", "Q4_K_XL",
    "Q4_K_M", "Q4_K_S", "IQ4_XS", "UD-Q3_K_XL", "Q3_K_M", "IQ3_XXS", "UD-IQ2_M", "IQ2_M",
]
NEVER_SUGGEST = {"BF16", "F16", "F32"}
_SHARD = re.compile(r"^(.+)-(\d{5})-of-(\d{5})$", re.IGNORECASE)
_TAG = re.compile(r"[-.]([A-Z0-9_]+)$", re.IGNORECASE)
_EXCLUDED_PREFIXES = ("mmproj", "mtp-", "eagle3-", "dflash-", "dspark-")


class QuantError(Exception):
    pass


def _basename(filename: str) -> str:
    return filename.rsplit("/", 1)[-1]


def shard_info(filename: str) -> tuple[str, int, int]:
    stem = _basename(filename)
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    match = _SHARD.match(stem)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return stem, 1, 1


def llama_tag(filename: str) -> str | None:
    prefix, _, _ = shard_info(filename)
    match = _TAG.search(prefix)
    return match.group(1).upper() if match else None


def quant_label(filename: str) -> str | None:
    prefix, _, _ = shard_info(filename)
    tag = llama_tag(filename)
    if tag is None:
        return None
    if prefix.upper().endswith("-UD-" + tag):
        return "UD-" + tag
    return tag


def is_model_file(filename: str) -> bool:
    name = _basename(filename).lower()
    return name.endswith(".gguf") and not name.startswith(_EXCLUDED_PREFIXES)


@dataclass
class QuantOption:
    tag: str
    label: str
    files: list[str]
    size: int
    mmproj: str | None = None
    mmproj_size: int = 0

    @property
    def total(self) -> int:
        return self.size + self.mmproj_size

    @property
    def primary(self) -> str:
        return self.files[0]


def pick_mmproj(files: list[tuple[str, int]]) -> tuple[str, int] | None:
    candidates = [
        (path, size) for path, size in files
        if _basename(path).lower().startswith("mmproj") and path.lower().endswith(".gguf")
    ]
    if not candidates:
        return None

    def rank(item: tuple[str, int]) -> int:
        name = _basename(item[0]).lower()
        return {"mmproj-f16.gguf": 0, "mmproj-bf16.gguf": 1}.get(name, 2)

    return sorted(candidates, key=rank)[0]


def quality_rank(label: str) -> int:
    try:
        return QUALITY_ORDER.index(label.upper())
    except ValueError:
        return len(QUALITY_ORDER)


def quant_options(files: list[tuple[str, int]]) -> list[QuantOption]:
    groups: dict[str, dict] = {}
    for path, size in files:
        if not is_model_file(path):
            continue
        label = quant_label(path)
        tag = llama_tag(path)
        if label is None or tag is None:
            continue
        prefix, index, _ = shard_info(path)
        directory = path.rsplit("/", 1)[0] if "/" in path else ""
        key = f"{directory}/{prefix}".lower()
        group = groups.setdefault(key, {"label": label, "tag": tag, "shards": {}, "size": 0})
        group["shards"][index] = path
        group["size"] += size
    projector = pick_mmproj(files)
    options = [
        QuantOption(
            tag=group["tag"],
            label=group["label"],
            files=[group["shards"][i] for i in sorted(group["shards"])],
            size=group["size"],
            mmproj=projector[0] if projector else None,
            mmproj_size=projector[1] if projector else 0,
        )
        for group in groups.values()
    ]
    options.sort(key=lambda option: (quality_rank(option.label), -option.size))
    return options


def suggest(options: list[QuantOption], budget: int) -> tuple[QuantOption | None, str]:
    """The best quantization that is comfortable here, else the best that fits, else the smallest."""
    known = [o for o in options if quality_rank(o.label) < len(QUALITY_ORDER)]
    if known:
        pool = sorted(known, key=lambda o: quality_rank(o.label))
    else:
        pool = sorted(
            [o for o in options if o.label.upper() not in NEVER_SUGGEST] or options,
            key=lambda o: -o.total,
        )
    if not pool:
        return None, "unknown"
    for wanted in ("comfortable", "fits"):
        for option in pool:
            if fit(estimate_bytes([option.total]), budget) == wanted:
                return option, wanted
    smallest = min(pool, key=lambda o: o.total)
    return smallest, fit(estimate_bytes([smallest.total]), budget)


def find_option(
    options: list[QuantOption], *, quant: str | None = None, file: str | None = None
) -> QuantOption:
    available = ", ".join(o.label for o in options) or "none"
    if file:
        for option in options:
            if file in option.files or _basename(file) in [_basename(f) for f in option.files]:
                return option
        raise QuantError(f"No file named {file} in this repo. Model files: "
                         + ", ".join(f for o in options for f in o.files))
    if quant:
        wanted = quant.upper()
        matches = [o for o in options if o.label.upper() == wanted]
        if not matches:
            matches = [o for o in options if o.tag.upper() == wanted]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(o.primary for o in matches)
            raise QuantError(
                f"{quant} matches more than one file ({names}); pick one with --file <name>"
            )
        raise QuantError(f"No quantization {quant} in this repo; available: {available}")
    raise QuantError(f"Choose a quantization; available: {available}")
