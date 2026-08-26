"""Read a GGUF file's header and turn it into memory facts.

A GGUF file starts with a magic, a version, a tensor count, a key-value count
and then typed key-value pairs; the tensor data comes after. Only the
key-value section is read here. No dependency is needed for that.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .estimate import estimate_bytes

MAGIC = b"GGUF"
MIN_CONTEXT = 4096
MAX_SUGGESTED_CONTEXT = 262144
_STRING = 8
_ARRAY = 9
_SCALAR = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}
_MAX_ARRAY = 4096  # longer arrays (vocabularies) are skipped, we never need them
_BYTES_PER_ELEMENT = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
    "q4_1": 0.625, "q4_0": 0.5625, "iq4_nl": 0.5625,
}


class GgufError(Exception):
    pass


@dataclass
class GgufHeader:
    architecture: str
    block_count: int
    context_length: int
    kv_heads: int
    attention_layers: int
    key_length: int
    value_length: int
    chat_template: str = ""
    size_label: str = ""
    parameter_count: int = 0

    @property
    def thinking(self) -> bool:
        return "<think>" in self.chat_template or "enable_thinking" in self.chat_template

    def kv_bytes_per_token(self, cache_type: str = "q8_0") -> int:
        per_layer = (self.key_length + self.value_length) * self.kv_heads
        return int(self.attention_layers * per_layer * bytes_per_element(cache_type))


def bytes_per_element(cache_type: str) -> float:
    return _BYTES_PER_ELEMENT.get(cache_type.lower(), 2.0)


class _Reader:
    def __init__(self, handle) -> None:
        self._handle = handle

    def read(self, count: int) -> bytes:
        data = self._handle.read(count)
        if len(data) != count:
            raise GgufError("truncated GGUF header")
        return data

    def skip(self, count: int) -> None:
        self._handle.seek(count, 1)

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.read(self.scalar("<Q")).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        self.skip(self.scalar("<Q"))

    def value(self, kind: int):
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            element_kind = self.scalar("<I")
            count = self.scalar("<Q")
            if element_kind == _STRING:
                if count > _MAX_ARRAY:
                    for _ in range(count):
                        self.skip_string()
                    return []
                return [self.string() for _ in range(count)]
            if element_kind == _ARRAY:
                return [self.value(_ARRAY) for _ in range(count)]
            fmt = _SCALAR.get(element_kind)
            if fmt is None:
                raise GgufError(f"unknown GGUF value type {element_kind}")
            size = struct.calcsize(fmt)
            if count > _MAX_ARRAY:
                self.skip(size * count)
                return []
            data = self.read(size * count)
            return list(struct.unpack("<" + fmt[1] * count, data))
        fmt = _SCALAR.get(kind)
        if fmt is None:
            raise GgufError(f"unknown GGUF value type {kind}")
        return self.scalar(fmt)


def read_header(path: Path) -> GgufHeader:
    try:
        with open(path, "rb") as handle:
            reader = _Reader(handle)
            if reader.read(4) != MAGIC:
                raise GgufError(f"not a GGUF file: {path}")
            version = reader.scalar("<I")
            if version < 2:
                raise GgufError(f"unsupported GGUF version {version}: {path}")
            reader.scalar("<Q")  # tensor count, unused
            count = reader.scalar("<Q")
            kv: dict[str, object] = {}
            for _ in range(count):
                key = reader.string()
                kind = reader.scalar("<I")
                kv[key] = reader.value(kind)
    except OSError as error:
        raise GgufError(f"cannot read {path}: {error}") from None
    return header_from_kv(kv)


def header_from_kv(kv: dict) -> GgufHeader:
    arch = str(kv.get("general.architecture") or "")
    if not arch:
        raise GgufError("GGUF header has no general.architecture")

    def number(key: str, default: int = 0) -> int:
        value = kv.get(f"{arch}.{key}", default)
        if isinstance(value, list):
            value = max(value) if value else default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    block_count = number("block_count")
    head_count = number("attention.head_count") or 1
    kv_heads = number("attention.head_count_kv") or head_count
    head_dim = number("embedding_length") // head_count if head_count else 0
    key_length = number("attention.key_length") or head_dim
    value_length = number("attention.value_length") or key_length
    interval = number("full_attention_interval")
    attention_layers = block_count // interval if interval > 1 else block_count
    per_layer = kv.get(f"{arch}.attention.head_count_kv")
    if isinstance(per_layer, list) and per_layer:
        attention_layers = sum(1 for heads in per_layer if heads)
    try:
        parameter_count = int(kv.get("general.parameter_count") or 0)
    except (TypeError, ValueError):
        parameter_count = 0
    return GgufHeader(
        architecture=arch,
        block_count=block_count,
        context_length=number("context_length"),
        kv_heads=kv_heads,
        attention_layers=attention_layers,
        key_length=key_length,
        value_length=value_length,
        chat_template=str(kv.get("tokenizer.chat_template") or ""),
        size_label=str(kv.get("general.size_label") or ""),
        parameter_count=parameter_count,
    )


def kv_bytes(header: GgufHeader, context: int, cache_type: str = "q8_0") -> int:
    return header.kv_bytes_per_token(cache_type) * context


def suggest_context(
    header: GgufHeader, weights_bytes: int, budget: int, cache_type: str = "q8_0"
) -> int:
    """Largest power-of-two context that fits, leaving two thirds of the free memory."""
    cap = min(header.context_length or MAX_SUGGESTED_CONTEXT, MAX_SUGGESTED_CONTEXT)
    weights = estimate_bytes([weights_bytes])
    free = budget - weights
    chosen = MIN_CONTEXT
    context = MIN_CONTEXT
    while context <= cap:
        cache = kv_bytes(header, context, cache_type)
        if weights + cache <= budget and cache * 3 <= free:
            chosen = context
        context *= 2
    return chosen


def refined_estimate(
    sizes: Iterable[int], header: GgufHeader | None, context: int, cache_type: str = "q8_0"
) -> int:
    base = estimate_bytes(sizes)
    if header is None:
        return base
    return base + kv_bytes(header, context, cache_type)
