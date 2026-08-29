import struct

import pytest

from local_llm.estimate import GIB, estimate_bytes
from local_llm.gguf import (
    GgufError,
    header_from_kv,
    kv_bytes,
    read_header,
    refined_estimate,
    suggest_context,
    suggest_context_floor,
)


def write_gguf(path, kv):
    """Write a minimal GGUF v3 file with only a key-value header."""
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))

    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    for key, value in kv.items():
        out += string(key)
        if isinstance(value, bool):
            out += struct.pack("<I", 7) + struct.pack("<?", value)
        elif isinstance(value, int):
            out += struct.pack("<I", 4) + struct.pack("<I", value)
        elif isinstance(value, float):
            out += struct.pack("<I", 6) + struct.pack("<f", value)
        elif isinstance(value, str):
            out += struct.pack("<I", 8) + string(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            out += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value))
            out += b"".join(string(v) for v in value)
        else:
            out += struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(value))
            out += struct.pack("<" + "I" * len(value), *value)
    path.write_bytes(bytes(out))


QWEN38 = {  # values read from unsloth/Qwen3.8-27B-GGUF UD-Q4_K_XL on 2026-08-26
    "general.architecture": "qwen35", "general.size_label": "27B",
    "qwen35.block_count": 65, "qwen35.context_length": 262144, "qwen35.embedding_length": 5120,
    "qwen35.attention.head_count": 24, "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
    "qwen35.full_attention_interval": 4,
    "tokenizer.chat_template": "{% if enable_thinking %}<think>{% endif %}",
}
CODER = {
    "general.architecture": "qwen3moe", "qwen3moe.block_count": 48,
    "qwen3moe.context_length": 262144, "qwen3moe.embedding_length": 2048,
    "qwen3moe.attention.head_count": 32, "qwen3moe.attention.head_count_kv": 4,
    "qwen3moe.attention.key_length": 128, "qwen3moe.attention.value_length": 128,
    "tokenizer.chat_template": "plain",
}
QWEN36 = {
    "general.architecture": "qwen35moe", "qwen35moe.block_count": 40,
    "qwen35moe.context_length": 262144, "qwen35moe.embedding_length": 2048,
    "qwen35moe.attention.head_count": 16, "qwen35moe.attention.head_count_kv": 2,
    "qwen35moe.attention.key_length": 256, "qwen35moe.attention.value_length": 256,
    "qwen35moe.full_attention_interval": 4,
}
SMALL = {
    "general.architecture": "qwen2", "qwen2.block_count": 28, "qwen2.context_length": 32768,
    "qwen2.embedding_length": 1536, "qwen2.attention.head_count": 12,
    "qwen2.attention.head_count_kv": 2,
}
BUDGET = 38 * GIB


def test_read_header_parses_real_layout(tmp_path):
    path = tmp_path / "m.gguf"
    write_gguf(
        path, {**QWEN38, "tokenizer.ggml.tokens": ["a", "b"], "tokenizer.ggml.scores": [1, 2]}
    )
    h = read_header(path)
    assert h.architecture == "qwen35" and h.block_count == 65 and h.context_length == 262144
    assert h.kv_heads == 4 and h.key_length == 256 and h.value_length == 256
    assert h.attention_layers == 16  # 65 // 4: only every fourth layer keeps a KV cache
    assert h.size_label == "27B" and h.thinking


def test_large_numeric_arrays_are_skipped_but_later_keys_still_read(tmp_path):
    path = tmp_path / "m.gguf"
    write_gguf(path, {"tokenizer.ggml.scores": list(range(5000)), **CODER})
    assert read_header(path).block_count == 48


def test_not_a_gguf(tmp_path):
    path = tmp_path / "x.gguf"
    path.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(GgufError, match="not a GGUF"):
        read_header(path)


def test_head_dim_defaults_and_per_layer_kv_array():
    small = header_from_kv(SMALL)
    assert small.key_length == 128 and small.value_length == 128  # 1536 / 12
    assert small.attention_layers == 28 and not small.thinking
    gemma = header_from_kv({
        "general.architecture": "gemma4", "gemma4.block_count": 4, "gemma4.context_length": 8192,
        "gemma4.attention.head_count": 8, "gemma4.attention.head_count_kv": [8, 8, 0, 8],
        "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
    })
    assert gemma.kv_heads == 8 and gemma.attention_layers == 3


def test_kv_bytes_per_token_matches_hand_calculation():
    assert header_from_kv(QWEN38).kv_bytes_per_token() == 16 * 512 * 4 * 1.0625  # 34816
    assert header_from_kv(CODER).kv_bytes_per_token() == 48 * 256 * 4 * 1.0625  # 52224
    assert header_from_kv(QWEN36).kv_bytes_per_token() == 10 * 512 * 2 * 1.0625  # 10880
    assert header_from_kv(SMALL).kv_bytes_per_token("f16") == 28 * 256 * 2 * 2
    assert kv_bytes(header_from_kv(QWEN38), 65536) == 34816 * 65536


def test_suggested_context_on_the_authors_mac():
    assert suggest_context(header_from_kv(QWEN38), int(17.2 * GIB), BUDGET) == 131072
    assert suggest_context(header_from_kv(CODER), int(16.5 * GIB), BUDGET) == 65536
    assert suggest_context(header_from_kv(QWEN36), int(20.8 * GIB), BUDGET) == 262144
    assert suggest_context(header_from_kv(SMALL), GIB, BUDGET, cache_type="f16") == 32768


def test_suggested_context_floors_at_minimum_and_caps_at_model_max():
    tiny_budget = 2 * GIB
    assert suggest_context(header_from_kv(QWEN38), int(17.2 * GIB), tiny_budget) == 4096
    huge = header_from_kv({**QWEN38, "qwen35.context_length": 1_000_000})
    assert suggest_context(huge, GIB, 1000 * GIB) == 262144


def test_refined_estimate_adds_kv_cache():
    header = header_from_kv(QWEN38)
    sizes = [int(17.2 * GIB)]
    assert refined_estimate(sizes, header, 65536) == estimate_bytes(sizes) + 34816 * 65536
    assert refined_estimate(sizes, None, 65536) == estimate_bytes(sizes)


def test_context_floor_is_the_working_minimum_when_the_machine_has_room():
    # The fitter will choose far more than this at load time; the floor only
    # says the model is not worth loading below it.
    assert suggest_context_floor(header_from_kv(QWEN38), int(17.2 * GIB), BUDGET) == 16384
    assert suggest_context_floor(header_from_kv(CODER), int(16.5 * GIB), BUDGET) == 16384


def test_context_floor_drops_to_what_a_small_machine_can_reach():
    # An unreachable floor is the one bad outcome: the fitter abandons the
    # attempt entirely and loads at full context.
    tiny_budget = 2 * GIB
    assert suggest_context_floor(header_from_kv(QWEN38), int(17.2 * GIB), tiny_budget) == 4096


def test_context_floor_never_exceeds_what_the_model_was_trained_for():
    short = header_from_kv({**SMALL, "qwen2.context_length": 8192})
    assert suggest_context_floor(short, GIB, BUDGET) == 8192


def test_context_floor_without_a_header_is_the_minimum():
    # No header means no way to estimate the cache cost, so no way to know
    # whether a higher floor is reachable.
    assert suggest_context_floor(None, GIB, BUDGET) == 4096
