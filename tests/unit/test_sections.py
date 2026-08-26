from datetime import date
from pathlib import Path

from local_llm.estimate import GIB
from local_llm.gguf import header_from_kv
from local_llm.hardware import Machine
from local_llm.sampling import SamplingResult
from local_llm.sections import build_section, local_name, section_name

from .test_gguf import QWEN38, SMALL

MAC = Machine("Darwin", "arm64", "Apple M4 Pro", 48 * GIB, "apple", 48 * GIB, 20, 10)
TODAY = date(2026, 8, 26)


def test_section_name_uses_llama_servers_tag():
    assert section_name("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf") == "unsloth/Qwen3.8-27B-GGUF:Q4_K_XL"
    assert section_name("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf") == "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
    assert section_name("x/y", "BF16/m-BF16-00001-of-00002.gguf") == "x/y:BF16"
    assert local_name(Path("/m/My-Model-Q4_K_M.gguf")) == "my-model-q4_k_m"


def test_full_section_for_a_big_thinking_vision_model():
    plan = build_section(
        name="unsloth/Qwen3.8-27B-GGUF:Q4_K_XL",
        model_path=Path("/hf/Qwen3.8-27B-UD-Q4_K_XL.gguf"), mmproj_path=Path("/hf/mmproj-F16.gguf"),
        total_bytes=int(18 * GIB), header=header_from_kv(QWEN38), machine=MAC,
        sampling=SamplingResult({"temp": "1.0", "top-k": "20", "reasoning-effort": "medium"}, "profile qwen3.8"),
        origin="unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_XL, 17.2 GB)", today=TODAY,
    )
    assert plan.name == "unsloth/Qwen3.8-27B-GGUF:Q4_K_XL"
    assert plan.keys == [
        ("model", "/hf/Qwen3.8-27B-UD-Q4_K_XL.gguf"), ("mmproj", "/hf/mmproj-F16.gguf"),
        ("c", "131072"), ("n-predict", "32768"), ("reasoning-format", "deepseek"),
        ("temp", "1.0"), ("top-k", "20"), ("reasoning-effort", "medium"),
        ("cache-type-k", "q8_0"), ("cache-type-v", "q8_0"),
    ]
    assert plan.comments == [
        "added by local-llm 2026-08-26 from unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_XL, 17.2 GB)",
        "context: 131072 suggested for this machine (38.0 GB usable)",
        "sampling: profile qwen3.8",
    ]


def test_small_model_without_kv_quantization_and_short_output():
    plan = build_section(
        name="Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M", model_path=Path("/hf/small.gguf"), mmproj_path=None,
        total_bytes=GIB, header=header_from_kv(SMALL), machine=MAC,
        sampling=SamplingResult({"temp": "0.7"}, "model card"), today=TODAY,
    )
    keys = dict(plan.keys)
    assert keys["c"] == "32768" and keys["n-predict"] == "4096"
    assert "cache-type-k" not in keys and "reasoning-format" not in keys and "mmproj" not in keys
    assert keys["temp"] == "0.7"


def test_overrides_no_tuning_and_missing_header():
    plan = build_section(
        name="x", model_path=Path("/m.gguf"), mmproj_path=None, total_bytes=int(18 * GIB),
        header=header_from_kv(QWEN38), machine=MAC, sampling=SamplingResult({"temp": "1.0"}, "profile q"),
        context=65536, extra=[("temp", "0.2"), ("top-p", "0.9")], today=TODAY,
    )
    keys = dict(plan.keys)
    assert keys["c"] == "65536" and keys["temp"] == "0.2" and keys["top-p"] == "0.9"
    assert plan.comments[1] == "context: 65536 (given)"
    bare = build_section(
        name="x", model_path=Path("/m.gguf"), mmproj_path=None, total_bytes=int(18 * GIB),
        header=header_from_kv(QWEN38), machine=MAC, sampling=SamplingResult({"temp": "1.0"}, "profile q"),
        no_tuning=True, today=TODAY,
    )
    assert [k for k, _ in bare.keys] == ["model", "c", "n-predict"]
    assert not any(c.startswith("sampling") for c in bare.comments)
    unreadable = build_section(
        name="x", model_path=Path("/m.gguf"), mmproj_path=None, total_bytes=GIB, header=None, machine=MAC,
        sampling=SamplingResult({}, "none"), today=TODAY,
    )
    assert dict(unreadable.keys)["c"] == "65536"
    assert "GGUF header not readable" in unreadable.comments[1]
