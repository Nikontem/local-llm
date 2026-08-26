import pytest

from local_llm.preset import Preset, PresetError

SAMPLE = """version = 1

[*]
jinja = true
n-gpu-layers = all   ; trailing comment
# Release weights after 5 minutes idle
sleep-idle-seconds = 300

[qwen-3.8-q4]
model = /models/Qwen3.8-27B-UD-Q4_K_XL.gguf
mmproj = /models/mmproj-F16.gguf
c = 65536
temp = 1.0

[tiny]
model = /models/tiny#1.gguf
c = 32768
"""


def test_round_trip_is_byte_identical():
    assert Preset.parse(SAMPLE).dump() == SAMPLE


def test_sections_excludes_star_and_keeps_order():
    assert Preset.parse(SAMPLE).sections() == ["qwen-3.8-q4", "tiny"]


def test_items_and_get_with_star_fallback():
    p = Preset.parse(SAMPLE)
    assert p.items("qwen-3.8-q4") == {
        "model": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "mmproj": "/models/mmproj-F16.gguf",
        "c": "65536",
        "temp": "1.0",
    }
    assert p.get("tiny", "c") == "32768"
    assert p.get("tiny", "jinja") == "true"                       # inherited from [*]
    assert p.get("tiny", "jinja", fallback_to_star=False) is None
    assert p.get("tiny", "nope") is None
    assert p.get("*", "n-gpu-layers") == "all"                    # inline comment stripped


def test_hash_inside_a_value_without_whitespace_is_kept():
    assert Preset.parse(SAMPLE).get("tiny", "model") == "/models/tiny#1.gguf"


def test_has_section():
    p = Preset.parse(SAMPLE)
    assert p.has_section("tiny") and not p.has_section("missing")


def test_load_missing_file_raises_preset_error(tmp_path):
    with pytest.raises(PresetError, match="Missing model config"):
        Preset.load(tmp_path / "none.ini")


def test_file_sizes_sums_model_and_mmproj(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x" * 10)
    (tmp_path / "p.gguf").write_bytes(b"y" * 5)
    p = Preset.parse(
        f"[a]\nmodel = {tmp_path}/m.gguf\nmmproj = {tmp_path}/p.gguf\n"
        f"[b]\nmodel = {tmp_path}/m.gguf\n"
    )
    assert p.file_sizes("a") == [10, 5]
    assert p.file_sizes("b") == [10]


def test_file_sizes_reports_missing_file_and_missing_key(tmp_path):
    p = Preset.parse(f"[a]\nmodel = {tmp_path}/gone.gguf\n[b]\nc = 1\n")
    with pytest.raises(PresetError, match="gone.gguf"):
        p.file_sizes("a")
    with pytest.raises(PresetError, match="no model key"):
        p.file_sizes("b")
