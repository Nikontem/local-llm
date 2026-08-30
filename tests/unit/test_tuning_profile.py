from local_llm.preset import Preset
from local_llm.tuning import BASELINE
from local_llm.tuning.profile import AGENT, baseline, candidates


def test_the_profile_fits_under_the_context_floor():
    # gguf.suggest_context_floor writes 16384; a candidate that cannot be loaded
    # would fail for reasons that have nothing to do with its speed.
    assert AGENT.total < 16384


def test_baseline_falls_back_to_llama_cpp_defaults_when_nothing_is_written():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\n")
    base = baseline(preset, "m")
    assert (base.batch, base.ubatch) == (2048, 512)
    assert base.flash_attn == "auto"
    assert base.cache_type == "f16"
    assert base.label == BASELINE


def test_baseline_inherits_from_the_star_block():
    # flash-attn lives in the [*] block of the shipped template, never in a
    # model's own section, so a baseline that only reads the section misses it.
    text = "[*]\nflash-attn = off\n\n[m]\nmodel = /tmp/m.gguf\ncache-type-k = q8_0\n"
    preset = Preset.parse(text)
    base = baseline(preset, "m")
    assert base.flash_attn == "off"
    assert base.cache_type == "q8_0"


def test_the_candidate_list_varies_each_knob_once():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\n")
    base = baseline(preset, "m")
    result = candidates(base)
    assert result[0] == base
    labels = [c.label for c in result]
    assert labels == [
        BASELINE,
        "batch 1024/256",
        "batch 4096/1024",
        "flash-attn off",
        "cache q8_0",
    ]
    # The baseline's own pair, 2048/512, is not offered a second time.
    assert [c.pair for c in result].count((2048, 512)) == 1


def test_a_baseline_outside_the_pair_table_keeps_all_three_pairs():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\nb = 512\nub = 128\n")
    base = baseline(preset, "m")
    result = candidates(base)
    assert len(result) == 6
    assert [c.pair for c in result[1:4]] == [(1024, 256), (2048, 512), (4096, 1024)]


def test_flipping_a_flash_attention_that_is_already_off():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\nflash-attn = off\n")
    result = candidates(baseline(preset, "m"))
    assert [c for c in result if c.label.startswith("flash-attn")][0].flash_attn == "on"
