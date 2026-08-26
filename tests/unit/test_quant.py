import pytest

from local_llm.estimate import GIB
from local_llm.quant import (
    QuantError, find_option, is_model_file, llama_tag, pick_mmproj, quality_rank, quant_label,
    quant_options, shard_info, suggest,
)

FILES = [
    ("Qwen3.8-27B-UD-Q4_K_XL.gguf", int(17.2 * GIB)),
    ("Qwen3.8-27B-Q4_K_M.gguf", int(16.5 * GIB)),
    ("Qwen3.8-27B-Q8_0.gguf", int(29 * GIB)),
    ("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", int(50 * GIB)),
    ("BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", int(4 * GIB)),
    ("mmproj-F16.gguf", int(0.8 * GIB)),
    ("mmproj-BF16.gguf", int(0.8 * GIB)),
    ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", GIB),
    ("README.md", 1000),
]


def test_tag_rules_match_llama_cpp():
    assert llama_tag("Qwen3.8-27B-UD-Q4_K_XL.gguf") == "Q4_K_XL"
    assert llama_tag("qwen2.5-1.5b-instruct-q4_k_m.gguf") == "Q4_K_M"
    assert llama_tag("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf") == "BF16"
    assert llama_tag("gemma.gguf") is None  # no "-" or "." token before the extension
    assert shard_info("a-b-00003-of-00010.gguf") == ("a-b", 3, 10)
    assert shard_info("dir/plain-Q4_0.gguf") == ("plain-Q4_0", 1, 1)


def test_quant_label_keeps_ud_prefix():
    assert quant_label("Qwen3.8-27B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert quant_label("Qwen3.8-27B-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_label("x-ud-iq2_m.gguf") == "UD-IQ2_M"


def test_model_file_filter():
    assert is_model_file("a/b-Q4_0.gguf")
    assert not is_model_file("mmproj-F16.gguf")
    assert not is_model_file("MTP/mtp-x-Q4_0.gguf")
    assert not is_model_file("README.md")


def test_quant_options_groups_shards_and_attaches_mmproj():
    options = quant_options(FILES)
    labels = [o.label for o in options]
    assert labels == ["Q8_0", "UD-Q4_K_XL", "Q4_K_M", "BF16"]  # quality order, unknown last
    bf16 = options[-1]
    assert bf16.files == ["BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf"]
    assert bf16.size == int(54 * GIB) and bf16.primary.endswith("00001-of-00002.gguf")
    assert all(o.mmproj == "mmproj-F16.gguf" and o.mmproj_size == int(0.8 * GIB) for o in options)
    assert options[1].total == int(17.2 * GIB) + int(0.8 * GIB)
    assert pick_mmproj([("mmproj-BF16.gguf", 1), ("mmproj-other.gguf", 2)]) == ("mmproj-BF16.gguf", 1)
    assert pick_mmproj([("x.gguf", 1)]) is None


def test_quality_rank():
    assert quality_rank("Q8_0") == 0 and quality_rank("ud-q4_k_xl") < quality_rank("Q4_K_M")
    assert quality_rank("MXFP4") == 15


def test_suggest_picks_highest_quality_that_is_comfortable():
    options = quant_options(FILES)
    on_48gb = suggest(options, 38 * GIB)          # Q8_0 ~34 GB fits but is not comfortable
    assert on_48gb[0].label == "UD-Q4_K_XL" and on_48gb[1] == "comfortable"
    on_128gb = suggest(options, 118 * GIB)
    assert on_128gb[0].label == "Q8_0" and on_128gb[1] == "comfortable"
    on_21gb = suggest(options, 21 * GIB)          # UD-Q4_K_XL needs 21.7 GB, Q4_K_M 20.9 GB: only the latter fits
    assert on_21gb[0].label == "Q4_K_M" and on_21gb[1] == "fits"
    on_8gb = suggest(options, 8 * GIB)
    assert on_8gb[0].label == "Q4_K_M" and on_8gb[1] == "too_big"  # smallest known, still too big
    assert suggest([], 8 * GIB) == (None, "unknown")


def test_suggest_with_only_unknown_tags_takes_the_largest_that_fits():
    options = quant_options([("gpt-oss-20b-MXFP4.gguf", int(12 * GIB)), ("gpt-oss-20b-F16.gguf", int(40 * GIB))])
    assert suggest(options, 38 * GIB)[0].label == "MXFP4"


def test_find_option_by_tag_label_and_file():
    options = quant_options(FILES)
    assert find_option(options, quant="q4_k_m").label == "Q4_K_M"
    assert find_option(options, quant="UD-Q4_K_XL").label == "UD-Q4_K_XL"
    assert find_option(options, file="Qwen3.8-27B-Q8_0.gguf").label == "Q8_0"
    with pytest.raises(QuantError, match="available: Q8_0, UD-Q4_K_XL, Q4_K_M, BF16"):
        find_option(options, quant="IQ2_M")
    assert find_option(quant_options([("m-UD-Q4_K_XL.gguf", 1), ("m-Q4_K_XL.gguf", 2)]), quant="Q4_K_XL").label == "Q4_K_XL"
    ambiguous = quant_options([("sub/m-UD-Q4_K_XL.gguf", 1), ("m-UD-Q4_K_XL.gguf", 2)])  # same label twice
    with pytest.raises(QuantError, match="--file"):
        find_option(ambiguous, quant="UD-Q4_K_XL")
