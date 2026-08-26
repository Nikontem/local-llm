from local_llm.sampling import (
    SamplingResult,
    load_profiles,
    parse_model_card,
    parse_preset_ini,
    profile_for,
    values_for,
)

CARD = """
# Qwen3.8-27B

We recommend `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `min_p=0` for thinking mode.
For non-thinking mode use temperature=0.7, top_p=0.8.
| Presence Penalty | 1.5 |
Repetition penalty: 1.05
The context window is 262144 tokens; top_k=999999 in the appendix is ignored.
"""


def test_parse_model_card_takes_the_first_plausible_value_of_each_key():
    assert parse_model_card(CARD) == {
        "temp": "0.6", "top-p": "0.95", "top-k": "20", "min-p": "0",
        "presence-penalty": "1.5", "repeat-penalty": "1.05",
    }


def test_parse_model_card_rejects_out_of_range_and_returns_empty_when_nothing_matches():
    assert parse_model_card("temperature = 7.5 and top_p: 3") == {}
    assert parse_model_card("no numbers here") == {}
    assert parse_model_card("TopP=0.9 Temp: 1") == {"top-p": "0.9", "temp": "1"}


def test_parse_preset_ini_merges_star_and_matching_section():
    text = (
        "[*]\nmmap = 1\ntemp = 0.5\n[Qwen3.8-27B]\n"
        "hf = unsloth/Qwen3.8-27B-GGUF:Q4_K_XL\ntop-k = 20\nmodel = /x\n[other]\ntemp = 2\n"
    )
    expected = {"mmap": "1", "temp": "0.5", "top-k": "20"}
    assert parse_preset_ini(text, "Qwen3.8-27B-GGUF") == expected
    assert parse_preset_ini(text, "zzz") == expected  # first section
    assert parse_preset_ini("garbage", "x") == {}


def test_profiles_are_matched_by_name_patterns_in_order():
    table = load_profiles()
    assert profile_for("unsloth/Qwen3.8-27B-GGUF", table)[0] == "qwen3.8"
    assert profile_for("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", table)[0] == "qwen3-instruct"
    assert profile_for("unsloth/Qwen3.6-35B-A3B-GGUF", table)[0] == "qwen3-thinking"
    assert profile_for("Qwen/Qwen2.5-1.5B-Instruct-GGUF", table)[0] == "qwen2.5-small"
    assert profile_for("unsloth/gemma-4-31B-it-GGUF", table)[0] == "gemma"
    assert profile_for("ggml-org/gpt-oss-20b-GGUF", table)[0] == "gpt-oss"
    assert profile_for("mistralai/Devstral-Small-2-GGUF", table)[0] == "mistral"
    assert profile_for("someone/mystery-GGUF", table) is None
    for name, profile in table["profiles"].items():
        assert profile["source"].startswith("http"), name


def test_values_for_precedence():
    repo = "unsloth/Qwen3.8-27B-GGUF"
    assert values_for(repo, no_tuning=True) == SamplingResult({}, "none")
    from_preset = values_for(repo, preset_ini="[*]\ntemp = 0.9\n", card=CARD)
    assert from_preset.source == "preset.ini in the repo" and from_preset.values == {"temp": "0.9"}
    from_card = values_for(repo, preset_ini=None, card=CARD)
    assert from_card.source == "model card" and from_card.values["temp"] == "0.6"
    from_profile = values_for(repo, card="nothing useful")
    assert from_profile.source == "profile qwen3.8"
    assert from_profile.values["reasoning-effort"] == "medium"
    assert values_for("someone/mystery-GGUF", card="") == SamplingResult({}, "none")
