from local_llm.preset import Preset
from local_llm.tuning import BASELINE, Candidate, Measurement, Profile, turn_time
from local_llm.tuning import report as report_module

AGENT = Profile(depth=4096, prompt=4096, generate=256)
BASE = Candidate(batch=2048, ubatch=512, flash_attn="auto", cache_type="f16", label="baseline")
FAST = Candidate(batch=4096, ubatch=1024, flash_attn="auto", cache_type="f16",
                 label="batch 4096/1024")


def _measured(candidate, prompt_rate, generation_rate):
    return Measurement(candidate, prompt_rate, generation_rate, repetitions=3)


def test_turn_time_is_prompt_plus_generation():
    # 4096 tokens in at 512/s is 8 seconds; 256 tokens out at 32/s is another 8.
    measurement = Measurement(BASE, prompt_rate=512.0, generation_rate=32.0, repetitions=3)
    assert turn_time(measurement, AGENT) == 16.0


def test_a_failed_measurement_sorts_last():
    failed = Measurement(BASE, prompt_rate=0.0, generation_rate=0.0, repetitions=0,
                         error="did not start")
    assert turn_time(failed, AGENT) == float("inf")
    assert not failed.ok


def test_a_candidate_names_the_preset_keys_it_implies():
    candidate = Candidate(batch=4096, ubatch=1024, flash_attn="off", cache_type="q8_0",
                          label="cache q8_0")
    assert candidate.preset_keys() == [
        ("b", "4096"),
        ("ub", "1024"),
        ("flash-attn", "off"),
        ("cache-type-k", "q8_0"),
        ("cache-type-v", "q8_0"),
    ]


def test_a_section_reports_every_comment_it_owns():
    preset = Preset.parse(
        "# added by local-llm\n"
        "# context chosen at load time\n"
        "[m]\n"
        "model = /tmp/m.gguf\n"
        "# hand-written note from the user\n"
        "b = 2048\n"
    )
    assert preset.comments("m") == [
        "added by local-llm",
        "context chosen at load time",
        "hand-written note from the user",
    ]


def test_a_hand_written_comment_survives_a_rewrite_by_moving_above_the_header():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\n# keep me\nb = 2048\n")
    preset.replace_section("m", [("model", "/tmp/m.gguf"), ("b", "4096")], preset.comments("m"))
    text = preset.dump()
    assert "# keep me" in text
    assert text.index("# keep me") < text.index("[m]")


def test_ranking_puts_the_shortest_turn_first_and_failures_last():
    slow = _measured(BASE, 400.0, 30.0)
    fast = _measured(FAST, 800.0, 30.0)
    broken = Measurement(FAST, 0.0, 0.0, repetitions=0, error="did not start")
    ordered = report_module.rank([slow, broken, fast], AGENT)
    assert [m.candidate.label for m in ordered] == [FAST.label, BASELINE, FAST.label]
    assert ordered[-1].error == "did not start"
    assert report_module.winner([slow, broken, fast], AGENT).candidate is FAST


def test_there_is_no_winner_when_everything_failed():
    broken = Measurement(BASE, 0.0, 0.0, repetitions=0, error="did not start")
    assert report_module.winner([broken], AGENT) is None


def test_the_table_names_the_baseline_and_the_change_against_it():
    rows = report_module.table([_measured(BASE, 400.0, 30.0), _measured(FAST, 800.0, 30.0)], AGENT)
    assert any(BASELINE in row and "b=2048" in row for row in rows)
    faster = [row for row in rows if FAST.label in row][0]
    assert "-" in faster and "%" in faster


def test_a_change_under_half_a_percent_is_not_printed_as_minus_zero():
    # 4096/402 + 256/30 against 4096/400 + 256/30 is a win of a third of a
    # percent. Rounded to whole percent it renders as "-0%", which reads as a
    # bug in the tool rather than as the tiny difference it is.
    rows = report_module.table([_measured(BASE, 400.0, 30.0), _measured(FAST, 402.0, 30.0)], AGENT)
    faster = [row for row in rows if FAST.label in row][0]
    assert "-0%" not in faster
    assert "-0.3%" in faster


def test_the_table_shows_the_spread_around_the_generation_rate():
    steady = Measurement(BASE, 400.0, 30.0, repetitions=3, generation_stddev=0.4)
    jumpy = Measurement(FAST, 800.0, 30.0, repetitions=3, generation_stddev=6.2)
    rows = report_module.table([steady, jumpy], AGENT)
    assert any("±0.4" in row for row in rows)
    assert any("±6.2" in row for row in rows)


def test_a_measurement_with_no_spread_to_report_prints_none():
    rows = report_module.table([_measured(BASE, 400.0, 30.0)], AGENT)
    assert "±" not in rows[0]


def test_the_margin_is_measured_against_the_baseline():
    base = _measured(BASE, 400.0, 30.0)
    fast = _measured(FAST, 800.0, 30.0)
    gain = report_module.improvement(fast, [base, fast], AGENT)
    base_seconds = 4096 / 400.0 + 256 / 30.0
    fast_seconds = 4096 / 800.0 + 256 / 30.0
    assert round(gain, 6) == round((base_seconds - fast_seconds) / base_seconds, 6)
    assert gain > report_module.MEANINGFUL_MARGIN


def test_a_win_of_one_percent_is_under_the_margin():
    base = _measured(BASE, 400.0, 30.0)
    barely = _measured(FAST, 410.0, 30.0)
    assert 0 < report_module.improvement(barely, [base, barely], AGENT) \
        < report_module.MEANINGFUL_MARGIN


def test_an_unmeasurable_baseline_gives_no_margin_rather_than_a_win():
    # None and zero must not be confused: with no baseline there is nothing to
    # compare against, which is not the same as "the winner tied with it".
    broken = Measurement(BASE, 0.0, 0.0, repetitions=0, error="did not start")
    fast = _measured(FAST, 800.0, 30.0)
    assert report_module.improvement(fast, [broken, fast], AGENT) is None


def test_a_failed_row_says_so_instead_of_printing_a_rate():
    broken = Measurement(FAST, 0.0, 0.0, repetitions=0, error="did not start")
    rows = report_module.table([_measured(BASE, 400.0, 30.0), broken], AGENT)
    assert any("did not start" in row for row in rows)


def test_only_keys_that_differ_from_the_effective_value_are_written():
    # flash-attn = auto comes from [*]; writing it into the section would add a
    # line that changes nothing. cache-type-k/v are set nowhere at all, so their
    # effective value is llama.cpp's own default ("f16"), which FAST also asks
    # for - so those two must not be written either.
    preset = Preset.parse(
        "[*]\nflash-attn = auto\n\n[m]\nmodel = /tmp/m.gguf\nb = 2048\nub = 512\n"
    )
    changes = report_module.changed_keys(FAST, preset, "m")
    assert changes == [("b", "4096"), ("ub", "1024")]


def test_an_unset_key_is_compared_against_llamacpps_own_default_not_none():
    # The section sets none of the five tuned keys at all, and neither does [*].
    # A candidate that only varies batch/ubatch must not also drag flash-attn and
    # cache-type along for the ride just because they were never written down.
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\n")
    changed = Candidate(batch=1024, ubatch=256, flash_attn="auto", cache_type="f16",
                        label="batch 1024/256")
    assert report_module.changed_keys(changed, preset, "m") == [
        ("b", "1024"), ("ub", "256"),
    ]


def test_a_winner_identical_to_the_file_implies_no_changes():
    preset = Preset.parse(
        "[m]\nmodel = /tmp/m.gguf\nb = 2048\nub = 512\nflash-attn = auto\n"
        "cache-type-k = f16\ncache-type-v = f16\n"
    )
    assert report_module.changed_keys(BASE, preset, "m") == []


def test_merging_keeps_the_keys_the_section_already_had():
    preset = Preset.parse("[m]\nmodel = /tmp/m.gguf\nfit-ctx = 16384\nb = 2048\n")
    merged = report_module.merged_keys(preset, "m", [("b", "4096"), ("ub", "1024")])
    assert merged[0] == ("model", "/tmp/m.gguf")
    assert ("fit-ctx", "16384") in merged
    assert ("b", "4096") in merged and ("b", "2048") not in merged
    assert ("ub", "1024") in merged


def test_the_json_shape_carries_the_rates_and_the_derived_turn():
    payload = report_module.as_json(
        [Measurement(BASE, 400.0, 30.0, repetitions=3,
                     prompt_stddev=1.5, generation_stddev=0.4)],
        AGENT,
    )
    assert payload[0]["label"] == BASELINE
    assert payload[0]["batch"] == 2048
    assert payload[0]["prompt_rate"] == 400.0
    assert payload[0]["generation_rate"] == 30.0
    assert payload[0]["prompt_stddev"] == 1.5
    assert payload[0]["generation_stddev"] == 0.4
    assert round(payload[0]["turn_seconds"], 3) == round(4096 / 400.0 + 256 / 30.0, 3)
