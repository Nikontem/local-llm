from local_llm.tuning import Candidate, Measurement, Profile, turn_time

AGENT = Profile(depth=4096, prompt=4096, generate=256)
BASE = Candidate(batch=2048, ubatch=512, flash_attn="auto", cache_type="f16", label="baseline")


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
