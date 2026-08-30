import json
import subprocess
from pathlib import Path

from local_llm.tuning import Candidate, Profile, TuningContext, bench

FIXTURE = Path(__file__).parent / "fixtures" / "llama_bench.json"
AGENT = Profile(depth=4096, prompt=4096, generate=256)
BASE = Candidate(batch=2048, ubatch=512, flash_attn="auto", cache_type="f16", label="baseline")


def test_the_arguments_ask_for_both_tests_at_depth():
    args = bench.bench_args("/opt/bin/llama-bench", Path("/m.gguf"), AGENT, BASE, 3)
    assert args[0] == "/opt/bin/llama-bench"
    joined = " ".join(args)
    assert "-m /m.gguf" in joined
    assert "-o json" in joined
    assert "-r 3" in joined
    assert "-d 4096" in joined and "-p 4096" in joined and "-n 256" in joined
    assert "-b 2048" in joined and "-ub 512" in joined
    assert "-fa auto" in joined
    assert "-ctk f16" in joined and "-ctv f16" in joined
    # No -ngl: llama-bench's default offloads every layer, which is what the
    # router pins with n-gpu-layers = all. Task 1 confirmed this.
    assert "-ngl" not in joined


def test_parsing_a_real_llama_bench_document():
    # The literals, not the same predicate the implementation uses to find its
    # rows: recomputing the expectation with the code under test would make a
    # filter that picks the wrong row agree with itself.
    prompt_rate, prompt_stddev, generation_rate, generation_stddev = bench.parse(
        FIXTURE.read_text()
    )
    assert prompt_rate == 1563.461981
    assert prompt_stddev == 0.408924
    assert generation_rate == 151.90858
    assert generation_stddev == 2.785915


def test_a_spread_the_document_does_not_report_becomes_zero():
    # One repetition, or a build old enough not to report stddev_ts at all.
    document = json.dumps([
        {"n_prompt": 4096, "n_gen": 0, "avg_ts": 500.0},
        {"n_prompt": 0, "n_gen": 256, "avg_ts": 40.0},
    ])
    assert bench.parse(document) == (500.0, 0.0, 40.0, 0.0)


def test_a_document_missing_a_row_is_an_error_not_a_zero():
    from local_llm.tuning import TuningError

    try:
        bench.parse('[{"n_prompt": 4096, "n_gen": 0, "avg_ts": 500.0}]')
    except TuningError as error:
        assert "generation" in str(error)
    else:
        raise AssertionError("expected TuningError")


def test_a_document_missing_the_prompt_row_is_an_error_too():
    from local_llm.tuning import TuningError

    try:
        bench.parse('[{"n_prompt": 0, "n_gen": 256, "avg_ts": 40.0}]')
    except TuningError as error:
        assert "prompt-processing" in str(error)
    else:
        raise AssertionError("expected TuningError")


def test_output_that_is_not_json_says_so_rather_than_raising_a_json_error():
    from local_llm.tuning import TuningError

    try:
        bench.parse("ggml_metal_init: loaded kernels\n")
    except TuningError as error:
        assert "did not return JSON" in str(error)
    else:
        raise AssertionError("expected TuningError")


def test_measure_runs_once_per_candidate_and_records_a_failure_without_stopping():
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[args.index("-b") + 1] == "4096":
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="out of memory")
        return subprocess.CompletedProcess(args, 0, stdout=FIXTURE.read_text(), stderr="")

    context = TuningContext(bench_binary="/opt/bin/llama-bench", run=fake_run)
    others = [
        BASE,
        Candidate(4096, 1024, "auto", "f16", "batch 4096/1024"),
        Candidate(2048, 512, "off", "f16", "flash-attn off"),
    ]
    results = bench.measure(Path("/m.gguf"), AGENT, others, context, 3)
    assert len(results) == 3
    assert len(calls) == 3
    assert results[0].ok and results[2].ok
    assert not results[1].ok
    assert "out of memory" in results[1].error
    # The spread llama-bench reported comes through with the rates.
    assert results[0].generation_stddev == 2.785915


def test_a_candidate_that_never_comes_back_is_recorded_and_the_run_continues():
    # TimeoutExpired is a SubprocessError and not an OSError. Uncaught, the
    # slowest candidate on the slowest machine would take every measurement
    # already paid for down with it.
    def fake_run(args, **kwargs):
        if args[args.index("-b") + 1] == "4096":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout=FIXTURE.read_text(), stderr="")

    context = TuningContext(bench_binary="/opt/bin/llama-bench", run=fake_run)
    candidates = [
        BASE,
        Candidate(4096, 1024, "auto", "f16", "batch 4096/1024"),
        Candidate(2048, 512, "off", "f16", "flash-attn off"),
    ]
    results = bench.measure(Path("/m.gguf"), AGENT, candidates, context, 3)
    assert [m.ok for m in results] == [True, False, True]
    assert "gave up after 30 minutes" in results[1].error


def test_a_missing_binary_is_refused_before_anything_runs():
    from local_llm.tuning import TuningError

    context = TuningContext(bench_binary=None, run=lambda *a, **k: None)
    try:
        bench.measure(Path("/m.gguf"), AGENT, [BASE], context, 3)
    except TuningError as error:
        assert "llama-bench" in str(error)
    else:
        raise AssertionError("expected TuningError")
