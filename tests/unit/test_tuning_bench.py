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
    prompt_rate, generation_rate = bench.parse(FIXTURE.read_text())
    rows = json.loads(FIXTURE.read_text())
    expected_pp = [r["avg_ts"] for r in rows if r["n_prompt"] and not r["n_gen"]][0]
    expected_tg = [r["avg_ts"] for r in rows if r["n_gen"] and not r["n_prompt"]][0]
    assert prompt_rate == expected_pp
    assert generation_rate == expected_tg


def test_a_document_missing_a_row_is_an_error_not_a_zero():
    from local_llm.tuning import TuningError

    try:
        bench.parse('[{"n_prompt": 4096, "n_gen": 0, "avg_ts": 500.0}]')
    except TuningError as error:
        assert "generation" in str(error)
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


def test_a_missing_binary_is_refused_before_anything_runs():
    from local_llm.tuning import TuningError

    context = TuningContext(bench_binary=None, run=lambda *a, **k: None)
    try:
        bench.measure(Path("/m.gguf"), AGENT, [BASE], context, 3)
    except TuningError as error:
        assert "llama-bench" in str(error)
    else:
        raise AssertionError("expected TuningError")
