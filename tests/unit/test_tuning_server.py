import json
from pathlib import Path

from local_llm.tuning import Candidate, Profile, TuningContext, TuningError, server

from .fakes import FakeBackend

FIXTURE = Path(__file__).parent / "fixtures" / "llama_server_timings.json"
AGENT = Profile(depth=4096, prompt=4096, generate=256)
BASE = Candidate(batch=2048, ubatch=512, flash_attn="auto", cache_type="f16", label="baseline")


def test_the_server_is_started_with_the_candidate_and_room_for_the_profile():
    args = server.server_args("/opt/bin/llama-server", Path("/m.gguf"), AGENT, BASE, 5699)
    joined = " ".join(args)
    assert "--port 5699" in joined
    assert "-m /m.gguf" in joined
    assert "-b 2048" in joined and "-ub 512" in joined
    assert "-fa auto" in joined
    assert "-ctk f16" in joined and "-ctv f16" in joined
    assert "-ngl all" in joined
    assert "--no-ui" in joined
    # The context must hold depth + prompt + generation with room to spare, or the
    # measured request is truncated and the rate is measured on the wrong work.
    size = int(args[args.index("-c") + 1])
    assert size > AGENT.total


def test_rates_come_from_the_response_not_from_the_request():
    response = json.loads(FIXTURE.read_text())
    prompt_rate, generation_rate = server.rates(response)
    timings = response["timings"]
    assert prompt_rate == timings["prompt_n"] / (timings["prompt_ms"] / 1000)
    assert generation_rate == timings["predicted_n"] / (timings["predicted_ms"] / 1000)


def test_a_response_without_timings_is_an_error():
    try:
        server.rates({"content": "hello"})
    except TuningError as error:
        assert "timings" in str(error)
    else:
        raise AssertionError("expected TuningError")


def _context(backend, posts):
    def post(url, body):
        posts.append((url, body))
        return json.loads(FIXTURE.read_text())

    return TuningContext(
        server_binary="/opt/bin/llama-server",
        backend=backend,
        post=post,
        port_in_use=lambda host, port: False,
        sleep=lambda seconds: None,
    )


def test_each_candidate_gets_its_own_server_which_is_stopped_afterwards():
    backend = FakeBackend()
    backend.spawn_listening = {5699}
    posts = []
    context = _context(backend, posts)
    candidates = [BASE, Candidate(4096, 1024, "auto", "f16", "batch 4096/1024")]
    results = server.measure(Path("/m.gguf"), AGENT, candidates, context, 2)
    assert len(results) == 2 and all(m.ok for m in results)
    assert len(backend.spawned) == 2
    # One priming request plus two measured requests, per candidate.
    assert len(posts) == 6
    assert posts[0][1]["n_predict"] == 1
    assert posts[1][1]["n_predict"] == AGENT.generate
    assert all(body["cache_prompt"] for _, body in posts)
    # Every server started was also stopped.
    assert len(backend.terminated) == 2


def test_a_server_that_never_comes_up_is_recorded_and_the_run_continues():
    backend = FakeBackend()
    backend.spawn_dies = True
    context = _context(backend, [])
    results = server.measure(Path("/m.gguf"), AGENT, [BASE], context, 2)
    assert len(results) == 1
    assert not results[0].ok
    assert "did not start" in results[0].error


def test_a_missing_binary_is_refused_before_anything_runs():
    context = TuningContext(server_binary=None, backend=FakeBackend())
    try:
        server.measure(Path("/m.gguf"), AGENT, [BASE], context, 2)
    except TuningError as error:
        assert "llama-server" in str(error)
    else:
        raise AssertionError("expected TuningError")
