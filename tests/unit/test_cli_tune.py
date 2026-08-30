from local_llm import cli
from local_llm.estimate import GIB
from local_llm.hardware import Machine
from local_llm.preset import Preset
from local_llm.router import Router, RouterError
from local_llm.tuning import Measurement

from .test_gguf import QWEN38, write_gguf


def _fake_engine(results, seen=None):
    def engine(model_path, profile, candidates, context, repetitions):
        if seen is not None:
            seen.append((list(candidates), repetitions))
        return [
            Measurement(c, results[i][0], results[i][1], repetitions)
            for i, c in enumerate(candidates)
        ]

    return engine


def _flat(rate=400.0):
    # Every candidate identical, so the baseline wins.
    return [(rate, 30.0)] * 8


def _second_is_fastest():
    return [(400.0, 30.0), (900.0, 30.0)] + [(300.0, 25.0)] * 6


def _second_is_barely_ahead():
    # 18.8 seconds against 18.5: a win of about one percent, which is less than
    # the run-to-run noise the design document recorded on real hardware.
    return [(400.0, 30.0), (410.0, 30.0)] + [(300.0, 25.0)] * 6


def running_router(h):
    h.backend.add(4242, ["llama-server", "--models-preset", str(h.paths.preset)],
                  listening={5678})
    h.paths.state_dir.mkdir(parents=True, exist_ok=True)
    h.paths.pid_file.write_text("4242")


def real_header_section(h, name, extra="", machine_ram_gib=64, reserve_gb=8):
    """A section whose model file has a header `read_header` can actually parse.

    The `harness` fixture's own models are a few hundred bytes of filler, which
    is all most commands need. Everything about fit — the refusal, and dropping
    a candidate that could not be adopted — reads the real key-value header for
    the cache geometry, so those tests need a real one, and a machine whose
    budget is a known number rather than whatever the developer's laptop has.
    """
    path = h.tmp / f"{name}.gguf"
    write_gguf(path, QWEN38)
    h.paths.preset.write_text(
        h.paths.preset.read_text() + f"\n[{name}]\nmodel = {path}\n{extra}"
    )
    machine = Machine("Darwin", "arm64", "Apple M4 Pro", machine_ram_gib * GIB, "apple",
                      machine_ram_gib * GIB, 20, reserve_gb)
    h.monkeypatch.setattr(cli, "_machine", lambda st: machine)
    return path


def test_tune_refuses_an_unknown_model(harness):
    result = harness.run("tune", "nope")
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_tune_reports_a_table_and_names_the_winner(harness):
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small")
    assert result.exit_code == 0
    assert "baseline" in result.output
    assert "batch 1024/256" in result.output
    assert "fastest" in result.output
    assert "Combinations of two changed settings were not tried." in result.output
    assert "worth measuring again" in result.output


def test_tune_offers_nothing_when_the_baseline_wins(harness):
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "small")
    assert result.exit_code == 0
    assert "already the fastest" in result.output
    assert Preset.load(harness.paths.preset).get("small", "b") is None


def test_a_win_too_small_to_reproduce_is_not_offered(harness):
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_second_is_barely_ahead()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 0
    assert "within measurement noise" in result.output
    # Even with --yes, which normally means "do not ask, just write it".
    assert Preset.load(harness.paths.preset).get("small", "b") is None


def test_tune_writes_only_the_keys_that_change(harness):
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 0
    preset = Preset.load(harness.paths.preset)
    assert preset.get("small", "b") == "1024"
    assert preset.get("small", "ub") == "256"
    # The model path the section already had is still there.
    assert preset.get("small", "model", fallback_to_star=False).endswith("small.gguf")
    # cache-type was never asked about by the winning candidate (only batch/ubatch
    # changed), so it must not be pinned in the file at llama.cpp's own default.
    assert preset.get("small", "cache-type-k", fallback_to_star=False) is None
    assert preset.get("small", "cache-type-v", fallback_to_star=False) is None


def test_a_second_run_replaces_the_note_rather_than_adding_one(harness):
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_second_is_fastest()))
    assert harness.run("tune", "small", "--yes").exit_code == 0
    assert harness.run("tune", "small", "--yes").exit_code == 0
    text = harness.paths.preset.read_text()
    assert text.count("tuned by local-llm") == 1


def test_the_repetitions_asked_for_reach_the_engine(harness):
    seen = []
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat(), seen))
    assert harness.run("tune", "small", "--repetitions", "7").exit_code == 0
    assert seen[0][1] == 7
    assert "7 runs each" in harness.run("tune", "small", "--repetitions", "7").output


def test_tune_leaves_the_file_alone_without_yes_when_not_interactive(harness):
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small")
    assert Preset.load(harness.paths.preset).get("small", "b") is None
    assert "--yes" in result.output


def test_tune_json_carries_every_candidate(harness):
    import json

    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small", "--json")
    payload = json.loads(result.output)
    assert len(payload) >= 5
    assert payload[0]["label"] == "batch 1024/256"
    assert "turn_seconds" in payload[0]
    assert "prompt_stddev" in payload[0] and "generation_stddev" in payload[0]


def test_tune_refuses_a_model_too_big_to_measure(harness):
    # 8.25 GB of RAM with 8 reserved leaves a quarter of a gigabyte usable, and
    # this model's key-value cache alone wants half of one at the measured depth.
    real_header_section(harness, "huge", machine_ram_gib=8, reserve_gb=8)
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "huge")
    assert result.exit_code == 1
    assert "to measure and this machine has" in result.output
    assert "local-llm doctor" in result.output


def test_a_cache_change_that_would_not_fit_at_the_serving_context_is_never_measured(harness):
    # The section serves at 65536 tokens. With a q8_0 cache that comes to 3.1 GB
    # all told; the same context at f16 comes to 5.0 GB, and this machine has
    # 4.0 GB usable - so `cache f16` is dropped before it is measured, rather
    # than measured, printed, and then found to be unusable.
    seen = []
    real_header_section(harness, "big-ctx", extra="c = 65536\ncache-type-k = q8_0\n"
                        "cache-type-v = q8_0\n", machine_ram_gib=8, reserve_gb=4)
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat(), seen))
    result = harness.run("tune", "big-ctx")
    assert result.exit_code == 0
    assert "not measuring cache f16" in result.output
    assert "65536-token context" in result.output
    labels = [c.label for c in seen[0][0]]
    assert "cache f16" not in labels
    assert "baseline" in labels and "flash-attn off" in labels
    assert "4 settings" in result.output
    # The header naming how many settings will be measured comes before the
    # sub-line explaining which one was dropped and why, not after it.
    assert result.output.index("4 settings") < result.output.index("not measuring cache f16")


def test_when_nothing_else_fits_the_message_says_so(harness):
    # Budget sits between what a q8_0 cache needs at this context (3.1 GB) and
    # what an f16 one needs (5.0 GB), so every non-baseline candidate is
    # dropped: the batch/ubatch and flash-attn variants share the baseline's
    # q8_0 cache type and so share its estimate, and the cache variant would
    # switch to f16. Only the baseline is left to measure.
    real_header_section(harness, "big-ctx", extra="c = 65536\ncache-type-k = q8_0\n"
                        "cache-type-v = q8_0\n", machine_ram_gib=8, reserve_gb=6)
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "big-ctx")
    assert result.exit_code == 0
    assert "No alternative setting could be measured" in result.output
    assert "already the fastest" not in result.output
    assert Preset.load(harness.paths.preset).get("big-ctx", "b") is None


def test_a_projector_that_has_moved_is_a_message_not_a_traceback(harness):
    real_header_section(harness, "vision", extra=f"mmproj = {harness.tmp}/gone.gguf\n")
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "vision")
    assert result.exit_code == 1
    assert "mmproj file is missing" in result.output
    assert "gone.gguf" in result.output


def test_tune_unloads_a_resident_model_and_puts_it_back(harness):
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 0
    assert ("POST", "/models/unload", {"model": "big"}) in harness.http.calls
    assert ("POST", "/models/load", {"model": "big"}) in harness.http.calls
    assert harness.http.calls.index(("POST", "/models/unload", {"model": "big"})) < \
        harness.http.calls.index(("POST", "/models/load", {"model": "big"}))


def test_an_unload_the_router_refused_stops_the_run(harness):
    # The router answers with an error *status*, which Router turns into an
    # ordinary reply body rather than an exception. Read as success, it would
    # start the benchmark with 20 GB of somebody else's model still resident.
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)
    harness.http.responses[("POST", "/models/unload")] = {
        "error": {"code": 500, "message": "model is busy"}
    }
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 1
    assert "Could not unload big" in result.output
    assert "model is busy" in result.output
    assert ("POST", "/models/load", {"model": "big"}) not in harness.http.calls


def test_a_model_still_resident_after_a_successful_unload_stops_the_run(harness):
    # The router says it unloaded and the process is still there. The reply is
    # the router's opinion; the process table is the fact, and the fact wins.
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)

    def deaf(method, path, body):
        harness.http.calls.append((method, path, body))
        return {"success": True}

    harness.monkeypatch.setattr(
        cli, "Router",
        lambda p, s, **kwargs: Router(p, s, backend=harness.backend, http=deaf,
                                      binary="/opt/bin/llama-server", help_text=lambda b: "",
                                      sleep=lambda seconds: None, log=harness.messages.append),
    )
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 1
    assert "still holding big" in result.output
    # It was asked to unload, so it is put back rather than left half-managed.
    assert ("POST", "/models/load", {"model": "big"}) in harness.http.calls


def test_a_reload_the_router_refused_says_so(harness):
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)
    harness.http.responses[("POST", "/models/load")] = {
        "error": {"message": "no memory left"}
    }
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert "big could not be reloaded" in result.output
    assert "no memory left" in result.output


def test_a_resident_model_is_reloaded_even_when_the_engine_explodes(harness):
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)

    def exploding(model_path, profile, candidates, context, repetitions):
        raise RuntimeError("boom")

    harness.monkeypatch.setattr(cli, "_measure", exploding)
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code != 0
    assert ("POST", "/models/load", {"model": "big"}) in harness.http.calls


def test_a_model_already_unloaded_is_still_reloaded_when_a_later_unload_fails(harness):
    # Two models resident. The second one refuses to unload. The first must still
    # come back - it was already put down before the router turned unhealthy.
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "first"], rss=5 * 1024**3, parent=4242)
    harness.backend.add(4244, ["llama-server", "--alias", "second"], rss=5 * 1024**3, parent=4242)

    def flaky(method, path, body):
        harness.http.calls.append((method, path, body))
        if (method, path, body) == ("POST", "/models/unload", {"model": "second"}):
            raise RouterError("router says no")
        return harness.http.responses.get((method, path), {"success": True})

    def make_flaky_router(p, s, **kwargs):
        return Router(p, s, backend=harness.backend, http=flaky,
                      binary="/opt/bin/llama-server", help_text=lambda binary: "",
                      sleep=lambda seconds: None, log=harness.messages.append)

    harness.monkeypatch.setattr(cli, "Router", make_flaky_router)
    harness.monkeypatch.setattr(cli, "_measure", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code != 0
    assert ("POST", "/models/unload", {"model": "first"}) in harness.http.calls
    assert ("POST", "/models/load", {"model": "first"}) in harness.http.calls
    assert ("POST", "/models/load", {"model": "second"}) not in harness.http.calls
