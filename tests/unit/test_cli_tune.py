from local_llm import cli
from local_llm.preset import Preset
from local_llm.router import Router, RouterError
from local_llm.tuning import Measurement


def _fake_engine(results):
    def engine(model_path, profile, candidates, context, repetitions):
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


def running_router(h):
    h.backend.add(4242, ["llama-server", "--models-preset", str(h.paths.preset)],
                  listening={5678})
    h.paths.state_dir.mkdir(parents=True, exist_ok=True)
    h.paths.pid_file.write_text("4242")


def test_tune_refuses_an_unknown_model(harness):
    result = harness.run("tune", "nope")
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_tune_reports_a_table_and_names_the_winner(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small")
    assert result.exit_code == 0
    assert "baseline" in result.output
    assert "batch 1024/256" in result.output
    assert "fastest" in result.output
    assert "Combinations of two changed settings were not tried." in result.output


def test_tune_offers_nothing_when_the_baseline_wins(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_flat()))
    result = harness.run("tune", "small")
    assert result.exit_code == 0
    assert "already the fastest" in result.output
    assert Preset.load(harness.paths.preset).get("small", "b") is None


def test_tune_writes_only_the_keys_that_change(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
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


def test_tune_leaves_the_file_alone_without_yes_when_not_interactive(harness):
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small")
    assert Preset.load(harness.paths.preset).get("small", "b") is None
    assert "--yes" in result.output


def test_tune_json_carries_every_candidate(harness):
    import json

    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_second_is_fastest()))
    result = harness.run("tune", "small", "--json")
    payload = json.loads(result.output)
    assert len(payload) >= 5
    assert payload[0]["label"] == "batch 1024/256"
    assert "turn_seconds" in payload[0]


def test_tune_unloads_a_resident_model_and_puts_it_back(harness):
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code == 0
    assert ("POST", "/models/unload", {"model": "big"}) in harness.http.calls
    assert ("POST", "/models/load", {"model": "big"}) in harness.http.calls
    assert harness.http.calls.index(("POST", "/models/unload", {"model": "big"})) < \
        harness.http.calls.index(("POST", "/models/load", {"model": "big"}))


def test_a_resident_model_is_reloaded_even_when_the_engine_explodes(harness):
    running_router(harness)
    harness.backend.add(4243, ["llama-server", "--alias", "big"], rss=20 * 1024**3, parent=4242)

    def exploding(model_path, profile, candidates, context, repetitions):
        raise RuntimeError("boom")

    harness.monkeypatch.setitem(cli._ENGINES, "bench", exploding)
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
    harness.monkeypatch.setitem(cli._ENGINES, "bench", _fake_engine(_flat()))
    result = harness.run("tune", "small", "--yes")
    assert result.exit_code != 0
    assert ("POST", "/models/unload", {"model": "first"}) in harness.http.calls
    assert ("POST", "/models/load", {"model": "first"}) in harness.http.calls
    assert ("POST", "/models/load", {"model": "second"}) not in harness.http.calls
