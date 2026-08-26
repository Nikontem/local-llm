from local_llm.preset import Preset

from .test_gguf import QWEN38, write_gguf

REPO = "unsloth/Qwen3.8-27B-GGUF"


def test_pull_with_suggestion_writes_a_tuned_section(hubbed):
    h = hubbed
    h.served[(REPO, "README.md")] = "We suggest temperature=0.6 and top_p=0.95."
    result = h.run("pull", REPO, "--yes")
    assert result.exit_code == 0, result.output
    assert "UD-Q4_K_XL" in result.output
    assert "downloading Qwen3.8-27B-UD-Q4_K_XL.gguf" in result.output
    assert "downloading mmproj-F16.gguf" in result.output
    assert f"Added [{REPO}:Q4_K_XL]" in result.output
    preset = Preset.load(h.paths.preset)
    section = f"{REPO}:Q4_K_XL"
    assert preset.has_section(section)
    keys = preset.items(section)
    assert keys["model"].endswith("Qwen3.8-27B-UD-Q4_K_XL.gguf")
    assert keys["mmproj"].endswith("mmproj-F16.gguf")
    assert keys["c"] == "131072" and keys["n-predict"] == "32768"
    assert keys["reasoning-format"] == "deepseek"
    assert keys["temp"] == "0.6" and keys["top-p"] == "0.95"
    assert keys["cache-type-k"] == "q8_0"
    text = h.paths.preset.read_text()
    assert "# sampling: model card" in text
    assert "# context: 131072 suggested for this machine" in text
    assert "local-llm up" in result.output  # router not running


def test_pull_explicit_quant_context_set_and_no_tuning(hubbed):
    h = hubbed
    result = h.run(
        "pull", f"{REPO}:Q8_0", "--context", "65536", "--set", "temp=0.2",
        "--set", "top-k=5", "--no-tuning", "--yes",
    )
    assert result.exit_code == 0, result.output
    keys = Preset.load(h.paths.preset).items(f"{REPO}:Q8_0")
    assert keys["model"].endswith("Qwen3.8-27B-Q8_0.gguf") and keys["c"] == "65536"
    assert keys["temp"] == "0.2" and keys["top-k"] == "5"
    assert "reasoning-format" not in keys and "cache-type-k" not in keys


def test_pull_rejects_unknown_quant_bad_repo_and_duplicates(hubbed):
    h = hubbed
    bad = h.run("pull", REPO, "--quant", "IQ2_M")
    assert bad.exit_code == 1 and "available: Q8_0, UD-Q4_K_XL" in bad.output
    missing = h.run("pull", "nobody/nothing-GGUF", "--yes")
    assert missing.exit_code == 1 and "No such repository" in missing.output
    malformed = h.run("pull", "just-a-name")
    assert malformed.exit_code == 1 and "org/repo" in malformed.output
    assert h.run("pull", f"{REPO}:Q8_0", "--yes").exit_code == 0
    again = h.run("pull", f"{REPO}:Q8_0", "--yes")
    assert again.exit_code == 1 and "already exists" in again.output
    assert "local-llm remove" in again.output
    bad_set = h.run("pull", REPO, "--set", "novalue", "--yes")
    assert bad_set.exit_code == 1 and "KEY=VALUE" in bad_set.output


def test_pull_refuses_without_disk_space(hubbed):
    h = hubbed
    from local_llm import cli
    h.monkeypatch.setattr(cli, "free_disk_bytes", lambda path: 10)
    result = h.run("pull", REPO, "--yes")
    assert result.exit_code == 1 and "Not enough disk space" in result.output


def test_pull_reloads_a_running_router(hubbed):
    h = hubbed
    h.backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    result = h.run("pull", REPO, "--yes")
    assert result.exit_code == 0, result.output
    assert ("GET", "/models?reload=1", None) in h.http.calls
    assert "reloaded its model list" in result.output


def test_add_local_file_and_remove_with_files(hubbed):
    h = hubbed
    model = h.tmp / "My-Model-UD-Q4_K_XL.gguf"
    write_gguf(model, QWEN38)
    blob = h.tmp / "blob"
    blob.write_bytes(b"x" * 10)
    link = h.tmp / "linked-Q4_0.gguf"
    link.symlink_to(blob)
    added = h.run("add", str(model), "--name", "mine", "--context", "8192")
    assert added.exit_code == 0, added.output
    keys = Preset.load(h.paths.preset).items("mine")
    assert keys["model"] == str(model.resolve())
    assert keys["c"] == "8192" and keys["n-predict"] == "4096"
    assert h.run("add", str(link)).exit_code == 0
    assert Preset.load(h.paths.preset).has_section("linked-q4_0")
    removed = h.run("remove", "linked-q4_0", "--delete-files", "--yes")
    assert removed.exit_code == 0, removed.output
    assert not link.exists() and not blob.exists()
    assert not Preset.load(h.paths.preset).has_section("linked-q4_0")
    assert h.run("remove", "nope").exit_code == 1


def test_status_uses_router_model_status(harness):
    h = harness
    h.backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    h.backend.add(43, ["/opt/bin/llama-server", "--alias", "big"], rss=100 * 1024**2, parent=42)
    h.paths.ensure_state_dirs()
    h.paths.pid_file.write_text("42\n")
    loading = {"data": [{"id": "big", "status": {"value": "loading"}}]}
    h.http.responses[("GET", "/models")] = loading
    result = h.run("status")
    assert "big" in result.output and "(loading)" in result.output
    assert "(asleep)" not in result.output
    sleeping = {"data": [{"id": "big", "status": {"value": "sleeping"}}]}
    h.http.responses[("GET", "/models")] = sleeping
    assert "(asleep)" in h.run("status").output


def test_load_uses_the_kv_cache_when_the_header_is_readable(harness):
    h = harness
    from local_llm import cli
    from local_llm.estimate import GIB
    h.backend.add(42, ["/opt/bin/llama-server", "--port", "5678"], listening={5678})
    h.paths.ensure_state_dirs()
    h.paths.pid_file.write_text("42\n")
    h.monkeypatch.setattr(cli, "total_ram", lambda: 48 * GIB)
    write_gguf(h.tmp / "big.gguf", QWEN38)
    h.paths.preset.write_text(f"[big]\nmodel = {h.tmp}/big.gguf\nc = 65536\ncache-type-k = q8_0\n")
    result = h.run("load", "big")
    assert result.exit_code == 0, result.output
    assert "weights + KV cache at c=65536" in result.output


def test_remove_while_running_does_not_suggest_loading_it(hubbed):
    h = hubbed
    assert h.run("pull", f"{REPO}:Q8_0", "--yes").exit_code == 0
    h.backend.add(42, ["/opt/bin/llama-server"], listening={5678})
    result = h.run("remove", f"{REPO}:Q8_0", "--yes")
    assert result.exit_code == 0, result.output
    assert "reloaded its model list" in result.output and "local-llm load" not in result.output
