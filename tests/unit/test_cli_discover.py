import json


def test_recommend_prints_machine_and_groups(hubbed):
    h = hubbed
    result = h.run("recommend")
    assert result.exit_code == 0, result.output
    assert "Apple M4 Pro, 20 GPU cores, 48 GB unified memory" in result.output
    for group in ("coding", "general", "small", "vision"):
        assert f"\n{group}\n" in result.output
    assert "Qwen3-Coder-30B-A3B-Instruct" in result.output and "UD-Q4_K_XL" in result.output
    assert "comfortable" in result.output and "vendor" in result.output
    assert "Remix" not in result.output
    assert "local-llm recommend --pick" in result.output


def test_recommend_use_filter_and_finetunes(hubbed):
    h = hubbed
    only_small = h.run("recommend", "--use", "small")
    assert "\nsmall\n" in only_small.output and "\ncoding\n" not in only_small.output
    assert "Q8_0" in only_small.output
    with_remix = h.run("recommend", "--include-finetunes", "--limit", "10")
    assert "Remix" in with_remix.output and "derivative" in with_remix.output
    bad = h.run("recommend", "--use", "audio")
    assert bad.exit_code == 1 and "coding, general, small, vision" in bad.output


def test_recommend_json(hubbed):
    result = hubbed.run("recommend", "--json", "--use", "coding")
    data = json.loads(result.output)
    entry = data["coding"][0]
    assert entry["repo_id"] == "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
    assert entry["suggested"] == "UD-Q4_K_XL" and entry["fit"] == "comfortable"
    assert entry["options"][0]["label"] == "UD-Q4_K_XL" and entry["options"][0]["size"] > 0


def test_search_lists_every_lineage_with_fit_and_pull_command(hubbed):
    result = hubbed.run("search", "Qwen3.8")
    assert result.exit_code == 0, result.output
    assert "unsloth/Qwen3.8-27B-GGUF" in result.output and "7,000,000 downloads" in result.output
    assert "UD-Q4_K_XL" in result.output and "comfortable" in result.output
    assert "Q8_0" in result.output and "fits" in result.output
    assert "local-llm pull unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL" in result.output
    assert "lmstudio-community/Qwen3.8-27B-GGUF" in result.output
    remix = hubbed.run("search", "Remix")
    assert "derivative" in remix.output
    nothing = hubbed.run("search", "zzzz-no-such-model")
    assert nothing.exit_code == 0 and "No GGUF repositories match" in nothing.output


def test_search_author_and_json(hubbed):
    result = hubbed.run("search", "Qwen3.8", "--author", "lmstudio-community", "--json")
    data = json.loads(result.output)
    assert [d["repo_id"] for d in data] == ["lmstudio-community/Qwen3.8-27B-GGUF"]


def test_recommend_offline_with_a_cold_cache_names_the_cache_file(hubbed):
    from local_llm import cli
    from local_llm.hub import HubError

    class Offline:
        def list_gguf(self, **kwargs):
            raise HubError("Could not list models on huggingface.co: no network")

    hubbed.monkeypatch.setattr(cli, "_make_hub", lambda st, refresh=False: Offline())
    result = hubbed.run("recommend")
    assert result.exit_code == 1
    assert "no network" in result.output and "hub-cache.json" in result.output
