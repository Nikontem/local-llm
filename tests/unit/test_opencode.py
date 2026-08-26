import json

from local_llm.integrations.opencode import (
    CONFIG_CANDIDATES,
    PLUGIN_NAME,
    agent_snippet,
    current_tiny_model,
    install_plugin,
    is_strict_json,
    merge_agent,
    opencode_paths,
    plugin_source,
    plugin_status,
    tiny_agent,
)


def test_paths_default_and_override(tmp_path):
    p = opencode_paths(home=tmp_path, env={})
    assert p.config_dir == tmp_path / ".config" / "opencode"
    assert p.plugin == p.config_dir / "plugins" / PLUGIN_NAME
    assert p.config_file is None and p.new_config == p.config_dir / "opencode.jsonc"
    (p.config_dir).mkdir(parents=True)
    (p.config_dir / "opencode.json").write_text("{}")
    (p.config_dir / "config.json").write_text("{}")
    assert opencode_paths(home=tmp_path, env={}).config_file == p.config_dir / "opencode.json"
    (p.config_dir / "opencode.jsonc").write_text("{}")
    assert opencode_paths(home=tmp_path, env={}).config_file == p.config_dir / "opencode.jsonc"
    custom = opencode_paths(home=tmp_path, env={"OPENCODE_CONFIG_DIR": str(tmp_path / "oc")})
    assert custom.config_dir == tmp_path / "oc"
    assert CONFIG_CANDIDATES == ("opencode.jsonc", "opencode.json", "config.json")


def test_plugin_status_and_install(tmp_path):
    p = opencode_paths(home=tmp_path, env={})
    assert plugin_status(p) == "missing"
    written = install_plugin(p)
    assert written == p.plugin and written.read_text() == plugin_source()
    assert "PROVIDER_ID" in plugin_source() and "LOCAL_LLM_PRESET" in plugin_source()
    assert plugin_status(p) == "same"
    p.plugin.write_text("// edited")
    assert plugin_status(p) == "different"


def test_tiny_agent_and_snippet():
    agent = tiny_agent("llamacpp/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M")
    assert agent["mode"] == "subagent"
    assert agent["model"] == "llamacpp/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
    assert agent["tools"] == {"write": False, "edit": False, "bash": False}
    snippet = agent_snippet(agent)
    assert snippet.startswith('"agent": {') and '"tiny"' in snippet


def test_merge_agent_into_strict_json_only():
    original = json.dumps(
        {"$schema": "https://opencode.ai/config.json", "agent": {"other": {"mode": "primary"}}},
        indent=2,
    )
    merged = merge_agent(original, tiny_agent("llamacpp/x"))
    data = json.loads(merged)
    assert data["$schema"] == "https://opencode.ai/config.json"
    assert data["agent"]["other"] == {"mode": "primary"}
    assert data["agent"]["tiny"]["model"] == "llamacpp/x"
    assert merged.endswith("\n")
    assert merge_agent("", tiny_agent("llamacpp/x")) is not None  # empty file counts as {}
    jsonc = '{\n  // a comment\n  "agent": {}\n}\n'
    assert not is_strict_json(jsonc) and merge_agent(jsonc, tiny_agent("llamacpp/x")) is None
    assert current_tiny_model(original) is None
    assert current_tiny_model(merged) == "llamacpp/x"
    commented = '{"agent": {"tiny": {"model": "llamacpp/old"}}, // c\n}'
    assert current_tiny_model(commented) == "llamacpp/old"
