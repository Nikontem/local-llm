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


def test_configure_installs_the_plugin_and_the_tiny_agent(tmp_path):
    from local_llm.integrations import HarnessContext, opencode
    from local_llm.paths import Paths
    from local_llm.preset import Preset
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True)
    blob = tmp_path / "small.gguf"
    blob.write_bytes(b"x" * 10)
    paths.preset.write_text(f"[*]\nc = 8192\n[small]\nmodel = {blob}\n")
    config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True)
    (config_dir / "opencode.json").write_text("{}\n")
    ctx = HarnessContext(
        paths=paths,
        settings=Settings(),
        preset=Preset.load(paths.preset),
        home=tmp_path,
        env={},
        say=lambda line: None,
        yes=True,
    )
    lines = opencode.configure(ctx)
    assert (config_dir / "plugins" / PLUGIN_NAME).is_file()
    assert '"tiny"' in (config_dir / "opencode.json").read_text()
    assert any("plugin installed" in line for line in lines)
    assert opencode.harness_status(ctx) == "same"
    assert any("already" in line for line in opencode.configure(ctx))


# A tiny agent somebody set up themselves, on a model this tool never serves.
THEIRS = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {"anthropic": {"options": {"apiKey": "sk-mine"}}},
    "agent": {"tiny": {"mode": "subagent", "model": "anthropic/claude-haiku-4-5"}},
}


def make_ctx(tmp_path, *, yes=True, answers=None, config=None):
    """A context in a scratch home, with one model called small and a config to match."""
    from local_llm.integrations import HarnessContext
    from local_llm.paths import Paths
    from local_llm.preset import Preset
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    blob = tmp_path / "small.gguf"
    blob.write_bytes(b"x" * 10)
    paths.preset.write_text(f"[*]\nc = 8192\n[small]\nmodel = {blob}\n")
    config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (config_dir / "opencode.json").write_text(config)
    asked: list[tuple[str, bool]] = []
    replies = list(answers or [])

    def confirm(prompt: str, default: bool) -> bool:
        asked.append((prompt, default))
        return replies.pop(0) if replies else default

    ctx = HarnessContext(
        paths=paths,
        settings=Settings(),
        preset=Preset.load(paths.preset),
        home=tmp_path,
        env={},
        say=lambda line: None,
        confirm=confirm,
        yes=yes,
    )
    ctx.asked = asked  # type: ignore[attr-defined]
    ctx.config = config_dir / "opencode.json"  # type: ignore[attr-defined]
    return ctx


def test_the_config_is_copied_aside_and_replaced_in_one_step(tmp_path):
    """A crash or a full disk mid-write used to truncate the whole opencode config."""
    from local_llm.integrations import opencode

    original = json.dumps(THEIRS, indent=2) + "\n"
    ctx = make_ctx(tmp_path, yes=False, answers=[True], config=original)

    opencode.configure(ctx)

    backup = ctx.config.with_name("opencode.json.local-llm.bak")
    assert backup.read_text() == original
    assert not list(ctx.config.parent.glob(".opencode.json.*")), "no temporary file left behind"


def test_a_symlinked_opencode_config_stays_a_symlink(tmp_path):
    """A config linked into a dotfiles repository is written through, not replaced."""
    from local_llm.integrations import opencode

    ctx = make_ctx(tmp_path, yes=True)
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real = dotfiles / "opencode.json"
    real.write_text('{"$schema": "https://opencode.ai/config.json"}\n')
    ctx.config.symlink_to(real)

    opencode.configure(ctx)

    assert ctx.config.is_symlink(), "the link was replaced by a regular file"
    assert json.loads(real.read_text())["agent"]["tiny"]["model"] == "llamacpp/small"
    assert not list(dotfiles.glob(".opencode.json.*")), "no temporary file left behind"


def test_a_write_that_fails_is_reported_not_raised(tmp_path, monkeypatch):
    from local_llm.integrations import opencode

    ctx = make_ctx(tmp_path, yes=True, config="{}\n")

    def refuse(source, destination):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(opencode.os, "replace", refuse)
    lines = opencode.configure(ctx)
    assert any("could not write" in line for line in lines)
    assert any("could not update" in line for line in lines)
