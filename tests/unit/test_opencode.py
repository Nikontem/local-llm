import json

from local_llm.integrations.opencode import (
    CONFIG_CANDIDATES,
    PLUGIN_NAME,
    agent_snippet,
    current_tiny_model,
    foreign_tiny_model,
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


def test_foreign_tiny_model_names_only_an_agent_we_did_not_write():
    assert foreign_tiny_model(json.dumps(THEIRS)) == "anthropic/claude-haiku-4-5"
    assert foreign_tiny_model('{"agent": {"tiny": {"model": "llamacpp/small"}}}') is None
    assert foreign_tiny_model("{}") is None


def test_a_tiny_agent_somebody_wrote_themselves_survives_yes_mode(tmp_path):
    """Under --yes and in the setup wizard nothing is asked, so nothing is replaced."""
    from local_llm.integrations import opencode

    ctx = make_ctx(tmp_path, yes=True, config=json.dumps(THEIRS, indent=2) + "\n")

    lines = opencode.configure(ctx)

    assert json.loads(ctx.config.read_text()) == THEIRS
    assert any("your own" in line and "anthropic/claude-haiku-4-5" in line for line in lines)
    assert any("without --yes" in line for line in lines)
    assert ctx.asked == [], "yes mode asks nothing"


def test_replacing_a_tiny_agent_of_theirs_is_a_question_that_names_the_model(tmp_path):
    from local_llm.integrations import opencode

    refusing = make_ctx(tmp_path, yes=False, answers=[False], config=json.dumps(THEIRS) + "\n")
    lines = opencode.configure(refusing)
    prompt, default = refusing.asked[-1]
    assert "anthropic/claude-haiku-4-5" in prompt and "llamacpp/small" in prompt
    assert "Replace" in prompt and not prompt.startswith("Add")
    assert default is False, "the default keeps what is already there"
    assert json.loads(refusing.config.read_text()) == THEIRS
    assert any("left on anthropic/claude-haiku-4-5" in line for line in lines)

    accepting = make_ctx(
        tmp_path / "other", yes=False, answers=[True], config=json.dumps(THEIRS) + "\n"
    )
    lines = opencode.configure(accepting)
    data = json.loads(accepting.config.read_text())
    assert data["agent"]["tiny"]["model"] == "llamacpp/small"
    assert data["provider"] == THEIRS["provider"], "the rest of the file is untouched"
    assert any("tiny agent set to llamacpp/small" in line for line in lines)


def test_a_tiny_agent_of_ours_is_updated_without_a_question(tmp_path):
    from local_llm.integrations import opencode

    ours = {"agent": {"tiny": {"mode": "subagent", "model": "llamacpp/retired"}}}
    ctx = make_ctx(tmp_path, yes=True, config=json.dumps(ours) + "\n")

    opencode.configure(ctx)

    assert json.loads(ctx.config.read_text())["agent"]["tiny"]["model"] == "llamacpp/small"


# The shape people actually write: the model sits after a nested object.
NESTED = {
    "$schema": "https://opencode.ai/config.json",
    "agent": {"tiny": {"tools": {"write": False}, "model": "anthropic/claude-haiku-4-5"}},
}


def test_a_model_after_a_nested_object_is_still_read(tmp_path):
    """A text match stops at the first nested brace, so it used to read nothing here."""
    from local_llm.integrations import opencode

    text = json.dumps(NESTED, indent=2) + "\n"
    assert current_tiny_model(text) == "anthropic/claude-haiku-4-5"
    assert foreign_tiny_model(text) == "anthropic/claude-haiku-4-5"

    ctx = make_ctx(tmp_path, yes=True, config=text)
    lines = opencode.configure(ctx)

    assert json.loads(ctx.config.read_text()) == NESTED, "somebody's paid model was replaced"
    assert any("your own" in line and "anthropic/claude-haiku-4-5" in line for line in lines)
    assert ctx.asked == [], "yes mode asks nothing"


def test_a_tiny_key_under_another_parent_never_decides_ownership(tmp_path):
    """Anything else called tiny - a provider, say - is not the agent in question."""
    from local_llm.integrations import opencode

    config = {
        "provider": {"tiny": {"model": "llamacpp/small"}},
        "agent": {"tiny": {"mode": "subagent", "model": "anthropic/claude-haiku-4-5"}},
    }
    text = json.dumps(config, indent=2) + "\n"
    assert current_tiny_model(text) == "anthropic/claude-haiku-4-5"
    assert foreign_tiny_model(text) == "anthropic/claude-haiku-4-5"

    ctx = make_ctx(tmp_path, yes=True, config=text)
    lines = opencode.configure(ctx)

    assert json.loads(ctx.config.read_text()) == config
    assert any("your own" in line for line in lines)


def test_our_own_agent_is_updated_even_when_the_model_sits_after_a_nested_object(tmp_path):
    from local_llm.integrations import opencode

    ours = {"agent": {"tiny": {"tools": {"write": False}, "model": "llamacpp/retired"}}}
    ctx = make_ctx(tmp_path, yes=True, config=json.dumps(ours, indent=2) + "\n")

    lines = opencode.configure(ctx)

    assert json.loads(ctx.config.read_text())["agent"]["tiny"]["model"] == "llamacpp/small"
    assert any("tiny agent set to llamacpp/small" in line for line in lines)


def test_a_config_that_is_not_strict_json_still_falls_back_to_the_text_match():
    """Nothing is ever written to a file with comments in it, so a match is enough there."""
    assert foreign_tiny_model('{\n  // mine\n  "agent": {"tiny": {"model": "anthropic/x"}}\n}') == (
        "anthropic/x"
    )
    assert foreign_tiny_model("{ not json at all") is None
    assert foreign_tiny_model('{"agent": []}') is None
    assert foreign_tiny_model('{"agent": {"tiny": "a string"}}') is None
    assert foreign_tiny_model('{"agent": {"tiny": {"model": 3}}}') is None
    assert foreign_tiny_model("[]") is None


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


def test_a_remote_router_is_warned_about_wherever_the_write_happens(tmp_path):
    """The menu warned; `local-llm integrate opencode` run on its own said nothing."""
    from local_llm.integrations import opencode
    from local_llm.settings import Settings

    ctx = make_ctx(tmp_path)
    ctx.settings = Settings(host="192.168.1.40")

    lines = opencode.configure(ctx)

    note = [line for line in lines if "192.168.1.40" in line and "loopback" in line]
    assert note, lines
    assert "http" in note[0] and "unencrypted" in note[0]

    assert not any("loopback" in line for line in opencode.configure(make_ctx(tmp_path)))


def test_the_plugin_is_read_once_so_the_diff_cannot_race_the_check(tmp_path, monkeypatch):
    """The second read was unguarded: between the two the file can change or stop
    being readable, which is the gap the Codex path closes by re-parsing inside its write."""
    from pathlib import Path

    from local_llm.integrations import opencode

    ctx = make_ctx(tmp_path, yes=True)
    plugin = tmp_path / ".config" / "opencode" / "plugins" / "local-llm-models.js"
    plugin.parent.mkdir(parents=True, exist_ok=True)
    plugin.write_text("// somebody else's plugin\n")

    reads: list[str] = []
    original = Path.read_text

    def counted(self, *args, **kwargs):
        if self == plugin:
            reads.append(str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    lines = opencode.configure(ctx)

    assert reads == [str(plugin)], "the plugin was read twice"
    assert any("plugin installed" in line for line in lines)


def test_the_config_is_read_as_utf8_and_an_unreadable_one_is_reported(tmp_path, monkeypatch):
    """It was read in whatever encoding the locale named and written back as UTF-8.

    On a machine whose locale is not UTF-8 that silently turned a person's own accented
    text into mojibake, in a file this tool does not own; on a config that is not UTF-8
    at all it raised straight past the caller.
    """
    from pathlib import Path

    from local_llm.integrations import opencode

    ctx = make_ctx(tmp_path, yes=True)
    ctx.config.write_text('{"note": "café résumé"}\n', encoding="utf-8")

    encodings: list[object] = []
    original = Path.read_text

    def watched(self, *args, encoding=None, **kwargs):
        if self == ctx.config:
            encodings.append(encoding)
        return original(self, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr(Path, "read_text", watched)
    opencode.configure(ctx)
    monkeypatch.undo()

    assert encodings and set(encodings) == {"utf-8"}, "the locale's encoding was used"
    assert "café résumé" in ctx.config.read_text(encoding="utf-8")

    ctx.config.write_bytes('{"note": "caf\xe9"}\n'.encode("iso-8859-1"))
    before = ctx.config.read_bytes()

    lines = opencode.configure(ctx)

    assert ctx.config.read_bytes() == before, "a config that is not UTF-8 was rewritten"
    assert any("not UTF-8" in line and str(ctx.config) in line for line in lines)


def test_a_config_that_cannot_be_opened_is_reported_not_raised(tmp_path, monkeypatch):
    from pathlib import Path

    from local_llm.integrations import opencode

    ctx = make_ctx(tmp_path, yes=True, config="{}\n")
    original = Path.read_text

    def refuse(self, *args, **kwargs):
        if self == ctx.config:
            raise PermissionError(13, "Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)
    lines = opencode.configure(ctx)

    assert any("cannot be read" in line and str(ctx.config) in line for line in lines), lines
