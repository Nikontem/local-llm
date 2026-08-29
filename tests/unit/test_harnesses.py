import pytest

from local_llm import harnesses


def which_only(*names):
    installed = set(names)
    return lambda binary: f"/usr/local/bin/{binary}" if binary in installed else None


def test_registry_shape():
    keys = [h.key for h in harnesses.REGISTRY]
    assert keys == [
        "codex", "opencode", "claude", "copilot", "aider", "qwen", "gemini", "antigravity"
    ]
    for harness in harnesses.REGISTRY:
        assert harness.kind in (harnesses.PROVIDER, harnesses.LAUNCHER, harnesses.INFORMATIONAL)
        if harness.kind == harnesses.PROVIDER:
            assert harness.configure is not None and harness.status is not None
        if harness.kind == harnesses.INFORMATIONAL:
            assert harness.note and harness.configure is None
        if harness.kind == harnesses.LAUNCHER:
            assert harness.alias is not None
    assert harnesses.find("codex").binary == "codex"
    assert harnesses.find("nope") is None


def test_detect_splits_installed_from_missing():
    installed, missing = harnesses.detect(which_only("codex", "claude"))
    assert [h.key for h in installed] == ["codex", "claude"]
    assert "opencode" in [h.key for h in missing]


def test_numbered_excludes_informational_rows():
    installed, _ = harnesses.detect(which_only("codex", "claude", "gemini"))
    assert [h.key for h in harnesses.numbered(installed)] == ["codex", "claude"]


def test_render_groups_and_names_what_was_not_found():
    installed, missing = harnesses.detect(which_only("codex", "claude", "gemini"))
    lines = harnesses.render(installed, missing, lambda h: "missing")
    text = "\n".join(lines)
    assert "Configured inside the agent" in text and "Launched through local-llm" in text
    assert "Detected, but cannot use the router" in text
    assert "1. OpenAI Codex CLI" in text and "2. Claude Code" in text
    assert "Gemini CLI" in text and "3." not in text
    assert "Not found:" in text and "opencode" in text and "Qwen Code" in text


def test_render_says_nothing_was_found():
    installed, missing = harnesses.detect(which_only())
    text = "\n".join(harnesses.render(installed, missing, lambda h: "missing"))
    assert "No coding agents found on PATH" in text
    assert "local-llm env" in text


def test_parse_choice():
    installed, _ = harnesses.detect(which_only("codex", "claude", "aider"))
    rows = harnesses.numbered(installed)
    assert [h.key for h in harnesses.parse_choice("1 3", rows)] == ["codex", "aider"]
    assert [h.key for h in harnesses.parse_choice("a", rows)] == ["codex", "claude", "aider"]
    assert harnesses.parse_choice("n", rows) == []
    assert harnesses.parse_choice("", rows) == []
    for bad in ("0", "4", "x", "1 9"):
        with pytest.raises(ValueError):
            harnesses.parse_choice(bad, rows)


def test_the_informational_notes_stay_honest():
    gemini = harnesses.find("gemini").note
    antigravity = harnesses.find("antigravity").note
    assert "Qwen Code" in gemini and "Google" in gemini
    assert "google-antigravity" in antigravity
    for note in (gemini, antigravity):
        assert "proxy" not in note.lower(), "never point anyone at a proxy patch"


def test_note_lines_warn_only_when_the_host_is_not_loopback():
    from local_llm.settings import Settings

    installed, _ = harnesses.detect(which_only("codex"))
    assert harnesses.note_lines(Settings(), installed) == []
    remote = harnesses.note_lines(Settings(host="0.0.0.0", allow_remote=True), installed)
    assert len(remote) == 2 and "not a loopback address" in remote[1]
    none_installed, _ = harnesses.detect(which_only())
    assert harnesses.note_lines(Settings(host="0.0.0.0"), none_installed) == []


def test_safe_status_never_raises():
    def explode(harness):
        raise OSError("permission denied")

    harness = harnesses.find("opencode")
    assert harnesses.safe_status(harness, explode) == "unknown"
    assert harnesses.safe_status(harness, lambda h: "same") == "same"
    assert harnesses.safe_status(harnesses.find("gemini"), explode) == ""


def alias_ctx(home, env=None):
    from local_llm.integrations import HarnessContext
    from local_llm.paths import Paths
    from local_llm.settings import Settings

    return HarnessContext(
        paths=Paths.from_env(env={}, home=home),
        settings=Settings(),
        home=home,
        env={"SHELL": "/bin/zsh"} if env is None else env,
    )


def test_a_launcher_reports_whether_its_alias_is_installed(tmp_path):
    """The menu's 'alias claude_local installed' row was unreachable: no launcher
    entry had a status, so the state could never be 'same'."""
    from local_llm.shellrc import block_text

    ctx = alias_ctx(tmp_path)
    claude = harnesses.find("claude")
    assert claude.status is not None
    assert claude.status(ctx) == "missing", "there is no shell startup file at all"

    rc = tmp_path / ".zshrc"
    rc.write_text("alias claude_local='local-llm claude'\n")
    assert claude.status(ctx) == "missing", "an alias outside our block is not ours"

    rc.write_text("export A=1\n" + block_text(["alias claude_local='local-llm claude'"]))
    assert claude.status(ctx) == "same"
    assert harnesses.find("aider").status(ctx) == "missing", "only the alias that is there"

    rc.write_bytes(b"\xff\xfe not valid utf-8")
    assert claude.status(ctx) == "unknown"


def test_the_menu_row_says_an_installed_alias_is_installed(tmp_path):
    from local_llm.shellrc import block_text

    (tmp_path / ".zshrc").write_text(block_text(["alias claude_local='local-llm claude'"]))
    ctx = alias_ctx(tmp_path)
    installed, missing = harnesses.detect(which_only("claude", "aider"))
    text = "\n".join(harnesses.render(installed, missing, lambda h: h.status(ctx)))
    assert "alias claude_local installed" in text
    assert "alias aider_local" in text and "alias aider_local installed" not in text


def test_one_parser_answers_for_every_menu_in_the_tool():
    """cli._numbers used to be a second copy, refusing the same bad answer differently."""
    import pytest

    from local_llm import cli
    from local_llm.harnesses import parse_numbers

    assert parse_numbers("1 3", 4) == [1, 3]
    assert parse_numbers("2 2", 4) == [2], "a repeat picks the same thing once"
    for bad in ("", "   ", "x", "0", "5", "-1", "1 x", "1.5"):
        with pytest.raises(ValueError, match="Pick numbers between 1 and 4"):
            parse_numbers(bad, 4)

    assert cli._numbers("1 3", 4) == parse_numbers("1 3", 4)
    assert "a for all" in str(
        pytest.raises(ValueError, parse_numbers, "x", 4, ", a for all").value
    )
