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
