
from local_llm.shellrc import (
    MARK_BEGIN,
    MARK_END,
    RETIRED_PREFIX,
    alias_lines,
    block_text,
    completion_script,
    detect_shell,
    old_source_lines,
    rc_file,
    remove_block,
    retire_old_source,
    upsert_block,
)


def test_detect_shell_and_rc_file(tmp_path):
    assert detect_shell({"SHELL": "/bin/zsh"}) == "zsh"
    assert detect_shell({"SHELL": "/usr/bin/fish"}) == "fish"
    assert detect_shell({"SHELL": "/bin/bash"}) == "bash"
    assert detect_shell({}, system="Darwin") == "zsh" and detect_shell({}, system="Linux") == "bash"
    assert rc_file("zsh", tmp_path) == tmp_path / ".zshrc"
    assert rc_file("bash", tmp_path) == tmp_path / ".bashrc"
    assert rc_file("fish", tmp_path) == tmp_path / ".config" / "fish" / "config.fish"


def test_alias_lines_per_shell():
    assert alias_lines("zsh") == [
        "alias local_llm='local-llm'",
        "alias claude_local='local-llm claude'",
        "alias copilot_local='local-llm copilot'",
    ]
    assert alias_lines("fish")[0] == "alias local_llm 'local-llm'"


def test_upsert_block_replaces_in_place_and_appends_once():
    original = "export A=1\n"
    once = upsert_block(original, ["alias x='y'"])
    assert once == f"export A=1\n\n{MARK_BEGIN}\nalias x='y'\n{MARK_END}\n"
    twice = upsert_block(once + "export B=2\n", ["alias x='z'"])
    assert twice == f"export A=1\n\n{MARK_BEGIN}\nalias x='z'\n{MARK_END}\nexport B=2\n"
    assert twice.count(MARK_BEGIN) == 1
    assert remove_block(twice) == "export A=1\n\nexport B=2\n"
    assert upsert_block("", ["a"]) == f"{MARK_BEGIN}\na\n{MARK_END}\n"
    assert block_text(["a", "b"]) == f"{MARK_BEGIN}\na\nb\n{MARK_END}\n"


def test_retire_old_source_lines():
    text = (
        "# shell setup\n"
        '[[ -r "$HOME/.config/local-llm/local_llm.zsh" ]] '
        '&& source "$HOME/.config/local-llm/local_llm.zsh"\n'
        "source ~/other.zsh\n"
        ". /Users/me/.config/local-llm/local_llm.zsh\n"
    )
    assert old_source_lines(text) == [1, 3]
    retired, count = retire_old_source(text)
    assert count == 2
    lines = retired.splitlines()
    assert lines[1].startswith(RETIRED_PREFIX) and lines[3].startswith(RETIRED_PREFIX)
    assert lines[2] == "source ~/other.zsh"
    assert retire_old_source(retired) == (retired, 0)


def test_completion_script_mentions_the_program():
    zsh = completion_script("zsh")
    assert "local-llm" in zsh and "_LOCAL_LLM_COMPLETE" in zsh
    assert "local-llm" in completion_script("bash") and "local-llm" in completion_script("fish")
