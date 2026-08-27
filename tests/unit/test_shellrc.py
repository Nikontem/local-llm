
from pathlib import Path

from local_llm.shellrc import (
    MARK_BEGIN,
    MARK_END,
    RETIRED_PREFIX,
    TOOL_ALIAS,
    alias_lines,
    alias_name,
    block_lines,
    block_text,
    completion_script,
    detect_shell,
    has_block,
    marker_problem,
    marker_warning,
    mentions_block,
    merge_alias_lines,
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


def test_alias_lines_take_the_pairs_they_are_given():
    pairs = [("local_llm", "local-llm"), ("claude_local", "local-llm claude")]
    assert alias_lines("zsh", pairs) == [
        "alias local_llm='local-llm'",
        "alias claude_local='local-llm claude'",
    ]
    assert alias_lines("fish", pairs)[0] == "alias local_llm 'local-llm'"
    assert TOOL_ALIAS == ("local_llm", "local-llm")


def test_aliases_are_add_only():
    existing = ["alias local_llm='local-llm'", "alias copilot_local='local-llm copilot'"]
    wanted = ["alias local_llm='local-llm'", "alias claude_local='local-llm claude'"]
    assert merge_alias_lines(existing, wanted) == [
        "alias local_llm='local-llm'",
        "alias copilot_local='local-llm copilot'",
        "alias claude_local='local-llm claude'",
    ]
    assert alias_name("alias claude_local='local-llm claude'") == "claude_local"
    assert alias_name("alias qwen_local 'local-llm qwen'") == "qwen_local"
    assert alias_name("export A=1") is None


def test_block_lines_reads_back_what_is_inside_the_markers():
    text = upsert_block("export A=1\n", ["alias x='y'", "alias z='w'"])
    assert block_lines(text) == ["alias x='y'", "alias z='w'"]
    assert block_lines("export A=1\n") == []


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


# A shell startup file worth losing: a PATH export, somebody's own alias, and the
# line that makes pyenv work. The reviewer's reproduction emptied all three.
THEIRS = (
    'export PATH="$HOME/bin:$PATH"\n'
    "alias gs='git status'\n"
    'eval "$(pyenv init -)"\n'
)


def test_half_a_block_is_never_completed_and_never_removed():
    """A begin marker whose end marker somebody deleted must stop both writers.

    Appending a second block below the stray marker leaves the stray begin and the new
    end able to pair up, and the next removal deletes every line between them.
    """
    stray_begin = THEIRS + MARK_BEGIN + "\nalias local_llm='local-llm'\n"
    problem = marker_problem(stray_begin)
    assert problem is not None and MARK_END in problem
    assert upsert_block(stray_begin, ["alias local_llm='local-llm'"]) == stray_begin
    assert remove_block(stray_begin) == stray_begin
    assert block_lines(stray_begin) == [] and not has_block(stray_begin)
    assert mentions_block(stray_begin), "uninstall still has to see the file to report it"

    stray_end = THEIRS + MARK_END + "\n"
    problem = marker_problem(stray_end)
    assert problem is not None and MARK_BEGIN in problem
    assert upsert_block(stray_end, ["alias local_llm='local-llm'"]) == stray_end
    assert remove_block(stray_end) == stray_end

    doubled = THEIRS + MARK_BEGIN + "\n" + MARK_BEGIN + "\n" + MARK_END + "\n"
    assert marker_problem(doubled) is not None
    assert upsert_block(doubled, ["alias x='y'"]) == doubled

    whole = THEIRS + block_text(["alias local_llm='local-llm'"])
    assert marker_problem(whole) is None and has_block(whole)
    assert remove_block(whole) == THEIRS


def test_a_marker_mentioned_inside_a_line_is_not_a_marker():
    """find() matched the marker anywhere, so a comment about it fenced off real lines."""
    mentioning = (
        f"# the block below is written by local-llm, between {MARK_BEGIN} and {MARK_END}\n"
        + THEIRS
    )
    assert marker_problem(mentioning) is None
    assert not has_block(mentioning) and not mentions_block(mentioning)
    assert remove_block(mentioning) == mentioning
    added = upsert_block(mentioning, ["alias local_llm='local-llm'"])
    assert added == mentioning + "\n" + block_text(["alias local_llm='local-llm'"])
    assert block_lines(added) == ["alias local_llm='local-llm'"]
    assert remove_block(added) == mentioning + "\n"

    echoed = THEIRS + f'echo "{MARK_BEGIN}"\n'
    assert not mentions_block(echoed) and remove_block(echoed) == echoed


def test_an_indented_marker_is_still_a_marker():
    indented = f"  {MARK_BEGIN}  \nalias x='y'\n\t{MARK_END}\n"
    assert marker_problem(indented) is None and has_block(indented)
    assert block_lines(indented) == ["alias x='y'"]
    assert remove_block(indented) == ""


def test_the_warning_names_the_file_and_says_what_to_do():
    problem = marker_problem(MARK_BEGIN + "\n")
    warning = marker_warning(Path("/home/me/.zshrc"), problem)
    assert warning.startswith("/home/me/.zshrc has a ")
    assert "by hand" in warning and "left exactly as it is" in warning


def test_two_whole_blocks_are_both_taken_out(tmp_path):
    """An older version, or a paste, could leave a second one; the first-only removal
    took the aliases out of one and reported the file clean."""
    text = (
        "export A=1\n"
        + block_text(["alias one='x'"])
        + "export B=2\n"
        + block_text(["alias two='y'"])
        + "export C=3\n"
    )

    assert remove_block(text) == "export A=1\nexport B=2\nexport C=3\n"


def test_two_whole_blocks_converge_to_one_and_keep_every_alias(tmp_path):
    text = (
        "export A=1\n"
        + block_text(["alias one='x'"])
        + "export B=2\n"
        + block_text(["alias two='y'"])
    )

    assert block_lines(text) == ["alias one='x'", "alias two='y'"]

    merged = merge_alias_lines(block_lines(text), ["alias three='z'"])
    updated = upsert_block(text, merged)

    assert updated.count(MARK_BEGIN) == 1 and updated.count(MARK_END) == 1
    assert block_lines(updated) == ["alias one='x'", "alias two='y'", "alias three='z'"]
    assert "export A=1" in updated and "export B=2" in updated
