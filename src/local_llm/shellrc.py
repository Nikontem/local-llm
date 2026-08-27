"""Shell integration: a marked block in the rc file, retiring the old zsh line, completion."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

MARK_BEGIN = "# >>> local-llm >>>"
MARK_END = "# <<< local-llm <<<"
RETIRED_PREFIX = "# retired by local-llm: "
SHELLS = ("zsh", "bash", "fish")
PROG_NAME = "local-llm"
COMPLETE_VAR = "_LOCAL_LLM_COMPLETE"
_OLD_SOURCE = re.compile(r"^\s*(\[\[.*\]\]\s*&&\s*)?(source|\.)\s+.*local-llm/local_llm\.zsh")
TOOL_ALIAS = ("local_llm", "local-llm")
_ALIAS_NAME = re.compile(r"^\s*alias\s+([A-Za-z_][A-Za-z0-9_]*)")


def detect_shell(env: Mapping[str, str] | None = None, system: str | None = None) -> str:
    env = os.environ if env is None else env
    name = Path(env.get("SHELL", "")).name
    if name in SHELLS:
        return name
    system = system or platform.system()
    return "zsh" if system == "Darwin" else "bash"


def rc_file(shell: str, home: Path) -> Path:
    if shell == "zsh":
        return home / ".zshrc"
    if shell == "fish":
        return home / ".config" / "fish" / "config.fish"
    return home / ".bashrc"


def alias_lines(shell: str, aliases: Sequence[tuple[str, str]]) -> list[str]:
    if shell == "fish":
        return [f"alias {name} '{command}'" for name, command in aliases]
    return [f"alias {name}='{command}'" for name, command in aliases]


def alias_name(line: str) -> str | None:
    match = _ALIAS_NAME.match(line)
    return match.group(1) if match else None


def block_text(lines: list[str]) -> str:
    return "\n".join([MARK_BEGIN, *lines, MARK_END]) + "\n"


def _markers(text: str) -> list[tuple[str, int, int]]:
    """Every marker line of ours, as ("begin"|"end", first character, past the newline).

    A marker counts only when it is the whole line. Matching it anywhere in the text
    turned a line that merely mentions it - inside a comment, or in a quoted string an
    installer echoes - into a fence around somebody else's shell configuration.
    """
    found: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == MARK_BEGIN:
            found.append(("begin", offset, offset + len(line)))
        elif stripped == MARK_END:
            found.append(("end", offset, offset + len(line)))
        offset += len(line)
    return found


def marker_problem(text: str) -> str | None:
    """Why this file's markers cannot be edited safely, or None when they can.

    Half a block - a begin line whose end line somebody deleted in a bad hand-edit or a
    dotfiles merge, or an end line with nothing above it - is never repaired and never
    removed. Appending a fresh block below a stray begin marker leaves those two able to
    pair up, and the next removal then deletes every line between them.
    """
    expected = "begin"
    for kind, _, _ in _markers(text):
        if kind != expected:
            if kind == "end":
                return f"has a {MARK_END!r} line with no {MARK_BEGIN!r} line above it"
            return f"has a second {MARK_BEGIN!r} line before the one above it was closed"
        expected = "end" if kind == "begin" else "begin"
    if expected == "end":
        return f"has a {MARK_BEGIN!r} line with no {MARK_END!r} line below it"
    return None


def marker_warning(path: Path, problem: str) -> str:
    """The one line both integration and uninstall print about a file they will not touch."""
    return (
        f"{path} {problem}, so it was left exactly as it is:"
        " fix the markers by hand, then run this again"
    )


def _block_span(text: str) -> tuple[int, int] | None:
    """Where our first whole block sits, or None: no block, or markers that do not pair up."""
    if marker_problem(text) is not None:
        return None
    found = _markers(text)
    return None if not found else (found[0][1], found[1][2])


def has_block(text: str) -> bool:
    """Is a whole block of ours in this text?"""
    return _block_span(text) is not None


def mentions_block(text: str) -> bool:
    """Is a marker line of ours in this text - a whole block, or half of one?

    Uninstall looks for this rather than for a whole block: half a block cannot be
    removed, but the person still has to be told it is there.
    """
    return bool(_markers(text))


def upsert_block(text: str, lines: list[str]) -> str:
    """Replace our block, or add one at the bottom. Markers that do not pair up are
    left alone: the text comes back exactly as it went in. See marker_problem."""
    if marker_problem(text) is not None:
        return text
    block = block_text(lines)
    span = _block_span(text)
    if span:
        return text[: span[0]] + block + text[span[1]:]
    if not text:
        return block
    separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return text + separator + block


def remove_block(text: str) -> str:
    """Take our block out. Markers that do not pair up are left alone, because the
    end marker they would pair with belongs to somebody else's lines."""
    span = _block_span(text)
    return text if span is None else text[: span[0]] + text[span[1]:]


def block_lines(text: str) -> list[str]:
    """The lines inside our marked block, or none when there is no block."""
    span = _block_span(text)
    if span is None:
        return []
    inner = text[span[0] : span[1]].splitlines()
    return [line for line in inner[1:-1]]


def merge_alias_lines(existing: list[str], wanted: list[str]) -> list[str]:
    """Add-only: an alias already there is never dropped, whatever we were asked for."""
    have = {alias_name(line) for line in existing}
    return [*existing, *(line for line in wanted if alias_name(line) not in have)]


def old_source_lines(text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines()) if _OLD_SOURCE.match(line)]


def retire_old_source(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    count = 0
    for index in old_source_lines(text):
        lines[index] = RETIRED_PREFIX + lines[index]
        count += 1
    return "".join(lines), count


def completion_script(shell: str) -> str:
    from typer._completion_shared import get_completion_script

    return get_completion_script(prog_name=PROG_NAME, complete_var=COMPLETE_VAR, shell=shell)


def install_completion(shell: str) -> Path:
    """Write the completion file where the shell looks for it; typer knows each shell's place."""
    from typer._completion_shared import install

    _, path = install(shell=shell, prog_name=PROG_NAME, complete_var=COMPLETE_VAR)
    return Path(path)
