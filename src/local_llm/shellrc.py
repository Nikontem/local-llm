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


def line_ending(text: str) -> str:
    """The terminator this file already uses, so anything added to it matches.

    A file saved on Windows, or kept in a dotfiles repository that normalises to CRLF,
    should not come back with our lines ending differently from every other line.
    """
    return "\r\n" if "\r\n" in text else "\n"


def block_text(lines: list[str], ending: str = "\n") -> str:
    return ending.join([MARK_BEGIN, *lines, MARK_END]) + ending


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


def encoding_warning(path: Path) -> str:
    """The one line both integration and uninstall print about a file that is not UTF-8.

    Reading it in whatever encoding the locale happens to name and writing it back in
    UTF-8 transcodes somebody's file behind their back, so it is not read at all.
    """
    return (
        f"{path} is not UTF-8, so it was left exactly as it is:"
        " save it as UTF-8, then run this again"
    )


def unreadable_warning(path: Path, error: Exception) -> str:
    """The one line printed about a file that cannot be opened at all.

    A permission this user does not have, or a device that is gone. Different from
    encoding_warning, which is about a file that opens and cannot be decoded, and
    said separately because "save it as UTF-8" is no help with either of those.
    """
    return f"{path} cannot be read ({error}), so it was left exactly as it is"


def _block_spans(text: str) -> list[tuple[int, int]]:
    """Where each whole block of ours sits. Empty: no block, or markers that do not pair up.

    More than one is unusual but real - an older version of this tool appended a
    second, or somebody pasted the same block twice. Acting on the first alone left
    the others behind while reporting the file clean, so every one is answered for.
    """
    if marker_problem(text) is not None:
        return []
    found = _markers(text)
    return [(begin[1], end[2]) for begin, end in zip(found[0::2], found[1::2], strict=True)]


def _block_span(text: str) -> tuple[int, int] | None:
    """Where our first whole block sits, or None when there is not one."""
    spans = _block_spans(text)
    return spans[0] if spans else None


def has_block(text: str) -> bool:
    """Is a whole block of ours in this text?"""
    return _block_span(text) is not None


def mentions_block(text: str) -> bool:
    """Is a marker line of ours in this text - a whole block, or half of one?

    Uninstall looks for this rather than for a whole block: half a block cannot be
    removed, but the person still has to be told it is there.
    """
    return bool(_markers(text))


def _without(text: str, spans: list[tuple[int, int]]) -> str:
    """The text with those stretches cut out, taken from the back so the offsets hold."""
    for begin, end in reversed(spans):
        text = text[:begin] + text[end:]
    return text


def upsert_block(text: str, lines: list[str]) -> str:
    """Replace our block, or add one at the bottom, leaving exactly one behind.

    Markers that do not pair up are left alone: the text comes back exactly as it
    went in. See marker_problem.
    """
    if marker_problem(text) is not None:
        return text
    ending = line_ending(text)
    block = block_text(lines, ending)
    spans = _block_spans(text)
    if spans:
        # The first one keeps its place in the file; any others are dropped, so a file
        # that somehow grew two of them comes back with one.
        pieces = [text[: spans[0][0]], block]
        previous = spans[0][1]
        for begin, end in spans[1:]:
            pieces.append(text[previous:begin])
            previous = end
        pieces.append(text[previous:])
        return "".join(pieces)
    if not text:
        return block
    if text.endswith(ending * 2):
        separator = ""
    elif text.endswith(ending):
        separator = ending
    else:
        separator = ending * 2
    return text + separator + block


def remove_block(text: str) -> str:
    """Take every block of ours out. Markers that do not pair up are left alone,
    because the end marker they would pair with belongs to somebody else's lines."""
    return _without(text, _block_spans(text))


def block_lines(text: str) -> list[str]:
    """The lines inside our marked blocks, in the order they appear in the file."""
    inner: list[str] = []
    for begin, end in _block_spans(text):
        inner.extend(text[begin:end].splitlines()[1:-1])
    return inner


def shadowed_aliases(text: str, names: Sequence[str]) -> list[tuple[str, str]]:
    """Aliases of ours that a line outside our block also defines, and which one wins.

    A shell takes the last definition it reads, so where the other line sits decides
    the outcome: above our block it loses to ours, below it wins. Either way one of
    the two silently does nothing, which is worth a line of output.

    The answer is ("ours") or ("theirs") per name. A name defined outside the block
    more than once is answered for by the last of them, which is the one that counts.
    """
    spans = _block_spans(text)
    if not spans:
        return []
    begin, end = spans[0]
    wanted = set(names)
    winners: dict[str, str] = {}
    offset = 0
    for line in text.splitlines(keepends=True):
        name = alias_name(line)
        # Only a line that starts at the left margin. An indented one is almost always
        # inside a function body, where it defines nothing until the function is called,
        # and warning about an alias that is not really there is worse than saying
        # nothing about an unusually indented one that is.
        if name in wanted and not line[:1].isspace() and not (begin <= offset < end):
            winners[name] = "ours" if offset < begin else "theirs"
        offset += len(line)
    return [(name, winners[name]) for name in names if name in winners]


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
