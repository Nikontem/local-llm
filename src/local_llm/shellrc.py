"""Shell integration: a marked block in the rc file, retiring the old zsh line, completion."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping
from pathlib import Path

MARK_BEGIN = "# >>> local-llm >>>"
MARK_END = "# <<< local-llm <<<"
RETIRED_PREFIX = "# retired by local-llm: "
SHELLS = ("zsh", "bash", "fish")
PROG_NAME = "local-llm"
COMPLETE_VAR = "_LOCAL_LLM_COMPLETE"
_OLD_SOURCE = re.compile(r"^\s*(\[\[.*\]\]\s*&&\s*)?(source|\.)\s+.*local-llm/local_llm\.zsh")
_ALIASES = [("local_llm", "local-llm"), ("claude_local", "local-llm claude"), ("copilot_local", "local-llm copilot")]


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


def alias_lines(shell: str) -> list[str]:
    if shell == "fish":
        return [f"alias {name} '{command}'" for name, command in _ALIASES]
    return [f"alias {name}='{command}'" for name, command in _ALIASES]


def block_text(lines: list[str]) -> str:
    return "\n".join([MARK_BEGIN, *lines, MARK_END]) + "\n"


def _block_span(text: str) -> tuple[int, int] | None:
    start = text.find(MARK_BEGIN)
    if start == -1:
        return None
    end = text.find(MARK_END, start)
    if end == -1:
        return None
    end += len(MARK_END)
    if end < len(text) and text[end] == "\n":
        end += 1
    return start, end


def upsert_block(text: str, lines: list[str]) -> str:
    block = block_text(lines)
    span = _block_span(text)
    if span:
        return text[: span[0]] + block + text[span[1]:]
    if not text:
        return block
    separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return text + separator + block


def remove_block(text: str) -> str:
    span = _block_span(text)
    return text if span is None else text[: span[0]] + text[span[1]:]


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
