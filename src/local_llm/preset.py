"""Read and edit llama-server preset files (INI) without losing comments or order.

The file is kept as a list of lines. Each line remembers what it is (header,
key = value, comment, blank) and which section it belongs to, so queries are
simple and writing back reproduces the file byte for byte.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_HEADER = re.compile(r"^\s*\[(?P<name>[^\]\n]+)\]\s*$")
# A comment starts with # or ; at the beginning of the line or after whitespace.
# A # glued to the previous character (as in a path) is part of the value.
_COMMENT = re.compile(r"(^|\s)[#;]")


class PresetError(Exception):
    pass


@dataclass
class _Line:
    raw: str
    kind: str  # "header" | "kv" | "comment" | "blank" | "other"
    section: str | None = None  # the header name, or the section a line sits in
    key: str | None = None
    value: str | None = None


def _strip_comment(line: str) -> str:
    match = _COMMENT.search(line)
    return line[: match.start()] if match else line


class Preset:
    def __init__(self, lines: list[_Line] | None = None) -> None:
        self._lines: list[_Line] = lines or []

    # ------------------------------------------------------------ parse / dump

    @classmethod
    def parse(cls, text: str) -> Preset:
        lines: list[_Line] = []
        current: str | None = None
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped == "":
                lines.append(_Line(raw, "blank", current))
                continue
            if stripped[0] in "#;":
                lines.append(_Line(raw, "comment", current))
                continue
            header = _HEADER.match(_strip_comment(raw))
            if header:
                current = header.group("name").strip()
                lines.append(_Line(raw, "header", current))
                continue
            body = _strip_comment(raw)
            if "=" in body:
                key, _, value = body.partition("=")
                lines.append(_Line(raw, "kv", current, key.strip(), value.strip()))
                continue
            lines.append(_Line(raw, "other", current))
        return cls(lines)

    @classmethod
    def load(cls, path: Path) -> Preset:
        try:
            text = path.read_text()
        except FileNotFoundError:
            raise PresetError(
                f"Missing model config: {path}\n"
                "Every model lives in that file. Create it with: local-llm setup"
            ) from None
        return cls.parse(text)

    def dump(self) -> str:
        return "".join(line.raw + "\n" for line in self._lines)

    # ------------------------------------------------------------ queries

    def sections(self) -> list[str]:
        names: list[str] = []
        for line in self._lines:
            if line.kind == "header" and line.section != "*" and line.section not in names:
                names.append(line.section)  # type: ignore[arg-type]
        return names

    def has_section(self, name: str) -> bool:
        return any(line.kind == "header" and line.section == name for line in self._lines)

    def items(self, section: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in self._lines:
            if line.kind == "kv" and line.section == section and line.key is not None:
                out[line.key] = line.value or ""
        return out

    def get(self, section: str, key: str, fallback_to_star: bool = True) -> str | None:
        own = self.items(section)
        if key in own:
            return own[key]
        if fallback_to_star:
            return self.items("*").get(key)
        return None

    def file_sizes(self, section: str) -> list[int]:
        """Sizes in bytes of the model file and, when present, the mmproj file."""
        model = self.get(section, "model", fallback_to_star=False)
        if not model:
            raise PresetError(f"Section [{section}] has no model key")
        sizes: list[int] = []
        for key in ("model", "mmproj"):
            value = self.get(section, key, fallback_to_star=False)
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_file():
                raise PresetError(f"Section [{section}]: {key} file is missing: {path}")
            sizes.append(path.stat().st_size)
        return sizes
