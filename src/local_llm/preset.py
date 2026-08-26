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
    def __init__(
        self,
        lines: list[_Line] | None = None,
        newline: str = "\n",
        trailing_newline: bool = True,
    ) -> None:
        self._lines: list[_Line] = lines or []
        # What line terminator the source text used, and whether its last line
        # had one. dump() reproduces both, so parse-then-write with no edits
        # is byte identical (spec 10) instead of always appending "\n".
        self._newline = newline
        self._trailing_newline = trailing_newline

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
        if "\r\n" in text:
            newline = "\r\n"
        elif "\r" in text:
            newline = "\r"
        else:
            newline = "\n"
        trailing_newline = text.endswith(("\n", "\r")) if text else True
        return cls(lines, newline=newline, trailing_newline=trailing_newline)

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
        if not self._lines:
            return ""
        body = self._newline.join(line.raw for line in self._lines)
        if self._trailing_newline:
            body += self._newline
        return body

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

    # ------------------------------------------------------------ editing

    def _span(self, name: str) -> tuple[int, int]:
        """Index of the header line and the index just past the section."""
        start = next(
            (
                i
                for i, line in enumerate(self._lines)
                if line.kind == "header" and line.section == name
            ),
            None,
        )
        if start is None:
            raise PresetError(f"No section [{name}] in the preset")
        end = next(
            (i for i in range(start + 1, len(self._lines)) if self._lines[i].kind == "header"),
            len(self._lines),
        )
        return start, end

    def _leading_comment_start(self, header_index: int) -> int:
        i = header_index
        while i > 0 and self._lines[i - 1].kind == "comment":
            i -= 1
        return i

    def _content_end(self, start: int, end: int) -> int:
        """Where a section's own lines actually stop.

        `_span` sets `end` to the index of the *next* header (or the end of
        the file), so the half-open range [start, end) also contains that
        next header's leading comment and the blank line before it - lines
        that belong to the section after this one, not to this one. Walk
        `end` back over the next header's leading comments first, then over
        blank layout lines, so only this section's own content remains.
        """
        boundary = end
        if boundary < len(self._lines):
            boundary = self._leading_comment_start(boundary)
        while boundary > start + 1 and self._lines[boundary - 1].kind == "blank":
            boundary -= 1
        return boundary

    @staticmethod
    def _section_lines(
        name: str, keys: Sequence[tuple[str, str]], comments: Sequence[str]
    ) -> list[_Line]:
        lines = [_Line(f"# {c}", "comment", name) for c in comments]
        lines.append(_Line(f"[{name}]", "header", name))
        lines.extend(_Line(f"{k} = {v}", "kv", name, k, v) for k, v in keys)
        return lines

    def add_section(
        self, name: str, keys: Sequence[tuple[str, str]], comments: Sequence[str] = ()
    ) -> None:
        if not name.strip() or "]" in name or "\n" in name:
            raise PresetError(f"Invalid section name: {name!r}")
        if self.has_section(name):
            raise PresetError(f"Section [{name}] already exists in the preset")
        if self._lines and self._lines[-1].kind != "blank":
            self._lines.append(_Line("", "blank", self._lines[-1].section))
        self._lines.extend(self._section_lines(name, keys, comments))

    def replace_section(
        self, name: str, keys: Sequence[tuple[str, str]], comments: Sequence[str] = ()
    ) -> None:
        start, end = self._span(name)
        body_end = self._content_end(start, end)
        lead = self._leading_comment_start(start)
        self._lines[lead:body_end] = self._section_lines(name, keys, comments)

    def remove_section(self, name: str) -> None:
        start, end = self._span(name)
        lead = self._leading_comment_start(start)
        del self._lines[lead : self._content_end(start, end)]
        # Two blank lines now touching each other read as a hole; keep one.
        if lead > 0 and lead < len(self._lines):
            if self._lines[lead - 1].kind == "blank" and self._lines[lead].kind == "blank":
                del self._lines[lead]
        # The separator that introduced a final section would otherwise trail the file.
        if lead >= len(self._lines) and self._lines and self._lines[-1].kind == "blank":
            self._lines.pop()

    # ------------------------------------------------------------ save

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(self.dump())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
