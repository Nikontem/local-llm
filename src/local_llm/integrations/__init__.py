"""What a harness integration is allowed to touch, all of it injectable for tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import Paths
from ..preset import Preset
from ..settings import Settings

BACKUP_SUFFIX = ".local-llm.bak"


def backup_path(target: Path) -> Path:
    """The backup name we own, not the plain .bak somebody may have made by hand.

    Uninstall deletes the files this tool wrote, and it must never be able to delete
    a person's own copy of their configuration.
    """
    return target.with_name(target.name + BACKUP_SUFFIX)


def atomic_write(target: Path, text: str, *, backup: bool = True) -> None:
    """Replace a file in one step, keeping a copy of what was in it.

    Writing in place truncates the file before the new bytes go in, so a crash or a
    full disk halfway through destroys everything that was there. This writes a
    temporary file beside the destination and renames it over the top, which no
    reader ever sees half done, and copies the old content aside first.

    People commonly symlink a configuration file into a dotfiles repository. Writing
    to the link path would replace the link with a regular file and leave the dotfiles
    copy holding the old content, so the write goes through to what the link points
    at. The temporary file has to sit beside that destination for os.replace to work
    across a filesystem boundary, and the backup stays where the link is.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = target.resolve() if target.is_symlink() else target
    if backup and target.is_file():
        copy = backup_path(target)
        if copy.is_symlink():
            copy.unlink()  # never write through a link somebody put there
        shutil.copy2(target, copy)
    handle, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        if destination.is_file():
            # A temporary file is created readable by its owner alone. Renaming it over
            # a file that was there already would quietly take away the permissions
            # somebody chose for it, so they are carried across first.
            os.chmod(temporary, destination.stat().st_mode & 0o7777)
        os.replace(temporary, destination)
    finally:
        # A failed write must not litter the directory with hidden half-files.
        if os.path.exists(temporary):
            os.unlink(temporary)


def _no_questions(prompt: str, default: bool) -> bool:
    """The default answer, used when nobody wired a real prompt."""
    return default


@dataclass
class HarnessContext:
    paths: Paths
    settings: Settings
    preset: Preset | None = None
    home: Path = field(default_factory=Path.home)
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    say: Callable[[str], None] = print
    confirm: Callable[[str, bool], bool] = _no_questions
    yes: bool = False

    def ask(self, prompt: str, default: bool = True) -> bool:
        """In yes-mode every question takes its default without being asked."""
        return default if self.yes else self.confirm(prompt, default)
