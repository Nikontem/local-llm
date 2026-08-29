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
                # A temporary file is created readable by its owner alone. Renaming it
                # over a file that was there already would quietly take away the
                # permissions somebody chose for it, so they are carried across first.
                # Through the open descriptor, not the path: nothing can be swapped in
                # between deciding on a file and changing it.
                os.fchmod(stream.fileno(), destination.stat().st_mode & 0o7777)
            # A rename is atomic against a process that dies, but the bytes behind it
            # may still be in the kernel's cache. Without this, a machine that loses
            # power just after the rename can come back to a file of zeros.
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        # A failed write must not litter the directory with hidden half-files.
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_directory(directory: Path) -> None:
    """Make the rename itself durable, not only the bytes it points at.

    Flushing the file guarantees its contents survive a power cut; the directory entry
    naming them is a separate write, and without this the file can come back under its
    old name. Not every filesystem allows a directory to be synced, and one that
    refuses is no reason to fail a write that has already succeeded.
    """
    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


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
    #: Set by a caller that has already said the remote-address warning itself. The
    #: menu says a fuller version of it before asking which agents to configure, so
    #: repeating it once per agent afterwards would be several paragraphs of the same
    #: thing. A command run on its own has said nothing, and leaves this alone.
    warned_remote: bool = False

    def ask(self, prompt: str, default: bool = True) -> bool:
        """In yes-mode every question takes its default without being asked."""
        return default if self.yes else self.confirm(prompt, default)


def remote_note(ctx: HarnessContext) -> list[str]:
    """What a write into another tool's config owes a person when the router is remote.

    Configuring an agent copies the router's address into a file on this machine, and
    the address the tool hands out is always plain http. With a host that is not this
    machine, that means the agent's prompts and the code it sends travel the network
    in the clear, which nothing else in the output would tell them. Said here so that
    a command run on its own says it too, and not only the menu.
    """
    if ctx.settings.is_local or ctx.warned_remote:
        return []
    return [
        f"note: {ctx.settings.host} is not a loopback address, and the base URL is plain"
        " http, so this agent's prompts and the code it sends cross your network"
        " unencrypted - and anything that can read its config file can see the address"
    ]
