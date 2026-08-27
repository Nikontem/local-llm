"""What a harness integration is allowed to touch, all of it injectable for tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import Paths
from ..preset import Preset
from ..settings import Settings


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
