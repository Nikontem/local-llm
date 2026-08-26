"""Facts about this machine that decide what fits."""

from __future__ import annotations

import psutil


def total_ram() -> int:
    """Physical memory in bytes."""
    return int(psutil.virtual_memory().total)
