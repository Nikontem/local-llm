"""One log file per router run, a pointer to the current one, and pruning."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from .paths import SERVICE


def new_run_log(log_dir: Path, service: str = SERVICE, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{service}.{now:%Y-%m-%dT%H-%M-%S}.log"
    path.touch()
    path.chmod(0o600)
    link = log_dir / f"{service}.log"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.name)
    except OSError:
        # No symlinks here (some filesystems, Windows without privileges): keep a pointer file.
        (log_dir / "current").write_text(str(path))
    return path


def current_log(log_dir: Path, service: str = SERVICE) -> Path | None:
    link = log_dir / f"{service}.log"
    if link.is_symlink():
        target = link.resolve()
        return target if target.is_file() else None
    pointer = log_dir / "current"
    if pointer.is_file():
        target = Path(pointer.read_text().strip())
        return target if target.is_file() else None
    return None


def prune_logs(
    log_dir: Path, days: int, service: str = SERVICE, now: datetime | None = None
) -> int:
    now = now or datetime.now()
    keep = current_log(log_dir, service)
    cutoff = now - timedelta(days=days)
    removed = 0
    for file in log_dir.glob(f"{service}.*.log"):
        if file.is_symlink():
            continue
        if keep is not None and file.resolve() == keep:
            continue
        if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
            file.unlink()
            removed += 1
    return removed


def tail_lines(path: Path, count: int) -> list[str]:
    with path.open(errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=count)]
