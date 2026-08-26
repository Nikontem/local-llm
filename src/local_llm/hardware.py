"""Facts about this machine that decide what fits."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from .estimate import GIB, budget_bytes, fit

_TIMEOUT = 5
_FIT_LABEL = {"comfortable": "comfortable", "fits": "fits", "too_big": "too big", "unknown": "unknown"}


def total_ram() -> int:
    """Physical memory in bytes."""
    return int(psutil.virtual_memory().total)


def _read(path: str) -> str:
    return Path(path).read_text()


@dataclass
class Probe:
    """Everything detection asks the machine, injectable for tests."""

    system: str = field(default_factory=platform.system)
    machine: str = field(default_factory=platform.machine)
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    which: Callable[[str], str | None] = shutil.which
    read_text: Callable[[str], str] = _read
    total_ram: Callable[[], int] = total_ram


@dataclass
class Machine:
    system: str
    machine: str
    chip: str
    total_ram: int
    gpu_kind: str  # "apple" | "nvidia" | "amd" | "none"
    gpu_vram: int
    gpu_cores: int
    reserve_gb: int

    @property
    def budget(self) -> int:
        return budget_bytes(self.total_ram, self.reserve_gb)

    @property
    def gpu_budget(self) -> int:
        if self.gpu_kind == "apple":
            return self.budget
        if self.gpu_kind in ("nvidia", "amd"):
            return max(0, self.gpu_vram - GIB)
        return 0

    def fit(self, estimate: int) -> str:
        return fit(estimate, self.budget)

    def fit_label(self, estimate: int) -> str:
        if self.gpu_kind in ("nvidia", "amd"):
            if estimate <= self.gpu_budget:
                return "fits on GPU"
            if estimate <= self.budget:
                return "fits with CPU offload (slower)"
            return "too big"
        return _FIT_LABEL[self.fit(estimate)]

    def describe(self) -> str:
        ram = f"{self.total_ram / GIB:.0f} GB"
        usable = f"{self.budget / GIB:.0f} GB usable for models ({self.reserve_gb} GB reserved)"
        if self.gpu_kind == "apple":
            cores = f", {self.gpu_cores} GPU cores" if self.gpu_cores else ""
            return f"{self.chip or 'Apple Silicon'}{cores}, {ram} unified memory → {usable}"
        name = self.chip or self.machine
        if self.gpu_kind in ("nvidia", "amd"):
            return (
                f"{name}, {ram} RAM, {self.gpu_kind.upper()} GPU with {self.gpu_vram / GIB:.0f} GB"
                f" → {usable}; {self.gpu_budget / GIB:.0f} GB on the GPU"
            )
        return (
            f"{name}, {ram} RAM, no GPU → {usable}; CPU only, models above about 8B"
            " parameters will be slow"
        )


def _output(probe: Probe, args: list[str]) -> str:
    try:
        result = probe.run(args, capture_output=True, text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return ""
    return result.stdout or ""


def detect(reserve_gb: int = 10, probe: Probe | None = None) -> Machine:
    probe = probe or Probe()
    system, arch = probe.system, probe.machine
    ram = probe.total_ram()
    chip, gpu_kind, vram, cores = "", "none", 0, 0

    if system == "Darwin":
        chip = _output(probe, ["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if arch == "arm64":
            gpu_kind, vram = "apple", ram
            found = re.search(
                r"Total Number of Cores:\s*(\d+)", _output(probe, ["system_profiler", "SPDisplaysDataType"])
            )
            cores = int(found.group(1)) if found else 0
    elif system == "Linux":
        try:
            info = probe.read_text("/proc/cpuinfo")
        except OSError:
            info = ""
        found = re.search(r"^model name\s*:\s*(.+)$", info, re.MULTILINE)
        chip = found.group(1).strip() if found else ""
        if probe.which("nvidia-smi"):
            out = _output(probe, ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            mib = [int(x) for x in re.findall(r"\d+", out)]
            if mib:
                gpu_kind, vram = "nvidia", sum(mib) * 1024 * 1024
        if gpu_kind == "none" and probe.which("rocm-smi"):
            out = _output(probe, ["rocm-smi", "--showmeminfo", "vram", "--json"])
            total = sum(int(x) for x in re.findall(r'"VRAM Total Memory \(B\)":\s*"?(\d+)', out))
            if total:
                gpu_kind, vram = "amd", total

    return Machine(system, arch, chip, ram, gpu_kind, vram, cores, reserve_gb)
