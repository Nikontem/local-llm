"""How much memory a set of model files needs, and whether it fits a budget.

Weights on disk plus room for the KV cache and compute buffers. Measured
against real loads: a 17.6 GB file settles near 19.9 GB resident, a 1.1 GB
file near 2.0 GB. Deliberately errs high - refusing a load that would have fit
is cheaper than swapping.
"""

from __future__ import annotations

from collections.abc import Iterable

GIB = 1024**3


def estimate_bytes(sizes: Iterable[int]) -> int:
    return sum(sizes) * 115 // 100 + GIB


def budget_bytes(total_ram: int, reserve_gb: int) -> int:
    return max(0, total_ram - reserve_gb * GIB)


def fit(estimate: int, budget: int) -> str:
    if budget <= 0:
        return "unknown"
    if estimate * 10 <= budget * 6:
        return "comfortable"
    if estimate <= budget:
        return "fits"
    return "too_big"


def human_gb(n: int) -> str:
    return f"{n / GIB:.1f} GB"
