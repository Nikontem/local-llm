from local_llm import hardware
from local_llm.estimate import GIB, budget_bytes, estimate_bytes, fit, human_gb


def test_estimate_matches_the_measured_rule():
    file_17_6_gb = int(17.6 * GIB)
    est = estimate_bytes([file_17_6_gb])
    assert est == file_17_6_gb * 115 // 100 + GIB
    assert 20.0 * GIB < est < 21.5 * GIB  # a 17.6 GB file settles near 19.9 GB resident; err high


def test_estimate_sums_several_files():
    assert estimate_bytes([GIB, GIB]) == 2 * GIB * 115 // 100 + GIB


def test_budget_subtracts_reserve_and_floors_at_zero():
    assert budget_bytes(48 * GIB, 10) == 38 * GIB
    assert budget_bytes(8 * GIB, 10) == 0


def test_fit_thresholds():
    budget = 100 * GIB
    assert fit(60 * GIB, budget) == "comfortable"
    assert fit(61 * GIB, budget) == "fits"
    assert fit(100 * GIB, budget) == "fits"
    assert fit(101 * GIB, budget) == "too_big"
    assert fit(1, 0) == "unknown"


def test_human_gb():
    assert human_gb(int(19.94 * GIB)) == "19.9 GB"
    assert human_gb(0) == "0.0 GB"


def test_total_ram_is_positive_int():
    total = hardware.total_ram()
    assert isinstance(total, int) and total > 0
