"""Needs the network. Run with: uv run pytest tests/live -m live -q"""

import pytest

from local_llm.discover import GROUPS, gather
from local_llm.estimate import GIB
from local_llm.hardware import Machine
from local_llm.hub import Hub

pytestmark = pytest.mark.live


def test_pipeline_returns_vendor_models_for_every_group():
    machine = Machine("Darwin", "arm64", "Apple M4 Pro", 48 * GIB, "apple", 48 * GIB, 20, 10)
    groups = gather(Hub(cache=None), machine, None, limit_per_group=3, max_bases=25)
    for group in GROUPS:
        assert groups[group], f"no candidates for {group}"
        assert all(c.lineage != "derivative" for c in groups[group])
        assert all(c.suggested is not None for c in groups[group])
