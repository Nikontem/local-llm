from pathlib import Path

import pytest

from local_llm.estimate import GIB
from local_llm.hub import (
    LIST_EXPAND,
    Hub,
    HubCache,
    HubError,
    Lineage,
    RepoListing,
    base_models_of,
    extra_tokens,
    free_disk_bytes,
)

from .fakes import FakeApi, fake_model

QWEN_BASE = fake_model(
    "Qwen/Qwen3.8-27B",
    tags=["base_model:Qwen/Qwen3.8-27B-Base", "base_model:finetune:Qwen/Qwen3.8-27B-Base"],
)
QWEN_ROOT = fake_model("Qwen/Qwen3.8-27B-Base", tags=[])
UNSLOTH = fake_model(
    "unsloth/Qwen3.8-27B-GGUF", downloads=7_000_000, likes=2900, trending=50,
    tags=["gguf", "base_model:Qwen/Qwen3.8-27B", "base_model:quantized:Qwen/Qwen3.8-27B"],
    base_models=["Qwen/Qwen3.8-27B"],
    gguf={
        "total": 27_320_697_856, "context_length": 262144,
        "architecture": "qwen35", "chat_template": "<think>",
    },
    files=[
        ("Qwen3.8-27B-UD-Q4_K_XL.gguf", int(17.2 * GIB)),
        ("mmproj-F16.gguf", int(0.8 * GIB)),
        ("README.md", 10),
    ],
)
REMIX_BASE = fake_model(
    "DavidAU/Qwen3.6-27B-Heretic", tags=["base_model:finetune:Qwen/Qwen3.6-27B"]
)
REMIX = fake_model(
    "DavidAU/Qwen3.6-27B-Heretic-GGUF", downloads=2_000_000,
    tags=["gguf", "base_model:quantized:DavidAU/Qwen3.6-27B-Heretic"],
    gguf={"total": 27e9}, files=[("Heretic-Q4_K_M.gguf", int(16 * GIB))],
)
ORPHAN = fake_model(
    "someone/mystery-GGUF", downloads=5, tags=["gguf"], gguf=None,
    files=[("mystery-Q4_0.gguf", GIB)],
)
LMSTUDIO = fake_model(
    "lmstudio-community/Qwen3.8-27B-GGUF", downloads=1_900_000,
    tags=["gguf", "base_model:quantized:Qwen/Qwen3.8-27B"], gguf={"total": 27e9},
    files=[("Qwen3.8-27B-Q4_K_M.gguf", int(16.5 * GIB))],
)
MODELS = {m.id: m for m in (QWEN_BASE, QWEN_ROOT, UNSLOTH, REMIX_BASE, REMIX, ORPHAN, LMSTUDIO)}


def make_hub(tmp_path, now=None, api=None):
    clock = {"t": 1000.0}
    cache = HubCache(tmp_path / "hub-cache.json", ttl=100, now=lambda: clock["t"])
    default_api = FakeApi(MODELS, {"unsloth/Qwen3.8-27B-GGUF": {"preset.ini", "README.md"}})
    hub = Hub(api=api or default_api, cache=cache)
    return hub, clock


def test_cache_round_trip_expiry_and_clear(tmp_path):
    clock = {"t": 0.0}
    cache = HubCache(tmp_path / "c.json", ttl=10, now=lambda: clock["t"])
    assert cache.get("k") is None
    cache.put("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    reopened = HubCache(tmp_path / "c.json", ttl=10, now=lambda: clock["t"])
    assert reopened.get("k") == {"a": 1}  # persisted
    clock["t"] = 11
    assert cache.get("k") is None
    cache.put("k", 2)
    cache.clear()
    assert cache.get("k") is None and not (tmp_path / "c.json").exists()


def test_list_gguf_maps_fields_and_caches(tmp_path):
    hub, clock = make_hub(tmp_path)
    listings = hub.list_gguf(limit=10)
    repo_ids = [item.repo_id for item in listings][:2]
    assert repo_ids == ["unsloth/Qwen3.8-27B-GGUF", "DavidAU/Qwen3.6-27B-Heretic-GGUF"]
    top = listings[0]
    assert isinstance(top, RepoListing)
    assert top.params == 27_320_697_856
    assert top.context_length == 262144 and top.architecture == "qwen35"
    assert top.chat_template == "<think>" and top.downloads == 7_000_000
    assert top.author == "unsloth"
    assert base_models_of(top) == ["Qwen/Qwen3.8-27B"]
    first_calls = len(hub.api.calls)
    hub.list_gguf(limit=10)
    assert len(hub.api.calls) == first_calls  # served from cache
    assert hub.api.calls[0][1]["expand"] == LIST_EXPAND
    assert hub.api.calls[0][1]["filter"] == "gguf"


def test_repo_files_and_errors(tmp_path):
    hub, _ = make_hub(tmp_path)
    files = hub.repo_files("unsloth/Qwen3.8-27B-GGUF")
    assert files.gguf() == [
        ("Qwen3.8-27B-UD-Q4_K_XL.gguf", int(17.2 * GIB)),
        ("mmproj-F16.gguf", int(0.8 * GIB)),
    ]
    assert files.has("README.md")
    with pytest.raises(HubError, match="No such repository"):
        hub.repo_files("nobody/nothing")


def test_lineage_rules(tmp_path):
    hub, _ = make_hub(tmp_path)
    assert hub.lineage("unsloth/Qwen3.8-27B-GGUF") == Lineage("vendor", "Qwen/Qwen3.8-27B")
    heretic_lineage = Lineage("derivative", "DavidAU/Qwen3.6-27B-Heretic", "Qwen/Qwen3.6-27B")
    assert hub.lineage("DavidAU/Qwen3.6-27B-Heretic-GGUF") == heretic_lineage
    assert hub.lineage("someone/mystery-GGUF") == Lineage("unknown")
    missing = fake_model("x/base-GGUF", tags=["base_model:quantized:gone/base"])
    hub.api.models["x/base-GGUF"] = missing
    assert hub.lineage("x/base-GGUF") == Lineage("unknown", "gone/base")


def test_model_card_preset_ini_and_download(tmp_path):
    served = {}

    def download_fn(repo_id, filename, **kwargs):
        target = tmp_path / "dl" / repo_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(served.get(filename, f"content of {filename}"))
        return str(target)

    served["README.md"] = "temperature=0.6"
    served["preset.ini"] = "[*]\ntemp = 1.0\n"

    def cached_path_fn(repo, name):
        return str(tmp_path / "dl" / repo / name) if name == "README.md" else None

    hub = Hub(
        api=FakeApi(MODELS, {"unsloth/Qwen3.8-27B-GGUF": {"preset.ini"}}),
        download_fn=download_fn, cached_path_fn=cached_path_fn,
    )
    assert hub.model_card("unsloth/Qwen3.8-27B-GGUF") == "temperature=0.6"
    assert hub.preset_ini("unsloth/Qwen3.8-27B-GGUF") == "[*]\ntemp = 1.0\n"
    assert hub.preset_ini("lmstudio-community/Qwen3.8-27B-GGUF") is None
    seen = []
    paths = hub.download(
        "unsloth/Qwen3.8-27B-GGUF", ["a.gguf", "mmproj-F16.gguf"], progress=seen.append
    )
    assert [p.name for p in paths] == ["a.gguf", "mmproj-F16.gguf"]
    assert seen == ["a.gguf", "mmproj-F16.gguf"]
    expected_readme = tmp_path / "dl" / "unsloth/Qwen3.8-27B-GGUF" / "README.md"
    assert hub.cached_path("unsloth/Qwen3.8-27B-GGUF", "README.md") == expected_readme
    assert hub.cached_path("unsloth/Qwen3.8-27B-GGUF", "a.gguf") is None


def test_download_gated_message(tmp_path):
    import httpx
    from huggingface_hub.errors import GatedRepoError

    def gated(repo_id, filename, **kwargs):
        response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))
        raise GatedRepoError("403", response=response)

    hub = Hub(api=FakeApi(MODELS), download_fn=gated)
    with pytest.raises(HubError, match="gated.*hf auth login"):
        hub.download("unsloth/Qwen3.8-27B-GGUF", ["a.gguf"])


def test_free_disk_bytes_walks_up_to_an_existing_parent(tmp_path):
    assert free_disk_bytes(tmp_path / "not" / "yet") > 0
    assert free_disk_bytes(Path(tmp_path)) > 0


def test_lineage_name_rule_catches_modified_uploads_tagged_as_quantizations(tmp_path):
    hub, _ = make_hub(tmp_path)
    hub.api.models["Qwen/Qwen3.6-35B-A3B"] = fake_model("Qwen/Qwen3.6-35B-A3B")
    hub.api.models["HauhauCS/Qwen3.6-35B-A3B-Uncensored-Aggressive"] = fake_model(
        "HauhauCS/Qwen3.6-35B-A3B-Uncensored-Aggressive",
        tags=["gguf", "base_model:quantized:Qwen/Qwen3.6-35B-A3B"],
    )
    assert hub.lineage("HauhauCS/Qwen3.6-35B-A3B-Uncensored-Aggressive") == Lineage(
        "derivative", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B"
    )
    hub.api.models["google/gemma-3-12b-it"] = fake_model("google/gemma-3-12b-it")
    hub.api.models["bartowski/google_gemma-3-12b-it-GGUF"] = fake_model(
        "bartowski/google_gemma-3-12b-it-GGUF",
        tags=["gguf", "base_model:quantized:google/gemma-3-12b-it"],
    )
    assert hub.lineage("bartowski/google_gemma-3-12b-it-GGUF").kind == "vendor"
    hub.api.models["mradermacher/Qwen3.6-35B-A3B-i1-GGUF"] = fake_model(
        "mradermacher/Qwen3.6-35B-A3B-i1-GGUF",
        tags=["gguf", "base_model:quantized:Qwen/Qwen3.6-35B-A3B"],
    )
    assert hub.lineage("mradermacher/Qwen3.6-35B-A3B-i1-GGUF").kind == "vendor"


def test_extra_tokens_ignores_packaging_words():
    assert extra_tokens("unsloth/Qwen3.8-27B-GGUF", "Qwen/Qwen3.8-27B") == set()
    assert extra_tokens("TheBloke/Llama-2-7B-Chat-GGUF", "meta-llama/Llama-2-7b-chat-hf") == set()
    heretic = extra_tokens("x/Qwen2.5-0.5B-Instruct-heretic", "Qwen/Qwen2.5-0.5B-Instruct")
    assert heretic == {"heretic"}
    assert extra_tokens("x/Model-Q4_K_M-imatrix-GGUF", "org/Model") == set()


def test_download_gives_up_when_the_file_stops_growing(tmp_path, monkeypatch):
    import time as _time

    def never_finishes(repo_id, filename, **kwargs):
        _time.sleep(5)
        return "/never"

    hub = Hub(api=FakeApi(MODELS), download_fn=never_finishes)
    monkeypatch.setattr(hub, "cache_dir", lambda: tmp_path)
    with pytest.raises(HubError, match=r"(?s)stalled.*hf auth login"):
        hub.download("unsloth/Qwen3.8-27B-GGUF", ["a.gguf"], stall_seconds=0.3)


def test_download_keeps_waiting_while_the_file_grows(tmp_path, monkeypatch):
    import time as _time

    blobs = tmp_path / "models--unsloth--Qwen3.8-27B-GGUF" / "blobs"
    blobs.mkdir(parents=True)
    part = blobs / "abc.incomplete"

    def slow_but_alive(repo_id, filename, **kwargs):
        for step in range(6):
            part.write_bytes(b"x" * (step + 1))
            _time.sleep(0.2)
        return str(tmp_path / "done.gguf")

    hub = Hub(api=FakeApi(MODELS), download_fn=slow_but_alive)
    monkeypatch.setattr(hub, "cache_dir", lambda: tmp_path)
    result = hub.download("unsloth/Qwen3.8-27B-GGUF", ["a.gguf"], stall_seconds=0.8)
    assert result == [tmp_path / "done.gguf"]


def test_plain_http_downloads_are_the_default():
    import os

    import local_llm.hub  # noqa: F401 - importing sets the default

    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
