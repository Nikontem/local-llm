from local_llm.discover import Candidate, classify, gather, rank, search
from local_llm.estimate import GIB
from local_llm.hardware import Machine
from local_llm.hub import Hub, RepoListing
from local_llm.preset import Preset

from .fakes import FakeApi, fake_model

# Apple M4 Pro, 48 GiB RAM -> 38 GiB budget
MAC = Machine("Darwin", "arm64", "Apple M4 Pro", 48 * GIB, "apple", 48 * GIB, 20, 10)
Q = "base_model:quantized:"
MODELS = {m.id: m for m in [
    fake_model("Qwen/Qwen3.8-27B", tags=["base_model:finetune:Qwen/Qwen3.8-27B-Base"]),
    fake_model("Qwen/Qwen3.8-27B-Base"),
    fake_model("Qwen/Qwen3-Coder-30B-A3B-Instruct"),
    fake_model("Qwen/Qwen2.5-1.5B-Instruct"),
    fake_model("google/gemma-4-31B-it"),
    fake_model("DavidAU/Remix", tags=["base_model:merge:Qwen/Qwen3.8-27B"]),
    fake_model(
        "unsloth/Qwen3.8-27B-GGUF", downloads=7_000_000, likes=2900, trending=90,
        tags=["gguf", Q + "Qwen/Qwen3.8-27B"],
        gguf={"total": 27e9, "context_length": 262144, "chat_template": "<think>"},
        files=[
            ("Qwen3.8-27B-UD-Q4_K_XL.gguf", int(17.2 * GIB)),
            ("Qwen3.8-27B-Q8_0.gguf", int(29 * GIB)),
            ("mmproj-F16.gguf", int(0.8 * GIB)),
        ],
    ),
    fake_model(
        "lmstudio-community/Qwen3.8-27B-GGUF", downloads=1_900_000,
        tags=["gguf", Q + "Qwen/Qwen3.8-27B"], gguf={"total": 27e9},
        files=[("Qwen3.8-27B-Q4_K_M.gguf", int(16.5 * GIB))],
    ),
    fake_model(
        "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", downloads=12_000_000,
        tags=["gguf", Q + "Qwen/Qwen3-Coder-30B-A3B-Instruct"],
        gguf={"total": 30e9, "context_length": 262144},
        files=[("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", int(16.5 * GIB))],
    ),
    fake_model(
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF", downloads=900_000,
        tags=["gguf", Q + "Qwen/Qwen2.5-1.5B-Instruct"],
        gguf={"total": 1.5e9, "context_length": 32768},
        files=[
            ("qwen2.5-1.5b-instruct-q4_k_m.gguf", GIB),
            ("qwen2.5-1.5b-instruct-q8_0.gguf", int(1.6 * GIB)),
        ],
    ),
    fake_model(
        "unsloth/gemma-4-31B-it-GGUF", downloads=3_000_000, trending=200,
        pipeline_tag="image-text-to-text",
        tags=["gguf", Q + "google/gemma-4-31B-it"],
        gguf={"total": 31e9, "context_length": 262144},
        files=[
            ("gemma-4-31B-it-UD-Q4_K_XL.gguf", int(17.5 * GIB)),
            ("mmproj-F16.gguf", int(0.9 * GIB)),
        ],
    ),
    fake_model(
        "DavidAU/Remix-GGUF", downloads=5_000_000,
        tags=["gguf", Q + "DavidAU/Remix"], gguf={"total": 27e9},
        files=[("Remix-Q4_K_M.gguf", int(16 * GIB))],
    ),
    fake_model(
        "huge/Model-405B-GGUF", downloads=50_000,
        tags=["gguf", Q + "huge/Model-405B"], gguf={"total": 405e9},
        files=[("Model-405B-Q4_K_M.gguf", int(230 * GIB))],
    ),
    fake_model("huge/Model-405B"),
]}


def make_hub(cached=()):
    return Hub(
        api=FakeApi(MODELS),
        cached_path_fn=lambda repo, name: f"/hf/{name}" if (repo, name) in cached else None,
    )


def test_classify_from_data_not_names_of_models():
    small = RepoListing("x/y", params=1_000_000_000)
    assert classify(small, "Qwen/Qwen2.5-1.5B-Instruct", False) == ["small"]
    coder = RepoListing("x/y", params=30e9)
    assert classify(coder, "Qwen/Qwen3-Coder-30B-A3B-Instruct", False) == ["coding"]
    vision = RepoListing("x/y", params=31e9, pipeline_tag="image-text-to-text")
    assert classify(vision, "google/gemma-4-31B-it", True) == ["vision", "general"]
    assert classify(RepoListing("x/y", params=7e9), "mistralai/Devstral-Small", False) == [
        "coding"
    ]
    # params unknown -> general
    assert classify(RepoListing("x/y"), "unknown/thing", False) == ["general"]


def test_gather_groups_ranks_and_hides_derivatives():
    preset = Preset.parse(
        "[unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL]\nmodel = /x.gguf\n"
    )
    hub = make_hub(cached={
        (
            "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        )
    })
    groups = gather(hub, MAC, preset)
    assert set(groups) == {"coding", "general", "small", "vision"}
    coding = groups["coding"]
    assert [c.repo_id for c in coding] == ["unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"]
    assert coding[0].suggested.label == "UD-Q4_K_XL" and coding[0].fit == "comfortable"
    assert coding[0].lineage == "vendor" and coding[0].downloaded and coding[0].configured
    general = groups["general"]
    assert [c.repo_id for c in general] == [
        "unsloth/gemma-4-31B-it-GGUF", "unsloth/Qwen3.8-27B-GGUF", "huge/Model-405B-GGUF"
    ]
    qwen = general[1]
    assert qwen.thinking and qwen.vision
    assert qwen.also_from == ["lmstudio-community/Qwen3.8-27B-GGUF"]
    assert qwen.display_name == "Qwen3.8-27B" and qwen.suggested.label == "UD-Q4_K_XL"
    assert general[2].fit == "too_big" and general[2].suggested.label == "Q4_K_M"
    assert [c.repo_id for c in groups["small"]] == ["Qwen/Qwen2.5-1.5B-Instruct-GGUF"]
    # better quality is cheap for a small model
    assert groups["small"][0].suggested.label == "Q8_0"
    assert [c.repo_id for c in groups["vision"]] == [
        "unsloth/gemma-4-31B-it-GGUF", "unsloth/Qwen3.8-27B-GGUF"
    ]
    assert all(c.lineage != "derivative" for group in groups.values() for c in group)


def test_gather_include_finetunes_limit_and_progress():
    hub = make_hub()
    seen = []
    groups = gather(
        hub, MAC, None, include_finetunes=True, limit_per_group=1, on_progress=seen.append
    )
    assert len(groups["general"]) == 1
    assert "DavidAU/Remix-GGUF" in seen
    all_groups = gather(hub, MAC, None, include_finetunes=True, limit_per_group=10)
    remix = next(c for c in all_groups["general"] if c.repo_id == "DavidAU/Remix-GGUF")
    assert remix.lineage == "derivative" and remix.base_model == "DavidAU/Remix"


def test_rank_orders_by_fit_then_size_then_downloads():
    def cand(repo, fit, params, downloads):
        return Candidate(
            repo, None, "vendor", downloads, 0, params, 0, ["general"], False, False, False,
            fit=fit,
        )

    ranked = rank([
        cand("a", "fits", 70e9, 1),
        cand("b", "comfortable", 27e9, 1),
        cand("c", "comfortable", 27e9, 9),
        cand("d", "too_big", 400e9, 1),
    ])
    assert [c.repo_id for c in ranked] == ["c", "b", "a", "d"]


def test_search_keeps_every_lineage_and_fetches_files_only_for_the_first_n():
    hub = make_hub()
    results = search(hub, MAC, None, text="Qwen3.8", limit=10, files_for=1)
    assert [c.repo_id for c in results] == [
        "unsloth/Qwen3.8-27B-GGUF", "lmstudio-community/Qwen3.8-27B-GGUF"
    ]
    assert results[0].suggested is not None
    assert results[1].suggested is None and results[1].fit == "unknown"
    remix = search(hub, MAC, None, text="Remix", limit=5)
    assert remix[0].lineage == "derivative"


def test_non_text_pipelines_are_skipped():
    from local_llm.discover import build_candidate

    hub = make_hub()
    speech = hub.api.models["handy/whisper-GGUF"] = fake_model(
        "handy/whisper-GGUF", downloads=10, pipeline_tag="automatic-speech-recognition",
        tags=["gguf"], gguf={"total": 1e9}, files=[("whisper-Q8_0.gguf", GIB)],
    )
    listing = hub.repo_meta(speech.id)
    assert build_candidate(hub, MAC, None, [listing]) is None


def test_files_too_small_for_the_parameter_count_are_not_offered():
    from local_llm.discover import build_candidate

    hub = make_hub()
    partial = hub.api.models["x/huge-partial-GGUF"] = fake_model(
        "x/huge-partial-GGUF", downloads=10, tags=["gguf", Q + "x/huge-partial"],
        gguf={"total": 100e9},
        files=[("huge-partial-0731.gguf", 5 * GIB), ("huge-partial-Q4_K_M.gguf", 55 * GIB)],
    )
    hub.api.models["x/huge-partial"] = fake_model("x/huge-partial")
    candidate = build_candidate(hub, MAC, None, [hub.repo_meta(partial.id)])
    assert [o.label for o in candidate.options] == ["Q4_K_M"]
