"""What to recommend: computed from this machine and live Hub data, never from a list."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .estimate import estimate_bytes
from .hardware import Machine
from .hub import Hub, HubError, Lineage, RepoListing, base_models_of
from .preset import Preset
from .quant import QuantOption, pick_mmproj, quant_options, suggest

GROUPS = ("coding", "general", "small", "vision")
CODING_WORDS = ("coder", "code", "devstral")
SMALL_PARAMS = 5_000_000_000
DEFAULT_BASES = 40  # distinct models examined per run; each costs a few Hub calls (cached a day)
_FIT_ORDER = {"comfortable": 0, "fits": 1, "too_big": 2, "unknown": 3}


@dataclass
class Candidate:
    repo_id: str
    base_model: str | None
    lineage: str
    downloads: int
    likes: int
    params: int
    context_length: int
    groups: list[str]
    thinking: bool
    vision: bool
    gated: bool
    options: list[QuantOption] = field(default_factory=list)
    suggested: QuantOption | None = None
    fit: str = "unknown"
    estimate: int = 0
    also_from: list[str] = field(default_factory=list)
    downloaded: bool = False
    configured: bool = False

    @property
    def display_name(self) -> str:
        return (self.base_model or self.repo_id).split("/")[-1]


def is_thinking(listing: RepoListing) -> bool:
    return "<think>" in listing.chat_template or "enable_thinking" in listing.chat_template


def classify(listing: RepoListing, base_model: str | None, has_mmproj: bool) -> list[str]:
    name = (base_model or listing.repo_id).lower()
    groups: list[str] = []
    if has_mmproj or listing.pipeline_tag == "image-text-to-text":
        groups.append("vision")
    coding = any(word in name for word in CODING_WORDS)
    small = 0 < listing.params < SMALL_PARAMS
    if coding:
        groups.append("coding")
    if small:
        groups.append("small")
    if not coding and not small:
        groups.append("general")
    return groups


def listings_for(hub: Hub) -> list[RepoListing]:
    """Popular now, trending now, and vision models; merged, first occurrence wins."""
    seen: dict[str, RepoListing] = {}
    batches = (
        hub.list_gguf(sort="downloads", limit=300),
        hub.list_gguf(sort="trending_score", limit=100),
        hub.list_gguf(sort="downloads", limit=100, pipeline_tag="image-text-to-text"),
    )
    for batch in batches:
        for listing in batch:
            seen.setdefault(listing.repo_id, listing)
    return list(seen.values())


def _by_base(listings: list[RepoListing]) -> list[list[RepoListing]]:
    """Group quantizer repos by the model they quantize, most downloaded first."""
    groups: dict[str, list[RepoListing]] = {}
    for listing in sorted(listings, key=lambda item: -item.downloads):
        bases = base_models_of(listing)
        key = (bases[0] if bases else listing.repo_id).lower()
        groups.setdefault(key, []).append(listing)
    return list(groups.values())


def build_candidate(
    hub: Hub, machine: Machine, preset: Preset | None, members: list[RepoListing], *,
    with_files: bool = True, skip_derivatives: bool = False,
) -> Candidate | None:
    listing = members[0]
    try:
        lineage = hub.lineage(listing.repo_id, listing)
    except HubError:
        lineage = Lineage("unknown")
    if skip_derivatives and lineage.kind == "derivative":
        return None

    options: list[QuantOption] = []
    suggested: QuantOption | None = None
    fit_kind, estimate, has_mmproj, gated = "unknown", 0, False, listing.gated
    if with_files:
        try:
            files = hub.repo_files(listing.repo_id)
        except HubError:
            return None
        options = quant_options(files.files)
        if not options:
            return None
        gated = gated or files.gated
        has_mmproj = pick_mmproj(files.files) is not None
        suggested, fit_kind = suggest(options, machine.budget)
        estimate = estimate_bytes([suggested.total]) if suggested else 0

    candidate = Candidate(
        repo_id=listing.repo_id,
        base_model=lineage.base_model,
        lineage=lineage.kind,
        downloads=listing.downloads,
        likes=listing.likes,
        params=listing.params,
        context_length=listing.context_length,
        groups=classify(listing, lineage.base_model, has_mmproj),
        thinking=is_thinking(listing),
        vision=has_mmproj or listing.pipeline_tag == "image-text-to-text",
        gated=gated,
        options=options,
        suggested=suggested,
        fit=fit_kind,
        estimate=estimate,
        also_from=[member.repo_id for member in members[1:]],
    )
    if suggested is not None:
        candidate.downloaded = hub.cached_path(listing.repo_id, suggested.primary) is not None
        candidate.configured = preset is not None and any(
            section.rsplit(":", 1)[0] == listing.repo_id for section in preset.sections()
        )
    return candidate


def rank(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates, key=lambda c: (_FIT_ORDER.get(c.fit, 3), -c.params, -c.downloads)
    )


def gather(
    hub: Hub, machine: Machine, preset: Preset | None = None, *, include_finetunes: bool = False,
    limit_per_group: int = 3, max_bases: int = DEFAULT_BASES,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, list[Candidate]]:
    candidates: list[Candidate] = []
    for members in _by_base(listings_for(hub))[:max_bases]:
        if on_progress:
            on_progress(members[0].repo_id)
        candidate = build_candidate(
            hub, machine, preset, members, skip_derivatives=not include_finetunes
        )
        if candidate is not None:
            candidates.append(candidate)
    return {
        group: rank([c for c in candidates if group in c.groups])[:limit_per_group]
        for group in GROUPS
    }


def search(
    hub: Hub, machine: Machine, preset: Preset | None = None, *, text: str, limit: int = 20,
    files_for: int = 10, author: str | None = None,
) -> list[Candidate]:
    results: list[Candidate] = []
    listings = hub.list_gguf(sort="downloads", limit=limit, search=text, author=author)
    for index, listing in enumerate(listings):
        candidate = build_candidate(hub, machine, preset, [listing], with_files=index < files_for)
        if candidate is not None:
            results.append(candidate)
    return results
