"""Talking to the Hugging Face Hub: token, listings, files, lineage, cards, downloads."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenStatus:
    state: str  # "valid" | "invalid" | "absent" | "unreachable"
    username: str | None = None


def token_status(timeout: float = 10.0) -> TokenStatus:
    """Validate the locally stored token against the Hub, without hanging."""
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    token = huggingface_hub.get_token()
    if not token:
        return TokenStatus("absent")

    # Run the probe on a daemon thread rather than a ThreadPoolExecutor: the pool's
    # worker is a non-daemon thread that concurrent.futures joins at interpreter
    # exit, so shutdown(wait=False) does not stop a blackholed whoami() call from
    # stalling process exit long past `timeout`. A daemon thread is abandoned
    # cleanly when the process exits, whether or not it ever finishes.
    done = threading.Event()
    outcome: dict[str, object] = {}

    def whoami() -> None:
        try:
            outcome["info"] = huggingface_hub.HfApi(token=token).whoami()
        except Exception as error:  # noqa: BLE001 - forwarded to the caller thread below
            outcome["error"] = error
        finally:
            done.set()

    threading.Thread(target=whoami, daemon=True).start()
    if not done.wait(timeout):
        return TokenStatus("unreachable")

    if "error" in outcome:
        error = outcome["error"]
        if isinstance(error, HfHubHTTPError):
            response = getattr(error, "response", None)
            if response is not None and getattr(response, "status_code", None) in (401, 403):
                return TokenStatus("invalid")
        return TokenStatus("unreachable")

    info = outcome.get("info")
    return TokenStatus("valid", info.get("name") if isinstance(info, dict) else None)


STALL_SECONDS = 180.0  # a download that has not grown for this long is reported, not waited on
LIST_EXPAND = [
    "author", "downloads", "likes", "trendingScore", "pipeline_tag", "gated", "tags",
    "baseModels", "gguf",
]


class HubError(Exception):
    pass


class HubCache:
    """A JSON file of {key: {"at": timestamp, "value": ...}} with a time-to-live."""

    def __init__(
        self, path: Path, ttl: float = 86400.0, now: Callable[[], float] = time.time
    ) -> None:
        self.path = path
        self.ttl = ttl
        self._now = now
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def get(self, key: str) -> Any | None:
        entry = self._load().get(key)
        if not entry or self._now() - entry.get("at", 0) > self.ttl:
            return None
        return entry.get("value")

    def put(self, key: str, value: Any) -> None:
        data = self._load()
        data[key] = {"at": self._now(), "value": value}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    def clear(self) -> None:
        self._data = {}
        self.path.unlink(missing_ok=True)


@dataclass
class RepoListing:
    repo_id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    trending: int = 0
    pipeline_tag: str | None = None
    gated: bool = False
    tags: list[str] = field(default_factory=list)
    base_models: list[str] = field(default_factory=list)
    params: int = 0
    context_length: int = 0
    architecture: str = ""
    chat_template: str = ""


@dataclass
class RepoFiles:
    repo_id: str
    files: list[tuple[str, int]]
    gated: bool = False

    def gguf(self) -> list[tuple[str, int]]:
        return [(path, size) for path, size in self.files if path.lower().endswith(".gguf")]

    def has(self, name: str) -> bool:
        return any(path == name for path, _ in self.files)


@dataclass
class Lineage:
    kind: str  # "vendor" | "derivative" | "unknown"
    base_model: str | None = None
    parent: str | None = None


def _tagged(tags: list[str], relation: str | None) -> list[str]:
    """base_model:quantized:X -> X for relation "quantized"; plain base_model:X for None."""
    found: list[str] = []
    for tag in tags:
        if not tag.startswith("base_model:"):
            continue
        rest = tag[len("base_model:"):]
        if relation is None:
            if ":" not in rest:
                found.append(rest)
        elif rest.startswith(relation + ":"):
            found.append(rest[len(relation) + 1:])
    return found


_NAME_SPLIT = re.compile(r"[-_.\s/]+")
# Tokens that describe how a model was packaged, not what it is.
_QUANT_VOCAB = {
    "gguf", "imatrix", "i1", "ud", "bf16", "f16", "f32", "fp16", "fp32", "fp8", "mxfp4", "nvfp4",
    "quantized", "quant", "quants", "hf", "llamacpp", "llama", "cpp", "repack",
}
_QUANT_TOKEN = re.compile(r"^(i?q\d\w*|iq\d\w*|\d+(bit|bpw|b|k)?|k|m|s|l|xl|xs|xxs)$")


def name_tokens(name: str) -> set[str]:
    return {token for token in _NAME_SPLIT.split(name.lower()) if token}


def extra_tokens(repo_id: str, base_model: str) -> set[str]:
    """Words in a repo's name that the base model's name does not explain.

    A quantizer repeats the base name and adds packaging words (GGUF, imatrix,
    a quantization tag). Anything else — "Uncensored", "Heretic", "Aggressive" —
    means the weights were changed, however the uploader tagged the repo.
    """
    extra = name_tokens(repo_id.split("/")[-1]) - name_tokens(base_model)
    return {t for t in extra if t not in _QUANT_VOCAB and not _QUANT_TOKEN.match(t)}


def base_models_of(listing: RepoListing) -> list[str]:
    return (
        _tagged(listing.tags, "quantized")
        or list(listing.base_models)
        or _tagged(listing.tags, None)
    )


def _listing_from_info(info) -> RepoListing:
    gguf = getattr(info, "gguf", None) or {}
    return RepoListing(
        repo_id=info.id,
        author=getattr(info, "author", None) or info.id.split("/")[0],
        downloads=int(getattr(info, "downloads", None) or 0),
        likes=int(getattr(info, "likes", None) or 0),
        trending=int(getattr(info, "trending_score", None) or 0),
        pipeline_tag=getattr(info, "pipeline_tag", None),
        gated=bool(getattr(info, "gated", False)),
        tags=list(getattr(info, "tags", None) or []),
        base_models=list(getattr(info, "base_models", None) or []),
        params=int(gguf.get("total") or 0),
        context_length=int(gguf.get("context_length") or 0),
        architecture=str(gguf.get("architecture") or ""),
        chat_template=str(gguf.get("chat_template") or ""),
    )


def _gated_message(repo_id: str) -> str:
    return (
        f"{repo_id} is gated: accept its terms on https://huggingface.co/{repo_id}"
        " and log in with: hf auth login"
    )


class Hub:
    def __init__(
        self, api=None, cache: HubCache | None = None, download_fn=None, cached_path_fn=None
    ) -> None:
        self.api = api
        self.cache = cache
        self._download_fn = download_fn
        self._cached_path_fn = cached_path_fn

    def _api(self):
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi()
        return self.api

    def _cached(self, key: str, compute: Callable[[], Any]) -> Any:
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        value = compute()
        if self.cache is not None:
            self.cache.put(key, value)
        return value

    def _model_info(self, repo_id: str, **kwargs):
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

        try:
            return self._api().model_info(repo_id, **kwargs)
        except GatedRepoError:
            raise HubError(_gated_message(repo_id)) from None
        except RepositoryNotFoundError:
            raise HubError(
                f"No such repository on huggingface.co: {repo_id}\n"
                "  local-llm search <text>    to find the right id"
            ) from None
        except HfHubHTTPError as error:
            raise HubError(f"huggingface.co returned an error for {repo_id}: {error}") from None
        except OSError as error:
            raise HubError(f"Could not reach huggingface.co: {error}") from None

    # ------------------------------------------------------------ listings and metadata

    def list_gguf(
        self, *, sort: str = "downloads", limit: int = 300, search: str | None = None,
        pipeline_tag: str | None = None, author: str | None = None,
    ) -> list[RepoListing]:
        key = f"list:{sort}:{limit}:{search or ''}:{pipeline_tag or ''}:{author or ''}"

        def compute() -> list[dict]:
            try:
                infos = self._api().list_models(
                    filter="gguf", search=search, pipeline_tag=pipeline_tag, author=author,
                    sort=sort, limit=limit, expand=LIST_EXPAND,
                )
                return [asdict(_listing_from_info(info)) for info in infos]
            except OSError as error:
                raise HubError(f"Could not list models on huggingface.co: {error}") from None

        return [RepoListing(**item) for item in self._cached(key, compute)]

    def repo_meta(self, repo_id: str) -> RepoListing:
        return RepoListing(**self._cached(
            f"meta:{repo_id}",
            lambda: asdict(_listing_from_info(self._model_info(repo_id, expand=LIST_EXPAND))),
        ))

    def repo_files(self, repo_id: str) -> RepoFiles:
        def compute() -> dict:
            info = self._model_info(repo_id, files_metadata=True)
            files = [
                (s.rfilename, int(getattr(s, "size", 0) or 0)) for s in (info.siblings or [])
            ]
            gated = bool(getattr(info, "gated", False))
            return {"repo_id": repo_id, "files": files, "gated": gated}

        data = self._cached(f"files:{repo_id}", compute)
        return RepoFiles(data["repo_id"], [tuple(item) for item in data["files"]], data["gated"])

    # ------------------------------------------------------------ lineage

    def lineage(self, repo_id: str, listing: RepoListing | None = None) -> Lineage:
        meta = listing or self.repo_meta(repo_id)
        bases = base_models_of(meta)
        if not bases:
            return Lineage("unknown")
        base = bases[0]
        if extra_tokens(repo_id, base):
            return Lineage("derivative", base, base)
        org = base.split("/")[0].lower()
        current = base
        for _ in range(3):
            try:
                node = self.repo_meta(current)
            except HubError:
                return Lineage("unknown", base)
            parents = _tagged(node.tags, "finetune") + _tagged(node.tags, "merge")
            if not parents:
                return Lineage("vendor", base)
            foreign = [p for p in parents if p.split("/")[0].lower() != org]
            if foreign:
                return Lineage("derivative", base, foreign[0])
            current = parents[0]
        return Lineage("vendor", base)

    # ------------------------------------------------------------ cards, presets, downloads

    def _fetch_text(self, repo_id: str, filename: str) -> str:
        try:
            path = self._download_or_real(repo_id, filename)
            return Path(path).read_text(errors="replace")
        except Exception:  # noqa: BLE001 - a missing card or preset is not an error
            return ""

    def model_card(self, repo_id: str) -> str:
        return self._cached(f"card:{repo_id}", lambda: self._fetch_text(repo_id, "README.md"))

    def preset_ini(self, repo_id: str) -> str | None:
        def compute() -> str:
            try:
                if not self._api().file_exists(repo_id, "preset.ini"):
                    return ""
            except Exception:  # noqa: BLE001
                return ""
            return self._fetch_text(repo_id, "preset.ini")

        return self._cached(f"preset:{repo_id}", compute) or None

    def _download_or_real(self, repo_id: str, filename: str):
        if self._download_fn is not None:
            return self._download_fn(repo_id=repo_id, filename=filename)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename)

    def download(
        self, repo_id: str, filenames: list[str], progress: Callable[[str], None] | None = None,
        stall_seconds: float = STALL_SECONDS,
    ) -> list[Path]:
        paths: list[Path] = []
        for name in filenames:
            if progress:
                progress(name)
            paths.append(self._download_with_watchdog(repo_id, name, stall_seconds))
        return paths

    def _download_with_watchdog(self, repo_id: str, name: str, stall_seconds: float) -> Path:
        """Run the download on a daemon thread and give up when the file stops growing.

        The Hub library retries quietly for a long time when a transfer is
        throttled or a CDN node stalls; a person deserves a message instead.
        """
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

        outcome: dict[str, object] = {}
        done = threading.Event()

        def work() -> None:
            try:
                outcome["path"] = self._download_or_real(repo_id, name)
            except Exception as error:  # noqa: BLE001 - forwarded to the caller thread below
                outcome["error"] = error
            finally:
                done.set()

        threading.Thread(target=work, daemon=True).start()
        blobs = self.cache_dir() / f"models--{repo_id.replace('/', '--')}" / "blobs"
        last_size, last_change = -1, time.monotonic()
        while not done.wait(1.0):
            size = _incomplete_bytes(blobs)
            if size != last_size:
                last_size, last_change = size, time.monotonic()
            elif time.monotonic() - last_change > stall_seconds:
                raise HubError(
                    f"Download of {repo_id}/{name} stalled: no data for {int(stall_seconds)} s"
                    f" ({size / 1e6:.0f} MB so far).\n"
                    "  Hugging Face throttles unauthenticated downloads; `hf auth login` lifts"
                    " the limit. The partial file is kept, so running the same command again"
                    " resumes."
                )
        if "error" in outcome:
            error = outcome["error"]
            if isinstance(error, GatedRepoError):
                raise HubError(_gated_message(repo_id)) from None
            if isinstance(error, HfHubHTTPError | OSError):
                raise HubError(f"Download of {repo_id}/{name} failed: {error}") from None
            raise HubError(f"Download of {repo_id}/{name} failed: {error}") from None
        return Path(str(outcome["path"]))

    def cache_dir(self) -> Path:
        return hf_cache_dir()

    def cached_path(self, repo_id: str, filename: str) -> Path | None:
        if self._cached_path_fn is not None:
            found = self._cached_path_fn(repo_id, filename)
            return Path(found) if found else None
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(repo_id, filename)
        return Path(found) if isinstance(found, str) else None


def _incomplete_bytes(blobs: Path) -> int:
    try:
        return sum(p.stat().st_size for p in blobs.glob("*.incomplete"))
    except OSError:
        return 0


def free_disk_bytes(path: Path) -> int:
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    return shutil.disk_usage(target).free


def hf_cache_dir() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)
