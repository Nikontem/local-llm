"""Sampling values for a model: the repo's preset.ini, else its model card, else a family table."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources

from .preset import Preset

KEYS = ["temp", "top-k", "top-p", "min-p", "presence-penalty", "repeat-penalty"]
ALIASES: dict[str, list[str]] = {
    "temp": ["temperature", "temp"],
    "top-p": ["top_p", "top-p", "top p", "topp"],
    "top-k": ["top_k", "top-k", "top k", "topk"],
    "min-p": ["min_p", "min-p", "min p", "minp"],
    "presence-penalty": ["presence_penalty", "presence-penalty", "presence penalty"],
    "repeat-penalty": [
        "repeat_penalty", "repeat-penalty", "repeat penalty",
        "repetition_penalty", "repetition-penalty", "repetition penalty",
    ],
}
RANGES: dict[str, tuple[float, float]] = {
    "temp": (0, 2), "top-p": (0, 1), "min-p": (0, 1), "top-k": (0, 1000),
    "presence-penalty": (0, 3), "repeat-penalty": (0, 3),
}
_PRESET_SKIP = {"model", "hf", "hf-repo", "hf-file", "alias", "host", "port", "api-key", "mmproj"}
_NUMBER = r"`?([0-9]*\.?[0-9]+)`?"


@dataclass
class SamplingResult:
    values: dict[str, str] = field(default_factory=dict)
    source: str = "none"


def _pattern(alias: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"\s*(?:=|:|\|)\s*" + _NUMBER, re.IGNORECASE)


def parse_model_card(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for key, aliases in ALIASES.items():
        low, high = RANGES[key]
        best: tuple[int, str] | None = None
        for alias in aliases:
            for match in _pattern(alias).finditer(text):
                try:
                    number = float(match.group(1))
                except ValueError:
                    continue
                if not (low <= number <= high):
                    continue
                if best is None or match.start() < best[0]:
                    best = (match.start(), match.group(1))
                break  # first plausible hit of this alias; earlier aliases may still win on position
        if best is not None:
            found[key] = best[1]
    return found


def parse_preset_ini(text: str, model_hint: str) -> dict[str, str]:
    try:
        preset = Preset.parse(text)
    except Exception:  # noqa: BLE001 - a broken preset.ini simply contributes nothing
        return {}
    names = preset.sections()
    hint = model_hint.lower()
    chosen = next(
        (n for n in names if hint and (hint in n.lower() or hint in (preset.get(n, "hf", False) or "").lower())),
        names[0] if names else None,
    )
    values = {**preset.items("*"), **(preset.items(chosen) if chosen else {})}
    return {key: value for key, value in values.items() if key not in _PRESET_SKIP}


def load_profiles() -> dict:
    return json.loads(resources.files("local_llm").joinpath("sampling.json").read_text())


def profile_for(repo_id: str, table: dict | None = None) -> tuple[str, dict[str, str]] | None:
    table = table or load_profiles()
    name = repo_id.lower()
    for rule in table["rules"]:
        if all(word.lower() in name for word in rule["contains"]):
            return rule["profile"], dict(table["profiles"][rule["profile"]]["values"])
    return None


def values_for(
    repo_id: str, *, preset_ini: str | None = None, card: str = "", no_tuning: bool = False,
    table: dict | None = None,
) -> SamplingResult:
    if no_tuning:
        return SamplingResult({}, "none")
    if preset_ini:
        values = parse_preset_ini(preset_ini, repo_id.split("/")[-1])
        if values:
            return SamplingResult(values, "preset.ini in the repo")
    values = parse_model_card(card)
    if values:
        return SamplingResult(values, "model card")
    profile = profile_for(repo_id, table)
    if profile:
        return SamplingResult(profile[1], f"profile {profile[0]}")
    return SamplingResult({}, "none")
