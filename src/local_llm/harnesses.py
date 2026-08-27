"""Every coding agent the tool knows: how to spot it, and what configuring it means."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from .integrations import HarnessContext
from .integrations import codex as codex_integration
from .integrations import opencode as opencode_integration
from .settings import Settings

PROVIDER = "provider"
LAUNCHER = "launcher"
INFORMATIONAL = "informational"

GROUP_TITLES = {
    PROVIDER: "Configured inside the agent",
    LAUNCHER: "Launched through local-llm",
    INFORMATIONAL: "Detected, but cannot use the router",
}


@dataclass(frozen=True)
class Harness:
    key: str
    title: str
    binary: str
    kind: str
    summary: str
    note: str = ""
    alias: tuple[str, str] | None = None
    configure: Callable[[HarnessContext], list[str]] | None = None
    status: Callable[[HarnessContext], str] | None = None


GEMINI_NOTE = (
    "Gemini CLI cannot be pointed at the router: it has no setting for an"
    " OpenAI-compatible endpoint, and its GOOGLE_GEMINI_BASE_URL variable needs a"
    " server speaking Google's own request format, which llama-server does not."
    " Qwen Code continues the same codebase and does run local models:"
    " local-llm qwen"
)
ANTIGRAVITY_NOTE = (
    "Antigravity CLI is Google's replacement for Gemini CLI. It also takes only a"
    " Gemini-format endpoint, so it would need a translating server in front of the"
    " router, which this tool does not ship. To drive a local model from Antigravity,"
    " its Python SDK supports it officially: pip install google-antigravity, then"
    " LocalOpenAIAgentConfig(base_url=..., model=...)"
)

REGISTRY: tuple[Harness, ...] = (
    Harness(
        key="codex",
        title="OpenAI Codex CLI",
        binary="codex",
        kind=PROVIDER,
        summary="~/.codex/config.toml: provider and profile local-llm (experimental)",
        alias=("codex_local", "codex --profile local-llm"),
        configure=codex_integration.configure,
        status=codex_integration.harness_status,
    ),
    Harness(
        key="opencode",
        title="opencode",
        binary="opencode",
        kind=PROVIDER,
        summary="plugin listing every model, and a tiny helper agent",
        configure=partial(opencode_integration.configure, agent=True),
        status=opencode_integration.harness_status,
    ),
    Harness(
        key="claude",
        title="Claude Code",
        binary="claude",
        kind=LAUNCHER,
        summary="local-llm claude",
        alias=("claude_local", "local-llm claude"),
    ),
    Harness(
        key="copilot",
        title="GitHub Copilot CLI",
        binary="copilot",
        kind=LAUNCHER,
        summary="local-llm copilot",
        alias=("copilot_local", "local-llm copilot"),
    ),
    Harness(
        key="aider",
        title="aider",
        binary="aider",
        kind=LAUNCHER,
        summary="local-llm aider",
        alias=("aider_local", "local-llm aider"),
    ),
    Harness(
        key="qwen",
        title="Qwen Code",
        binary="qwen",
        kind=LAUNCHER,
        summary="local-llm qwen",
        alias=("qwen_local", "local-llm qwen"),
    ),
    Harness(
        key="gemini",
        title="Gemini CLI",
        binary="gemini",
        kind=INFORMATIONAL,
        summary="speaks only Google's own API format, which the router does not serve",
        note=GEMINI_NOTE,
    ),
    Harness(
        key="antigravity",
        title="Antigravity CLI",
        binary="agy",
        kind=INFORMATIONAL,
        summary="speaks only Google's own API format, which the router does not serve",
        note=ANTIGRAVITY_NOTE,
    ),
)


def find(key: str) -> Harness | None:
    return next((harness for harness in REGISTRY if harness.key == key), None)


def detect(
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[list[Harness], list[Harness]]:
    """(installed, missing), in registry order. Names on PATH only, nothing is run."""
    installed: list[Harness] = []
    missing: list[Harness] = []
    for harness in REGISTRY:
        (installed if which(harness.binary) else missing).append(harness)
    return installed, missing


def numbered(installed: Sequence[Harness]) -> list[Harness]:
    """The rows a person can actually choose."""
    return [h for h in installed if h.kind in (PROVIDER, LAUNCHER)]


def note_lines(settings: Settings, installed: Sequence[Harness]) -> list[str]:
    """The one warning the menu owes a person before it writes another tool's config."""
    if settings.is_local or not numbered(installed):
        return []
    return [
        "",
        f"  Note: {settings.host} is not a loopback address, so configuring an agent"
        " writes that address into its config file, where anything reading that file"
        " can see it.",
    ]


def safe_status(harness: Harness, status_of: Callable[[Harness], str]) -> str:
    """A harness that cannot say how it is configured still gets a row."""
    if harness.status is None:
        return ""
    try:
        return status_of(harness)
    except Exception:  # a menu must render whatever the disk is doing
        return "unknown"


def _status_note(harness: Harness, state: str) -> str:
    if harness.kind == PROVIDER:
        return {
            "same": "configured",
            "different": "configured, but differs from ours",
            "unreadable": "its config file does not parse",
            "unknown": "cannot tell — its config file could not be read",
        }.get(state, "not configured")
    if harness.kind == LAUNCHER:
        alias = harness.alias[0] if harness.alias else ""
        return f"alias {alias} installed" if state == "same" else f"alias {alias}"
    return ""


def render(
    installed: Sequence[Harness],
    missing: Sequence[Harness],
    status_of: Callable[[Harness], str],
) -> list[str]:
    """The whole menu as lines, ready to print. Numbering matches numbered()."""
    lines: list[str] = []
    choices = numbered(installed)
    if not installed:
        lines.append("  No coding agents found on PATH.")
        lines.append("  local-llm env prints the exports for any other tool.")
    for kind in (PROVIDER, LAUNCHER, INFORMATIONAL):
        group = [h for h in installed if h.kind == kind]
        if not group:
            continue
        lines.append("")
        lines.append(f"  {GROUP_TITLES[kind]}")
        for harness in group:
            note = _status_note(harness, status_of(harness))
            tail = f"   · {note}" if note else ""
            if kind == INFORMATIONAL:
                lines.append(f"      {harness.title:<20} {harness.summary}")
            else:
                number = choices.index(harness) + 1
                lines.append(f"   {number}. {harness.title:<20} {harness.summary}{tail}")
    if missing:
        lines.append("")
        lines.append("  Not found: " + ", ".join(h.title for h in missing))
    return lines


def parse_choice(answer: str, rows: Sequence[Harness]) -> list[Harness]:
    """'1 3' picks two, 'a' picks all, 'n' or empty picks none. Anything else raises."""
    answer = answer.strip().lower()
    if answer in ("", "n", "no", "none"):
        return []
    if answer in ("a", "all"):
        return list(rows)
    picked: list[Harness] = []
    for token in answer.split():
        if not token.isdigit():
            raise ValueError(f"Pick numbers between 1 and {len(rows)}, a for all, n for none")
        number = int(token)
        if number < 1 or number > len(rows):
            raise ValueError(f"Pick numbers between 1 and {len(rows)}, a for all, n for none")
        if rows[number - 1] not in picked:
            picked.append(rows[number - 1])
    return picked
