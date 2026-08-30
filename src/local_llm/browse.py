"""The model menu: what Hugging Face has, numbered, and what of it you take.

The wizard's third step and `local-llm browse-models` put the same question to the
person, so the loop that asks it lives here rather than in either of them. It
reaches the world outside only through the callbacks on `BrowseContext` - one to
print, one to ask, and three to fetch and to choose - which is what lets a test
drive a whole session of listing, searching and picking from a list of scripted
answers, with no terminal and no network.

The loop refuses an answer it cannot read and asks again rather than giving up:
someone half way through choosing a 17 GB download should not be sent back to the
shell over a mistyped number.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .discover import GROUPS, Candidate
from .estimate import human_gb
from .hardware import Machine
from .harnesses import parse_numbers
from .hub import Hub, HubError
from .preset import Preset
from .quant import QuantOption, quant_options, suggest

PROMPT = "  Numbers to download (e.g. 1 3), s <text> to search, n for none"

# What `parse_numbers` adds to its refusal, so the menu's other answers are named
# in the same breath as the numbers it would not read.
OTHER_ANSWERS = ", s <text> to search, n for none"


@dataclass
class BrowseContext:
    """Everything the menu needs, and nothing else.

    Deliberately not the wizard's `SetupContext`: the menu has no business with the
    router, the shell or the coding agents, and keeping it to these five callbacks
    is what stops the wizard and the standalone command from having to agree on
    anything larger than the question being asked.
    """

    say: Callable[[str], None]
    ask: Callable[[str, str], str]
    hub: Hub
    recommend: Callable[[Machine, Preset | None, str | None], dict[str, list[Candidate]]]
    search: Callable[[Machine, Preset | None, str], list[Candidate]]
    choose_quant: Callable[[list[QuantOption], QuantOption | None, Machine], QuantOption]


def candidate_line(candidate: Candidate, machine: Machine) -> str:
    """One numbered line: what the model is, what it costs, and whether you have it."""
    option = candidate.suggested
    marks = ", ".join(
        m for m, on in (("thinking", candidate.thinking), ("vision", candidate.vision)) if on
    )
    marks = f" ({marks})" if marks else ""
    state = (
        "  · in models.ini"
        if candidate.configured
        else ("  · downloaded" if candidate.downloaded else "")
    )
    if option is None:
        return f"{candidate.display_name}{marks}  {candidate.repo_id}"
    return (
        f"{candidate.display_name}{marks}  {candidate.repo_id}  {option.label}"
        f"  {human_gb(option.total)} on disk, ~{human_gb(candidate.estimate)} in memory"
        f"  {machine.fit_label(candidate.estimate)}{state}"
    )


def pick_models(
    ctx: BrowseContext, machine: Machine, preset: Preset | None, *, use: str | None
) -> list[tuple[str, QuantOption]]:
    """Show what fits, take numbers or a search, and return what was chosen.

    Returns the repository and the quantization for each pick, in the order they
    were typed; nothing is downloaded here. An empty list means the person asked
    for none, or that the Hub could not be reached - both are ordinary answers,
    not failures, because the caller has something to say either way.
    """
    ctx.say("  Looking at what is popular on Hugging Face and what fits here...")
    try:
        groups = ctx.recommend(machine, preset, use)
    except HubError as error:
        ctx.say(f"  Could not reach Hugging Face: {error}")
        return []
    wanted = [use] if use else list(GROUPS)
    numbered: list[Candidate] = []
    while True:
        for group in wanted:
            ctx.say(f"  {group}")
            for candidate in groups.get(group, []):
                if candidate in numbered:
                    continue
                numbered.append(candidate)
                ctx.say(f"   {len(numbered):>2}. {candidate_line(candidate, machine)}")
        answer = ctx.ask(PROMPT, "n").strip()
        if answer.lower() in ("n", "none", ""):
            return []
        if answer.lower().startswith("s "):
            try:
                results = ctx.search(machine, preset, answer[2:].strip())
            except HubError as error:
                ctx.say(f"  Search failed: {error}")
                continue
            groups = {"search": results}
            wanted = ["search"]
            continue
        try:
            picks = [numbered[n - 1] for n in parse_numbers(answer, len(numbered), OTHER_ANSWERS)]
        except ValueError as error:
            ctx.say(f"  {error}")
            continue
        chosen: list[tuple[str, QuantOption]] = []
        for candidate in picks:
            if not candidate.options:
                # `discover.search` reads the file list of only its first `files_for`
                # results, so a long enough search returns candidates that know their
                # name but not their quantizations. Both callers currently ask for ten
                # and get ten with files, which makes this unreachable today and worth
                # keeping anyway: asking the Hub about the one repository someone just
                # picked is cheap, and a repository that will not answer is skipped
                # rather than taking the rest of the picks with it.
                try:
                    files = ctx.hub.repo_files(candidate.repo_id)
                    candidate.options = quant_options(files.files)
                    candidate.suggested, _ = suggest(candidate.options, machine.budget)
                except HubError as error:
                    ctx.say(f"  {candidate.repo_id}: {error}")
                    continue
            option = ctx.choose_quant(candidate.options, candidate.suggested, machine)
            chosen.append((candidate.repo_id, option))
        return chosen
