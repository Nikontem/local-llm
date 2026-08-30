from local_llm.browse import BrowseContext, pick_models
from local_llm.discover import gather
from local_llm.discover import search as discover_search
from local_llm.hub import Hub, HubError

from .fakes import FakeApi
from .test_discover import MAC, MODELS


class Menu:
    """Scripted answers: ask() pops from `answers`, everything said is kept."""

    def __init__(self, answers, *, recommend=None, search=None):
        self.answers = list(answers)
        self.said: list[str] = []
        self.hub = Hub(api=FakeApi(MODELS), cache=None, cached_path_fn=lambda repo, name: None)
        self.ctx = BrowseContext(
            say=self.said.append,
            ask=lambda prompt, default: self.answers.pop(0) if self.answers else default,
            hub=self.hub,
            recommend=recommend
            or (lambda machine, preset, use: gather(self.hub, machine, preset, limit_per_group=2)),
            search=search
            or (
                lambda machine, preset, text: discover_search(
                    self.hub, machine, preset, text=text, limit=5
                )
            ),
            choose_quant=lambda options, suggested, machine: suggested,
        )

    def pick(self, use=None):
        return pick_models(self.ctx, MAC, None, use=use)

    def text(self):
        return "\n".join(self.said)

    def numbered(self):
        """The listing as {number: repo_id}, so a test picks by name, not by luck."""
        found = {}
        for line in self.said:
            stripped = line.strip()
            number, _, rest = stripped.partition(". ")
            if number.isdigit() and "/" in rest:
                found[int(number)] = next(w for w in rest.split() if "/" in w)
        return found


def test_none_downloads_nothing():
    menu = Menu(["n"])
    assert menu.pick() == []
    assert "coding" in menu.text() and "Qwen3-Coder-30B-A3B-Instruct" in menu.text()


def test_several_numbers_are_returned_in_the_order_they_were_typed():
    scout = Menu([])  # no answers: the default "n" ends the menu after one listing
    scout.pick()
    listing = scout.numbered()
    coder = next(n for n, repo in listing.items() if "Coder" in repo)
    small = next(n for n, repo in listing.items() if "1.5B" in repo)

    again = Menu([f"{small} {coder}"])
    chosen = again.pick()
    assert [repo for repo, _ in chosen] == [
        listing[small], listing[coder]
    ], "picks come back in typed order, not listing order"
    assert all(option is not None for _, option in chosen)


def test_a_repeated_number_picks_the_thing_once():
    menu = Menu(["1 1"])
    assert len(menu.pick()) == 1


def test_an_unreadable_answer_asks_again_instead_of_giving_up():
    menu = Menu(["nope", "0", "999", "n"])
    assert menu.pick() == []
    said = menu.text()
    assert said.count("Pick numbers between 1 and") == 3
    assert "s <text> to search, n for none" in said


def test_search_results_join_the_listing_and_can_be_picked():
    scout = Menu(["s Remix", "n"])
    scout.pick()
    listing = scout.numbered()
    remix = next(n for n, repo in listing.items() if repo == "DavidAU/Remix-GGUF")
    coder = next(n for n, repo in listing.items() if "Coder" in repo)
    assert "search" in scout.text()

    menu = Menu(["s Remix", str(remix)])
    assert [repo for repo, _ in menu.pick()] == ["DavidAU/Remix-GGUF"]

    # Searching does not renumber what came before it: a model seen in the first
    # listing is still pickable by the number it was given there.
    kept = Menu(["s Remix", str(coder)])
    assert [repo for repo, _ in kept.pick()] == [listing[coder]]


def test_a_search_that_fails_leaves_the_menu_standing():
    def boom(machine, preset, text):
        raise HubError("no route to huggingface.co")

    menu = Menu(["s Remix", "n"], search=boom)
    assert menu.pick() == []
    assert "Search failed: no route to huggingface.co" in menu.text()


def test_an_unreachable_hub_is_reported_and_picks_nothing():
    def boom(machine, preset, use):
        raise HubError("no route to huggingface.co")

    menu = Menu([], recommend=boom)
    assert menu.pick() == []
    assert "Could not reach Hugging Face: no route to huggingface.co" in menu.text()


def test_use_shows_only_that_group():
    menu = Menu(["n"])
    menu.pick(use="small")
    assert "  small" in menu.said and "  coding" not in menu.said
