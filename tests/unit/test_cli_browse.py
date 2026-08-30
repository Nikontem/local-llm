from local_llm import cli
from local_llm.preset import Preset

SMALL = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
SMALL_SECTION = f"{SMALL}:Q8_0"  # the quantization suggested for this machine


def menu(hubbed):
    """The harness with a terminal, which is the only place the menu will run."""
    hubbed.monkeypatch.setattr(cli, "_interactive", lambda: True)
    return hubbed


def numbered(output):
    """The listing as {number: repo_id}, so a test picks by name, not by luck."""
    found = {}
    for line in output.splitlines():
        number, _, rest = line.strip().partition(". ")
        if number.isdigit() and "/" in rest:
            found[int(number)] = next(w for w in rest.split() if "/" in w)
    return found


def test_without_a_terminal_it_says_what_to_use_instead(hubbed):
    result = hubbed.run("browse-models")  # the harness is not a terminal by default
    assert result.exit_code == 1
    assert "needs a terminal" in result.output
    assert "local-llm recommend --json" in result.output
    assert "local-llm pull" in result.output


def test_bad_use_is_refused_by_name(hubbed):
    result = menu(hubbed).run("browse-models", "--use", "audio")
    assert result.exit_code == 1 and "coding, general, small, vision" in result.output
    assert "Looking at what is popular" not in result.output, "refused before the scan"


def test_limit_finetunes_and_refresh_reach_the_scan(hubbed):
    h = menu(hubbed)
    one = h.run("browse-models", "--use", "coding", "--limit", "1", input="n\n")
    assert len(numbered(one.output)) == 1, "--limit caps the models listed per group"

    vendor_only = h.run("browse-models", "--limit", "10", input="n\n")
    assert "Remix" not in vendor_only.output
    with_remixes = h.run("browse-models", "--limit", "10", "--include-finetunes", input="n\n")
    assert "DavidAU/Remix-GGUF" in with_remixes.output, "--include-finetunes widens the listing"

    asked = []

    def note_refresh(st, refresh=False):
        asked.append(refresh)
        return h.hub

    h.monkeypatch.setattr(cli, "_make_hub", note_refresh)
    h.run("browse-models", "--use", "small", input="n\n")
    h.run("browse-models", "--use", "small", "--refresh", input="n\n")
    assert asked == [False, True], "--refresh is what decides whether the cache is ignored"


def test_none_downloads_nothing_and_says_so(hubbed):
    result = menu(hubbed).run("browse-models", "--use", "small", input="n\n")
    assert result.exit_code == 0, result.output
    assert "Apple M4 Pro" in result.output
    assert "nothing downloaded" in result.output
    assert not hubbed.paths.preset.read_text().count(SMALL)


def test_two_picks_become_two_sections_in_one_run(hubbed):
    h = menu(hubbed)
    listing = numbered(h.run("browse-models", input="n\n").output)
    coder = next(n for n, repo in listing.items() if "Coder" in repo)
    small = next(n for n, repo in listing.items() if repo == SMALL)

    # The menu answer, then the suggested quantization for each pick in turn.
    result = h.run("browse-models", input=f"{coder} {small}\n1\n1\n")
    assert result.exit_code == 0, result.output
    preset = Preset.load(h.paths.preset)
    sections = preset.sections()
    assert f"{listing[coder]}:Q4_K_XL" in sections
    assert SMALL_SECTION in sections
    assert "local-llm up" in result.output


def test_a_model_already_in_models_ini_is_skipped_not_re_downloaded(hubbed):
    h = menu(hubbed)
    listing = numbered(h.run("browse-models", input="n\n").output)
    small = next(n for n, repo in listing.items() if repo == SMALL)
    first = h.run("browse-models", input=f"{small}\n1\n")
    assert first.exit_code == 0, first.output

    again = h.run("browse-models", input=f"{small}\n1\n")
    assert again.exit_code == 0, again.output
    assert f"[{SMALL_SECTION}] is already in" in again.output
    assert "downloading" not in again.output


def test_a_download_that_fails_does_not_take_the_later_picks_with_it(hubbed):
    from local_llm.estimate import GIB

    h = menu(hubbed)
    listing = numbered(h.run("browse-models", input="n\n").output)
    coder = next(n for n, repo in listing.items() if "Coder" in repo)
    small = next(n for n, repo in listing.items() if repo == SMALL)
    # Room for the 1.6 GB pick and nowhere near enough for the 16.5 GB one, so the
    # first pull fails the way a full disk fails and the second could still succeed.
    h.monkeypatch.setattr(cli, "free_disk_bytes", lambda path: 5 * GIB)

    result = h.run("browse-models", input=f"{coder} {small}\n1\n1\n")
    assert "Not enough disk space" in result.output
    assert f"{listing[coder]} was not added" in result.output
    assert SMALL_SECTION in Preset.load(h.paths.preset).sections(), (
        "the pick after the failed one is still downloaded"
    )
    assert result.exit_code == 1 and f"Not added: {listing[coder]}" in result.output


def test_search_reaches_a_model_the_recommendations_never_show(hubbed):
    h = menu(hubbed)
    assert "Remix" not in h.run("browse-models", input="n\n").output
    listing = numbered(h.run("browse-models", input="s Remix\nn\n").output)
    remix = next(n for n, repo in listing.items() if repo == "DavidAU/Remix-GGUF")

    result = h.run("browse-models", input=f"s Remix\n{remix}\n1\n")
    assert result.exit_code == 0, result.output
    assert "DavidAU/Remix-GGUF:Q4_K_M" in Preset.load(h.paths.preset).sections()


def test_the_recommend_listing_points_at_the_menu(hubbed):
    assert "local-llm browse-models" in hubbed.run("recommend").output
