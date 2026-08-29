
from local_llm.doctor import Env
from local_llm.estimate import GIB
from local_llm.hardware import Machine
from local_llm.hub import Hub, TokenStatus
from local_llm.paths import Paths
from local_llm.preset import Preset
from local_llm.router import Router
from local_llm.settings import load_settings
from local_llm.setup import Io, SetupAbort, SetupContext, run_setup, smallest_model, step_models

from .fakes import FakeApi, FakeBackend, FakeHttp
from .test_discover import MODELS

MAC = Machine("Darwin", "arm64", "Apple M4 Pro", 48 * GIB, "apple", 48 * GIB, 20, 10)


class Script:
    """Scripted answers: confirm() pops from `confirms`, ask() pops from `asks`."""

    def __init__(self, confirms=(), asks=()):
        self.confirms = list(confirms)
        self.asks = list(asks)
        self.said: list[str] = []
        self.ran: list[list[str]] = []

    def io(self, yes=False):
        return Io(
            say=self.said.append,
            ask=lambda prompt, default: self.asks.pop(0) if self.asks else default,
            confirm=lambda prompt, default: self.confirms.pop(0) if self.confirms else default,
            run=lambda cmd: self.ran.append(cmd) or 0,
            yes=yes,
        )

    def text(self):
        return "\n".join(self.said)


def make_ctx(tmp_path, script, *, yes=False, tools=None, token="valid", backend=None, http=None):
    tools = (
        tools
        if tools is not None
        else {
            "brew": "/opt/homebrew/bin/brew",
            "llama-server": "/opt/homebrew/bin/llama-server",
            "hf": "/opt/homebrew/bin/hf",
            "opencode": "/usr/local/bin/opencode",
            "claude": "/usr/local/bin/claude",
        }
    )
    paths = Paths.from_env(env={}, home=tmp_path)
    settings = load_settings(paths, env={})
    hub = Hub(api=FakeApi(MODELS), cache=None, cached_path_fn=lambda repo, name: None)
    backend = backend or FakeBackend()
    backend.spawn_listening = {5678}
    http = http or FakeHttp(
        {
            ("GET", "/health"): {"status": "ok"},
            ("POST", "/v1/chat/completions"): {
                "choices": [{"message": {"content": "Hello there friend"}}]
            },
        }
    )
    pulled: list[tuple[str, str]] = []
    integrations: list[str] = []

    def outputs(binary, flag):
        return {
            "--version": "version: 0.3.0\n",
            "--help": "--models-preset --list-devices\n",
            "--list-devices": "Available devices:\n  MTL0: Apple M4 Pro (38338 MiB)\n",
        }.get(flag, "")

    import subprocess

    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=outputs(args[0], args[1]), stderr="")

    env = Env(system="Darwin", machine="arm64", which=lambda n: tools.get(n), run=run,
              token_status=lambda: TokenStatus(token, "nikos" if token == "valid" else None),
              port_in_use=lambda host, port: False)

    def pull(machine, repo, option):
        section = f"{repo}:{option.tag}"
        pulled.append((repo, option.label))
        preset = (
            Preset.parse(paths.preset.read_text())
            if paths.preset.is_file()
            else Preset.parse("[*]\njinja = true\n")
        )
        model = tmp_path / f"{option.primary}"
        model.write_bytes(b"x" * 16)
        preset.add_section(section, [("model", str(model)), ("c", "8192")])
        preset.save(paths.preset)
        return section

    from local_llm.discover import gather
    from local_llm.discover import search as discover_search

    ctx = SetupContext(
        paths=paths, settings=settings, io=script.io(yes=yes), env=env, hub=hub,
        detect=lambda reserve: Machine(
            "Darwin", "arm64", "Apple M4 Pro", 48 * GIB, "apple", 48 * GIB, 20, reserve
        ),
        router=lambda: Router(
            paths, settings, backend=backend, http=http,
            binary="/opt/bin/llama-server", sleep=lambda s: None,
        ),
        which=lambda n: tools.get(n),
        recommend=lambda machine, preset, use: gather(hub, machine, preset, limit_per_group=2),
        search=lambda machine, preset, text: discover_search(
            hub, machine, preset, text=text, limit=5
        ),
        choose_quant=lambda options, suggested, machine: suggested,
        pull=pull,
        integrate_shell=lambda shell, extra=(): integrations.append(f"shell:{shell}")
        or [f"completion for {shell} installed"],
        harness_status=lambda harness: "missing",
        configure_harness=lambda harness: [],
        shell="zsh",
    )
    ctx.pulled = pulled  # type: ignore[attr-defined]
    ctx.integrations = integrations  # type: ignore[attr-defined]
    ctx.backend = backend  # type: ignore[attr-defined]
    ctx.http = http  # type: ignore[attr-defined]
    return ctx


def test_full_run_with_yes_and_a_preselected_model(tmp_path):
    script = Script()
    ctx = make_ctx(tmp_path, script, yes=True)
    code = run_setup(ctx, models=["Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q8_0"])
    assert code == 0, script.text()
    assert ctx.pulled == [("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "Q8_0")]
    text = script.text()
    assert "1. Prerequisites" in text and "logged in as nikos" in text
    assert "Apple M4 Pro, 20 GPU cores, 48 GB unified memory" in text
    assert ctx.paths.settings_file.is_file()
    settings_text = ctx.paths.settings_file.read_text()
    assert 'default_model = "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q8_0"' in settings_text
    assert ctx.integrations == ["shell:zsh"]  # agents now go through configure_harness
    assert ctx.backend.spawned, "router was started"
    expected_call = (
        "POST",
        "/v1/chat/completions",
        {
            "model": "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q8_0",
            "messages": [{"role": "user", "content": "Say hello in three words."}],
            "max_tokens": 12,
        },
    )
    assert expected_call in ctx.http.calls
    assert "Hello there friend" in text
    assert "http://127.0.0.1:5678/v1" in text and "local-llm status" in text


def test_yes_without_models_skips_downloads_and_start(tmp_path):
    script = Script()
    ctx = make_ctx(tmp_path, script, yes=True)
    assert run_setup(ctx) == 0
    assert ctx.pulled == [] and not ctx.backend.spawned
    assert "No models configured yet" in script.text() and "local-llm recommend" in script.text()
    assert ctx.paths.settings_file.is_file()


def test_interactive_pick_from_recommendations_and_reserve_change(tmp_path):
    script = Script(confirms=[True, True, True, True], asks=["12", "4"])
    ctx = make_ctx(tmp_path, script)
    assert run_setup(ctx) == 0, script.text()
    assert ctx.settings.reserve_gb == 12
    assert "36 GB usable" in script.text()
    accepted_vendor_repos = {
        c
        for c in (
            "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
            "unsloth/Qwen3.8-27B-GGUF",
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        )
    }
    assert (
        len(ctx.pulled) == 1
        and ctx.pulled[0][0] == "unsloth/Qwen3.5-4B-GGUF"
        or ctx.pulled[0][0] in accepted_vendor_repos
    )


def test_search_then_skip(tmp_path):
    script = Script(asks=["10", "s Remix", "n"], confirms=[False, False, False])
    ctx = make_ctx(tmp_path, script)
    assert run_setup(ctx) == 0
    assert "DavidAU/Remix-GGUF" in script.text() and ctx.pulled == []


def test_missing_brew_on_mac_aborts_with_instructions(tmp_path):
    script = Script()
    ctx = make_ctx(tmp_path, script, yes=True, tools={})
    assert run_setup(ctx) == 1
    assert "Homebrew" in script.text() and "install.sh" in script.text()


def test_missing_llama_server_is_installed_when_confirmed(tmp_path):
    script = Script(confirms=[True])
    tools = {"brew": "/opt/homebrew/bin/brew"}
    ctx = make_ctx(tmp_path, script, tools=tools, token="absent")
    # once brew install runs, pretend llama-server appeared
    original_run = ctx.io.run

    def run_and_install(cmd):
        tools["llama-server"] = "/opt/homebrew/bin/llama-server"
        return original_run(cmd)

    ctx.io.run = run_and_install
    ctx.io.confirm = lambda prompt, default: True if "Install llama.cpp" in prompt else False
    ctx.io.ask = lambda prompt, default: "n" if "Numbers" in prompt else default
    assert run_setup(ctx) == 0, script.text()
    assert ["/opt/homebrew/bin/brew", "install", "llama.cpp"] in script.ran
    assert "gated" in script.text().lower()


def test_smallest_model_and_step_models_with_bad_preselection(tmp_path):
    preset = Preset.parse(f"[a]\nmodel = {tmp_path}/a.gguf\n[b]\nmodel = {tmp_path}/b.gguf\n")
    (tmp_path / "a.gguf").write_bytes(b"x" * 20)
    (tmp_path / "b.gguf").write_bytes(b"x" * 5)
    assert smallest_model(preset) == "b"
    assert smallest_model(Preset.parse("[*]\n")) is None
    script = Script()
    ctx = make_ctx(tmp_path, script, yes=True)
    try:
        step_models(ctx, MAC, None, use=None, preselected=["nobody/nothing-GGUF"])
    except SetupAbort as error:
        assert "No such repository" in str(error)
    else:
        raise AssertionError("expected SetupAbort")


def test_step_six_offers_installed_agents_and_configures_the_chosen(tmp_path):
    from local_llm.setup import step_integrations

    script = Script(confirms=[True], asks=["1"])
    ctx = make_ctx(tmp_path, script, tools={
        "codex": "/usr/local/bin/codex",
        "claude": "/usr/local/bin/claude",
        "gemini": "/usr/local/bin/gemini",
    })
    done: list[str] = []
    ctx.configure_harness = lambda h: done.append(h.key) or [f"{h.key} done"]
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    text = script.text()
    assert "1. OpenAI Codex CLI" in text and "2. Claude Code" in text
    assert "Gemini CLI" in text and "Not found:" in text
    assert done == ["codex"] and "codex done" in text


def test_step_six_with_yes_configures_everything_installed(tmp_path):
    from local_llm.setup import step_integrations

    script = Script()
    ctx = make_ctx(tmp_path, script, yes=True, tools={"codex": "/x", "aider": "/y"})
    done: list[str] = []
    ctx.configure_harness = lambda h: done.append(h.key) or []
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    assert done == ["codex", "aider"]


def test_step_six_says_when_nothing_is_installed(tmp_path):
    from local_llm.setup import step_integrations

    script = Script(confirms=[True])
    ctx = make_ctx(tmp_path, script, tools={})
    ctx.configure_harness = lambda h: []
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    assert "No coding agents found on PATH" in script.text()


def test_step_six_warns_when_the_router_is_not_loopback(tmp_path):
    from local_llm.setup import step_integrations

    script = Script(confirms=[True], asks=["n"])
    ctx = make_ctx(tmp_path, script, tools={"codex": "/usr/local/bin/codex"})
    ctx.settings.host = "0.0.0.0"
    ctx.configure_harness = lambda h: []
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    assert "not a loopback address" in script.text()


def test_step_six_asks_harness_questions_through_io(tmp_path):
    from local_llm.setup import step_integrations

    script = Script(confirms=[True], asks=["1"])
    ctx = make_ctx(tmp_path, script, tools={"codex": "/usr/local/bin/codex"})
    asked: list[str] = []

    def configure(harness):
        asked.append(harness.key)
        return ["did it"]

    ctx.configure_harness = configure
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    assert asked == ["codex"] and "did it" in script.text()


def test_step_six_survives_a_harness_that_raises_anything(tmp_path):
    """A failing agent is one printed line; the wizard must reach step 7 regardless."""
    from local_llm.setup import step_integrations

    script = Script(confirms=[True])
    ctx = make_ctx(tmp_path, script, yes=True, tools={"codex": "/x", "aider": "/y"})
    done: list[str] = []

    def configure(harness):
        if harness.key == "codex":
            raise ValueError("something nobody predicted")
        done.append(harness.key)
        return [f"{harness.key} done"]

    ctx.configure_harness = configure
    ctx.harness_status = lambda h: "missing"
    step_integrations(ctx)
    text = script.text()
    assert "could not configure OpenAI Codex CLI: something nobody predicted" in text
    assert done == ["aider"] and "aider done" in text
