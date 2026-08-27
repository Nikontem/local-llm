import subprocess

from local_llm.doctor import BREW_INSTALL, Check, Env, port_in_use, run_checks
from local_llm.hub import TokenStatus
from local_llm.paths import Paths
from local_llm.settings import Settings


def fake_run(outputs):
    """Build a subprocess.run replacement answering from {(binary, flag): text}."""

    def run(args, **kwargs):
        text = outputs.get((args[0], args[1]), "")
        return subprocess.CompletedProcess(args, 0, stdout=text, stderr="")

    return run


def mac_env(**overrides):
    tools = {
        "brew": "/opt/homebrew/bin/brew",
        "llama-server": "/opt/homebrew/bin/llama-server",
        "hf": "/opt/homebrew/bin/hf",
        "claude": "/usr/local/bin/claude",
    }
    outputs = {
        ("/opt/homebrew/bin/llama-server", "--version"):
            "version: 0.3.0 (build 10621, commit c1d0e7a00)\n",
        ("/opt/homebrew/bin/llama-server", "--help"):
            "... --models-preset PATH ... --list-devices ...\n",
        ("/opt/homebrew/bin/llama-server", "--list-devices"):
            "Available devices:\n  MTL0: Apple M4 Pro (38338 MiB, 38338 MiB free)\n",
    }
    env = Env(
        system="Darwin", machine="arm64", which=lambda name: tools.get(name),
        run=fake_run(outputs),
        token_status=lambda: TokenStatus("valid", "nikos"),
        port_in_use=lambda host, port: False,
    )
    for key, value in overrides.items():
        setattr(env, key, value)
    return env


def by_name(checks):
    return {c.name: c for c in checks}


def healthy_paths(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "m.gguf").write_bytes(b"x")
    paths.preset.write_text(f"[*]\njinja = true\n[m]\nmodel = {tmp_path}/m.gguf\n")
    return paths


def test_everything_ok_on_a_healthy_mac(tmp_path):
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=mac_env()))
    assert all(c.status == "ok" for c in checks.values()), checks
    assert "0.3.0" in checks["llama-server"].detail
    assert "Apple M4 Pro" in checks["llama-server"].detail
    assert checks["router support"].status == "ok"
    assert checks["hf token"].detail == "logged in as nikos"
    assert checks["models.ini"].detail == "1 model(s), all files present"
    assert checks["port"].detail == "5678 is free"
    assert "Claude Code" in checks["agents"].detail


def test_missing_brew_on_mac_is_fatal_and_optional_on_linux(tmp_path):
    no_brew = mac_env(which=lambda name: None)
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=no_brew))
    assert checks["brew"].status == "fail" and BREW_INSTALL in checks["brew"].fix
    assert checks["llama-server"].status == "fail" and BREW_INSTALL in checks["llama-server"].fix
    linux = mac_env(system="Linux", which=lambda name: None)
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=linux))
    assert checks["brew"].status == "warn"
    assert "releases" in checks["llama-server"].fix


def test_missing_llama_server_with_brew_offers_safe_fix(tmp_path):
    env = mac_env(which=lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=env))
    assert checks["llama-server"].fix_cmd == ["/opt/homebrew/bin/brew", "install", "llama.cpp"]
    assert checks["hf"].status == "warn"
    assert checks["hf"].fix_cmd == ["/opt/homebrew/bin/brew", "install", "hf"]
    assert "router support" not in checks


def test_old_llama_server_without_presets(tmp_path):
    env = mac_env(run=fake_run({("/opt/homebrew/bin/llama-server", "--help"): "no such flag\n"}))
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=env))
    assert checks["router support"].status == "fail"
    assert "December 2025" in checks["router support"].detail


def test_old_llama_server_without_list_devices_flag_does_not_dump_usage(tmp_path):
    # A build predating --list-devices answers it with an error plus a full usage
    # dump on stdout/stderr, same as it does for --models-preset. That text must
    # not be folded into the llama-server check's detail (spec 4.2: devices are
    # shown "when the flag exists").
    outputs = {
        ("/opt/homebrew/bin/llama-server", "--help"): "no such flag\n",
        ("/opt/homebrew/bin/llama-server", "--list-devices"):
            "error: invalid argument: --list-devices\nusage: llama-server [options]\n...\n",
    }
    env = mac_env(run=fake_run(outputs))
    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=env))
    assert "devices" not in checks["llama-server"].detail
    assert "invalid argument" not in checks["llama-server"].detail


def test_token_states(tmp_path):
    for status, expected_status, expected_fix in [
        (TokenStatus("absent"), "warn", "hf auth login"),
        (TokenStatus("invalid"), "warn", "hf auth login --force"),
        (TokenStatus("unreachable"), "warn", None),
    ]:
        env = mac_env(token_status=lambda s=status: s)
        check = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=env))["hf token"]
        assert check.status == expected_status and check.fix == expected_fix


def test_preset_problems(tmp_path):
    paths = Paths.from_env(env={}, home=tmp_path)
    checks = by_name(run_checks(paths, Settings(), env=mac_env()))
    assert checks["models.ini"].status == "warn" and checks["models.ini"].fix == "local-llm setup"
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.preset.write_text("[a]\nmodel = /nope/a.gguf\n[b]\nc = 1\n")
    check = by_name(run_checks(paths, Settings(), env=mac_env()))["models.ini"]
    assert check.status == "fail"
    assert "[a]: /nope/a.gguf" in check.detail and "[b]: no model key" in check.detail


def test_port_states(tmp_path):
    paths = healthy_paths(tmp_path)
    busy = mac_env(port_in_use=lambda host, port: True)
    ours = by_name(run_checks(paths, Settings(), env=busy, router_pid=42))["port"]
    assert ours.status == "ok" and "pid 42" in ours.detail
    other = by_name(run_checks(paths, Settings(), env=busy))["port"]
    assert other.status == "fail" and "--port" in other.fix


def test_check_is_a_plain_record():
    assert Check("x", "ok", "fine").fix is None


def test_port_in_use_handles_ipv6_loopback():
    # settings.py accepts "::1" as a loopback host; binding it needs AF_INET6, not
    # the AF_INET socket port_in_use used to open unconditionally (which raises
    # gaierror on an IPv6 literal and used to be misread as "port in use").
    assert port_in_use("::1", 0) is False


def test_llama_server_that_cannot_run_is_reported_as_such(tmp_path):
    import subprocess as sp

    def broken_run(args, **kwargs):
        text = "llama-server: error while loading shared libraries: libgomp.so.1: cannot open"
        return sp.CompletedProcess(args, 127, stdout="", stderr=text + "\n")

    checks = by_name(run_checks(healthy_paths(tmp_path), Settings(), env=mac_env(run=broken_run)))
    assert checks["llama-server"].status == "fail"
    assert "does not run" in checks["llama-server"].detail
    assert "libgomp" in checks["llama-server"].detail
    assert "router support" not in checks


def test_agents_check_names_every_harness_in_the_registry(tmp_path):
    from local_llm.doctor import Env, run_checks
    from local_llm.paths import Paths
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    env = Env(
        system="Darwin",
        machine="arm64",
        which=lambda name: "/usr/local/bin/codex" if name == "codex" else None,
        token_status=lambda: __import__(
            "local_llm.hub", fromlist=["TokenStatus"]
        ).TokenStatus("valid", "someone"),
        port_in_use=lambda host, port: False,
    )
    checks = {c.name: c for c in run_checks(paths, Settings(), env=env)}
    detail = checks["agents"].detail
    assert "OpenAI Codex CLI" in detail
    assert "Qwen Code" in detail and "Antigravity CLI" in detail


def test_codex_config_check_warns_when_the_address_moved(tmp_path):
    from local_llm.doctor import Env, run_checks
    from local_llm.integrations import codex
    from local_llm.paths import Paths
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    cx = codex.codex_paths(home=tmp_path, env={})
    codex.write(cx, codex.provider_table(Settings(port=5678)), None)
    env = Env(
        system="Darwin",
        machine="arm64",
        which=lambda name: None,
        token_status=lambda: __import__(
            "local_llm.hub", fromlist=["TokenStatus"]
        ).TokenStatus("valid", "someone"),
        port_in_use=lambda host, port: False,
        home=tmp_path,
        environ={},
    )
    ok = {c.name: c for c in run_checks(paths, Settings(port=5678), env=env)}
    assert "codex config" not in ok
    moved = {c.name: c for c in run_checks(paths, Settings(port=9999), env=env)}
    assert moved["codex config"].status == "warn"
    assert "integrate codex" in moved["codex config"].fix


def test_agents_check_reports_how_each_provider_is_configured(tmp_path):
    from local_llm.doctor import Env, run_checks
    from local_llm.integrations import codex
    from local_llm.paths import Paths
    from local_llm.settings import Settings

    paths = Paths.from_env(env={}, home=tmp_path)
    env = Env(
        system="Darwin",
        machine="arm64",
        which=lambda name: f"/usr/local/bin/{name}" if name in ("codex", "claude") else None,
        token_status=lambda: TokenStatus("valid", "someone"),
        port_in_use=lambda host, port: False,
        home=tmp_path,
        environ={},
    )
    detail = by_name(run_checks(paths, Settings(), env=env))["agents"].detail
    assert "OpenAI Codex CLI (not configured)" in detail
    assert "Claude Code" in detail and "Claude Code (" not in detail, "launchers stay plain"

    codex.write(codex.codex_paths(home=tmp_path, env={}), codex.provider_table(Settings()), None)
    detail = by_name(run_checks(paths, Settings(), env=env))["agents"].detail
    assert "OpenAI Codex CLI (configured)" in detail
